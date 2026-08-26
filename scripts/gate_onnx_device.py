#!/usr/bin/env python3
"""Gate: does the device half export, run, and stay inside the operator set an edge
compiler will take?

Runs on macOS, needs no accelerator and no Dataflow Compiler. It answers the question
the compiler answers, one stage earlier and on the machine the work happens on.

THIS IS THE PRIMARY GATE, AND THE DFC GATE IS SECONDARY. That is a statement about
CADENCE and not about truth. This one runs on any desk, on every change, and blocks;
`gate_dfc_parse.py` needs Linux x86-64 because that is the only platform the DFC wheel
builds for, so requiring it everywhere makes routine work wait on a machine somebody
has to stand up and authenticate to.

WHAT DID NOT FLIP: WHICH ONE IS RIGHT. `DEVICE_OPS` below is a hand-maintained
allowlist, which is a CLAIM about the compiler. The compiler is the compiler. When the
two disagree the allowlist is still what gets corrected -- a proxy that outranks its
own measurement is PITFALLS 4 written into a gate, and the cost of being wrong here is
discovering it after a training run rather than before one.

So: this gate gives a fast, portable, blocking answer, and it is allowed to be wrong in
the direction of caution. The DFC gate is run before anything is committed to hardware,
and its verdict wins.

THREE CHECKS, and each fails loudly rather than skipping:

1. EXPORT. Two operators block a plain export and they are not the same kind of
   problem. `aten::_upsample_bicubic2d_aa` resizes DINOv2's position embedding to the
   deployment resolution; it reads a parameter and a fixed size, never the image, so at
   one resolution its output is constant and folding it is exact. `LWDETR.export()`
   swaps the training forward for `forward_export`, which drops aux outputs: 12427
   nodes to 3632 at 1272. Both are applied here, not left to the caller.

2. NUMERIC. The export is run in onnxruntime and diffed against PyTorch on the same
   input. The bound is the port's own: `test_keypoints` holds keypoints at 4.2e-3.

3. OPERATOR. Every operator is checked against DEVICE_OPS below. An operator outside
   that set is a FAIL naming the operator, because that is what the compiler will do.

Run:  pixi run -e gate \
          python scripts/gate_onnx_device.py [--self-test]
"""
from __future__ import annotations

import argparse
import os
import collections
import sys
import time
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

#: What the accelerator is expected to take. Ordinary conv-net and transformer
#: arithmetic, plus the shape plumbing a fixed-resolution graph folds away.
#:
#: This list is a CLAIM about the compiler, not a fact, and `gate_dfc_parse.py` is what
#: turns it into one. Every entry below was observed in a device-half export that the
#: numeric check passed; nothing is here on the strength of a datasheet.
DEVICE_OPS = {
    # OneHot is accepted ONLY inside an exact pattern, which is why it sat in
    # KNOWN_BLOCKERS after a first refusal. `is_supported_one_hot` in the translator's
    # onnx_graph.py walks a chain of OneHot(axis=-1) -> Transpose(perm=[0,4,2,3,1]) ->
    # Squeeze(axis 4 or -1) with exactly one predecessor, and anything between those three
    # nodes breaks the match. A `Cast` emitted by `.to(dtype)` before the Transpose was the
    # entire difference between REFUSED and PARSED OK on identical arithmetic; the cast
    # belongs after the Squeeze. See build/onehot/pattern.py for the exported shape.
    "OneHot",
    "Add", "AveragePool", "Cast", "Clip", "Concat", "Constant", "ConstantOfShape",
    "Conv", "Div", "Equal", "Erf", "Expand", "Flatten", "Gather", "Gemm", "Identity",
    "LayerNormalization", "MatMul", "MaxPool", "Mul", "Pad", "Pow", "Range",
    "ReduceMean", "ReduceSum", "Relu", "Reshape", "Shape", "Sigmoid", "Slice",
    "Softmax", "Split", "Sqrt", "Squeeze", "Sub", "Tanh", "Transpose",
    "Unsqueeze", "Where",
}

#: Operators measured to be absent from the device half and present in the DETR
#: decoder. Named rather than merely excluded, so a regression reports the reason.
KNOWN_BLOCKERS = {
    "GridSample": "deformable attention; Hailo state it is unsupported with no plan to add it",
    "ScatterND": "keypoint-schema scatter in the decoder",
    "TopK": "two-stage query selection",
    "GatherElements": "dynamic gather, also what a GridSample decomposition produces",
    # Moved out of DEVICE_OPS by the Linux gate, which is what that gate is for. The
    # macOS gate passed the device half with Tile x1 and DFC 5.3.0 rejected it:
    #   UnsupportedConcatLayerError in op /encoder/encoder/embeddings/Tile:
    #   Unsupported concat over axis batch
    # It broadcasts the CLS and position tokens across the batch axis. At batch 1 that
    # is a no-op, so it should fold rather than compile -- the same class of fix as the
    # antialias resize above, and not yet applied.
    "Tile": "DFC 5.3.0: unsupported concat over axis batch; fold it at batch 1",
    "NonZero": "data-dependent output shape",
    "NonMaxSuppression": "data-dependent output shape",
    "Loop": "control flow",
    "If": "control flow",
}

KEYPOINT_TOL = 4.2e-3   # the bound `test_keypoints` already holds


def _fold_antialias():
    """Constant-fold antialiased resizes. Exact: the inputs are a parameter and a size."""
    orig = F.interpolate
    state = {"n": 0}

    def patched(*a, **kw):
        if not kw.get("antialias"):
            return orig(*a, **kw)
        with torch.no_grad():
            out = orig(*a, **kw)
        state["n"] += 1
        # from_numpy severs the traced dependency, so the graph records a constant.
        return torch.from_numpy(out.detach().cpu().numpy())

    F.interpolate = patched
    return orig, state


def build_device_half(resolution: int, num_windows: int | None = None):
    """The backbone and projector, which is exactly what would be compiled.

    NUM_WINDOWS IS THE WHOLE COMPATIBILITY QUESTION, so it is a parameter rather than a
    default nobody reads. `RFDETRKeypointPreviewConfig.num_windows` is 2, which exports 868
    nodes -- the graph DFC 5.3.0 rejects with "Unsupported concat over axis batch" on the
    Tile that replicates the CLS token once per window. At 1 there is no per-window
    replication, the Tile does not exist, and the export is the 825-node graph the compiler
    takes."""
    from rfdetr import RFDETRKeypointPreview

    kw = {} if num_windows is None else {"num_windows": num_windows}
    model = RFDETRKeypointPreview(resolution=resolution, **kw)
    backbone = model.model.model.backbone[0].eval()

    class DeviceHalf(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, x):
            feats, _masks, cross = self.b.forward_export(x)
            return (feats[0], cross[0]) if cross is not None else feats[0]

    return DeviceHalf(backbone).eval()


def export_and_check(module, resolution, path, tol=KEYPOINT_TOL, allow=DEVICE_OPS):
    """Return (problems, facts). An empty problems list is the only pass."""
    problems, facts = [], {}
    x = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        ref = module(x)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)

    with torch.no_grad():
        torch.onnx.export(module, (x,), path, opset_version=17,
                          input_names=["image"], dynamo=False, verbose=False)

    graph = onnx.load(path).graph
    ops = collections.Counter(n.op_type for n in graph.node)
    facts["nodes"] = len(graph.node)
    facts["ops"] = len(ops)

    so = ort.SessionOptions()
    so.log_severity_level = 3

    # EVERY AVAILABLE PROVIDER, NOT JUST THE CPU, AND THE ONE THAT ANSWERED IS REPORTED.
    #
    # This line read `providers=["CPUExecutionProvider"]`, and the docstring above still says
    # the gate "needs no accelerator". Both are true statements that together produced a false
    # one: `logbook-edge-npu-and-the-anny-forward.md` opens "macOS, no accelerator" and reports
    # the backbone at 1399.3 ms, which reads as what this machine can do. It is what this line
    # allowed it to do. A hardcoded backend list turns a capability question into a tautology.
    #
    # So each provider onnxruntime actually offers is run, and each is diffed against PyTorch
    # separately. That matters more than the timing: CoreML is free to use different kernels
    # and lower precision, so a provider can be fast and WRONG, and the gate's 4.2e-3 bound is
    # the contract that decides. A provider outside the bound is a deployability finding, not a
    # threshold to loosen.
    #
    # MEASURED, AND THE ANSWER WAS NOT THE ONE EXPECTED. At 576 with num_windows=1, CoreML runs
    # 1685.1 ms against the CPU's 476.4 -- 0.28x, nearly four times SLOWER -- and lands at
    # 4.924e-03 against a 4.2e-03 bound. It is slower AND outside tolerance. The likely reason
    # is in the caveat below: the provider partitions the graph and hands most of it back, so
    # the transfers cost more than the acceleration returns.
    #
    # That does not make the old hardcoded list correct. It made a capability claim nobody had
    # tested, and it happened to pick the right backend for the wrong reason; the next model or
    # the next runtime version moves that answer and nothing would have noticed.
    #
    # THE CPU RESULT REMAINS THE VERDICT. This gate answers "will the export deploy", and the
    # accelerator comparison is evidence beside that rather than a replacement for it.
    available = [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
    facts["providers_available"] = available
    facts["by_provider"] = {}
    outs = None
    for prov in available:
        try:
            sess = ort.InferenceSession(path, so, providers=[prov])
            t0 = time.perf_counter()
            got = sess.run(None, {"image": x.numpy()})
            elapsed = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            # Named and counted. A provider that will not build is a fact about the deployment
            # surface, and absorbing it silently is how this file got here.
            facts["by_provider"][prov] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        diff = max((float(np.abs(r.detach().numpy() - o).max())
                    for r, o in zip(refs, got) if tuple(r.shape) == tuple(o.shape)), default=None)
        facts["by_provider"][prov] = {
            # `get_providers()` reports what was REGISTERED. CoreML partitions a graph and
            # silently runs unsupported nodes on the CPU, so this does not prove the
            # accelerator took the whole model -- stated because the opposite is easy to assume.
            "registered": sess.get_providers(),
            "ms": elapsed,
            "max_abs_diff": diff,
        }
        if prov == "CPUExecutionProvider":
            outs = got
    if outs is None:
        problems.append("CPUExecutionProvider did not run; it is the gate's reference and "
                        "an accelerator result cannot stand in for it")
        return problems, facts

    for prov, r in facts["by_provider"].items():
        if prov == "CPUExecutionProvider" or "error" in r or r["max_abs_diff"] is None:
            continue
        if r["max_abs_diff"] > tol:
            # The wording here asserted the provider was "faster and does not agree", written
            # before it was run. It is not faster -- CoreML measured 0.28x the CPU at 576 --
            # so the sentence stated a speed nobody had measured while reporting a difference
            # that was. Say only what the numbers say.
            problems.append(
                f"provider {prov}: max|diff| {r['max_abs_diff']:.3e} exceeds {tol:.1e}, so it "
                f"does not agree with PyTorch inside the port's own bound. See the timing "
                f"table above for whether it is even faster")

    if len(outs) != len(refs):
        problems.append(f"onnx returned {len(outs)} outputs, pytorch produced {len(refs)}")
    worst = 0.0
    for r, o in zip(refs, outs):
        if tuple(r.shape) != tuple(o.shape):
            problems.append(f"shape drift: pytorch {tuple(r.shape)} vs onnx {tuple(o.shape)}")
            continue
        worst = max(worst, float(np.abs(r.detach().numpy() - o).max()))
    facts["max_abs_diff"] = worst
    if worst > tol:
        problems.append(f"numeric: max|diff| {worst:.3e} exceeds {tol:.1e}")

    outside = {o: c for o, c in ops.items() if o not in allow}
    facts["outside"] = outside
    for op, count in sorted(outside.items()):
        why = KNOWN_BLOCKERS.get(op, "not in DEVICE_OPS")
        problems.append(f"operator: {op} x{count} -- {why}")
    return problems, facts


def run(resolution, path, num_windows=None):
    orig, folds = _fold_antialias()
    try:
        module = build_device_half(resolution, num_windows)
        problems, facts = export_and_check(module, resolution, path)
    finally:
        F.interpolate = orig
    if folds["n"] == 0:
        # A silent skip reads exactly like a pass. If the fold never fired, either the
        # model changed or the patch missed, and the export below is not the one meant.
        problems.append("the antialias fold never fired, so this export is not the gated one")
    print(f"  resolution {resolution}: {facts['nodes']} nodes, {facts['ops']} operators, "
          f"max|diff| {facts['max_abs_diff']:.3e}, {folds['n']} resize folded")
    if facts["outside"]:
        print(f"  operators outside DEVICE_OPS: {facts['outside']}")

    # THE PROVIDER TABLE, PRINTED, BECAUSE ITS ABSENCE IS WHAT MADE 1399.3 ms READ AS THE MAC'S
    # THROUGHPUT. A run that does not say which backend answered cannot be quoted safely.
    by = facts.get("by_provider") or {}
    if by:
        cpu_ms = (by.get("CPUExecutionProvider") or {}).get("ms")
        print("  provider                       ms      max|diff|   vs CPU")
        for prov, r in by.items():
            if "error" in r:
                print(f"  {prov:28s} FAILED TO BUILD -- {r['error']}")
                continue
            rel = f"{cpu_ms / r['ms']:.2f}x" if cpu_ms and r["ms"] else "--"
            print(f"  {prov:28s} {r['ms']:8.1f}   {r['max_abs_diff']:.3e}   {rel}")
        print("  `registered` is what onnxruntime accepted, not what ran: CoreML partitions a")
        print("  graph and puts unsupported nodes back on the CPU, so a fast row is not proof")
        print("  the accelerator took the whole model.")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=576)
    # NOT "/tmp/...". This gate answers a go/no-go about hardware, so it has to run on the
    # desk the work happens on, and this workspace is on Windows where /tmp does not exist.
    # The export reached this line and died on FileNotFoundError, which reads like a model
    # problem in a traceback and is a path problem.
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "device_half.onnx"))
    ap.add_argument("--num-windows", type=int, default=None,
                    help="1 is the configuration DFC 5.3.0 accepts; the checkpoint default "
                         "is 2, which exports the rejected graph")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    print("device-half gate")
    problems = run(a.resolution, a.out, a.num_windows)
    if problems:
        print(f"\nFAIL ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nPASS: exports, runs, and every operator is inside DEVICE_OPS")
    return 0


# --- negative controls -------------------------------------------------------------
# Each breaks one thing and asserts the gate says so. A check that passes on broken
# input has certified the defect rather than caught it. These use small synthetic
# modules on purpose: a control that takes a minute to run is a control nobody runs.

class _WithGridSample(torch.nn.Module):
    """The blocker itself, as a graph. The operator check must reject this."""
    def forward(self, x):
        n, c, h, w = x.shape
        grid = torch.zeros(n, h, w, 2)
        return F.grid_sample(x, grid, mode="bilinear", align_corners=False)


class _Ordinary(torch.nn.Module):
    """Conv and relu only. The operator check must accept this."""
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.c(x))


def _check(module, res, path, **kw):
    return export_and_check(module, res, path, **kw)[0]


def self_test():
    controls, fails = [], []

    def control(name):
        def deco(fn):
            controls.append((name, fn))
            return fn
        return deco

    @control("1 a graph containing GridSample is rejected")
    def _c1():
        problems = _check(_WithGridSample(), 32, "/tmp/_ctl_grid.onnx")
        assert any("GridSample" in p for p in problems), \
            f"GridSample was not reported; problems were {problems}"
        assert any("Hailo" in p for p in problems), "the refusal does not say why"
        return next(p for p in problems if "GridSample" in p)

    @control("2 an ordinary conv graph passes")
    def _c2():
        problems = _check(_Ordinary(), 32, "/tmp/_ctl_ok.onnx")
        assert not problems, f"a clean graph was rejected: {problems}"
        return "conv+relu accepted, so control 1 is not rejecting everything"

    @control("3 a tightened tolerance turns the same graph into a failure")
    def _c3():
        # Same module, impossible bound. If this still passes, the numeric check is
        # not wired to the tolerance at all and control 2's pass means nothing.
        problems = _check(_Ordinary(), 32, "/tmp/_ctl_tol.onnx", tol=-1.0)
        assert any("numeric" in p for p in problems), \
            f"an impossible tolerance did not fail: {problems}"
        return next(p for p in problems if "numeric" in p)

    @control("4 an operator removed from the allowlist is named")
    def _c4():
        narrowed = DEVICE_OPS - {"Conv"}
        problems = _check(_Ordinary(), 32, "/tmp/_ctl_allow.onnx", allow=narrowed)
        assert any("Conv" in p for p in problems), \
            f"removing Conv from the allowlist changed nothing: {problems}"
        return "Conv reported once removed, so the allowlist is load-bearing"

    @control("5 KNOWN_BLOCKERS and DEVICE_OPS do not overlap")
    def _c5():
        both = DEVICE_OPS & set(KNOWN_BLOCKERS)
        assert not both, f"these are both allowed and blocked: {both}"
        return f"{len(DEVICE_OPS)} allowed, {len(KNOWN_BLOCKERS)} blocked, disjoint"

    for name, fn in controls:
        try:
            print(f"  ok   {name}\n         {fn()}")
        except AssertionError as e:
            fails.append(name)
            print(f"  FAIL {name}\n         {e}")
    print()
    if fails:
        print(f"{len(fails)} of {len(controls)} controls failed.")
        return 1
    print(f"{len(controls)} controls, each failing for its own reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
