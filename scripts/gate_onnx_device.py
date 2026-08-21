#!/usr/bin/env python3
"""Gate: does the device half export, run, and stay inside the operator set an edge
compiler will take?

Runs on macOS, needs no accelerator and no Dataflow Compiler. It answers the question
the compiler answers, one stage earlier and on the machine the work happens on. The
Linux gate beside it (`gate_dfc_parse.py`) then runs the real compiler and must agree;
where the two disagree, this file's allowlist is wrong and gets corrected from that.

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

Run:  uv run --with torch --with numpy --with onnx --with onnxruntime --with rfdetr \
          python scripts/gate_onnx_device.py [--self-test]
"""
from __future__ import annotations

import argparse
import collections
import sys
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
    "Add", "AveragePool", "Cast", "Clip", "Concat", "Constant", "ConstantOfShape",
    "Conv", "Div", "Equal", "Erf", "Expand", "Flatten", "Gather", "Gemm", "Identity",
    "LayerNormalization", "MatMul", "MaxPool", "Mul", "Pad", "Pow", "Range",
    "ReduceMean", "ReduceSum", "Relu", "Reshape", "Shape", "Sigmoid", "Slice",
    "Softmax", "Split", "Sqrt", "Squeeze", "Sub", "Tanh", "Tile", "Transpose",
    "Unsqueeze", "Where",
}

#: Operators measured to be absent from the device half and present in the DETR
#: decoder. Named rather than merely excluded, so a regression reports the reason.
KNOWN_BLOCKERS = {
    "GridSample": "deformable attention; Hailo state it is unsupported with no plan to add it",
    "ScatterND": "keypoint-schema scatter in the decoder",
    "TopK": "two-stage query selection",
    "GatherElements": "dynamic gather, also what a GridSample decomposition produces",
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


def build_device_half(resolution: int):
    """The backbone and projector, which is exactly what would be compiled."""
    from rfdetr import RFDETRKeypointPreview

    model = RFDETRKeypointPreview(resolution=resolution)
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
    outs = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).run(
        None, {"image": x.numpy()})

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


def run(resolution, path):
    orig, folds = _fold_antialias()
    try:
        module = build_device_half(resolution)
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
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=576)
    ap.add_argument("--out", default="/tmp/device_half.onnx")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    print("device-half gate")
    problems = run(a.resolution, a.out)
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
