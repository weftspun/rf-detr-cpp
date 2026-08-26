"""Where does the device half actually RUN under Core ML, and does it still agree.

WHY THIS EXISTS BESIDE `gate_onnx_device.py`. That gate measures onnxruntime's
`CoreMLExecutionProvider` and warns, in its own output, that "CoreML partitions a graph and
puts unsupported nodes back on the CPU, so a fast row is not proof the accelerator took the
whole model". The warning is right and it cuts both ways: a SLOW row is not proof the
accelerator took the whole model either.

`rfd1122-plan.usda` carries `neuralEngineUsefulForBackbone = 0` from that slow row -- 1685.1
ms at 576 against the CPU's 476.4, reproduced here at 3758.9 against 710.7 with
`--num-windows 1`. The row is consistent with two opposite worlds: the Neural Engine ran the
graph badly, or the Neural Engine barely saw it. The plan records the first and the evidence
cannot distinguish them.

WHAT A NATIVE CONVERSION ADDS. Converting the traced module rather than the `.onnx` gives an
`mlprogram` whose every operation can be interrogated with `MLComputePlan`, which reports the
device Core ML prefers per operation. The question stops being "how fast was it" and becomes
"how much of it ran where, and how fast was that".

coremltools 9 dropped its ONNX front end, so this converts the same torch module
`gate_onnx_device.build_device_half` hands the exporter. Same graph, different back end.

THE NUMERIC BOUND IS THE PORT'S OWN, 4.2e-3 from `test_keypoints`, imported rather than
restated so the two gates cannot disagree about what passing means. Speed without agreement
is not deployability: the ONNX CoreML row failed on both, and either alone is a finding.
"""

import argparse
import collections
import os
import statistics
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_onnx_device import KEYPOINT_TOL, _fold_antialias, build_device_half

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from coremltools.converters.mil.frontend.torch import ops as _torch_ops
from coremltools.converters.mil.mil import Builder as mb


def _install_rank1_const_cast_shim():
    """Work around a coremltools bug in `_cast`, and leave the graph alone.

    THE BUG IS IN THE CONVERTER, NOT IN THE MODEL. `_cast` validates its input with its own
    comment -- "Input must either be a scalar or a (1 x 1 x ... x 1) tensor" -- and then
    takes two branches. The non-constant branch squeezes:

        x = mb.squeeze(x=x, name=node.name + "_item")

    The compile-time-constant branch does not:

        res = mb.const(val=dtype(x.val), name=node.name)

    `int(np.array([48]))` raises "only 0-dimensional arrays can be converted to Python
    scalars", so a rank-1 constant its own validation permits reaches a call that rejects
    it. The device half hits this at `encoder/encoder/embeddings/68`, where transformers'
    Dinov2 `interpolate_pos_encoding` computes `sqrt_num_positions` as shape (1,) holding
    48 rather than as a scalar.

    THIS SHIM CHANGES NO SEMANTICS. It squeezes a size-1 constant before the same cast the
    converter was already attempting, which is what the sibling branch does for the same
    shape. Nothing about the traced module, its operators, or its numerics moves, and the
    numeric check below is what confirms that rather than this paragraph.

    It is applied here rather than vendored into coremltools so the fault stays visible: a
    patched dependency looks like a working one, and this way the workaround is deleted by
    grep when upstream fixes it.
    """
    original = _torch_ops._cast

    def cast_with_rank1_consts(context, node, dtype, dtype_name):
        try:
            return original(context, node, dtype, dtype_name)
        except TypeError as exc:
            if "0-dimensional" not in str(exc):
                raise
            x = _torch_ops._get_inputs(context, node, expected=1)[0]
            val = np.asarray(x.val)
            if val.size != 1:
                raise
            context.add(mb.const(val=dtype(val.reshape(-1)[0]), name=node.name), node.name)

    _torch_ops._cast = cast_with_rank1_consts

# Confining a run to a unit set is how a comparison stays honest: ALL lets Core ML pick and
# then the placement column explains the timing, while the narrower sets answer "what does
# this device alone give me".
UNITS = {
    "all": ct.ComputeUnit.ALL,
    "ane": ct.ComputeUnit.CPU_AND_NE,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu": ct.ComputeUnit.CPU_ONLY,
}

SHORT = {
    "MLNeuralEngineComputeDevice": "ane",
    "MLGPUComputeDevice": "gpu",
    "MLCPUComputeDevice": "cpu",
}


def placement(pkg, units):
    """Per-operation device assignment, constants excluded from the denominator.

    `MLComputePlan` needs a COMPILED model. Handed an `.mlpackage` it aborts the process from
    libc++ rather than raising, so the compile happens here where no caller can miss it.
    Constants carry no device, and leaving them in makes a graph of mostly weights look well
    placed wherever its arithmetic went.
    """
    loaded = ct.models.MLModel(pkg, compute_units=units)
    plan = MLComputePlan.load_from_path(loaded.get_compiled_model_path(), compute_units=units)
    counts, consts = collections.Counter(), 0
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name == "const":
            consts += 1
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        counts[SHORT.get(type(usage.preferred_compute_device).__name__, "unreported")
               if usage else "unreported"] += 1
    return counts, consts, sum(counts.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resolution", type=int, default=576)
    ap.add_argument("--num-windows", type=int, default=1)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--units", nargs="+", default=["all", "ane", "gpu", "cpu"])
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args(argv)

    _install_rank1_const_cast_shim()
    module = build_device_half(args.resolution, args.num_windows).eval()
    x = torch.randn(1, 3, args.resolution, args.resolution)

    # THE SAME FOLD THE ONNX GATE APPLIES, borrowed rather than reimplemented so the two
    # gates cannot disagree about what graph they measured. coremltools has no converter
    # for `_upsample_bicubic2d_aa`, and the fold removes the operator instead of emulating
    # it: at a fixed resolution the resize reads a parameter and a size rather than the
    # image, so its output is constant and folding is exact. The reference below is
    # computed with the fold active, exactly as `export_and_check` does.
    orig_interpolate, folds = _fold_antialias()
    try:
        with torch.no_grad():
            reference = module(x)
        traced = torch.jit.trace(module, x)
    finally:
        F.interpolate = orig_interpolate
    reference = [t.numpy() for t in (reference if isinstance(reference, (list, tuple))
                                     else [reference])]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape)],
        convert_to="mlprogram",
        compute_precision=(ct.precision.FLOAT32 if args.precision == "fp32"
                           else ct.precision.FLOAT16),
        minimum_deployment_target=ct.target.macOS15,
    )
    pkg = os.path.join(tempfile.mkdtemp(prefix="devhalf_"), "device_half.mlpackage")
    mlmodel.save(pkg)

    print(f"device half: resolution {args.resolution}, num_windows {args.num_windows}, "
          f"{folds['n']} resize folded, {args.precision}")
    print(f"bound: {KEYPOINT_TOL:.1e} (the port's own, from test_keypoints)")
    print()
    header = (f"{'units':<6} {'ms med':>9} {'max|diff|':>11} {'bound':>6}  placement")
    print(header)
    print("-" * len(header))

    cpu_ms, rows = None, []
    for name in args.units:
        counts, consts, total = placement(pkg, UNITS[name])
        loaded = ct.models.MLModel(pkg, compute_units=UNITS[name])
        feed = {"x": x.numpy().astype(np.float32)}
        out = loaded.predict(feed)
        loaded.predict(feed)                      # warm-up discarded: the first predict pays
        samples = []                              # for lazy compile and weight staging
        for _ in range(args.reps):
            t = time.perf_counter()
            loaded.predict(feed)
            samples.append((time.perf_counter() - t) * 1e3)
        ms = statistics.median(samples)
        if name == "cpu":
            cpu_ms = ms

        # PAIR BY AGREEMENT, NOT BY SHAPE. The device half returns two tensors of the
        # SAME shape, (1, 256, 48, 48), so matching on shape alone silently compared one
        # reference against the other output and reported 2.5e+00 on every row -- including
        # the CPU row, which is what exposed it: Core ML's own CPU cannot disagree with
        # PyTorch by 2.5, so the comparison was wrong rather than the device. Each output is
        # assigned to its closest reference, and each reference is used once.
        got = {k: np.asarray(v) for k, v in out.items()}
        worst, pairing, taken = 0.0, [], set()
        for idx, ref in enumerate(reference):
            best_key, best_err = None, None
            for k, g in got.items():
                if k in taken or g.shape != ref.shape:
                    continue
                err = float(np.max(np.abs(g - ref)))
                if best_err is None or err < best_err:
                    best_key, best_err = k, err
            if best_key is None:
                raise RuntimeError(f"no output matches reference {idx} of shape {ref.shape}")
            taken.add(best_key)
            pairing.append(f"{best_key}~ref{idx}:{best_err:.2e}")
            worst = max(worst, best_err)
        verdict = "ok" if worst <= KEYPOINT_TOL else "FAIL"
        rows.append((name, ms, worst, verdict, dict(counts), pairing))
        print(f"{name:<6} {ms:>9.1f} {worst:>11.3e} {verdict:>6}  {dict(counts)}")

    if cpu_ms:
        print()
        for name, ms, worst, verdict, counts, pairing in rows:
            print(f"  {name:<5} {cpu_ms / ms:>6.2f}x CPU   {' '.join(pairing)}")
    print()
    print("A placement column is the point. A slow row whose ANE fraction is near zero is a")
    print("statement about partitioning; a slow row at 1.000 is a statement about the device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
