# `DEVICE_OPS`, checked against the exports and against the compiler

`gate_onnx_device.py` carries a 40-operator allowlist and its docstring makes a specific,
checkable promise about how entries got there:

> Every entry below was observed in a device-half export that the numeric check passed;
> nothing is here on the strength of a datasheet.

This is that check. It is the first time the list has been compared either to the exports it
claims to come from or to what Hailo documents, and it does not come out clean.

Reproduce with `python scripts/hailo_supported_ops.py --verify <dir-of-exports>`, where the
directory holds `device_half_nw1.onnx` and `device_half_default.onnx` from
`gate_onnx_device.py`. `build/onehot/*.onnx` is picked up from the tree.

## Two axes, because one is not enough

**Observed** means the operator appears in an ONNX export on this desk. **Documented** means
Hailo names it in the Dataflow Compiler 5.3.0 guide, Table 3 (layers) or Table 4 (activations)
or inside one of Table 4's multi-operator patterns. `scripts/hailo_supported_ops.py` holds
those tables; the guide itself is confidential and is not checked in.

|                | documented     | undocumented                   |
| -------------- | -------------- | ------------------------------ |
| **observed**   | 18 — solid     | 7 — pass by folding            |
| **unobserved** | 10 — plausible | **5 — nothing supports these** |

    observed AND documented (18)
      Add Concat Conv Div Equal Erf LayerNormalization MatMul Mul OneHot
      ReduceSum Reshape Sigmoid Slice Softmax Split Sqrt Transpose

    observed, documented NOWHERE (7)
      Cast Constant ConstantOfShape Expand Shape Squeeze Where

    unobserved, documented by Hailo (10)
      AveragePool Clip Gemm MaxPool Pad Pow ReduceMean Relu Sub Tanh

    neither observed nor documented (5)
      Flatten Gather Identity Range Unsqueeze

## What each cell means

**The 7 that pass while documented nowhere are the interesting ones.** They are shape plumbing,
and they pass because a fixed-resolution graph folds them away before anything reaches
hardware — not because the accelerator implements them. That is support with a precondition,
and the precondition is invisible in the allowlist. Two of these are already known to break
when the fold cannot happen: a standalone `Where` is rejected by the parser, and a
value-changing `Cast` is accepted and then not performed, returning an `f32 -> s32 -> f32`
round trip untruncated and wrong by 0.875.

**The 5 in the last cell contradict the docstring.** `Flatten`, `Gather`, `Identity`, `Range`
and `Unsqueeze` appear in no export on this desk and in neither Hailo table. The docstring says
nothing is there on the strength of a datasheet; for these five there is no evidence of either
kind to point at.

**Stated rather than implied: this is not proof they are wrong.** Exports at resolutions other
than 576 were not retained, so "not observed here" is weaker than "never observed". What the
check establishes is that the claim is currently **unverifiable for 15 of 40 entries**, five of
them with no documentary support either. The honest repair is to record the export each entry
came from, so the claim is checkable rather than remembered.

**Nothing is missing in the dangerous direction.** Every operator in every export on disk is
already in `DEVICE_OPS` — the allowlist is a superset of what was seen, so the gate does not
pass an operator it never considered.

## The union is smaller than the list

    device_half_nw1.onnx        22 distinct operators
    device_half_default.onnx    23
    build/onehot/OneHot.onnx     8
    union across all exports    26
    DEVICE_OPS                  40

The default-windows export is the one DFC 5.3.0 rejects, on `Tile` — _"unsupported concat over
axis batch; fold it at batch 1"_ — and it is included here anyway, because an export that the
compiler refuses still shows which operators the exporter emits, which is what this table is
counting.
