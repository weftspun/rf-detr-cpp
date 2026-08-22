# 5. What an edge NPU compiler does with this backbone

- Status: accepted
- Date: 2026-08-21

## Context and Problem Statement

The question was whether the keypoint model can deploy to a Hailo-10H class
accelerator (ASUS UGen300, 8 GB, 20 TOPS INT8 / 40 INT4). Two facts this
repository already recorded turn out to decide it, and neither was written with
a compiler in mind:

- `backbone-windowing.md`: windowing happens once, at the embeddings stage,
  **by expanding the batch dimension**.
- `0002-position-embed-bicubic.md`: position embeddings are bicubic+antialias
  interpolated to the runtime grid at every forward pass.

Both are correct descriptions of the model. Both are also exactly what Hailo
Dataflow Compiler 5.3.0 refuses.

## Decision

Compile the backbone with `num_windows=1`. Everything else follows from
measurement rather than from choice.

### The three findings, in the order they were hit

**1. The antialias resize does not export at all.**
`aten::_upsample_bicubic2d_aa` has no ONNX symbolic at opset 17 through 20. It
reads a learned parameter and a fixed target size and never the input image, so
at one resolution its output is constant and folding it is exact rather than an
approximation. `scripts/gate_onnx_device.py` applies the fold, and treats a fold
that never fires as a FAIL, because a silent skip reads as a pass.

**2. `Tile` is rejected, and it is the CLS token per window.**

    UnsupportedConcatLayerError in op /encoder/encoder/embeddings/Tile:
    Unsupported concat over axis batch

Constant at a fixed input shape, [1,1,384] to [4,1,384]. Folded exactly by
`scripts/fold_static_tile.py`, measured max|diff| 0.0. `onnxsim` alone does not
remove it: simplifying at a fixed input shape takes 868 nodes to 569 and clears
`Expand`, `ConstantOfShape`, `Where` and `Shape`, and `Tile` survives that pass.

**3. Folding it is not sufficient. Windowing is the actual blocker.**
The concat that joins the CLS token to the patch tokens fails next. Setting
`num_windows=1` removes both constructs at the source: `Tile` count 0, and the
embeddings concat is absent from the graph entirely.

| graph | DFC 5.3.0, hailo10h |
| ----- | ------------------- |
| `num_windows=2`, 868 nodes | REJECTED |
| `num_windows=1`, 825 nodes | **PARSE OK** |

Same model, same resolution, same fold applied to both.

### Why windowing and not the CLS token

Hailo's own Model Zoo ships timm ViTs and parses them whole, with no start or
end node overrides. Their ViT-Base is 197 tokens, and 197 is prime. So neither a
CLS token nor a token count that is not a spatial grid is a blocker. What this
backbone has that theirs does not is the batch-expanded windowing that
`backbone-windowing.md` describes.

### What it costs

Global attention replaces four windows, measured at 576: **1.35x wall-clock**,
238.4 ms against 175.9 ms. MAC counting initially disagreed and reported 0.83x;
that number was wrong because ONNX shape inference failed on 51 nodes of the
`num_windows=1` graph and silently undercounted. Simplifying first resolves the
shapes, and the corrected count is 107.9 GMAC against 76.5, a 1.37x ratio that
agrees with wall-clock.

`num_windows=1` also changes the attention pattern, so the checkpoint needs
retraining. RFD 107a is training a new head regardless.

## How to diagnose this compiler

Two habits cost several attempts and are recorded so they are not repeated.

**Its first reported error is often not the real one.** A plain parse of the
unfolded graph raises `IndexError: list index out of range` inside LayerNorm axis
mapping, from an empty `input_format`. That is a downstream symptom of the
simplify-and-retry path. Passing `end_node_names` makes it report the honest
error and a recommendation.

**Minimal reproductions are unreliable here.** Isolated rank-3 LayerNorm, an
isolated window reshape, and an isolated CLS concat producing 577 tokens all
parse cleanly, while the same constructs fail inside the real graph. Format
inference is whole-graph rather than local, so a toy that passes proves nothing.

## The two gates

`scripts/gate_onnx_device.py` runs on macOS with no accelerator and no compiler,
and checks export, numerics against the `test_keypoints` bound, and every
operator against `DEVICE_OPS`. `scripts/gate_dfc_parse.py` runs the real
compiler on Linux and exits non-zero when the two disagree, including when the
parse succeeded.

That pair has already paid for itself once. The macOS gate passed the device half
with `Tile x1`; the compiler rejected it; `DEVICE_OPS` was the side that was
wrong and was narrowed. `DEVICE_OPS` is a claim about a compiler, and only the
compiler can check it.

## Consequences

The device half is 868 nodes, 23 operators, 25 MB INT8, and carries 95% of the
wall-clock at 1200. Everything the compiler refuses -- 8 GridSample, 17
ScatterND, TopK, GatherElements -- sits in the DETR decoder, which is 4.7%.

Note `forward_export` matters when measuring any of this. The training forward is
12,427 nodes at 1272 and the deployment graph is 3,632, with ScatterND falling
from 85 to 17 and `IsNaN`/`IsInf` disappearing.
