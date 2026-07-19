# 3. Phase-2 finetuning/training — design research + LayerNorm backward fix

* Status: in progress (backward prerequisite done, scope decided, loss/matching/dataloader not yet implemented)
* Date: 2026-07-18

## Context and Problem Statement

All three inference milestones (detection, segmentation, keypoints) are
done for the Nano-family variants plus RFDETRBase. The project's original
scope defers "phase 2" (finetuning/training support) until inference is
solid — that condition is now met. Per explicit instruction: research and
design this phase properly before writing any training code (unlike every
inference milestone so far, there's no precedent in the sibling ports —
`see-through-cpp` and `trellis2cpp` are both inference-only). What would a
C++/ggml training loop for this codebase actually require?

## Finding 1: ggml has a real training/autodiff framework — this isn't from scratch

Contrary to the "no precedent" framing above at the *port* level, `ggml`
itself ships genuine training infrastructure, not just inference primitives:

- `ggml_set_param()` marks trainable tensors; `ggml_build_backward_expand()`
  builds the gradient graph from a forward graph.
- `ggml_opt_step_adamw()` / `ggml_opt_step_sgd()` — optimizer step ops,
  operate as ordinary graph nodes (so they run through the same
  CPU/Vulkan backend dispatch as everything else).
- `ggml-opt.h` — a higher-level dataset/context/epoch API
  (`ggml_opt_dataset_*`, `ggml_opt_context_t`, `ggml_opt_fit`,
  `ggml_opt_epoch`) for a standard supervised training loop, with a
  progress-bar callback already provided.
- `ggml_cross_entropy_loss()` / `_back()` exist as first-class ops.

This means the *mechanics* of a training loop (graph-based backward pass,
optimizer step, epoch/batch iteration) are provided by ggml — this port
doesn't need to hand-roll autodiff or an optimizer.

## Finding 2: this port's *existing* graphs would crash `ggml_build_backward_expand` today

Checked ggml's backward-pass switch (`ggml_compute_backward` in
`third_party/ggml/src/ggml.c`) against every op this codebase actually
uses. **The `default` case is `GGML_ABORT("unsupported ggml op for backward
pass")`** — not a silent no-op, not a warning, a hard process abort. Ops
with **no backward case at all**, cross-referenced against where this port
uses them:

| Op (no backward) | Used where in this port | Impact |
|---|---|---|
| `GGML_OP_NORM` (plain LayerNorm via `ggml_norm`) | `layer_norm_affine`, `spatial_layer_norm_affine` (`ops.h`) — used in **every** block: backbone, decoder, segmentation, keypoints | **Blocking.** This is the single most-used normalization op in the whole port. `GGML_OP_RMS_NORM` *does* have backward (`ggml_rms_norm_back`, a dedicated op) — LayerNorm's is a well-defined, structurally similar formula, just not implemented upstream yet. |
| `GGML_OP_TOP_K` | `rfdetr_decoder`'s two-stage query selection | Not blocking in a useful sense — top-k *selection* is inherently non-differentiable (a discrete index choice); the reference DETR implementation doesn't backprop through it either, so this needs isolating from the trainable subgraph, not a ggml patch. |
| `GGML_OP_POOL_1D` | decoder's max-over-classes reduction, segmentation's would-be use (not currently used for anything trainable-adjacent) | Same as top-k — used for score ranking, not a differentiable path in the reference implementation either. |
| `GGML_OP_CLAMP` | `deform_attn.cpp`'s bilinear sampling (index clamping) | Expected: index computations aren't meant to be differentiable *through the index itself*, but the deformable-attention *sampling weights* (bilinear interpolation coefficients) mathematically **do** have a well-defined gradient w.r.t. the learned offsets — see Finding 3. |
| `GGML_OP_UPSCALE`/interpolate (bicubic/bilinear) | segmentation's spatial upsample, backbone's position-embed resize | The position-embed resize isn't trained (it's a fixed conversion-time-baked matrix, see `0002-position-embed-bicubic.md`) — not blocking. Segmentation's upsample of *learned* features would need backward if segmentation is finetuned. |

Ops confirmed **to have** backward support and used extensively here:
`MUL_MAT` (every `linear()`), `GET_ROWS` (deformable attention's corner
gather, embedding lookups), `IM2COL` (`conv2d`'s convolution), `SOFT_MAX`
(every attention block), `ADD`/`MUL`/`SUB`/elementwise, `RESHAPE`/`VIEW`/
`PERMUTE`/`CONT` (pure bookkeeping, always differentiable). This is most of
the FLOPs in the network — the gap is narrower than "nothing works," but
`GGML_OP_NORM`'s absence alone blocks naive end-to-end backprop through
literally every existing block.

## Finding 3: deformable attention needs a *hand-derived* backward, not just missing-op patches

`src/deform_attn.cpp`'s bilinear sampling is built from `ggml_floor` +
`ggml_clamp` + `ggml_get_rows` + elementwise weight multiplies — a
decomposition chosen because ggml has no native `grid_sample`-equivalent
(see `docs/decisions/decoder.md`). Even if every individual op in that
chain had a backward case, composing them via autodiff would differentiate
through `ggml_floor`'s corner-index computation, which is not how
deformable attention's real gradient is defined (the *indices* aren't
differentiable; the *bilinear weights* are, w.r.t. the continuous sampling
location which itself depends on the learned `sampling_offsets`). This
needs either: (a) a hand-derived custom backward op mirroring
`ggml_rms_norm_back`'s pattern (a dedicated `ms_deform_attn_back`
implementing the known-closed-form gradient from the original Deformable
DETR paper), or (b) marking the sampling-location computation as a
"stop-gradient" boundary and only training what's downstream/upstream of
it separately. Not a small task — flag as its own sub-milestone.

## RF-DETR's loss requirements (from `rfdetr` source, read this session)

- **Hungarian matching** (`scipy.optimize.linear_sum_assignment` in
  upstream, or an equivalent): a **non-differentiable, discrete assignment**
  between predicted queries and ground-truth targets, computed *outside*
  the gradient graph as ordinary CPU preprocessing on detached tensors —
  this is standard for every DETR-family model and is not a ggml-autodiff
  concern at all; it just needs a plain C++ (or even a small Python
  preprocessing step feeding fixed assignment indices into the ggml graph)
  implementation, decoupled from backward-pass concerns.
- **Detection loss**: focal loss (classification) + L1 + GIoU (box
  regression), applied only to matched pairs. Focal/L1 are elementwise —
  fine once `NORM` backward exists upstream of them. GIoU needs its own
  elementwise formula (intersection/union of predicted vs. target boxes) —
  not yet checked whether this decomposes into already-backward-capable
  ops; likely yes (min/max/mul/div/add are all in the supported set) but
  unverified.
- **Segmentation loss**: (not yet read in detail — deferred, this
  research pass focused on the backward-op-support blocker first since
  it's a hard prerequisite for *any* task's training, not task-specific).
- **Keypoint loss**: partially read this session
  (`compute_l1_keypoint_loss`, `docs/decisions/keypoints.md`'s earlier
  research) — an L1 term + BCE (findable/visible) + a Gaussian NLL with
  Cholesky-parameterized precision. All elementwise/reduction ops, same
  "blocked only by NORM" caveat as detection.

## Prerequisite resolved: `layer_norm_affine_diff` (`src/ops.{h,cpp}`)

Rather than patching vendored ggml (adding a new `ggml_norm_back` op), the
LayerNorm-backward gap from Finding 2 is resolved by decomposing
`ggml_norm` into primitive ops that each individually have a backward case.
This turned out to need more care than "pick primitives with a `case` in
`ggml_compute_backward`'s switch" — two additional, non-obvious shape
constraints surfaced during validation (`tests/test_norm_backward.cpp`):

- **`ggml_sub`/`ggml_div`'s backward doesn't broadcast.** `GGML_OP_ADD` and
  `GGML_OP_MUL`'s backward call `ggml_repeat_back` on the gradient before
  writing it to a smaller (broadcast) `src1`; `GGML_OP_SUB`/`GGML_OP_DIV`
  don't — they assume `src1` is already the output shape and hit
  `GGML_ASSERT(ggml_are_same_shape(src1, cgraph->grads[isrc1]))`. Centering
  (`x - mean`, `mean` is `(1,T,N)` broadcasting against `x`'s `(C,T,N)`)
  and normalizing (`centered / std`) both broadcast, so both are rewritten
  through the add/mul path instead: `add(x, scale(mean, -1))` and
  `mul(centered, reciprocal)` where `reciprocal = div(ones, std)` (that div
  is same-shape, non-broadcast — safe).
- **`ggml_mean`'s own backward assumes a true scalar output.** It's
  implemented via `ggml_add1_or_set`, which asserts its operand is a scalar
  (`ggml_is_scalar`) — fine for the "mean of a whole tensor down to one
  number" case ggml's own code uses it for, but `layer_norm_affine_diff`
  needs a per-row mean (output `(1,T,N)`, not `(1,1,1,1)`). `ggml_sum_rows`
  has the identical forward reduction shape but a shape-general backward
  (plain `ggml_repeat`, no scalar assumption), so mean is computed as
  `scale(sum_rows(x), 1/C)` instead of `ggml_mean(x)`.

Validated in `tests/test_norm_backward.cpp`: forward output matches
`layer_norm_affine` (fused `ggml_norm`) to 7.2e-7; `ggml_build_backward_expand`
on a graph using `layer_norm_affine_diff` completes without aborting; the
resulting gradient matches a central-difference numerical check to <0.1%
relative error on 5 sampled elements. The existing fused `layer_norm_affine`
is untouched (still used by every inference graph) — `layer_norm_affine_diff`
is a separate, opt-in function for graphs that need backward.

## Scope decision: detection-head-only finetune (decided)

**Chosen scope: freeze the DINOv2 backbone AND the transformer decoder;
finetune only `class_embed`/`bbox_embed`** (the final linear/MLP heads
reading the last decoder layer's hidden state). This is the smallest
useful slice flagged in the prior draft of this doc, and it's the one that
sidesteps the most open work:

- Freezing the decoder means `ms_deform_attn`'s backward (Finding 3, a
  hand-derived analytical gradient, not a small task) is **not needed** —
  gradients never need to flow through it since nothing upstream of
  `class_embed`/`bbox_embed` requires a gradient.
- Freezing the backbone means the windowed-attention machinery
  (`dinov2_backbone`) also never needs backward support.
- `class_embed`/`bbox_embed` are plain `linear()`/`mlp()` calls — already
  fully backward-capable (`MUL_MAT`, `ADD` both have backward cases) with
  **zero new primitive work** beyond what's already validated.
- This scope still exercises the real prerequisite this phase closed:
  `class_embed`/`bbox_embed`'s INPUT (the decoder's final `hidden_states`)
  is itself the output of a `layer_norm_affine`-heavy stack, but since that
  stack is frozen (no gradient requested through it), `layer_norm_affine`
  (the fast, backward-incapable fused version) can stay as-is there too —
  `layer_norm_affine_diff` isn't even needed for THIS scope specifically,
  though it remains available (and validated) for whenever the scope
  widens to include the decoder.

Remaining steps for this scope, in order:

1. Loss: focal loss (classification) + L1 + GIoU (box regression) —
   elementwise/reduction ops, all already backward-capable
   (`SUB`/`MUL`/`DIV`/`SQR`/`SQRT`/`MEAN`/`SUM` per Finding 2's audit); GIoU
   needs its own formula checked but likely decomposes the same way.
2. Hungarian matching (`scipy.optimize.linear_sum_assignment` or
   equivalent) — non-differentiable CPU preprocessing on detached
   `pred_boxes`/`pred_logits`, decoupled from the ggml graph entirely; can
   be plain C++ (e.g. a straightforward Jonker-Volgenant/Hungarian
   implementation) or shell out to a small Python step, since it never
   touches the gradient graph.
3. Dataset/dataloader (COCO-format annotations → ggml input tensors) — not
   researched yet.
4. `ggml_opt_step_adamw`/`ggml_opt_epoch` wiring (Finding 1's existing
   framework) once 1-3 land.

Widening the scope later (decoder trainable, then backbone trainable) is a
separate future decision — each widening reuses `layer_norm_affine_diff`
directly (already validated) except the decoder-trainable step, which
additionally needs Finding 3.

## Consequences

This doc intentionally stops short of a full training loop, per the
explicit instruction not to rush Phase B implementation. The load-bearing
finding is Finding 2: **training was blocked on a real ggml gap
(`GGML_OP_NORM` backward), not on this port's own code** — that gap is now
closed (`layer_norm_affine_diff`, validated), which unblocks every
finetuning scope from "just the classifier head" to "everything" (all of
them route through at least one LayerNorm). The remaining prerequisite
before any RF-DETR-specific training-loop code is Finding 3's deformable-
attention backward, and only if the chosen scope needs the decoder
trainable.
