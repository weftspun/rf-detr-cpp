# 2. Position-embedding bicubic+antialias interpolation (RFDETRBase)

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

`patch_size==14` variants (RFDETRBase, RFDETRLarge-deprecated) run at a
resolution that differs from DINOv2's native 518px training grid (37×37
patches) — RFDETRBase runs at 560px (40×40 patches). Upstream genuinely
bicubic-interpolates the position embeddings to the runtime grid at every
forward pass (`interpolate_pos_encoding`,
`rfdetr/models/backbone/dinov2_with_windowed_attn.py`), with
`antialias=True` on non-MPS devices. Every variant validated before this
(Nano, SegNano, KeypointPreview) uses `patch_size!=14` and skips this
entirely (`docs/decisions/backbone-windowing.md`), so it had been deferred
repeatedly. How should this port reproduce PyTorch's antialiased bicubic
resize, given ggml has no matching built-in?

## Why not `ggml_interpolate(..., GGML_SCALE_MODE_BICUBIC)`

ggml's bicubic mode exists but **cannot be combined with antialiasing** —
`ggml_interpolate` hard-asserts `GGML_SCALE_FLAG_ANTIALIAS` is only valid
with `GGML_SCALE_MODE_BILINEAR` (confirmed by triggering the assert
directly). And the difference isn't negligible: even though this is
*upsampling* (37→40, where classic image-processing antialiasing
intuitively "shouldn't matter" since there's no decimation) — PyTorch's own
antialias path measurably diverges from its non-antialias path here (max
abs diff 0.297 on a synthetic test), and ggml's plain bicubic matches
PyTorch's **non**-antialiased bicubic almost exactly (1.4e-5) but not the
antialiased one upstream actually uses.

## Considered options

1. Hand-transcribe PyTorch's antialias bicubic kernel formula (Keys cubic,
   `a=-0.5`, kernel support widened by `scale` when `scale>=1`) into a new
   ggml custom CPU op (`ggml_map_custom`/`ggml_custom_4d`).
2. Same, but as a GPU (Vulkan) compute shader (user's initial ask, using
   Slang) — requires wiring up Vulkan for this repo for the first time plus
   a SPIR-V build step, for an op that resizes a 37×37 tensor once per
   inference.
3. **Precompute the exact resize as a dense weight matrix, probed directly
   from PyTorch's real implementation (not hand-derived), and apply it via
   two `ggml_mul_mat` calls (separable resize: width pass, height pass).**

## Decision Outcome

Option 3. Scoped down from option 2 after clarifying the goal isn't
"establish a GPU pipeline," just "make this one op work" — a precomputed
matrix + `ggml_mul_mat` needs **no new op at all**, reuses the same
primitive already used everywhere else in this port (`linear()`,
`ms_deform_attn`'s outer products, etc.), and — the actual point of the
user's "GPU-first" ask — runs on whichever backend is active (Vulkan or
CPU) automatically, with zero backend-specific code, the moment this repo's
build gains a Vulkan target. No custom kernel, no shader compilation step,
no transcription risk.

**Extraction method** (`extract_bicubic_aa_resize_matrix` in
`scripts/convert_dinov2_to_gguf.py`): since PyTorch's bicubic-AA resize is
separable (independent 1-D filters along each axis), probe it with an
identity-impulse input per source index, using a *constant* (not
degenerate size-1) value along the orthogonal axis so a sum-to-1-normalized
filter leaves it invariant — this isolates one axis's exact filter response
without a degenerate-dimension edge case (an earlier attempt using a
literal width-1 probe tensor produced garbage: PyTorch's antialias path
apparently doesn't handle a size-1 spatial dimension the way a naive
per-axis probe assumes). Verified the extracted `(40,37)` matrix reproduces
the real 2-D `F.interpolate(..., antialias=True)` output on an
independent random test to 4.8e-7 before trusting it.

**ggml application** (`src/backbone.cpp`, gated on
`BackboneParams::native_grid != 0` and differing from the runtime grid):
two `ggml_mul_mat` passes against the same baked `(native_grid, gw)` weight
matrix — width pass (contract over native width, produce runtime width),
then a permute + height pass (contract over native height, produce runtime
height) — exactly mirroring the `linear()` convention already used
throughout this codebase (`ggml_mul_mat(weight, x)`).

## Consequences

Validated end-to-end for RFDETRBase's backbone (`test_backbone_base.cpp`):
all 4 tapped feature maps match the real checkpoint to max-abs-diff ≤1.6e-4
(gate 5e-2) — the same tight scale as every non-interpolated backbone
milestone, confirming the precomputed-matrix approach reproduces PyTorch's
antialiased bicubic essentially exactly, not just approximately.
`BackboneParams` gained one field (`native_grid`, default 0 = no
interpolation); every previously-validated config is unaffected
(re-confirmed: `test_backbone`, `test_decoder`, `test_segmentation`,
`test_keypoints` all still pass after this change).

The precomputed matrix is baked once per (native_grid, runtime_grid) pair
at GGUF-conversion time — it's specific to one checkpoint's training grid
and one runtime resolution, not recomputed per inference call. This is
correct for this project's scope (one canonical resolution per config) but
would need generalizing (e.g. a small library of precomputed matrices, or
falling back to option 1/2 for a real runtime-configurable resolution) if
arbitrary runtime resolutions were ever required.
