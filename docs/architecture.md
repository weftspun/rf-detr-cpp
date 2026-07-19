# RF-DETR upstream architecture notes

Source: `roboflow/rf-detr` (`develop` branch), package under `src/rfdetr/`.
These are porting notes, not upstream documentation — verify against the
actual upstream source at conversion time, pin a commit SHA once conversion
scripts start reading it.

## Backbone — DINOv2 with windowed attention

`src/rfdetr/models/backbone/dinov2_with_windowed_attn.py` — a local fork of
HF `transformers`' `Dinov2WithRegisters` (targets transformers v5 API).
Configs: `src/rfdetr/models/backbone/dinov2_configs/*.json`.

| Variant | hidden | layers | heads | patch | registers |
|---|---|---|---|---|---|
| Small | 384 | 12 | 6 | 14 | 4 |
| Base  | 768 | 12 | 12 | 14 | 0 (plain `Dinov2Model`) |
| Large | 1024 | 24 | 16 | 14 | 4 |

- MLP ratio 4, GELU, LayerScale, learned/interpolated absolute position
  embeddings (`nn.Parameter([1, num_patches+1, hidden])`, bicubic-interpolated
  per input resolution). **Not RoPE** — this differs from trellis2cpp's
  DINOv3 port, which uses axial 2D RoPE. Do not reuse that code path as-is.
- Windowed attention: per-variant `num_windows` (e.g. Nano/Small/Medium/Seg=2,
  Base=4) reshapes the batch into `num_windows²` spatial windows for layers
  listed in `window_block_indexes`; other layers run full/global attention.
  Input resolution must be divisible by `patch_size * num_windows`.
- Runtime `patch_size` in `ModelConfig` (12/14/16, varies per variant) can
  differ from the checkpoint's native patch=14 — position embeddings are
  re-interpolated accordingly.
- `out_feature_indexes` / `projector_scale` (e.g. `["P3","P5"]` for Large,
  `["P4"]` for Base/Nano/2XLarge) select which transformer-layer outputs feed
  `src/rfdetr/models/backbone/projector.py`, which projects/resizes to
  multi-scale P3/P4/P5 feature maps for the decoder. The backbone itself is
  single-scale ViT; multi-scale is synthesized by the projector.

## Detection head/decoder

Deformable-DETR-style (`models/ops/modules/ms_deform_attn.py` for the CUDA/
PyTorch deformable-attention op), driven from `models/lwdetr.py` (`LWDETR`)
and `models/transformer.py`. Relevant `ModelConfig` fields (`config.py`):
`dec_layers` (2-6, variant-dependent), `hidden_dim` (256 for Base/Nano, 384
for Large), `sa_nheads`/`ca_nheads`, `dec_n_points` (deformable sample points
per head/level), `num_feature_levels = len(projector_scale)`, `num_queries`
(100-300). `group_detr=13` is GroupDETR-style duplicated query groups used
only during training (matching/loss), not needed for inference.

## Instance segmentation (RF-DETR-Seg)

`models/heads/segmentation.py`, `SegmentationHead`: upsamples backbone
spatial features `(B,C,H,W)` bilinearly by `downsample_ratio`, runs
`DepthwiseConvBlock`s (dwconv3x3 + LN-style pointwise linear), projects both
spatial features and per-decoder-layer query features, then produces masks
via `einsum("bchw,bnc->bnhw", spatial_proj, query_proj) + bias` — a
dot-product mask head per query, computed once per decoder layer.

## Keypoint detection (RFDETRKeypointPreview)

`models/heads/keypoints.py`: an MLP head predicting `(x, y, visibility)`-like
values per keypoint per query, `num_keypoints_per_class` (COCO-17 by
default). Dedicated matching cost in `keypoint_oks.py`
(`KeypointTrainConfig`) — training-only, not needed for the inference port.

## Checkpoints

Plain PyTorch `.pth` state dicts (not HuggingFace/safetensors), hosted at
`https://storage.googleapis.com/rfdetr/*.pth`, enumerated in
`src/rfdetr/assets/model_weights.py`. Tensor names are raw `nn.Module`
state-dict names (e.g. `backbone.encoder.layer.{i}...`) — GGUF conversion
should preserve these verbatim, per the sibling-ports convention.

## Export paths (useful for shape cross-checking, not for conversion)

`src/rfdetr/export/_onnx/exporter.py`, `_tflite/`, `_tensorrt.py`, plus
`forward_export()` methods on `LWDETR`/`SegmentationHead` (fixed
control-flow paths built for tracing).

## Reference: trellis2cpp's DINOv3 encoder port

`weftspun/trellis2cpp` already has a validated DINOv2/DINOv3 ViT-L/16 encoder
(relative L2 ≤ 7e-7 vs. HF reference) whose **ggml plumbing** is a reusable
template — but whose **attention/pos-embed code is not directly reusable**
for RF-DETR's backbone:

- Axial 2D RoPE (theta=100) over patch centers, not learned absolute
  pos-embed — RF-DETR needs bicubic-interpolated absolute pos-embed instead.
- No windowed attention (full attention every layer) — RF-DETR needs
  per-layer window/global switching keyed on `window_block_indexes`.
- q/v/o biases, no k-bias; exact (erf) GELU; register tokens; LayerScale both
  branches; affine-free final LayerNorm — these parts likely do transfer.

Inference lives inline in `trellis2.cpp` (`trellis2_dino_encode`, roughly
lines 1286-1494) and structs in `trellis2.h` (roughly lines 570-700), not a
separate file. Conversion script `convert_dino_to_gguf.py` reads HF
safetensors + config.json directly with numpy (no gguf/torch runtime dep),
writes tensors under raw HF state-dict names, hyperparams as
`trellis2.dino.*` GGUF KV pairs, `ftype 0=f32 / 1=f16` (matrices f16,
1-D/embedding tensors always f32). Has a tap mechanism
(`trellis2_dino_taps`) dumping named intermediates matching a
`scripts/dump_dino_reference.py` reference dumper — same milestone-diff
methodology this project follows.

**Port plan**: reuse the linear/layernorm/SDPA-lambda plumbing and the
tap/GGUF-writer conventions; rewrite pos-embed (interpolated absolute) and
add windowed-attention batch reshaping; support the no-register Base variant
and variable patch_size (12/14/16).
