# rf-detr-cpp

A C++/[ggml](https://github.com/ggml-org/ggml) port of
[RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow, Apache-2.0 for
Nano–Large/Seg/Keypoint; PML 1.0 for XL/2XL): a DINOv2-backbone real-time
detection transformer supporting object detection, instance segmentation, and
keypoint detection. Modeled on
[trellis2cpp](https://github.com/weftspun/trellis2cpp) and
[see-through-cpp](https://github.com/weftspun/see-through-cpp): GGUF
conversion first, per-milestone validated ggml graphs, no PyTorch at runtime.

See [docs/architecture.md](docs/architecture.md) for the upstream model
architecture notes this port is working from, and
[docs/decisions/](docs/decisions/) for divergence/port-decision records.

## Status

Scaffolding stage — no validated milestones yet. Plan:

1. Backbone: DINOv2-with-windowed-attention encoder (Small/Base/Large configs)
2. Detection head: Deformable-DETR-style decoder (`lwdetr`/`transformer`)
3. Instance segmentation head (dot-product mask head)
4. Keypoint head (RFDETRKeypointPreview)
5. Finetuning/training (C++/ggml training loop) — phase 2, after 1-4 are
   validated for inference

Each milestone will be validated against a PyTorch reference
(`gen_reference/*.py` via `uv run` CPU torch → `tests/test_*.cpp` max-abs-diff
gate), following the sibling ports' pattern.

## Weights

Converted GGUF weights will be published on this repo's GitHub Releases
(zstd-split parts for files >2GB), mirroring see-through-cpp's release
layout. Upstream PyTorch checkpoints are plain `.pth` state dicts hosted at
`https://storage.googleapis.com/rfdetr/*.pth` (not HuggingFace).
