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
architecture notes this port is working from,
[docs/decisions/](docs/decisions/) for divergence/port-decision records, and
[docs/decisions/0001-open-work.md](docs/decisions/0001-open-work.md) for the
consolidated open-task checklist across all milestones.

## Status

| Milestone | Test | Result |
|---|---|---|
| DINOv2-windowed-attention backbone (RFDETRNano, 4 taps) | `test_backbone` | ≤2.5e-4 (gate 5e-2) |
| Multi-scale projector (C2f fusion, RFDETRNano) | `test_projector` | 1.0e-5 |
| Deformable attention core (isolated, synthetic) | `test_deform_attn` | 0.0 (exact) |
| **Object detection end-to-end (RFDETRNano)** | `test_decoder` | boxes 3.3e-4, logits 7.3e-4 |
| **Instance segmentation end-to-end (RFDETRSegNano)** | `test_segmentation` | boxes 1.1e-2, logits 2.9e-3, masks 0.109 (gate 0.15, see `docs/decisions/segmentation.md`) |
| **Keypoint detection end-to-end (RFDETRKeypointPreview)** | `test_keypoints` | boxes 3.5e-3, logits 9.3e-4, keypoints 4.2e-3 |
| RFDETRBase backbone (patch_size==14, bicubic+antialias pos-embed interpolation) | `test_backbone_base` | ≤1.6e-4 |

**All three inference milestones are done for the Nano-family variants:
object detection, instance segmentation, and keypoint detection.** Now
extending to other model-size variants — see `docs/decisions/0001-open-work.md`.

1. ~~Backbone: DINOv2-with-windowed-attention encoder~~ — validated for
   RFDETRNano, RFDETRSegNano, RFDETRKeypointPreview, and RFDETRBase's
   backbone (including bicubic+antialias position-embedding interpolation,
   see `docs/decisions/0002-position-embed-bicubic.md`); Small/Large/XL/2XL
   still to do, and RFDETRBase's projector+decoder aren't wired up yet.
2. ~~Detection head: Deformable-DETR-style decoder~~ — validated end-to-end
   (backbone → projector → decoder → boxes/logits) for RFDETRNano.
3. ~~Instance segmentation head~~ — validated end-to-end for RFDETRSegNano
   (dot-product mask head over all decoder-layer query features).
4. ~~Keypoint head~~ — validated end-to-end for RFDETRKeypointPreview (dual
   projector, AdaLN-modulated GroupPose keypoint decoder stream).
5. Finetuning/training (C++/ggml training loop) — phase 2, after 1-4 are
   validated for inference

Each milestone is validated against a PyTorch reference
(`gen_reference/*.py` via `uv run` CPU torch → `tests/test_*.cpp` max-abs-diff
gate), following the sibling ports' pattern:

```sh
cmake -B build -G Ninja && cmake --build build --target test_backbone
uv run --with torch --with numpy --with gguf scripts/convert_dinov2_to_gguf.py models/rf-detr-nano.pth models/rf-detr-nano-backbone.gguf
uv run --with torch --with numpy --with rfdetr gen_reference/gen_reference_backbone.py models/rf-detr-nano.pth gen_reference/reference_backbone_nano.bin
./build/test_backbone.exe
```

Checkpoints download from `https://storage.googleapis.com/rfdetr/*` (see
`docs/decisions/backbone-windowing.md`); `models/` is gitignored.

## Weights

Converted GGUF weights will be published on this repo's GitHub Releases
(zstd-split parts for files >2GB), mirroring see-through-cpp's release
layout. Upstream PyTorch checkpoints are plain `.pth` state dicts hosted at
`https://storage.googleapis.com/rfdetr/*.pth` (not HuggingFace).
