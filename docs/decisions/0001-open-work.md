# 1. Open work

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

`docs/decisions/*.md` each record one milestone's architecture findings
(backbone windowing, the detection decoder, segmentation) plus their own
"still unverified"/open-question notes. Those notes were scattered across
four separate files with no single place to see what's actually
outstanding across the whole port. This record is that single place: an
actionable checklist, consolidated from every decision doc's open items.
The architecture rationale stays in each doc; only the open-task tracking
moves here. New open items get added here as they're found; items get
checked off (not deleted) as they land, so the record stays a history of
what was open and when it closed.

## Checklist

### Object detection (RFDETRNano) — done

- [x] Backbone validated (`test_backbone`, max-abs-diff ≤2.5e-4)
- [x] Projector validated (`test_projector`, 1.0e-5)
- [x] Deformable attention core validated in isolation
      (`test_deform_attn`, exact — see `docs/decisions/decoder.md`)
- [x] Detection decoder validated end-to-end (`test_decoder`: boxes 3.3e-4,
      logits 7.3e-4)
- [x] Position-embedding bicubic+antialias interpolation implemented and
      validated (`test_backbone_base`: max-abs-diff ≤1.6e-4) — needed for
      `patch_size==14` variants at a non-native resolution (RFDETRBase);
      see `docs/decisions/0002-position-embed-bicubic.md`. ggml has no
      bicubic+antialias built-in; solved via a precomputed resize matrix
      (probed directly from PyTorch's real implementation) applied through
      two `ggml_mul_mat` calls — no custom op needed, and it's inherently
      GPU-backend-ready (runs on whichever ggml backend is active).
- [x] RFDETRBase backbone validated end-to-end (`test_backbone_base`)
- [x] RFDETRBase full detection pipeline validated (`test_decoder_base`:
      boxes 1.1e-3, logits 7.2e-4) — 3 decoder layers (vs. Nano-family's
      2/4), confirms `src/decoder.cpp` generalizes cleanly to a third
      `dec_layers` value with no code changes
- [ ] `qkv_bias` (config default `True`) never explicitly checked for
      presence against a real checkpoint key — low risk, `linear()`
      handles bias-optional generically either way, but not confirmed
- [ ] Other backbone/decoder size variants (Small/Large/XL/2XL, and
      Large-deprecated which shares RFDETRBase's patch_size==14 path)
      unvalidated — only Nano-family configs + RFDETRBase's backbone
      confirmed so far
- [ ] Multi-image batching (N>1) not supported — `backbone.cpp`'s
      windowed/global merge-reshape trick assumes a single image per graph

### Instance segmentation (RFDETRSegNano) — done

- [x] Segmentation head validated end-to-end (`test_segmentation`: boxes
      1.1e-2, logits 2.9e-3, masks 0.109 against a relaxed 0.15 gate)
- [x] Mask-gate relaxation independently reviewed (second-opinion agent) —
      no bug found; the original quantitative justification was corrected
      per review feedback (`docs/decisions/segmentation.md`)
- [ ] The *exact* mechanism behind the mask head's larger residual diff
      (random channel-sign accumulation vs. boundary-pixel sensitivity vs.
      both) is not nailed down — only that it isn't a wiring/aliasing bug
- [ ] Other segmentation variants (Small/Medium/Large/XL/2XL) unvalidated
- [ ] `RFDETRSegPreviewConfig` and other non-Nano segmentation configs
      unvalidated

### Keypoint detection (RFDETRKeypointPreview) — done

- [x] Resolve where `init_kp_ref_xy` is actually consumed in the inference
      forward path — resolved: it's a presence-guard only, never read
      again; `keypoint_query_initializer_enc`/`enc_out_keypoint_embed` are
      dead code for inference, safely skippable
- [x] Read `self.keypoint_embed`'s definition (`lwdetr.py`) and
      `_aggregate_keypoint_class_logits` in full — both fully traced, see
      `docs/decisions/keypoints.md`'s "Output formatting" section
- [x] Download `rf-detr-keypoint-preview-xlarge.pth` and verify actual
      backbone/decoder dims against the checkpoint's own state-dict keys —
      confirmed same Small/Nano-family backbone (12 layers, patch=12, no
      registers) and 4 decoder layers as the config class implies; the
      "xlarge" filename is misleading, not a real size mismatch. Also
      found a dead legacy `keypoint_head.keypoint_proj.*` weight set
      (confirmed unused via `weights.py`'s own comment) that no amount of
      source reading would have surfaced.
- [x] Implement the dual projector (reuse `projector_p4` with the
      `cross_attn_projector` weight prefix)
- [x] Implement `ConditionalQueryInitializer` (AdaLN-modulated keypoint
      query initialization)
- [x] Implement the per-layer keypoint sublayer: keypoint-instance
      self-attention, keypoint-specific deformable cross-attention (reuse
      `ms_deform_attn()` verbatim), keypoint-specific FFN
- [x] Implement final keypoint head decode + `_format_keypoint_output` +
      `_aggregate_keypoint_class_logits` — caught a real bug here: the
      checkpoint's actual `num_keypoints_per_class` schema is `[0,17]`
      (active class index **1**), not the initially-guessed `[17,0]`
      (index 0) — a shape-only check (`_kp_active_mask`'s `(2,17)` shape)
      didn't distinguish the two; only inspecting the mask's boolean
      *values* did. See `docs/decisions/keypoints.md`.
- [x] GGUF conversion script (`scripts/convert_keypoints_to_gguf.py`) +
      `gen_reference/gen_reference_keypoints.py` + `tests/test_keypoints.cpp`
      — validated end-to-end: boxes 3.5e-3, logits 9.3e-4, keypoints 4.2e-3
      (gate 5e-2 all three)

Full architecture + implementation notes: `docs/decisions/keypoints.md`.

**All three inference milestones (object detection, instance segmentation,
keypoint detection) are now done.** Remaining scope: other model-size
variants for each (see their sections above), then phase-2 training.

### Phase 2: finetuning/training — in progress

- [x] Design research doc (`docs/decisions/0003-training.md`): surveyed
      ggml's real autodiff/optimizer API, and audited every op this port
      uses against `ggml_compute_backward`'s switch for backward-pass
      support
- [x] Fixed the biggest blocker found: `ggml_norm` (plain LayerNorm) has no
      backward case at all, and it's used in every block (backbone,
      decoder, segmentation, keypoints). Added `layer_norm_affine_diff`
      (`src/ops.{h,cpp}`) — the same LayerNorm rebuilt from primitives that
      do have backward support, validated forward-exact-match plus a
      finite-difference gradient check (`tests/test_norm_backward.cpp`).
      Two non-obvious ggml backward-shape gotchas surfaced and were routed
      around: `ggml_sub`/`ggml_div`'s backward doesn't broadcast a smaller
      `src1`'s gradient back up (unlike `ggml_add`/`ggml_mul`, which do via
      `ggml_repeat_back`), and `ggml_mean`'s own backward assumes a true
      scalar output (asserts on a per-row `(1,T,N)` mean) — worked around
      with `ggml_sum_rows`, whose backward is shape-general. Existing
      `layer_norm_affine` (fused `ggml_norm`) is untouched, still used by
      all 8 already-validated inference tests.
- [ ] Decide the actual finetuning scope (which layers trainable — see
      `0003-training.md`'s "Recommended scope")
- [ ] Wire `layer_norm_affine_diff` into the graphs the chosen scope needs
      trainable
- [ ] Resolve deformable-attention's backward (Finding 3 in
      `0003-training.md`) if the decoder is in-scope
- [ ] Hungarian matching + loss assembly (detection first)
- [ ] Dataset/dataloader (COCO-format annotations → ggml tensors)

### Documentation / process

- [ ] Weights not yet published to a GitHub Release on
      `github.com/weftspun/rf-detr-cpp` (per the original instruction to
      mirror see-through-cpp's release layout, zstd-split for >2GB files)
      — `models/*.gguf`/`*.pth` currently exist only locally, gitignored
