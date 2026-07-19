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
- [ ] `qkv_bias` (config default `True`) never explicitly checked for
      presence against a real checkpoint key — low risk, `linear()`
      handles bias-optional generically either way, but not confirmed
- [ ] Other backbone/decoder size variants (Small/Base/Large/XL/2XL)
      unvalidated — only Nano-family configs (Nano, SegNano) confirmed
- [ ] Position-embedding bicubic interpolation not implemented — needed
      for `patch_size==14` variants at a non-native resolution
      (RFDETRBase/Large-deprecated); Nano-family variants (`patch_size!=14`)
      don't need it (see `docs/decisions/backbone-windowing.md`)
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

### Keypoint detection (RFDETRKeypointPreview) — not started

- [ ] Resolve where `init_kp_ref_xy` is actually consumed in the inference
      forward path (`transformer.py`) — affects whether
      `keypoint_query_initializer_enc`/`enc_out_keypoint_embed` are needed
      for inference at all, or training-loss-only
- [ ] Read `self.keypoint_embed`'s definition (`lwdetr.py`) and
      `_aggregate_keypoint_class_logits` in full
- [ ] Download `rf-detr-keypoint-preview-xlarge.pth` and verify actual
      backbone/decoder dims against the checkpoint's own state-dict keys —
      the filename says "xlarge" but `RFDETRKeypointPreviewConfig` doesn't
      override encoder size; unresolved discrepancy, don't trust the
      config-class assumption until checked
- [ ] Implement the dual projector (reuse `projector_p4` with the
      `cross_attn_projector` weight prefix)
- [ ] Implement `ConditionalQueryInitializer` (AdaLN-modulated keypoint
      query initialization)
- [ ] Implement the per-layer keypoint sublayer: keypoint-instance
      self-attention (masked, but the mask is a no-op for
      `num_keypoints_per_class=[17]`'s single class), keypoint-specific
      deformable cross-attention (reuse `ms_deform_attn()` verbatim against
      the dual projector's second memory), keypoint-specific FFN
- [ ] Implement final keypoint head decode + `_format_keypoint_output`
      (a no-op pass-through for the single-class case) +
      `_aggregate_keypoint_class_logits`
- [ ] GGUF conversion script (`scripts/convert_keypoints_to_gguf.py`) +
      `gen_reference/gen_reference_keypoints.py` (real upstream module via
      `uv run --with rfdetr`) + a C++ test with a max-abs-diff gate

Full architecture research: `docs/decisions/keypoints.md`.

### Phase 2: finetuning/training — not started

- [ ] Design and implement a C++/ggml training loop (backprop, optimizer,
      dataloader) — deliberately deferred until all three inference
      milestones (detection, segmentation, keypoints) pass, per the
      project's original scope (`docs/architecture.md`)

### Documentation / process

- [ ] Weights not yet published to a GitHub Release on
      `github.com/weftspun/rf-detr-cpp` (per the original instruction to
      mirror see-through-cpp's release layout, zstd-split for >2GB files)
      — `models/*.gguf`/`*.pth` currently exist only locally, gitignored
