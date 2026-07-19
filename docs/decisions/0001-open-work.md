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
- [x] `qkv_bias` (config default `True`) checkpoint-verified: scanned the
      raw GGUF tensor-name strings for `encoder.layer.0.attention.attention.
      {query,key,value}.bias` in both `rf-detr-nano-backbone.gguf` and
      `rf-detr-base-backbone.gguf` — present in both. `linear()` already
      loads and applies any `.bias` tensor it finds (`m.has(pre+".bias")`),
      so this was a verification-only task, no code change needed.
- [x] RFDETRSmall validated end-to-end (`test_decoder_small`: boxes 4.5e-4,
      logits 9.3e-4) — checkpoint-verified against
      `small_coco/checkpoint_best_regular.pth` (MD5 confirmed). Same
      Nano-family backbone/window config (`out_feature_indexes_raw=[3,6,9,12]`,
      `num_windows=2`, patch_size=16, no bicubic interpolation needed), just
      resolution 512 (grid 32) and `dec_layers=3` — pure config-value
      extension, no code changes, confirming the pattern generalizes.
- [x] RFDETRMedium validated end-to-end (`test_decoder_medium`: boxes
      1.8e-3, logits 2.2e-3) — checkpoint-verified against
      `medium_coco/checkpoint_best_regular.pth` (MD5 confirmed). Same
      pattern as Small: resolution 576 (grid 36), `dec_layers=4`, no code
      changes.
- [x] RFDETRLarge (current, non-deprecated) validated end-to-end
      (`test_decoder_large`: boxes 1.5e-3, logits 1.8e-3) —
      checkpoint-verified against `rf-detr-large-2026.pth` (MD5 confirmed).
      Same pattern as Small/Medium: resolution 704 (grid 44), `dec_layers=4`,
      no code changes.
- [x] There is no detection XL/2XL variant upstream — only Nano, Small,
      Medium, Base, Large, and LargeDeprecated exist for detection (XL/2XL
      only exist for segmentation). **Every actively-used detection size is
      now validated.** `RFDETRLargeDeprecated` (multi-feature-level
      `projector_scale=['P3','P5']`, 768-wide backbone) is a real config
      class upstream still ships, but it's explicitly superseded by the
      current `RFDETRLarge` (already validated) and no active code in this
      port or upstream depends on it — **deprioritized per explicit
      instruction, not pursuing further**. The multi-level building blocks
      built while investigating it are kept (they're general-purpose, not
      LargeDeprecated-specific) but validated only in isolation against
      synthetic weights, not against LargeDeprecated's real checkpoint:
      - `ms_deform_attn_multilevel` (`src/deform_attn.{h,cpp}`) — exact
        match in isolation (`test_deform_attn_multilevel.cpp`).
      - `projector_multiscale` (`src/projector.{h,cpp}`) — exact match in
        isolation (`test_projector_multiscale.cpp`).
      - `DecoderParams::levels` multi-level wiring (`src/decoder.{h,cpp}`)
        — zero regression on all 7 existing decoder tests, not yet
        end-to-end validated against a real multi-level checkpoint.
      - Sine-embedding dim parametrization (`hd/2` instead of hardcoded
        128) — a real latent bug fix, independently valuable, zero
        regression confirmed.
      - **A real, unresolved ggml graph-allocator bug was found and
        documented** while chasing this (see the SegXLarge entry below) —
        that finding stands on its own regardless of LargeDeprecated's
        priority, since it affects any sufficiently large decoder graph.
- [x] Multi-image batching: `dinov2_backbone`'s windowed-attention trick
      already spends ggml's only spare batch axis on per-window batching
      (token-major tensors are `(C,T,nw2)`, not `(C,T,1,N_img)`), so a true
      4th image-batch dimension isn't available without restructuring
      windowed attention itself. Resolved as: call `dinov2_backbone` (and
      `projector_p4`/`rfdetr_decoder`) once per image, sharing one `Model`
      — validated (`test_backbone_multi_image.cpp`) that two independent
      graph-build calls sharing a `Model`'s weights produce genuinely
      independent, non-aliased results even within a single ggml
      context/graph. No code changes needed; this is the standard
      static-graph-inference batching pattern.

### Instance segmentation (RFDETRSegNano) — done

- [x] Segmentation head validated end-to-end (`test_segmentation`: boxes
      1.1e-2, logits 2.9e-3, masks 0.109 against a relaxed 0.15 gate)
- [x] Mask-gate relaxation independently reviewed (second-opinion agent) —
      no bug found; the original quantitative justification was corrected
      per review feedback (`docs/decisions/segmentation.md`)
- [ ] The *exact* mechanism behind the mask head's larger residual diff
      (random channel-sign accumulation vs. boundary-pixel sensitivity vs.
      both) is not nailed down — only that it isn't a wiring/aliasing bug
- [x] RFDETRSegSmall validated end-to-end (`test_segmentation_small`: boxes
      4.5e-4, logits 4.9e-4, masks 4.9e-2 against the 0.15 gate —
      noticeably tighter than SegNano's 0.109, consistent with the
      "amplified decoder float-drift" explanation in
      `docs/decisions/segmentation.md`, not a variant-specific issue) —
      checkpoint-verified against `rf-detr-seg-s-ft.pth` (MD5 confirmed).
      Same encoder family as SegNano, `num_windows=2` (vs SegNano's 1),
      resolution 384 (grid 32) — pure config-value extension, no code
      changes.
- [x] RFDETRSegMedium validated end-to-end (`test_segmentation_medium`:
      boxes 6.2e-3, logits 4.1e-3, masks 8.4e-2 against the 0.15 gate) —
      checkpoint-verified against `rf-detr-seg-m-ft.pth` (MD5 confirmed).
      Caught a real bug while wiring this one up: `SegmentationParams::
      num_blocks` must equal `dec_layers` (one `DepthwiseConvBlock` per
      decoder layer, per `segmentation.h`'s own doc comment) — copying
      SegSmall's test file verbatim with `num_blocks=4` left over (SegMedium
      is `dec_layers=5`) produced a real, large mask divergence
      (max-abs-diff 20, nothing like the small "amplified float drift"
      residual every other variant shows) until fixed.
- [x] RFDETRSegLarge validated end-to-end (`test_segmentation_large`:
      boxes 2.4e-3, logits 7.1e-3, masks 7.8e-2 against the 0.15 gate) —
      checkpoint-verified against `rf-detr-seg-l-ft.pth` (MD5 confirmed).
      Caught a real config-vs-checkpoint drift: `RFDETRSegLargeConfig`'s
      own `num_queries` default is 200, but the actual published
      checkpoint's `refpoint_embed.weight`/`query_feat.weight` are shaped
      for 300 queries (`3900 = 300*13 group_detr` rows) — the default
      produces a `load_state_dict` size-mismatch error; fixed by passing
      `num_queries=300` explicitly (both in `gen_reference_segmentation.py`
      and the C++ `DecoderParams`), not trusting the config class's own
      default. Same encoder/window pattern as SegMedium, resolution 504
      (grid 42), `dec_layers=5`.
- [ ] **RFDETRSegXLarge fails with a large, real divergence** (`test_segmentation_xlarge`:
      boxes max-abs-diff 0.996, logits 3.4, masks ~59 — nothing like the
      small amplified-float-drift residual every other variant shows).
      Extensively bisected this session (checkpoint MD5-verified, all
      config fields checkpoint-verified, GGUF weight VALUES spot-checked
      byte-for-byte against the checkpoint for the projector and
      enc_output/enc_out_bbox_embed — all correct):
      - `dinov2_backbone`'s taps match the real encoder exactly (1.2e-4) in
        isolation.
      - `projector_p4`'s output matches the real `MultiScaleProjector`
        exactly (3e-6) **when computed in an isolated graph** (backbone+
        projector only, nothing else).
      - The SAME computation, embedded in the full backbone+projector+
        decoder graph (dec_layers=6, 2704 tokens — the largest decoder
        graph in this project), produces a **corrupted** `memory` tensor
        (the reshaped/permuted/cont'd projector output fed to the
        decoder) — max-abs-diff ~4 against the same reference.
      - Explicitly calling `ggml_set_output()` on `memory` (and
        `output_memory`) before building the graph **fixes `memory`'s own
        corruption** (confirmed exact match again), but a further
        downstream tensor (`enc_delta`, the two-stage box-decode MLP
        output) is **still corrupted** even with both of those protected.
      - `tgt` (the learned content query, independent of `memory`) matches
        exactly throughout, confirming the corruption is specific to the
        `memory`-derived computation chain, not a general graph problem.
      This strongly points to a **ggml graph-allocator buffer-reuse bug**
      triggered specifically by this graph's size/topology (not a port
      logic bug — every architecture piece it depends on, checkpoint-
      verified independently, is correct) — but fully resolving it needs
      either finding every affected intermediate tensor (`ggml_set_output`
      whack-a-mole) or a deeper look at `ggml-alloc.c`'s buffer-lifetime
      analysis for graphs this large. Not resolved this session;
      `test_segmentation_xlarge.cpp`/GGUF conversion scaffolding is
      committed and ready to re-validate once fixed. Seg2XLarge (even
      larger, 768 res) is likely to hit the same issue and should wait for
      this to be root-caused first.
- [x] RFDETRSegPreview validated end-to-end (`test_segmentation_preview`:
      boxes 4.7e-4, logits 1.6e-3, masks 5.6e-2 against the 0.15 gate) —
      checkpoint-verified against `rf-detr-seg-preview.pt` (MD5 confirmed).
      Same resolution as SegMedium (432, grid 36) but `dec_layers=4` (not
      5), `num_queries=200` — pure config-value extension, no code
      changes.

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
- [x] Decided the finetuning scope: freeze the DINOv2 backbone AND the
      transformer decoder, finetune only `class_embed`/`bbox_embed` (the
      smallest useful slice) — see `0003-training.md`. This sidesteps
      Finding 3 (deformable-attention backward, not a small task) entirely
      and doesn't even need `layer_norm_affine_diff` for THIS scope (the
      frozen decoder stack never has a gradient requested through it) —
      that work stays validated and ready for whenever the scope widens.
- [x] Loss + Hungarian matching implemented and validated
      (`src/loss.{h,cpp}`, `tests/test_loss.cpp`): sigmoid focal loss + L1
      + GIoU (upstream's `ia_bce_loss=False` path, not the actual default
      `ia_bce_loss=True` — needs a stop-gradient mechanism ggml doesn't
      have natively, documented as a follow-up in `0003-training.md`) +
      a standard host-side Kuhn-Munkres Hungarian matcher reproducing
      upstream's exact cost formula. Matched pairs match scipy's real
      `HungarianMatcher` exactly; loss VALUE matches the real
      `SetCriterion` to 2e-6; a finite-difference check confirms the
      gradient is correct. Caught two real bugs: (1) a wrong `.mean(1)`
      axis assumption that made the loss ~2.2x too large before checking
      the actual formula, (2) `ggml_sigmoid` has no backward case (same
      gap class as `ggml_norm`), fixed via the same primitive-
      decomposition trick as `layer_norm_affine_diff`. Also surfaced a
      SECOND real ggml training-infra gap (repeated graph execution
      silently corrupting results — see `0003-training.md`), worked
      around by rebuilding a fresh graph per evaluation in the test.
- [x] End-to-end training-step demo (`demos/train_step_demo.cpp`): loads
      the real RFDETRNano checkpoint, runs frozen backbone→decoder
      forward, `hungarian_match`, `detection_loss`, backward, and a
      hand-rolled AdamW update (same formula as `ggml_opt_step_adamw`'s
      CPU kernel) on `class_embed`/`bbox_embed` only, training against a
      REAL COCO image + real annotations
      (`gen_reference/gen_reference_real_image.py`). Loss decreases
      steadily and stays bounded over 30 steps (0.59 → ~0.10 at lr=0.01).
      Proves the full scope-decided plumbing composes end-to-end. Found
      and fixed four real bugs: `ggml_concat` has no backward case either
      (fixed via a `ggml_set`-based `bbox_reparam_decode_diff`,
      forward-identical, zero regression on all 7 existing inference
      tests); MSVC Debug builds pop a blocking assert dialog that hangs
      headless runs forever (fixed via `_CrtSetReportMode` at startup —
      likely explains an earlier session's "hung modal" report too); the
      demo never uploaded `rfdetr_decoder`'s required `output_proposals`
      input tensor, leaving it as allocator garbage that fed a huge
      `base_wh` into the (correctly-clamped-per-element) box decode,
      producing an aggregate loss around -1e26 — this, not synthetic
      input, was the real root cause of the previously-suspected
      "synthetic-noise numerical blowup"; and the demo's trainable
      `class_embed`/`bbox_embed` tensors weren't protected from
      graph-allocator buffer reuse (missing `ggml_set_output`), so a
      later node's output silently overwrote a param tensor's buffer
      before host readback — same bug class as the `g.loss`-buffer-reuse
      fix already documented below, just not yet applied to param
      tensors. See `0003-training.md` for the full writeup.
- [ ] Dataset/dataloader beyond this one hardcoded real image (COCO-format
      annotations → ggml tensors for an arbitrary image, batching,
      iteration over a full split) — not yet researched.
- [ ] `ggml_opt_step_adamw`/`ggml_opt_epoch` wiring (the demo hand-rolls
      identical AdamW math as a host-side loop instead, to sidestep the
      repeated-graph-execution gap above) once the dataloader lands.
- [ ] (Deferred, only if scope widens later) Resolve deformable-attention's
      backward (Finding 3) if the decoder becomes trainable; wire
      `layer_norm_affine_diff` into the decoder's LayerNorms at that point.

### Documentation / process

- [ ] Weights not yet published to a GitHub Release on
      `github.com/weftspun/rf-detr-cpp` (per the original instruction to
      mirror see-through-cpp's release layout, zstd-split for >2GB files)
      — `models/*.gguf`/`*.pth` currently exist only locally, gitignored
