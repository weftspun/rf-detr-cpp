# Keypoint head port decisions (RFDETRKeypointPreview) — research draft

Status: **architecture research only, no implementation yet**. This is a
genuinely new architecture component (a whole second "GroupPose" decoder
stream interleaved with the detection decoder), unlike segmentation which
reused the existing decoder/backbone verbatim with different config values.
Read this file in full before starting `src/keypoints.cpp` — the upstream
code (`rfdetr/models/transformer.py`'s `if self.enable_keypoint_processing:`
branches, explicitly skipped during the detection-decoder research) is
intricate; get the wiring right on paper first.

## Config: `RFDETRKeypointPreviewConfig` (`rfdetr/config.py`)

Good news: **same backbone as other Nano-family configs** —
`encoder="dinov2_windowed_small"`, `patch_size=12`, `num_windows=2`,
`resolution=576` (so `positional_encoding_size=576/12=48`,
`gw=gh=48`), `dec_layers=4`, `num_queries=100`. No new backbone size, no
position-embedding bicubic interpolation needed (same reasoning as
Nano/SegNano: `patch_size != 14` → `load_dinov2_weights=False` → checkpoint's
position embeddings are already sized for the training resolution).
**Not yet checkpoint-verified** — the checkpoint filename
(`rf-detr-keypoint-preview-xlarge.pth`) is suspicious (says "xlarge" despite
the config class not overriding encoder size); confirm actual backbone
dims against the downloaded checkpoint's state-dict keys before trusting
this, the way every prior milestone's assumptions were checkpoint-verified
(e.g. the backbone-windowing off-by-one, the decoder's non-zero
`refpoint_embed` — assumptions from reading source alone have been wrong
before in this project).

What's new vs. detection/segmentation: `use_grouppose_keypoints=True`,
`dual_projector=True`, `dual_projector_kp_only=True`,
`num_keypoints_per_class=[17]` (single class, COCO-17 layout),
`keypoint_cross_attn=True`, `inter_instance_kp_attn` not set → **defaults
False** (a whole optional sub-branch, "cross-instance keypoint attention",
can be skipped entirely for this checkpoint — see below),
`grouppose_keypoint_dim_downscale=1` (default — this simplifies several
`nn.Linear`s in `TransformerDecoderLayer` to `nn.Identity()`, see below).

## Dual projector

`Backbone.__init__` builds a **second** `MultiScaleProjector` instance
(`self.cross_attn_projector`) with the same architecture as the main one
(same C2f block, `docs/decisions/decoder.md`'s projector notes apply
verbatim) but independently-learned weights, when `dual_projector=True`.
`Backbone.forward` runs `raw_feats` (the DINOv2 backbone taps) through
**both** projectors independently — they do NOT share the C2f computation,
each has its own full weight set. Checkpoint key prefix:
`backbone.0.cross_attn_projector.*` (parallel to `backbone.0.projector.*`).

Since `dual_projector_kp_only=True`, `LWDETR.forward` routes them
differently (confirmed from source, `models/lwdetr.py` around the
`self.dual_projector_kp_only` check seen during earlier decoder research):
`decoder_memory = memory` (**main** projector output — detection/
segmentation's cross-attention keeps using the ordinary projector, kp-only
mode does NOT redirect the main stream), `kp_cross_attn_memory =
cross_attn_srcs` (**second** projector output, used *only* by the
keypoint-specific deformable cross-attention inside
`TransformerDecoderLayer`, see below).

## Keypoint query initialization (`ConditionalQueryInitializer`, `rfdetr/models/heads/keypoints.py`)

AdaLN-modulated learned queries, conditioned on a per-instance feature
vector:
```
normed = LayerNorm(no affine)(self.queries)                    # (total_kp, out_dim), queries is a learned Parameter
scale, shift, gate = split3(MLP(Linear→GELU→Linear(zero-init))(cond))  # cond: (..., dim) -> (..., out_dim*3)
modulated = out_proj((scale+1)*normed + shift) * gate + self.queries   # (..., total_kp, out_dim)
```
`adaLN_modulation`'s final `Linear`'s weight+bias are **zero-initialized**
at construction (so untrained modulation is a no-op) — irrelevant for
inference (loaded from checkpoint) but confirms the intended residual-like
behavior: `modulated ≈ self.queries` when scale/shift/gate are near their
zero-init values, drifting via training.

Two independent instances (`grouppose_keypoint_dim_downscale=1` → both
`out_dim = d_model = 256`, no separate `kp_dim`):
- `keypoint_query_initializer` (decoder-level): conditioned on `tgt`
  (the learned static content-query embedding, same `tgt` fed to the main
  decoder) → produces `tgt_keypoints`, the **initial keypoint tokens** fed
  into `TransformerDecoder.forward`'s per-layer keypoint self-attention.
- `keypoint_query_initializer_enc` (two-stage/encoder-level): conditioned
  on `memory_ts` (the two-stage top-k-gathered encoder memory rows, same
  gather used for `hs_enc`/`refpoint_embed_ts` in the main two-stage path)
  → produces `keypoint_memory_ts`, fed through `enc_out_keypoint_embed[0]`
  (an `MLP(kp_dim, d_model, kp_dim, 2)`, 3-layer per the `MLP` class's
  `(in,hidden,out,n_layer=2)` signature meaning 2 *hidden* layers — verify
  exact layer count against checkpoint keys) to produce `kp_delta`, decoded
  via the same bbox-reparam-style formula as everywhere else:
  `kp_xy = kp_delta[...,:2]*ref_wh + ref_xy` (using the two-stage detection
  box's `ref_wh`/`ref_xy`, broadcast per keypoint) → `enc_kp_predictions`
  → `init_kp_ref_xy = enc_kp_predictions[...,:2].detach()`, the **initial
  keypoint xy reference points** used by keypoint deformable cross-attention
  in layer 0 (reference points don't refine per-layer — same
  `lite_refpoint_refine` reasoning as the main decoder;
  `TransformerDecoder.forward` asserts
  `enable_keypoint_processing requires lite_refpoint_refine`).

**Important**: `init_kp_ref_xy` is asserted required by
`TransformerDecoder.forward` but the actual keypoint cross-attention code
(`TransformerDecoderLayer.forward_post`'s keypoint branch, read below) uses
`bbox_ref_for_kp` = the **parent detection query's own box reference**
(`reference_points`, expanded per-keypoint), *not* `init_kp_ref_xy`
directly, for the deformable sampling. Re-check where `init_kp_ref_xy`
actually gets consumed (likely only for the *loss*/matching machinery in
training, or for `_format_keypoint_output`'s xy — not yet traced to its
consumer in the inference forward path; **resolve this before
implementing**, since it changes whether `keypoint_query_initializer_enc`
is needed for inference at all).

## Per-layer keypoint processing (`TransformerDecoderLayer.forward_post`, keypoint branch)

Runs *after* the ordinary detection self-attn/cross-attn/FFN sublayers
(same as `src/decoder.cpp`'s existing 3 sublayers — unchanged), as a 4th
stage, only when `enable_keypoint_processing`:

With `grouppose_keypoint_dim_downscale=1`: `inst_in_proj`,
`inst_pos_in_proj`, `inst_out_proj`, `memory_in_proj` are all
`nn.Identity()` (the `if grouppose_keypoint_dim_downscale > 1 else
nn.Identity()` ternary) — **no extra projection weights to load/apply for
this specific config**, `kp_dim == d_model == 256` throughout. Don't be
misled by the general code supporting a downscaled keypoint dimension;
this checkpoint doesn't use it.

1. **Keypoint-instance self-attention**: for each detection query
   (`num_queries=100`), concatenate `[tgt_query_token, keypoint_tgt[0..16]]`
   → 18 tokens, run ordinary `nn.MultiheadAttention` (`sa_nhead` heads,
   same fused `in_proj_weight` pattern as the main self-attn — needs the
   same q/k/v split at conversion time) **within that group of 18** (batch
   dimension becomes `B*num_queries`, i.e. every instance's
   query+keypoints attend only among themselves — structurally similar to
   the backbone's per-window batching trick), masked by
   `keypoint_class_mask` (a `(1+total_kp, 1+total_kp)` boolean mask
   blocking cross-class keypoint interactions — **trivially all-zero
   (no masking) for `num_keypoints_per_class=[17]`**, a single class, so
   this mask can be skipped/ignored for this checkpoint: `_create_keypoint_class_mask`
   only sets `True` entries between *different* classes' keypoint ranges,
   and there's only one class here). Output splits back into an updated
   `tgt` (index 0) and updated `keypoint_tgt` (indices 1-17), each with
   its own residual+LayerNorm (`instance_kp_layer_scale`, a *learned
   scalar* initialized to `1e-6`, gates the `tgt` update specifically —
   `tgt = tgt + dropout(inst_out_proj(tgt2)) * instance_kp_layer_scale`).
2. **Inter-instance keypoint attention**: **skipped entirely** —
   `inter_instance_kp_attn=False` for this config (not set, defaults
   False), so `self.inter_instance_kp_attn` is `False` and this whole
   block (`swapped_keypoint_tgt`/`inter_inst_kp_attn`) never runs. No
   weights for it exist in this checkpoint either (verify).
3. **Keypoint-specific deformable cross-attention**: each keypoint token
   (`query = keypoint_tgt + kp_query_pos`, flattened to
   `(B, num_queries*17, kp_dim)`) cross-attends via a **separate**
   `MSDeformAttn` instance (`kp_cross_attn`, its own learned
   `sampling_offsets`/`attention_weights`/`value_proj`/`output_proj` —
   reuse `src/deform_attn.cpp`'s `ms_deform_attn()` verbatim, just a
   different weight prefix and different `query`/`value_input`/
   `ref_points`), using `bbox_ref_for_kp` = the **parent query's own
   `reference_points`** (the same fixed detection-decoder reference box,
   expanded/repeated across all 17 keypoints of that instance — *not* a
   per-keypoint xy reference), sampling from `kp_cross_attn_memory` (the
   **second/dual** projector's output, `memory_in_proj`'d — identity here)
   rather than the main `memory`. Residual + LayerNorm (`kp_cross_attn_norm`).
4. **Keypoint-specific FFN**: `kp_linear1(kp_dim→4*d_model)` → activation
   (ReLU, same `activation` as the main FFN) → `kp_linear3(4*d_model→kp_dim)`
   → residual + `kp_norm5`. Note the *expansion width* is `d_model*4` (1024)
   even though `kp_dim` may be `<d_model` in the general (downscaled) case
   — for this config `kp_dim=d_model=256` so it's the ordinary 256→1024→256
   FFN, same shape as the main decoder's FFN.

Returns `(tgt, keypoint_tgt)` — both feed into the next layer (unlike the
main `hs`/pred_boxes/pred_logits, which only need the *last* layer,
`TransformerDecoder.forward` collects **all** layers' `keypoint_tgt` into
`intermediate_keypoints` — check whether the keypoint head, like
segmentation, needs all layers or just the last one before assuming
"last only" the way detection works).

## Output formatting (`LWDETR`, `rfdetr/models/lwdetr.py`)

- `keypoint_hs` (all-layers stack from `TransformerDecoder`) →
  `self.keypoint_embed(keypoint_hs)` (not yet read — locate this module,
  likely another small MLP mapping `kp_dim → KEYPOINT_PRED_DIM=8`) →
  `outputs_keypoints_delta` → decode via the box-reparam-style formula
  against the **parent query's** `ref_wh`/`ref_xy` (not a separate keypoint
  reference) → `outputs_keypoints_compact` (`x,y` reparam-decoded,
  remaining 6 channels — `findable`, `visible`, 3 Cholesky params, 1
  class-logit — passed through unchanged from the delta, per the module
  docstring's `KEYPOINT_PRED_DIM=8` slot layout in
  `rfdetr/models/heads/keypoints.py`).
- `_format_keypoint_output`: converts "compact" (`total_actual_keypoints`)
  to "class-padded" (`num_classes * max_keypoints_per_class`) layout —
  **for `num_keypoints_per_class=[17]` (single class) these are equal**
  (`17 == 1*17`), so this is a **no-op pass-through** for this checkpoint.
  Don't implement the general multi-class padding logic yet; not needed.
- `_aggregate_keypoint_class_logits`: sums each keypoint's slot-7
  "class-logit contribution" into the main detection `class_embed` output
  (`outputs_class = outputs_class + self._aggregate_keypoint_class_logits(...)`)
  — **not yet read in detail**, needed before implementing final logits.
- Final `out["pred_keypoints"] = outputs_keypoints[-1]` — last layer only
  (mirrors detection's `[-1]`, unlike segmentation's "all layers").

## Implementation plan (next iterations)

1. Resolve the `init_kp_ref_xy` consumer question above (read
   `keypoint_embed`, confirm whether `keypoint_query_initializer_enc`/
   `enc_out_keypoint_embed` are needed for inference or training-loss-only).
2. Read `self.keypoint_embed`'s definition and `_aggregate_keypoint_class_logits`
   in full.
3. Download `rf-detr-keypoint-preview-xlarge.pth`, verify backbone/decoder
   dims against real checkpoint keys (don't trust the "xlarge" filename).
4. Extend `src/decoder.cpp` (or a new `src/keypoints.cpp` calling into it)
   with: dual projector (reuse `src/projector.cpp`'s `projector_p4` with a
   different weight prefix — it's already parameterized by prefix
   implicitly via `Model`'s global weight map, just needs the
   `cross_attn_projector` prefix wired through), `ConditionalQueryInitializer`
   (AdaLN modulation — new small op sequence, straightforward), the 4th
   per-layer keypoint sublayer (self-attn reusing the per-window-batching
   trick from `backbone.cpp`, deformable cross-attn reusing
   `ms_deform_attn()` verbatim), and final keypoint head decode.
5. GGUF conversion script, `gen_reference_keypoints.py` using the real
   upstream module (same `uv run --with rfdetr` pattern), C++ test with a
   max-abs-diff gate — same methodology as every prior milestone.

## Still unverified

Everything above is read-only source analysis this session; **no line of
`src/keypoints.cpp` has been written yet**, and none of the checkpoint-key
assumptions have been checkpoint-verified (unlike backbone/decoder/
segmentation, where every prefix was double-checked against a real
downloaded `.pth`'s `data.pkl` string scan before writing conversion code).
Do that verification pass first in the next iteration, the same way it was
done for every prior milestone — this project's own history
(`docs/decisions/backbone-windowing.md`'s off-by-one,
`docs/decisions/decoder.md`'s non-zero `refpoint_embed`) shows source
reading alone gets non-obvious details wrong often enough that checkpoint
verification is not optional.
