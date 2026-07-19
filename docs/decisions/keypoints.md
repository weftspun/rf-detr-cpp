# Keypoint head port decisions (RFDETRKeypointPreview) — research draft

Status: **architecture fully researched and checkpoint-verified, no C++
implementation yet**. This is a genuinely new architecture component (a
whole second "GroupPose" decoder stream interleaved with the detection
decoder), unlike segmentation which reused the existing decoder/backbone
verbatim with different config values. Every open question this doc
originally flagged is now resolved (see "Resolved" markers throughout) —
implementation can start directly from this spec. Read this file in full
before starting `src/keypoints.cpp` — the upstream code
(`rfdetr/models/transformer.py`'s `if self.enable_keypoint_processing:`
branches, explicitly skipped during the detection-decoder research) is
intricate; get the wiring right on paper first.

## Config: `RFDETRKeypointPreviewConfig` (`rfdetr/config.py`)

**Same backbone as other Nano-family configs, checkpoint-confirmed** —
`encoder="dinov2_windowed_small"`, `patch_size=12`, `num_windows=2`,
`resolution=576` (so `positional_encoding_size=576/12=48`,
`gw=gh=48`), `dec_layers=4`, `num_queries=100`. No new backbone size, no
position-embedding bicubic interpolation needed (same reasoning as
Nano/SegNano: `patch_size != 14` → `load_dinov2_weights=False` → checkpoint's
position embeddings are already sized for the training resolution).
**Checkpoint-verified**: downloaded `rf-detr-keypoint-preview-xlarge.pth`
(163MB) and confirmed via the `data.pkl` string-scan technique — 12
backbone layers, 4 decoder layers, no register tokens, patch conv weight
`(384,3,12,12)` (hidden=384/Small, patch=12), position embeddings
`(1,2305,384)` (2305-1=2304=48×48, matches `resolution=576/patch=12`),
`query_feat.weight`/`refpoint_embed.weight` shape `(1300,...)`
(`num_queries*group_detr = 100*13`). **The "xlarge" in the filename is
misleading** — this is the same Small/Nano-family backbone as every other
config validated so far, not a larger encoder. `class_embed.weight` shape
`(2,256)` confirms this checkpoint is trained for a **single foreground
class** (person, +1 background = 2), consistent with
`num_keypoints_per_class=[17]` (COCO-17 person keypoints).

**Also found by checkpoint verification, not present in any source
reading**: a `keypoint_head.keypoint_proj.{0,1,2}.*` weight set exists in
this checkpoint but is **dead** — `rfdetr/models/weights.py`'s own comment
confirms it: *"The preview keypoint checkpoint still stores the old
standalone MLP projection head, but the current GroupPose inference path
no longer consumes it."* Ignore these keys entirely; `keypoint_embed.*`
(not `keypoint_head.*`) is the real, currently-used weight set. This is
exactly the kind of thing source-reading alone misses and checkpoint
verification catches — consistent with every prior milestone's experience
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

**Resolved: `init_kp_ref_xy` is dead code for inference.** Traced fully —
`TransformerDecoder.forward` takes `init_kp_ref_xy` as a parameter and
raises `ValueError` if it's `None` (a presence guard only), but the
parameter is **never read again anywhere else in that function body**. The
per-layer keypoint cross-attention (`TransformerDecoderLayer.forward_post`)
uses `bbox_ref_for_kp` = the **parent detection query's own
`reference_points`** (expanded per-keypoint), not `init_kp_ref_xy`. And the
final keypoint xy decode (`LWDETR.forward`, see "Output formatting" below)
also uses the parent query's `ref_unsigmoid` (the same fixed reference
every other head decodes against), not `init_kp_ref_xy` either.
**Consequence: `keypoint_query_initializer_enc`, `enc_out_keypoint_embed`,
and the whole `init_kp_ref_xy`/`enc_kp_predictions` computation chain can
be skipped entirely in this port** — they exist upstream only to populate
`out["enc_outputs"]["pred_keypoints"]` (training-loss-only, like all
`enc_outputs`) and to satisfy that one guard, neither of which this
inference-only port needs to replicate. Only `keypoint_query_initializer`
(the **decoder-level** one, conditioned on `tgt`) is actually needed.

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

`self.keypoint_embed` (`LWDETR.__init__`): `MLP(hidden_dim, hidden_dim, 8,
3)` — same `MLP` class as `bbox_embed`, 3 `nn.Linear` layers
(`layers.0`: 256→256, `layers.1`: 256→256, `layers.2`: 256→8, ReLU between
the first two, none after the last — identical structure/weight-prefix
convention to `bbox_embed.layers.{0,1,2}`, reuse `mlp()` from `ops.h`
verbatim). Last layer zero-initialized at construction (irrelevant —
loaded from checkpoint).

Per-layer forward (only the **last** decoder layer's `keypoint_tgt` is
actually needed for the final output — `outputs_keypoints[-1]`, exactly
like detection's `pred_boxes`/`pred_logits` only needing `hs[-1]`; unlike
segmentation, which genuinely needs every layer):
```
outputs_keypoints_delta = keypoint_embed(keypoint_tgt)      # (..., num_kp, 8)
ref_wh = ref_unsigmoid[..., 2:].unsqueeze(-2)                # parent query's OWN box, same one everywhere else
ref_xy = ref_unsigmoid[..., :2].unsqueeze(-2)
keypoints_xy = outputs_keypoints_delta[..., :2] * ref_wh + ref_xy   # same reparam formula as bbox/mask heads
keypoints_other = outputs_keypoints_delta[..., 2:]           # findable, visible, 3 Cholesky params, class-logit -- passthrough
outputs_keypoints_compact = cat([keypoints_xy, keypoints_other], dim=-1)   # (..., num_kp, 8)
```
`_format_keypoint_output`: converts "compact" (`total_actual_keypoints`) to
"class-padded" (`num_classes * max_keypoints_per_class`) layout — **for
`num_keypoints_per_class=[17]` (single class) these are equal** (`17 ==
1*17`), so this is a **no-op pass-through** for this checkpoint. Don't
implement the general multi-class padding logic; not needed.

`_aggregate_keypoint_class_logits` (fully traced): takes slot-7
("class-logit contribution") from every keypoint,
`class_contrib.view(..., num_keypoint_classes, max_num_keypoints) *
self._kp_active_mask` (mask is all-`True`/trivial for one class with all
17 keypoints active — safe to skip the mask, just sum), `.sum(dim=-1)` →
one scalar boost per keypoint-class → **zero-padded** up to the detection
head's full `num_classes+1` (91) output width. For this checkpoint
(`num_keypoints_per_class=[17]`, one class): `pred_logits[..., 0] +=
sum_{k=0..16}(keypoint_predictions[..., k, 7])`, **all other 90 classes
(including background) unchanged**. Applied as
`outputs_class = outputs_class + self._aggregate_keypoint_class_logits(outputs_keypoints)`
— i.e. this is an *additive correction to `pred_logits`*, not a separate
output; `src/decoder.cpp`'s existing `class_embed` linear output needs this
term added in before being returned, once the keypoint head is wired in.

Final: `out["pred_keypoints"] = outputs_keypoints[-1]` (last layer only).
- Final `out["pred_keypoints"] = outputs_keypoints[-1]` — last layer only
  (mirrors detection's `[-1]`, unlike segmentation's "all layers").

## Implementation plan and open tasks

All research questions this doc originally raised are resolved (init_kp_ref_xy
is dead code, keypoint_embed's shape and consumers are fully traced, the
checkpoint's dims and the dead keypoint_head.keypoint_proj.* keys are
confirmed). Remaining work is implementation, not research — tracked as an
actionable checklist in [`0001-open-work.md`](0001-open-work.md)'s
"Keypoint detection" section, not duplicated here — this file stays the
architecture reference. Summary: implement the dual projector (reuse
`projector_p4` with the `cross_attn_projector` prefix — verified present
in the checkpoint), `ConditionalQueryInitializer`, the per-layer keypoint
sublayer (self-attn + deformable cross-attn + FFN, all shapes confirmed
above), the final keypoint decode + class-logit aggregation, then the
conversion script, reference dump, and C++ test — same methodology as
every prior milestone. **No line of `src/keypoints.cpp` has been written
yet**; everything above is architecture research, now checkpoint-verified.
