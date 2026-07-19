# Backbone port decisions (DINOv2 windowed attention)

Status: **validated** for RFDETRNano — `test_backbone` diffs all 4 tapped
feature maps against the real upstream `WindowedDinov2WithRegistersBackbone`
module run on the actual `rf-detr-nano.pth` checkpoint, max-abs-diff ≤2.5e-4
(gate 5e-2). Everything below that predates this validation is now confirmed
correct for the Nano config; other variants (different patch_size/registers/
resolution, and position-embedding interpolation) are still unvalidated.

Originally written mostly verified against the real upstream source
(`roboflow/rf-detr` @ `develop`, cloned locally and read directly — not a
guess anymore for the architecture; still unverified against an actual
downloaded `.pth` checkpoint's `state_dict().keys()`, which is the next
concrete step).

## How windowing actually works (verified from `dinov2_with_windowed_attn.py`)

This is **not** "spatial windows extracted per attention layer" as the first
draft of this port assumed. Windowing happens **once**, at the embeddings
stage, by expanding the batch dimension:

1. Patch tokens get CLS-prepended and position-embedded as a normal
   `(B, 1+n_patch, C)` sequence first.
2. Then the whole thing is partitioned into `num_windows²` non-overlapping
   spatial windows, each becoming its own batch entry: `(B*num_windows²,
   1+patches_per_window, C)`. **The CLS token is repeated identically into
   every window** (`cls_token.repeat(num_windows**2, 1, 1)`), so each window
   ends up with its own independently-evolving CLS token.
3. Register tokens are inserted *after* windowing and are **also repeated
   per window** (`register_tokens.expand(B*num_windows², ...)`), inserted at
   position 1 (between CLS and patches).
4. Every subsequent encoder layer, windowed or not, operates on this
   `(B*num_windows², T, C)` windowed-batch tensor:
   - **Windowed layers** (`i in window_block_indexes`): ordinary self-attention
     per batch entry — since each batch entry already is exactly one window's
     tokens (CLS + registers + that window's patches), no extra windowing
     logic is needed inside the layer at all.
   - **"Full"/global layers** (`i not in window_block_indexes`): temporarily
     **merge** all `num_windows²` windows belonging to the same image back
     into one long sequence (`(B, num_windows²·T, C)`), run ordinary
     self-attention over that (so every window's CLS/register/patch tokens
     attend to every other window's), then **split back** to `(B·nw², T, C)`
     for the next layer.
5. `window_block_indexes` is **derived**, not a free hyperparameter — but
   with an off-by-one quirk that matters a lot, verified against a real
   downloaded checkpoint (see "Verified against a real checkpoint" below):
   `rfdetr/models/backbone/dinov2.py` computes
   `window_block_indexes = set(range(out_feature_indexes[-1]+1)) -
   set(out_feature_indexes)` **using the config's raw `out_feature_indexes`
   values, which are 1-based "stage" numbers** (`stage{i}` = output *after*
   0-indexed encoder layer `i-1`; see `WindowedDinov2WithRegistersEncoder`'s
   `stage_names`/`all_hidden_states` bookkeeping). That computed set is then
   used **directly, unshifted**, against the 0-indexed layer loop variable
   `i` (`run_full_attention = i not in config.window_block_indexes`) — i.e.
   upstream does NOT shift back to layer-index space before reusing the set.
   Net effect for RFDETRNano's `out_feature_indexes=[3,6,9,12]` (12 layers,
   indices 0-11): `window_block_indexes = {0,1,2,4,5,7,8,10,11}` (computed
   from stage-space, used as layer-space), so **only layers 3, 6, 9 run
   global/full attention** — layer 11 (the last layer, whose output is
   tapped as `stage12`) runs **windowed**, not global. The earlier version
   of this doc claimed "every tapped layer runs global attention" —
   **that's wrong**; it happened to hold for the earlier taps but not the
   last one, precisely because of this off-by-one. When populating
   `BackboneParams`, reproduce the exact upstream formula, not a
   re-derivation in layer-index space:
   ```
   raw = config.out_feature_indexes                    # e.g. [3, 6, 9, 12]
   window_block_indexes = sorted(set(range(raw[-1] + 1)) - set(raw))  # used as-is, layer-space
   out_feature_indexes   = [x - 1 for x in raw]          # shift to 0-based layer-output index
   ```
6. Feature-map extraction (`WindowedDinov2WithRegistersBackbone.forward`):
   for each tapped layer output (still in windowed-batch form), apply the
   model's final `layernorm` (yes — to *every* tap, not just the last),
   strip the CLS+register prefix, then undo the window partition back to a
   `(B, C, H, W)` spatial map for the projector.

`src/backbone.cpp` now implements exactly this (see `dinov2_backbone`):
window-partition once via a small loop over `num_windows²` (cheap — nw is 1,
2, or 4), then either run `dinov2_block` directly (windowed layer, batch =
`num_windows²`) or reshape-merge to batch=1/seq=`nw²·T`, run `dinov2_block`,
reshape-split back (global layer) — a pair of `ggml_reshape` calls with no
data movement, since token-major layout already puts the window index as
the natural outer/slower axis when merged with the token axis. **Batching
(N>1 images per graph) is not yet supported** — the merge/split reshape
trick assumes a single image; multi-image batching needs the windows to stay
grouped per-image, which the current code doesn't arrange for. Fine for
single-image inference (the common case for this port so far).

## Per-variant configs (verified from `rfdetr/config.py`)

| Config (`ModelConfig` subclass) | encoder size | patch | num_windows | out_feature_indexes | resolution | positional_encoding_size |
|---|---|---|---|---|---|---|
| RFDETRBase (`dinov2_windowed_small`) | Small | 14 | 4 | [2,5,8,11] | 560 | 37 |
| RFDETRLarge-deprecated (`dinov2_windowed_base`) | Base | 14 (inherited) | 4 | [2,5,8,11] | 560 (inherited) | 37 |
| RFDETRNano | Small | 16 | 2 | [3,6,9,12] | 384 | — |
| (Small/Medium/2xLarge — see config.py for the rest, same shape as Nano's block with different resolution/dec_layers) | | | | | | |

**Position-embedding interpolation is conditionally required, not always
skippable** as the first draft assumed:
- `patch_size == 14` (native DINOv2) *and* `positional_encoding_size *
  patch_size == checkpoint's native image_size` (518, the standard DINOv2
  training resolution) → DINOv2 pretrained weights load, `image_size` stays
  518, so position embeddings are stored at a 37×37 grid. **If the model's
  actual `resolution` differs from 518 (e.g. RFDETRBase runs at 560px),
  bicubic interpolation to the runtime grid size genuinely happens at
  inference** (`interpolate_pos_encoding`, bicubic, `antialias=True` off
  MPS). Not implemented in this port yet.
- `patch_size != 14` (Nano and most other variants use 16 or 12) → DINOv2
  pretrained weights are **not** loaded at all (`load_dinov2_weights=False`);
  `image_size` is set directly to the model's own training resolution, so
  the checkpoint's `position_embeddings` are already sized exactly for
  inference and **no interpolation is needed**.

**Recommended first validation target**: a `patch_size != 14` variant (e.g.
RFDETRNano) — it needs zero position-embedding interpolation and can be
validated with the current code as-is. Save the interpolation implementation
for whichever `patch_size == 14` variant (RFDETRBase/Large-deprecated) comes
next.

## Verified against a real checkpoint

Downloaded `rf-detr-nano.pth`
(`https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth`,
366MB) and read its parameter names straight out of the torch zip archive's
`data.pkl` (regex-scanned the embedded pickle strings — no torch install
needed for this). Confirms:

- **State-dict key prefix is `backbone.0.encoder.encoder.`** for the
  embeddings/encoder-layer/layernorm tensors — one level deeper than the
  original guess (`backbone.0.encoder.`). Full paths look like
  `backbone.0.encoder.encoder.embeddings.cls_token`,
  `backbone.0.encoder.encoder.encoder.layer.{i}.norm1.weight` (yes, three
  `encoder`s: `Joiner[0]` → `DinoV2.encoder` (the
  `WindowedDinov2WithRegistersBackbone`) → its own `.encoder`
  (`WindowedDinov2WithRegistersEncoder`) → `.layer.{i}`),
  `backbone.0.encoder.encoder.layernorm.weight`. The conversion script
  should strip `backbone.0.encoder.encoder.` to land on exactly the
  unprefixed names `backbone.cpp` already uses (`embeddings.*`,
  `encoder.layer.{i}.*`, `layernorm.*`) — no other changes needed there.
- **RFDETRNano has no register tokens** — no `register_tokens` key anywhere
  in the checkpoint, confirming `dinov2_windowed_small` (no `"registers"` in
  the encoder name) uses the register-free config path.
  `BackboneParams::n_register = 0` for Nano.
- 12 encoder layers present (`encoder.layer.0` through `.11`), matching
  `dinov2_small.json`'s `num_hidden_layers: 12`.
- Other top-level checkpoint keys confirm the wider model: `bbox_embed`,
  `class_embed`, `query_feat`, `refpoint_embed`, `transformer` (the
  Deformable-DETR decoder — no `segmentation`/`keypoint` head keys, as
  expected for a plain detection checkpoint).

## Confirmed, previously listed as unverified

- `hidden_act` default `"gelu"` → HF's `ACT2FN["gelu"]` is erf-based exact
  GELU (`ggml_gelu_erf`), not tanh-approximate — used in `dinov2_block`.
  Confirmed correct by `test_backbone`'s numeric pass.

Open tasks (backbone variants, position-embedding interpolation, `qkv_bias`
presence, multi-image batching) are tracked in
[`0001-open-work.md`](0001-open-work.md), not duplicated here.
