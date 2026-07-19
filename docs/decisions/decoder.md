# Decoder port decisions (Deformable-DETR detection head)

Status: **validated end-to-end** for RFDETRNano. `tests/test_decoder.cpp`
runs the full pipeline (`dinov2_backbone` -> `projector_p4` -> `rfdetr_decoder`)
against the real checkpoint and diffs against `gen_reference/gen_reference_decoder.py`'s
dump of the actual upstream `LWDETR` model: `pred_boxes` max-abs-diff 3.3e-4,
`pred_logits` max-abs-diff 7.3e-4 (gate 5e-2 both). **Object detection
inference for RFDETRNano is complete.** The `topk_idx_override` mechanism
(see "Top-k selection" below) is used for this validation; normal inference
uses `ggml_top_k` directly.

## Checkpoint keys (all confirmed present in rf-detr-nano.pth, no guessing)

`refpoint_embed.weight` (3900,4), `query_feat.weight` (3900,256),
`class_embed.{weight,bias}` (91,256)/(91,), `bbox_embed.layers.{0,1,2}.*`,
`transformer.enc_output.0.*`, `transformer.enc_output_norm.0.*`,
`transformer.enc_out_class_embed.0.*`, `transformer.enc_out_bbox_embed.0.layers.{0,1,2}.*`
(groups 1-12 also present, training-only, not loaded),
`transformer.decoder.layers.{0,1}.{self_attn.{in_proj_weight,in_proj_bias,
out_proj.weight,out_proj.bias}, norm1.*, cross_attn.{sampling_offsets,
attention_weights,value_proj,output_proj}.*, norm2.*, linear1.*, linear2.*,
norm3.*}`, `transformer.decoder.ref_point_head.layers.{0,1}.*`,
`transformer.decoder.norm.*`. No `register`/`keypoint`/`grouppose` keys —
those code paths are inactive for a plain-detection Nano checkpoint, skipped
entirely in this port.

**Important correction to the initial plan**: `refpoint_embed.weight[:300]`
is **not** all-zero at inference — it's a real trained parameter (checked:
std≈0.04 on the loaded checkpoint). The plan's description of "decoder
reference points = the two-stage top-k proposal boxes" was incomplete; the
actual reference point is a **reparameterized combination** of the two-stage
proposal box and this learned per-query delta (see "Reference points" below)
— this must not be skipped.

## Two-stage proposals (`gen_encoder_output_proposals`, single level, no padding)

Per spatial location `(row,col)` in the `gw×gh` grid: fixed (non-learned)
candidate box `cx=(col+0.5)/gw, cy=(row+0.5)/gh, w=h=0.05`. With no padding
mask, `valid_height=gh, valid_width=gw` exactly (no dynamic masking needed).
The upstream code additionally zeros out any candidate whose 4 coords fall
outside `(0.01, 0.99)` — **for Nano's 24×24 grid every candidate already
satisfies this** (`0.5/24≈0.021` to `23.5/24≈0.979`, and `w=h=0.05` is
comfortably inside), so **this port skips the masking step entirely** as a
documented simplification valid only because of that arithmetic fact; it
would need implementing for a grid where some cell fails the check.

`output_memory = enc_output_norm[0](enc_output[0](memory))`, then
`enc_out_class_embed[0]`/`enc_out_bbox_embed[0]` (group 0 only, per the
established "group_detr is a no-op at inference" finding) score/decode each
location's proposal via the same `bbox_reparam` delta formula used
everywhere else in this model (`cxcy = delta[:2]*prop_wh + prop_cxcy; wh =
exp(delta[2:])*prop_wh`).

## Top-k selection: **ggml_top_k's output order is not guaranteed to match
`torch.topk`'s**

`torch.topk` returns indices in descending-score order; the resulting
`pred_boxes[i]`/`pred_logits[i]` array position depends on that order.
`ggml_top_k`'s docstring explicitly says "the resulting top k indices are in
no particular order." **Validating `test_decoder` therefore cannot use a
plain positional max-abs-diff** the way `test_backbone`/`test_projector` do
— the set of top-300 boxes/logits should match, but possibly permuted.
Handle this by sorting both the ggml output and the reference by (e.g.)
score or box cx before diffing, or by a nearest-box matching comparison,
when writing `tests/test_decoder.cpp`.

Score-per-location = max over the 91 class logits, computed via
`ggml_pool_1d(..., GGML_OP_POOL_MAX, k0=num_classes, s0=num_classes, 0)`
(no native reduce-max-along-ne0 op in this ggml version; max-pooling with a
kernel spanning the whole class axis is the workaround).

## Reference points fed to the decoder (fixed for all layers, `lite_refpoint_refine=True`)

```
ts_boxes = gather(enc_boxes, topk_idx)                     # (4, num_queries) two-stage proposal boxes
subset   = refpoint_embed.weight[:num_queries]              # LEARNED, checkpoint-loaded, not zero
final_cxcy = subset[:2] * ts_boxes.wh + ts_boxes.cxcy
final_wh   = exp(subset[2:]) * ts_boxes.wh
ref_points = concat(final_cxcy, final_wh)                   # (4, num_queries) -- fixed for every decoder layer
```
Content queries: `tgt = query_feat.weight[:num_queries]` directly (learned,
unrelated to the two-stage `memory`/`output_memory` gather — that gather
result, `memory_ts`/`tgt_undetach` upstream, is training/aux-output only and
is not computed by this port).

## Query positional embedding (`gen_sineembed_for_position`, dim=128 per coord)

Applied to all 4 ref-point coords, concatenated in **`(pos_y, pos_x, pos_w,
pos_h)`** order (not x,y,w,h — confirmed from source: `x_embed =
pos_tensor[...,0]`, `y_embed = pos_tensor[...,1]`, then
`cat((pos_y,pos_x,pos_w,pos_h))`) → 512-wide, then
`ref_point_head` (2-layer MLP, ReLU between, 512→256→256) → `query_pos`,
computed **once** before the layer loop (per `lite_refpoint_refine`).

Per-coordinate embedding: `dim_t[i] = 10000^(2*(i//2)/128)` (a **constant**,
not learned) — ported as a GGUF-baked constant tensor (`sine_scale = 2π /
dim_t`, precomputed by the conversion script) rather than recomputed at
graph-build time, since ggml has no direct pow/arange-with-fractional-exp
primitive convenient for this. `raw[d,q] = v[q] * sine_scale[d]` is an outer
product, computed via `ggml_mul_mat` on `(1,dim)` × `(1,n_query)` tensors.
The interleaved-sin/cos reassembly (`stack(sin(raw[0::2]),
cos(raw[1::2])).flatten()`) is done via a `reshape(2,dim/2,...)` + strided
view + `ggml_sin`/`ggml_cos` + concat + reshape-back trick — see
`sine_embed_1coord` in `src/decoder.cpp`.

## Decoder layer (`forward_post`, post-norm, ×2 for Nano)

1. Self-attn: standard MHA, `q=k=tgt+query_pos, v=tgt`, 8 heads. PyTorch's
   `nn.MultiheadAttention` stores a **fused** `in_proj_weight` (768,256) /
   `in_proj_bias` (768,) — split into three (256,256)/(256,) q/k/v tensors
   at GGUF-conversion time (rows `[0:256]`=q, `[256:512]`=k, `[512:768]`=v),
   written as separate `self_attn.{q,k,v}_proj.{weight,bias}` tensors so
   `src/decoder.cpp` can reuse the existing generic multi-head-attention
   pattern instead of special-casing fused QKV in `ops.h`.
2. Deformable cross-attn (`src/deform_attn.cpp`, `ms_deform_attn`):
   `query = tgt + query_pos`, `value_input = memory` (the projector's fused
   P4 map, flattened), `ref_points` = the fixed reference points above
   (same for every layer).
3. FFN: `linear1(256→2048) → ReLU → linear2(2048→256)` (`dim_feedforward`
   hardcoded 2048, not a `ModelConfig` field).
4. All three sublayers are residual + **post**-LayerNorm (`norm1/2/3`).

**`decoder.norm` is not applied between layers** — `return_intermediate`
copies+norms each layer's output for the (unused-at-inference) aux list,
but the **unnormed** `output` tensor is what feeds the next layer. Only
apply `decoder.norm` once, to the last layer's raw output, before the final
heads.

## Final heads

`class_embed` (single linear, 256→91) and `bbox_embed` (3-layer MLP,
256→256→256→4, ReLU between, no activation after last) applied to the
final normed decoder output. Box decode uses the **same fixed reference
points** as the decoder's cross-attention (not re-derived):
`cxcy = delta[:2]*ref_wh + ref_cxcy; wh = exp(delta[2:])*ref_wh`. No final
sigmoid on `pred_logits` (upstream applies it in post-processing, not the
model forward) — worth confirming to the reference dump either way.
