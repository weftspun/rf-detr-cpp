// RF-DETR detection decoder: two-stage encoder proposals + top-k query
// selection + Deformable-DETR decoder (lite_refpoint_refine, bbox_reparam)
// + final box/class heads. See docs/decisions/decoder.md for the verified
// architecture (checkpoint keys, the topk-ordering caveat, the
// gen_sineembed_for_position (pos_y,pos_x,pos_w,pos_h) ordering, etc).
#pragma once

#include "ops.h"

#include <utility>
#include <vector>

struct DecoderParams {
    int hidden_dim  = 256;
    int dec_layers  = 2;
    int sa_nheads   = 8;
    int ca_nheads   = 16;
    int dec_n_points = 2;
    int num_queries = 300;
    int num_classes = 91;
    int gw = 24, gh = 24; // spatial grid of the single (P4) feature level

    // Multi-level (num_feature_levels>1, e.g. RFDETRLargeDeprecated's P3+P5)
    // support: empty (default) means single-level -- use gw/gh above exactly
    // as every already-validated variant does, zero behavior change. Non-
    // empty selects the multi-level path (masked two-stage proposals +
    // ms_deform_attn_multilevel), levels in the same order as memory's
    // concatenation (matches upstream's spatial_shapes/projector_scale
    // order, e.g. [P3,P5]) -- see docs/decisions/0001-open-work.md.
    std::vector<std::pair<int, int>> levels;
};

struct DecoderOutput {
    ggml_tensor * pred_boxes;        // (4, num_queries, 1), normalized cxcywh
    ggml_tensor * pred_logits;       // (num_classes, num_queries, 1), raw logits
    ggml_tensor * output_proposals;  // (4, total_tokens, 1) INPUT tensor -- caller must
                                      // ggml_backend_tensor_set from output_proposals_data()
                                      // (single-level) or output_proposals_data_multilevel()
                                      // (p.levels non-empty); total_tokens = gw*gh or
                                      // sum(gw_l*gh_l) respectively.
    ggml_tensor * valid_mask = nullptr; // (1, total_tokens, 1) INPUT tensor, ONLY set when
                                        // p.levels is non-empty -- caller must
                                        // ggml_backend_tensor_set from
                                        // valid_mask_data_multilevel(). Masks `memory` before
                                        // the two-stage proposal heads (upstream zeroes tokens
                                        // whose grid-cell proposal falls outside (0.01,0.99) --
                                        // never happens for any single-level grid validated so
                                        // far, but P3's 80x80-scale border cells genuinely do).
    std::vector<ggml_tensor *> hidden_states; // per-layer decoder.norm'd output (dec_layers
                                              // entries, (hidden_dim, num_queries, 1) each) --
                                              // hidden_states.back() == the tensor pred_boxes/
                                              // pred_logits are derived from. Needed by the
                                              // segmentation head (one block per decoder layer).
    ggml_tensor * pred_keypoints = nullptr; // (8, num_kp, num_queries, 1), only set when a
                                            // KeypointParams was passed to rfdetr_decoder.

    // Debug taps (docs/decisions/0001-open-work.md's SegXLarge investigation)
    // -- always populated, not gated behind a flag, since they're cheap
    // (no extra ggml ops, just exposing existing internal tensors).
    ggml_tensor * dbg_ts_boxes = nullptr;   // (4, num_queries, 1) -- top-k-selected two-stage proposal boxes
    ggml_tensor * dbg_ref_points = nullptr; // (4, num_queries, 1) -- fixed reference points fed to every decoder layer
    ggml_tensor * dbg_enc_boxes = nullptr;  // (4, gwh, 1) -- ALL two-stage proposal boxes, before top-k gather
    ggml_tensor * dbg_enc_delta = nullptr;  // (4, gwh, 1) -- enc_out_bbox_embed's raw MLP output, before decode
    ggml_tensor * dbg_topk_idx = nullptr;   // (num_queries,) I32 -- the actual indices used for the gather
    ggml_tensor * dbg_bbox_delta = nullptr; // (4, num_queries, 1) -- bbox_embed's raw MLP output, before final decode
    ggml_tensor * dbg_final_hs = nullptr;   // (hidden_dim, num_queries, 1) -- last decoder layer's normed output
};

// Host-side constant matching gen_encoder_output_proposals (single level, no
// padding): per grid cell (row,col), (cx,cy,w,h) = ((col+.5)/gw,(row+.5)/gh,.05,.05).
// Row-major, width-fastest, matching the backbone/projector flatten convention.
std::vector<float> output_proposals_data(int gw, int gh);

// Multi-level variant of output_proposals_data (p.levels non-empty): per
// level lvl (0-indexed, in `levels`' order), wh = 0.05 * 2^lvl (matches
// upstream's gen_encoder_output_proposals exactly). Concatenates all
// levels' proposals in order, matching memory's own concatenation.
// Proposals whose (cx,cy,w,h) has ANY coordinate outside (0.01,0.99) are
// zeroed (all 4 coords) here directly -- matches upstream's masked_fill(0)
// path (unsigmoid=False, i.e. bbox_reparam=True, this port's only mode).
std::vector<float> output_proposals_data_multilevel(const std::vector<std::pair<int, int>> & levels);

// Companion to output_proposals_data_multilevel: 1.0 for tokens whose
// proposal is valid (see above), 0.0 otherwise -- used to mask `memory`
// itself (a real graph tensor, not host data) before the two-stage
// proposal heads, matching upstream's output_memory.masked_fill(...,0).
std::vector<float> valid_mask_data_multilevel(const std::vector<std::pair<int, int>> & levels);

struct KeypointParams; // src/keypoints.h

// memory: (hidden_dim, gw*gh, 1) token-major flattened projector output
// (or (hidden_dim, sum(gw_l*gh_l), 1) concatenated multi-level, when
// p.levels is non-empty -- RAW, unmasked; masking is applied internally
// before the two-stage proposal heads only, matching upstream, while
// deformable cross-attention still reads the raw memory).
// topk_idx_override: if non-null, used instead of ggml_top_k for the
// two-stage query selection (I32, shape (num_queries)) -- ggml_top_k's
// output order is not guaranteed to match torch.topk's (see
// docs/decisions/decoder.md), so numeric validation against a reference
// dump injects the reference's own indices here rather than fighting an
// order-invariant comparison. Leave null for normal inference.
// kp: if non-null, runs the GroupPose keypoint sublayer after every
// ordinary decoder layer (see docs/decisions/keypoints.md); kp_memory must
// then also be non-null -- the dual projector's second (gw,gh,hidden_dim,1)
// feature map, flattened the same way as `memory`. Not supported together
// with p.levels (no multi-level keypoint variant exists upstream).
// trainable_boxes: if true, the final pred_boxes decode uses
// bbox_reparam_decode_diff (SET-based, clamps delta_wh to [-4,4] before
// exp -- needed because ggml_concat has no backward case, see
// docs/decisions/0003-training.md). If false (default), uses the ORIGINAL
// bbox_reparam_decode (CONCAT-based, no clamp) -- this matches upstream's
// exact unclamped `outputs_coord_wh = delta.exp() * ref_wh` numerically,
// including its float32 exp-underflow-to-EXACTLY-0 behavior for extreme
// delta values. The clamp is a genuine, deliberate training-time
// numerical-safety measure, not an inference-time correctness fix --
// applying it unconditionally silently diverges from upstream whenever
// delta_wh legitimately exceeds +-4 (confirmed: this is exactly what was
// behind RFDETRSegXLarge's large, previously-unexplained divergence,
// see docs/decisions/0001-open-work.md -- ~102/300 queries' delta_wh
// exceeds the clamp for that model+input, so the two versions genuinely
// disagree, not because of a bug in either one).
DecoderOutput rfdetr_decoder(Model & m, ggml_tensor * memory, const DecoderParams & p,
                             ggml_tensor * topk_idx_override = nullptr,
                             const KeypointParams * kp = nullptr, ggml_tensor * kp_memory = nullptr,
                             bool trainable_boxes = false);
