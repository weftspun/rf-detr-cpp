// RF-DETR detection decoder: two-stage encoder proposals + top-k query
// selection + Deformable-DETR decoder (lite_refpoint_refine, bbox_reparam)
// + final box/class heads. See docs/decisions/decoder.md for the verified
// architecture (checkpoint keys, the topk-ordering caveat, the
// gen_sineembed_for_position (pos_y,pos_x,pos_w,pos_h) ordering, etc).
#pragma once

#include "ops.h"

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
};

struct DecoderOutput {
    ggml_tensor * pred_boxes;        // (4, num_queries, 1), normalized cxcywh
    ggml_tensor * pred_logits;       // (num_classes, num_queries, 1), raw logits
    ggml_tensor * output_proposals;  // (4, gw*gh, 1) INPUT tensor -- caller must
                                      // ggml_backend_tensor_set from output_proposals_data()
    std::vector<ggml_tensor *> hidden_states; // per-layer decoder.norm'd output (dec_layers
                                              // entries, (hidden_dim, num_queries, 1) each) --
                                              // hidden_states.back() == the tensor pred_boxes/
                                              // pred_logits are derived from. Needed by the
                                              // segmentation head (one block per decoder layer).
    ggml_tensor * pred_keypoints = nullptr; // (8, num_kp, num_queries, 1), only set when a
                                            // KeypointParams was passed to rfdetr_decoder.
};

// Host-side constant matching gen_encoder_output_proposals (single level, no
// padding): per grid cell (row,col), (cx,cy,w,h) = ((col+.5)/gw,(row+.5)/gh,.05,.05).
// Row-major, width-fastest, matching the backbone/projector flatten convention.
std::vector<float> output_proposals_data(int gw, int gh);

struct KeypointParams; // src/keypoints.h

// memory: (hidden_dim, gw*gh, 1) token-major flattened projector output.
// topk_idx_override: if non-null, used instead of ggml_top_k for the
// two-stage query selection (I32, shape (num_queries)) -- ggml_top_k's
// output order is not guaranteed to match torch.topk's (see
// docs/decisions/decoder.md), so numeric validation against a reference
// dump injects the reference's own indices here rather than fighting an
// order-invariant comparison. Leave null for normal inference.
// kp: if non-null, runs the GroupPose keypoint sublayer after every
// ordinary decoder layer (see docs/decisions/keypoints.md); kp_memory must
// then also be non-null -- the dual projector's second (gw,gh,hidden_dim,1)
// feature map, flattened the same way as `memory`.
DecoderOutput rfdetr_decoder(Model & m, ggml_tensor * memory, const DecoderParams & p,
                             ggml_tensor * topk_idx_override = nullptr,
                             const KeypointParams * kp = nullptr, ggml_tensor * kp_memory = nullptr);
