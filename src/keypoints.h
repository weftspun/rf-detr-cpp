// RF-DETR GroupPose keypoint head (rfdetr/models/heads/keypoints.py +
// the `enable_keypoint_processing` branches of transformer.py). See
// docs/decisions/keypoints.md for the fully-verified architecture:
// dual projector, AdaLN-modulated ConditionalQueryInitializer, a 4th
// per-decoder-layer keypoint sublayer (instance self-attn + keypoint-
// specific deformable cross-attn + FFN), final decode + class-logit boost.
//
// Scoped to grouppose_keypoint_dim_downscale=1 (RFDETRKeypointPreview's
// value): kp_dim == hidden_dim throughout, so inst_in_proj/inst_pos_in_proj/
// inst_out_proj/memory_in_proj are all identities and carry no weights.
// Also scoped to inter_instance_kp_attn=False (also this config's value):
// that whole sublayer is omitted. And to a single keypoint class (COCO-17,
// num_keypoints_per_class=[17]): keypoint_class_mask is a no-op, skipped.
#pragma once

#include "ops.h"

#include <vector>

struct KeypointParams {
    int num_kp   = 17;
    int sa_nheads = 8;  // kp_inst_self_attn
    int ca_nheads = 16; // kp_cross_attn
    int dec_n_points = 2;
    // _format_keypoint_output schema (see keypoint_final_decode): total
    // *declared* keypoint classes and which one is actually active (has
    // real, nonzero keypoints) for this checkpoint. [1,0] is a true
    // single-class schema (no padding); RFDETRKeypointPreview's real
    // checkpoint is [2,0] (two declared classes, only class 0 active).
    int num_keypoint_classes = 1;
    int active_class_idx = 0;
};

// cond: (hidden_dim, 1, num_queries) -- the decoder's content query tgt,
// reshaped so num_queries is the batch dim (ne2). Returns the initial
// keypoint tokens (hidden_dim, num_kp, num_queries), one set of num_kp
// AdaLN-modulated queries per detection query. Weight prefix (unprefixed):
// "<pre>.{queries,adaLN_modulation.0,adaLN_modulation.2,out_proj}".
ggml_tensor * conditional_query_initializer(Model & m, ggml_tensor * cond, const std::string & pre,
                                            int num_kp, int hidden_dim);

// One decoder layer's keypoint sublayer (runs after the ordinary detection
// self-attn/cross-attn/FFN sublayers). tgt/query_pos: (hidden_dim,
// num_queries, 1). keypoint_tgt: (hidden_dim, num_kp, num_queries).
// keypoint_pos: (hidden_dim, num_kp, 1) -- the shared learned
// keypoint_pos_embed, same for every query/layer. ref_points: (4,
// num_queries, 1), the fixed decoder reference boxes. kp_memory: (hidden_dim,
// gw*gh, 1), the dual projector's second feature map. Mutates neither input;
// returns the updated {tgt, keypoint_tgt} pair.
struct KeypointLayerOutput { ggml_tensor * tgt; ggml_tensor * keypoint_tgt; };
KeypointLayerOutput keypoint_decoder_layer(Model & m, ggml_tensor * tgt, ggml_tensor * query_pos,
                                           ggml_tensor * keypoint_tgt, ggml_tensor * keypoint_pos,
                                           ggml_tensor * ref_points, ggml_tensor * kp_memory,
                                           const std::string & pre, const KeypointParams & kp,
                                           int gw, int gh);

struct KeypointHeadOutput {
    ggml_tensor * pred_keypoints;      // (8, num_keypoint_classes*max_kp_per_class, num_queries, 1)
    ggml_tensor * class_logit_boost;   // (1, num_queries, 1) -- add to pred_logits' class-0 row
};

// keypoint_tgt_last: the LAST decoder layer's keypoint tokens (hidden_dim,
// num_kp, num_queries) -- only the last layer is needed, matching
// detection's pred_boxes/pred_logits pattern. ref_points: the same fixed
// (4, num_queries, 1) decoder reference boxes. Weight prefix "keypoint_embed".
//
// _format_keypoint_output (rfdetr/models/lwdetr.py): upstream's "compact"
// (num_kp real keypoints) output gets padded to a per-class layout,
// num_keypoint_classes*max_kp_per_class total slots, with each class'
// keypoints placed at its own class_idx*max_kp_per_class offset and all
// other slots zero. This port is scoped to a single ACTIVE class (the
// checkpoint's real schema) at `active_class_idx` -- pass
// num_keypoint_classes=1 and active_class_idx=0 for a true single-class
// schema (a no-op pad). For rf-detr-keypoint-preview specifically:
// num_keypoint_classes=2, active_class_idx=1 -- confirmed against the
// checkpoint's own `_kp_active_mask` buffer (row 0 all-False, row 1
// all-True), NOT the initially-guessed [17,0]/active_class_idx=0, which
// looked self-consistent (matched a reference dump built with the same
// wrong guess) but was wrong -- the class-logit boost silently came out
// to exactly 0 for every query with that guess, only caught by directly
// inspecting _kp_active_mask's boolean values, not its shape alone.
KeypointHeadOutput keypoint_final_decode(Model & m, ggml_tensor * keypoint_tgt_last,
                                         ggml_tensor * ref_points, int num_kp,
                                         int num_keypoint_classes = 1, int active_class_idx = 0);
