// RF-DETR detection loss for the Phase-2 detection-head-only finetune
// scope (docs/decisions/0003-training.md): DINOv2 backbone + transformer
// decoder frozen, only class_embed/bbox_embed trainable. Hungarian
// matching (host-side, non-differentiable, matches upstream's exact
// matcher.py cost formula) + a ggml loss graph (plain sigmoid focal loss,
// L1, GIoU -- upstream's ia_bce_loss=False path; see
// docs/decisions/0004-loss.md for exactly what was and wasn't
// reproduced and why).
#pragma once

#include "ops.h"

#include <cstdint>
#include <vector>

// One image's ground truth: labels[i] is the 0-indexed class of boxes[i]
// (no background class -- absence of a match is background implicitly).
// boxes is (N*4) cxcywh, normalized to [0,1] relative to image size,
// row-major (boxes[i*4+0..3] = cx,cy,w,h for instance i).
struct DetectionTarget {
    std::vector<int32_t> labels;
    std::vector<float> boxes;
    size_t size() const { return labels.size(); }
};

// query_idx[k]/tgt_idx[k] is one matched (query, target-instance) pair;
// unmatched queries are implicitly background, never listed here.
struct MatchResult {
    std::vector<int32_t> query_idx;
    std::vector<int32_t> tgt_idx;
};

// Host-side Hungarian matching (scipy.optimize.linear_sum_assignment
// equivalent -- a standard O(num_targets^2 * num_queries) Kuhn-Munkres
// with potentials), using RF-DETR's exact matcher cost formula
// (checkpoint-verified via uv run --with rfdetr reading matcher.py):
// cost = cost_bbox*L1(pred_box,tgt_box) + cost_class*focal_cost(pred_logit,tgt_label)
//        + cost_giou*(-GIoU(pred_box,tgt_box))
// where focal_cost is the sigmoid-focal-style classification cost (alpha=
// 0.25, gamma=2 hardcoded upstream), NOT the same as the final loss's
// focal formula (matcher and loss are two separate, similarly-shaped
// computations upstream). pred_logits/pred_boxes are HOST arrays,
// query-major: pred_logits[q*num_classes+c], pred_boxes[q*4+{0..3}]
// (cxcywh, normalized).
MatchResult hungarian_match(const std::vector<float> & pred_logits, const std::vector<float> & pred_boxes,
                            int num_queries, int num_classes, const DetectionTarget & tgt,
                            float cost_class = 2.0f, float cost_bbox = 5.0f, float cost_giou = 2.0f,
                            float focal_alpha = 0.25f);

// Companion host-data builders for detection_loss's three INPUT tensors
// (caller must ggml_backend_tensor_set from these before compute):
// "loss_targets_onehot" (num_classes, num_queries, 1) -- 1.0 at every
// matched (query, target's class) position, 0 elsewhere.
std::vector<float> targets_onehot_data(const MatchResult & match, const DetectionTarget & tgt,
                                       int num_queries, int num_classes);
// "loss_matched_query_idx" (num_matched,) I32 -- match.query_idx verbatim,
// exposed as its own function only for symmetry/clarity (trivial copy).
std::vector<int32_t> matched_query_idx_data(const MatchResult & match);
// "loss_matched_tgt_boxes" (4, num_matched, 1) -- tgt.boxes gathered in
// match order (matched_tgt_boxes[k] = tgt.boxes[match.tgt_idx[k]]).
std::vector<float> matched_tgt_boxes_data(const MatchResult & match, const DetectionTarget & tgt);

// Builds the RF-DETR detection loss graph. pred_logits: (num_classes,
// num_queries, 1). pred_boxes: (4, num_queries, 1) cxcywh normalized --
// both are graphs already built elsewhere (e.g. rfdetr_decoder's output),
// NOT recomputed here. `match` must already be computed (via
// hungarian_match, host-side, on the SAME pred_logits/pred_boxes VALUES --
// run the forward pass once to get those values, match, then build this
// loss graph and re-run/backward on the same input tensor).
//
// Adds three new INPUT tensors to m.ctx_g (named above); caller must
// ggml_set_input+ggml_backend_tensor_set them from the *_data() functions
// above before computing/backward-ing this graph. Returns a scalar loss
// tensor (sum of cls_coef*loss_ce + bbox_coef*loss_bbox + giou_coef*loss_giou,
// matching upstream's default weight_dict).
ggml_tensor * detection_loss(Model & m, ggml_tensor * pred_logits, ggml_tensor * pred_boxes,
                             const DetectionTarget & tgt, const MatchResult & match,
                             int num_queries, int num_classes,
                             float cls_coef = 1.0f, float bbox_coef = 5.0f, float giou_coef = 2.0f,
                             float focal_alpha = 0.25f);
