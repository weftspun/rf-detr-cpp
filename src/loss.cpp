#include "loss.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

static float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// cxcywh -> xyxy, host floats.
static std::array<float, 4> to_xyxy(const float * b) {
    float hw = b[2] * 0.5f, hh = b[3] * 0.5f;
    return { b[0] - hw, b[1] - hh, b[0] + hw, b[1] + hh };
}

static float giou_host(const float * pred, const float * tgt) {
    auto b1 = to_xyxy(pred), b2 = to_xyxy(tgt);
    float ix1 = std::max(b1[0], b2[0]), iy1 = std::max(b1[1], b2[1]);
    float ix2 = std::min(b1[2], b2[2]), iy2 = std::min(b1[3], b2[3]);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    float area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    float uni = area1 + area2 - inter + 1e-7f;
    float iou = inter / uni;
    float ex1 = std::min(b1[0], b2[0]), ey1 = std::min(b1[1], b2[1]);
    float ex2 = std::max(b1[2], b2[2]), ey2 = std::max(b1[3], b2[3]);
    float earea = std::max(0.0f, ex2 - ex1) * std::max(0.0f, ey2 - ey1) + 1e-7f;
    return iou - (earea - uni) / earea;
}

// Standard O(n^2*m) Kuhn-Munkres with potentials (n <= m), 1-indexed
// internally (classic e-maxx formulation). cost is n x m, row-major.
// Returns, for each row (0-indexed), its assigned column (0-indexed).
static std::vector<int> kuhn_munkres(const std::vector<float> & cost, int n, int m) {
    const float INF = std::numeric_limits<float>::max() / 2;
    std::vector<float> u(n + 1, 0.0f), v(m + 1, 0.0f);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(m + 1, INF);
        std::vector<bool> used(m + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = -1;
            float delta = INF;
            for (int j = 1; j <= m; j++) {
                if (!used[j]) {
                    float cur = cost[(size_t) (i0 - 1) * m + (j - 1)] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= m; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> result(n, -1);
    for (int j = 1; j <= m; j++) {
        if (p[j] > 0) {
            result[p[j] - 1] = j - 1;
        }
    }
    return result;
}

MatchResult hungarian_match(const std::vector<float> & pred_logits, const std::vector<float> & pred_boxes,
                            int num_queries, int num_classes, const DetectionTarget & tgt,
                            float cost_class, float cost_bbox, float cost_giou, float focal_alpha) {
    MatchResult result;
    const int n = (int) tgt.size();
    if (n == 0) {
        return result;
    }
    const float gamma = 2.0f;

    std::vector<float> cost((size_t) n * num_queries);
    for (int t = 0; t < n; t++) {
        const float * tbox = &tgt.boxes[(size_t) t * 4];
        int label = tgt.labels[t];
        for (int q = 0; q < num_queries; q++) {
            const float * qbox = &pred_boxes[(size_t) q * 4];
            float logit = pred_logits[(size_t) q * num_classes + label];
            float prob = sigmoidf_(logit);
            // -log(sigmoid(x)) and -log(1-sigmoid(x)), numerically simple
            // (matches upstream's -F.logsigmoid form) since matching-time
            // logits are finite and this isn't backprop'd through.
            float neg_log_p = -std::log(std::max(prob, 1e-12f));
            float neg_log_1mp = -std::log(std::max(1.0f - prob, 1e-12f));
            float pos_cost = focal_alpha * std::pow(1.0f - prob, gamma) * neg_log_p;
            float neg_cost = (1.0f - focal_alpha) * std::pow(prob, gamma) * neg_log_1mp;
            float cost_class_val = pos_cost - neg_cost;

            float l1 = 0.0f;
            for (int k = 0; k < 4; k++) {
                l1 += std::fabs(qbox[k] - tbox[k]);
            }
            float giou = giou_host(qbox, tbox);

            cost[(size_t) t * num_queries + q] =
                cost_bbox * l1 + cost_class * cost_class_val + cost_giou * (-giou);
        }
    }

    std::vector<int> assign = kuhn_munkres(cost, n, num_queries); // assign[target] = query
    // Defensive: a non-finite cost value (e.g. a frozen/untrained head fed
    // wildly out-of-distribution input producing NaN/Inf logits) can make
    // the algorithm's internal comparisons misbehave and leave a target
    // unassigned (-1) even though n<=num_queries guarantees a real
    // assignment exists for finite costs -- skip those rather than emit an
    // invalid query index downstream.
    for (int t = 0; t < n; t++) {
        if (assign[t] >= 0 && assign[t] < num_queries) {
            result.query_idx.push_back(assign[t]);
            result.tgt_idx.push_back(t);
        }
    }
    return result;
}

std::vector<float> targets_onehot_data(const MatchResult & match, const DetectionTarget & tgt,
                                       int num_queries, int num_classes) {
    std::vector<float> out((size_t) num_classes * num_queries, 0.0f);
    for (size_t k = 0; k < match.query_idx.size(); k++) {
        int q = match.query_idx[k];
        int c = tgt.labels[match.tgt_idx[k]];
        out[(size_t) q * num_classes + c] = 1.0f;
    }
    return out;
}

std::vector<int32_t> matched_query_idx_data(const MatchResult & match) {
    return match.query_idx;
}

std::vector<float> matched_tgt_boxes_data(const MatchResult & match, const DetectionTarget & tgt) {
    std::vector<float> out(match.tgt_idx.size() * 4);
    for (size_t k = 0; k < match.tgt_idx.size(); k++) {
        const float * b = &tgt.boxes[(size_t) match.tgt_idx[k] * 4];
        for (int j = 0; j < 4; j++) {
            out[k * 4 + j] = b[j];
        }
    }
    return out;
}

// sigmoid(x) = 1/(1+exp(-x)) built from primitives with backward support
// (ggml_sigmoid itself has NO backward case -- see
// docs/decisions/0003-training.md's LayerNorm precedent for the same
// class of gap; SCALE/EXP/ADD/DIV are all confirmed backward-capable).
static ggml_tensor * sigmoid_diff(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * exp_neg = ggml_exp(ctx, ggml_scale(ctx, x, -1.0f));
    ggml_tensor * denom = ggml_scale_bias(ctx, exp_neg, 1.0f, 1.0f);
    ggml_tensor * ones = ggml_scale_bias(ctx, denom, 0.0f, 1.0f);
    return ggml_div(ctx, ones, denom);
}

static ggml_tensor * elementwise_max(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    return ggml_add(ctx, a, ggml_relu(ctx, ggml_sub(ctx, b, a)));
}
static ggml_tensor * elementwise_min(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    return ggml_sub(ctx, a, ggml_relu(ctx, ggml_sub(ctx, a, b)));
}
// clamp_diff (backward-capable clamp) is a shared primitive -- see
// ops.h/ops.cpp. Used below to keep sigmoid's output away from EXACTLY
// 0.0/1.0 before log() (a frozen model fed out-of-distribution input can
// legitimately saturate sigmoid in float32).

ggml_tensor * detection_loss(Model & m, ggml_tensor * pred_logits, ggml_tensor * pred_boxes,
                             const DetectionTarget & tgt, const MatchResult & match,
                             int num_queries, int num_classes,
                             float cls_coef, float bbox_coef, float giou_coef, float focal_alpha) {
    ggml_context * ctx = m.ctx_g;
    const int num_matched = (int) match.query_idx.size();
    const float gamma = 2.0f;

    // --- classification: plain sigmoid focal loss over ALL queries/classes ---
    ggml_tensor * targets_onehot = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, num_classes, num_queries, 1);
    ggml_set_name(targets_onehot, "loss_targets_onehot");
    ggml_set_input(targets_onehot);

    ggml_tensor * prob = clamp_diff(ctx, sigmoid_diff(ctx, pred_logits), 1e-6f, 1.0f - 1e-6f);
    ggml_tensor * one_minus_p = ggml_scale_bias(ctx, prob, -1.0f, 1.0f);
    ggml_tensor * one_minus_t = ggml_scale_bias(ctx, targets_onehot, -1.0f, 1.0f);

    ggml_tensor * log_p = ggml_log(ctx, prob);
    ggml_tensor * log_1mp = ggml_log(ctx, one_minus_p);
    ggml_tensor * bce = ggml_scale(ctx, ggml_add(ctx, ggml_mul(ctx, targets_onehot, log_p),
                                                  ggml_mul(ctx, one_minus_t, log_1mp)), -1.0f);

    ggml_tensor * pt = ggml_add(ctx, ggml_mul(ctx, prob, targets_onehot),
                                ggml_mul(ctx, one_minus_p, one_minus_t));
    ggml_tensor * one_minus_pt = ggml_scale_bias(ctx, pt, -1.0f, 1.0f);
    ggml_tensor * modulator = ggml_mul(ctx, one_minus_pt, one_minus_pt); // gamma=2, hardcoded upstream too

    ggml_tensor * alpha_t = ggml_add(ctx, ggml_scale(ctx, targets_onehot, focal_alpha),
                                     ggml_scale(ctx, one_minus_t, 1.0f - focal_alpha));

    ggml_tensor * focal_elem = ggml_mul(ctx, ggml_mul(ctx, alpha_t, modulator), bce);
    ggml_tensor * loss_ce_sum = ggml_sum(ctx, focal_elem);
    // upstream: sigmoid_focal_loss(...).mean(1).sum()/num_boxes * num_queries.
    // src_logits is (batch,queries,classes) -- mean(1) averages over
    // QUERIES (dim index 1), not classes; that 1/num_queries exactly
    // cancels the outer "* num_queries" multiply in loss_labels(), leaving
    // simply sum-over-everything / num_boxes (checkpoint-verified by
    // reading criterion.py directly after an initial wrong guess that
    // mean(1) was over the class axis produced a ~2.2x-too-large loss).
    ggml_tensor * loss_ce = ggml_scale(ctx, loss_ce_sum, 1.0f / (float) std::max(1, num_matched));

    ggml_tensor * total = ggml_scale(ctx, loss_ce, cls_coef);
    if (num_matched == 0) {
        return total; // nothing to match this batch -- classification-only loss
    }

    // --- box losses: L1 + GIoU, matched pairs only ---
    ggml_tensor * matched_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_matched);
    ggml_set_name(matched_idx, "loss_matched_query_idx");
    ggml_set_input(matched_idx);

    ggml_tensor * pred_boxes_2d = ggml_reshape_2d(ctx, pred_boxes, 4, num_queries);
    ggml_tensor * matched_pred = ggml_get_rows(ctx, pred_boxes_2d, matched_idx); // (4,num_matched)
    matched_pred = ggml_reshape_3d(ctx, matched_pred, 4, num_matched, 1);

    ggml_tensor * tgt_boxes_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, num_matched, 1);
    ggml_set_name(tgt_boxes_t, "loss_matched_tgt_boxes");
    ggml_set_input(tgt_boxes_t);

    ggml_tensor * diff = ggml_sub(ctx, matched_pred, tgt_boxes_t); // same shape, safe (no broadcast)
    ggml_tensor * abs_diff = ggml_abs(ctx, diff);
    ggml_tensor * loss_bbox = ggml_scale(ctx, ggml_sum(ctx, abs_diff), 1.0f / num_matched);
    total = ggml_add(ctx, total, ggml_scale(ctx, loss_bbox, bbox_coef));

    auto coord = [&](ggml_tensor * box, int off) {
        return ggml_cont(ctx, ggml_view_3d(ctx, box, 1, num_matched, 1, box->nb[1], box->nb[2],
                                           (size_t) off * sizeof(float)));
    };
    auto cxcywh_to_xyxy = [&](ggml_tensor * box) {
        ggml_tensor * cx = coord(box, 0), * cy = coord(box, 1), * w = coord(box, 2), * h = coord(box, 3);
        ggml_tensor * hw = ggml_scale(ctx, w, 0.5f), * hh = ggml_scale(ctx, h, 0.5f);
        return std::array<ggml_tensor *, 4>{
            ggml_sub(ctx, cx, hw), ggml_sub(ctx, cy, hh), ggml_add(ctx, cx, hw), ggml_add(ctx, cy, hh)
        };
    };
    std::array<ggml_tensor *, 4> b1 = cxcywh_to_xyxy(matched_pred);
    std::array<ggml_tensor *, 4> b2 = cxcywh_to_xyxy(tgt_boxes_t);

    ggml_tensor * ix1 = elementwise_max(ctx, b1[0], b2[0]), * iy1 = elementwise_max(ctx, b1[1], b2[1]);
    ggml_tensor * ix2 = elementwise_min(ctx, b1[2], b2[2]), * iy2 = elementwise_min(ctx, b1[3], b2[3]);
    ggml_tensor * iw = ggml_relu(ctx, ggml_sub(ctx, ix2, ix1));
    ggml_tensor * ih = ggml_relu(ctx, ggml_sub(ctx, iy2, iy1));
    ggml_tensor * inter = ggml_mul(ctx, iw, ih);

    ggml_tensor * area1 = ggml_mul(ctx, ggml_sub(ctx, b1[2], b1[0]), ggml_sub(ctx, b1[3], b1[1]));
    ggml_tensor * area2 = ggml_mul(ctx, ggml_sub(ctx, b2[2], b2[0]), ggml_sub(ctx, b2[3], b2[1]));
    ggml_tensor * uni = ggml_scale_bias(ctx, ggml_sub(ctx, ggml_add(ctx, area1, area2), inter), 1.0f, 1e-7f);
    ggml_tensor * iou = ggml_div(ctx, inter, uni);

    ggml_tensor * ex1 = elementwise_min(ctx, b1[0], b2[0]), * ey1 = elementwise_min(ctx, b1[1], b2[1]);
    ggml_tensor * ex2 = elementwise_max(ctx, b1[2], b2[2]), * ey2 = elementwise_max(ctx, b1[3], b2[3]);
    ggml_tensor * ew = ggml_relu(ctx, ggml_sub(ctx, ex2, ex1));
    ggml_tensor * eh = ggml_relu(ctx, ggml_sub(ctx, ey2, ey1));
    ggml_tensor * earea = ggml_scale_bias(ctx, ggml_mul(ctx, ew, eh), 1.0f, 1e-7f);

    ggml_tensor * giou = ggml_sub(ctx, iou, ggml_div(ctx, ggml_sub(ctx, earea, uni), earea));
    ggml_tensor * loss_giou_elem = ggml_scale_bias(ctx, giou, -1.0f, 1.0f); // 1 - giou
    ggml_tensor * loss_giou = ggml_scale(ctx, ggml_sum(ctx, loss_giou_elem), 1.0f / num_matched);
    total = ggml_add(ctx, total, ggml_scale(ctx, loss_giou, giou_coef));

    return total;
}
