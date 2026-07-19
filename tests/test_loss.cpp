// Isolated validation of src/loss.cpp (hungarian_match + detection_loss)
// against the real upstream HungarianMatcher + SetCriterion
// (gen_reference/gen_reference_loss.py, ia_bce_loss=False path) on
// synthetic pred_logits/pred_boxes/targets. Checks BOTH the matched
// (query,target) index pairs (must match exactly -- Hungarian assignment
// is discrete, no tolerance) and the final scalar loss value.
#include "loss.h"
#include "test_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

int main(int argc, char ** argv) {
    const char * ref_path = argc > 1 ? argv[1] : "gen_reference/reference_loss.bin";

    std::vector<NpyArray> ref;
    if (!read_ref(ref_path, ref, 7)) {
        fprintf(stderr, "failed to read %s (run gen_reference/gen_reference_loss.py first)\n", ref_path);
        return 1;
    }
    const NpyArray & logits_ref = ref[0], & boxes_ref = ref[1];
    const NpyArray & tgt_labels_ref = ref[2], & tgt_boxes_ref = ref[3];
    const NpyArray & query_idx_ref = ref[4], & tgt_idx_ref = ref[5];
    const NpyArray & total_loss_ref = ref[6];

    const int num_queries = (int) logits_ref.shape[0];
    const int num_classes = (int) logits_ref.shape[1];
    const int num_targets = (int) tgt_labels_ref.shape[0];

    DetectionTarget tgt;
    tgt.labels.resize(num_targets);
    for (int i = 0; i < num_targets; i++) {
        tgt.labels[i] = (int32_t) tgt_labels_ref.data[i];
    }
    tgt.boxes = tgt_boxes_ref.data;

    // --- 1. Hungarian matching: exact SET of (query,target) pairs required
    // (order differs from scipy's own convention -- scipy's row_ind comes
    // back query-sorted, hungarian_match here emits target-order -- same
    // assignment, just listed differently, so compare as an unordered set
    // of pairs rather than positionally).
    MatchResult match = hungarian_match(logits_ref.data, boxes_ref.data, num_queries, num_classes, tgt);
    printf("matched %zu pairs\n", match.query_idx.size());
    bool idx_ok = match.query_idx.size() == (size_t) num_targets;
    std::vector<int32_t> expect_by_tgt(num_targets, -1);
    for (size_t k = 0; k < query_idx_ref.data.size(); k++) {
        expect_by_tgt[(int32_t) tgt_idx_ref.data[k]] = (int32_t) query_idx_ref.data[k];
    }
    for (size_t k = 0; idx_ok && k < match.query_idx.size(); k++) {
        int32_t t = match.tgt_idx[k];
        if (match.query_idx[k] != expect_by_tgt[t]) {
            fprintf(stderr, "target %d matched query %d, expected query %d\n",
                    t, match.query_idx[k], expect_by_tgt[t]);
            idx_ok = false;
        }
    }
    printf("hungarian_match: %s\n", idx_ok ? "VALIDATION PASS" : "VALIDATION FAIL");

    // --- 2. Loss graph: scalar value match (loose tolerance -- log/exp-heavy) ---
    Model m;
    ggml_init_params wip = { 1024, nullptr, /*no_alloc*/ false };
    ggml_context * wctx = ggml_init(wip);
    m.ctx_w.push_back(wctx);

    init_graph_ctx(m, 4096);
    ggml_tensor * pred_logits = ggml_new_tensor_3d(m.ctx_g, GGML_TYPE_F32, num_classes, num_queries, 1);
    ggml_tensor * pred_boxes = ggml_new_tensor_3d(m.ctx_g, GGML_TYPE_F32, 4, num_queries, 1);
    ggml_set_input(pred_logits);
    ggml_set_input(pred_boxes);

    ggml_tensor * loss = detection_loss(m, pred_logits, pred_boxes, tgt, match, num_queries, num_classes);

    bool ok = compute_cpu(m, loss, 4096, [&] {
        ggml_backend_tensor_set(pred_logits, logits_ref.data.data(), 0, logits_ref.data.size() * sizeof(float));
        ggml_backend_tensor_set(pred_boxes, boxes_ref.data.data(), 0, boxes_ref.data.size() * sizeof(float));
        ggml_tensor * onehot = ggml_get_tensor(m.ctx_g, "loss_targets_onehot");
        std::vector<float> onehot_data = targets_onehot_data(match, tgt, num_queries, num_classes);
        ggml_backend_tensor_set(onehot, onehot_data.data(), 0, onehot_data.size() * sizeof(float));

        ggml_tensor * midx = ggml_get_tensor(m.ctx_g, "loss_matched_query_idx");
        std::vector<int32_t> midx_data = matched_query_idx_data(match);
        ggml_backend_tensor_set(midx, midx_data.data(), 0, midx_data.size() * sizeof(int32_t));

        ggml_tensor * mtgt = ggml_get_tensor(m.ctx_g, "loss_matched_tgt_boxes");
        std::vector<float> mtgt_data = matched_tgt_boxes_data(match, tgt);
        ggml_backend_tensor_set(mtgt, mtgt_data.data(), 0, mtgt_data.size() * sizeof(float));
    });
    if (!ok) {
        return 1;
    }
    int rc = compare_ref(loss, total_loss_ref, 1e-2) || !idx_ok;

    // --- 3. Backward pass: confirm detection_loss's graph is genuinely
    // trainable (ggml_build_backward_expand doesn't abort) and the
    // gradient w.r.t. pred_logits/pred_boxes matches a finite-difference
    // check -- the actual point of this function, unlike every earlier
    // milestone which only needed forward correctness.
    {
        Model m2;
        ggml_init_params wip2 = { 1024, nullptr, false };
        ggml_context * wctx2 = ggml_init(wip2);
        m2.ctx_w.push_back(wctx2);

        size_t meta = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, true);
        ggml_init_params ip2 = { meta, nullptr, true };
        m2.ctx_g = ggml_init(ip2);

        ggml_tensor * pl = ggml_new_tensor_3d(m2.ctx_g, GGML_TYPE_F32, num_classes, num_queries, 1);
        ggml_tensor * pb = ggml_new_tensor_3d(m2.ctx_g, GGML_TYPE_F32, 4, num_queries, 1);
        ggml_set_input(pl); ggml_set_param(pl);
        ggml_set_input(pb); ggml_set_param(pb);

        ggml_tensor * loss2 = detection_loss(m2, pl, pb, tgt, match, num_queries, num_classes);
        ggml_set_loss(loss2);

        ggml_cgraph * gf = ggml_new_graph_custom(m2.ctx_g, 4096, true);
        ggml_build_forward_expand(gf, loss2);
        ggml_build_backward_expand(m2.ctx_g, gf, nullptr);

        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }

        auto set_inputs = [&]() {
            ggml_backend_tensor_set(pl, logits_ref.data.data(), 0, logits_ref.data.size() * sizeof(float));
            ggml_backend_tensor_set(pb, boxes_ref.data.data(), 0, boxes_ref.data.size() * sizeof(float));
            ggml_tensor * onehot = ggml_get_tensor(m2.ctx_g, "loss_targets_onehot");
            std::vector<float> onehot_data = targets_onehot_data(match, tgt, num_queries, num_classes);
            ggml_backend_tensor_set(onehot, onehot_data.data(), 0, onehot_data.size() * sizeof(float));
            ggml_tensor * midx = ggml_get_tensor(m2.ctx_g, "loss_matched_query_idx");
            std::vector<int32_t> midx_data = matched_query_idx_data(match);
            ggml_backend_tensor_set(midx, midx_data.data(), 0, midx_data.size() * sizeof(int32_t));
            ggml_tensor * mtgt = ggml_get_tensor(m2.ctx_g, "loss_matched_tgt_boxes");
            std::vector<float> mtgt_data = matched_tgt_boxes_data(match, tgt);
            ggml_backend_tensor_set(mtgt, mtgt_data.data(), 0, mtgt_data.size() * sizeof(float));
        };
        set_inputs();
        ggml_graph_reset(gf);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "BACKWARD PASS FAILED/ABORTED\n");
            return 1;
        }
        printf("backward pass completed without aborting\n");

        ggml_tensor * pl_grad = ggml_graph_get_grad(gf, pl);
        ggml_tensor * pb_grad = ggml_graph_get_grad(gf, pb);
        if (!pl_grad || !pb_grad) {
            fprintf(stderr, "no gradient computed\n");
            return 1;
        }
        std::vector<float> gl(logits_ref.data.size()), gb(boxes_ref.data.size());
        ggml_backend_tensor_get(pl_grad, gl.data(), 0, gl.size() * sizeof(float));
        ggml_backend_tensor_get(pb_grad, gb.data(), 0, gb.size() * sizeof(float));

        // Fresh forward-only graph PER CALL (context/graph/allocator all
        // rebuilt from scratch) -- reusing the single backward-enabled
        // graph across repeated ggml_backend_graph_compute calls (even
        // with ggml_graph_reset between them) was observed to silently
        // corrupt the result (a plain re-evaluation at the SAME
        // unperturbed point gave 67.9 instead of 23.6 the second time).
        // Root cause not tracked down -- possibly related to the ggml
        // graph-allocator buffer-reuse issue already found and documented
        // for the SegXLarge decoder graph (docs/decisions/0001-open-work.md).
        // Rebuilding fresh sidesteps it entirely for this finite-difference
        // check, which only needs the scalar loss VALUE, not a live
        // backward-capable graph.
        auto eval_loss = [&](std::vector<float> logits, std::vector<float> boxes) -> float {
            ggml_init_params ipf = { ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false), nullptr, true };
            ggml_context * ctxf = ggml_init(ipf);
            ggml_tensor * plf = ggml_new_tensor_3d(ctxf, GGML_TYPE_F32, num_classes, num_queries, 1);
            ggml_tensor * pbf = ggml_new_tensor_3d(ctxf, GGML_TYPE_F32, 4, num_queries, 1);
            ggml_set_input(plf);
            ggml_set_input(pbf);
            Model mf;
            mf.ctx_g = ctxf;
            ggml_tensor * lossf = detection_loss(mf, plf, pbf, tgt, match, num_queries, num_classes);

            ggml_backend_t backendf = ggml_backend_cpu_init();
            ggml_cgraph * gff = ggml_new_graph_custom(ctxf, 512, false);
            ggml_build_forward_expand(gff, lossf);
            ggml_gallocr_t allocf = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backendf));
            ggml_gallocr_alloc_graph(allocf, gff);

            ggml_backend_tensor_set(plf, logits.data(), 0, logits.size() * sizeof(float));
            ggml_backend_tensor_set(pbf, boxes.data(), 0, boxes.size() * sizeof(float));
            ggml_tensor * onehotf = ggml_get_tensor(ctxf, "loss_targets_onehot");
            std::vector<float> onehot_data = targets_onehot_data(match, tgt, num_queries, num_classes);
            ggml_backend_tensor_set(onehotf, onehot_data.data(), 0, onehot_data.size() * sizeof(float));
            ggml_tensor * midxf = ggml_get_tensor(ctxf, "loss_matched_query_idx");
            std::vector<int32_t> midx_data = matched_query_idx_data(match);
            ggml_backend_tensor_set(midxf, midx_data.data(), 0, midx_data.size() * sizeof(int32_t));
            ggml_tensor * mtgtf = ggml_get_tensor(ctxf, "loss_matched_tgt_boxes");
            std::vector<float> mtgt_data = matched_tgt_boxes_data(match, tgt);
            ggml_backend_tensor_set(mtgtf, mtgt_data.data(), 0, mtgt_data.size() * sizeof(float));

            ggml_backend_graph_compute(backendf, gff);
            float v;
            ggml_backend_tensor_get(lossf, &v, 0, sizeof(float));

            ggml_gallocr_free(allocf);
            ggml_backend_free(backendf);
            ggml_free(ctxf);
            return v;
        };
        printf("sanity re-eval at unperturbed point: %.6f (expect ~23.6096)\n",
               eval_loss(logits_ref.data, boxes_ref.data));

        const float h = 1e-3f;
        bool grad_ok = true;
        // pass if EITHER the relative error is small OR the absolute
        // difference is tiny -- a pure relative check is noisy for
        // near-zero gradients (e.g. a well-classified negative example
        // whose focal modulator has already suppressed it toward zero),
        // where a numerically-fine 0.0008 absolute difference can still
        // read as a large ratio.
        auto check = [&](const char * label, int idx, float analytic, float numeric) {
            float abs_err = std::fabs(numeric - analytic);
            float rel = abs_err / (std::fabs(numeric) + 1e-3f);
            printf("d(loss)/d(%s[%d]): analytic=%.6f numeric=%.6f abs_err=%.6f rel_err=%.6f\n",
                   label, idx, analytic, numeric, abs_err, rel);
            if (rel > 5e-2f && abs_err > 2e-3f) {
                grad_ok = false;
            }
        };

        // a few logits indices + a few box indices (matched-query coords,
        // where both loss_ce and loss_bbox/giou gradients are nonzero)
        int box_check_idx = match.query_idx.empty() ? 0 : match.query_idx[0] * 4;
        int checks[] = { 0, num_classes * 3 + 1, num_classes * num_queries - 1 };
        for (int idx : checks) {
            std::vector<float> lp = logits_ref.data, lm = logits_ref.data;
            lp[idx] += h; lm[idx] -= h;
            float vp = eval_loss(lp, boxes_ref.data), vm = eval_loss(lm, boxes_ref.data);
            check("logit", idx, gl[idx], (vp - vm) / (2 * h));
        }
        {
            std::vector<float> bp = boxes_ref.data, bm = boxes_ref.data;
            bp[box_check_idx] += h; bm[box_check_idx] -= h;
            float vp = eval_loss(logits_ref.data, bp), vm = eval_loss(logits_ref.data, bm);
            check("box", box_check_idx, gb[box_check_idx], (vp - vm) / (2 * h));
        }
        ggml_gallocr_free(alloc);
        ggml_backend_free(backend);
        if (!grad_ok) {
            fprintf(stderr, "GRADIENT MISMATCH\n");
            rc = 1;
        }
    }

    return rc;
}
