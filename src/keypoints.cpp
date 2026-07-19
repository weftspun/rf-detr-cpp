#include "keypoints.h"
#include "deform_attn.h"

#include <cmath>

ggml_tensor * conditional_query_initializer(Model & m, ggml_tensor * cond, const std::string & pre,
                                            int num_kp, int hidden_dim) {
    ggml_context * ctx = m.ctx_g;
    const int64_t nq = cond->ne[2];

    ggml_tensor * queries = ggml_reshape_3d(ctx, m.get(pre + ".queries"), hidden_dim, num_kp, 1); // (C,K,1)
    ggml_tensor * normed = ggml_norm(ctx, queries, 1e-5f); // elementwise_affine=False: no weight/bias

    ggml_tensor * h = linear(m, cond, pre + ".adaLN_modulation.0"); // (C,1,nq)
    h = ggml_gelu_erf(ctx, h);
    h = linear(m, h, pre + ".adaLN_modulation.2"); // (3C,1,nq)

    auto slice = [&](int off) {
        return ggml_cont(ctx, ggml_view_3d(ctx, h, hidden_dim, 1, nq, h->nb[1], h->nb[2], (size_t) off * hidden_dim * sizeof(float)));
    };
    ggml_tensor * scale = slice(0), * shift = slice(1), * gate = slice(2); // each (C,1,nq)

    ggml_tensor * normed_full = ggml_repeat_4d(ctx, normed, hidden_dim, num_kp, nq, 1);
    ggml_tensor * scale_full  = ggml_repeat_4d(ctx, scale,  hidden_dim, num_kp, nq, 1);
    ggml_tensor * shift_full  = ggml_repeat_4d(ctx, shift,  hidden_dim, num_kp, nq, 1);
    ggml_tensor * gate_full   = ggml_repeat_4d(ctx, gate,   hidden_dim, num_kp, nq, 1);
    ggml_tensor * queries_full = ggml_repeat_4d(ctx, queries, hidden_dim, num_kp, nq, 1);

    ggml_tensor * modulated = ggml_add(ctx, ggml_mul(ctx, ggml_scale_bias(ctx, scale_full, 1.0f, 1.0f), normed_full), shift_full);
    modulated = linear(m, ggml_reshape_3d(ctx, modulated, hidden_dim, num_kp, nq), pre + ".out_proj");
    modulated = ggml_mul(ctx, modulated, gate_full);
    return ggml_add(ctx, modulated, queries_full); // (C, num_kp, nq)
}

// Ordinary MHA over T-token groups, batch N, with q=k=x+pos, v=x (pos not
// added to v) -- same pattern as decoder.cpp's main self-attn, generalized
// to arbitrary (T,N). weight prefix "<pre>.{q,k,v}_proj", "<pre>.out_proj".
static ggml_tensor * grouped_self_attn(Model & m, ggml_tensor * x, ggml_tensor * pos, const std::string & pre, int n_head) {
    ggml_context * ctx = m.ctx_g;
    const int64_t C = x->ne[0], T = x->ne[1], N = x->ne[2];
    const int64_t hd = C / n_head;

    ggml_tensor * qk_in = ggml_add(ctx, x, pos);
    ggml_tensor * q = linear(m, qk_in, pre + ".q_proj");
    ggml_tensor * k = linear(m, qk_in, pre + ".k_proj");
    ggml_tensor * v = linear(m, x, pre + ".v_proj");

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, n_head, T, N), 0, 2, 1, 3)); // (hd,T,H,N)
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, n_head, T, N), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, n_head, T, N), 1, 2, 0, 3)); // (T,hd,H,N)

    ggml_tensor * kq = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / sqrtf((float) hd)));
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);              // (hd,T,H,N)
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // (hd,H,T,N)
    kqv = ggml_reshape_3d(ctx, kqv, C, T, N);
    return linear(m, kqv, pre + ".out_proj");
}

KeypointLayerOutput keypoint_decoder_layer(Model & m, ggml_tensor * tgt, ggml_tensor * query_pos,
                                           ggml_tensor * keypoint_tgt, ggml_tensor * keypoint_pos,
                                           ggml_tensor * ref_points, ggml_tensor * kp_memory,
                                           const std::string & pre, const KeypointParams & kp,
                                           int gw, int gh) {
    ggml_context * ctx = m.ctx_g;
    const int64_t C = tgt->ne[0], nq = tgt->ne[1];
    const int K = kp.num_kp;

    // ---- 1. keypoint-instance self-attention: [query_token, K keypoint tokens] per instance ----
    ggml_tensor * tgt3 = ggml_reshape_3d(ctx, tgt, C, 1, nq);         // (C,1,nq)
    ggml_tensor * zero_pos = ggml_scale(ctx, tgt3, 0.0f);             // query slot gets zero pos embed
    ggml_tensor * kp_pos_full = ggml_repeat_4d(ctx, keypoint_pos, C, K, nq, 1); // (C,K,nq)

    ggml_tensor * combined_feat = ggml_concat(ctx, tgt3, keypoint_tgt, 1);   // (C,1+K,nq)
    ggml_tensor * combined_pos  = ggml_concat(ctx, zero_pos, kp_pos_full, 1); // (C,1+K,nq)

    ggml_tensor * combined_out = grouped_self_attn(m, combined_feat, combined_pos, pre + "kp_inst_self_attn", kp.sa_nheads);

    ggml_tensor * tgt2 = ggml_cont(ctx, ggml_view_3d(ctx, combined_out, C, 1, nq, combined_out->nb[1], combined_out->nb[2], 0));
    ggml_tensor * keypoint_tgt2 = ggml_cont(ctx, ggml_view_3d(ctx, combined_out, C, K, nq, combined_out->nb[1], combined_out->nb[2], combined_out->nb[1]));

    ggml_tensor * layer_scale = m.get(pre + "instance_kp_layer_scale"); // (1,)
    ggml_tensor * tgt_delta = ggml_mul(ctx, ggml_reshape_2d(ctx, tgt2, C, nq), layer_scale);
    ggml_tensor * tgt_new = ggml_add(ctx, tgt, tgt_delta);
    tgt_new = layer_norm_affine(m, tgt_new, pre + "kp_inst_norm");

    ggml_tensor * keypoint_tgt_new = ggml_add(ctx, keypoint_tgt, keypoint_tgt2);
    keypoint_tgt_new = layer_norm_affine(m, keypoint_tgt_new, pre + "kp_norm");

    // ---- 2. inter-instance keypoint attention: skipped (inter_instance_kp_attn=False) ----

    // ---- 3. keypoint-specific deformable cross-attention ----
    // query = keypoint_tgt + parent query_pos (broadcast per keypoint), flattened to (C, K*nq, 1)
    ggml_tensor * qpos3 = ggml_reshape_4d(ctx, query_pos, C, 1, nq, 1);
    ggml_tensor * qpos_full = ggml_repeat_4d(ctx, qpos3, C, K, nq, 1);
    ggml_tensor * kp_query = ggml_add(ctx, keypoint_tgt_new, qpos_full);
    kp_query = ggml_reshape_3d(ctx, ggml_cont(ctx, kp_query), C, K * nq, 1);

    // ref box repeated per keypoint: (4,nq,1) -> (4,K,nq,1) -> (4,K*nq,1)
    ggml_tensor * ref4 = ggml_reshape_4d(ctx, ref_points, 4, 1, nq, 1);
    ggml_tensor * ref_full = ggml_repeat_4d(ctx, ref4, 4, K, nq, 1);
    ref_full = ggml_reshape_3d(ctx, ggml_cont(ctx, ref_full), 4, K * nq, 1);

    ggml_tensor * kp_cross_out = ms_deform_attn(m, kp_query, kp_memory, ref_full, pre + "kp_cross_attn",
                                                kp.ca_nheads, kp.dec_n_points, gw, gh);
    kp_cross_out = ggml_reshape_3d(ctx, kp_cross_out, C, K, nq);

    keypoint_tgt_new = ggml_add(ctx, keypoint_tgt_new, kp_cross_out);
    keypoint_tgt_new = layer_norm_affine(m, keypoint_tgt_new, pre + "kp_cross_attn_norm");

    // ---- 4. keypoint-specific FFN ----
    ggml_tensor * ff = linear(m, keypoint_tgt_new, pre + "kp_linear1");
    ff = ggml_relu(ctx, ff);
    ff = linear(m, ff, pre + "kp_linear3");
    keypoint_tgt_new = ggml_add(ctx, keypoint_tgt_new, ff);
    keypoint_tgt_new = layer_norm_affine(m, keypoint_tgt_new, pre + "kp_norm5");

    return { tgt_new, keypoint_tgt_new };
}

KeypointHeadOutput keypoint_final_decode(Model & m, ggml_tensor * keypoint_tgt_last,
                                         ggml_tensor * ref_points, int num_kp,
                                         int num_keypoint_classes, int active_class_idx) {
    ggml_context * ctx = m.ctx_g;
    const int64_t nq = keypoint_tgt_last->ne[2];

    ggml_tensor * delta = mlp(m, keypoint_tgt_last, "keypoint_embed", 3); // (8, num_kp, nq)

    ggml_tensor * ref4 = ggml_reshape_4d(ctx, ref_points, 4, 1, nq, 1);
    ggml_tensor * ref_full = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_repeat_4d(ctx, ref4, 4, num_kp, nq, 1)), 4, num_kp, nq);
    ggml_tensor * ref_xy = ggml_cont(ctx, ggml_view_3d(ctx, ref_full, 2, num_kp, nq, ref_full->nb[1], ref_full->nb[2], 0));
    ggml_tensor * ref_wh = ggml_cont(ctx, ggml_view_3d(ctx, ref_full, 2, num_kp, nq, ref_full->nb[1], ref_full->nb[2], 2 * sizeof(float)));

    ggml_tensor * delta_xy = ggml_cont(ctx, ggml_view_3d(ctx, delta, 2, num_kp, nq, delta->nb[1], delta->nb[2], 0));
    ggml_tensor * delta_other = ggml_cont(ctx, ggml_view_3d(ctx, delta, 6, num_kp, nq, delta->nb[1], delta->nb[2], 2 * sizeof(float)));

    ggml_tensor * xy = ggml_add(ctx, ggml_mul(ctx, delta_xy, ref_wh), ref_xy);
    ggml_tensor * pred_keypoints_compact = ggml_concat(ctx, xy, delta_other, 0); // (8, num_kp, nq)

    // class-logit boost: slot 7 (last of the 6 "other" channels) summed over keypoints.
    // Only the active class' real keypoints contribute -- every other declared
    // class' keypoints are zero-padded (see _format_keypoint_output below), so
    // their class-boost is provably 0 and adding it is a no-op; this port
    // only computes the active class' (nonzero) boost.
    ggml_tensor * class_logit = ggml_cont(ctx, ggml_view_3d(ctx, delta_other, 1, num_kp, nq, delta_other->nb[1], delta_other->nb[2], 5 * sizeof(float)));
    class_logit = ggml_cont(ctx, ggml_permute(ctx, class_logit, 1, 0, 2, 3)); // (num_kp,1,nq)
    ggml_tensor * boost = ggml_sum_rows(ctx, ggml_reshape_2d(ctx, class_logit, num_kp, nq)); // (1,nq)
    boost = ggml_reshape_3d(ctx, boost, 1, nq, 1);

    // _format_keypoint_output: pad "compact" (num_kp real keypoints) to
    // "class-padded" (num_keypoint_classes*num_kp total slots), the active
    // class' keypoints at [active_class_idx*num_kp, (active_class_idx+1)*num_kp),
    // every other class' slots zero. No-op when num_keypoint_classes==1.
    ggml_tensor * pred_keypoints = pred_keypoints_compact;
    if (num_keypoint_classes > 1) {
        ggml_tensor * zeros_before = active_class_idx > 0
            ? ggml_scale(ctx, ggml_repeat_4d(ctx, pred_keypoints_compact, 8, num_kp * active_class_idx, nq, 1), 0.0f)
            : nullptr;
        int trailing = num_keypoint_classes - active_class_idx - 1;
        ggml_tensor * zeros_after = trailing > 0
            ? ggml_scale(ctx, ggml_repeat_4d(ctx, pred_keypoints_compact, 8, num_kp * trailing, nq, 1), 0.0f)
            : nullptr;
        if (zeros_before) pred_keypoints = ggml_concat(ctx, zeros_before, pred_keypoints, 1);
        if (zeros_after) pred_keypoints = ggml_concat(ctx, pred_keypoints, zeros_after, 1);
    }

    return { ggml_reshape_4d(ctx, pred_keypoints, 8, num_keypoint_classes * num_kp, nq, 1), boost };
}
