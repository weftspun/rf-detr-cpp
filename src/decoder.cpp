#include "decoder.h"
#include "deform_attn.h"

#include <cmath>

std::vector<float> output_proposals_data(int gw, int gh) {
    std::vector<float> out((size_t) gw * gh * 4);
    for (int row = 0; row < gh; row++) {
        for (int col = 0; col < gw; col++) {
            float * p = &out[((size_t) row * gw + col) * 4];
            p[0] = (col + 0.5f) / gw; // cx
            p[1] = (row + 0.5f) / gh; // cy
            p[2] = 0.05f;             // w
            p[3] = 0.05f;             // h
        }
    }
    return out;
}

// (cx,cy,w,h)-reparam box decode: cxcy = delta[:2]*base_wh + base_cxcy,
// wh = exp(delta[2:])*base_wh. delta/base are (4,T,N).
static ggml_tensor * bbox_reparam_decode(ggml_context * ctx, ggml_tensor * delta, ggml_tensor * base) {
    ggml_tensor * base_cxcy = ggml_cont(ctx, ggml_view_3d(ctx, base, 2, base->ne[1], base->ne[2], base->nb[1], base->nb[2], 0));
    ggml_tensor * base_wh   = ggml_cont(ctx, ggml_view_3d(ctx, base, 2, base->ne[1], base->ne[2], base->nb[1], base->nb[2], 2 * sizeof(float)));
    ggml_tensor * delta_cxcy = ggml_cont(ctx, ggml_view_3d(ctx, delta, 2, delta->ne[1], delta->ne[2], delta->nb[1], delta->nb[2], 0));
    ggml_tensor * delta_wh   = ggml_cont(ctx, ggml_view_3d(ctx, delta, 2, delta->ne[1], delta->ne[2], delta->nb[1], delta->nb[2], 2 * sizeof(float)));

    ggml_tensor * cxcy = ggml_add(ctx, ggml_mul(ctx, delta_cxcy, base_wh), base_cxcy);
    ggml_tensor * wh   = ggml_mul(ctx, ggml_exp(ctx, delta_wh), base_wh);
    return ggml_concat(ctx, cxcy, wh, 0);
}

// gen_sineembed_for_position, one coordinate: v is (1,n_query,1); sine_scale
// is the GGUF constant (1,dim) = 2*pi/dim_t. Returns (dim,n_query,1) with
// interleaved sin(raw[2k])/cos(raw[2k+1]).
static ggml_tensor * sine_embed_1coord(ggml_context * ctx, ggml_tensor * v, ggml_tensor * sine_scale, int dim) {
    const int64_t n_query = v->ne[1];
    ggml_tensor * v2 = ggml_reshape_2d(ctx, ggml_cont(ctx, v), 1, n_query);              // (1,n_query)
    ggml_tensor * raw = ggml_mul_mat(ctx, sine_scale, v2);                               // (dim,n_query)

    ggml_tensor * pairs = ggml_reshape_3d(ctx, raw, 2, dim / 2, n_query);                // (2,dim/2,n_query)
    ggml_tensor * even = ggml_cont(ctx, ggml_view_3d(ctx, pairs, 1, dim / 2, n_query, pairs->nb[1], pairs->nb[2], 0));
    ggml_tensor * odd  = ggml_cont(ctx, ggml_view_3d(ctx, pairs, 1, dim / 2, n_query, pairs->nb[1], pairs->nb[2], sizeof(float)));
    ggml_tensor * s = ggml_sin(ctx, even);
    ggml_tensor * c = ggml_cos(ctx, odd);
    ggml_tensor * interleaved = ggml_concat(ctx, s, c, 0); // (2,dim/2,n_query)
    return ggml_reshape_3d(ctx, interleaved, dim, n_query, 1);
}

DecoderOutput rfdetr_decoder(Model & m, ggml_tensor * memory, const DecoderParams & p,
                             ggml_tensor * topk_idx_override) {
    ggml_context * ctx = m.ctx_g;
    const int gwh = p.gw * p.gh;
    const int hd = p.hidden_dim;

    ggml_tensor * proposals = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, gwh, 1);
    ggml_set_name(proposals, "output_proposals");
    ggml_set_input(proposals);

    // two-stage encoder proposals (group 0 only; group_detr is a no-op at inference)
    ggml_tensor * output_memory = layer_norm_affine(m, linear(m, memory, "enc_output.0"), "enc_output_norm.0");
    ggml_tensor * enc_class = linear(m, output_memory, "enc_out_class_embed.0");     // (num_classes, gwh, 1)
    ggml_tensor * enc_delta = mlp(m, output_memory, "enc_out_bbox_embed.0", 3);      // (4, gwh, 1)
    ggml_tensor * enc_boxes = bbox_reparam_decode(ctx, enc_delta, proposals);        // (4, gwh, 1)

    // top-k by per-location max class score (no reduce-max-along-ne0 op --
    // max-pool with a kernel spanning the whole class axis instead)
    ggml_tensor * topk_idx;
    if (topk_idx_override) {
        topk_idx = topk_idx_override;
    } else {
        ggml_tensor * score = ggml_pool_1d(ctx, enc_class, GGML_OP_POOL_MAX, p.num_classes, p.num_classes, 0); // (1,gwh,1)
        ggml_tensor * score1d = ggml_reshape_1d(ctx, ggml_cont(ctx, score), gwh);
        topk_idx = ggml_reshape_1d(ctx, ggml_top_k(ctx, score1d, p.num_queries), p.num_queries); // I32 (num_queries)
    }

    ggml_tensor * ts_boxes = ggml_get_rows(ctx, ggml_reshape_2d(ctx, enc_boxes, 4, gwh), topk_idx); // (4,num_queries)
    ts_boxes = ggml_reshape_3d(ctx, ts_boxes, 4, p.num_queries, 1);

    // learned static content queries + per-query reference-point delta
    ggml_tensor * query_feat_w = m.get("query_feat.weight");     // (hidden_dim, num_queries*group_detr)
    ggml_tensor * refpt_w      = m.get("refpoint_embed.weight"); // (4, num_queries*group_detr)
    ggml_tensor * tgt = ggml_cont(ctx, ggml_view_2d(ctx, query_feat_w, hd, p.num_queries, query_feat_w->nb[1], 0));
    tgt = ggml_reshape_3d(ctx, tgt, hd, p.num_queries, 1);
    ggml_tensor * subset = ggml_cont(ctx, ggml_view_2d(ctx, refpt_w, 4, p.num_queries, refpt_w->nb[1], 0));
    subset = ggml_reshape_3d(ctx, subset, 4, p.num_queries, 1);

    ggml_tensor * ref_points = bbox_reparam_decode(ctx, subset, ts_boxes); // (4, num_queries, 1) -- fixed for every layer

    // query positional embedding (once, lite_refpoint_refine)
    ggml_tensor * sine_scale = m.get("decoder.sine_scale"); // (1,128) constant, GGUF-baked: 2*pi/dim_t
    auto coord = [&](int off) {
        return ggml_cont(ctx, ggml_view_3d(ctx, ref_points, 1, p.num_queries, 1, ref_points->nb[1], ref_points->nb[2], (size_t) off * sizeof(float)));
    };
    ggml_tensor * pos_y = sine_embed_1coord(ctx, coord(1), sine_scale, 128);
    ggml_tensor * pos_x = sine_embed_1coord(ctx, coord(0), sine_scale, 128);
    ggml_tensor * pos_w = sine_embed_1coord(ctx, coord(2), sine_scale, 128);
    ggml_tensor * pos_h = sine_embed_1coord(ctx, coord(3), sine_scale, 128);
    ggml_tensor * sine = ggml_concat(ctx, ggml_concat(ctx, pos_y, pos_x, 0), ggml_concat(ctx, pos_w, pos_h, 0), 0); // (512,nq,1)
    ggml_tensor * query_pos = mlp(m, sine, "decoder.ref_point_head", 2); // (256,nq,1), ReLU between layers 0/1

    // decoder layers
    ggml_tensor * tgt_cur = tgt;
    for (int l = 0; l < p.dec_layers; l++) {
        const std::string pre = "decoder.layers." + std::to_string(l) + ".";

        // Self-attention with v computed from tgt_cur (not tgt_cur+query_pos) --
        // ops.h's generic self_attn() applies q/k/v to the same input, which
        // doesn't fit nn.MultiheadAttention(query=tgt+pos, key=tgt+pos, value=tgt)
        // here, so it's inlined instead.
        ggml_tensor * q_in = ggml_add(ctx, tgt_cur, query_pos);
        const int64_t T = tgt_cur->ne[1];
        ggml_tensor * qv = linear(m, q_in, pre + "self_attn.q_proj");
        ggml_tensor * kv = linear(m, q_in, pre + "self_attn.k_proj");
        ggml_tensor * vv = linear(m, tgt_cur, pre + "self_attn.v_proj");
        const int64_t hdh = hd / p.sa_nheads;
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, qv, hdh, p.sa_nheads, T, 1), 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, kv, hdh, p.sa_nheads, T, 1), 0, 2, 1, 3));
        ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, vv, hdh, p.sa_nheads, T, 1), 1, 2, 0, 3));
        ggml_tensor * kq = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, kh, qh), 1.0f / sqrtf((float) hdh)));
        ggml_tensor * kqv = ggml_mul_mat(ctx, vh, kq);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_3d(ctx, kqv, hd, T, 1);
        ggml_tensor * self_out = linear(m, kqv, pre + "self_attn.out_proj");

        tgt_cur = layer_norm_affine(m, ggml_add(ctx, tgt_cur, self_out), pre + "norm1");

        ggml_tensor * cross_q = ggml_add(ctx, tgt_cur, query_pos);
        ggml_tensor * cross_out = ms_deform_attn(m, cross_q, memory, ref_points, pre + "cross_attn",
                                                 p.ca_nheads, p.dec_n_points, p.gw, p.gh);
        tgt_cur = layer_norm_affine(m, ggml_add(ctx, tgt_cur, cross_out), pre + "norm2");

        ggml_tensor * ff = linear(m, tgt_cur, pre + "linear1");
        ff = ggml_relu(ctx, ff);
        ff = linear(m, ff, pre + "linear2");
        tgt_cur = layer_norm_affine(m, ggml_add(ctx, tgt_cur, ff), pre + "norm3");
    }

    ggml_tensor * final_hs = layer_norm_affine(m, tgt_cur, "decoder.norm");

    DecoderOutput out;
    out.pred_logits = linear(m, final_hs, "class_embed");
    ggml_tensor * bbox_delta = mlp(m, final_hs, "bbox_embed", 3);
    out.pred_boxes = bbox_reparam_decode(ctx, bbox_delta, ref_points);
    out.output_proposals = proposals;
    return out;
}
