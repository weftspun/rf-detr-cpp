#include "deform_attn.h"

#include <vector>

// 1 if val is in [0, size-1], else 0 (float mask); val is a float tensor of
// arbitrary shape, evaluated elementwise. step(x) = x>0 ? 1 : 0, so the
// +0.5 offsets turn it into >=0 / <=size-1 with margin for exact integers.
static ggml_tensor * in_bounds_mask(ggml_context * ctx, ggml_tensor * val, int size) {
    ggml_tensor * ge0 = ggml_step(ctx, ggml_scale_bias(ctx, val, 1.0f, 0.5f));
    ggml_tensor * le  = ggml_step(ctx, ggml_scale_bias(ctx, val, -1.0f, (float) (size - 1) + 0.5f));
    return ggml_mul(ctx, ge0, le);
}

ggml_tensor * ms_deform_attn(Model & m, ggml_tensor * query, ggml_tensor * value_input,
                             ggml_tensor * ref_points, const std::string & pre,
                             int n_heads, int n_points, int gw, int gh) {
    ggml_context * ctx = m.ctx_g;
    const int64_t d_model  = query->ne[0];
    const int64_t n_query  = query->ne[1];
    const int64_t head_dim = d_model / n_heads;

    ggml_tensor * value = ggml_cont(ctx, linear(m, value_input, pre + ".value_proj")); // (d_model, gw*gh, 1)
    ggml_tensor * value4 = ggml_reshape_3d(ctx, value, head_dim, n_heads, gw * gh);     // (head_dim, n_heads, gw*gh)

    ggml_tensor * off_raw = ggml_cont(ctx, linear(m, query, pre + ".sampling_offsets")); // (n_heads*n_points*2, n_query, 1)
    ggml_tensor * off = ggml_reshape_4d(ctx, off_raw, 2, n_points, n_heads, n_query);     // xy fastest, then point, then head

    ggml_tensor * aw_raw = ggml_cont(ctx, linear(m, query, pre + ".attention_weights"));  // (n_heads*n_points, n_query, 1)
    ggml_tensor * aw = ggml_reshape_4d(ctx, aw_raw, n_points, n_heads, n_query, 1);
    aw = ggml_soft_max(ctx, aw); // softmax over n_points (num_levels=1, so this is the full level*point softmax)

    ggml_tensor * ref_cxcy = ggml_cont(ctx, ggml_view_2d(ctx, ref_points, 2, n_query, ref_points->nb[1], 0));
    ggml_tensor * ref_wh   = ggml_cont(ctx, ggml_view_2d(ctx, ref_points, 2, n_query, ref_points->nb[1], 2 * sizeof(float)));

    ggml_tensor * out = nullptr;
    for (int h = 0; h < n_heads; h++) {
        ggml_tensor * value_h = ggml_cont(ctx, ggml_view_2d(ctx, value4, head_dim, gw * gh,
                                                            value4->nb[2], (size_t) h * value4->nb[1])); // (head_dim, gw*gh)

        ggml_tensor * head_out = nullptr;
        for (int pt = 0; pt < n_points; pt++) {
            ggml_tensor * off_hp = ggml_cont(ctx, ggml_view_2d(ctx, off, 2, n_query, off->nb[3],
                                                (size_t) pt * off->nb[1] + (size_t) h * off->nb[2])); // (2,n_query)

            ggml_tensor * sloc = ggml_add(ctx, ggml_mul(ctx, ggml_scale(ctx, off_hp, 0.5f / n_points), ref_wh), ref_cxcy);
            ggml_tensor * sloc_x = ggml_cont(ctx, ggml_view_2d(ctx, sloc, 1, n_query, sloc->nb[1], 0));
            ggml_tensor * sloc_y = ggml_cont(ctx, ggml_view_2d(ctx, sloc, 1, n_query, sloc->nb[1], sizeof(float)));

            ggml_tensor * ix = ggml_scale_bias(ctx, sloc_x, (float) gw, -0.5f); // align_corners=False unnormalize
            ggml_tensor * iy = ggml_scale_bias(ctx, sloc_y, (float) gh, -0.5f);
            ggml_tensor * ix0 = ggml_floor(ctx, ix);
            ggml_tensor * iy0 = ggml_floor(ctx, iy);
            ggml_tensor * ix1 = ggml_scale_bias(ctx, ix0, 1.0f, 1.0f);
            ggml_tensor * iy1 = ggml_scale_bias(ctx, iy0, 1.0f, 1.0f);
            ggml_tensor * wx1 = ggml_sub(ctx, ix, ix0), * wy1 = ggml_sub(ctx, iy, iy0);
            ggml_tensor * wx0 = ggml_scale_bias(ctx, wx1, -1.0f, 1.0f), * wy0 = ggml_scale_bias(ctx, wy1, -1.0f, 1.0f);

            ggml_tensor * mx0 = in_bounds_mask(ctx, ix0, gw), * mx1 = in_bounds_mask(ctx, ix1, gw);
            ggml_tensor * my0 = in_bounds_mask(ctx, iy0, gh), * my1 = in_bounds_mask(ctx, iy1, gh);
            // ggml_clamp is in-place (aliases its input's buffer, see ggml_view_tensor
            // in ggml.c) -- clamp a ggml_cont'd copy so ix0/iy0/ix1/iy1's original
            // values (still read by wx1/wy1/in_bounds_mask above) aren't corrupted.
            ggml_tensor * cx0 = ggml_clamp(ctx, ggml_cont(ctx, ix0), 0.0f, (float) (gw - 1));
            ggml_tensor * cx1 = ggml_clamp(ctx, ggml_cont(ctx, ix1), 0.0f, (float) (gw - 1));
            ggml_tensor * cy0 = ggml_clamp(ctx, ggml_cont(ctx, iy0), 0.0f, (float) (gh - 1));
            ggml_tensor * cy1 = ggml_clamp(ctx, ggml_cont(ctx, iy1), 0.0f, (float) (gh - 1));

            struct Corner { ggml_tensor * cx; ggml_tensor * cy; ggml_tensor * wx; ggml_tensor * wy; ggml_tensor * mx; ggml_tensor * my; };
            Corner corners[4] = {
                { cx0, cy0, wx0, wy0, mx0, my0 },
                { cx1, cy0, wx1, wy0, mx1, my0 },
                { cx0, cy1, wx0, wy1, mx0, my1 },
                { cx1, cy1, wx1, wy1, mx1, my1 },
            };
            ggml_tensor * sample_sum = nullptr;
            for (const Corner & c : corners) {
                ggml_tensor * row_f = ggml_add(ctx, ggml_scale(ctx, c.cy, (float) gw), c.cx); // (1,n_query)
                ggml_tensor * row_i = ggml_cast(ctx, ggml_reshape_1d(ctx, row_f, n_query), GGML_TYPE_I32);
                ggml_tensor * gathered = ggml_get_rows(ctx, value_h, row_i); // (head_dim, n_query)
                ggml_tensor * weight = ggml_mul(ctx, ggml_mul(ctx, c.wx, c.wy), ggml_mul(ctx, c.mx, c.my)); // (1,n_query)
                ggml_tensor * contrib = ggml_mul(ctx, gathered, weight); // broadcast (1,n_query) over (head_dim,n_query)
                sample_sum = sample_sum ? ggml_add(ctx, sample_sum, contrib) : contrib;
            }

            ggml_tensor * aw_hp = ggml_cont(ctx, ggml_view_2d(ctx, aw, 1, n_query, aw->nb[2],
                                               (size_t) pt * aw->nb[0] + (size_t) h * aw->nb[1])); // (1,n_query)
            ggml_tensor * weighted = ggml_mul(ctx, sample_sum, aw_hp);
            head_out = head_out ? ggml_add(ctx, head_out, weighted) : weighted;
        }
        out = out ? ggml_concat(ctx, out, head_out, 0) : head_out; // concat heads along channel dim
    }

    out = ggml_reshape_3d(ctx, out, d_model, n_query, 1);
    return linear(m, out, pre + ".output_proj");
}

ggml_tensor * ms_deform_attn_multilevel(Model & m, ggml_tensor * query, ggml_tensor * value_input,
                                        ggml_tensor * ref_points, const std::string & pre,
                                        int n_heads, int n_points,
                                        const std::vector<std::pair<int, int>> & levels) {
    ggml_context * ctx = m.ctx_g;
    const int64_t d_model  = query->ne[0];
    const int64_t n_query  = query->ne[1];
    const int64_t head_dim = d_model / n_heads;
    const int n_levels = (int) levels.size();

    ggml_tensor * value = ggml_cont(ctx, linear(m, value_input, pre + ".value_proj")); // (d_model, total_tokens, 1)
    ggml_tensor * value4 = ggml_reshape_3d(ctx, value, head_dim, n_heads, value->ne[1]); // (head_dim, n_heads, total_tokens)

    // raw sampling_offsets/attention_weights are never reshaped past 2D here --
    // ggml caps tensors at 4 dims, and (xy, point, level, head, query) would need
    // 5; instead every (head,level,point) lookup below computes its own flat
    // byte offset into the row-major-contiguous (n_heads*n_levels*n_points*{2,1})
    // first axis directly (matches PyTorch's .view() flattening order exactly).
    ggml_tensor * off_raw = ggml_cont(ctx, linear(m, query, pre + ".sampling_offsets")); // (n_heads*n_levels*n_points*2, n_query, 1)
    ggml_tensor * aw_raw  = ggml_cont(ctx, linear(m, query, pre + ".attention_weights")); // (n_heads*n_levels*n_points, n_query, 1)
    ggml_tensor * aw = ggml_reshape_4d(ctx, aw_raw, n_levels * n_points, n_heads, n_query, 1);
    aw = ggml_soft_max(ctx, aw); // joint softmax over the flattened (n_levels*n_points) axis, per head/query

    ggml_tensor * ref_cxcy = ggml_cont(ctx, ggml_view_2d(ctx, ref_points, 2, n_query, ref_points->nb[1], 0));
    ggml_tensor * ref_wh   = ggml_cont(ctx, ggml_view_2d(ctx, ref_points, 2, n_query, ref_points->nb[1], 2 * sizeof(float)));

    ggml_tensor * out = nullptr;
    for (int h = 0; h < n_heads; h++) {
        ggml_tensor * head_out = nullptr;
        int64_t level_off = 0; // token offset of this level within value4's total_tokens axis
        for (int lvl = 0; lvl < n_levels; lvl++) {
            const int gw = levels[lvl].first, gh = levels[lvl].second;
            ggml_tensor * value_h = ggml_cont(ctx, ggml_view_2d(ctx, value4, head_dim, gw * gh,
                                                                value4->nb[2], (size_t) h * value4->nb[1] + (size_t) level_off * value4->nb[2]));
            for (int pt = 0; pt < n_points; pt++) {
                const size_t flat_idx = (size_t) h * n_levels * n_points + (size_t) lvl * n_points + (size_t) pt;
                ggml_tensor * off_hlp = ggml_cont(ctx, ggml_view_2d(ctx, off_raw, 2, n_query, off_raw->nb[1],
                                                    flat_idx * 2 * sizeof(float))); // (2,n_query)

                ggml_tensor * sloc = ggml_add(ctx, ggml_mul(ctx, ggml_scale(ctx, off_hlp, 0.5f / n_points), ref_wh), ref_cxcy);
                ggml_tensor * sloc_x = ggml_cont(ctx, ggml_view_2d(ctx, sloc, 1, n_query, sloc->nb[1], 0));
                ggml_tensor * sloc_y = ggml_cont(ctx, ggml_view_2d(ctx, sloc, 1, n_query, sloc->nb[1], sizeof(float)));

                ggml_tensor * ix = ggml_scale_bias(ctx, sloc_x, (float) gw, -0.5f); // align_corners=False unnormalize
                ggml_tensor * iy = ggml_scale_bias(ctx, sloc_y, (float) gh, -0.5f);
                ggml_tensor * ix0 = ggml_floor(ctx, ix);
                ggml_tensor * iy0 = ggml_floor(ctx, iy);
                ggml_tensor * ix1 = ggml_scale_bias(ctx, ix0, 1.0f, 1.0f);
                ggml_tensor * iy1 = ggml_scale_bias(ctx, iy0, 1.0f, 1.0f);
                ggml_tensor * wx1 = ggml_sub(ctx, ix, ix0), * wy1 = ggml_sub(ctx, iy, iy0);
                ggml_tensor * wx0 = ggml_scale_bias(ctx, wx1, -1.0f, 1.0f), * wy0 = ggml_scale_bias(ctx, wy1, -1.0f, 1.0f);

                ggml_tensor * mx0 = in_bounds_mask(ctx, ix0, gw), * mx1 = in_bounds_mask(ctx, ix1, gw);
                ggml_tensor * my0 = in_bounds_mask(ctx, iy0, gh), * my1 = in_bounds_mask(ctx, iy1, gh);
                // ggml_clamp is in-place -- clamp a ggml_cont'd copy, see the
                // single-level ms_deform_attn above for the full explanation.
                ggml_tensor * cx0 = ggml_clamp(ctx, ggml_cont(ctx, ix0), 0.0f, (float) (gw - 1));
                ggml_tensor * cx1 = ggml_clamp(ctx, ggml_cont(ctx, ix1), 0.0f, (float) (gw - 1));
                ggml_tensor * cy0 = ggml_clamp(ctx, ggml_cont(ctx, iy0), 0.0f, (float) (gh - 1));
                ggml_tensor * cy1 = ggml_clamp(ctx, ggml_cont(ctx, iy1), 0.0f, (float) (gh - 1));

                struct Corner { ggml_tensor * cx; ggml_tensor * cy; ggml_tensor * wx; ggml_tensor * wy; ggml_tensor * mx; ggml_tensor * my; };
                Corner corners[4] = {
                    { cx0, cy0, wx0, wy0, mx0, my0 },
                    { cx1, cy0, wx1, wy0, mx1, my0 },
                    { cx0, cy1, wx0, wy1, mx0, my1 },
                    { cx1, cy1, wx1, wy1, mx1, my1 },
                };
                ggml_tensor * sample_sum = nullptr;
                for (const Corner & c : corners) {
                    ggml_tensor * row_f = ggml_add(ctx, ggml_scale(ctx, c.cy, (float) gw), c.cx); // (1,n_query)
                    ggml_tensor * row_i = ggml_cast(ctx, ggml_reshape_1d(ctx, row_f, n_query), GGML_TYPE_I32);
                    ggml_tensor * gathered = ggml_get_rows(ctx, value_h, row_i); // (head_dim, n_query)
                    ggml_tensor * weight = ggml_mul(ctx, ggml_mul(ctx, c.wx, c.wy), ggml_mul(ctx, c.mx, c.my)); // (1,n_query)
                    ggml_tensor * contrib = ggml_mul(ctx, gathered, weight);
                    sample_sum = sample_sum ? ggml_add(ctx, sample_sum, contrib) : contrib;
                }

                ggml_tensor * aw_hlp = ggml_cont(ctx, ggml_view_2d(ctx, aw, 1, n_query, aw->nb[2],
                                                   (size_t) (lvl * n_points + pt) * aw->nb[0] + (size_t) h * aw->nb[1])); // (1,n_query)
                ggml_tensor * weighted = ggml_mul(ctx, sample_sum, aw_hlp);
                head_out = head_out ? ggml_add(ctx, head_out, weighted) : weighted;
            }
            level_off += (int64_t) gw * gh;
        }
        out = out ? ggml_concat(ctx, out, head_out, 0) : head_out; // concat heads along channel dim
    }

    out = ggml_reshape_3d(ctx, out, d_model, n_query, 1);
    return linear(m, out, pre + ".output_proj");
}
