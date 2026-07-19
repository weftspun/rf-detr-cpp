#include "segmentation.h"

// DepthwiseConvBlock: depthwise 3x3 conv (residual) -> channels-last LN
// (eps=1e-6) -> pointwise Linear(dim,dim) -> exact GELU -> (no LayerScale,
// gamma is None since layer_scale_init_value=0 default) -> residual add.
static ggml_tensor * depthwise_conv_block(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];

    ggml_tensor * w = m.get(pre + ".dwconv.weight"); // (kw=3,kh=3,1,C)
    ggml_tensor * y = ggml_conv_2d_dw_direct(ctx, w, x, 1, 1, 1, 1, 1, 1);
    ggml_tensor * b = ggml_reshape_4d(ctx, m.get(pre + ".dwconv.bias"), 1, 1, C, 1);
    y = ggml_add(ctx, y, b);

    y = spatial_layer_norm_affine(m, y, pre + ".norm", 1e-6f);

    ggml_tensor * y_tok = ggml_reshape_3d(ctx, y, W * H, C, 1);
    y_tok = ggml_cont(ctx, ggml_permute(ctx, y_tok, 1, 0, 2, 3)); // (C, W*H, 1)
    y_tok = linear(m, y_tok, pre + ".pwconv1");
    y_tok = ggml_gelu_erf(ctx, y_tok);
    y = ggml_cont(ctx, ggml_permute(ctx, y_tok, 1, 0, 2, 3));     // (W*H, C, 1)
    y = ggml_reshape_4d(ctx, y, W, H, C, 1);

    return ggml_add(ctx, x, y);
}

// MLPBlock (query_features_block): pre-LN (eps=1e-5, nn.LayerNorm default)
// -> Linear(dim,4dim) -> exact GELU -> Linear(4dim,dim) -> residual (no
// LayerScale, same gamma=None default as DepthwiseConvBlock).
static ggml_tensor * mlp_block(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * h = layer_norm_affine(m, x, pre + ".norm_in", 1e-5f);
    h = linear(m, h, pre + ".layers.0");
    h = ggml_gelu_erf(ctx, h);
    h = linear(m, h, pre + ".layers.2");
    return ggml_add(ctx, x, h);
}

std::vector<ggml_tensor *> segmentation_head(Model & m, ggml_tensor * spatial_features,
                                             const std::vector<ggml_tensor *> & hidden_states,
                                             const SegmentationParams & p) {
    ggml_context * ctx = m.ctx_g;
    const int64_t target_w = p.image_w / p.downsample_ratio;
    const int64_t target_h = p.image_h / p.downsample_ratio;

    ggml_tensor * feat = ggml_interpolate(ctx, spatial_features, target_w, target_h,
                                          spatial_features->ne[2], spatial_features->ne[3],
                                          GGML_SCALE_MODE_BILINEAR);

    ggml_tensor * bias = m.get("segmentation_head.bias"); // (1,)

    std::vector<ggml_tensor *> masks;
    for (int i = 0; i < p.num_blocks; i++) {
        const std::string bpre = "segmentation_head.blocks." + std::to_string(i);
        feat = depthwise_conv_block(m, feat, bpre);

        ggml_tensor * feat_proj = conv2d(m, feat, "segmentation_head.spatial_features_proj", 1, 0); // (tw,th,hd,1)
        ggml_tensor * feat_tok = ggml_reshape_3d(ctx, feat_proj, target_w * target_h, p.hidden_dim, 1);
        feat_tok = ggml_cont(ctx, ggml_permute(ctx, feat_tok, 1, 0, 2, 3)); // (hidden_dim, tw*th, 1)

        ggml_tensor * qf = mlp_block(m, hidden_states[i], "segmentation_head.query_features_block");
        qf = linear(m, qf, "segmentation_head.query_features_proj"); // (hidden_dim, num_queries, 1)

        ggml_tensor * mask = ggml_mul_mat(ctx, feat_tok, qf); // (tw*th, num_queries, 1)
        mask = ggml_add(ctx, mask, bias);
        mask = ggml_reshape_4d(ctx, mask, target_w, target_h, qf->ne[1], 1);
        masks.push_back(mask);
    }
    return masks;
}
