#include "projector.h"

// ConvX: bias-free conv -> spatial LayerNorm (ConvNeXt-style, called ".bn"
// in the checkpoint even though it's a LayerNorm since layer_norm=True is
// threaded through every ConvX in this model) -> SiLU.
static ggml_tensor * conv_x(Model & m, ggml_tensor * x, const std::string & pre, int kernel, int stride) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * y = conv2d(m, x, pre + ".conv", stride, kernel / 2);
    y = spatial_layer_norm_affine(m, y, pre + ".bn");
    return ggml_silu(ctx, y);
}

// Bottleneck(c,c,shortcut=False): two 3x3 ConvX, no residual add (C2f always
// constructs its bottlenecks with shortcut=False here).
static ggml_tensor * bottleneck(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_tensor * y = conv_x(m, x, pre + ".cv1", 3, 1);
    return conv_x(m, y, pre + ".cv2", 3, 1);
}

// C2f(in_channels -> out_channels, n=3): cv1 (1x1, in->out) split into two
// out/2 halves, n bottlenecks chained off the second half (each appended to
// the running list), all n+2 chunks concatenated, cv2 (1x1, (n+2)*out/2 -> out).
static ggml_tensor * c2f(Model & m, ggml_tensor * x, const std::string & pre, int out_channels, int n) {
    ggml_context * ctx = m.ctx_g;
    const int half = out_channels / 2;

    ggml_tensor * y = conv_x(m, x, pre + ".cv1", 1, 1); // (W,H,out_channels,1)
    ggml_tensor * y0 = ggml_cont(ctx, ggml_view_4d(ctx, y, y->ne[0], y->ne[1], half, y->ne[3],
                                                   y->nb[1], y->nb[2], y->nb[3], 0));
    ggml_tensor * y1 = ggml_cont(ctx, ggml_view_4d(ctx, y, y->ne[0], y->ne[1], half, y->ne[3],
                                                   y->nb[1], y->nb[2], y->nb[3], (size_t) half * y->nb[2]));

    ggml_tensor * cat = ggml_concat(ctx, y0, y1, 2);
    ggml_tensor * cur = y1;
    for (int i = 0; i < n; i++) {
        cur = bottleneck(m, cur, pre + ".m." + std::to_string(i));
        cat = ggml_concat(ctx, cat, cur, 2);
    }
    return conv_x(m, cat, pre + ".cv2", 1, 1);
}

ggml_tensor * projector_p4(Model & m, const std::vector<ggml_tensor *> & taps, int out_channels,
                           const std::string & prefix) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * fused = taps[0];
    for (size_t i = 1; i < taps.size(); i++) fused = ggml_concat(ctx, fused, taps[i], 2);

    ggml_tensor * y = c2f(m, fused, prefix + ".stages.0.0", out_channels, 3);
    return spatial_layer_norm_affine(m, y, prefix + ".stages.0.1");
}
