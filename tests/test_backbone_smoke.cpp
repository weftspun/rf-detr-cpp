// Structural smoke test for the DINOv2 windowed-attention backbone graph:
// builds it with small random weights (no GGUF/checkpoint needed) and checks
// it computes finite output of the expected shape. This is NOT a numerical
// validation against upstream — see docs/decisions/backbone-windowing.md.
#include "backbone.h"
#include "test_common.h"

#include <cmath>
#include <cstdio>
#include <random>

static ggml_tensor * add_weight(Model & m, ggml_context * ctx, const std::string & name,
                                 std::vector<int64_t> ne, std::mt19937 & rng) {
    while (ne.size() < 4) ne.push_back(1);
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    ggml_set_name(t, name.c_str());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> buf(ggml_nelements(t));
    for (float & v : buf) v = dist(rng);
    memcpy(t->data, buf.data(), buf.size() * sizeof(float));
    m.weights[name] = t;
    return t;
}

int main() {
    BackboneParams p;
    p.hidden      = 32;
    p.n_layer     = 4;
    p.n_head      = 2;
    p.patch_size  = 8;
    p.n_register  = 2;
    p.num_windows = 2;
    // window_block_indexes is derived upstream: range(out_feature_indexes.back()+1)
    // minus out_feature_indexes (see docs/decisions/backbone-windowing.md).
    p.out_feature_indexes  = { 1, 3 };
    p.window_block_indexes = { 0, 2 };

    const int64_t img_w = 64, img_h = 64; // gw=gh=8, divisible by num_windows=2
    const int64_t gw = img_w / p.patch_size, gh = img_h / p.patch_size;
    const int64_t n_patch = gw * gh;

    std::mt19937 rng(42);
    Model m;
    // weight context large enough for every named tensor below
    ggml_init_params wip = { 64 * 1024 * 1024, nullptr, /*no_alloc*/ false };
    ggml_context * wctx = ggml_init(wip);
    m.ctx_w.push_back(wctx);

    add_weight(m, wctx, "embeddings.patch_embeddings.projection.weight", { p.patch_size, p.patch_size, 3, p.hidden }, rng);
    add_weight(m, wctx, "embeddings.patch_embeddings.projection.bias", { p.hidden }, rng);
    add_weight(m, wctx, "embeddings.cls_token", { p.hidden }, rng);
    add_weight(m, wctx, "embeddings.register_tokens", { p.hidden, p.n_register }, rng);
    add_weight(m, wctx, "embeddings.position_embeddings", { p.hidden, 1 + n_patch, 1 }, rng);
    add_weight(m, wctx, "layernorm.weight", { p.hidden }, rng);
    add_weight(m, wctx, "layernorm.bias", { p.hidden }, rng);

    for (int l = 0; l < p.n_layer; l++) {
        std::string pre = "encoder.layer." + std::to_string(l) + ".";
        add_weight(m, wctx, pre + "norm1.weight", { p.hidden }, rng);
        add_weight(m, wctx, pre + "norm1.bias", { p.hidden }, rng);
        for (const char * proj : { "query", "key", "value" }) {
            add_weight(m, wctx, pre + "attention.attention." + proj + ".weight", { p.hidden, p.hidden }, rng);
            add_weight(m, wctx, pre + "attention.attention." + proj + ".bias", { p.hidden }, rng);
        }
        add_weight(m, wctx, pre + "attention.output.dense.weight", { p.hidden, p.hidden }, rng);
        add_weight(m, wctx, pre + "attention.output.dense.bias", { p.hidden }, rng);
        add_weight(m, wctx, pre + "layer_scale1.lambda1", { p.hidden }, rng);
        add_weight(m, wctx, pre + "norm2.weight", { p.hidden }, rng);
        add_weight(m, wctx, pre + "norm2.bias", { p.hidden }, rng);
        add_weight(m, wctx, pre + "mlp.fc1.weight", { p.hidden, p.hidden * 4 }, rng);
        add_weight(m, wctx, pre + "mlp.fc1.bias", { p.hidden * 4 }, rng);
        add_weight(m, wctx, pre + "mlp.fc2.weight", { p.hidden * 4, p.hidden }, rng);
        add_weight(m, wctx, pre + "mlp.fc2.bias", { p.hidden }, rng);
        add_weight(m, wctx, pre + "layer_scale2.lambda1", { p.hidden }, rng);
    }

    init_graph_ctx(m, 4096);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, img_w, img_h, 3, 1);
    ggml_set_name(x, "pixel_values");
    ggml_set_input(x);

    std::vector<ggml_tensor *> taps = dinov2_backbone(m, x, p);
    if (taps.size() != p.out_feature_indexes.size()) {
        fprintf(stderr, "expected %zu taps, got %zu\n", p.out_feature_indexes.size(), taps.size());
        return 1;
    }

    bool ok = compute_cpu_multi(m, taps, 4096, [&] {
        std::vector<float> px(ggml_nelements(x));
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float & v : px) v = dist(rng);
        ggml_backend_tensor_set(x, px.data(), 0, px.size() * sizeof(float));
    });
    if (!ok) return 1;

    for (size_t i = 0; i < taps.size(); i++) {
        ggml_tensor * t = taps[i];
        if (t->ne[0] != gw || t->ne[1] != gh || t->ne[2] != p.hidden) {
            fprintf(stderr, "tap %zu wrong shape: (%lld,%lld,%lld,%lld)\n", i,
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
            return 1;
        }
        std::vector<float> y(ggml_nelements(t));
        ggml_backend_tensor_get(t, y.data(), 0, y.size() * sizeof(float));
        for (float v : y) {
            if (!std::isfinite(v)) { fprintf(stderr, "tap %zu has non-finite value\n", i); return 1; }
        }
        printf("tap %zu: shape (%lld,%lld,%lld,%lld) OK\n", i,
               (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    }
    printf("SMOKE TEST PASS\n");
    return 0;
}
