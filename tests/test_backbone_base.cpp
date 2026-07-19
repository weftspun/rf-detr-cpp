// Validates src/backbone.cpp's position-embedding bicubic+antialias
// interpolation path (native_grid != runtime grid) against the real
// RFDETRBase checkpoint. See docs/decisions/0002-position-embed-bicubic.md.
#include "backbone.h"
#include "test_common.h"

#include <cstdio>

int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : "models/rf-detr-base-backbone.gguf";
    const char * ref_path  = argc > 2 ? argv[2] : "gen_reference/reference_backbone_base.bin";

    Model m;
    if (!rfdetr_load(m, gguf_path)) { fprintf(stderr, "failed to load %s\n", gguf_path); return 1; }

    BackboneParams p;
    p.hidden      = 384;
    p.n_layer     = 12;
    p.n_head      = 6;
    p.patch_size  = 14;
    p.n_register  = 0;
    p.num_windows = 4;
    p.window_block_indexes = { 0, 1, 3, 4, 6, 7, 9, 10 };
    p.out_feature_indexes  = { 1, 4, 7, 10 };
    p.native_grid = 37; // checkpoint's native 37x37 DINOv2 grid; runtime is 40x40

    const int64_t res = 560;

    std::vector<NpyArray> ref;
    if (!read_ref(ref_path, ref, 1 + p.out_feature_indexes.size())) {
        fprintf(stderr, "failed to read %s (run gen_reference/gen_reference_backbone.py first)\n", ref_path);
        return 1;
    }

    init_graph_ctx(m, 16384);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_set_name(x, "pixel_values");
    ggml_set_input(x);

    std::vector<ggml_tensor *> taps = dinov2_backbone(m, x, p);

    bool ok = compute_cpu_multi(m, taps, 16384, [&] {
        const NpyArray & px = ref[0];
        std::vector<float> whc(px.data.size());
        int64_t H = px.shape[0], W = px.shape[1], C = px.shape[2];
        for (int64_t h = 0; h < H; h++)
            for (int64_t w = 0; w < W; w++)
                for (int64_t c = 0; c < C; c++)
                    whc[c * H * W + h * W + w] = px.data[h * W * C + w * C + c];
        ggml_backend_tensor_set(x, whc.data(), 0, whc.size() * sizeof(float));
    });
    if (!ok) return 1;

    int rc = 0;
    for (size_t i = 0; i < taps.size(); i++) {
        const NpyArray & r = ref[1 + i];
        int64_t H = r.shape[0], W = r.shape[1], C = r.shape[2];
        NpyArray r_whc;
        r_whc.shape = { W, H, C, 1 };
        r_whc.data.resize(r.data.size());
        for (int64_t h = 0; h < H; h++)
            for (int64_t w = 0; w < W; w++)
                for (int64_t c = 0; c < C; c++)
                    r_whc.data[c * H * W + h * W + w] = r.data[h * W * C + w * C + c];
        printf("tap %zu (layer %d):\n", i, p.out_feature_indexes[i]);
        if (compare_ref(taps[i], r_whc, 5e-2) != 0) rc = 1;
    }
    return rc;
}
