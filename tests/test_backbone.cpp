// Validates src/backbone.cpp's DINOv2-windowed-attention graph against a
// real RFDETRNano checkpoint, diffed against the actual upstream PyTorch
// module (gen_reference/gen_reference_backbone.py). See
// docs/decisions/backbone-windowing.md for the verified architecture.
#include "backbone.h"
#include "test_common.h"

#include <cstdio>

int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : "models/rf-detr-nano-backbone.gguf";
    const char * ref_path  = argc > 2 ? argv[2] : "gen_reference/reference_backbone_nano.bin";

    Model m;
    if (!rfdetr_load(m, gguf_path)) { fprintf(stderr, "failed to load %s\n", gguf_path); return 1; }

    BackboneParams p;
    p.hidden      = 384;
    p.n_layer     = 12;
    p.n_head      = 6;
    p.patch_size  = 16;
    p.n_register  = 0;
    p.num_windows = 2;
    p.window_block_indexes = { 0, 1, 2, 4, 5, 7, 8, 10, 11 };
    p.out_feature_indexes  = { 2, 5, 8, 11 };

    const int64_t res = 384;
    const int64_t gw  = res / p.patch_size; // 24

    std::vector<NpyArray> ref;
    if (!read_ref(ref_path, ref, 1 + p.out_feature_indexes.size())) {
        fprintf(stderr, "failed to read %s (run gen_reference/gen_reference_backbone.py first)\n", ref_path);
        return 1;
    }

    init_graph_ctx(m, 8192);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_set_name(x, "pixel_values");
    ggml_set_input(x);

    std::vector<ggml_tensor *> taps = dinov2_backbone(m, x, p);

    bool ok = compute_cpu_multi(m, taps, 8192, [&] {
        // ref[0] is (H,W,C) numpy from the (1,3,H,W) input, permuted to
        // (H,W,C); ggml wants WHCN (W fastest). Row-major (H,W,C) with W
        // adjacent-fastest-after-C means we need (C fastest within a
        // pixel) -> transpose to (W,H,C) order for ggml's im2col/conv2d,
        // which reads x as ne0=W,ne1=H,ne2=C.
        const NpyArray & px = ref[0]; // shape (H,W,C)
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
        // ref[1+i] is (H,W,C); taps[i] is ggml WHCN (W,H,C,1) -- same
        // transpose as the input pixels before compare_ref's flat memcmp.
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
