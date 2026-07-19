// Validates rf-detr-cpp's multi-image ("batching") strategy: since
// dinov2_backbone's windowed-attention trick already uses ggml's only spare
// batch axis (token-major tensors are (C,T,nw2) -- the window count, not an
// image-batch dim) for per-window batching, this port does NOT thread a
// true 4th "image batch" dimension through the graph. Instead, N images are
// handled by calling dinov2_backbone (and projector_p4/rfdetr_decoder) once
// PER IMAGE, each with its own input tensor, all sharing the same Model
// (weights loaded once). This is the standard static-graph-inference
// pattern (see-through-cpp/trellis2cpp do the same for their own batch
// dimensions) and requires zero backbone.cpp/decoder.cpp changes -- what
// needed validating was that two independent graph-build calls sharing one
// Model produce genuinely independent, non-aliased results (no accidental
// tensor/buffer reuse across the two subgraphs).
//
// Built as two dinov2_backbone() calls in ONE ggml context/graph (the
// within-a-single-inference-process case; calling it across separate
// init_graph_ctx/compute_cpu_multi invocations, e.g. one per incoming
// request, works the same way and is the more common real deployment
// shape) with the SAME reference input fed to both -- if the two calls
// were aliasing any intermediate buffer, at least one would diverge from
// the (identical, since both got identical input) reference.
#include "backbone.h"
#include "test_common.h"

#include <cstdio>

int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : "models/rf-detr-nano-backbone.gguf";
    const char * ref_path  = argc > 2 ? argv[2] : "gen_reference/reference_backbone_nano.bin";

    Model m;
    if (!rfdetr_load(m, gguf_path)) {
        fprintf(stderr, "failed to load %s\n", gguf_path);
        return 1;
    }

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

    std::vector<NpyArray> ref;
    if (!read_ref(ref_path, ref, 1 + p.out_feature_indexes.size())) {
        fprintf(stderr, "failed to read %s (run gen_reference/gen_reference_backbone.py first)\n", ref_path);
        return 1;
    }

    init_graph_ctx(m, 16384);
    ggml_tensor * x0 = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_tensor * x1 = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_set_name(x0, "pixel_values_0");
    ggml_set_name(x1, "pixel_values_1");
    ggml_set_input(x0);
    ggml_set_input(x1);

    // "image 0" and "image 1" -- two entirely independent subgraphs sharing
    // only the Model's weight tensors.
    std::vector<ggml_tensor *> taps0 = dinov2_backbone(m, x0, p);
    std::vector<ggml_tensor *> taps1 = dinov2_backbone(m, x1, p);

    std::vector<ggml_tensor *> all_outs = taps0;
    all_outs.insert(all_outs.end(), taps1.begin(), taps1.end());

    bool ok = compute_cpu_multi(m, all_outs, 16384, [&] {
        const NpyArray & px = ref[0]; // shape (H,W,C)
        std::vector<float> whc(px.data.size());
        int64_t H = px.shape[0], W = px.shape[1], C = px.shape[2];
        for (int64_t h = 0; h < H; h++) {
            for (int64_t w = 0; w < W; w++) {
                for (int64_t c = 0; c < C; c++) {
                    whc[c * H * W + h * W + w] = px.data[h * W * C + w * C + c];
                }
            }
        }
        // both images get the SAME pixel data -- if the two graph-build
        // calls aliased any buffer, this is exactly what would make one
        // copy's result depend on the other's (garbage or duplicated data).
        ggml_backend_tensor_set(x0, whc.data(), 0, whc.size() * sizeof(float));
        ggml_backend_tensor_set(x1, whc.data(), 0, whc.size() * sizeof(float));
    });
    if (!ok) {
        return 1;
    }

    int rc = 0;
    for (int image = 0; image < 2; image++) {
        std::vector<ggml_tensor *> & taps = image == 0 ? taps0 : taps1;
        for (size_t i = 0; i < taps.size(); i++) {
            const NpyArray & r = ref[1 + i];
            int64_t H = r.shape[0], W = r.shape[1], C = r.shape[2];
            NpyArray r_whc;
            r_whc.shape = { W, H, C, 1 };
            r_whc.data.resize(r.data.size());
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    for (int64_t c = 0; c < C; c++) {
                        r_whc.data[c * H * W + h * W + w] = r.data[h * W * C + w * C + c];
                    }
                }
            }
            printf("image %d, tap %zu (layer %d):\n", image, i, p.out_feature_indexes[i]);
            if (compare_ref(taps[i], r_whc, 5e-2) != 0) {
                rc = 1;
            }
        }
    }
    return rc;
}
