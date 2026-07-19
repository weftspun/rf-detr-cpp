// Validates src/projector.cpp against the real RFDETRNano checkpoint: feeds
// the backbone's real taps into the projector and diffs the fused P4 map
// against gen_reference/gen_reference_projector.py's dump of the real
// upstream Backbone module. Input pixels are read from
// reference_backbone_nano.bin (same torch.manual_seed(0) + randn(1,3,384,384)
// call sequence as gen_reference_projector.py, so bit-identical).
#include "backbone.h"
#include "projector.h"
#include "test_common.h"

#include <cstdio>

static void whc_to_ggml(const NpyArray & r, ggml_tensor * dst) {
    int64_t H = r.shape[0], W = r.shape[1], C = r.shape[2];
    std::vector<float> whc(r.data.size());
    for (int64_t h = 0; h < H; h++)
        for (int64_t w = 0; w < W; w++)
            for (int64_t c = 0; c < C; c++)
                whc[c * H * W + h * W + w] = r.data[h * W * C + w * C + c];
    ggml_backend_tensor_set(dst, whc.data(), 0, whc.size() * sizeof(float));
}

static NpyArray hwc_to_whc_ref(const NpyArray & r) {
    int64_t H = r.shape[0], W = r.shape[1], C = r.shape[2];
    NpyArray out;
    out.shape = { W, H, C, 1 };
    out.data.resize(r.data.size());
    for (int64_t h = 0; h < H; h++)
        for (int64_t w = 0; w < W; w++)
            for (int64_t c = 0; c < C; c++)
                out.data[c * H * W + h * W + w] = r.data[h * W * C + w * C + c];
    return out;
}

int main(int argc, char ** argv) {
    const char * backbone_gguf  = argc > 1 ? argv[1] : "models/rf-detr-nano-backbone.gguf";
    const char * projector_gguf = argc > 2 ? argv[2] : "models/rf-detr-nano-projector.gguf";
    const char * backbone_ref_path  = argc > 3 ? argv[3] : "gen_reference/reference_backbone_nano.bin";
    const char * projector_ref_path = argc > 4 ? argv[4] : "gen_reference/reference_projector_nano.bin";

    Model m;
    if (!rfdetr_load(m, backbone_gguf))  { fprintf(stderr, "failed to load %s\n", backbone_gguf); return 1; }
    if (!rfdetr_load(m, projector_gguf)) { fprintf(stderr, "failed to load %s\n", projector_gguf); return 1; }

    BackboneParams p;
    p.hidden = 384; p.n_layer = 12; p.n_head = 6; p.patch_size = 16; p.n_register = 0; p.num_windows = 2;
    p.window_block_indexes = { 0, 1, 2, 4, 5, 7, 8, 10, 11 };
    p.out_feature_indexes  = { 2, 5, 8, 11 };

    const int64_t res = 384;

    std::vector<NpyArray> backbone_ref;
    if (!read_ref(backbone_ref_path, backbone_ref, 5)) {
        fprintf(stderr, "failed to read %s\n", backbone_ref_path); return 1;
    }
    std::vector<NpyArray> projector_ref;
    if (!read_ref(projector_ref_path, projector_ref, 5)) {
        fprintf(stderr, "failed to read %s\n", projector_ref_path); return 1;
    }

    init_graph_ctx(m, 8192);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_set_name(x, "pixel_values");
    ggml_set_input(x);

    std::vector<ggml_tensor *> taps = dinov2_backbone(m, x, p);
    ggml_tensor * fused = projector_p4(m, taps, 256);

    std::vector<ggml_tensor *> outs = taps;
    outs.push_back(fused);

    bool ok = compute_cpu_multi(m, outs, 8192, [&] {
        whc_to_ggml(backbone_ref[0], x); // same input as test_backbone.cpp
    });
    if (!ok) return 1;

    printf("fused P4 map:\n");
    NpyArray fused_ref = hwc_to_whc_ref(projector_ref[4]);
    return compare_ref(fused, fused_ref, 5e-2);
}
