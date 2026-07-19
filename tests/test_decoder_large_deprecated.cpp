// Validates the full RFDETRLargeDeprecated pipeline (backbone -> multi-scale
// projector -> multi-level decoder) against the real checkpoint -- the one
// remaining detection variant needing real new code (num_feature_levels=2,
// projector_scale=["P3","P5"], 768-wide backbone, hidden_dim=384 decoder)
// rather than a config-value extension. See docs/decisions/0001-open-work.md
// for the full architecture summary and every checkpoint-verification
// surprise found while building this (stages_sampling's ".0" Sequential
// index, P5's act="relu" not "silu", the sine-embedding dim being
// hidden_dim/2 not a hardcoded 128). Same topk-override technique as every
// other test_decoder_*.cpp (see docs/decisions/decoder.md).
#include "backbone.h"
#include "decoder.h"
#include "projector.h"
#include "test_common.h"

#include <cstdio>

int main(int argc, char ** argv) {
    const char * backbone_gguf  = argc > 1 ? argv[1] : "models/rf-detr-large-deprecated-backbone.gguf";
    const char * projector_gguf = argc > 2 ? argv[2] : "models/rf-detr-large-deprecated-projector.gguf";
    const char * decoder_gguf   = argc > 3 ? argv[3] : "models/rf-detr-large-deprecated-decoder.gguf";
    const char * backbone_ref_path = argc > 4 ? argv[4] : "gen_reference/reference_backbone_large_deprecated.bin";
    const char * decoder_ref_path  = argc > 5 ? argv[5] : "gen_reference/reference_decoder_large_deprecated.bin";

    Model m;
    if (!rfdetr_load(m, backbone_gguf)) {
        fprintf(stderr, "failed to load %s\n", backbone_gguf);
        return 1;
    }
    if (!rfdetr_load(m, projector_gguf)) {
        fprintf(stderr, "failed to load %s\n", projector_gguf);
        return 1;
    }
    if (!rfdetr_load(m, decoder_gguf)) {
        fprintf(stderr, "failed to load %s\n", decoder_gguf);
        return 1;
    }

    BackboneParams bp;
    bp.hidden = 768; bp.n_layer = 12; bp.n_head = 12; bp.patch_size = 14; bp.n_register = 0; bp.num_windows = 4;
    bp.window_block_indexes = { 0, 1, 3, 4, 6, 7, 9, 10 };
    bp.out_feature_indexes  = { 1, 4, 7, 10 };
    bp.native_grid = 37;

    DecoderParams dp;
    dp.hidden_dim = 384; dp.dec_layers = 3; dp.sa_nheads = 12; dp.ca_nheads = 24; dp.dec_n_points = 4;
    dp.num_queries = 300; dp.num_classes = 91;
    dp.levels = { { 80, 80 }, { 20, 20 } }; // P3 (2x upsample of the 40x40 P4-equivalent grid), P5 (0.5x downsample)

    const int64_t res = 560;

    std::vector<NpyArray> backbone_ref;
    if (!read_ref(backbone_ref_path, backbone_ref, 5)) {
        fprintf(stderr, "failed to read %s\n", backbone_ref_path);
        return 1;
    }
    std::vector<NpyArray> decoder_ref;
    if (!read_ref(decoder_ref_path, decoder_ref, 3)) {
        fprintf(stderr, "failed to read %s\n", decoder_ref_path);
        return 1;
    }
    const NpyArray & topk_ref = decoder_ref[0], & boxes_ref = decoder_ref[1], & logits_ref = decoder_ref[2];

    init_graph_ctx(m, 200000);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_set_name(x, "pixel_values");
    ggml_set_input(x);
    ggml_tensor * topk_override = ggml_new_tensor_1d(m.ctx_g, GGML_TYPE_I32, dp.num_queries);
    ggml_set_name(topk_override, "topk_override");
    ggml_set_input(topk_override);

    std::vector<ggml_tensor *> taps = dinov2_backbone(m, x, bp);
    std::vector<ggml_tensor *> fused = projector_multiscale(m, taps, { 2.0f, 0.5f }, dp.hidden_dim); // [P3,P5]

    ggml_tensor * memory = nullptr;
    for (size_t lvl = 0; lvl < fused.size(); lvl++) {
        const int gw = dp.levels[lvl].first, gh = dp.levels[lvl].second;
        ggml_tensor * lvl_mem = ggml_reshape_3d(m.ctx_g, fused[lvl], gw * gh, dp.hidden_dim, 1);
        lvl_mem = ggml_cont(m.ctx_g, ggml_permute(m.ctx_g, lvl_mem, 1, 0, 2, 3)); // (hidden_dim,gw*gh,1)
        memory = memory ? ggml_concat(m.ctx_g, memory, lvl_mem, 1) : lvl_mem;
    }

    DecoderOutput out = rfdetr_decoder(m, memory, dp, topk_override);

    std::vector<ggml_tensor *> outs = { out.pred_boxes, out.pred_logits };
    bool ok = compute_cpu_multi(m, outs, 200000, [&] {
        std::vector<float> whc(backbone_ref[0].data.size());
        int64_t H = backbone_ref[0].shape[0], W = backbone_ref[0].shape[1], C = backbone_ref[0].shape[2];
        for (int64_t h = 0; h < H; h++) {
            for (int64_t w = 0; w < W; w++) {
                for (int64_t c = 0; c < C; c++) {
                    whc[c * H * W + h * W + w] = backbone_ref[0].data[h * W * C + w * C + c];
                }
            }
        }
        ggml_backend_tensor_set(x, whc.data(), 0, whc.size() * sizeof(float));

        std::vector<float> prop = output_proposals_data_multilevel(dp.levels);
        ggml_backend_tensor_set(out.output_proposals, prop.data(), 0, prop.size() * sizeof(float));
        std::vector<float> vmask = valid_mask_data_multilevel(dp.levels);
        ggml_backend_tensor_set(out.valid_mask, vmask.data(), 0, vmask.size() * sizeof(float));

        std::vector<int32_t> idx(dp.num_queries);
        for (int i = 0; i < dp.num_queries; i++) {
            idx[i] = (int32_t) topk_ref.data[i];
        }
        ggml_backend_tensor_set(topk_override, idx.data(), 0, idx.size() * sizeof(int32_t));
    });
    if (!ok) {
        return 1;
    }

    printf("pred_boxes:\n");
    int rc1 = compare_ref(out.pred_boxes, boxes_ref, 5e-2);
    printf("pred_logits:\n");
    int rc2 = compare_ref(out.pred_logits, logits_ref, 5e-2);
    return (rc1 || rc2) ? 1 : 0;
}
