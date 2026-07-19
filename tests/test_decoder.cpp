// Validates src/decoder.cpp end-to-end (backbone -> projector -> decoder)
// against the real RFDETRNano checkpoint. Uses the reference's own
// torch.topk indices as an override (ggml_top_k's output order is not
// guaranteed to match torch.topk's -- see docs/decisions/decoder.md) so
// this validates every other part of the decoder exactly.
#include "backbone.h"
#include "decoder.h"
#include "projector.h"
#include "test_common.h"

#include <cstdio>

int main(int argc, char ** argv) {
    const char * backbone_gguf  = argc > 1 ? argv[1] : "models/rf-detr-nano-backbone.gguf";
    const char * projector_gguf = argc > 2 ? argv[2] : "models/rf-detr-nano-projector.gguf";
    const char * decoder_gguf   = argc > 3 ? argv[3] : "models/rf-detr-nano-decoder.gguf";
    const char * backbone_ref_path = argc > 4 ? argv[4] : "gen_reference/reference_backbone_nano.bin";
    const char * decoder_ref_path  = argc > 5 ? argv[5] : "gen_reference/reference_decoder_nano.bin";

    Model m;
    if (!rfdetr_load(m, backbone_gguf))  { fprintf(stderr, "failed to load %s\n", backbone_gguf); return 1; }
    if (!rfdetr_load(m, projector_gguf)) { fprintf(stderr, "failed to load %s\n", projector_gguf); return 1; }
    if (!rfdetr_load(m, decoder_gguf))   { fprintf(stderr, "failed to load %s\n", decoder_gguf); return 1; }

    BackboneParams bp;
    bp.hidden = 384; bp.n_layer = 12; bp.n_head = 6; bp.patch_size = 16; bp.n_register = 0; bp.num_windows = 2;
    bp.window_block_indexes = { 0, 1, 2, 4, 5, 7, 8, 10, 11 };
    bp.out_feature_indexes  = { 2, 5, 8, 11 };

    DecoderParams dp;
    dp.hidden_dim = 256; dp.dec_layers = 2; dp.sa_nheads = 8; dp.ca_nheads = 16; dp.dec_n_points = 2;
    dp.num_queries = 300; dp.num_classes = 91; dp.gw = 24; dp.gh = 24;

    const int64_t res = 384;

    std::vector<NpyArray> backbone_ref;
    if (!read_ref(backbone_ref_path, backbone_ref, 5)) { fprintf(stderr, "failed to read %s\n", backbone_ref_path); return 1; }
    std::vector<NpyArray> decoder_ref;
    if (!read_ref(decoder_ref_path, decoder_ref, 3)) { fprintf(stderr, "failed to read %s\n", decoder_ref_path); return 1; }
    const NpyArray & topk_ref = decoder_ref[0], & boxes_ref = decoder_ref[1], & logits_ref = decoder_ref[2];

    init_graph_ctx(m, 65536);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_set_name(x, "pixel_values");
    ggml_set_input(x);
    ggml_tensor * topk_override = ggml_new_tensor_1d(m.ctx_g, GGML_TYPE_I32, dp.num_queries);
    ggml_set_name(topk_override, "topk_override");
    ggml_set_input(topk_override);

    std::vector<ggml_tensor *> taps = dinov2_backbone(m, x, bp);
    ggml_tensor * fused = projector_p4(m, taps, 256); // (gw,gh,256,1)
    ggml_tensor * memory = ggml_reshape_3d(m.ctx_g, fused, dp.gw * dp.gh, dp.hidden_dim, 1);
    memory = ggml_cont(m.ctx_g, ggml_permute(m.ctx_g, memory, 1, 0, 2, 3)); // (256, gw*gh, 1)

    DecoderOutput out = rfdetr_decoder(m, memory, dp, topk_override);

    std::vector<ggml_tensor *> outs = { out.pred_boxes, out.pred_logits };
    bool ok = compute_cpu_multi(m, outs, 65536, [&] {
        std::vector<float> whc(backbone_ref[0].data.size());
        int64_t H = backbone_ref[0].shape[0], W = backbone_ref[0].shape[1], C = backbone_ref[0].shape[2];
        for (int64_t h = 0; h < H; h++)
            for (int64_t w = 0; w < W; w++)
                for (int64_t c = 0; c < C; c++)
                    whc[c * H * W + h * W + w] = backbone_ref[0].data[h * W * C + w * C + c];
        ggml_backend_tensor_set(x, whc.data(), 0, whc.size() * sizeof(float));

        std::vector<float> prop = output_proposals_data(dp.gw, dp.gh);
        ggml_backend_tensor_set(out.output_proposals, prop.data(), 0, prop.size() * sizeof(float));

        std::vector<int32_t> idx(dp.num_queries);
        for (int i = 0; i < dp.num_queries; i++) idx[i] = (int32_t) topk_ref.data[i];
        ggml_backend_tensor_set(topk_override, idx.data(), 0, idx.size() * sizeof(int32_t));
    });
    if (!ok) return 1;

    printf("pred_boxes:\n");
    int rc1 = compare_ref(out.pred_boxes, boxes_ref, 5e-2);
    printf("pred_logits:\n");
    int rc2 = compare_ref(out.pred_logits, logits_ref, 5e-2);
    return (rc1 || rc2) ? 1 : 0;
}
