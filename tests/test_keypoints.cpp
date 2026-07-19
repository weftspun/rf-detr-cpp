// Validates the GroupPose keypoint head end-to-end (backbone -> dual
// projector -> decoder+keypoint stream) against the real
// rf-detr-keypoint-preview checkpoint. Uses the reference's own torch.topk
// indices as an override, same as test_decoder.cpp/test_segmentation.cpp.
#include "backbone.h"
#include "decoder.h"
#include "keypoints.h"
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

int main(int argc, char ** argv) {
    const char * backbone_gguf   = argc > 1 ? argv[1] : "models/rf-detr-keypoint-preview-backbone.gguf";
    const char * projector_gguf  = argc > 2 ? argv[2] : "models/rf-detr-keypoint-preview-projector.gguf";
    const char * cross_proj_gguf = argc > 3 ? argv[3] : "models/rf-detr-keypoint-preview-cross-projector.gguf";
    const char * decoder_gguf    = argc > 4 ? argv[4] : "models/rf-detr-keypoint-preview-decoder.gguf";
    const char * kp_gguf         = argc > 5 ? argv[5] : "models/rf-detr-keypoint-preview-keypoints.gguf";
    const char * ref_path        = argc > 6 ? argv[6] : "gen_reference/reference_keypoints_preview.bin";

    Model m;
    if (!rfdetr_load(m, backbone_gguf))   { fprintf(stderr, "failed to load %s\n", backbone_gguf); return 1; }
    if (!rfdetr_load(m, projector_gguf))  { fprintf(stderr, "failed to load %s\n", projector_gguf); return 1; }
    if (!rfdetr_load(m, cross_proj_gguf)) { fprintf(stderr, "failed to load %s\n", cross_proj_gguf); return 1; }
    if (!rfdetr_load(m, decoder_gguf))    { fprintf(stderr, "failed to load %s\n", decoder_gguf); return 1; }
    if (!rfdetr_load(m, kp_gguf))         { fprintf(stderr, "failed to load %s\n", kp_gguf); return 1; }

    BackboneParams bp;
    bp.hidden = 384; bp.n_layer = 12; bp.n_head = 6; bp.patch_size = 12; bp.n_register = 0; bp.num_windows = 2;
    bp.window_block_indexes = { 0, 1, 2, 4, 5, 7, 8, 10, 11 };
    bp.out_feature_indexes  = { 2, 5, 8, 11 };

    DecoderParams dp;
    dp.hidden_dim = 256; dp.dec_layers = 4; dp.sa_nheads = 8; dp.ca_nheads = 16; dp.dec_n_points = 2;
    dp.num_queries = 100; dp.num_classes = 2; dp.gw = 48; dp.gh = 48;

    KeypointParams kp;
    kp.num_kp = 17; kp.sa_nheads = 8; kp.ca_nheads = 16; kp.dec_n_points = 2;
    kp.num_keypoint_classes = 2; kp.active_class_idx = 1;

    const int64_t res = 576;

    std::vector<NpyArray> ref;
    if (!read_ref(ref_path, ref, 5)) { fprintf(stderr, "failed to read %s\n", ref_path); return 1; }
    const NpyArray & pixels_ref = ref[0], & topk_ref = ref[1], & boxes_ref = ref[2],
                   & logits_ref = ref[3], & kpts_ref = ref[4];

    init_graph_ctx(m, 300000);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_set_name(x, "pixel_values");
    ggml_set_input(x);
    ggml_tensor * topk_override = ggml_new_tensor_1d(m.ctx_g, GGML_TYPE_I32, dp.num_queries);
    ggml_set_input(topk_override);

    std::vector<ggml_tensor *> taps = dinov2_backbone(m, x, bp);
    ggml_tensor * fused = projector_p4(m, taps, 256, "projector");
    ggml_tensor * cross_fused = projector_p4(m, taps, 256, "cross_attn_projector");

    auto flatten = [&](ggml_tensor * f) {
        ggml_tensor * t = ggml_reshape_3d(m.ctx_g, f, dp.gw * dp.gh, dp.hidden_dim, 1);
        return ggml_cont(m.ctx_g, ggml_permute(m.ctx_g, t, 1, 0, 2, 3));
    };
    ggml_tensor * memory = flatten(fused);
    ggml_tensor * kp_memory = flatten(cross_fused);

    DecoderOutput out = rfdetr_decoder(m, memory, dp, topk_override, &kp, kp_memory);

    bool ok = compute_cpu_multi(m, { out.pred_boxes, out.pred_logits, out.pred_keypoints }, 300000, [&] {
        whc_to_ggml(pixels_ref, x);
        std::vector<float> prop = output_proposals_data(dp.gw, dp.gh);
        ggml_backend_tensor_set(out.output_proposals, prop.data(), 0, prop.size() * sizeof(float));
        std::vector<int32_t> idx(dp.num_queries);
        for (int i = 0; i < dp.num_queries; i++) idx[i] = (int32_t) topk_ref.data[i];
        ggml_backend_tensor_set(topk_override, idx.data(), 0, idx.size() * sizeof(int32_t));
    });
    if (!ok) return 1;

    int rc = 0;
    printf("pred_boxes:\n");
    if (compare_ref(out.pred_boxes, boxes_ref, 5e-2) != 0) rc = 1;
    printf("pred_logits:\n");
    if (compare_ref(out.pred_logits, logits_ref, 5e-2) != 0) rc = 1;
    printf("pred_keypoints:\n");
    if (compare_ref(out.pred_keypoints, kpts_ref, 5e-2) != 0) rc = 1;
    return rc;
}
