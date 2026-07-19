// Validates src/segmentation.cpp end-to-end (backbone -> projector ->
// decoder -> segmentation head) against the real RFDETRSegMedium
// checkpoint -- same encoder/window pattern as SegSmall, resolution 432
// (grid 36), dec_layers=5, num_queries=200. Same topk-override technique
// as test_segmentation.cpp (see docs/decisions/decoder.md).
#include "backbone.h"
#include "decoder.h"
#include "projector.h"
#include "segmentation.h"
#include "test_common.h"

#include <cstdio>

static void whc_to_ggml(const NpyArray & r, ggml_tensor * dst) {
    int64_t H = r.shape[0], W = r.shape[1], C = r.shape[2];
    std::vector<float> whc(r.data.size());
    for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
            for (int64_t c = 0; c < C; c++) {
                whc[c * H * W + h * W + w] = r.data[h * W * C + w * C + c];
            }
        }
    }
    ggml_backend_tensor_set(dst, whc.data(), 0, whc.size() * sizeof(float));
}

static NpyArray hwc_to_whc_ref(const NpyArray & r) {
    int64_t H = r.shape[0], W = r.shape[1], C = r.shape[2];
    NpyArray out;
    out.shape = { W, H, C, 1 };
    out.data.resize(r.data.size());
    for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
            for (int64_t c = 0; c < C; c++) {
                out.data[c * H * W + h * W + w] = r.data[h * W * C + w * C + c];
            }
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    const char * backbone_gguf  = argc > 1 ? argv[1] : "models/rf-detr-seg-medium-backbone.gguf";
    const char * projector_gguf = argc > 2 ? argv[2] : "models/rf-detr-seg-medium-projector.gguf";
    const char * decoder_gguf   = argc > 3 ? argv[3] : "models/rf-detr-seg-medium-decoder.gguf";
    const char * seg_gguf       = argc > 4 ? argv[4] : "models/rf-detr-seg-medium-segmentation.gguf";
    const char * seg_ref_path   = argc > 5 ? argv[5] : "gen_reference/reference_segmentation_medium.bin";

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
    if (!rfdetr_load(m, seg_gguf)) {
        fprintf(stderr, "failed to load %s\n", seg_gguf);
        return 1;
    }

    BackboneParams bp;
    bp.hidden = 384; bp.n_layer = 12; bp.n_head = 6; bp.patch_size = 12; bp.n_register = 0; bp.num_windows = 2;
    bp.window_block_indexes = { 0, 1, 2, 4, 5, 7, 8, 10, 11 };
    bp.out_feature_indexes  = { 2, 5, 8, 11 };

    DecoderParams dp;
    dp.hidden_dim = 256; dp.dec_layers = 5; dp.sa_nheads = 8; dp.ca_nheads = 16; dp.dec_n_points = 2;
    dp.num_queries = 200; dp.num_classes = 91; dp.gw = 36; dp.gh = 36;

    SegmentationParams sp;
    sp.hidden_dim = 256; sp.num_blocks = 5; sp.downsample_ratio = 4; sp.image_w = 432; sp.image_h = 432;

    const int64_t res = 432;

    std::vector<NpyArray> seg_ref;
    if (!read_ref(seg_ref_path, seg_ref, 5)) {
        fprintf(stderr, "failed to read %s\n", seg_ref_path);
        return 1;
    }
    const NpyArray & pixels_ref = seg_ref[0], & topk_ref = seg_ref[1], & masks_ref = seg_ref[2];
    const NpyArray & boxes_ref = seg_ref[3], & logits_ref = seg_ref[4];

    init_graph_ctx(m, 200000);
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

    DecoderOutput dout = rfdetr_decoder(m, memory, dp, topk_override);
    std::vector<ggml_tensor *> masks = segmentation_head(m, fused, dout.hidden_states, sp);
    ggml_tensor * final_mask = masks.back();

    bool ok = compute_cpu_multi(m, { final_mask, dout.pred_boxes, dout.pred_logits }, 200000, [&] {
        whc_to_ggml(pixels_ref, x);

        std::vector<float> prop = output_proposals_data(dp.gw, dp.gh);
        ggml_backend_tensor_set(dout.output_proposals, prop.data(), 0, prop.size() * sizeof(float));

        std::vector<int32_t> idx(dp.num_queries);
        for (int i = 0; i < dp.num_queries; i++) {
            idx[i] = (int32_t) topk_ref.data[i];
        }
        ggml_backend_tensor_set(topk_override, idx.data(), 0, idx.size() * sizeof(int32_t));
    });
    if (!ok) {
        return 1;
    }

    printf("pred_boxes (sanity check vs test_decoder-style path):\n");
    compare_ref(dout.pred_boxes, boxes_ref, 5e-2);
    printf("pred_logits (sanity check):\n");
    compare_ref(dout.pred_logits, logits_ref, 5e-2);

    printf("pred_masks (last block):\n");
    // Same relaxed gate as test_segmentation.cpp (SegNano) -- see
    // docs/decisions/segmentation.md for why (256-channel dot-product mask
    // head amplifies this port's ~1e-3-level backbone/decoder float drift).
    NpyArray masks_whc = hwc_to_whc_ref(masks_ref);
    return compare_ref(final_mask, masks_whc, 0.15);
}
