// RF-DETR backbone: DINOv2 with windowed attention
// (rfdetr/models/backbone/dinov2_with_windowed_attn.py upstream).
//
// Verified against real upstream checkpoints (see
// docs/decisions/backbone-windowing.md). Single-image inference only (N=1).
#pragma once

#include "ops.h"

#include <vector>

struct BackboneParams {
    int hidden      = 384;   // hidden_size
    int n_layer     = 12;    // num_hidden_layers
    int n_head      = 6;     // num_attention_heads
    int patch_size  = 16;
    int n_register  = 4;     // 0 for the register-free variant
    int num_windows = 2;     // spatial grid divided into num_windows^2 windows
    float ln_eps    = 1e-6f;
    // window_block_indexes is derived upstream, not a free parameter:
    // range(out_feature_indexes.back()+1) minus out_feature_indexes. Pass it
    // precomputed; out_feature_indexes are the (0-based) encoder layer
    // indices to tap (always run global/full attention, per the derivation).
    std::vector<int> window_block_indexes;
    std::vector<int> out_feature_indexes;
    // Position-embedding bicubic interpolation (interpolate_pos_encoding,
    // needed only when patch_size==14 and the runtime resolution differs
    // from the checkpoint's native training grid -- see
    // docs/decisions/backbone-windowing.md and 0002-position-embed-bicubic.md).
    // 0 (default) = no interpolation, position_embeddings used as-is
    // (correct for every patch_size!=14 variant validated so far).
    int native_grid = 0;
};

// x: image pixels (W, H, 3, 1), already resized/normalized per the DINOv2
// preprocessor. W and H must be divisible by patch_size * num_windows.
// Returns one (patch_w, patch_h, hidden, 1) WHCN feature map per entry in
// params.out_feature_indexes, in that order (CLS/register tokens dropped,
// final backbone layernorm applied per upstream).
std::vector<ggml_tensor *> dinov2_backbone(Model & m, ggml_tensor * x, const BackboneParams & params);
