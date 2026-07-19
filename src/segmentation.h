// RF-DETR instance segmentation head (rfdetr/models/heads/segmentation.py,
// class SegmentationHead). One DepthwiseConvBlock per decoder layer,
// progressively refining a shared spatial-features map; each block is paired
// with that layer's query features to produce one set of per-query masks via
// a channel dot-product ("einsum bchw,bnc->bnhw"). See
// docs/decisions/segmentation.md.
#pragma once

#include "ops.h"

#include <vector>

struct SegmentationParams {
    int hidden_dim = 256;
    int num_blocks = 4;          // == dec_layers for the paired config
    int downsample_ratio = 4;
    int image_w = 312, image_h = 312; // original input resolution (pre-patch-embed)
};

// spatial_features: (gw, gh, hidden_dim, 1) -- the projector's fused P4 map
// (same one used as decoder memory). hidden_states: per-decoder-layer normed
// hidden states (hidden_dim, num_queries, 1), i.e. DecoderOutput::hidden_states.
// Returns one (target_w, target_h, num_queries, 1) mask-logit map per block,
// in layer order; masks.back() is the final pred_masks equivalent.
std::vector<ggml_tensor *> segmentation_head(Model & m, ggml_tensor * spatial_features,
                                             const std::vector<ggml_tensor *> & hidden_states,
                                             const SegmentationParams & p);
