// RF-DETR projector: fuses the backbone's tapped feature maps into one
// multi-scale-ready map per decoder feature level (rfdetr/models/backbone/
// projector.py, class MultiScaleProjector). For RFDETRNano, projector_scale
// = ["P4"] -> exactly one output level, one stage, and stages_sampling is
// the identity (all taps already share the same H,W) -- see
// docs/decisions/decoder.md.
#pragma once

#include "ops.h"

#include <vector>

// taps: the backbone's tapped (gw,gh,C,1) feature maps, channel-concatenated
// then run through C2f(len(taps)*C -> out_channels, n=3) + a final spatial
// LayerNorm. Weight prefix (unprefixed, GGUF-stripped): "<prefix>.stages.0.*"
// -- "projector" for the main projector (the default), "cross_attn_projector"
// for the dual/keypoint-only second projector (docs/decisions/keypoints.md).
ggml_tensor * projector_p4(Model & m, const std::vector<ggml_tensor *> & taps, int out_channels = 256,
                           const std::string & prefix = "projector");

// Multi-scale variant (num_feature_levels>1, e.g. RFDETRLargeDeprecated's
// projector_scale=["P3","P5"]). Each scale_factors[lvl] first resamples
// EVERY tap independently via "<prefix>.stages_sampling.<lvl>.<tap_idx>"
// before the same per-level C2f+LayerNorm fusion projector_p4 uses
// ("<prefix>.stages.<lvl>.0"/".1"): scale>1 (upsample, e.g. P3=2.0) uses a
// bias-optional ConvTranspose2d(in_ch -> in_ch/scale, k=s=(int)scale), no
// norm/act; scale<1 (downsample, e.g. P5=0.5) uses the same ConvX
// (conv+spatialLN+SiLU) as C2f's own blocks, k=3, s=(int)(1/scale),
// channels UNCHANGED (matches upstream's `in_channel // max(1,scale)`
// channel-count formula). scale==1 is not expected here -- use
// projector_p4 for the single-P4-level case instead. Returns one fused
// (gw_lvl,gh_lvl,out_channels,1) map per entry in scale_factors, in the
// same order (matches upstream's spatial_shapes/projector_scale order).
std::vector<ggml_tensor *> projector_multiscale(Model & m, const std::vector<ggml_tensor *> & taps,
                                                const std::vector<float> & scale_factors,
                                                int out_channels = 256,
                                                const std::string & prefix = "projector");
