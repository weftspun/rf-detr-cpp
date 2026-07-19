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
