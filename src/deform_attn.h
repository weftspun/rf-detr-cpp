// Multi-scale deformable attention (rfdetr/models/ops/modules/ms_deform_attn.py
// + ms_deform_attn_core_pytorch), specialized to num_levels=1 (RFDETRNano's
// single P4 feature level -- see docs/decisions/decoder.md). No ggml
// built-in matches torch.grid_sample, so the bilinear sampling is
// hand-implemented via floor/clamp/get_rows; see deform_attn.cpp.
#pragma once

#include "ops.h"

#include <utility>
#include <vector>

// query: (d_model, n_query, 1) content+pos query. value_input: (d_model,
// gw*gh, 1) token-major flattened feature map (row-major, width-fastest,
// matching the backbone/projector convention) -- value_proj is applied
// inside. ref_points: (4, n_query, 1), (cx,cy,w,h) normalized [0,1].
// Weight prefix (unprefixed): "<pre>.{sampling_offsets,attention_weights,
// value_proj,output_proj}". Returns (d_model, n_query, 1).
ggml_tensor * ms_deform_attn(Model & m, ggml_tensor * query, ggml_tensor * value_input,
                             ggml_tensor * ref_points, const std::string & pre,
                             int n_heads, int n_points, int gw, int gh);

// Multi-level variant (num_levels>1, e.g. RFDETRLargeDeprecated's P3+P5).
// value_input is the FULL concatenated multi-level flattened feature map
// (d_model, sum(gw_l*gh_l), 1), levels in the same order as `levels`
// (matching upstream's spatial_shapes / projector_scale order, e.g.
// [P3,P5]). value_proj is applied once to the whole concatenated tensor
// (mathematically identical to per-level projection, since Linear is
// per-token) and then sliced per level. Sampling-location math is
// IDENTICAL across levels (box-form ref points; no per-level scale
// factor -- verified against ms_deform_attn_core_pytorch), but
// attention_weights' softmax is JOINT over the flattened (n_levels *
// n_points) axis per head, not per-level independently -- see
// docs/decisions/0001-open-work.md's LargeDeprecated note.
ggml_tensor * ms_deform_attn_multilevel(Model & m, ggml_tensor * query, ggml_tensor * value_input,
                                        ggml_tensor * ref_points, const std::string & pre,
                                        int n_heads, int n_points,
                                        const std::vector<std::pair<int, int>> & levels);
