// Multi-scale deformable attention (rfdetr/models/ops/modules/ms_deform_attn.py
// + ms_deform_attn_core_pytorch), specialized to num_levels=1 (RFDETRNano's
// single P4 feature level -- see docs/decisions/decoder.md). No ggml
// built-in matches torch.grid_sample, so the bilinear sampling is
// hand-implemented via floor/clamp/get_rows; see deform_attn.cpp.
#pragma once

#include "ops.h"

// query: (d_model, n_query, 1) content+pos query. value_input: (d_model,
// gw*gh, 1) token-major flattened feature map (row-major, width-fastest,
// matching the backbone/projector convention) -- value_proj is applied
// inside. ref_points: (4, n_query, 1), (cx,cy,w,h) normalized [0,1].
// Weight prefix (unprefixed): "<pre>.{sampling_offsets,attention_weights,
// value_proj,output_proj}". Returns (d_model, n_query, 1).
ggml_tensor * ms_deform_attn(Model & m, ggml_tensor * query, ggml_tensor * value_input,
                             ggml_tensor * ref_points, const std::string & pre,
                             int n_heads, int n_points, int gw, int gh);
