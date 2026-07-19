#!/usr/bin/env python3
"""Isolated reference for src/deform_attn.cpp: uses the REAL upstream
MSDeformAttn module (num_levels=1) with synthetic random weights and inputs,
dumps everything needed to reproduce it in ggml. This validates the
hand-implemented bilinear sampling (no ggml built-in matches grid_sample)
in isolation before it's wired into the full decoder -- see the approved
plan at the time this was written (docs/decisions/decoder.md).

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_deform_attn.py gen_reference/reference_deform_attn.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

D_MODEL, N_HEADS, N_POINTS = 8, 2, 2
GW, GH = 4, 4
N_QUERY = 5


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "gen_reference/reference_deform_attn.bin"

    torch.manual_seed(0)
    m = MSDeformAttn(d_model=D_MODEL, n_levels=1, n_heads=N_HEADS, n_points=N_POINTS)
    m.eval()
    # random-but-fixed weights (module's own init is fine, just re-seed so
    # sampling_offsets.bias's structured init is reproducible too)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3)

    query = torch.randn(1, N_QUERY, D_MODEL) * 0.5
    value_input = torch.randn(1, GW * GH, D_MODEL) * 0.5
    # reference points: cx,cy well inside [0.2,0.8], w,h moderate so offsets
    # can push some samples out of bounds (exercises the zero-padding path)
    cxcy = torch.rand(1, N_QUERY, 2) * 0.6 + 0.2
    wh = torch.rand(1, N_QUERY, 2) * 0.5 + 0.1
    ref_points = torch.cat([cxcy, wh], dim=-1)  # (1,N_QUERY,4)

    spatial_shapes = torch.tensor([[GH, GW]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)
    padding_mask = None

    with torch.no_grad():
        out = m(query, ref_points.unsqueeze(2), value_input, spatial_shapes, level_start_index, padding_mask)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        write_arr(f, m.value_proj.weight.detach().numpy())
        write_arr(f, m.value_proj.bias.detach().numpy())
        write_arr(f, m.sampling_offsets.weight.detach().numpy())
        write_arr(f, m.sampling_offsets.bias.detach().numpy())
        write_arr(f, m.attention_weights.weight.detach().numpy())
        write_arr(f, m.attention_weights.bias.detach().numpy())
        write_arr(f, m.output_proj.weight.detach().numpy())
        write_arr(f, m.output_proj.bias.detach().numpy())
        write_arr(f, query.squeeze(0).numpy())
        write_arr(f, value_input.squeeze(0).numpy())
        write_arr(f, ref_points.squeeze(0).numpy())
        write_arr(f, out.squeeze(0).numpy())
    print(f"wrote 12 arrays to {out_path}; out shape {tuple(out.shape)}")


if __name__ == "__main__":
    main()
