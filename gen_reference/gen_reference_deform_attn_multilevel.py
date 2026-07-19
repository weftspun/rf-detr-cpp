#!/usr/bin/env python3
"""Isolated reference for src/deform_attn.cpp's ms_deform_attn_multilevel
(num_levels=2, needed for RFDETRLargeDeprecated's P3+P5 projector_scale) --
same synthetic-weights/inputs approach as gen_reference_deform_attn.py
(num_levels=1), but with two DIFFERENTLY-SIZED levels (5x5 and 3x3) to
stress-test the per-level value slicing and the joint level*point softmax.

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_deform_attn_multilevel.py \
        gen_reference/reference_deform_attn_multilevel.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

D_MODEL, N_HEADS, N_POINTS = 8, 2, 2
LEVELS = [(5, 5), (3, 3)]  # (gw, gh) per level, P3-then-P5 order
N_QUERY = 5


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "gen_reference/reference_deform_attn_multilevel.bin"

    torch.manual_seed(0)
    n_levels = len(LEVELS)
    m = MSDeformAttn(d_model=D_MODEL, n_levels=n_levels, n_heads=N_HEADS, n_points=N_POINTS)
    m.eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3)

    query = torch.randn(1, N_QUERY, D_MODEL) * 0.5
    total_tokens = sum(gw * gh for gw, gh in LEVELS)
    value_input = torch.randn(1, total_tokens, D_MODEL) * 0.5
    cxcy = torch.rand(1, N_QUERY, 2) * 0.6 + 0.2
    wh = torch.rand(1, N_QUERY, 2) * 0.5 + 0.1
    ref_points = torch.cat([cxcy, wh], dim=-1)  # (1,N_QUERY,4)

    spatial_shapes = torch.tensor([[gh, gw] for gw, gh in LEVELS], dtype=torch.long)
    level_start_index = torch.tensor([0] + list(np.cumsum([gw * gh for gw, gh in LEVELS]))[:-1], dtype=torch.long)
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
    print(f"wrote 12 arrays to {out_path}; out shape {tuple(out.shape)}; levels={LEVELS}")


if __name__ == "__main__":
    main()
