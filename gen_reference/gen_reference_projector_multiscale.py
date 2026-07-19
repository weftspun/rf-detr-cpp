#!/usr/bin/env python3
"""Isolated reference for src/projector.cpp's projector_multiscale
(scale_factors=[2.0,0.5], i.e. P3+P5, needed for RFDETRLargeDeprecated) --
constructs the REAL upstream MultiScaleProjector directly with small
synthetic random weights/inputs (same technique as
gen_reference_deform_attn.py), so this validates against upstream's actual
module without needing the ~1.6GB LargeDeprecated checkpoint at all.

Dumps every learnable parameter by its real state-dict key (so the C++ test
can load them under the exact same names src/projector.cpp expects) plus 4
synthetic input taps and the resulting fused P3/P5 outputs.

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_projector_multiscale.py \
        gen_reference/reference_projector_multiscale.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.models.backbone.projector import MultiScaleProjector

IN_CH = 8       # per-tap channel count (all 4 taps the same, like the real backbone)
OUT_CH = 6
GW, GH = 6, 6   # P4-equivalent tap spatial size (before per-level resampling)
N_TAPS = 4


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "gen_reference/reference_projector_multiscale.bin"

    torch.manual_seed(0)
    m = MultiScaleProjector(in_channels=[IN_CH] * N_TAPS, out_channels=OUT_CH,
                            scale_factors=[2.0, 0.5], num_blocks=3, layer_norm=True)
    m.eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)

    taps = [torch.randn(1, IN_CH, GH, GW) * 0.5 for _ in range(N_TAPS)]
    with torch.no_grad():
        outs = m(taps)  # list: [P3 (1,OUT_CH,2*GH,2*GW), P5 (1,OUT_CH,GH//2,GW//2)]

    names_and_params = list(m.named_parameters())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", len(names_and_params)))
        for name, p in names_and_params:
            name_b = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_b)))
            f.write(name_b)
            write_arr(f, p.detach().numpy())
        f.write(struct.pack("<i", N_TAPS))
        for t in taps:
            write_arr(f, t.squeeze(0).permute(1, 2, 0).numpy())  # (H,W,C)
        f.write(struct.pack("<i", len(outs)))
        for o in outs:
            write_arr(f, o.squeeze(0).permute(1, 2, 0).numpy())  # (H,W,C)
    print(f"wrote {len(names_and_params)} params, {N_TAPS} taps, {len(outs)} outputs to {out_path}")
    for i, o in enumerate(outs):
        print(f"  out[{i}]: {tuple(o.shape)}")


if __name__ == "__main__":
    main()
