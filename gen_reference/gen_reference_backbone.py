#!/usr/bin/env python3
"""Dump a DINOv2-windowed-attention backbone reference (input pixels + tapped
feature maps) from the REAL upstream module, for diffing against
src/backbone.cpp. Binary format matches tests/test_common.h's read_ref:
repeated {i32 ndim, i64 dims[ndim], f32 data (row-major, numpy order)}.

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_backbone.py models/rf-detr-nano.pth \
        gen_reference/reference_backbone_nano.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
)

PREFIX = "backbone.0.encoder.encoder."

NANO = dict(
    hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
    patch_size=16, num_register_tokens=0, num_windows=2,
    image_size=384, resolution=384,
    out_feature_indexes_raw=[3, 6, 9, 12],
)

# RFDETRBaseConfig: patch_size=14 (native), image_size stays DINOv2's native
# 518 (37x37 grid) since positional_encoding_size(37)*patch_size(14)==518, but
# resolution=560 (40x40 grid) triggers real bicubic+antialias interpolation
# inside the model -- see docs/decisions/0002-position-embed-bicubic.md.
BASE = dict(
    hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
    patch_size=14, num_register_tokens=0, num_windows=4,
    image_size=518, resolution=560,
    out_feature_indexes_raw=[2, 5, 8, 11],
)

# RFDETRSmallConfig: same Nano-family backbone, resolution=512 (grid 32).
SMALL = dict(
    hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
    patch_size=16, num_register_tokens=0, num_windows=2,
    image_size=512, resolution=512,
    out_feature_indexes_raw=[3, 6, 9, 12],
)

# RFDETRMediumConfig: same Nano-family backbone, resolution=576 (grid 36).
MEDIUM = dict(
    hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
    patch_size=16, num_register_tokens=0, num_windows=2,
    image_size=576, resolution=576,
    out_feature_indexes_raw=[3, 6, 9, 12],
)

VARIANTS = {"nano": NANO, "base": BASE, "small": SMALL, "medium": MEDIUM}


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "models/rf-detr-nano.pth"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "gen_reference/reference_backbone_nano.bin"
    variant = sys.argv[3] if len(sys.argv) > 3 else "nano"
    cfg = VARIANTS[variant]

    raw = cfg["out_feature_indexes_raw"]
    window_block_indexes = sorted(set(range(raw[-1] + 1)) - set(raw))

    hf_cfg = WindowedDinov2WithRegistersConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        patch_size=cfg["patch_size"],
        num_register_tokens=cfg["num_register_tokens"],
        num_windows=cfg["num_windows"],
        image_size=cfg["image_size"],
        out_features=[f"stage{i}" for i in raw],
        window_block_indexes=window_block_indexes,
        return_dict=False,
    )
    model = WindowedDinov2WithRegistersBackbone(hf_cfg)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    own_sd = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX) and k != PREFIX + "embeddings.mask_token"}
    missing, unexpected = model.load_state_dict(own_sd, strict=False)
    print(f"missing={missing}\nunexpected={unexpected}")
    # mask_token is unused at inference (bool_masked_pos is always None) and
    # intentionally not written to the GGUF either -- see convert_dinov2_to_gguf.py.
    real_missing = [k for k in missing if k != "embeddings.mask_token"]
    assert not real_missing and not unexpected, "state dict mismatch -- see docs/decisions/backbone-windowing.md"

    torch.manual_seed(0)
    res = cfg["resolution"]
    pixel_values = torch.randn(1, 3, res, res, dtype=torch.float32)

    with torch.no_grad():
        out = model(pixel_values, return_dict=True)
    feature_maps = out.feature_maps  # tuple, one (1,C,H,W) per out_feature, in raw out_features order

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        write_arr(f, pixel_values.permute(0, 2, 3, 1).squeeze(0).numpy())  # (H,W,C) -- caller transposes to WHCN
        for fm in feature_maps:
            write_arr(f, fm.squeeze(0).permute(1, 2, 0).numpy())  # (H,W,C)
    print(f"wrote {1 + len(feature_maps)} arrays to {out_path}")
    for i, fm in enumerate(feature_maps):
        print(f"  feature_maps[{i}]: {tuple(fm.shape)}")


if __name__ == "__main__":
    main()
