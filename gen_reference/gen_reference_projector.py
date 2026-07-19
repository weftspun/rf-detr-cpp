#!/usr/bin/env python3
"""Dump a projector reference (backbone taps + fused P4 feature map) from the
REAL upstream Backbone module (encoder + projector), for diffing against
src/projector.cpp fed with src/backbone.cpp's own taps.

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_projector.py models/rf-detr-nano.pth \
        gen_reference/reference_projector_nano.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.utilities.tensors import NestedTensor

PREFIX = "backbone.0."


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "models/rf-detr-nano.pth"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "gen_reference/reference_projector_nano.bin"

    model = Backbone(
        name="dinov2_windowed_small",
        out_feature_indexes=[3, 6, 9, 12],
        projector_scale=["P4"],
        target_shape=(384, 384),
        layer_norm=True,
        load_dinov2_weights=False,
        patch_size=16,
        num_windows=2,
        positional_encoding_size=24,
    )
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    own_sd = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX) and k != PREFIX + "encoder.encoder.embeddings.mask_token"}
    missing, unexpected = model.load_state_dict(own_sd, strict=False)
    real_missing = [k for k in missing if k != "encoder.encoder.embeddings.mask_token"]
    print(f"missing={missing}\nunexpected={unexpected}")
    assert not real_missing and not unexpected, "state dict mismatch"

    torch.manual_seed(0)
    res = 384
    pixel_values = torch.randn(1, 3, res, res, dtype=torch.float32)
    mask = torch.zeros(1, res, res, dtype=torch.bool)  # no padding
    nt = NestedTensor(pixel_values, mask)

    with torch.no_grad():
        raw_feats = model.encoder(pixel_values)
        (feat_nested,), _ = model(nt)
    fused = feat_nested.tensors  # (1,256,H,W)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        for feat in raw_feats:
            write_arr(f, feat.squeeze(0).permute(1, 2, 0).numpy())  # (H,W,C)
        write_arr(f, fused.squeeze(0).permute(1, 2, 0).numpy())     # (H,W,256)
    print(f"wrote {len(raw_feats) + 1} arrays to {out_path}")
    print(f"fused: {tuple(fused.shape)}")


if __name__ == "__main__":
    main()
