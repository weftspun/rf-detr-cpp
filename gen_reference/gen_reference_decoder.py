#!/usr/bin/env python3
"""Dump a full end-to-end RFDETRNano reference (LWDETR forward: backbone ->
projector -> two-stage proposals -> decoder -> heads) for diffing against
src/decoder.cpp fed with src/backbone.cpp + src/projector.cpp's own outputs.

Dumps torch.topk's exact index selection too, since ggml_top_k's output
order is not guaranteed to match torch.topk's (see
docs/decisions/decoder.md) -- tests/test_decoder.cpp uses these indices as
an override input rather than trusting ggml_top_k's order for this
milestone's numeric validation.

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_decoder.py models/rf-detr-nano.pth \
        gen_reference/reference_decoder_nano.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.config import RFDETRBaseConfig, RFDETRNanoConfig
from rfdetr.models.lwdetr import build_model_from_config

CONFIGS = {"nano": (RFDETRNanoConfig, 384, 300), "base": (RFDETRBaseConfig, 560, 300)}


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "models/rf-detr-nano.pth"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "gen_reference/reference_decoder_nano.bin"
    variant = sys.argv[3] if len(sys.argv) > 3 else "nano"
    config_cls, res, num_queries = CONFIGS[variant]

    cfg = config_cls(pretrain_weights=None, device="cpu")
    model = build_model_from_config(cfg)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "mask_token" not in k and k != "_kp_active_mask"]
    print(f"missing={real_missing}\nunexpected={unexpected}")
    assert not real_missing and not unexpected, "state dict mismatch"

    # capture torch.topk's exact indices (group 0, dec_layers>0 branch) via a hook
    captured = {}
    orig_topk = torch.topk

    def spy_topk(*args, **kwargs):
        result = orig_topk(*args, **kwargs)
        if "topk_idx" not in captured and args and args[0].dim() == 2 and args[0].shape[-1] > num_queries:
            captured["topk_idx"] = result.indices.clone()
        return result

    torch.topk = spy_topk
    try:
        torch.manual_seed(0)
        pixel_values = torch.randn(1, 3, res, res, dtype=torch.float32)
        with torch.no_grad():
            out = model(pixel_values)
    finally:
        torch.topk = orig_topk

    assert "topk_idx" in captured, "failed to capture topk indices via spy hook"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        write_arr(f, captured["topk_idx"].squeeze(0).numpy().astype(np.float32))  # (300,) as f32, cast to i32 on read
        write_arr(f, out["pred_boxes"].squeeze(0).numpy())   # (300,4)
        write_arr(f, out["pred_logits"].squeeze(0).numpy())  # (300,91)
    print(f"wrote 3 arrays to {out_path}")
    print(f"topk_idx[:10] = {captured['topk_idx'][0, :10].tolist()}")
    print(f"pred_boxes shape {tuple(out['pred_boxes'].shape)}, pred_logits shape {tuple(out['pred_logits'].shape)}")


if __name__ == "__main__":
    main()
