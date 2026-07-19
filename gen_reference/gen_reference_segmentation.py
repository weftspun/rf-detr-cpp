#!/usr/bin/env python3
"""Dump a full end-to-end RFDETRSegNano reference (backbone -> projector ->
decoder -> segmentation head) for diffing against src/segmentation.cpp fed
with the C++ backbone/projector/decoder's own outputs. Same torch.topk
spy-hook technique as gen_reference_decoder.py, since ggml_top_k's output
order doesn't match torch.topk's.

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_segmentation.py models/rf-detr-seg-nano.pt \
        gen_reference/reference_segmentation_nano.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.config import (
    RFDETRSegLargeConfig, RFDETRSegMediumConfig, RFDETRSegNanoConfig, RFDETRSegPreviewConfig,
    RFDETRSegSmallConfig, RFDETRSegXLargeConfig, RFDETRSeg2XLargeConfig,
)
from rfdetr.models.lwdetr import build_model_from_config

CONFIGS = {
    "seg-nano": (RFDETRSegNanoConfig, 312, 100),
    "seg-small": (RFDETRSegSmallConfig, 384, 100),
    "seg-medium": (RFDETRSegMediumConfig, 432, 200),
    "seg-preview": (RFDETRSegPreviewConfig, 432, 200),
    # RFDETRSegLargeConfig's own num_queries default (200) does NOT match
    # the actual published rf-detr-seg-l-ft.pth checkpoint, which has
    # refpoint_embed.weight/query_feat.weight shaped for 300 queries
    # (3900 = 300*13 group_detr rows) -- checkpoint-verified via a direct
    # torch.load state-dict shape check after the config-default 200
    # produced a load_state_dict size-mismatch error. Passed as an
    # explicit override below, not the config class's own default.
    "seg-large": (RFDETRSegLargeConfig, 504, 300),
    "seg-xlarge": (RFDETRSegXLargeConfig, 624, 300),
    "seg-2xlarge": (RFDETRSeg2XLargeConfig, 768, 300),
}


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "models/rf-detr-seg-nano.pt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "gen_reference/reference_segmentation_nano.bin"
    variant = sys.argv[3] if len(sys.argv) > 3 else "seg-nano"
    config_cls, res, num_queries = CONFIGS[variant]

    cfg = config_cls(pretrain_weights=None, device="cpu", num_queries=num_queries)
    model = build_model_from_config(cfg)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "mask_token" not in k and k != "_kp_active_mask"]
    print(f"missing={real_missing}\nunexpected={unexpected}")
    assert not real_missing and not unexpected, "state dict mismatch"

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

    pred_masks = out["pred_masks"]  # (1, num_queries, H', W')
    print(f"pred_boxes {tuple(out['pred_boxes'].shape)}, pred_masks {tuple(pred_masks.shape)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        write_arr(f, pixel_values.permute(0, 2, 3, 1).squeeze(0).numpy())  # (H,W,3), WHCN-transposed by the reader
        write_arr(f, captured["topk_idx"].squeeze(0).numpy().astype(np.float32))
        write_arr(f, pred_masks.squeeze(0).permute(1, 2, 0).numpy())  # (H',W',num_queries)
        write_arr(f, out["pred_boxes"].squeeze(0).numpy())   # (num_queries,4) -- cross-check vs test_decoder-style path
        write_arr(f, out["pred_logits"].squeeze(0).numpy())  # (num_queries,91)
    print(f"wrote 5 arrays to {out_path}")


if __name__ == "__main__":
    main()
