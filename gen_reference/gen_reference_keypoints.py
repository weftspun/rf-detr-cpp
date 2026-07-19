#!/usr/bin/env python3
"""Dump a full end-to-end RFDETRKeypointPreview reference (backbone -> dual
projector -> decoder+GroupPose keypoint stream -> heads) for diffing
against src/keypoints.cpp fed with the C++ backbone/projector/decoder's own
outputs. Same torch.topk spy-hook technique as gen_reference_decoder.py.

num_classes=1 override: the real checkpoint's class_embed is (2,256) i.e.
num_classes+1=2 -- the config class default (90) doesn't match this
specific trained checkpoint (single class: person).

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_keypoints.py models/rf-detr-keypoint-preview.pth \
        gen_reference/reference_keypoints_preview.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.config import RFDETRKeypointPreviewConfig
from rfdetr.models.lwdetr import build_model_from_config


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "models/rf-detr-keypoint-preview.pth"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "gen_reference/reference_keypoints_preview.bin"

    # num_classes=1 (class_embed.weight is (2,256) = num_classes+1 background) and
    # num_keypoints_per_class=[0,17] are the real trained checkpoint's values, not
    # the config class's defaults (num_classes=90, [17]). [0,17] (not [17,0]) is
    # confirmed by the checkpoint's own _kp_active_mask buffer, shape (2,17):
    # row 0 is all-False (inactive), row 1 is all-True (active) -- so the "real"
    # class is index 1, and _format_keypoint_output pads the real 17 keypoints
    # into slots [17:34), zeros into [0:17). Getting this backwards ([17,0])
    # still produces a self-consistent-looking but WRONG reference (the same 17
    # real keypoint values end up at the wrong padding offset, and the
    # class-logit boost lands on class 0 instead of class 1 -- caught only by
    # cross-checking against _kp_active_mask directly, not by the dump alone
    # looking "reasonable").
    cfg = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu", num_classes=1,
                                      num_keypoints_per_class=[0, 17])
    model = build_model_from_config(cfg)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ignore_missing = {"_kp_active_mask"}
    ignore_unexpected_prefixes = ("keypoint_head.keypoint_proj.", "transformer.keypoint_query_initializer_enc.",
                                  "transformer.enc_out_keypoint_embed.")
    real_missing = [k for k in missing if "mask_token" not in k and k not in ignore_missing]
    real_unexpected = [k for k in unexpected if not any(k.startswith(p) for p in ignore_unexpected_prefixes)]
    print(f"missing={real_missing}\nunexpected={real_unexpected}")
    assert not real_missing and not real_unexpected, "state dict mismatch"

    captured = {}
    orig_topk = torch.topk

    def spy_topk(*args, **kwargs):
        result = orig_topk(*args, **kwargs)
        if "topk_idx" not in captured and args and args[0].dim() == 2 and args[0].shape[-1] > 100:
            captured["topk_idx"] = result.indices.clone()
        return result

    torch.topk = spy_topk
    try:
        torch.manual_seed(0)
        res = 576
        pixel_values = torch.randn(1, 3, res, res, dtype=torch.float32)
        with torch.no_grad():
            out = model(pixel_values)
    finally:
        torch.topk = orig_topk

    assert "topk_idx" in captured, "failed to capture topk indices via spy hook"

    pred_keypoints = out["pred_keypoints"]  # (1, num_queries, num_kp, 8) or similar
    print(f"pred_boxes {tuple(out['pred_boxes'].shape)}, pred_logits {tuple(out['pred_logits'].shape)}, "
          f"pred_keypoints {tuple(pred_keypoints.shape)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        write_arr(f, pixel_values.permute(0, 2, 3, 1).squeeze(0).numpy())  # (H,W,3)
        write_arr(f, captured["topk_idx"].squeeze(0).numpy().astype(np.float32))
        write_arr(f, out["pred_boxes"].squeeze(0).numpy())    # (num_queries,4)
        write_arr(f, out["pred_logits"].squeeze(0).numpy())   # (num_queries,2)
        write_arr(f, pred_keypoints.squeeze(0).numpy())        # (num_queries,num_kp,8)
    print(f"wrote 5 arrays to {out_path}")


if __name__ == "__main__":
    main()
