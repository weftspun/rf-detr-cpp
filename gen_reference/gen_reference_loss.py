#!/usr/bin/env python3
"""Isolated reference for src/loss.cpp (hungarian_match + detection_loss)
against the REAL upstream HungarianMatcher + SetCriterion on synthetic
pred_logits/pred_boxes/targets -- same isolation-before-wiring-in
discipline as every other milestone in this project.

Uses ia_bce_loss=False (plain sigmoid focal loss) to match this port's
chosen simpler formula -- see docs/decisions/0004-loss.md for why the
IA-BCE default wasn't reproduced.

Usage:
    uv run --with torch --with numpy --with rfdetr \
        gen_reference/gen_reference_loss.py gen_reference/reference_loss.bin
"""
import os
import struct
import sys

import numpy as np
import torch
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.models.criterion import SetCriterion

NUM_QUERIES, NUM_CLASSES, NUM_TARGETS = 20, 8, 3


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "gen_reference/reference_loss.bin"

    torch.manual_seed(0)
    pred_logits = torch.randn(1, NUM_QUERIES, NUM_CLASSES) * 1.5
    pred_boxes = torch.rand(1, NUM_QUERIES, 4) * 0.6 + 0.2  # cx,cy,w,h in [0.2,0.8]
    pred_boxes[..., 2:] = torch.rand(1, NUM_QUERIES, 2) * 0.3 + 0.05  # w,h in [0.05,0.35]

    tgt_labels = torch.randint(0, NUM_CLASSES, (NUM_TARGETS,))
    tgt_boxes = torch.rand(NUM_TARGETS, 4) * 0.6 + 0.2
    tgt_boxes[:, 2:] = torch.rand(NUM_TARGETS, 2) * 0.3 + 0.05
    targets = [{"labels": tgt_labels, "boxes": tgt_boxes}]

    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=0.25)
    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    with torch.no_grad():
        indices = matcher(outputs, targets)
    query_idx, tgt_idx = indices[0]
    print("matched query_idx:", query_idx.tolist())
    print("matched tgt_idx:", tgt_idx.tolist())

    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        ia_bce_loss=False,
        group_detr=1,
    )
    pred_logits.requires_grad_(True)
    pred_boxes.requires_grad_(True)
    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    loss_dict = criterion(outputs, targets)
    total_loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)
    print("loss_dict:", {k: v.item() for k, v in loss_dict.items()})
    print("total_loss:", total_loss.item())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        write_arr(f, pred_logits.detach().squeeze(0).numpy())
        write_arr(f, pred_boxes.detach().squeeze(0).numpy())
        write_arr(f, tgt_labels.numpy().astype(np.float32))
        write_arr(f, tgt_boxes.numpy())
        write_arr(f, query_idx.numpy().astype(np.float32))
        write_arr(f, tgt_idx.numpy().astype(np.float32))
        write_arr(f, np.array([total_loss.item()], dtype=np.float32))
    print(f"wrote 7 arrays to {out_path}")


if __name__ == "__main__":
    main()
