#!/usr/bin/env python3
"""Preprocess a REAL COCO image + its real annotations using upstream's
OWN transform pipeline (rfdetr.datasets.coco.make_coco_transforms,
image_set="val_speed" -- the fixed-square-resize variant matching this
port's fixed-resolution backbone, as opposed to "val"'s aspect-ratio-
preserving resize), for demos/train_step_demo.cpp to train on real data
instead of synthetic noise.

Category IDs are used AS-IS (COCO's raw 1-90 category_id, no remapping) --
checkpoint-verified via rfdetr.datasets.coco.build_coco: remap_category_ids
is only True for keypoint datasets, False (i.e. cat2label=None, so
ConvertCoco appends category_id directly) for plain detection.

Usage:
    uv run --with torch --with numpy --with rfdetr --with pillow --with albumentations \
        gen_reference/gen_reference_real_image.py \
        data/000000289343.jpg data/instances_val2017.json 289343 384 \
        gen_reference/real_image_384.bin
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from PIL import Image
from rfdetr.datasets.coco import ConvertCoco, make_coco_transforms


def write_arr(f, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(struct.pack("<i", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
    f.write(arr.tobytes())


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/000000289343.jpg"
    ann_path = sys.argv[2] if len(sys.argv) > 2 else "data/instances_val2017.json"
    image_id = int(sys.argv[3]) if len(sys.argv) > 3 else 289343
    resolution = int(sys.argv[4]) if len(sys.argv) > 4 else 384
    out_path = sys.argv[5] if len(sys.argv) > 5 else "gen_reference/real_image_384.bin"

    with open(ann_path) as f:
        coco = json.load(f)
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id and not a.get("iscrowd", 0)]
    assert anns, f"no annotations found for image_id {image_id}"

    image = Image.open(img_path).convert("RGB")
    target_raw = {"image_id": image_id, "annotations": anns}

    convert = ConvertCoco()  # cat2label=None -> raw COCO category_id used directly
    image, target = convert(image, target_raw)

    transforms = make_coco_transforms("val_speed", resolution)  # fixed square resize+normalize
    image_t, target_t = transforms(image, target)

    print("image_t shape", tuple(image_t.shape))
    print("boxes (cxcywh, normalized):", target_t["boxes"])
    print("labels:", target_t["labels"])

    boxes = target_t["boxes"].numpy()  # (N,4) cxcywh normalized [0,1]
    labels = target_t["labels"].numpy().astype(np.int64)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        write_arr(f, image_t.permute(1, 2, 0).numpy())  # (H,W,3), WHCN-transposed by the C++ reader
        write_arr(f, boxes)
        write_arr(f, labels.astype(np.float32))  # cast to i32 on the C++ side
    print(f"wrote 3 arrays to {out_path}: image {tuple(image_t.shape)}, {len(labels)} boxes/labels")


if __name__ == "__main__":
    main()
