#!/usr/bin/env python3
"""Preprocess a whole directory of real COCO images + their real annotations
into a directory of per-image binary dumps, generalizing
gen_reference_real_image.py (which only handled one hardcoded image) into a
real dataloader dataset for src/dataset.{h,cpp} to iterate over.

Same transform pipeline as gen_reference_real_image.py (upstream's own
rfdetr.datasets.coco.make_coco_transforms, image_set="val_speed" -- the
fixed-square-resize variant matching this port's fixed-resolution backbone),
same per-file format (write_arr'd image/boxes/labels), plus a manifest.txt
listing one relative filename per line so the C++ loader doesn't need to
glob a directory (portable across filesystems, deterministic order).

Skips any image with zero non-crowd annotations after filtering (nothing
for the loss to match against).

Usage:
    uv run --with torch --with numpy --with rfdetr --with pillow --with albumentations \
        gen_reference/gen_coco_dataset.py \
        data/val2017_sample data/instances_val2017.json 384 \
        gen_reference/coco_sample_384
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
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "data/val2017_sample"
    ann_path = sys.argv[2] if len(sys.argv) > 2 else "data/instances_val2017.json"
    resolution = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "gen_reference/coco_sample_384"

    with open(ann_path) as f:
        coco = json.load(f)
    by_filename = {im["file_name"]: im for im in coco["images"]}
    anns_by_image = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        anns_by_image.setdefault(a["image_id"], []).append(a)

    convert = ConvertCoco()  # cat2label=None -> raw COCO category_id used directly
    transforms = make_coco_transforms("val_speed", resolution)

    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    filenames = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    for fname in filenames:
        im_meta = by_filename.get(fname)
        if im_meta is None:
            print(f"skip {fname}: not found in {ann_path}")
            continue
        image_id = im_meta["id"]
        anns = anns_by_image.get(image_id, [])
        if not anns:
            print(f"skip {fname} (image_id {image_id}): no non-crowd annotations")
            continue

        image = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        target_raw = {"image_id": image_id, "annotations": anns}
        image, target = convert(image, target_raw)
        image_t, target_t = transforms(image, target)

        boxes = target_t["boxes"].numpy()  # (N,4) cxcywh normalized [0,1]
        labels = target_t["labels"].numpy().astype(np.int64)

        out_name = f"{image_id:012d}.bin"
        with open(os.path.join(out_dir, out_name), "wb") as f:
            write_arr(f, image_t.permute(1, 2, 0).numpy())  # (H,W,3)
            write_arr(f, boxes)
            write_arr(f, labels.astype(np.float32))
        manifest.append(out_name)
        print(f"{fname} (image_id {image_id}): {len(labels)} boxes/labels -> {out_name}")

    manifest_path = os.path.join(out_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"wrote {len(manifest)} dataset entries to {out_dir} (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
