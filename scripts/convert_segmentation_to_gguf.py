#!/usr/bin/env python3
"""Convert an RF-DETR-Seg .pth checkpoint's segmentation head
(rfdetr/models/heads/segmentation.py, class SegmentationHead) into a GGUF
file for src/segmentation.cpp. Top-level "segmentation_head.*" keys, no
prefix stripping needed.

Usage:
    uv run --with torch --with numpy --with gguf scripts/convert_segmentation_to_gguf.py \
        models/rf-detr-seg-nano.pt models/rf-detr-seg-nano-segmentation.gguf
"""
import sys

import gguf
import numpy as np
import torch

PREFIX = "segmentation_head."


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <checkpoint.pth> <out.gguf>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    writer = gguf.GGUFWriter(out_path, "rfdetr-segmentation")

    n_written = 0
    for k, v in sd.items():
        if not k.startswith(PREFIX):
            continue
        arr = v.detach().to(torch.float32).numpy()
        writer.add_tensor(k, np.ascontiguousarray(arr))
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n_written} tensors to {out_path}")


if __name__ == "__main__":
    main()
