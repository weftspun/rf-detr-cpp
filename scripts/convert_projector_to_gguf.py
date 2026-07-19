#!/usr/bin/env python3
"""Convert an RF-DETR .pth checkpoint's projector (rfdetr/models/backbone/
projector.py, MultiScaleProjector) into a GGUF file for src/projector.cpp.

Strips the "backbone.0." prefix, leaving "projector.stages.0.*" -- matches
what projector.cpp expects.

Usage:
    uv run --with torch --with numpy --with gguf scripts/convert_projector_to_gguf.py \
        models/rf-detr-nano.pth models/rf-detr-nano-projector.gguf
"""
import sys

import gguf
import numpy as np
import torch

PREFIX = "backbone.0."


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <checkpoint.pth> <out.gguf>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    writer = gguf.GGUFWriter(out_path, "rfdetr-projector")
    writer.add_uint32("rfdetr.projector.out_channels", 256)

    n_written = 0
    for k, v in sd.items():
        if not (k.startswith(PREFIX + "projector.")):
            continue
        name = k[len(PREFIX):]
        arr = v.detach().to(torch.float32).numpy()
        writer.add_tensor(name, np.ascontiguousarray(arr))
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n_written} tensors to {out_path}")


if __name__ == "__main__":
    main()
