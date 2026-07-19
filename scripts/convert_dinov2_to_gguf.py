#!/usr/bin/env python3
"""Convert an RF-DETR .pth checkpoint's DINOv2-with-windowed-attention
backbone into a GGUF file for rf-detr-cpp's src/backbone.cpp.

Strips the "backbone.0.encoder.encoder." prefix (see
docs/decisions/backbone-windowing.md for how that prefix was found) so
tensor names land on exactly what backbone.cpp expects: "embeddings.*",
"encoder.layer.{i}.*", "layernorm.*".

Usage:
    uv run --with torch --with numpy --with gguf scripts/convert_dinov2_to_gguf.py \
        models/rf-detr-nano.pth models/rf-detr-nano-backbone.gguf
"""
import sys

import gguf
import numpy as np
import torch

PREFIX = "backbone.0.encoder.encoder."

# Per-variant backbone hyperparameters (see docs/decisions/backbone-windowing.md
# for the out_feature_indexes / window_block_indexes derivation).
VARIANTS = {
    "nano": dict(
        hidden=384, n_layer=12, n_head=6, patch_size=16, n_register=0,
        num_windows=2, ln_eps=1e-6,
        out_feature_indexes_raw=[3, 6, 9, 12],
    ),
}


def derive_indexes(raw):
    window_block_indexes = sorted(set(range(raw[-1] + 1)) - set(raw))
    out_feature_indexes = [x - 1 for x in raw]  # stage number -> 0-based layer-output index
    return window_block_indexes, out_feature_indexes


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <checkpoint.pth> <out.gguf> [--variant nano]", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    variant = "nano"
    cfg = VARIANTS[variant]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    writer = gguf.GGUFWriter(out_path, "rfdetr-backbone")
    writer.add_string("rfdetr.backbone.variant", variant)
    writer.add_uint32("rfdetr.backbone.hidden", cfg["hidden"])
    writer.add_uint32("rfdetr.backbone.n_layer", cfg["n_layer"])
    writer.add_uint32("rfdetr.backbone.n_head", cfg["n_head"])
    writer.add_uint32("rfdetr.backbone.patch_size", cfg["patch_size"])
    writer.add_uint32("rfdetr.backbone.n_register", cfg["n_register"])
    writer.add_uint32("rfdetr.backbone.num_windows", cfg["num_windows"])
    writer.add_float32("rfdetr.backbone.ln_eps", cfg["ln_eps"])

    window_block_indexes, out_feature_indexes = derive_indexes(cfg["out_feature_indexes_raw"])
    writer.add_array("rfdetr.backbone.window_block_indexes", window_block_indexes)
    writer.add_array("rfdetr.backbone.out_feature_indexes", out_feature_indexes)

    n_written = 0
    for k, v in sd.items():
        if not k.startswith(PREFIX):
            continue
        name = k[len(PREFIX):]
        if name == "embeddings.mask_token":
            continue  # inference-only; bool_masked_pos is always None
        arr = v.detach().to(torch.float32).numpy()
        writer.add_tensor(name, np.ascontiguousarray(arr))
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n_written} tensors to {out_path}")
    print(f"window_block_indexes={window_block_indexes}")
    print(f"out_feature_indexes={out_feature_indexes}")


if __name__ == "__main__":
    main()
