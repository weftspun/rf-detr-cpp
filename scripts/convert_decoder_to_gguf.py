#!/usr/bin/env python3
"""Convert an RF-DETR .pth checkpoint's detection decoder (two-stage proposal
heads, transformer decoder, final box/class heads) into a GGUF file for
src/decoder.cpp. See docs/decisions/decoder.md for the verified key layout.

Strips no fixed prefix (these are mostly top-level keys already); splits
nn.MultiheadAttention's fused in_proj_weight/bias into separate q/k/v
tensors; bakes the DETR sine-embedding scale (2*pi/dim_t, dim=128) as a
constant tensor "decoder.sine_scale" since it isn't learned.

Usage:
    uv run --with torch --with numpy --with gguf scripts/convert_decoder_to_gguf.py \
        models/rf-detr-nano.pth models/rf-detr-nano-decoder.gguf
"""
import sys

import gguf
import numpy as np
import torch

HIDDEN = 256

TOP_LEVEL = ["class_embed", "bbox_embed", "query_feat", "refpoint_embed"]
TRANSFORMER_PREFIXES = [
    "transformer.enc_output.0",
    "transformer.enc_output_norm.0",
    "transformer.enc_out_class_embed.0",
    "transformer.enc_out_bbox_embed.0",
    "transformer.decoder.ref_point_head",
    "transformer.decoder.norm",
]


def dest_name(k: str, dec_layers: int) -> str | None:
    for name in TOP_LEVEL:
        if k == name or k.startswith(name + "."):
            return k  # keep as-is: class_embed.*, bbox_embed.*, query_feat.weight, refpoint_embed.weight
    for pre in TRANSFORMER_PREFIXES:
        if k == pre or k.startswith(pre + "."):
            return k[len("transformer."):]  # enc_output.0.*, enc_out_class_embed.0.*, decoder.ref_point_head.*, decoder.norm.*
    for i in range(dec_layers):
        pre = f"transformer.decoder.layers.{i}."
        if k.startswith(pre) and "self_attn.in_proj" not in k:
            return "decoder.layers." + str(i) + "." + k[len(pre):]
    return None


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <checkpoint.pth> <out.gguf> [dec_layers=2]", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    dec_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    writer = gguf.GGUFWriter(out_path, "rfdetr-decoder")

    n_written = 0
    for k, v in sd.items():
        name = dest_name(k, dec_layers)
        if name is None:
            continue
        arr = v.detach().to(torch.float32).numpy()
        writer.add_tensor(name, np.ascontiguousarray(arr))
        n_written += 1

    for i in range(dec_layers):
        pre = f"transformer.decoder.layers.{i}.self_attn"
        w = sd[f"{pre}.in_proj_weight"].detach().to(torch.float32).numpy()  # (768,256)
        b = sd[f"{pre}.in_proj_bias"].detach().to(torch.float32).numpy()    # (768,)
        for j, part in enumerate(["q", "k", "v"]):
            writer.add_tensor(f"decoder.layers.{i}.self_attn.{part}_proj.weight",
                              np.ascontiguousarray(w[j * HIDDEN:(j + 1) * HIDDEN]))
            writer.add_tensor(f"decoder.layers.{i}.self_attn.{part}_proj.bias",
                              np.ascontiguousarray(b[j * HIDDEN:(j + 1) * HIDDEN]))
            n_written += 1

    dim = 128
    dim_t = 10000.0 ** (2 * (np.arange(dim) // 2) / dim)
    # numpy shape (dim,1) -> gguf writer reverses dims -> ggml ne=(1,dim), the
    # (k=1,n=dim) shape ggml_mul_mat needs for the outer-product trick in
    # sine_embed_1coord (src/decoder.cpp).
    sine_scale = (2 * np.pi / dim_t).astype(np.float32).reshape(dim, 1)
    writer.add_tensor("decoder.sine_scale", sine_scale)
    n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n_written} tensors to {out_path}")


if __name__ == "__main__":
    main()
