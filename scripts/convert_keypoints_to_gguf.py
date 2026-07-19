#!/usr/bin/env python3
"""Convert an RF-DETR-Keypoint .pth checkpoint's GroupPose keypoint head
into a GGUF file for src/keypoints.cpp: ConditionalQueryInitializer
(decoder-level), the per-layer keypoint sublayer, keypoint_pos_embed,
keypoint_embed. See docs/decisions/keypoints.md for the verified key
layout. Splits kp_inst_self_attn's fused in_proj_weight/bias into q/k/v,
same as the main decoder self-attn.

Ignores the dead legacy `keypoint_head.keypoint_proj.*` weight set
(confirmed unused upstream, see keypoints.md).

Usage:
    uv run --with torch --with numpy --with gguf scripts/convert_keypoints_to_gguf.py \
        models/rf-detr-keypoint-preview.pth models/rf-detr-keypoint-preview-keypoints.gguf
"""
import sys

import gguf
import numpy as np
import torch

HIDDEN = 256

TOP_LEVEL = ["keypoint_embed"]
DECODER_LEVEL = ["transformer.decoder.keypoint_pos_embed"]
QUERY_INIT_PREFIX = "transformer.keypoint_query_initializer"

# per-layer keypoint sublayer key fragments (besides kp_inst_self_attn's
# fused in_proj, handled separately below)
PER_LAYER_FRAGMENTS = [
    "instance_kp_layer_scale", "kp_inst_self_attn.out_proj", "kp_inst_norm", "kp_norm",
    "kp_cross_attn.sampling_offsets", "kp_cross_attn.attention_weights",
    "kp_cross_attn.value_proj", "kp_cross_attn.output_proj", "kp_cross_attn_norm",
    "kp_linear1", "kp_linear3", "kp_norm5",
]


def dest_name(k, dec_layers):
    for name in TOP_LEVEL:
        if k == name or k.startswith(name + "."):
            return k
    for name in DECODER_LEVEL:
        if k == name:
            return k[len("transformer."):]
    if k.startswith(QUERY_INIT_PREFIX + "."):
        return "keypoint_query_initializer" + k[len(QUERY_INIT_PREFIX):]
    for i in range(dec_layers):
        pre = f"transformer.decoder.layers.{i}."
        if not k.startswith(pre):
            continue
        rest = k[len(pre):]
        if any(rest.startswith(frag) for frag in PER_LAYER_FRAGMENTS):
            return f"decoder.layers.{i}." + rest
    return None


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <checkpoint.pth> <out.gguf> [dec_layers=4]", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    dec_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    writer = gguf.GGUFWriter(out_path, "rfdetr-keypoints")

    n_written = 0
    for k, v in sd.items():
        name = dest_name(k, dec_layers)
        if name is None:
            continue
        arr = v.detach().to(torch.float32).numpy()
        writer.add_tensor(name, np.ascontiguousarray(arr))
        n_written += 1

    for i in range(dec_layers):
        pre = f"transformer.decoder.layers.{i}.kp_inst_self_attn"
        w = sd[f"{pre}.in_proj_weight"].detach().to(torch.float32).numpy()  # (768,256)
        b = sd[f"{pre}.in_proj_bias"].detach().to(torch.float32).numpy()    # (768,)
        for j, part in enumerate(["q", "k", "v"]):
            writer.add_tensor(f"decoder.layers.{i}.kp_inst_self_attn.{part}_proj.weight",
                              np.ascontiguousarray(w[j * HIDDEN:(j + 1) * HIDDEN]))
            writer.add_tensor(f"decoder.layers.{i}.kp_inst_self_attn.{part}_proj.bias",
                              np.ascontiguousarray(b[j * HIDDEN:(j + 1) * HIDDEN]))
            n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {n_written} tensors to {out_path}")


if __name__ == "__main__":
    main()
