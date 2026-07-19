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
    # RFDETRSmallConfig: same Nano-family backbone/window config, just a
    # larger resolution (512 -> grid 32, still patch_size==16 so
    # runtime_grid==native_grid, no bicubic interpolation needed).
    "small": dict(
        hidden=384, n_layer=12, n_head=6, patch_size=16, n_register=0,
        num_windows=2, ln_eps=1e-6,
        out_feature_indexes_raw=[3, 6, 9, 12],
    ),
    # RFDETRMediumConfig: same pattern again, resolution 576 (grid 36).
    "medium": dict(
        hidden=384, n_layer=12, n_head=6, patch_size=16, n_register=0,
        num_windows=2, ln_eps=1e-6,
        out_feature_indexes_raw=[3, 6, 9, 12],
    ),
    # RFDETRLargeConfig (current, non-deprecated): same pattern again,
    # resolution 704 (grid 44). Distinct from RFDETRLargeDeprecatedConfig,
    # which is patch_size=14/768-wide/2-feature-level -- not this one.
    "large": dict(
        hidden=384, n_layer=12, n_head=6, patch_size=16, n_register=0,
        num_windows=2, ln_eps=1e-6,
        out_feature_indexes_raw=[3, 6, 9, 12],
    ),
    # RFDETRSegNanoConfig: same Small/no-registers encoder, but num_windows=1
    # (windowed vs global attention become numerically identical -- a useful
    # sanity check that both code paths in backbone.cpp agree), patch_size=12,
    # resolution=312 (gw=gh=26).
    "seg-nano": dict(
        hidden=384, n_layer=12, n_head=6, patch_size=12, n_register=0,
        num_windows=1, ln_eps=1e-6,
        out_feature_indexes_raw=[3, 6, 9, 12],
    ),
    # RFDETRSegSmallConfig: same encoder family as seg-nano, but
    # num_windows=2 (checkpoint-verified via RFDETRSegSmallConfig's own
    # fields, uv run --with rfdetr), resolution=384 (gw=gh=32).
    "seg-small": dict(
        hidden=384, n_layer=12, n_head=6, patch_size=12, n_register=0,
        num_windows=2, ln_eps=1e-6,
        out_feature_indexes_raw=[3, 6, 9, 12],
    ),
    # RFDETRKeypointPreviewConfig: patch_size=12, num_windows=2, resolution=576 (gw=gh=48)
    "keypoint-preview": dict(
        hidden=384, n_layer=12, n_head=6, patch_size=12, n_register=0,
        num_windows=2, ln_eps=1e-6,
        out_feature_indexes_raw=[3, 6, 9, 12],
    ),
    # RFDETRBaseConfig: patch_size=14 (native DINOv2), num_windows=4,
    # resolution=560, positional_encoding_size=37 (37*14=518=DINOv2's native
    # training resolution) -- runtime grid gw=gh=560/14=40 != native grid 37,
    # so position embeddings need bicubic+antialias interpolation at
    # inference (see docs/decisions/backbone-windowing.md and
    # 0002-position-embed-bicubic.md). native_grid triggers baking the
    # resize weight matrix below.
    "base": dict(
        hidden=384, n_layer=12, n_head=6, patch_size=14, n_register=0,
        num_windows=4, ln_eps=1e-6,
        out_feature_indexes_raw=[2, 5, 8, 11],
        native_grid=37, runtime_grid=40,
    ),
    # RFDETRLargeDeprecatedConfig: dinov2_windowed_base (NOT _small like
    # every other variant) -- hidden=768, n_head=12, same patch_size==14/
    # num_windows=4/out_feature_indexes_raw/native+runtime grid as "base"
    # (checkpoint-verified via rfdetr.models.backbone.dinov2.get_config("base")
    # and Backbone.__init__'s DinoV2(size=...) resolution, not assumed).
    # Distinct from every other size (Nano/Small/Medium/Base/Large) in that
    # the DECODER hidden_dim is also 384 (not the usual 256) and
    # projector_scale=["P3","P5"] (num_feature_levels=2) -- see
    # docs/decisions/0001-open-work.md.
    "large-deprecated": dict(
        hidden=768, n_layer=12, n_head=12, patch_size=14, n_register=0,
        num_windows=4, ln_eps=1e-6,
        out_feature_indexes_raw=[2, 5, 8, 11],
        native_grid=37, runtime_grid=40,
    ),
}


def derive_indexes(raw):
    window_block_indexes = sorted(set(range(raw[-1] + 1)) - set(raw))
    out_feature_indexes = [x - 1 for x in raw]  # stage number -> 0-based layer-output index
    return window_block_indexes, out_feature_indexes


def extract_bicubic_aa_resize_matrix(in_size, out_size):
    """(out_size, in_size) separable resize matrix reproducing PyTorch's
    F.interpolate(mode='bicubic', antialias=True, align_corners=False)
    EXACTLY (verified to 4.8e-7 against the real 2D op) -- probed directly
    from the real implementation (impulse response per input position, with
    the orthogonal axis held constant so a single probe isolates one axis)
    rather than hand-transcribing PyTorch's internal antialias-kernel
    formula, to avoid a transcription bug. See
    docs/decisions/0002-position-embed-bicubic.md."""
    eye = torch.eye(in_size).unsqueeze(1).unsqueeze(-1).expand(in_size, 1, in_size, in_size).contiguous()
    out = torch.nn.functional.interpolate(eye, size=(out_size, out_size), mode="bicubic",
                                          align_corners=False, antialias=True)
    return out[:, 0, :, 0].T.contiguous().numpy()  # (out_size, in_size)


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <checkpoint.pth> <out.gguf> [variant=nano]", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else "nano"
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
    if "native_grid" in cfg:
        native_grid, runtime_grid = cfg["native_grid"], cfg["runtime_grid"]
        writer.add_uint32("rfdetr.backbone.native_grid", native_grid)
        writer.add_uint32("rfdetr.backbone.runtime_grid", runtime_grid)
        resize_w = extract_bicubic_aa_resize_matrix(native_grid, runtime_grid)  # (runtime_grid, native_grid)
        # numpy shape (runtime_grid, native_grid) -> gguf reverses -> ggml
        # ne=(native_grid, runtime_grid), the (k=native_grid, n=runtime_grid)
        # shape ggml_mul_mat needs directly (see backbone.cpp's pos-embed
        # resize: out[o, batch] = sum_i resize_w[i, o] * in[i, batch]).
        writer.add_tensor("embeddings.pos_resize", resize_w.astype(np.float32))
        n_written += 1
        print(f"baked pos_resize matrix ({native_grid}->{runtime_grid}), row sums "
              f"[{resize_w.sum(axis=1).min():.6f}, {resize_w.sum(axis=1).max():.6f}]")

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
