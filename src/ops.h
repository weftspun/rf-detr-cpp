// Shared ggml building blocks for rf-detr-cpp: multi-gguf weight container
// plus the linear/layernorm/conv/attention vocabulary used by every ported
// graph. Token-major activations are (C, T[, N]); spatial activations are
// ggml WHCN, matching the see-through-cpp/trellis2cpp convention.
#pragma once

#include "ggml.h"

#include <map>
#include <string>
#include <vector>

// Weights merged from one or more ggufs (raw PyTorch state-dict names, which
// do not collide across rf-detr-cpp's components: backbone/decoder/heads).
struct Model {
    std::vector<ggml_context *> ctx_w;   // one per gguf
    ggml_context * ctx_g = nullptr;      // graph (no_alloc), set by the caller

    std::map<std::string, ggml_tensor *> weights;
    std::map<std::string, std::string>   config_json;  // per-component KV, if present

    Model() = default;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    ~Model();

    bool load(const char * path);        // merge tensors from a gguf (host RAM)
    bool load_backend(const char * path, struct ggml_backend_buffer_type * buft);
    std::vector<struct ggml_backend_buffer *> bufs;   // owned backend buffers
    ggml_tensor * get(const std::string & name) const;   // exits on miss
    bool has(const std::string & name) const;
};

// conv + bias; pad 0 for 1x1 convs
ggml_tensor * conv2d(Model & m, ggml_tensor * x, const std::string & pre,
                     int stride = 1, int pad = 1);

// LayerNorm over ne[0] + affine, for token-major (C, T[, N]) activations
ggml_tensor * layer_norm_affine(Model & m, ggml_tensor * x, const std::string & pre,
                                float eps = 1e-6f);

// y = W x + b for token-major (C_in, T[, N]) activations (bias optional:
// skipped when "<pre>.bias" is absent)
ggml_tensor * linear(Model & m, ggml_tensor * x, const std::string & pre);

// ConvNeXt-style channels-last LayerNorm for spatial WHCN activations:
// permutes to channel-fastest, ggml_norm over that axis (+ affine), permutes
// back. Distinct from layer_norm_affine, which normalizes token-major ne[0].
ggml_tensor * spatial_layer_norm_affine(Model & m, ggml_tensor * x, const std::string & pre,
                                        float eps = 1e-6f);
