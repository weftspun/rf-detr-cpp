#include "backbone.h"

#include <algorithm>
#include <cmath>

// Multi-head self-attention over a token-major (C, T, N) tensor, N is
// whatever batch the caller has arranged (per-window batch for windowed
// layers, or a single merged sequence for global layers). Weight prefix
// follows the verified HF Dinov2WithRegisters layout (see
// docs/decisions/backbone-windowing.md).
static ggml_tensor * dinov2_self_attn(Model & m, ggml_tensor * x, const std::string & pre, int n_head) {
    ggml_context * ctx = m.ctx_g;
    const int64_t C = x->ne[0], T = x->ne[1], N = x->ne[2];
    const int64_t hd = C / n_head;

    ggml_tensor * q = linear(m, x, pre + "attention.attention.query");
    ggml_tensor * k = linear(m, x, pre + "attention.attention.key");
    ggml_tensor * v = linear(m, x, pre + "attention.attention.value");

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, n_head, T, N), 0, 2, 1, 3)); // (hd,T,H,N)
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, n_head, T, N), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, n_head, T, N), 1, 2, 0, 3)); // (T,hd,H,N)

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                       // (T,T,H,N)
    kq = ggml_soft_max(ctx, ggml_scale(ctx, kq, 1.0f / sqrtf((float) hd)));
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                     // (hd,T,H,N)
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));         // (hd,H,T,N)
    kqv = ggml_reshape_3d(ctx, kqv, C, T, N);

    return linear(m, kqv, pre + "attention.output.dense");
}

// One WindowedDinov2WithRegistersLayer block: pre-LN attention (LayerScale +
// residual), pre-LN MLP (LayerScale + residual). x is token-major (C, T, N);
// N is treated purely as an independent batch by ordinary self-attention,
// which is correct for BOTH windowed layers (N = num_windows^2, one window
// per batch entry) and global layers (N = 1, all windows merged into T).
static ggml_tensor * dinov2_block(Model & m, ggml_tensor * x, const std::string & pre,
                                  int n_head, float ln_eps) {
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * h = layer_norm_affine(m, x, pre + "norm1", ln_eps);
    h = dinov2_self_attn(m, h, pre, n_head);
    h = ggml_mul(ctx, h, m.get(pre + "layer_scale1.lambda1"));
    x = ggml_add(ctx, x, h);

    h = layer_norm_affine(m, x, pre + "norm2", ln_eps);
    h = linear(m, h, pre + "mlp.fc1");
    h = ggml_gelu_erf(ctx, h);
    h = linear(m, h, pre + "mlp.fc2");
    h = ggml_mul(ctx, h, m.get(pre + "layer_scale2.lambda1"));
    return ggml_add(ctx, x, h);
}

// Extract window (wx,wy) of size (ws,ws) from a (gw,gh,C,1) spatial grid and
// return it token-major as (C, ws*ws, 1). Token order within the window is
// row-major (matches PyTorch's flatten(2) on a (C,H,W) conv output).
static ggml_tensor * window_extract(Model & m, ggml_tensor * grid, int wx, int wy, int ws) {
    ggml_context * ctx = m.ctx_g;
    const int64_t C = grid->ne[2];
    ggml_tensor * win = ggml_view_4d(ctx, grid, ws, ws, C, 1,
                                     grid->nb[1], grid->nb[2], grid->nb[3],
                                     (size_t) wx * ws * grid->nb[0] + (size_t) wy * ws * grid->nb[1]);
    win = ggml_cont(ctx, win);                                   // (ws,ws,C,1)
    win = ggml_reshape_3d(ctx, win, ws * ws, C, 1);               // (ws*ws,C,1)
    return ggml_cont(ctx, ggml_permute(ctx, win, 1, 0, 2, 3));    // (C,ws*ws,1)
}

// Inverse: (C, ws*ws, 1) token-major -> (ws,ws,C,1) grid tile.
static ggml_tensor * window_to_tile(ggml_context * ctx, ggml_tensor * tok, int ws) {
    const int64_t C = tok->ne[0];
    ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, tok, 1, 0, 2, 3)); // (ws*ws,C,1)
    return ggml_reshape_4d(ctx, t, ws, ws, C, 1);
}

// Slice out batch entry `idx` of a (C, T, N) tensor as (C, T, 1).
static ggml_tensor * batch_slice(ggml_context * ctx, ggml_tensor * x, int idx) {
    ggml_tensor * v = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], 1, x->nb[1], x->nb[2],
                                   (size_t) idx * x->nb[2]);
    return ggml_cont(ctx, v);
}

std::vector<ggml_tensor *> dinov2_backbone(Model & m, ggml_tensor * x, const BackboneParams & p) {
    ggml_context * ctx = m.ctx_g;
    const int nw = p.num_windows;
    const int nw2 = nw * nw;

    ggml_tensor * patches = conv2d(m, x, "embeddings.patch_embeddings.projection", p.patch_size, 0);
    const int64_t gw = patches->ne[0], gh = patches->ne[1];
    const int64_t n_patch = gw * gh;
    const int ws = (int) (gw / nw); // window side, in patches (gw==gh assumed)

    // token-major patch sequence, row-major (width fastest) to match
    // PyTorch's flatten(2) on (C,H,W): grid ne0=width, ne1=height already
    // matches that order once flattened with ne0 fastest.
    ggml_tensor * patch_tok = ggml_reshape_3d(ctx, patches, n_patch, p.hidden, 1);
    patch_tok = ggml_cont(ctx, ggml_permute(ctx, patch_tok, 1, 0, 2, 3)); // (hidden, n_patch, 1)

    ggml_tensor * pos = m.get("embeddings.position_embeddings"); // (hidden, 1+n_patch, 1)
    ggml_tensor * cls_pos = ggml_cont(ctx, ggml_view_3d(ctx, pos, p.hidden, 1, 1, pos->nb[1], pos->nb[2], 0));
    ggml_tensor * patch_pos = ggml_cont(ctx, ggml_view_3d(ctx, pos, p.hidden, n_patch, 1, pos->nb[1], pos->nb[2],
                                                          pos->nb[1]));
    patch_tok = ggml_add(ctx, patch_tok, patch_pos);

    ggml_tensor * cls_tok = ggml_reshape_3d(ctx, m.get("embeddings.cls_token"), p.hidden, 1, 1);
    cls_tok = ggml_add(ctx, cls_tok, cls_pos);

    ggml_tensor * grid_pos = ggml_cont(ctx, ggml_permute(ctx, patch_tok, 1, 0, 2, 3)); // (n_patch,hidden,1)
    grid_pos = ggml_reshape_4d(ctx, grid_pos, gw, gh, p.hidden, 1);

    ggml_tensor * reg = p.n_register > 0 ? m.get("embeddings.register_tokens") : nullptr; // (hidden, n_register)
    if (reg) reg = ggml_reshape_3d(ctx, reg, p.hidden, p.n_register, 1);

    const int64_t T = 1 + p.n_register + (int64_t) ws * ws;

    // Window partition: batch index w = q*nw + j (q = window row, j = window
    // col), matching the derivation in docs/decisions/backbone-windowing.md.
    ggml_tensor * seq = nullptr;
    for (int q = 0; q < nw; q++) {
        for (int j = 0; j < nw; j++) {
            ggml_tensor * win_patch = window_extract(m, grid_pos, j, q, ws); // (hidden, ws*ws, 1)
            ggml_tensor * seq_w = reg ? ggml_concat(ctx, ggml_concat(ctx, cls_tok, reg, 1), win_patch, 1)
                                       : ggml_concat(ctx, cls_tok, win_patch, 1);
            seq = seq ? ggml_concat(ctx, seq, seq_w, 2) : seq_w;
        }
    }

    std::vector<ggml_tensor *> taps;
    for (int l = 0; l < p.n_layer; l++) {
        const std::string pre = "encoder.layer." + std::to_string(l) + ".";
        const bool windowed = std::find(p.window_block_indexes.begin(), p.window_block_indexes.end(), l)
                               != p.window_block_indexes.end();

        if (windowed) {
            seq = dinov2_block(m, seq, pre, p.n_head, p.ln_eps); // batch = nw2, per-window attention
        } else {
            ggml_tensor * merged = ggml_reshape_3d(ctx, ggml_cont(ctx, seq), p.hidden, T * nw2, 1);
            merged = dinov2_block(m, merged, pre, p.n_head, p.ln_eps); // batch = 1, all windows attend to each other
            seq = ggml_reshape_3d(ctx, ggml_cont(ctx, merged), p.hidden, T, nw2);
        }

        if (std::find(p.out_feature_indexes.begin(), p.out_feature_indexes.end(), l) != p.out_feature_indexes.end()) {
            ggml_tensor * normed = layer_norm_affine(m, seq, "layernorm", p.ln_eps); // (hidden, T, nw2)

            ggml_tensor * grid = nullptr;
            for (int q = 0; q < nw; q++) {
                ggml_tensor * row = nullptr;
                for (int j = 0; j < nw; j++) {
                    ggml_tensor * win_seq = batch_slice(ctx, normed, q * nw + j); // (hidden, T, 1)
                    ggml_tensor * patch_part = ggml_cont(ctx, ggml_view_3d(ctx, win_seq, p.hidden, ws * ws, 1,
                                                                           win_seq->nb[1], win_seq->nb[2],
                                                                           (size_t) (1 + p.n_register) * win_seq->nb[1]));
                    ggml_tensor * tile = window_to_tile(ctx, patch_part, ws); // (ws,ws,hidden,1)
                    row = row ? ggml_concat(ctx, row, tile, 0) : tile;        // grow along width
                }
                grid = grid ? ggml_concat(ctx, grid, row, 1) : row;          // grow along height
            }
            taps.push_back(grid);
        }
    }
    return taps;
}
