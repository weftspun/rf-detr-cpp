#include "ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

Model::~Model() {
    for (ggml_backend_buffer * b : bufs) ggml_backend_buffer_free(b);
    for (ggml_context * c : ctx_w) ggml_free(c);
}

static void collect_config_kv(gguf_context * g, std::map<std::string, std::string> & config_json) {
    for (int64_t i = 0; i < gguf_get_n_kv(g); i++) {
        const char * key = gguf_get_key(g, i);
        if (gguf_get_kv_type(g, i) != GGUF_TYPE_STRING) continue;
        std::string k = key;
        for (const char * suf : { ".config_json" }) {
            size_t n = strlen(suf);
            if (k.size() > n && k.compare(k.size() - n, n, suf) == 0) {
                config_json[k] = gguf_get_val_str(g, i);
            }
        }
    }
}

bool Model::load(const char * path) {
    ggml_context * c = nullptr;
    gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &c };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;
    collect_config_kv(g, config_json);
    for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
        weights[ggml_get_name(t)] = t;
    }
    ctx_w.push_back(c);
    gguf_free(g);
    return true;
}

bool Model::load_backend(const char * path, ggml_backend_buffer_type * buft) {
    ggml_context * c = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &c };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;
    collect_config_kv(g, config_json);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!buf) { gguf_free(g); return false; }
    bufs.push_back(buf);

    FILE * f = fopen(path, "rb");
    if (!f) { gguf_free(g); return false; }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<uint8_t> staging;
    for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
        int64_t idx = gguf_find_tensor(g, ggml_get_name(t));
        if (idx < 0) { fclose(f); gguf_free(g); return false; }
        size_t nbytes = ggml_nbytes(t);
        staging.resize(nbytes);
#ifdef _WIN32
        int seek_rc = _fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(g, idx)), SEEK_SET);
#else
        int seek_rc = fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(g, idx)), SEEK_SET);
#endif
        if (seek_rc != 0 || fread(staging.data(), 1, nbytes, f) != nbytes) {
            fclose(f); gguf_free(g); return false;
        }
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        weights[ggml_get_name(t)] = t;
    }
    fclose(f);
    ctx_w.push_back(c);
    gguf_free(g);
    return true;
}

ggml_tensor * Model::get(const std::string & name) const {
    auto it = weights.find(name);
    if (it == weights.end()) { fprintf(stderr, "missing tensor: %s\n", name.c_str()); exit(1); }
    return it->second;
}

bool Model::has(const std::string & name) const {
    return weights.count(name) != 0;
}

ggml_tensor * conv2d(Model & m, ggml_tensor * x, const std::string & pre, int stride, int pad) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * w = m.get(pre + ".weight");
    ggml_type it = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im = ggml_im2col(ctx, w, x, stride, stride, pad, pad, 1, 1, true, it);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]),
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
    r = ggml_reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], w->ne[3]);
    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
    if (m.has(pre + ".bias")) {
        ggml_tensor * b = ggml_reshape_4d(ctx, m.get(pre + ".bias"), 1, 1, w->ne[3], 1);
        r = ggml_add(ctx, r, b);
    }
    return r;
}

ggml_tensor * layer_norm_affine(Model & m, ggml_tensor * x, const std::string & pre, float eps) {
    ggml_context * ctx = m.ctx_g;
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, m.get(pre + ".weight"));
    return ggml_add(ctx, x, m.get(pre + ".bias"));
}

ggml_tensor * linear(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * y = ggml_mul_mat(ctx, m.get(pre + ".weight"), x);
    if (m.has(pre + ".bias")) y = ggml_add(ctx, y, m.get(pre + ".bias"));
    return y;
}
