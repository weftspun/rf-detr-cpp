// Isolated validation of src/projector.cpp's projector_multiscale
// (scale_factors=[2.0,0.5], i.e. P3+P5) against the real upstream
// MultiScaleProjector module on synthetic weights/inputs
// (gen_reference/gen_reference_projector_multiscale.py) -- needed for
// RFDETRLargeDeprecated's projector_scale=["P3","P5"]. Custom binary
// format (named params, since the number/shape of weight tensors depends
// on scale_factors and doesn't fit test_common.h's fixed-order read_ref).
#include "ops.h"
#include "projector.h"
#include "test_common.h"

#include <cstdio>
#include <cstring>
#include <string>

struct NamedArray {
    std::string name;
    NpyArray arr;
};

static bool read_arr(FILE * f, NpyArray & arr) {
    int32_t ndim = 0;
    if (fread(&ndim, 4, 1, f) != 1 || ndim <= 0 || ndim > 8) {
        return false;
    }
    arr.shape.resize(ndim);
    if (fread(arr.shape.data(), 8, ndim, f) != (size_t) ndim) {
        return false;
    }
    int64_t n = 1;
    for (int64_t d : arr.shape) {
        n *= d;
    }
    arr.data.resize(n);
    return fread(arr.data.data(), 4, n, f) == (size_t) n;
}

int main(int argc, char ** argv) {
    const char * ref_path = argc > 1 ? argv[1] : "gen_reference/reference_projector_multiscale.bin";
    FILE * f = fopen(ref_path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s (run gen_reference/gen_reference_projector_multiscale.py first)\n", ref_path);
        return 1;
    }

    int32_t n_params = 0;
    fread(&n_params, 4, 1, f);
    std::vector<NamedArray> params(n_params);
    for (auto & p : params) {
        int32_t name_len = 0;
        fread(&name_len, 4, 1, f);
        p.name.resize(name_len);
        fread(&p.name[0], 1, name_len, f);
        if (!read_arr(f, p.arr)) { fprintf(stderr, "bad param array\n"); return 1; }
    }
    int32_t n_taps = 0;
    fread(&n_taps, 4, 1, f);
    std::vector<NpyArray> taps(n_taps);
    for (auto & t : taps) {
        if (!read_arr(f, t)) { fprintf(stderr, "bad tap array\n"); return 1; }
    }
    int32_t n_outputs = 0;
    fread(&n_outputs, 4, 1, f);
    std::vector<NpyArray> outs_ref(n_outputs);
    for (auto & o : outs_ref) {
        if (!read_arr(f, o)) { fprintf(stderr, "bad output array\n"); return 1; }
    }
    fclose(f);
    printf("loaded %d params, %d taps, %d outputs\n", n_params, n_taps, n_outputs);

    const int out_channels = (int) outs_ref[0].shape[2];
    const std::string prefix = "projector";

    Model m;
    size_t wbytes = 4 * 1024 * 1024;
    for (auto & p : params) {
        wbytes += p.arr.data.size() * 4 + 1024;
    }
    ggml_init_params wip = { wbytes, nullptr, /*no_alloc*/ false };
    ggml_context * wctx = ggml_init(wip);
    m.ctx_w.push_back(wctx);
    for (auto & p : params) {
        // PyTorch shapes are (out,...,in) row-major; GGUF's convention (already
        // used by every convert_*.py script) reverses to ne=(in,...,out) -- do
        // the same reversal here so weight layout matches what conv2d/conv_x/
        // conv_transpose2d/linear all expect.
        std::vector<int64_t> ne(p.arr.shape.rbegin(), p.arr.shape.rend());
        while (ne.size() < 1) ne.push_back(1);
        ggml_tensor * t;
        if (ne.size() == 1) t = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, ne[0]);
        else if (ne.size() == 2) t = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, ne[0], ne[1]);
        else if (ne.size() == 4) t = ggml_new_tensor_4d(wctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        else { fprintf(stderr, "unexpected ndim for %s\n", p.name.c_str()); return 1; }
        memcpy(t->data, p.arr.data.data(), p.arr.data.size() * sizeof(float));
        m.weights[prefix + "." + p.name] = t;
    }

    init_graph_ctx(m, 4096);
    std::vector<ggml_tensor *> tap_tensors;
    for (auto & t : taps) {
        int64_t H = t.shape[0], W = t.shape[1], C = t.shape[2];
        ggml_tensor * gt = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, W, H, C, 1);
        ggml_set_input(gt);
        tap_tensors.push_back(gt);
    }

    std::vector<float> scale_factors = { 2.0f, 0.5f };
    std::vector<ggml_tensor *> outs = projector_multiscale(m, tap_tensors, scale_factors, out_channels, prefix);

    bool ok = compute_cpu_multi(m, outs, 4096, [&] {
        for (size_t i = 0; i < taps.size(); i++) {
            int64_t H = taps[i].shape[0], W = taps[i].shape[1], C = taps[i].shape[2];
            std::vector<float> whc(taps[i].data.size());
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    for (int64_t c = 0; c < C; c++) {
                        whc[c * H * W + h * W + w] = taps[i].data[h * W * C + w * C + c];
                    }
                }
            }
            ggml_backend_tensor_set(tap_tensors[i], whc.data(), 0, whc.size() * sizeof(float));
        }
    });
    if (!ok) {
        return 1;
    }

    int rc = 0;
    for (size_t i = 0; i < outs.size(); i++) {
        printf("level %zu:\n", i);
        // outs_ref[i] is (H,W,C) numpy order; out tensor is ggml (W,H,C,1) --
        // reorder the reference the same way test_backbone.cpp compares taps.
        int64_t H = outs_ref[i].shape[0], W = outs_ref[i].shape[1], C = outs_ref[i].shape[2];
        NpyArray whc_ref;
        whc_ref.shape = { C, H, W };
        whc_ref.data.resize(outs_ref[i].data.size());
        for (int64_t h = 0; h < H; h++) {
            for (int64_t w = 0; w < W; w++) {
                for (int64_t c = 0; c < C; c++) {
                    whc_ref.data[c * H * W + h * W + w] = outs_ref[i].data[h * W * C + w * C + c];
                }
            }
        }
        rc |= compare_ref(outs[i], whc_ref, 1e-3);
    }
    return rc;
}
