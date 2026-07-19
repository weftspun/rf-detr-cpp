// Isolated validation of src/deform_attn.cpp's hand-implemented bilinear
// sampling against the real upstream MSDeformAttn module on synthetic
// weights/inputs (gen_reference/gen_reference_deform_attn.py) -- done in
// isolation, before wiring into the full decoder, since this is the port's
// most novel component (no ggml built-in matches torch.grid_sample).
#include "deform_attn.h"
#include "test_common.h"

#include <cstdio>

int main(int argc, char ** argv) {
    const char * ref_path = argc > 1 ? argv[1] : "gen_reference/reference_deform_attn.bin";

    std::vector<NpyArray> ref;
    if (!read_ref(ref_path, ref, 12)) {
        fprintf(stderr, "failed to read %s (run gen_reference/gen_reference_deform_attn.py first)\n", ref_path);
        return 1;
    }
    const NpyArray & value_proj_w = ref[0], & value_proj_b = ref[1];
    const NpyArray & samp_off_w   = ref[2], & samp_off_b   = ref[3];
    const NpyArray & attn_w_w     = ref[4], & attn_w_b     = ref[5];
    const NpyArray & out_proj_w   = ref[6], & out_proj_b   = ref[7];
    const NpyArray & query_ref    = ref[8], & value_ref    = ref[9];
    const NpyArray & refpts_ref   = ref[10], & out_ref     = ref[11];

    const int d_model = 8, n_heads = 2, n_points = 2, gw = 4, gh = 4;
    const int64_t n_query = query_ref.shape[0];

    Model m;
    ggml_init_params wip = { 16 * 1024 * 1024, nullptr, /*no_alloc*/ false };
    ggml_context * wctx = ggml_init(wip);
    m.ctx_w.push_back(wctx);

    auto add_weight_2d = [&](const std::string & name, const NpyArray & a) {
        // a.shape is PyTorch (out,in); ggml wants ne=(in,out) matching linear()'s ggml_mul_mat convention.
        ggml_tensor * t = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, a.shape[1], a.shape[0]);
        ggml_set_name(t, name.c_str());
        memcpy(t->data, a.data.data(), a.data.size() * sizeof(float));
        m.weights[name] = t;
    };
    auto add_weight_1d = [&](const std::string & name, const NpyArray & a) {
        ggml_tensor * t = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, a.shape[0]);
        ggml_set_name(t, name.c_str());
        memcpy(t->data, a.data.data(), a.data.size() * sizeof(float));
        m.weights[name] = t;
    };
    add_weight_2d("deform.value_proj.weight", value_proj_w);
    add_weight_1d("deform.value_proj.bias", value_proj_b);
    add_weight_2d("deform.sampling_offsets.weight", samp_off_w);
    add_weight_1d("deform.sampling_offsets.bias", samp_off_b);
    add_weight_2d("deform.attention_weights.weight", attn_w_w);
    add_weight_1d("deform.attention_weights.bias", attn_w_b);
    add_weight_2d("deform.output_proj.weight", out_proj_w);
    add_weight_1d("deform.output_proj.bias", out_proj_b);

    init_graph_ctx(m, 4096);
    ggml_tensor * query = ggml_new_tensor_3d(m.ctx_g, GGML_TYPE_F32, d_model, n_query, 1);
    ggml_tensor * value = ggml_new_tensor_3d(m.ctx_g, GGML_TYPE_F32, d_model, gw * gh, 1);
    ggml_tensor * refpts = ggml_new_tensor_3d(m.ctx_g, GGML_TYPE_F32, 4, n_query, 1);
    ggml_set_input(query); ggml_set_input(value); ggml_set_input(refpts);

    ggml_tensor * out = ms_deform_attn(m, query, value, refpts, "deform", n_heads, n_points, gw, gh);

    bool ok = compute_cpu(m, out, 4096, [&] {
        ggml_backend_tensor_set(query, query_ref.data.data(), 0, query_ref.data.size() * sizeof(float));
        ggml_backend_tensor_set(value, value_ref.data.data(), 0, value_ref.data.size() * sizeof(float));
        ggml_backend_tensor_set(refpts, refpts_ref.data.data(), 0, refpts_ref.data.size() * sizeof(float));
    });
    if (!ok) return 1;

    return compare_ref(out, out_ref, 1e-3);
}
