#include "models.h"
#include <cmath>

// Gemma 4 audio encoder: USM-style conformer
// Reference: HuggingFace transformers Gemma4AudioModel
//
// Block order: FF1 → self_attn → conv1d → FF2 → norm_out
// Attention: global (simplified from chunked local) with relative position bias

static constexpr float SOFTCAP        = 50.0f;
static constexpr float RESIDUAL_WEIGHT = 0.5f;

ggml_cgraph * clip_graph_gemma4a::build() {
    const float eps = hparams.eps;

    ggml_tensor * inp = build_inp_raw(1);

    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    cur = ggml_reshape_4d(ctx0, cur, cur->ne[0], cur->ne[1], 1, 1);

    // conv subsampling: 2 conv2d layers, stride=2, ReLU
    {
        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[1], cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_relu(ctx0, cur);
    }

    // flatten: [OW, OH, C=32, 1] → [C*OH, OW] then input_projection
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
    cur = build_mm(model.audio_inp_proj_w, cur);

    // Precompute sinusoidal relative position embeddings as input tensor
    // HF: 13 positions (arange(12, -1, -1)), hidden_size=1024, [sin, cos] layout
    // This becomes an input to the graph that we set_input on
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, 13);
    ggml_set_name(pos_emb, "audio_pos_emb");
    ggml_set_input(pos_emb);
    ggml_build_forward_expand(gf, pos_emb);

    const float q_scale = (1.0f / std::sqrt((float)d_head)) / std::log(2.0f);
    const float k_scale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);

    // conformer blocks - data layout: [hidden=1024, seq_len]
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];

        // === FeedForward 1 (half-step) ===
        {
            auto * residual = cur;
            auto * ff = ggml_rms_norm(ctx0, cur, eps);
            ff = ggml_mul(ctx0, ff, layer.ff_norm_w);
            ff = build_ffn(ff, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, FFN_SILU, il);
            ff = ggml_rms_norm(ctx0, ff, eps);
            ff = ggml_mul(ctx0, ff, layer.ff_post_norm_w);
            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, ff, RESIDUAL_WEIGHT));
        }

        // === Self-Attention with relative position bias ===
        {
            auto * residual = cur;
            cur = ggml_rms_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, layer.attn_pre_norm_w);

            const int64_t seq_len = cur->ne[1];

            // Q with softplus(per_dim_scale) * q_scale
            ggml_tensor * Qcur = build_mm(layer.q_w, cur); // [hidden, seq]
            {
                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, seq_len);
                ggml_tensor * scale = ggml_softplus(ctx0, layer.per_dim_scale_w);
                scale = ggml_scale(ctx0, scale, q_scale);
                Qcur = ggml_mul(ctx0, Qcur, scale);
            }
            // Qcur: [d_head, n_head, seq] → permute to [d_head, seq, n_head]
            Qcur = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 2, 1, 3));

            // K with k_scale
            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            Kcur = ggml_scale(ctx0, Kcur, k_scale);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, seq_len);
            Kcur = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            // V
            ggml_tensor * Vcur = build_mm(layer.v_w, cur);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, seq_len);
            Vcur = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));

            // Content attention: Q @ K^T → [seq, seq, n_head]
            ggml_tensor * scores = ggml_mul_mat(ctx0, Qcur, Kcur);

            // Relative position bias: Q @ rel_K^T
            // rel_K = relative_k_proj(pos_emb): [hidden, 13] → [hidden, 13]
            ggml_tensor * rel_k = build_mm(layer.attn_k_rel_w, pos_emb); // [hidden, 13]
            rel_k = ggml_reshape_3d(ctx0, rel_k, d_head, n_head, 13);
            rel_k = ggml_cont(ctx0, ggml_permute(ctx0, rel_k, 0, 2, 1, 3)); // [d_head, 13, n_head]

            // Relative position bias: approximate by adding Q @ rel_K^T summed over positions
            // Full implementation needs chunked blocking + rel_shift
            // pos_scores: [seq, 13, n_head] → sum over dim1 → [seq, 1, n_head] → broadcast
            ggml_tensor * pos_scores = ggml_mul_mat(ctx0, Qcur, rel_k); // [seq, 13, n_head]
            // Sum over the 13 positions and scale
            // Use ggml_sum_rows which sums over dim0... we need sum over dim1
            // Reshape: [seq, 13, n_head] → [13, seq, n_head] then sum_rows → [1, seq, n_head]
            pos_scores = ggml_cont(ctx0, ggml_permute(ctx0, pos_scores, 1, 0, 2, 3));
            pos_scores = ggml_sum_rows(ctx0, pos_scores); // [1, seq, n_head]
            pos_scores = ggml_scale(ctx0, pos_scores, 1.0f / 13.0f); // mean
            pos_scores = ggml_cont(ctx0, ggml_permute(ctx0, pos_scores, 1, 0, 2, 3)); // [seq, 1, n_head]
            scores = ggml_add(ctx0, scores, pos_scores);

            // Softcap: tanh(scores/cap) * cap
            scores = ggml_scale(ctx0, scores, 1.0f / SOFTCAP);
            scores = ggml_tanh(ctx0, scores);
            scores = ggml_scale(ctx0, scores, SOFTCAP);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);
            ggml_tensor * x = ggml_mul_mat(ctx0, attn, Vcur);
            x = ggml_permute(ctx0, x, 2, 0, 1, 3);
            x = ggml_cont_2d(ctx0, x, x->ne[0] * x->ne[1], x->ne[2]);

            ggml_tensor * out = build_mm(layer.o_w, x);
            out = ggml_rms_norm(ctx0, out, eps);
            out = ggml_mul(ctx0, out, layer.attn_post_norm_w);
            cur = ggml_add(ctx0, residual, out);
        }

        // === LightConv1d ===
        {
            auto * residual = cur;
            cur = ggml_rms_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, layer.norm_conv_w);

            auto * x = build_mm(layer.conv_pw1_w, cur);
            {
                int64_t d = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0]));
                x = ggml_mul(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }

            x = ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);

            x = ggml_rms_norm(ctx0, x, eps);
            x = ggml_mul(ctx0, x, layer.conv_norm_w);
            x = ggml_silu(ctx0, x);

            x = build_mm(layer.conv_pw2_w, x);
            cur = ggml_add(ctx0, residual, x);
        }

        // === FeedForward 2 (half-step) ===
        {
            auto * residual = cur;
            auto * ff = ggml_rms_norm(ctx0, cur, eps);
            ff = ggml_mul(ctx0, ff, layer.ff_norm_1_w);
            ff = build_ffn(ff, layer.ff_up_1_w, nullptr, nullptr, nullptr, layer.ff_down_1_w, nullptr, FFN_SILU, il);
            ff = ggml_rms_norm(ctx0, ff, eps);
            ff = ggml_mul(ctx0, ff, layer.ff_post_norm_1_w);
            cur = ggml_add(ctx0, residual, ggml_scale(ctx0, ff, RESIDUAL_WEIGHT));
        }

        // === Final norm ===
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ln_2_w);
    }

    // output projection: 1024 → 1536
    cur = build_mm(model.pre_encode_out_w, cur);
    cur = ggml_add(ctx0, cur, model.pre_encode_out_b);

    // audio adapter: 1536 → 2560
    cur = build_mm(model.mm_audio_inp_proj_w, cur);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
