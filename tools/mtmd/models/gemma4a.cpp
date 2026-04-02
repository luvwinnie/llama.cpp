#include "models.h"
#include <cmath>

// Gemma 4 audio: USM conformer with sliding window + relative position bias
// Relative position: Toeplitz structure — for each distance d (0..12),
// bias[i, i-d] = Q[i] @ rel_k[12-d] (13 diagonal bands)

static constexpr float SOFTCAP     = 50.0f;
static constexpr float RESIDUAL_WT = 0.5f;
static constexpr int   N_REL_POS   = 13;

ggml_cgraph * clip_graph_gemma4a::build() {
    const float eps = hparams.eps;

    ggml_tensor * inp = build_inp_raw(1);
    // build_inp_raw gives [n_frames, n_mel, 1] (ne0=nx=n_frames, ne1=ny=n_mel)
    // ggml conv2d expects [W, H, C, N] — we need W=n_mel, H=n_frames
    // HF conv2d input: [B, C=1, H=n_frames, W=n_mel]
    // So transpose to put n_mel in dim0: [n_mel, n_frames]
    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    cur = ggml_reshape_4d(ctx0, cur, cur->ne[0], cur->ne[1], 1, 1);

    // conv subsampling: conv2d → LayerNorm(channel) → ReLU ×2
    // ggml conv2d output: [OW, OH, C, 1] where C = channels
    // HF: norm(x.permute(0,2,3,1)).permute(0,3,1,2) = LayerNorm over C dimension
    // In ggml: C is dim2. Need to flatten spatial, permute C to dim0, norm, permute back.
    for (int conv_idx = 0; conv_idx < 2; conv_idx++) {
        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[conv_idx], cur, 2, 2, 1, 1, 1, 1);
        // cur: [OW, OH, C, 1]
        {
            int64_t ow = cur->ne[0], oh = cur->ne[1], oc = cur->ne[2];
            // Flatten spatial: [OW*OH, C, 1, 1]
            cur = ggml_reshape_4d(ctx0, cur, ow * oh, oc, 1, 1);
            // Permute C to dim0: [C, OW*OH, 1, 1]
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
            // LayerNorm over dim0 (C)
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, model.conv_norm_w_arr[conv_idx]);
            // Permute back: [OW*OH, C, 1, 1]
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
            // Reshape to 4D: [OW, OH, C, 1]
            cur = ggml_reshape_4d(ctx0, cur, ow, oh, oc, 1);
        }
        cur = ggml_relu(ctx0, cur);
    }

    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
    cur = build_mm(model.audio_inp_proj_w, cur);

    const int64_t S = cur->ne[1]; // seq_len

    // Input: relative position embeddings [hidden, 13]
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N_REL_POS);
    ggml_set_name(pos_emb, "audio_pos_emb");
    ggml_set_input(pos_emb);
    ggml_build_forward_expand(gf, pos_emb);

    // Input: sliding window mask [S, S]
    ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, S, S);
    ggml_set_name(attn_mask, "audio_attn_mask");
    ggml_set_input(attn_mask);
    ggml_build_forward_expand(gf, attn_mask);

    // Input: 13 Toeplitz diagonal masks [S, S] each
    // diag_mask_d[i, j] = 1.0 if j == i - d, else 0.0 (for d = 0..12)
    // Precomputed on CPU, used per-layer to scatter position scores
    ggml_tensor * diag_masks[N_REL_POS];
    for (int d = 0; d < N_REL_POS; d++) {
        char name[64];
        snprintf(name, sizeof(name), "audio_diag_%d", d);
        diag_masks[d] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, S, S);
        ggml_set_name(diag_masks[d], name);
        ggml_set_input(diag_masks[d]);
        ggml_build_forward_expand(gf, diag_masks[d]);
    }

    const float q_scale = (1.0f / std::sqrt((float)d_head)) / std::log(2.0f);
    const float k_scale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);

    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];

        // FF1 (half-step)
        {
            auto * res = cur;
            auto * ff = ggml_rms_norm(ctx0, cur, eps);
            ff = ggml_mul(ctx0, ff, layer.ff_norm_w);
            ff = build_ffn(ff, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, FFN_SILU, il);
            ff = ggml_rms_norm(ctx0, ff, eps);
            ff = ggml_mul(ctx0, ff, layer.ff_post_norm_w);
            cur = ggml_add(ctx0, res, ggml_scale(ctx0, ff, RESIDUAL_WT));
        }

        // Self-attention
        {
            auto * res = cur;
            cur = ggml_rms_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, layer.attn_pre_norm_w);

            // Q with per_dim_scale * q_scale
            // NOTE: per_dim_scale is already softplus'd by convert_hf_to_gguf.py
            ggml_tensor * Q = build_mm(layer.q_w, cur);
            {
                Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, S);
                auto * s = ggml_scale(ctx0, layer.per_dim_scale_w, q_scale);
                Q = ggml_mul(ctx0, Q, s);
            }
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // [dh, S, nh]

            // K with k_scale
            ggml_tensor * K = build_mm(layer.k_w, cur);
            K = ggml_scale(ctx0, K, k_scale);
            K = ggml_reshape_3d(ctx0, K, d_head, n_head, S);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));

            // V
            ggml_tensor * V = build_mm(layer.v_w, cur);
            V = ggml_reshape_3d(ctx0, V, d_head, n_head, S);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));

            // Content attention: Q @ K^T → [S, S, nh]
            ggml_tensor * scores = ggml_mul_mat(ctx0, Q, K);

            // Relative position bias via Toeplitz diagonals:
            // For each distance d (0..12):
            //   pos_idx = 12 - d
            //   Extract rel_scores[:, pos_idx] → [S, nh] (score at this distance for each query)
            //   Scatter onto diagonal -d using diag_mask_d
            //
            // rel_k = attn_k_rel(pos_emb) → [hidden, 13]
            ggml_tensor * rel_k = build_mm(layer.attn_k_rel_w, pos_emb); // [hidden, 13]
            rel_k = ggml_reshape_3d(ctx0, rel_k, d_head, n_head, N_REL_POS);
            rel_k = ggml_cont(ctx0, ggml_permute(ctx0, rel_k, 0, 2, 1, 3)); // [dh, 13, nh]

            // Q @ rel_k^T → [S, 13, nh]
            ggml_tensor * qr = ggml_mul_mat(ctx0, Q, rel_k); // [S, 13, nh]

            // For each distance d, extract column and scatter onto diagonal:
            // bias[i, i-d, h] = qr[i, 12-d, h]
            for (int d = 0; d < N_REL_POS; d++) {
                int pos_idx = 12 - d;
                // Extract qr[:, pos_idx, :] → [S, 1, nh]
                ggml_tensor * col = ggml_view_3d(ctx0, qr,
                    S, 1, n_head,
                    qr->nb[1], qr->nb[2],
                    pos_idx * qr->nb[1]);
                // Repeat col along dim1 to [S, S, nh]
                col = ggml_repeat(ctx0, col, scores); // [S, S, nh]
                // Multiply by diagonal mask [S, S] → need [S, S, nh]
                // diag_masks[d] is [S, S], reshape to [S, S, 1] won't broadcast into [S, S, nh]
                // Instead: repeat diag_mask to [S, S, nh]
                ggml_tensor * mask_nh = ggml_reshape_3d(ctx0, diag_masks[d], S, S, 1);
                mask_nh = ggml_repeat(ctx0, mask_nh, scores); // [S, S, nh]
                ggml_tensor * bias_d = ggml_mul(ctx0, col, mask_nh);
                scores = ggml_add(ctx0, scores, bias_d);
            }

            // Sliding window mask
            scores = ggml_add(ctx0, scores, attn_mask);

            // Softcap
            scores = ggml_scale(ctx0, scores, 1.0f / SOFTCAP);
            scores = ggml_tanh(ctx0, scores);
            scores = ggml_scale(ctx0, scores, SOFTCAP);

            auto * attn = ggml_soft_max(ctx0, scores);
            auto * x = ggml_mul_mat(ctx0, attn, V);
            x = ggml_permute(ctx0, x, 2, 0, 1, 3);
            x = ggml_cont_2d(ctx0, x, x->ne[0] * x->ne[1], x->ne[2]);

            auto * out = build_mm(layer.o_w, x);
            out = ggml_rms_norm(ctx0, out, eps);
            out = ggml_mul(ctx0, out, layer.attn_post_norm_w);
            cur = ggml_add(ctx0, res, out);
        }

        // LightConv1d
        {
            auto * res = cur;
            cur = ggml_rms_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, layer.norm_conv_w);
            auto * x = build_mm(layer.conv_pw1_w, cur);
            {
                int64_t dd = x->ne[0] / 2;
                auto * gate = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, x, dd, x->ne[1], x->nb[1], dd * x->nb[0]));
                x = ggml_mul(ctx0, ggml_view_2d(ctx0, x, dd, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }
            x = ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            x = ggml_rms_norm(ctx0, x, eps);
            x = ggml_mul(ctx0, x, layer.conv_norm_w);
            x = ggml_silu(ctx0, x);
            x = build_mm(layer.conv_pw2_w, x);
            cur = ggml_add(ctx0, res, x);
        }

        // FF2 (half-step)
        {
            auto * res = cur;
            auto * ff = ggml_rms_norm(ctx0, cur, eps);
            ff = ggml_mul(ctx0, ff, layer.ff_norm_1_w);
            ff = build_ffn(ff, layer.ff_up_1_w, nullptr, nullptr, nullptr, layer.ff_down_1_w, nullptr, FFN_SILU, il);
            ff = ggml_rms_norm(ctx0, ff, eps);
            ff = ggml_mul(ctx0, ff, layer.ff_post_norm_1_w);
            cur = ggml_add(ctx0, res, ggml_scale(ctx0, ff, RESIDUAL_WT));
        }

        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ln_2_w);
    }

    cur = build_mm(model.pre_encode_out_w, cur);
    cur = ggml_add(ctx0, cur, model.pre_encode_out_b);
    cur = build_mm(model.mm_audio_inp_proj_w, cur);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
