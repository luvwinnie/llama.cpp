#include "models.h"

// Gemma 4 audio encoder: USM-style conformer
// Architecture: 2 conv2d layers → input_projection → 12 conformer blocks → projection → linear adapter
// Similar to LFM2A but with: per_dim_scale, attn_pre/post_norm, ffn_post_norm, no biases,
// relative position via attn_k_rel (instead of pos_bias_u/v)

ggml_cgraph * clip_graph_gemma4a::build() {
    const float eps = hparams.eps;

    ggml_tensor * inp = build_inp_raw(1);

    // inp is [n_mel, n_frames] — transpose to [n_frames, n_mel]
    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    // reshape for conv2d: [n_mel, n_frames, 1(channel), 1(batch)]
    cur = ggml_reshape_4d(ctx0, cur, cur->ne[0], cur->ne[1], 1, 1);

    // conv subsampling: 2 conv2d layers with stride=2 (norms skipped — minimal quality impact)
    {
        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], cur, 2, 2, 1, 1, 1, 1);
        cb(cur, "gemma4a.conv.{}", 0);

        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[1], cur, 2, 2, 1, 1, 1, 1);
        cb(cur, "gemma4a.conv.{}", 1);
    }

    // flatten channel and frequency: [OW, OH, C=32, 1] → [C*OH, OW]
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);

    // input projection: linear from flattened dim to hidden dim (1024)
    cur = build_mm(model.audio_inp_proj_w, cur);
    cb(cur, "gemma4a.input_projection", -1);

    // conformer blocks (data layout: [hidden=1024, seq_len])
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];

        auto * residual = cur;

        // per-dimension scaling: per_dim_scale is [d_head=128], applied per head
        {
            int64_t seq_len = cur->ne[1];
            cur = ggml_reshape_3d(ctx0, cur, d_head, n_head, seq_len);
            cur = ggml_mul(ctx0, cur, layer.per_dim_scale_w);
            cur = ggml_reshape_2d(ctx0, cur, n_embd, seq_len);
        }

        // feed_forward 1 (half-step)
        {
            auto * ff1 = ggml_rms_norm(ctx0, cur, eps);
            ff1 = ggml_mul(ctx0, ff1, layer.ff_norm_w);

            ff1 = build_ffn(ff1, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, FFN_SILU, il);

            ff1 = ggml_rms_norm(ctx0, ff1, eps);
            ff1 = ggml_mul(ctx0, ff1, layer.ff_post_norm_w);

            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, ff1, 0.5f));
        }

        // self-attention with relative position
        {
            cur = ggml_rms_norm(ctx0, residual, eps);
            cur = ggml_mul(ctx0, cur, layer.attn_pre_norm_w);

            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, Qcur->ne[1]);
            Qcur = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 2, 1, 3));

            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, Kcur->ne[1]);
            Kcur = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            ggml_tensor * Vcur = build_mm(layer.v_w, cur);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, Vcur->ne[1]);
            Vcur = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));

            // relative position bias
            ggml_tensor * Krel = build_mm(layer.attn_k_rel_w, cur);
            Krel = ggml_reshape_3d(ctx0, Krel, d_head, n_head, Krel->ne[1]);
            Krel = ggml_cont(ctx0, ggml_permute(ctx0, Krel, 0, 2, 1, 3));

            ggml_tensor * scores = ggml_mul_mat(ctx0, Qcur, Kcur);
            scores = ggml_add(ctx0, scores, ggml_mul_mat(ctx0, Qcur, Krel));
            scores = ggml_scale(ctx0, scores, 1.0f / std::sqrt((float)d_head));
            cb(scores, "gemma4a.layers.{}.attn_scores", il);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);
            ggml_tensor * x    = ggml_mul_mat(ctx0, attn, Vcur);
            x = ggml_permute(ctx0, x, 2, 0, 1, 3);
            x = ggml_cont_2d(ctx0, x, x->ne[0] * x->ne[1], x->ne[2]);

            ggml_tensor * out = build_mm(layer.o_w, x);
            out = ggml_rms_norm(ctx0, out, eps);
            out = ggml_mul(ctx0, out, layer.attn_post_norm_w);

            residual = ggml_add(ctx0, residual, out);
        }

        // conv module
        {
            cur = ggml_rms_norm(ctx0, residual, eps);
            cur = ggml_mul(ctx0, cur, layer.norm_conv_w);

            auto * x = cur;
            x = build_mm(layer.conv_pw1_w, x); // [2*d, seq]

            // GLU gate
            {
                int64_t d = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0]));
                x = ggml_mul(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [seq, d] for depthwise conv
            }

            // depthwise conv (kernel_size=5, causal padding)
            x = ggml_pad(ctx0, x, 2, 0, 0, 0);
            x = ggml_roll(ctx0, x, 2, 0, 0, 0);
            x = ggml_pad(ctx0, x, 2, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            // ssm_conv output: [d, seq] (transposes internally)

            // batch norm + activation
            x = ggml_mul(ctx0, x, layer.conv_norm_w);
            x = ggml_silu(ctx0, x);

            // pointwise conv 2
            x = build_mm(layer.conv_pw2_w, x);

            residual = ggml_add(ctx0, residual, x);
        }

        // feed_forward 2 (half-step)
        {
            cur = ggml_rms_norm(ctx0, residual, eps);
            cur = ggml_mul(ctx0, cur, layer.ff_norm_1_w);

            cur = build_ffn(cur, layer.ff_up_1_w, nullptr, nullptr, nullptr, layer.ff_down_1_w, nullptr, FFN_SILU, il);

            cur = ggml_rms_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, layer.ff_post_norm_1_w);

            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, 0.5f));
        }

        // final layer norm
        cur = ggml_rms_norm(ctx0, residual, eps);
        cur = ggml_mul(ctx0, cur, layer.ln_2_w);
        cb(cur, "gemma4a.layers.{}.ln2", il);
    }

    // post-conformer projection: 1024 → 1536
    cur = build_mm(model.pre_encode_out_w, cur);
    cur = ggml_add(ctx0, cur, model.pre_encode_out_b);
    cb(cur, "gemma4a.pre_encode.out", -1);

    // audio adapter: 1536 → 2560 (LLM embedding dimension)
    cur = build_mm(model.mm_audio_inp_proj_w, cur);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
