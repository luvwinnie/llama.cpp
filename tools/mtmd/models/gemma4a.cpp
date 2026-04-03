/**
 * Gemma 4 Audio Conformer Encoder
 * Ported from ollama model/models/gemma4/model_audio.go
 * Reference: HF transformers modular_gemma4.py
 */
#include "models.h"
#include <cmath>
#include <cfloat>

static constexpr float SOFTCAP     = 50.0f;
static constexpr float RES_WEIGHT  = 0.5f;
static constexpr float GRAD_CLIP   = 1e10f;
static constexpr float NORM_EPS    = 1e-6f;
static constexpr int   CHUNK_SZ    = 12;
static constexpr int   MAX_PAST    = 12;
static constexpr int   CTX_SZ      = 24;
static constexpr int   N_POS       = 13;

// Gemma4ClippableLinear: clamp input/output
ggml_tensor * clip_graph_gemma4a::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    auto it = model.clamp_info_map.find(w->name);
    if (it == model.clamp_info_map.end()) {
        return ggml_mul_mat(ctx0, w, x);
    }
    const auto & ci = it->second;
    if (ci.inp_max < FLT_MAX) x = ggml_clamp(ctx0, x, ci.inp_min, ci.inp_max);
    ggml_tensor * out = ggml_mul_mat(ctx0, w, x);
    if (ci.out_max < FLT_MAX) out = ggml_clamp(ctx0, out, ci.out_min, ci.out_max);
    return out;
}

ggml_cgraph * clip_graph_gemma4a::build() {
    ggml_tensor * inp = build_inp_raw(1);

    // inp: [nx=frames, ny=mel, 1] → transpose to [mel, frames] for conv2d
    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    cur = ggml_reshape_4d(ctx0, cur, cur->ne[0], cur->ne[1], 1, 1);

    // ── Conv2d subsampling ──
    // Two conv blocks: conv2d → LayerNorm(channels) → ReLU
    // Ollama: permute(1,2,0,3) to get channels in ne[0], norm, permute(2,0,1,3) back
    for (int ci = 0; ci < 2; ci++) {
        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[ci], cur, 2, 2, 1, 1, 1, 1);
        // cur: [OW=freq', OH=time', OC=channels, 1]
        if (model.conv_norm_w_arr[ci]) {
            // LayerNorm over channels: flatten spatial, put channels in ne[0]
            int64_t ow = cur->ne[0], oh = cur->ne[1], oc = cur->ne[2];
            cur = ggml_reshape_2d(ctx0, cur, ow * oh, oc);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3)); // [OC, spatial]
            cur = ggml_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, model.conv_norm_w_arr[ci]);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3)); // [spatial, OC]
            cur = ggml_reshape_4d(ctx0, cur, ow, oh, oc, 1);
        }
        cur = ggml_relu(ctx0, cur);
    }

    // ── Flatten for linear projection ──
    // cur: [OW=F, OH=T, OC=C, 1]
    // HF: permute(0,2,3,1) → [B,T,F,C] → reshape [B,T,F*C]
    // Feature i = f*C + c (freq slowest, channel fastest? No: f varies over F, c over C)
    // In HF [T,F,C], reshape to [T, F*C]: feat index = f*C + c
    // In ggml [F,T,C,1]: to match, need [F,C,T,1] → reshape [F*C, T]
    // = ggml permute(0,2,1,3) on [F,T,C,1] → [F,C,T,1]
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3)); // [F, C, T, 1]
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]); // [F*C, T]

    // Input projection
    cur = build_mm(model.audio_inp_proj_w, cur); // [1024, T]

    const int64_t S = cur->ne[1]; // sequence length

    // ── Conformer blocks ──
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];

        // ── FFW 1 (half-residual) ──
        {
            auto * res = cur;
            cur = ggml_clamp(ctx0, cur, -GRAD_CLIP, GRAD_CLIP);
            cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, layer.ff_norm_w);
            cur = build_mm(layer.ff_up_w, cur);
            cur = ggml_silu(ctx0, cur);
            cur = build_mm(layer.ff_down_w, cur);
            cur = ggml_clamp(ctx0, cur, -GRAD_CLIP, GRAD_CLIP);
            cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, layer.ff_post_norm_w);
            cur = ggml_scale(ctx0, cur, RES_WEIGHT);
            cur = ggml_add(ctx0, res, cur);
        }

        // ── Self-Attention (per-chunk loop, matching ollama) ──
        {
            auto * res = cur;
            cur = ggml_clamp(ctx0, cur, -GRAD_CLIP, GRAD_CLIP);
            cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, layer.attn_pre_norm_w);

            // QKV: [hidden, S]
            ggml_tensor * Q = build_mm(layer.q_w, cur);
            ggml_tensor * K = build_mm(layer.k_w, cur);
            ggml_tensor * V = build_mm(layer.v_w, cur);

            // Reshape to [d_head, n_head, S]
            Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, S);
            K = ggml_reshape_3d(ctx0, K, d_head, n_head, S);
            V = ggml_reshape_3d(ctx0, V, d_head, n_head, S);

            // Q scaling: q_scale * per_dim_scale (already softplus'd)
            const float q_scale = (1.0f / std::sqrt((float)d_head)) / std::log(2.0f);
            Q = ggml_scale(ctx0, Q, q_scale);
            if (layer.per_dim_scale_w) {
                Q = ggml_mul(ctx0, Q, layer.per_dim_scale_w); // [d_head] broadcasts
            }

            // K scaling
            const float k_scale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
            K = ggml_scale(ctx0, K, k_scale);

            // Permute to [d_head, S, n_head] for easier blocking
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

            // Build sinusoidal position embeddings: [hidden, N_POS]
            ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N_POS);
            ggml_set_name(pos_emb, "audio_pos_emb");
            ggml_set_input(pos_emb);
            if (il == 0) ggml_build_forward_expand(gf, pos_emb);

            // Project position embeddings through attn_k_rel (linear_pos)
            ggml_tensor * pos_proj = build_mm(layer.attn_k_rel_w, pos_emb); // [hidden, N_POS]
            pos_proj = ggml_reshape_3d(ctx0, pos_proj, d_head, n_head, N_POS);
            // Permute to [d_head, N_POS, n_head]
            pos_proj = ggml_cont(ctx0, ggml_permute(ctx0, pos_proj, 0, 2, 1, 3));

            // Pad Q to multiple of CHUNK_SZ
            int64_t n_blocks = (S + CHUNK_SZ - 1) / CHUNK_SZ;
            int64_t S_pad = n_blocks * CHUNK_SZ;
            int64_t pad_t = S_pad - S;
            if (pad_t > 0) {
                Q = ggml_pad(ctx0, Q, 0, (int)pad_t, 0, 0);
                K = ggml_pad(ctx0, K, 0, (int)pad_t, 0, 0);
                V = ggml_pad(ctx0, V, 0, (int)pad_t, 0, 0);
            }

            // Pad K,V for context: left by MAX_PAST, right by CHUNK_SZ-1
            // Use pad+roll for left padding (ggml_pad only pads right)
            K = ggml_pad(ctx0, K, 0, MAX_PAST + CHUNK_SZ - 1, 0, 0);
            K = ggml_roll(ctx0, K, 0, MAX_PAST, 0, 0);
            V = ggml_pad(ctx0, V, 0, MAX_PAST + CHUNK_SZ - 1, 0, 0);
            V = ggml_roll(ctx0, V, 0, MAX_PAST, 0, 0);

            // Reshape Q into chunks: [d_head, CHUNK_SZ, n_head, n_blocks]
            ggml_tensor * Q_chunked = ggml_reshape_4d(ctx0, Q, d_head, CHUNK_SZ, n_head, n_blocks);

            // Per-chunk attention loop (matching ollama exactly)
            ggml_tensor * chunk_outs[128]; // max blocks
            GGML_ASSERT(n_blocks <= 128);

            for (int u = 0; u < n_blocks; u++) {
                // Extract Q block: [d_head, CHUNK_SZ, n_head]
                ggml_tensor * q_blk = ggml_view_3d(ctx0, Q_chunked,
                    d_head, CHUNK_SZ, n_head,
                    Q_chunked->nb[1], Q_chunked->nb[2],
                    u * Q_chunked->nb[3]);
                q_blk = ggml_cont(ctx0, q_blk);

                // Extract K,V context: [d_head, CTX_SZ, n_head]
                int64_t c_start = u * CHUNK_SZ;
                ggml_tensor * k_ctx = ggml_view_3d(ctx0, K,
                    d_head, CTX_SZ, n_head,
                    K->nb[1], K->nb[2],
                    c_start * K->nb[1]);
                k_ctx = ggml_cont(ctx0, k_ctx);

                ggml_tensor * v_ctx = ggml_view_3d(ctx0, V,
                    d_head, CTX_SZ, n_head,
                    V->nb[1], V->nb[2],
                    c_start * V->nb[1]);
                v_ctx = ggml_cont(ctx0, v_ctx);

                // Permute for matmul: [d_head, X, n_head] → [d_head, X, n_head]
                // Already in right layout for batched matmul over n_head

                // Content logits: k_ctx^T @ q_blk = [CTX_SZ, CHUNK_SZ, n_head]
                auto * qP = ggml_cont(ctx0, ggml_permute(ctx0, q_blk, 0, 2, 1, 3)); // [dh, nh, chunk]
                auto * kP = ggml_cont(ctx0, ggml_permute(ctx0, k_ctx, 0, 2, 1, 3)); // [dh, nh, ctx]

                // Wait: ollama does qP=[dh,chunk,nh], kP=[dh,ctx,nh] then kP.Mulmat(qP)=[ctx,chunk,nh]
                // In ggml: mul_mat(a,b) contracts over ne[0]. Need ne[0]=dh for both.
                // q_blk: [dh, CHUNK, nh], k_ctx: [dh, CTX, nh]
                // mul_mat(k_ctx, q_blk) = [CTX, CHUNK, nh] ← correct!
                auto * term_ac = ggml_mul_mat(ctx0, k_ctx, q_blk); // [CTX, CHUNK, nh]

                // Position logits: pos_proj^T @ q_blk = [N_POS, CHUNK, nh]
                auto * term_bd_raw = ggml_mul_mat(ctx0, pos_proj, q_blk); // [N_POS, CHUNK, nh]

                // Relative shift: [N_POS, CHUNK, nh] → [CTX_SZ, CHUNK, nh]
                {
                    int pad_amt = CTX_SZ + 1 - N_POS; // 12
                    if (pad_amt > 0) {
                        term_bd_raw = ggml_pad(ctx0, term_bd_raw, pad_amt, 0, 0, 0);
                    }
                    // [CTX+1, CHUNK, nh] → reshape [(CTX+1)*CHUNK, nh]
                    term_bd_raw = ggml_reshape_2d(ctx0, term_bd_raw, (CTX_SZ + 1) * CHUNK_SZ, n_head);
                    // Slice first CTX*CHUNK elements
                    term_bd_raw = ggml_view_2d(ctx0, term_bd_raw,
                        CTX_SZ * CHUNK_SZ, n_head,
                        term_bd_raw->nb[1], 0);
                    term_bd_raw = ggml_cont(ctx0, term_bd_raw);
                    // Reshape to [CTX, CHUNK, nh]
                    term_bd_raw = ggml_reshape_3d(ctx0, term_bd_raw, CTX_SZ, CHUNK_SZ, n_head);
                }

                // Combined logits
                auto * logits = ggml_add(ctx0, term_ac, term_bd_raw);

                // Softcap
                logits = ggml_scale(ctx0, logits, 1.0f / SOFTCAP);
                logits = ggml_tanh(ctx0, logits);
                logits = ggml_scale(ctx0, logits, SOFTCAP);

                // Causal-validity mask
                ggml_tensor * mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, CTX_SZ, CHUNK_SZ, 1);
                {
                    char name[64];
                    snprintf(name, sizeof(name), "attn_mask_%d_%d", il, u);
                    ggml_set_name(mask, name);
                    ggml_set_input(mask);
                    if (il == 0) ggml_build_forward_expand(gf, mask);
                }
                logits = ggml_add(ctx0, logits, mask);

                // Softmax over CTX dimension (ne[0])
                logits = ggml_soft_max(ctx0, logits);

                // Weighted sum: v_ctx^T @ logits
                // v_ctx: [dh, CTX, nh] → permute to [CTX, dh, nh]
                auto * v_t = ggml_cont(ctx0, ggml_permute(ctx0, v_ctx, 1, 0, 2, 3)); // [CTX, dh, nh]
                auto * chunk_out = ggml_mul_mat(ctx0, v_t, logits); // [dh, CHUNK, nh]

                // Permute to [dh, nh, CHUNK]
                chunk_out = ggml_cont(ctx0, ggml_permute(ctx0, chunk_out, 0, 2, 1, 3));
                chunk_outs[u] = chunk_out;
            }

            // Concatenate chunks along time dimension (dim 2)
            ggml_tensor * attn_out = chunk_outs[0];
            for (int u = 1; u < n_blocks; u++) {
                attn_out = ggml_concat(ctx0, attn_out, chunk_outs[u], 2);
            }

            // Trim to original seq length
            if (S_pad > S) {
                attn_out = ggml_view_3d(ctx0, attn_out,
                    d_head, n_head, S,
                    attn_out->nb[1], attn_out->nb[2], 0);
                attn_out = ggml_cont(ctx0, attn_out);
            }

            // Reshape to [hidden, S]
            attn_out = ggml_reshape_2d(ctx0, attn_out, d_head * n_head, S);

            // Output projection
            cur = build_mm(layer.o_w, attn_out);
            cur = ggml_clamp(ctx0, cur, -GRAD_CLIP, GRAD_CLIP);
            cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, layer.attn_post_norm_w);
            cur = ggml_add(ctx0, res, cur);
        }

        // ── Light Conv1d ──
        {
            auto * res = cur;
            cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, layer.norm_conv_w);

            cur = build_mm(layer.conv_pw1_w, cur); // [2*hidden, S]

            // GLU: split and gate
            {
                int64_t d = cur->ne[0] / 2;
                auto * data_part = ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], 0);
                auto * gate_part = ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], d * cur->nb[0]);
                gate_part = ggml_sigmoid(ctx0, gate_part);
                cur = ggml_mul(ctx0, data_part, gate_part);
            }

            // Manual depthwise conv (kernel_size=5, causal)
            // cur: [hidden, S] — need per-tap shifted copies
            {
                int kernel_size = 5;
                ggml_tensor * conv_out = nullptr;
                // Transpose kernel [K, D] → [D, K] for per-tap slicing
                auto * kern_t = ggml_cont(ctx0, ggml_permute(ctx0, layer.conv_dw_w, 1, 0, 2, 3)); // [D, K]

                for (int k = 0; k < kernel_size; k++) {
                    int shift = kernel_size - 1 - k; // causal shift
                    ggml_tensor * shifted;
                    if (shift == 0) {
                        shifted = cur;
                    } else {
                        // Trim last 'shift' time steps, left-pad with zeros
                        auto * trimmed = ggml_view_2d(ctx0, cur,
                            cur->ne[0], cur->ne[1] - shift,
                            cur->nb[1], 0);
                        trimmed = ggml_cont(ctx0, trimmed);
                        shifted = ggml_pad(ctx0, trimmed, 0, shift, 0, 0);
                        // Roll to left-pad
                        shifted = ggml_roll(ctx0, shifted, 0, shift, 0, 0);
                    }

                    // Extract tap weight: [D, 1]
                    auto * wk = ggml_view_2d(ctx0, kern_t,
                        kern_t->ne[0], 1,
                        kern_t->nb[1], k * kern_t->nb[1]);
                    wk = ggml_cont(ctx0, wk);

                    auto * term = ggml_mul(ctx0, shifted, wk); // [D, S] * [D, 1] broadcast
                    if (conv_out == nullptr) {
                        conv_out = term;
                    } else {
                        conv_out = ggml_add(ctx0, conv_out, term);
                    }
                }
                cur = conv_out;
            }

            cur = ggml_clamp(ctx0, cur, -GRAD_CLIP, GRAD_CLIP);
            cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, layer.conv_norm_w);
            cur = ggml_silu(ctx0, cur);

            cur = build_mm(layer.conv_pw2_w, cur);
            cur = ggml_add(ctx0, res, cur);
        }

        // ── FFW 2 (half-residual) ──
        {
            auto * res = cur;
            cur = ggml_clamp(ctx0, cur, -GRAD_CLIP, GRAD_CLIP);
            cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, layer.ff_norm_1_w);
            cur = build_mm(layer.ff_up_1_w, cur);
            cur = ggml_silu(ctx0, cur);
            cur = build_mm(layer.ff_down_1_w, cur);
            cur = ggml_clamp(ctx0, cur, -GRAD_CLIP, GRAD_CLIP);
            cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
            cur = ggml_mul(ctx0, cur, layer.ff_post_norm_1_w);
            cur = ggml_scale(ctx0, cur, RES_WEIGHT);
            cur = ggml_add(ctx0, res, cur);
        }

        // ── Final norm ──
        cur = ggml_clamp(ctx0, cur, -GRAD_CLIP, GRAD_CLIP);
        cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
        cur = ggml_mul(ctx0, cur, layer.ln_2_w);
    }

    // Output projection: 1024 → 1536
    cur = build_mm(model.pre_encode_out_w, cur);
    cur = ggml_add(ctx0, cur, model.pre_encode_out_b);

    // embed_audio: RMSNorm (no weight) + linear projection
    cur = ggml_rms_norm(ctx0, cur, NORM_EPS);
    cur = build_mm(model.mm_audio_inp_proj_w, cur);

    cb(cur, "projected", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
