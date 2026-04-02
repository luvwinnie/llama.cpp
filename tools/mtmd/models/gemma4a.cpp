#include "models.h"
#include <cmath>

// Gemma 4 audio: USM conformer with chunked local attention + relative position
// Implements HF's _convert_to_block, _extract_block_context, _rel_shift exactly

static constexpr float SOFTCAP     = 50.0f;
static constexpr float RESIDUAL_WT = 0.5f;
static constexpr int   N_REL_POS   = 13;
static constexpr int   CHUNK       = 12;
static constexpr int   MAX_PAST    = 12;
static constexpr int   CTX_SIZE    = 24;  // CHUNK + MAX_PAST

ggml_cgraph * clip_graph_gemma4a::build() {
    const float eps = hparams.eps;

    ggml_tensor * inp = build_inp_raw(1);
    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    cur = ggml_reshape_4d(ctx0, cur, cur->ne[0], cur->ne[1], 1, 1);

    // conv subsampling with LayerNorm + ReLU
    for (int ci = 0; ci < 2; ci++) {
        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[ci], cur, 2, 2, 1, 1, 1, 1);
        {
            int64_t ow = cur->ne[0], oh = cur->ne[1], oc = cur->ne[2];
            cur = ggml_reshape_4d(ctx0, cur, ow * oh, oc, 1, 1);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, model.conv_norm_w_arr[ci]);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
            cur = ggml_reshape_4d(ctx0, cur, ow, oh, oc, 1);
        }
        cur = ggml_relu(ctx0, cur);
    }

    // flatten + input projection
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
    cur = build_mm(model.audio_inp_proj_w, cur);

    const int64_t S = cur->ne[1];
    const int64_t NB = (S + CHUNK - 1) / CHUNK; // num blocks
    const int64_t S_PAD = NB * CHUNK;

    // Inputs
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N_REL_POS);
    ggml_set_name(pos_emb, "audio_pos_emb");
    ggml_set_input(pos_emb);
    ggml_build_forward_expand(gf, pos_emb);

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

        // === CHUNKED LOCAL ATTENTION ===
        {
            auto * res = cur;
            cur = ggml_rms_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, layer.attn_pre_norm_w);
            // cur: [hidden, S]

            // Q with per_dim_scale * q_scale
            ggml_tensor * Q = build_mm(layer.q_w, cur); // [hidden, S]
            {
                Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, S);
                auto * s = ggml_scale(ctx0, layer.per_dim_scale_w, q_scale);
                Q = ggml_mul(ctx0, Q, s);
            }
            // Q: [d_head, n_head, S] → permute to [d_head, S, n_head]
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

            // K with k_scale
            ggml_tensor * K_raw = build_mm(layer.k_w, cur);
            K_raw = ggml_scale(ctx0, K_raw, k_scale);
            K_raw = ggml_reshape_3d(ctx0, K_raw, d_head, n_head, S);
            K_raw = ggml_cont(ctx0, ggml_permute(ctx0, K_raw, 0, 2, 1, 3)); // [dh, S, nh]

            // V
            ggml_tensor * V_raw = build_mm(layer.v_w, cur);
            V_raw = ggml_reshape_3d(ctx0, V_raw, d_head, n_head, S);
            V_raw = ggml_cont(ctx0, ggml_permute(ctx0, V_raw, 0, 2, 1, 3)); // [dh, S, nh]

            // _convert_to_block(Q): pad S to S_PAD, reshape to [dh, CHUNK, nh, NB]
            ggml_tensor * Q_blk = Q;
            if (S_PAD > S) {
                Q_blk = ggml_pad(ctx0, Q_blk, 0, (int)(S_PAD - S), 0, 0); // pad dim1
            }
            Q_blk = ggml_reshape_4d(ctx0, Q_blk, d_head, CHUNK, n_head, NB);
            // permute to [dh, CHUNK, nh, NB] → we want [dh, CHUNK, NB, nh] for batched matmul
            // Actually: scores = Q_blk @ K_ctx^T per (block, head)
            // Let's arrange: [dh, CHUNK, NB, nh] so matmul over dh gives [CHUNK, CTX, NB, nh]
            // We need Q: [dh, CHUNK, NB*nh] and K: [dh, CTX, NB*nh]
            // Then matmul gives [CHUNK, CTX, NB*nh]
            // Reshape back to [CHUNK, CTX, NB, nh]

            // _extract_block_context(K): pad left by MAX_PAST, right by MAX_PAST+CHUNK-1=23
            // K: [dh, S, nh]
            ggml_tensor * K_pad = ggml_pad(ctx0, K_raw, 0, (int)(MAX_PAST), 0, 0); // left-pad dim1
            K_pad = ggml_pad(ctx0, K_pad, 0, (int)(CHUNK - 1), 0, 0); // right-pad by CHUNK-1
            // K_pad: [dh, S+MAX_PAST+CHUNK-1, nh] = [dh, S+23, nh]
            // Wait: ggml_pad pads at the END of each dimension. For left-padding we need a different approach.
            // ggml_pad(x, p0, p1, p2, p3) pads dim0 by p0, dim1 by p1, etc — ALL AT THE END.
            // For left-padding, we need to use ggml_pad then ggml_roll.

            // Actually, left-pad dim1 by MAX_PAST: pad end by MAX_PAST, then roll by MAX_PAST
            K_pad = ggml_pad(ctx0, K_raw, 0, (int)(MAX_PAST + CHUNK - 1), 0, 0); // pad dim1 end
            K_pad = ggml_roll(ctx0, K_pad, 0, (int)(MAX_PAST), 0, 0); // roll dim1 to left-pad

            // V same treatment
            ggml_tensor * V_pad = ggml_pad(ctx0, V_raw, 0, (int)(MAX_PAST + CHUNK - 1), 0, 0);
            V_pad = ggml_roll(ctx0, V_pad, 0, (int)(MAX_PAST), 0, 0);

            // Now extract NB windows of size CTX_SIZE with stride CHUNK along dim1
            // K_pad: [dh, S+MAX_PAST+CHUNK-1, nh]
            // For block b: K_ctx[b] = K_pad[:, b*CHUNK : b*CHUNK+CTX_SIZE, :]
            // We can do this with a strided view if ggml supports it
            // Alternatively: reshape K_pad to [dh, NB, CTX_SIZE+overlap, nh] using view tricks

            // Simpler approach: reshape K_pad to blocks using ggml_view + concat
            // Or: use im2col-like unfolding
            // Let's use manual extraction with views and concat

            // Total padded length: S + MAX_PAST + CHUNK - 1
            // For NB blocks with stride CHUNK and window CTX_SIZE:
            // Block b starts at b*CHUNK, length CTX_SIZE
            // This works if NB*CHUNK + CTX_SIZE - CHUNK <= padded_length
            // = S_PAD + CTX_SIZE - CHUNK = S_PAD + 12 = total padded = S + 23

            // Build context blocks by extracting views and concatenating
            // K_ctx: [dh, CTX_SIZE, nh, NB]
            ggml_tensor * k_blocks[64]; // max blocks
            ggml_tensor * v_blocks[64];
            GGML_ASSERT(NB <= 64);

            for (int b = 0; b < NB; b++) {
                int64_t start = b * CHUNK;
                // K_pad dim1 view at offset start, length CTX_SIZE
                k_blocks[b] = ggml_view_3d(ctx0, K_pad,
                    d_head, CTX_SIZE, n_head,
                    K_pad->nb[1], K_pad->nb[2],
                    start * K_pad->nb[1]);
                k_blocks[b] = ggml_cont(ctx0, k_blocks[b]);

                v_blocks[b] = ggml_view_3d(ctx0, V_pad,
                    d_head, CTX_SIZE, n_head,
                    V_pad->nb[1], V_pad->nb[2],
                    start * V_pad->nb[1]);
                v_blocks[b] = ggml_cont(ctx0, v_blocks[b]);
            }

            // Stack blocks: [dh, CTX, nh] × NB → [dh, CTX, nh*NB]
            ggml_tensor * K_ctx = k_blocks[0];
            for (int b = 1; b < NB; b++) {
                K_ctx = ggml_concat(ctx0, K_ctx, k_blocks[b], 2); // concat along dim2
            }
            // K_ctx: [dh, CTX, nh*NB]

            ggml_tensor * V_ctx = v_blocks[0];
            for (int b = 1; b < NB; b++) {
                V_ctx = ggml_concat(ctx0, V_ctx, v_blocks[b], 2);
            }

            // Q_blk: [dh, CHUNK, nh, NB] → reshape to [dh, CHUNK, nh*NB]
            Q_blk = ggml_reshape_3d(ctx0, Q_blk, d_head, CHUNK, n_head * NB);

            // Content attention: Q_blk @ K_ctx^T
            // Q_blk: [dh, CHUNK, nh*NB], K_ctx: [dh, CTX, nh*NB]
            // mul_mat(K_ctx, Q_blk) → [CTX, CHUNK, nh*NB] (ggml contracts over dim0)
            // Wait: ggml_mul_mat(a, b) = a^T @ b for each batch dim
            // For Q=[dh, CHUNK, nh*NB] and K=[dh, CTX, nh*NB]:
            // mul_mat(Q, K) transposes Q over dim0, giving [CHUNK, CTX, nh*NB]
            // That's scores[chunk_q, ctx_k, batch] — wrong order
            // We want scores[ctx_k, chunk_q, batch] so we can softmax over ctx_k
            // So: mul_mat(K_ctx, Q_blk) → [CTX, CHUNK, nh*NB] — but we need [CHUNK, CTX, nh*NB]
            // Actually ggml_mul_mat(a, b): result[i,j,k] = sum_d a[d,i,k] * b[d,j,k]
            // So mul_mat(K_ctx[dh,CTX,b], Q_blk[dh,CHUNK,b]) → [CTX, CHUNK, b]
            // For softmax over CTX (dim0), this is good!
            ggml_tensor * ac = ggml_mul_mat(ctx0, K_ctx, Q_blk); // [CTX, CHUNK, nh*NB]

            // Relative position: Q_flat @ rel_k^T → [CHUNK*NB, 13]
            // Then _rel_shift
            ggml_tensor * rel_k = build_mm(layer.attn_k_rel_w, pos_emb); // [hidden, 13]
            rel_k = ggml_reshape_3d(ctx0, rel_k, d_head, n_head, N_REL_POS);
            rel_k = ggml_cont(ctx0, ggml_permute(ctx0, rel_k, 0, 2, 1, 3)); // [dh, 13, nh]

            // Use original Q for relative position (before blocking)
            // Pad to S_PAD: [dh, S_PAD, nh]
            ggml_tensor * Q_padded = Q;
            if (S_PAD > S) {
                Q_padded = ggml_pad(ctx0, Q_padded, 0, (int)(S_PAD - S), 0, 0);
            }
            // Q_padded: [dh, S_PAD, nh]
            // Q @ rel_k^T: mul_mat(rel_k[dh,13,nh], Q_padded[dh,S_PAD,nh])
            // → [13, S_PAD, nh] — score for each relative distance per query per head
            ggml_tensor * qr = ggml_mul_mat(ctx0, rel_k, Q_padded); // [13, S_PAD, nh]

            // _rel_shift: reshape qr to [NB, CHUNK, 13, nh]
            // Then for each head: pad right to CTX+1-13=12, flatten, truncate, reshape
            // qr: [13, S_PAD, nh] → permute to [S_PAD, 13, nh] → reshape [NB, CHUNK, 13, nh]
            qr = ggml_cont(ctx0, ggml_permute(ctx0, qr, 1, 0, 2, 3)); // [S_PAD, 13, nh]
            qr = ggml_reshape_4d(ctx0, qr, CHUNK, N_REL_POS, n_head, NB); // Wait, dims wrong
            // Actually: [S_PAD, 13, nh] → reshape to [CHUNK, NB, 13, nh]
            // S_PAD = NB * CHUNK, so first dim splits into [CHUNK, NB]
            // But the sequence order matters! The first CHUNK elements are block 0's queries.
            // [S_PAD, 13, nh] with S_PAD=NB*CHUNK → reshape to [CHUNK, NB, 13, nh]
            // That's wrong — need [NB, CHUNK, 13, nh]
            // So: reshape to [CHUNK, NB, 13, nh] then permute [1,0,2,3] → [NB, CHUNK, 13, nh]
            qr = ggml_reshape_4d(ctx0, qr, CHUNK, NB, N_REL_POS, n_head);
            qr = ggml_cont(ctx0, ggml_permute(ctx0, qr, 0, 2, 1, 3)); // [CHUNK, 13, NB, nh]
            // Actually I need [NB, CHUNK, 13] per head for _rel_shift
            // Let me work per-head by flattening nh into the batch

            // Flatten: [CHUNK, 13, NB*nh]
            qr = ggml_reshape_3d(ctx0, qr, CHUNK, N_REL_POS, NB * n_head);

            // _rel_shift:
            // 1. Pad dim1 right by CTX_SIZE+1-N_REL_POS = 12
            qr = ggml_pad(ctx0, qr, 0, (int)(CTX_SIZE + 1 - N_REL_POS), 0, 0);
            // qr: [CHUNK, 25, NB*nh]

            // 2. Reshape to [CHUNK*(CTX_SIZE+1), NB*nh] = [300, NB*nh]
            qr = ggml_reshape_2d(ctx0, qr, CHUNK * (CTX_SIZE + 1), NB * n_head);

            // 3. Truncate to [CHUNK*CTX_SIZE, NB*nh] = [288, NB*nh]
            qr = ggml_view_2d(ctx0, qr,
                CHUNK * CTX_SIZE, NB * n_head,
                qr->nb[1], 0);
            qr = ggml_cont(ctx0, qr);

            // 4. Reshape to [CTX_SIZE, CHUNK, NB*nh]
            // Wait: HF reshapes to [NB, CHUNK, CTX_SIZE]
            // Our flatten was [CHUNK, 25] → [CHUNK*25] → truncate to [CHUNK*24] → reshape [CHUNK, CTX]
            // So: [CHUNK*CTX, NB*nh] → [CTX, CHUNK, NB*nh]
            // Hmm, need to be careful with reshape order
            // Original rel_shift: [NB, CHUNK, 25] → flatten last two → [NB, CHUNK*25] → truncate → [NB, CHUNK*24] → reshape [NB, CHUNK, 24]
            // Our data: [CHUNK, 25, NB*nh] → reshape [CHUNK*25, NB*nh] → truncate → [CHUNK*24, NB*nh] → reshape [CHUNK, 24, NB*nh] → permute [24, CHUNK, NB*nh]
            // Wait that gives [24, CHUNK, NB*nh] but we need [CTX, CHUNK, NB*nh]
            // CTX=24, so [24, CHUNK, NB*nh] is correct!
            // Reshape: [CHUNK*CTX, NB*nh] → [CHUNK, CTX, NB*nh]
            // IMPORTANT: HF reshapes (CHUNK, CTX) not (CTX, CHUNK)!
            qr = ggml_reshape_3d(ctx0, qr, CHUNK, CTX_SIZE, NB * n_head);
            // Permute to [CTX, CHUNK, NB*nh] to match content scores layout
            qr = ggml_cont(ctx0, ggml_permute(ctx0, qr, 1, 0, 2, 3));
            // Reorder batch from NB*nh to nh*NB
            qr = ggml_reshape_4d(ctx0, qr, CTX_SIZE, CHUNK, NB, n_head);
            qr = ggml_cont(ctx0, ggml_permute(ctx0, qr, 0, 1, 3, 2));
            qr = ggml_reshape_3d(ctx0, qr, CTX_SIZE, CHUNK, n_head * NB);

            // Now both ac and qr: [CTX, CHUNK, nh*NB] with same batch ordering
            ggml_tensor * scores = ggml_add(ctx0, ac, qr);

            // Softcap
            scores = ggml_scale(ctx0, scores, 1.0f / SOFTCAP);
            scores = ggml_tanh(ctx0, scores);
            scores = ggml_scale(ctx0, scores, SOFTCAP);

            // Softmax over dim0 (CTX_SIZE)
            scores = ggml_soft_max(ctx0, scores);

            // Attention output: scores @ V_ctx
            // scores: [CTX, CHUNK, NB*nh]
            // V_ctx: [dh, CTX, NB*nh] — need V transposed for matmul
            // We want: out = scores^T @ V_ctx^T ... hmm
            // Actually: for each batch b, out[j] = sum_i scores[i,j,b] * V[d,i,b]
            // This is: out[d,j,b] = V[d,:,b] @ scores[:,j,b]
            // = mul_mat(scores[CTX,CHUNK,b], V_ctx[dh,CTX,b]) but mul_mat transposes first arg
            // mul_mat(a[CTX,CHUNK,b], V[dh,CTX,b]) → [CHUNK, dh, b]... no, dim0 must match
            // mul_mat(a, b): result[i,j,k] = sum_d a[d,i,k] * b[d,j,k]
            // Need: sum over CTX. So CTX must be in dim0 of both.
            // scores: [CTX, CHUNK, NB*nh] — CTX in dim0 ✓
            // V_ctx: [dh, CTX, NB*nh] — CTX in dim1, need to permute
            // V_ctx_t: [CTX, dh, NB*nh]
            ggml_tensor * V_ctx_t = ggml_cont(ctx0, ggml_permute(ctx0, V_ctx, 1, 0, 2, 3));
            // mul_mat(scores[CTX,CHUNK,b], V_ctx_t[CTX,dh,b]) → [CHUNK, dh, b]
            ggml_tensor * attn_out = ggml_mul_mat(ctx0, scores, V_ctx_t); // [CHUNK, dh, NB*nh]

            // Reshape back: [CHUNK, dh, NB*nh] → [CHUNK, dh, nh, NB]
            attn_out = ggml_reshape_4d(ctx0, attn_out, CHUNK, d_head, n_head, NB);
            // Permute to [dh, CHUNK, nh, NB] → [dh*nh, CHUNK*NB] = [hidden, S_PAD]
            attn_out = ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 1, 0, 3, 2)); // [dh, CHUNK, NB, nh]
            attn_out = ggml_reshape_2d(ctx0, attn_out, d_head * n_head, CHUNK * NB); // [hidden, S_PAD]

            // Truncate to original seq length
            if (S_PAD > S) {
                attn_out = ggml_view_2d(ctx0, attn_out, n_embd, S, attn_out->nb[1], 0);
                attn_out = ggml_cont(ctx0, attn_out);
            }

            // Output projection + post-norm
            ggml_tensor * out = build_mm(layer.o_w, attn_out);
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
