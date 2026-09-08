// Copyright (c) 2026 codec.cpp contributors
// SPDX-License-Identifier: MIT
#include "models.h"
#include <cmath>
#include <cfloat>
#include <stdexcept>

namespace {
struct s3_attn_params { float scale = 0; };
enum s3_unary_op { S3_UNARY_SIGMOID, S3_UNARY_ELU, S3_UNARY_SILU, S3_UNARY_GELU_ERF, S3_UNARY_MISH };
static ggml_tensor * s3_cast_f32(ggml_context * ctx, ggml_tensor * x) {
    return x && x->type != GGML_TYPE_F32 ? ggml_cast(ctx, x, GGML_TYPE_F32) : x;
}
static ggml_tensor * s3_weight(ggml_context *, const clip_model * model, const std::string & name) {
    auto it = model->chatterbox_weights.find(name);
    if (it == model->chatterbox_weights.end()) { throw std::runtime_error("missing Chatterbox tensor: " + name); }
    return it->second;
}
constexpr int32_t kHiftInChannels = 80;
constexpr int32_t kHiftNFft = 16;
constexpr int32_t kHiftHop = 4;
constexpr int32_t kHiftNFftBins = kHiftNFft / 2 + 1;          // 9
constexpr int32_t kHiftNbHarmonics = 8;
constexpr float   kHiftNsfAlpha = 0.1f;
constexpr float   kHiftNsfSigma = 0.003f;
constexpr float   kHiftNsfVoicedThreshold = 10.0f;
constexpr float   kHiftLreluSlope = 0.1f;
constexpr float   kHiftLreluSlopeDefault = 0.01f; // PyTorch F.leaky_relu default
constexpr float   kHiftAudioLimit = 0.99f;
constexpr int32_t kHiftF0NumLayers = 5;
constexpr int32_t kHiftNumUps = 3;
constexpr int32_t kHiftUpsampleRates[kHiftNumUps] = {8, 5, 3};
constexpr int32_t kHiftUpsampleKernels[kHiftNumUps] = {16, 11, 7};
constexpr int32_t kHiftSourceDownStrides[kHiftNumUps] = {15, 3, 1};
constexpr int32_t kHiftSourceDownPads[kHiftNumUps] = {7, 1, 0};
constexpr int32_t kHiftResblockKernels[kHiftNumUps] = {3, 7, 11};

constexpr int32_t kHiftSourceResblockKernels[kHiftNumUps] = {7, 7, 11};
constexpr int32_t kHiftResblockDilations[3] = {1, 3, 5};

constexpr int32_t kHiftSourceUpsample = 480;

constexpr int32_t kFlowSpkEmbedDim = 192;
constexpr int32_t kFlowVocabSize = 6561;
constexpr int32_t kFlowEncoderHidden = 512;
constexpr int32_t kFlowEncoderLayers = 6;
constexpr int32_t kFlowEncoderUpLayers = 4;


constexpr int32_t kCfmInChannels = 320;        // packed [x, mu, spks, cond] = 80*4
constexpr int32_t kCfmOutChannels = 80;
constexpr int32_t kCfmChannels = 256;          // channels[0]
constexpr int32_t kCfmTimeEmbedDim = kCfmChannels * 4;  // 1024
constexpr int32_t kCfmNumMidBlocks = 12;
constexpr int32_t kCfmTransformersPerBlock = 4;
constexpr int32_t kCfmAttentionHeadDim = 64;
constexpr int32_t kCfmAttentionHeads = 8;
constexpr int32_t kCfmAttnInner = kCfmAttentionHeadDim * kCfmAttentionHeads;  // 512
constexpr float   kCfmTimeEmbedScale = 1000.0f;

static ggml_tensor * s3_conv1d_pointwise_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w) {

    if (ctx == nullptr || x == nullptr || w == nullptr) {
        return nullptr;
    }
    if (x->ne[1] != w->ne[1] || w->ne[0] != 1) {
        return nullptr;
    }

    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x));       // [c_in, t]
    ggml_tensor * w_ic = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);  // [c_in, c_out]
    ggml_tensor * y_ct = ggml_mul_mat(ctx, w_ic, x_ct);                // [c_out, t]
    if (y_ct == nullptr) {
        return nullptr;
    }

    return ggml_cont(ctx, ggml_transpose(ctx, y_ct));                  // [t, c_out]
}

static ggml_tensor * s3_conv1d_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    if (w->ne[0] == 1 && stride == 1 && dilation == 1 && padding == 0) {
        return s3_conv1d_pointwise_impl(ctx, x, w);
    }

    const ggml_type im2col_type = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im2col = ggml_im2col(ctx, w, x, stride, 0, padding, 0, dilation, 0, false, im2col_type);
    if (im2col == nullptr) {
        return nullptr;
    }

    ggml_tensor * lhs = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * rhs = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor * y = ggml_mul_mat(ctx, lhs, rhs);
    if (y == nullptr) {
        return nullptr;
    }

    return ggml_reshape_3d(ctx, y, im2col->ne[1], w->ne[2], im2col->ne[2]);
}

static ggml_tensor * s3_conv1d_prepare_w(ggml_context * ctx, ggml_tensor * w) {
    if (w == nullptr) {
        return nullptr;
    }
    if (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16) {
        return w;
    }
    return ggml_cast(ctx, w, GGML_TYPE_F32);
}

static ggml_tensor * s3_conv1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    w = s3_conv1d_prepare_w(ctx, w);
    b = s3_cast_f32(ctx, b);

    ggml_tensor * y = s3_conv1d_impl(ctx, x, w, stride, padding, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    y = ggml_cont(ctx, y);
    if (x->ne[2] <= 1 && y->ne[2] == 1 && y->ne[3] == 1) {
        y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    }
    return y;
}

static ggml_tensor * s3_pad_1d(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right) {
    if (ctx == nullptr || x == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    return ggml_pad_ext(ctx, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
}

static ggml_tensor * s3_conv1d_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    if (w->ne[0] < stride) {
        return nullptr;
    }

    w = s3_conv1d_prepare_w(ctx, w);
    b = s3_cast_f32(ctx, b);

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = s3_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = s3_conv1d_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return ggml_cont(ctx, y);
}

static ggml_tensor * s3_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t crop_left, int32_t crop_right) {
    if (ctx == nullptr || x == nullptr || crop_left < 0 || crop_right < 0) {
        return nullptr;
    }
    const int64_t out_t = x->ne[0] - crop_left - crop_right;
    if (out_t <= 0) {
        return nullptr;
    }
    ggml_tensor * view = ggml_view_2d(ctx, x, out_t, x->ne[1], x->nb[1], (size_t) crop_left * sizeof(float));
    return ggml_cont(ctx, view);
}

static ggml_tensor * s3_convtr1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) {
        w = ggml_cast(ctx, w, GGML_TYPE_F32);
    }
    b = s3_cast_f32(ctx, b);

    ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    if (padding > 0) {
        y = s3_crop_1d(ctx, y, padding, padding);
    }
    return ggml_cont(ctx, y);
}

static ggml_tensor * s3_layer_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma, ggml_tensor * beta) {
    if (ctx == nullptr || x_ct == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }

    gamma = s3_cast_f32(ctx, gamma);
    beta = s3_cast_f32(ctx, beta);

    ggml_tensor * y = ggml_norm(ctx, x_ct, eps);
    ggml_tensor * g2 = gamma;
    ggml_tensor * b2 = beta;
    y = ggml_mul(ctx, y, ggml_repeat(ctx, g2, y));
    y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    return y;
}

static ggml_tensor * s3_layer_norm_tc(ggml_context * ctx, ggml_tensor * x_tc, float eps, ggml_tensor * gamma, ggml_tensor * beta) {
    if (ctx == nullptr || x_tc == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }

    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x_tc));
    ggml_tensor * y_ct = s3_layer_norm_ct(ctx, x_ct, eps, gamma, beta);
    if (y_ct == nullptr) {
        return nullptr;
    }
    return ggml_cont(ctx, ggml_transpose(ctx, y_ct));
}

static ggml_tensor * s3_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    if (ctx == nullptr || x == nullptr || w == nullptr) {
        return nullptr;
    }

    ggml_tensor * w_f32 = s3_cast_f32(ctx, w);
    ggml_tensor * b_f32 = s3_cast_f32(ctx, b);

    ggml_tensor * y = ggml_mul_mat(ctx, w_f32, x);
    if (b_f32 != nullptr) {
        ggml_tensor * b2 = b_f32;
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return y;
}

static ggml_tensor * s3_lm_attn_ctx_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    const s3_attn_params * params) {

    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr) {
        return nullptr;
    }
    if (q_dth->ne[0] != k_dth->ne[0] || q_dth->ne[0] != v_dth->ne[0] ||
        q_dth->ne[1] != k_dth->ne[1] || q_dth->ne[1] != v_dth->ne[1] ||
        q_dth->ne[2] != k_dth->ne[2] || q_dth->ne[2] != v_dth->ne[2]) {
        return nullptr;
    }

    const int32_t head_dim = (int32_t) q_dth->ne[0];
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max(1, head_dim)));
    ggml_tensor * scores = ggml_mul_mat(ctx, ggml_cont(ctx, k_dth), q_dth);
    ggml_tensor * probs = ggml_soft_max(ctx, ggml_scale(ctx, scores, scale));
    ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, values, probs);
}

static ggml_tensor * s3_unary(ggml_context * ctx, ggml_tensor * x, s3_unary_op op) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }

    switch (op) {
        case S3_UNARY_SIGMOID: {
            ggml_tensor * x_half = ggml_scale(ctx, x, 0.5f);
            ggml_tensor * th = ggml_tanh(ctx, x_half);
            return ggml_scale_bias(ctx, th, 0.5f, 0.5f);
        }
        case S3_UNARY_ELU:
            return ggml_elu(ctx, x);
        case S3_UNARY_SILU:
            return ggml_silu(ctx, x);
        case S3_UNARY_GELU_ERF:
            return ggml_gelu_erf(ctx, x);
        case S3_UNARY_MISH: {
            ggml_tensor * sp = ggml_softplus(ctx, x);
            ggml_tensor * t = ggml_tanh(ctx, sp);
            return ggml_mul(ctx, x, t);
        }
        default:
            return nullptr;
    }
}

static ggml_tensor * s3_basic_transformer_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * norm1_w, ggml_tensor * norm1_b,
    ggml_tensor * qw, ggml_tensor * kw, ggml_tensor * vw,
    ggml_tensor * ow, ggml_tensor * ob,
    ggml_tensor * norm3_w, ggml_tensor * norm3_b,
    ggml_tensor * ff1_w, ggml_tensor * ff1_b,
    ggml_tensor * ff2_w, ggml_tensor * ff2_b,
    int32_t head_dim,
    int32_t num_heads) {
    if (ctx == nullptr || x_tc == nullptr || qw == nullptr || kw == nullptr || vw == nullptr ||
        ow == nullptr || ff1_w == nullptr || ff2_w == nullptr || head_dim <= 0 || num_heads <= 0) {
        return nullptr;
    }
    const int32_t inner = head_dim * num_heads;
    const int32_t T = (int32_t) x_tc->ne[0];

    ggml_tensor * h_tc = s3_layer_norm_tc(ctx, x_tc, 1e-5f, norm1_w, norm1_b);
    if (h_tc == nullptr) return nullptr;
    ggml_tensor * h_ct = ggml_cont(ctx, ggml_transpose(ctx, h_tc));
    ggml_tensor * q_ct = s3_linear(ctx, h_ct, qw, nullptr);
    ggml_tensor * k_ct = s3_linear(ctx, h_ct, kw, nullptr);
    ggml_tensor * v_ct = s3_linear(ctx, h_ct, vw, nullptr);
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr) return nullptr;

    auto to_dth = [&](ggml_tensor * x_ct_in) {
        ggml_tensor * r = ggml_reshape_3d(ctx, x_ct_in, head_dim, num_heads, T);
        return ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
    };
    s3_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    ggml_tensor * attn_dth = s3_lm_attn_ctx_dth(ctx, to_dth(q_ct), to_dth(k_ct), to_dth(v_ct), &attn_p);
    if (attn_dth == nullptr) return nullptr;

    ggml_tensor * attn_ct = ggml_reshape_2d(
        ctx,
        ggml_cont(ctx, ggml_permute(ctx, attn_dth, 0, 2, 1, 3)),
        inner, T);
    ggml_tensor * proj_ct = s3_linear(ctx, attn_ct, ow, ob);
    if (proj_ct == nullptr) return nullptr;
    x_tc = ggml_add(ctx, x_tc, ggml_cont(ctx, ggml_transpose(ctx, proj_ct)));

    ggml_tensor * f_tc = s3_layer_norm_tc(ctx, x_tc, 1e-5f, norm3_w, norm3_b);
    if (f_tc == nullptr) return nullptr;
    ggml_tensor * f_ct = ggml_cont(ctx, ggml_transpose(ctx, f_tc));
    ggml_tensor * ff = s3_linear(ctx, f_ct, ff1_w, ff1_b);
    if (ff == nullptr) return nullptr;
    ff = s3_unary(ctx, ff, S3_UNARY_GELU_ERF);
    ff = s3_linear(ctx, ff, ff2_w, ff2_b);
    if (ff == nullptr) return nullptr;
    return ggml_add(ctx, x_tc, ggml_cont(ctx, ggml_transpose(ctx, ff)));
}

static ggml_tensor * s3_causal_block1d_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * conv_w,
    ggml_tensor * conv_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b) {
    if (ctx == nullptr || x_tc == nullptr || conv_w == nullptr) return nullptr;
    ggml_tensor * y = s3_conv1d_causal(ctx, x_tc, conv_w, conv_b, 1, 1);
    if (y == nullptr) return nullptr;
    y = s3_layer_norm_tc(ctx, y, 1e-5f, ln_w, ln_b);
    if (y == nullptr) return nullptr;
    return s3_unary(ctx, y, S3_UNARY_MISH);
}

static ggml_tensor * s3_cfm_causal_resnet_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * t_emb,
    ggml_tensor * b1_conv_w, ggml_tensor * b1_conv_b,
    ggml_tensor * b1_ln_w,   ggml_tensor * b1_ln_b,
    ggml_tensor * b2_conv_w, ggml_tensor * b2_conv_b,
    ggml_tensor * b2_ln_w,   ggml_tensor * b2_ln_b,
    ggml_tensor * mlp_w,     ggml_tensor * mlp_b,
    ggml_tensor * res_w,     ggml_tensor * res_b) {
    if (ctx == nullptr || x_tc == nullptr || t_emb == nullptr ||
        b1_conv_w == nullptr || b2_conv_w == nullptr ||
        mlp_w == nullptr || res_w == nullptr) return nullptr;

    ggml_tensor * h = s3_causal_block1d_tc(ctx, x_tc, b1_conv_w, b1_conv_b, b1_ln_w, b1_ln_b);
    if (h == nullptr) return nullptr;

    ggml_tensor * tm = s3_unary(ctx, t_emb, S3_UNARY_MISH);
    if (tm == nullptr) return nullptr;
    ggml_tensor * tm_2d = ggml_reshape_2d(ctx, tm, tm->ne[0], 1);
    ggml_tensor * tm_proj = ggml_mul_mat(ctx, mlp_w, tm_2d);  // [out, 1]
    if (tm_proj == nullptr) return nullptr;
    if (mlp_b != nullptr) {
        ggml_tensor * mlp_b_2d = ggml_reshape_2d(ctx, mlp_b, mlp_b->ne[0], 1);
        tm_proj = ggml_add(ctx, tm_proj, mlp_b_2d);
    }
    ggml_tensor * tm_to = ggml_cont(ctx, ggml_transpose(ctx, tm_proj));  // [1, out]
    h = ggml_add(ctx, h, ggml_repeat(ctx, tm_to, h));

    h = s3_causal_block1d_tc(ctx, h, b2_conv_w, b2_conv_b, b2_ln_w, b2_ln_b);
    if (h == nullptr) return nullptr;

    ggml_tensor * res = s3_conv1d(ctx, x_tc, res_w, res_b, 1, 1, 0);
    if (res == nullptr) return nullptr;
    return ggml_add(ctx, h, res);
}

static ggml_tensor * s3_espnet_rel_pos_emb(
    ggml_context * ctx,
    int32_t t,
    int32_t d_model) {
    if (ctx == nullptr || t <= 0 || d_model <= 0 || (d_model & 1) != 0) return nullptr;
    const int32_t half = d_model / 2;
    const int32_t pe_len = 2 * t - 1;

    ggml_tensor * row_idx = ggml_arange(ctx, 0.0f, (float) pe_len, 1.0f);    // [pe_len]
    ggml_tensor * pos = ggml_scale_bias(ctx, row_idx, -1.0f, (float) (t - 1));

    ggml_tensor * k_idx = ggml_arange(ctx, 0.0f, (float) half, 1.0f);
    ggml_tensor * log_k = ggml_scale(ctx, k_idx, -2.0f * std::log(10000.0f) / (float) d_model);
    ggml_tensor * freqs = ggml_exp(ctx, log_k);                                // [half]

    ggml_tensor * pos_2d = ggml_reshape_2d(ctx, pos, 1, pe_len);
    ggml_tensor * freqs_2d = ggml_reshape_2d(ctx, freqs, half, 1);
    ggml_tensor * tmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, pe_len);
    ggml_tensor * angle = ggml_mul(ctx,
        ggml_repeat(ctx, freqs_2d, tmpl),
        ggml_repeat(ctx, pos_2d, tmpl));

    ggml_tensor * sin_3d = ggml_reshape_3d(ctx, ggml_sin(ctx, angle), half, 1, pe_len);
    ggml_tensor * cos_3d = ggml_reshape_3d(ctx, ggml_cos(ctx, angle), half, 1, pe_len);
    ggml_tensor * stacked = ggml_concat(ctx, sin_3d, cos_3d, 1);       // [half, 2, pe_len]
    ggml_tensor * permuted = ggml_cont(ctx, ggml_permute(ctx, stacked, 1, 0, 2, 3));
    return ggml_reshape_2d(ctx, permuted, d_model, pe_len);
}

static ggml_tensor * s3_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps) {
    if (ctx == nullptr || x == nullptr || alpha == nullptr) {
        return nullptr;
    }

    alpha = s3_cast_f32(ctx, alpha);

    ggml_tensor * alpha_2d = ggml_reshape_2d(ctx, alpha, 1, x->ne[1]);
    ggml_tensor * alpha_rep = ggml_repeat(ctx, alpha_2d, x);
    ggml_tensor * alpha_clamped = ggml_clamp(ctx, alpha_rep, eps, FLT_MAX);
    ggml_tensor * ax = ggml_mul(ctx, alpha_clamped, x);
    ggml_tensor * s = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    ggml_tensor * frac = ggml_div(ctx, s2, alpha_clamped);
    return ggml_add(ctx, x, frac);
}

static ggml_tensor * s3_hifigan_resblock_branch_ct(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * a1,
    ggml_tensor * a2,
    ggml_tensor * c1_w,
    ggml_tensor * c1_b,
    ggml_tensor * c2_w,
    ggml_tensor * c2_b,
    int32_t kernel_size,
    int32_t dilation) {
    if (ctx == nullptr || x_tc == nullptr || a1 == nullptr || a2 == nullptr ||
        c1_w == nullptr || c2_w == nullptr || kernel_size <= 0 || dilation <= 0) {
        return nullptr;
    }
    const int32_t pad1 = (kernel_size * dilation - dilation) / 2;
    const int32_t pad2 = (kernel_size - 1) / 2;
    ggml_tensor * h = s3_snake(ctx, x_tc, a1, 1e-9f);
    if (h == nullptr) return nullptr;
    h = s3_conv1d(ctx, h, c1_w, c1_b, 1, dilation, pad1);
    if (h == nullptr) return nullptr;
    h = s3_snake(ctx, h, a2, 1e-9f);
    if (h == nullptr) return nullptr;
    h = s3_conv1d(ctx, h, c2_w, c2_b, 1, 1, pad2);
    if (h == nullptr) return nullptr;
    return ggml_add(ctx, h, x_tc);
}

static ggml_tensor * s3_rel_shift_espnet(ggml_context * ctx, ggml_tensor * x) {
    if (ctx == nullptr || x == nullptr) return nullptr;
    const int64_t two_t1 = x->ne[0];
    const int64_t t = x->ne[1];
    const int64_t h = x->ne[2];
    if (two_t1 != 2 * t - 1) return nullptr;

    ggml_tensor * x_pad = ggml_pad_ext(ctx, x, 1, 0, 0, 0, 0, 0, 0, 0);
    if (x_pad == nullptr) return nullptr;

    ggml_tensor * x_pad_cont = ggml_cont(ctx, x_pad);
    ggml_tensor * x_view = ggml_reshape_3d(ctx, x_pad_cont, t, 2 * t, h);

    ggml_tensor * x_drop = ggml_view_3d(
        ctx, x_view,
        t, 2 * t - 1, h,
        x_view->nb[1],
        x_view->nb[2],
        x_view->nb[1]);
    x_drop = ggml_cont(ctx, x_drop);

    ggml_tensor * x_back = ggml_reshape_3d(ctx, x_drop, 2 * t - 1, t, h);
    ggml_tensor * x_out = ggml_view_3d(
        ctx, x_back,
        t, t, h,
        x_back->nb[1],
        x_back->nb[2],
        0);
    return ggml_cont(ctx, x_out);
}

static ggml_tensor * s3_lm_attn_rel_pos_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    ggml_tensor * p_dth,
    ggml_tensor * pos_bias_u,
    ggml_tensor * pos_bias_v,
    const s3_attn_params * params) {
    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr ||
        p_dth == nullptr || pos_bias_u == nullptr || pos_bias_v == nullptr) {
        return nullptr;
    }
    const int64_t head_dim = q_dth->ne[0];
    const int64_t t = q_dth->ne[1];
    const int64_t h = q_dth->ne[2];
    if (k_dth->ne[0] != head_dim || v_dth->ne[0] != head_dim || p_dth->ne[0] != head_dim ||
        k_dth->ne[1] != t || v_dth->ne[1] != t || p_dth->ne[1] != 2 * t - 1 ||
        k_dth->ne[2] != h || v_dth->ne[2] != h || p_dth->ne[2] != h) {
        return nullptr;
    }
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max<int64_t>(1, head_dim)));

    auto add_bias = [&](ggml_tensor * q, ggml_tensor * bias_dh) -> ggml_tensor * {
        ggml_tensor * b3 = ggml_reshape_3d(ctx, bias_dh, head_dim, 1, h);
        return ggml_add(ctx, q, ggml_repeat(ctx, b3, q));
    };
    ggml_tensor * q_u = add_bias(q_dth, pos_bias_u);
    ggml_tensor * q_v = add_bias(q_dth, pos_bias_v);
    if (q_u == nullptr || q_v == nullptr) return nullptr;

    ggml_tensor * mat_ac = ggml_mul_mat(ctx, ggml_cont(ctx, k_dth), q_u);
    ggml_tensor * mat_bd = ggml_mul_mat(ctx, ggml_cont(ctx, p_dth), q_v);
    if (mat_ac == nullptr || mat_bd == nullptr) return nullptr;
    mat_bd = s3_rel_shift_espnet(ctx, mat_bd);
    if (mat_bd == nullptr) return nullptr;

    ggml_tensor * scores = ggml_add(ctx, mat_ac, mat_bd);
    scores = ggml_scale(ctx, scores, scale);
    ggml_tensor * attn_w = ggml_soft_max(ctx, scores);

    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, v_tdh, attn_w);
}

static ggml_tensor * s3_sinusoidal_time_emb(
    ggml_context * ctx,
    float t_v,
    int32_t dim,
    float scale) {
    if (ctx == nullptr || dim <= 0 || (dim & 1) != 0) return nullptr;
    const int32_t half = dim / 2;
    ggml_tensor * idx = ggml_arange(ctx, 0.0f, (float) half, 1.0f);
    ggml_tensor * log_idx = ggml_scale(ctx, idx, -std::log(10000.0f) / (float) (half - 1));
    ggml_tensor * freqs = ggml_exp(ctx, log_idx);
    ggml_tensor * e = ggml_scale(ctx, freqs, t_v * scale);
    return ggml_concat(ctx, ggml_sin(ctx, e), ggml_cos(ctx, e), 0);
}
static ggml_tensor * s3_hift_f0_forward(
    ggml_context * ctx, ggml_tensor * mel_tc, const clip_model * model) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx, model, n); };
    ggml_tensor * x = mel_tc;
    for (int32_t i = 0; i < kHiftF0NumLayers; ++i) {
        ggml_tensor * w = W("a.gen.wav.hift.f0.cn." + std::to_string(i) + ".weight");
        ggml_tensor * b = W("a.gen.wav.hift.f0.cn." + std::to_string(i) + ".bias");
        if (w == nullptr || b == nullptr) return nullptr;
        x = s3_conv1d(ctx, x, w, b, 1, 1, 1);
        if (x == nullptr) return nullptr;
        x = s3_unary(ctx, x, S3_UNARY_ELU);
        if (x == nullptr) return nullptr;
    }
    ggml_tensor * cls_w = W("a.gen.wav.hift.f0.cls.weight");
    ggml_tensor * cls_b = W("a.gen.wav.hift.f0.cls.bias");
    if (cls_w == nullptr || cls_b == nullptr) return nullptr;
    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor * f0 = ggml_mul_mat(ctx, cls_w, x_ct);
    if (f0 == nullptr) return nullptr;
    f0 = ggml_add(ctx, f0, cls_b);
    f0 = ggml_abs(ctx, f0);
    return ggml_reshape_1d(ctx, f0, x->ne[0]);
}


static ggml_tensor * s3_apply_resblock(
    ggml_context * ctx_eval,
    ggml_tensor * x,
    const clip_model * model,
    const std::string & prefix,
    int32_t kernel_size) {
    for (int32_t idx = 0; idx < 3; ++idx) {
        const int32_t d = kHiftResblockDilations[idx];
        ggml_tensor * a1 = s3_weight(ctx_eval, model, prefix + ".a1." + std::to_string(idx));
        ggml_tensor * a2 = s3_weight(ctx_eval, model, prefix + ".a2." + std::to_string(idx));
        ggml_tensor * c1w = s3_weight(ctx_eval, model, prefix + ".cv1." + std::to_string(idx) + ".weight");
        ggml_tensor * c1b = s3_weight(ctx_eval, model, prefix + ".cv1." + std::to_string(idx) + ".bias");
        ggml_tensor * c2w = s3_weight(ctx_eval, model, prefix + ".cv2." + std::to_string(idx) + ".weight");
        ggml_tensor * c2b = s3_weight(ctx_eval, model, prefix + ".cv2." + std::to_string(idx) + ".bias");
        x = s3_hifigan_resblock_branch_ct(ctx_eval, x, a1, a2, c1w, c1b, c2w, c2b, kernel_size, d);
        if (x == nullptr) return nullptr;
    }
    return x;
}

static ggml_tensor * s3_hift_main_forward(
    ggml_context * ctx_eval,
    ggml_tensor * t_mel,
    ggml_tensor * t_stft,
    const clip_model * model) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx_eval, model, n); };
    if (t_mel == nullptr || t_stft == nullptr) return nullptr;

    ggml_tensor * conv_pre_w = W("a.gen.wav.hift.conv_pre.weight");
    ggml_tensor * conv_pre_b = W("a.gen.wav.hift.conv_pre.bias");
    if (conv_pre_w == nullptr || conv_pre_b == nullptr) return nullptr;
    ggml_tensor * x = s3_conv1d(ctx_eval, t_mel, conv_pre_w, conv_pre_b, 1, 1, 3);
    if (x == nullptr) return nullptr;

    for (int32_t i = 0; i < kHiftNumUps; ++i) {
        const int32_t u = kHiftUpsampleRates[i];
        const int32_t k = kHiftUpsampleKernels[i];
        const int32_t up_pad = (k - u) / 2;

        x = ggml_leaky_relu(ctx_eval, x, kHiftLreluSlope, false);

        ggml_tensor * up_w = W("a.gen.wav.hift.up." + std::to_string(i) + ".weight");
        ggml_tensor * up_b = W("a.gen.wav.hift.up." + std::to_string(i) + ".bias");
        if (up_w == nullptr || up_b == nullptr) return nullptr;
        x = s3_convtr1d(ctx_eval, x, up_w, up_b, u, up_pad, 1);
        if (x == nullptr) return nullptr;

        if (i == kHiftNumUps - 1) {
            const int32_t T = (int32_t) x->ne[0];
            ggml_tensor * x_left = ggml_view_2d(
                ctx_eval, x,
                1, x->ne[1],
                x->nb[1],
                (size_t) 1 * x->nb[0]);
            x_left = ggml_cont(ctx_eval, x_left);
            ggml_tensor * x_full = ggml_view_2d(
                ctx_eval, x,
                T, x->ne[1],
                x->nb[1],
                0);
            x_full = ggml_cont(ctx_eval, x_full);
            x = ggml_concat(ctx_eval, x_left, x_full, 0);
            if (x == nullptr) return nullptr;
        }

        ggml_tensor * sd_w = W("a.gen.wav.hift.src_dn." + std::to_string(i) + ".weight");
        ggml_tensor * sd_b = W("a.gen.wav.hift.src_dn." + std::to_string(i) + ".bias");
        if (sd_w == nullptr || sd_b == nullptr) return nullptr;
        ggml_tensor * si = s3_conv1d(
            ctx_eval, t_stft, sd_w, sd_b,
            kHiftSourceDownStrides[i],
            1,
            kHiftSourceDownPads[i]);
        if (si == nullptr) return nullptr;

        si = s3_apply_resblock(
            ctx_eval, si, model,
            "a.gen.wav.hift.src_rb." + std::to_string(i),
            kHiftSourceResblockKernels[i]);
        if (si == nullptr) return nullptr;

        if (si->ne[0] != x->ne[0]) {
            const int32_t common = (int32_t) std::min<int64_t>(si->ne[0], x->ne[0]);
            ggml_tensor * si_trim = ggml_cont(ctx_eval, ggml_view_2d(
                ctx_eval, si,
                common, si->ne[1],
                si->nb[1],
                0));
            ggml_tensor * x_trim = ggml_cont(ctx_eval, ggml_view_2d(
                ctx_eval, x,
                common, x->ne[1],
                x->nb[1],
                0));
            si = si_trim;
            x = x_trim;
        }
        x = ggml_add(ctx_eval, x, si);

        ggml_tensor * xs = nullptr;
        for (int32_t j = 0; j < 3; ++j) {
            ggml_tensor * branch = s3_apply_resblock(
                ctx_eval, x, model,
                "a.gen.wav.hift.rb." + std::to_string(i * 3 + j),
                kHiftResblockKernels[j]);
            if (branch == nullptr) return nullptr;
            xs = (xs == nullptr) ? branch : ggml_add(ctx_eval, xs, branch);
        }
        x = ggml_scale(ctx_eval, xs, 1.0f / 3.0f);
    }

    x = ggml_leaky_relu(ctx_eval, x, kHiftLreluSlopeDefault, false);

    ggml_tensor * cp_w = W("a.gen.wav.hift.conv_post.weight");
    ggml_tensor * cp_b = W("a.gen.wav.hift.conv_post.bias");
    if (cp_w == nullptr || cp_b == nullptr) return nullptr;
    x = s3_conv1d(ctx_eval, x, cp_w, cp_b, 1, 1, 3);  // [t_pcm/4, 18]
    if (x == nullptr) return nullptr;
    return x;
}




static ggml_tensor * s3_cfm_causal_resnet(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * t_emb,
    const clip_model * model,
    const std::string & prefix) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx, model, n); };
    return s3_cfm_causal_resnet_block_tc(ctx, x_tc, t_emb,
        W(prefix + ".b1.cv.weight"), W(prefix + ".b1.cv.bias"),
        W(prefix + ".b1.ln.weight"), W(prefix + ".b1.ln.bias"),
        W(prefix + ".b2.cv.weight"), W(prefix + ".b2.cv.bias"),
        W(prefix + ".b2.ln.weight"), W(prefix + ".b2.ln.bias"),
        W(prefix + ".mlp.weight"),   W(prefix + ".mlp.bias"),
        W(prefix + ".res.weight"),   W(prefix + ".res.bias"));
}

static ggml_tensor * s3_cfm_basic_transformer(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    const clip_model * model,
    const std::string & prefix) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx, model, n); };
    return s3_basic_transformer_block_tc(ctx, x_tc,
        W(prefix + ".norm1.weight"), W(prefix + ".norm1.bias"),
        W(prefix + ".attn.q.weight"), W(prefix + ".attn.k.weight"), W(prefix + ".attn.v.weight"),
        W(prefix + ".attn.o.weight"), W(prefix + ".attn.o.bias"),
        W(prefix + ".norm3.weight"), W(prefix + ".norm3.bias"),
        W(prefix + ".ff.w1.weight"), W(prefix + ".ff.w1.bias"),
        W(prefix + ".ff.w2.weight"), W(prefix + ".ff.w2.bias"),
        kCfmAttentionHeadDim, kCfmAttentionHeads);
}


static ggml_tensor * s3_cfm_time_emb(ggml_context * ctx, float t_v, const clip_model * model) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx, model, n); };
    ggml_tensor * t_emb_in = s3_sinusoidal_time_emb(ctx, t_v, kCfmInChannels, kCfmTimeEmbedScale);
    if (t_emb_in == nullptr) return nullptr;

    ggml_tensor * t_l1w = W("a.gen.wav.cfm.t.l1.weight");
    ggml_tensor * t_l1b = W("a.gen.wav.cfm.t.l1.bias");
    ggml_tensor * t_l2w = W("a.gen.wav.cfm.t.l2.weight");
    ggml_tensor * t_l2b = W("a.gen.wav.cfm.t.l2.bias");
    if (t_l1w == nullptr || t_l1b == nullptr || t_l2w == nullptr || t_l2b == nullptr) return nullptr;
    ggml_tensor * te_2d = ggml_reshape_2d(ctx, t_emb_in, kCfmInChannels, 1);
    ggml_tensor * te = s3_linear(ctx, te_2d, t_l1w, t_l1b);
    if (te == nullptr) return nullptr;
    te = s3_unary(ctx, te, S3_UNARY_SILU);
    te = s3_linear(ctx, te, t_l2w, t_l2b);
    if (te == nullptr) return nullptr;
    return ggml_reshape_1d(ctx, te, kCfmTimeEmbedDim);
}

static ggml_tensor * s3_cfm_estimator_forward(
    ggml_context * ctx,
    ggml_tensor * x_in,    // [T, 80]
    ggml_tensor * mu_in,   // [T, 80]
    ggml_tensor * spks_in, // [80]
    ggml_tensor * cond_in, // [T, 80]
    ggml_tensor * t_emb,   // [time_embed_dim]
    const clip_model * model) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx, model, n); };
    if (x_in == nullptr || mu_in == nullptr || spks_in == nullptr || cond_in == nullptr || t_emb == nullptr) {
        return nullptr;
    }

    ggml_tensor * spks_2d = ggml_reshape_2d(ctx, spks_in, 1, kCfmOutChannels);
    ggml_tensor * spks_rep = ggml_repeat(ctx, spks_2d, x_in);
    ggml_tensor * pack1 = ggml_concat(ctx, x_in, mu_in, 1);
    ggml_tensor * pack2 = ggml_concat(ctx, pack1, spks_rep, 1);
    ggml_tensor * x_tc  = ggml_concat(ctx, pack2, cond_in, 1);
    if (x_tc == nullptr) return nullptr;

    ggml_tensor * skip = nullptr;
    {
        const std::string p_dn = "a.gen.wav.cfm.dn.0";
        x_tc = s3_cfm_causal_resnet(ctx, x_tc, t_emb, model, p_dn + ".r");
        if (x_tc == nullptr) return nullptr;
        for (int32_t ti = 0; ti < kCfmTransformersPerBlock; ++ti) {
            x_tc = s3_cfm_basic_transformer(ctx, x_tc, model, p_dn + ".t." + std::to_string(ti));
            if (x_tc == nullptr) return nullptr;
        }
        skip = x_tc;
        ggml_tensor * dnw = W(p_dn + ".x.weight");
        ggml_tensor * dnb = W(p_dn + ".x.bias");
        if (dnw == nullptr || dnb == nullptr) return nullptr;
        x_tc = s3_conv1d_causal(ctx, x_tc, dnw, dnb, 1, 1);
        if (x_tc == nullptr) return nullptr;
    }

    for (int32_t bi = 0; bi < kCfmNumMidBlocks; ++bi) {
        const std::string p_md = "a.gen.wav.cfm.md." + std::to_string(bi);
        x_tc = s3_cfm_causal_resnet(ctx, x_tc, t_emb, model, p_md + ".r");
        if (x_tc == nullptr) return nullptr;
        for (int32_t ti = 0; ti < kCfmTransformersPerBlock; ++ti) {
            x_tc = s3_cfm_basic_transformer(ctx, x_tc, model, p_md + ".t." + std::to_string(ti));
            if (x_tc == nullptr) return nullptr;
        }
    }

    {
        const std::string p_up = "a.gen.wav.cfm.up.0";
        x_tc = ggml_concat(ctx, x_tc, skip, 1);
        if (x_tc == nullptr) return nullptr;
        x_tc = s3_cfm_causal_resnet(ctx, x_tc, t_emb, model, p_up + ".r");
        if (x_tc == nullptr) return nullptr;
        for (int32_t ti = 0; ti < kCfmTransformersPerBlock; ++ti) {
            x_tc = s3_cfm_basic_transformer(ctx, x_tc, model, p_up + ".t." + std::to_string(ti));
            if (x_tc == nullptr) return nullptr;
        }
        ggml_tensor * upw = W(p_up + ".x.weight");
        ggml_tensor * upb = W(p_up + ".x.bias");
        if (upw == nullptr || upb == nullptr) return nullptr;
        x_tc = s3_conv1d_causal(ctx, x_tc, upw, upb, 1, 1);
        if (x_tc == nullptr) return nullptr;
    }

    ggml_tensor * fcw = W("a.gen.wav.cfm.final.cv.weight");
    ggml_tensor * fcb = W("a.gen.wav.cfm.final.cv.bias");
    ggml_tensor * flw = W("a.gen.wav.cfm.final.ln.weight");
    ggml_tensor * flb = W("a.gen.wav.cfm.final.ln.bias");
    ggml_tensor * pw  = W("a.gen.wav.cfm.proj.weight");
    ggml_tensor * pb  = W("a.gen.wav.cfm.proj.bias");
    if (fcw == nullptr || fcb == nullptr || flw == nullptr || flb == nullptr || pw == nullptr || pb == nullptr) return nullptr;
    x_tc = s3_causal_block1d_tc(ctx, x_tc, fcw, fcb, flw, flb);
    if (x_tc == nullptr) return nullptr;
    return s3_conv1d(ctx, x_tc, pw, pb, 1, 1, 0);
}



static ggml_tensor * s3_flow_pre_lookahead(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    const clip_model * model) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx, model, n); };
    ggml_tensor * c1w = W("a.gen.wav.flow.enc.pre.cv1.weight");
    ggml_tensor * c1b = W("a.gen.wav.flow.enc.pre.cv1.bias");
    ggml_tensor * c2w = W("a.gen.wav.flow.enc.pre.cv2.weight");
    ggml_tensor * c2b = W("a.gen.wav.flow.enc.pre.cv2.bias");
    if (c1w == nullptr || c1b == nullptr || c2w == nullptr || c2b == nullptr) return nullptr;

    ggml_tensor * h = s3_pad_1d(ctx, x_tc, 0, 3);
    if (h == nullptr) return nullptr;
    h = s3_conv1d(ctx, h, c1w, c1b, 1, 1, 0);
    if (h == nullptr) return nullptr;
    h = ggml_leaky_relu(ctx, h, 0.01f, false);
    h = s3_pad_1d(ctx, h, 2, 0);
    if (h == nullptr) return nullptr;
    h = s3_conv1d(ctx, h, c2w, c2b, 1, 1, 0);
    if (h == nullptr) return nullptr;
    return ggml_add(ctx, h, x_tc);
}

static ggml_tensor * s3_flow_up_layer(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    const clip_model * model) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx, model, n); };
    ggml_tensor * uw = W("a.gen.wav.flow.enc.up.weight");
    ggml_tensor * ub = W("a.gen.wav.flow.enc.up.bias");
    if (uw == nullptr || ub == nullptr) return nullptr;

    const int64_t t2 = x_tc->ne[0] * 2;
    const int64_t c = x_tc->ne[1];
    ggml_tensor * up = ggml_interpolate(ctx, x_tc, t2, c, x_tc->ne[2], x_tc->ne[3], GGML_SCALE_MODE_NEAREST);
    if (up == nullptr) return nullptr;

    ggml_tensor * up_pad = s3_pad_1d(ctx, up, 4, 0);
    if (up_pad == nullptr) return nullptr;
    return s3_conv1d(ctx, up_pad, uw, ub, 1, 1, 0);
}

static ggml_tensor * s3_flow_conformer_block(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * pos_emb,  // [d, pe_len] column-major (d=hidden)
    const clip_model * model,
    const std::string & prefix) {
    auto W = [&](const std::string & n) -> ggml_tensor * { return s3_weight(ctx, model, n); };
    ggml_tensor * nmw = W(prefix + ".norm_mha.weight");
    ggml_tensor * nmb = W(prefix + ".norm_mha.bias");
    ggml_tensor * nfw = W(prefix + ".norm_ff.weight");
    ggml_tensor * nfb = W(prefix + ".norm_ff.bias");
    ggml_tensor * qw = W(prefix + ".attn.q.weight");
    ggml_tensor * qb = W(prefix + ".attn.q.bias");
    ggml_tensor * kw = W(prefix + ".attn.k.weight");
    ggml_tensor * kb = W(prefix + ".attn.k.bias");
    ggml_tensor * vw = W(prefix + ".attn.v.weight");
    ggml_tensor * vb = W(prefix + ".attn.v.bias");
    ggml_tensor * ow = W(prefix + ".attn.o.weight");
    ggml_tensor * ob = W(prefix + ".attn.o.bias");
    ggml_tensor * pw = W(prefix + ".attn.pos.weight");
    ggml_tensor * pbu = W(prefix + ".attn.pbu");
    ggml_tensor * pbv = W(prefix + ".attn.pbv");
    ggml_tensor * f1w = W(prefix + ".ff.w1.weight");
    ggml_tensor * f1b = W(prefix + ".ff.w1.bias");
    ggml_tensor * f2w = W(prefix + ".ff.w2.weight");
    ggml_tensor * f2b = W(prefix + ".ff.w2.bias");
    if (nmw == nullptr || nfw == nullptr || qw == nullptr || pw == nullptr || pbu == nullptr) return nullptr;

    const int32_t T = (int32_t) x_tc->ne[0];

    ggml_tensor * h_tc = s3_layer_norm_tc(ctx, x_tc, 1e-12f, nmw, nmb);
    if (h_tc == nullptr) return nullptr;
    ggml_tensor * h_ct = ggml_cont(ctx, ggml_transpose(ctx, h_tc));  // [c, t]

    ggml_tensor * q_ct = s3_linear(ctx, h_ct, qw, qb);
    ggml_tensor * k_ct = s3_linear(ctx, h_ct, kw, kb);
    ggml_tensor * v_ct = s3_linear(ctx, h_ct, vw, vb);
    ggml_tensor * p_ct = s3_linear(ctx, pos_emb, pw, nullptr);  // [c, pe_len]
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr || p_ct == nullptr) return nullptr;

    auto to_dth = [&](ggml_tensor * x_ct_in, int32_t T_in) -> ggml_tensor * {
        ggml_tensor * r = ggml_reshape_3d(ctx, x_ct_in, kCfmAttentionHeadDim, kCfmAttentionHeads, T_in);
        return ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
    };

    ggml_tensor * q_dth = to_dth(q_ct, T);
    ggml_tensor * k_dth = to_dth(k_ct, T);
    ggml_tensor * v_dth = to_dth(v_ct, T);
    ggml_tensor * p_dth = to_dth(p_ct, 2 * T - 1);
    if (q_dth == nullptr || k_dth == nullptr || v_dth == nullptr || p_dth == nullptr) return nullptr;

    s3_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) kCfmAttentionHeadDim);
    ggml_tensor * attn_dth = s3_lm_attn_rel_pos_dth(ctx, q_dth, k_dth, v_dth, p_dth, pbu, pbv, &attn_p);
    if (attn_dth == nullptr) return nullptr;

    ggml_tensor * attn_ct = ggml_reshape_2d(
        ctx,
        ggml_cont(ctx, ggml_permute(ctx, attn_dth, 0, 2, 1, 3)),
        kCfmAttnInner, T);
    ggml_tensor * proj_ct = s3_linear(ctx, attn_ct, ow, ob);  // [c, t]
    if (proj_ct == nullptr) return nullptr;
    ggml_tensor * proj_tc = ggml_cont(ctx, ggml_transpose(ctx, proj_ct));
    x_tc = ggml_add(ctx, x_tc, proj_tc);

    ggml_tensor * ff_tc = s3_layer_norm_tc(ctx, x_tc, 1e-12f, nfw, nfb);
    if (ff_tc == nullptr) return nullptr;
    ggml_tensor * ff_ct = ggml_cont(ctx, ggml_transpose(ctx, ff_tc));
    ggml_tensor * ff = s3_linear(ctx, ff_ct, f1w, f1b);  // [ff_inner, t]
    if (ff == nullptr) return nullptr;
    ff = s3_unary(ctx, ff, S3_UNARY_SILU);
    ff = s3_linear(ctx, ff, f2w, f2b);  // [c, t]
    if (ff == nullptr) return nullptr;
    ggml_tensor * ff_out_tc = ggml_cont(ctx, ggml_transpose(ctx, ff));
    return ggml_add(ctx, x_tc, ff_out_tc);
}

static ggml_tensor * s3_encode(ggml_context * ctx_eval, const clip_model * model, int T) {
    auto W = [&](const std::string & n) { return s3_weight(ctx_eval, model, n); };
    const int T_up = T * 2;
    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, T);
    ggml_set_name(t_tok, "a.gen.wav.flow.tokens");
    ggml_set_input(t_tok);
    ggml_tensor * t_pe1 = s3_espnet_rel_pos_emb(ctx_eval, T, kFlowEncoderHidden);
    ggml_tensor * t_pe2 = s3_espnet_rel_pos_emb(ctx_eval, T_up, kFlowEncoderHidden);
    if (t_pe1 == nullptr || t_pe2 == nullptr) return nullptr;

    ggml_tensor * emb_table = W("a.gen.wav.flow.input_emb.weight");
    if (emb_table == nullptr) return nullptr;
    ggml_tensor * x_ct = ggml_get_rows(ctx_eval, emb_table, t_tok);  // [hidden, T]
    if (x_ct == nullptr) return nullptr;

    ggml_tensor * el_w = W("a.gen.wav.flow.enc.embed.lin.weight");
    ggml_tensor * el_b = W("a.gen.wav.flow.enc.embed.lin.bias");
    ggml_tensor * en_w = W("a.gen.wav.flow.enc.embed.ln.weight");
    ggml_tensor * en_b = W("a.gen.wav.flow.enc.embed.ln.bias");
    if (el_w == nullptr || el_b == nullptr || en_w == nullptr || en_b == nullptr) return nullptr;
    ggml_tensor * h_proj = s3_linear(ctx_eval, x_ct, el_w, el_b);
    if (h_proj == nullptr) return nullptr;
    ggml_tensor * h_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, h_proj));
    h_tc = s3_layer_norm_tc(ctx_eval, h_tc, 1e-5f, en_w, en_b);
    if (h_tc == nullptr) return nullptr;
    h_tc = ggml_scale(ctx_eval, h_tc, std::sqrt((float) kFlowEncoderHidden));

    h_tc = s3_flow_pre_lookahead(ctx_eval, h_tc, model);
    if (h_tc == nullptr) return nullptr;

    for (int32_t bi = 0; bi < kFlowEncoderLayers; ++bi) {
        h_tc = s3_flow_conformer_block(ctx_eval, h_tc, t_pe1, model,
                                              "a.gen.wav.flow.enc.blk." + std::to_string(bi));
        if (h_tc == nullptr) return nullptr;
    }

    h_tc = s3_flow_up_layer(ctx_eval, h_tc, model);
    if (h_tc == nullptr) return nullptr;

    ggml_tensor * eul_w = W("a.gen.wav.flow.enc.up_embed.lin.weight");
    ggml_tensor * eul_b = W("a.gen.wav.flow.enc.up_embed.lin.bias");
    ggml_tensor * eun_w = W("a.gen.wav.flow.enc.up_embed.ln.weight");
    ggml_tensor * eun_b = W("a.gen.wav.flow.enc.up_embed.ln.bias");
    if (eul_w == nullptr || eul_b == nullptr || eun_w == nullptr || eun_b == nullptr) return nullptr;
    {
        ggml_tensor * h_ct2 = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, h_tc));
        ggml_tensor * h_proj2 = s3_linear(ctx_eval, h_ct2, eul_w, eul_b);
        if (h_proj2 == nullptr) return nullptr;
        h_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, h_proj2));
        h_tc = s3_layer_norm_tc(ctx_eval, h_tc, 1e-5f, eun_w, eun_b);
        if (h_tc == nullptr) return nullptr;
        h_tc = ggml_scale(ctx_eval, h_tc, std::sqrt((float) kFlowEncoderHidden));
    }

    for (int32_t bi = 0; bi < kFlowEncoderUpLayers; ++bi) {
        h_tc = s3_flow_conformer_block(ctx_eval, h_tc, t_pe2, model,
                                              "a.gen.wav.flow.enc.up_blk." + std::to_string(bi));
        if (h_tc == nullptr) return nullptr;
    }

    ggml_tensor * an_w = W("a.gen.wav.flow.enc.after_norm.weight");
    ggml_tensor * an_b = W("a.gen.wav.flow.enc.after_norm.bias");
    if (an_w == nullptr || an_b == nullptr) return nullptr;
    h_tc = s3_layer_norm_tc(ctx_eval, h_tc, 1e-5f, an_w, an_b);
    if (h_tc == nullptr) return nullptr;

    ggml_tensor * pj_w = W("a.gen.wav.flow.proj.weight");
    ggml_tensor * pj_b = W("a.gen.wav.flow.proj.bias");
    if (pj_w == nullptr || pj_b == nullptr) return nullptr;
    ggml_tensor * h_ct_after = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, h_tc));
    ggml_tensor * mu_ct = s3_linear(ctx_eval, h_ct_after, pj_w, pj_b);  // [80, T_total]
    if (mu_ct == nullptr) return nullptr;
    ggml_tensor * mu_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, mu_ct));  // [T_total, 80]
    ggml_set_name(mu_tc, "a.gen.wav.flow.mu");

    return mu_tc;
}
static ggml_tensor * s3_denoise(ggml_context * ctx_eval, const clip_model * model, int T_total, float t_v, float r_v, bool unconditional, const clip_chatterbox_reference * ref) {
    auto W = [&](const std::string & n) { return s3_weight(ctx_eval, model, n); };
    const int mel_len1 = ref ? ref->decoder_mel.size() / 80 : W("a.gen.wav.cond.prompt_feat")->ne[1];
    auto * mu_tc = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, T_total, 80);
    ggml_set_name(mu_tc, "a.gen.wav.flow.mu");
    ggml_set_input(mu_tc);
    auto * x = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, T_total, 80);
    ggml_set_name(x, "a.gen.wav.flow.noise_z");
    ggml_set_input(x);
    ggml_tensor * sp_emb = ref ? ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 192) : W("a.gen.wav.cond.embedding");
    if (ref) { ggml_set_name(sp_emb, "inp_reference_embedding"); ggml_set_input(sp_emb); }           // ne[0]=192, ne[1]=1
    ggml_tensor * sa_w = W("a.gen.wav.flow.spk_aff.weight");
    ggml_tensor * sa_b = W("a.gen.wav.flow.spk_aff.bias");
    if (sp_emb == nullptr || sa_w == nullptr || sa_b == nullptr) return nullptr;
    ggml_tensor * sp_normed = ggml_rms_norm(ctx_eval, sp_emb, 1e-12f);
    sp_normed = ggml_scale(ctx_eval, sp_normed, 1.0f / std::sqrt((float) kFlowSpkEmbedDim));
    ggml_tensor * sp_proj = s3_linear(ctx_eval, sp_normed, sa_w, sa_b);  // [80, 1]
    ggml_tensor * spks = ggml_reshape_1d(ctx_eval, sp_proj, kCfmOutChannels);
    ggml_set_name(spks, "a.gen.wav.flow.spks");

    ggml_tensor * pf = ref ? ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, 80, mel_len1) : W("a.gen.wav.cond.prompt_feat");
    if (ref) { ggml_set_name(pf, "inp_reference_mel"); ggml_set_input(pf); }
    if (pf == nullptr) return nullptr;
    ggml_tensor * pf_2d = ggml_reshape_2d(ctx_eval, pf, kCfmOutChannels, mel_len1);
    ggml_tensor * pf_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, pf_2d));  // [mel_len1, 80]
    const int32_t tail = T_total - mel_len1;
    ggml_tensor * cond = ggml_pad(ctx_eval, pf_tc, tail, 0, 0, 0);
    ggml_set_name(cond, "a.gen.wav.flow.cond");


    auto * t_emb = s3_cfm_time_emb(ctx_eval, t_v, model);
    if (model->hparams.chatterbox_meanflow) {
        auto * r_emb = s3_cfm_time_emb(ctx_eval, r_v, model);
        t_emb = ggml_mul_mat(ctx_eval, W("a.gen.wav.cfm.t_mix.weight"), ggml_concat(ctx_eval, t_emb, r_emb, 0));
    }
    if (unconditional) {
        mu_tc = ggml_scale(ctx_eval, mu_tc, 0);
        spks = ggml_scale(ctx_eval, spks, 0);
        cond = ggml_scale(ctx_eval, cond, 0);
    }
    return s3_cfm_estimator_forward(ctx_eval, x, mu_tc, spks, cond, t_emb, model);
}
static ggml_tensor * s3_vocode(ggml_context * ctx_eval, const clip_model * model, int T_total, const clip_chatterbox_reference * ref) {
    auto W = [&](const std::string & n) { return s3_weight(ctx_eval, model, n); };
    const int mel_len1 = ref ? ref->decoder_mel.size() / 80 : W("a.gen.wav.cond.prompt_feat")->ne[1];
    const int sample_rate = 24000;
    auto * x = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, T_total, 80);
    ggml_set_name(x, "a.gen.wav.flow.noise_z");
    ggml_set_input(x);
    const int32_t T_speech = T_total - mel_len1;
    ggml_tensor * mel_full = ggml_cont(ctx_eval, x);
    ggml_tensor * mel_view = ggml_view_2d(
        ctx_eval, mel_full,
        T_speech, kCfmOutChannels,
        mel_full->nb[1],
        (size_t) mel_len1 * mel_full->nb[0]);
    ggml_tensor * mel = ggml_cont(ctx_eval, mel_view);
    ggml_set_name(mel, "a.gen.wav.flow.mel");

    ggml_tensor * f0 = s3_hift_f0_forward(ctx_eval, mel, model);
    if (f0 == nullptr) return nullptr;

    const int32_t T_pcm = T_speech * kHiftSourceUpsample;
    const int32_t T_stft = T_pcm / kHiftHop + 1;
    const int32_t kHarm = kHiftNbHarmonics + 1;
    ggml_tensor * t_phase = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, kHarm);                    // [9]
    ggml_tensor * t_nsf_noise = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, T_pcm, kHarm);         // [T_pcm, 9]
    ggml_set_name(t_phase, "a.gen.wav.hift.nsf_phase");
    ggml_set_input(t_phase);
    ggml_set_name(t_nsf_noise, "a.gen.wav.hift.nsf_noise");
    ggml_set_input(t_nsf_noise);
    auto * t_basis_re_k = W("a.gen.wav.hift.stft_basis_re_k");
    auto * t_basis_im_k = W("a.gen.wav.hift.stft_basis_im_k");
    auto * t_istft_re = W("a.gen.wav.hift.istft_basis_re");
    auto * t_istft_im = W("a.gen.wav.hift.istft_basis_im");
    auto * t_hann = W("a.gen.wav.hift.hann");
    auto * t_ola_w = W("a.gen.wav.hift.ola_w");

    ggml_tensor * f0_4d = ggml_reshape_4d(ctx_eval, f0, T_speech, 1, 1, 1);
    ggml_tensor * f0_pcm_4d = ggml_interpolate(ctx_eval, f0_4d, T_pcm, 1, 1, 1, GGML_SCALE_MODE_NEAREST);
    ggml_tensor * f0_pcm = ggml_reshape_2d(ctx_eval, f0_pcm_4d, T_pcm, 1);  // [T_pcm, 1]

    ggml_tensor * h_idx = ggml_arange(ctx_eval, 1.0f, (float) (kHarm + 1), 1.0f);  // [9] = [1..9]
    ggml_tensor * scales = ggml_scale(ctx_eval, h_idx, 1.0f / (float) sample_rate);
    ggml_tensor * scales_2d = ggml_reshape_2d(ctx_eval, scales, 1, kHarm);
    ggml_tensor * f_harm_template = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, T_pcm, kHarm);
    ggml_tensor * f_harm = ggml_mul(ctx_eval,
        ggml_repeat(ctx_eval, f0_pcm, f_harm_template),
        ggml_repeat(ctx_eval, scales_2d, f_harm_template));

    ggml_tensor * theta = ggml_scale(ctx_eval, ggml_cumsum(ctx_eval, f_harm), 2.0f * (float) M_PI);

    ggml_tensor * phase_2d = ggml_reshape_2d(ctx_eval, t_phase, 1, kHarm);
    ggml_tensor * sine_waves = ggml_sin(ctx_eval,
        ggml_add(ctx_eval, theta, ggml_repeat(ctx_eval, phase_2d, f_harm_template)));
    sine_waves = ggml_scale(ctx_eval, sine_waves, kHiftNsfAlpha);

    ggml_tensor * uv = ggml_step(ctx_eval, ggml_scale_bias(ctx_eval, f0_pcm, 1.0f, -kHiftNsfVoicedThreshold));
    ggml_tensor * uv_bcast = ggml_repeat(ctx_eval, uv, f_harm_template);

    ggml_tensor * noise_amp = ggml_scale_bias(ctx_eval, uv_bcast,
        kHiftNsfSigma - kHiftNsfAlpha / 3.0f,
        kHiftNsfAlpha / 3.0f);
    ggml_tensor * noise = ggml_mul(ctx_eval, noise_amp, t_nsf_noise);

    ggml_tensor * waves = ggml_add(ctx_eval, ggml_mul(ctx_eval, sine_waves, uv_bcast), noise);
    ggml_tensor * lin_w = W("a.gen.wav.hift.src.lin.weight");  // [1, 9] PyTorch to ggml ne[0]=9, ne[1]=1.
    ggml_tensor * lin_b = W("a.gen.wav.hift.src.lin.bias");
    if (lin_w == nullptr || lin_b == nullptr) return nullptr;
    ggml_tensor * waves_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, waves));  // [9, T_pcm]
    ggml_tensor * sine_merge_ct = s3_linear(ctx_eval, waves_ct, lin_w, lin_b);  // [1, T_pcm]
    ggml_tensor * sine_merge_1d = ggml_tanh(ctx_eval, ggml_reshape_1d(ctx_eval, sine_merge_ct, T_pcm));

    ggml_tensor * sm_2d = ggml_reshape_2d(ctx_eval, sine_merge_1d, T_pcm, 1);          // [T_pcm, 1]
    ggml_tensor * sm_padded = s3_pad_1d(ctx_eval, sm_2d,
        kHiftNFft / 2, kHiftNFft / 2);                       // [T_pcm + n_fft, 1]
    ggml_tensor * stft_re = s3_conv1d(ctx_eval, sm_padded, t_basis_re_k, nullptr,
        kHiftHop, 1, 0);                            // [T_stft, n_bins]
    ggml_tensor * stft_im = s3_conv1d(ctx_eval, sm_padded, t_basis_im_k, nullptr,
        kHiftHop, 1, 0);                            // [T_stft, n_bins]
    if (stft_re == nullptr || stft_im == nullptr) return nullptr;
    ggml_tensor * s_stft = ggml_concat(ctx_eval, stft_re, stft_im, 1);          // [T_stft, n_fft+2]
    (void) T_stft;

    ggml_tensor * head = s3_hift_main_forward(ctx_eval, mel, s_stft, model);     // [T_head, n_fft+2]
    if (head == nullptr) return nullptr;

    const int32_t T_head = (int32_t) head->ne[0];
    ggml_tensor * head_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, head));  // [n_fft+2, T_head]
    ggml_tensor * mag_log = ggml_view_2d(ctx_eval, head_ct, kHiftNFftBins, T_head, head_ct->nb[1], 0);
    ggml_tensor * phase_v = ggml_view_2d(ctx_eval, head_ct, kHiftNFftBins, T_head, head_ct->nb[1],
        (size_t) kHiftNFftBins * head_ct->nb[0]);
    mag_log = ggml_cont(ctx_eval, mag_log);
    phase_v = ggml_cont(ctx_eval, phase_v);

    ggml_tensor * mag = ggml_exp(ctx_eval, ggml_clamp(ctx_eval, mag_log, -1e30f, 1e2f));
    ggml_tensor * phase_sin = ggml_sin(ctx_eval, phase_v);
    ggml_tensor * re_F = ggml_mul(ctx_eval, mag, ggml_cos(ctx_eval, phase_sin));   // [n_bins, T_head]
    ggml_tensor * im_F = ggml_mul(ctx_eval, mag, ggml_sin(ctx_eval, phase_sin));

    ggml_tensor * frame_re = s3_linear(ctx_eval, re_F, t_istft_re, nullptr); // [n_fft, T_head]
    ggml_tensor * frame_im = s3_linear(ctx_eval, im_F, t_istft_im, nullptr);
    if (frame_re == nullptr || frame_im == nullptr) return nullptr;
    ggml_tensor * frame = ggml_scale(ctx_eval, ggml_sub(ctx_eval, frame_re, frame_im), 1.0f / (float) kHiftNFft);

    ggml_tensor * windowed = frame;

    ggml_tensor * windowed_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, windowed));
    ggml_tensor * ola_signal = s3_convtr1d(ctx_eval, windowed_tc, t_ola_w, nullptr,
        kHiftHop, 0, 1);                       // [T_pcm + n_fft, 1]

    ggml_tensor * hann_sq = ggml_mul(ctx_eval, t_hann, t_hann);
    ggml_tensor * hann_sq_2d = ggml_reshape_2d(ctx_eval, hann_sq, kHiftNFft, 1);
    ggml_tensor * env_template = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, T_head, kHiftNFft);
    ggml_tensor * env_in = ggml_repeat(ctx_eval, ggml_cont(ctx_eval, ggml_transpose(ctx_eval, hann_sq_2d)), env_template);
    ggml_tensor * env = s3_convtr1d(ctx_eval, env_in, t_ola_w, nullptr,
        kHiftHop, 0, 1);

    ggml_tensor * env_safe = ggml_clamp(ctx_eval, env, 1e-11f, 1e30f);
    ggml_tensor * signal = ggml_div(ctx_eval, ola_signal, env_safe);                // [out_size, 1]

    const int32_t out_size = (T_head - 1) * kHiftHop + kHiftNFft;
    const int32_t pad = kHiftNFft / 2;
    const int32_t pcm_len = out_size - 2 * pad;
    ggml_tensor * signal_cont = ggml_cont(ctx_eval, signal);
    ggml_tensor * signal_view = ggml_view_2d(ctx_eval, signal_cont, pcm_len, 1,
        signal_cont->nb[1], (size_t) pad * signal_cont->nb[0]);
    ggml_tensor * pcm = ggml_reshape_1d(ctx_eval, ggml_cont(ctx_eval, signal_view), pcm_len);
    pcm = ggml_clamp(ctx_eval, pcm, -kHiftAudioLimit, kHiftAudioLimit);
    ggml_set_name(pcm, "a.gen.wav.flow.pcm");

    return pcm;
}


// T3 conditioning: speaker projection, optional Perceiver, emotion, text and BOS.
static ggml_tensor * s3_t3_prompt(ggml_context * ctx, const clip_model * model, int n_text, const clip_chatterbox_reference * ref) {
    auto W = [&](const std::string & n) { return s3_weight(ctx, model, "a.gen.wav.t3." + n); };
    const bool turbo = model->hparams.chatterbox_meanflow;
    auto * speaker_emb = ref ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256) : W("cond.speaker_emb");
    auto * codes = ref ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ref->t3_codes.size()) : ggml_cast(ctx, W("cond.cond_prompt_speech_tokens"), GGML_TYPE_I32);
    if (ref) {
        ggml_set_name(speaker_emb, "inp_reference_speaker"); ggml_set_input(speaker_emb);
        ggml_set_name(codes, "inp_reference_codes"); ggml_set_input(codes);
    }
    auto * speaker = s3_linear(ctx, speaker_emb, W("cond_enc.spkr_enc.weight"), W("cond_enc.spkr_enc.bias"));
    auto * prompt = ggml_get_rows(ctx, W("speech_emb.weight"), codes);
    if (!turbo) {
        auto * positions = ggml_cast(ctx, ggml_arange(ctx, 0, (float) codes->ne[0], 1), GGML_TYPE_I32);
        prompt = ggml_add(ctx, prompt, ggml_get_rows(ctx, W("speech_pos_emb.emb.weight"), positions));
        auto * query = W("cond_enc.perceiver.pre_attention_query");
        auto attention = [&](ggml_tensor * q_input, ggml_tensor * kv_input) {
            auto norm = [&](ggml_tensor * x) {
                return s3_layer_norm_ct(ctx, x, 1e-5f, W("cond_enc.perceiver.attn.norm.weight"), W("cond_enc.perceiver.attn.norm.bias"));
            };
            auto project = [&](ggml_tensor * x, const std::string & name) {
                auto * y = s3_linear(ctx, x, W("cond_enc.perceiver.attn.to_" + name + ".weight"), W("cond_enc.perceiver.attn.to_" + name + ".bias"));
                return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, y, 256, 4, x->ne[1]), 0, 2, 1, 3));
            };
            auto * q = project(norm(q_input), "q");
            auto * normalized = norm(kv_input);
            auto * k = project(normalized, "k");
            auto * v = project(normalized, "v");
            auto * scores = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / 16));
            auto * h = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), scores);
            h = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, h, 0, 2, 1, 3)), 1024, q_input->ne[1]);
            h = s3_linear(ctx, h, W("cond_enc.perceiver.attn.proj_out.weight"), W("cond_enc.perceiver.attn.proj_out.bias"));
            return ggml_add(ctx, s3_cast_f32(ctx, q_input), h);
        };
        prompt = attention(query, prompt);
        prompt = attention(prompt, prompt);
    }
    auto * prefix = ggml_concat(ctx, speaker, prompt, 1);
    if (!turbo) {
        auto * emotion = s3_linear(ctx, W("cond.emotion_adv"), W("cond_enc.emotion_adv_fc.weight"), nullptr);
        prefix = ggml_concat(ctx, prefix, emotion, 1);
    }
    auto * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text);
    ggml_set_name(tokens, "inp_tokens");
    ggml_set_input(tokens);
    auto * text = ggml_get_rows(ctx, W("text_emb.weight"), tokens);
    auto * uncond = ggml_scale(ctx, text, 0);
    auto * bos_id = ggml_cast(ctx, ggml_arange(ctx, 6561, 6562, 1), GGML_TYPE_I32);
    auto * bos = ggml_get_rows(ctx, W("speech_emb.weight"), bos_id);
    if (!turbo) {
        auto * positions = ggml_cast(ctx, ggml_arange(ctx, 0, (float) n_text, 1), GGML_TYPE_I32);
        auto * pos = ggml_get_rows(ctx, W("text_pos_emb.emb.weight"), positions);
        text = ggml_add(ctx, text, pos);
        uncond = ggml_add(ctx, uncond, pos);
        auto * zero = ggml_cast(ctx, ggml_arange(ctx, 0, 1, 1), GGML_TYPE_I32);
        bos = ggml_add(ctx, bos, ggml_get_rows(ctx, W("speech_pos_emb.emb.weight"), zero));
        // The reference inference appends BOS again after prepare_input_embeds.
        bos = ggml_concat(ctx, bos, bos, 1);
    }
    auto * output = ggml_concat(ctx, ggml_concat(ctx, prefix, text, 1), bos, 1);
    if (!turbo) {
        auto * negative = ggml_concat(ctx, ggml_concat(ctx, prefix, uncond, 1), bos, 1);
        output = ggml_concat(ctx, output, negative, 1);
    }
    return output;
}

static ggml_tensor * s3_t3_next(ggml_context * ctx, const clip_model * model, int code, int position) {
    auto W = [&](const std::string & n) { return s3_weight(ctx, model, "a.gen.wav.t3." + n); };
    auto * id = ggml_cast(ctx, ggml_arange(ctx, (float) code, (float) code + 1, 1), GGML_TYPE_I32);
    auto * x = ggml_get_rows(ctx, W("speech_emb.weight"), id);
    if (!model->hparams.chatterbox_meanflow) {
        auto * pos = ggml_cast(ctx, ggml_arange(ctx, (float) position, (float) position + 1, 1), GGML_TYPE_I32);
        x = ggml_add(ctx, x, ggml_get_rows(ctx, W("speech_pos_emb.emb.weight"), pos));
    }
    return x;
}

static ggml_tensor * s3_reference_tokens(ggml_context * ctx, const clip_model * model, int n_frames) {
    auto W = [&](const std::string & n) { return s3_weight(ctx, model, "a.gen.wav.ref.tok." + n); };
    auto * mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, 128);
    ggml_set_name(mel, "inp_features"); ggml_set_input(mel);
    auto * x = s3_conv1d(ctx, mel, W("enc.conv1.weight"), W("enc.conv1.bias"), 2, 1, 1);
    x = ggml_gelu_erf(ctx, x);
    x = s3_conv1d(ctx, x, W("enc.conv2.weight"), W("enc.conv2.bias"), 2, 1, 1);
    x = ggml_cont(ctx, ggml_transpose(ctx, ggml_gelu_erf(ctx, x)));
    const int T = x->ne[1];
    auto * positions = ggml_cast(ctx, ggml_arange(ctx, 0, (float) T, 1), GGML_TYPE_I32);
    for (int i = 0; i < 6; ++i) {
        const std::string base = "enc.blk." + std::to_string(i) + ".";
        auto norm = [&](ggml_tensor * h, const std::string & n) {
            return s3_layer_norm_ct(ctx, h, 1e-5f, W(base + n + ".weight"), W(base + n + ".bias"));
        };
        auto linear = [&](ggml_tensor * h, const std::string & n, bool bias = true) {
            return s3_linear(ctx, h, W(base + n + ".weight"), bias ? W(base + n + ".bias") : nullptr);
        };
        auto * h = norm(x, "attn_ln");
        auto * q = linear(h, "attn.query");
        auto * k = linear(h, "attn.key", false);
        auto * v = linear(h, "attn.value");
        auto split = [&](ggml_tensor * z, bool rope) {
            z = ggml_reshape_3d(ctx, z, 64, 20, T);
            if (rope) { z = ggml_rope_ext(ctx, z, positions, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000, 1, 0, 1, 0, 0); }
            return ggml_cont(ctx, ggml_permute(ctx, z, 0, 2, 1, 3));
        };
        s3_attn_params ap{}; ap.scale = 0.125f;
        auto * attention = s3_lm_attn_ctx_dth(ctx, split(q, true), split(k, true), split(v, false), &ap);
        attention = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attention, 0, 2, 1, 3)), 1280, T);
        attention = linear(attention, "attn.out");
        auto * fsmn = ggml_conv_1d_dw(ctx, W(base + "attn.fsmn_block.weight"), ggml_cont(ctx, ggml_transpose(ctx, v)), 1, 15, 1);
        fsmn = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, fsmn, T, 1280)));
        x = ggml_add(ctx, x, ggml_add(ctx, attention, ggml_add(ctx, fsmn, v)));
        h = ggml_gelu_erf(ctx, linear(norm(x, "mlp_ln"), "mlp.0"));
        x = ggml_add(ctx, x, linear(h, "mlp.2"));
    }
    x = s3_linear(ctx, x, W("proj.weight"), W("proj.bias"));
    x = ggml_scale_bias(ctx, ggml_round(ctx, ggml_scale(ctx, ggml_tanh(ctx, x), 0.9990000128746033f)), 1, 1);
    auto * powers = ggml_exp(ctx, ggml_scale(ctx, ggml_arange(ctx, 0, 8, 1), std::log(3.0f)));
    powers = ggml_round(ctx, powers);
    x = ggml_sum_rows(ctx, ggml_mul(ctx, x, powers));
    return ggml_cast(ctx, ggml_reshape_1d(ctx, x, T), GGML_TYPE_I32);
}

// Each LSTM layer is a separate graph, batched over the overlapping 160-frame windows.
static ggml_tensor * s3_voice_encoder(ggml_context * ctx, const clip_model * model, int n_partials, int layer) {
    auto W = [&](const std::string & n) { return s3_weight(ctx, model, "a.gen.wav.ref.ve." + n); };
    const int D = layer == 0 ? 40 : 256;
    const std::string base = "blk." + std::to_string(layer) + ".";
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_partials, 160);
    ggml_set_name(input, "inp_features"); ggml_set_input(input);
    auto * gates = ggml_mul_mat(ctx, W(base + "input.weight"), input);
    gates = ggml_add(ctx, gates, W(base + "bias"));
    auto * zero = ggml_scale(ctx, ggml_arange(ctx, 0, 256.0f * n_partials, 1), 0);
    auto * h = ggml_reshape_2d(ctx, zero, 256, n_partials);
    auto * c = h;
    ggml_tensor * sequence = nullptr;
    for (int t = 0; t < 160; ++t) {
        auto * wx = ggml_view_2d(ctx, gates, 1024, n_partials, gates->nb[1], t * gates->nb[2]);
        auto * g = ggml_add(ctx, wx, ggml_mul_mat(ctx, W(base + "recurrent.weight"), h));
        auto slice = [&](int i) { return ggml_cont(ctx, ggml_view_2d(ctx, g, 256, n_partials, g->nb[1], i * 256 * g->nb[0])); };
        auto * gate_i = ggml_sigmoid(ctx, slice(0));
        auto * gate_f = ggml_sigmoid(ctx, slice(1));
        auto * gate_g = ggml_tanh(ctx, slice(2));
        auto * gate_o = ggml_sigmoid(ctx, slice(3));
        c = ggml_add(ctx, ggml_mul(ctx, gate_f, c), ggml_mul(ctx, gate_i, gate_g));
        h = ggml_mul(ctx, gate_o, ggml_tanh(ctx, c));
        if (layer < 2) {
            auto * row = ggml_reshape_3d(ctx, h, 256, n_partials, 1);
            sequence = sequence ? ggml_concat(ctx, sequence, row, 2) : row;
        }
    }
    if (layer < 2) { return sequence; }
    auto * embedding = ggml_relu(ctx, s3_linear(ctx, h, W("proj.weight"), W("proj.bias")));
    embedding = ggml_scale(ctx, ggml_rms_norm(ctx, embedding, 1e-12f), 1.0f / 16);
    embedding = ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, embedding)));
    embedding = ggml_cont(ctx, ggml_transpose(ctx, embedding));
    return ggml_scale(ctx, ggml_rms_norm(ctx, embedding, 1e-12f), 1.0f / 16);
}

static ggml_tensor * s3_campplus(ggml_context * ctx, const clip_model * model, int T) {
    auto W = [&](const std::string & name, bool required = true) -> ggml_tensor * {
        auto it = model->chatterbox_weights.find("a.gen.wav.ref.camp." + name);
        if (it != model->chatterbox_weights.end()) { return it->second; }
        if (required) { throw std::runtime_error("missing CAMPPlus tensor " + name); }
        return nullptr;
    };
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, 80);
    ggml_set_name(input, "inp_features"); ggml_set_input(input);
    auto affine = [&](ggml_tensor * x, const std::string & name, bool conv2d) {
        auto * w = W(name + ".weight");
        auto * b = W(name + ".bias");
        if (conv2d) { w = ggml_reshape_3d(ctx, w, 1, 1, w->ne[0]); b = ggml_reshape_3d(ctx, b, 1, 1, b->ne[0]); }
        else { w = ggml_reshape_2d(ctx, w, 1, w->ne[0]); b = ggml_reshape_2d(ctx, b, 1, b->ne[0]); }
        return ggml_add(ctx, ggml_mul(ctx, x, w), b);
    };
    auto conv2d = [&](ggml_tensor * x, const std::string & name, int stride, int pad) {
        return ggml_conv_2d(ctx, W(name + ".weight"), x, 1, stride, pad, pad, 1, 1);
    };
    auto * x = ggml_reshape_3d(ctx, input, T, 80, 1);
    x = ggml_relu(ctx, affine(conv2d(x, "head.conv1", 1, 1), "head.bn1", true));
    for (int stage = 1; stage <= 2; ++stage) {
        for (int layer = 0; layer < 2; ++layer) {
            const std::string base = "head.layer" + std::to_string(stage) + "." + std::to_string(layer) + ".";
            auto * residual = x;
            auto * h = ggml_relu(ctx, affine(conv2d(x, base + "conv1", layer == 0 ? 2 : 1, 1), base + "bn1", true));
            h = affine(conv2d(h, base + "conv2", 1, 1), base + "bn2", true);
            if (layer == 0) { residual = affine(conv2d(x, base + "shortcut.0", 2, 0), base + "shortcut.1", true); }
            x = ggml_relu(ctx, ggml_add(ctx, h, residual));
        }
    }
    x = ggml_relu(ctx, affine(conv2d(x, "head.conv2", 2, 1), "head.bn2", true));
    x = ggml_reshape_2d(ctx, x, T, 320);
    auto conv = [&](ggml_tensor * h, const std::string & name, int stride = 1, int dilation = 1, int pad = 0) {
        return s3_conv1d(ctx, h, W(name + ".weight"), W(name + ".bias", false), stride, dilation, pad);
    };
    auto nonlinear = [&](ggml_tensor * h, const std::string & name) { return ggml_relu(ctx, affine(h, name, false)); };
    x = nonlinear(conv(x, "x.tdnn.linear", 2, 1, 2), "x.tdnn.n");
    const int n_layers[] = {12, 24, 16};
    for (int stage = 0; stage < 3; ++stage) {
        for (int layer = 1; layer <= n_layers[stage]; ++layer) {
            const std::string base = "x.block" + std::to_string(stage + 1) + ".d" + std::to_string(layer) + ".";
            auto * h = conv(nonlinear(x, base + "n1"), base + "linear1");
            h = nonlinear(h, base + "n2");
            auto * local = conv(h, base + "cam.linear_local", 1, stage == 0 ? 1 : 2, stage == 0 ? 1 : 2);
            auto * mean = ggml_mean(ctx, h);
            ggml_tensor * segments = nullptr;
            for (int start = 0; start < h->ne[0]; start += 100) {
                const int length = std::min<int64_t>(100, h->ne[0] - start);
                auto * segment = ggml_cont(ctx, ggml_view_2d(ctx, h, length, h->ne[1], h->nb[1], start * h->nb[0]));
                auto * average = ggml_mean(ctx, segment);
                segment = ggml_repeat(ctx, average, segment);
                segments = segments ? ggml_concat(ctx, segments, segment, 0) : segment;
            }
            auto * context = ggml_add(ctx, segments, mean);
            context = ggml_relu(ctx, conv(context, base + "cam.linear1"));
            context = ggml_sigmoid(ctx, conv(context, base + "cam.linear2"));
            x = ggml_concat(ctx, x, ggml_mul(ctx, local, context), 1);
        }
        const std::string base = "x.transit" + std::to_string(stage + 1);
        x = conv(nonlinear(x, base + ".n"), base + ".linear");
    }
    x = nonlinear(x, "x.out_n");
    auto * mean = ggml_mean(ctx, x);
    auto * centered = ggml_sub(ctx, x, mean);
    auto * variance = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, centered)), 1.0f / (x->ne[0] - 1));
    auto * stats = ggml_concat(ctx, mean, ggml_sqrt(ctx, variance), 1);
    return ggml_reshape_1d(ctx, affine(conv(stats, "x.dense.linear"), "x.dense.n", false), 192);
}

} // namespace

ggml_cgraph * clip_graph_chatterbox_gen::build() {
    ggml_tensor * output = nullptr;
    if (params.stage == clip_chatterbox_stage::ENCODE) {
        output = s3_encode(ctx0, &model, params.n_tokens);
    } else if (params.stage == clip_chatterbox_stage::DENOISE || params.stage == clip_chatterbox_stage::DENOISE_UNCOND) {
        output = s3_denoise(ctx0, &model, params.n_tokens * 2, params.time_t, params.time_r, params.stage == clip_chatterbox_stage::DENOISE_UNCOND, params.reference);
    } else if (params.stage == clip_chatterbox_stage::EULER_STEP) {
        auto input = [&](const char * name) {
            auto * x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, params.n_tokens * 2, 80);
            ggml_set_name(x, name); ggml_set_input(x); return x;
        };
        auto * x = input("a.gen.wav.flow.noise_z");
        auto * velocity = input("inp_velocity");
        if (!hparams.chatterbox_meanflow) {
            auto * negative = input("inp_unconditional");
            velocity = ggml_sub(ctx0, ggml_scale(ctx0, velocity, 1.7f), ggml_scale(ctx0, negative, 0.7f));
        }
        output = ggml_add(ctx0, x, ggml_scale(ctx0, velocity, params.time_r - params.time_t));
    } else if (params.stage == clip_chatterbox_stage::VOICE_ENCODER) {
        output = s3_voice_encoder(ctx0, &model, params.n_tokens, params.layer);
    } else if (params.stage == clip_chatterbox_stage::TOKENIZE) {
        output = s3_reference_tokens(ctx0, &model, params.n_tokens);
    } else if (params.stage == clip_chatterbox_stage::CAMPPLUS) {
        output = s3_campplus(ctx0, &model, params.n_tokens);
    } else if (params.stage == clip_chatterbox_stage::PROMPT) {
        output = s3_t3_prompt(ctx0, &model, params.n_tokens, params.reference);
    } else if (params.stage == clip_chatterbox_stage::SPEECH_EMBD) {
        output = s3_t3_next(ctx0, &model, params.code, params.position);
    } else if (params.stage == clip_chatterbox_stage::VOCODE) {
        output = s3_vocode(ctx0, &model, params.n_tokens * 2, params.reference);
    }
    if (!output) { throw std::runtime_error("failed to build Chatterbox graph"); }
    ggml_set_name(output, params.stage == clip_chatterbox_stage::VOCODE ? "out_audio" : params.stage == clip_chatterbox_stage::TOKENIZE ? "out_codes" : "out_feats");
    ggml_build_forward_expand(gf, output);
    return gf;
}
