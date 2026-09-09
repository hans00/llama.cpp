// Copyright (c) 2026 codec.cpp contributors
// SPDX-License-Identifier: MIT
#include "models.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {
struct snac_graph {
    ggml_context *     ctx;
    const clip_model & model;

    ggml_tensor * weight(const std::string & name) {
        const auto it = model.snac_weights.find("a.gen.wav." + name);
        if (it == model.snac_weights.end()) {
            throw std::runtime_error("missing SNAC tensor: " + name);
        }
        return it->second;
    }

    ggml_tensor * snake(ggml_tensor * x, const std::string & name) {
        auto * a    = weight(name + ".alpha");
        auto * wave = ggml_sin(ctx, ggml_mul(ctx, x, a));
        return ggml_add(ctx, x, ggml_div(ctx, ggml_sqr(ctx, wave), ggml_scale_bias(ctx, a, 1, 1e-9f)));
    }

    ggml_tensor * conv(ggml_tensor *       x,
                       const std::string & name,
                       int                 stride    = 1,
                       int                 dilation  = 1,
                       int                 padding   = 0,
                       bool                depthwise = false,
                       bool                bias      = true) {
        auto *        w = weight(name + ".weight");
        ggml_tensor * y;
        if (name.compare(0, 4, "enc.") == 0 && padding > 0) {
            // SNAC's encoder needs F32 im2col: F16 errors accumulate through Snake.
            const int kernel = stride > 1 ? 2 * stride : 7;
            auto * shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel, depthwise ? 1 : x->ne[1],
                                              depthwise ? x->ne[1] : w->ne[1]);
            auto * input = depthwise ? ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], 1) : x;
            auto * columns   = ggml_im2col(ctx, shape, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
            if (depthwise) {
                y = ggml_mul_mat(ctx, columns, w);
                y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[2]);
            } else {
                y = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, w, columns)));
            }
        } else if (w->ne[0] == x->ne[1] && w->ne[2] == 1 && padding == 0) {
            y = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, w, ggml_cont(ctx, ggml_transpose(ctx, x)))));
        } else if (depthwise) {
            y = ggml_conv_1d_dw(ctx, w, x, stride, padding, dilation);
        } else {
            y = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
        }
        y = bias ? ggml_add(ctx, y, weight(name + ".bias")) : y;
        ggml_set_name(y, ("snac_" + name).c_str());
        return y;
    }

    ggml_tensor * residual(ggml_tensor * x, const std::string & name, int dilation) {
        auto * y = snake(x, name + ".0");
        y        = conv(y, name + ".1", 1, dilation, 3 * dilation, true);
        y        = conv(snake(y, name + ".2"), name + ".3");
        return ggml_add(ctx, x, y);
    }

    ggml_tensor * expand(ggml_tensor * x, int factor) {
        if (factor == 1) {
            return x;
        }
        const int64_t T = x->ne[0], C = x->ne[1];
        x            = ggml_reshape_3d(ctx, ggml_cont(ctx, x), 1, T, C);
        auto * shape = ggml_new_tensor_3d(ctx, x->type, factor, T, C);
        return ggml_reshape_2d(ctx, ggml_repeat(ctx, x, shape), factor * T, C);
    }

    ggml_tensor * reconstruct(ggml_tensor * codes, int level) {
        const std::string base = "q." + std::to_string(level);
        auto *            x    = ggml_get_rows(ctx, weight(base + ".codebook.weight"), codes);
        x                      = ggml_cont(ctx, ggml_transpose(ctx, x));
        return expand(conv(x, base + ".out_proj"), 4 >> level);
    }

    ggml_tensor * encode(int n_frames) {
        auto * pcm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames * 2048, 1);
        ggml_set_name(pcm, "inp_pcm");
        ggml_set_input(pcm);
        auto *    x       = conv(pcm, "enc.0", 1, 1, 3);
        const int rates[] = { 2, 4, 8, 8 }, dilations[] = { 1, 3, 9 };
        for (int i = 0; i < 4; ++i) {
            std::string name = "enc." + std::to_string(i + 1);
            for (int j = 0; j < 3; ++j) {
                x = residual(x, name + "." + std::to_string(j), dilations[j]);
            }
            x = conv(snake(x, name + ".3"), name + ".4", rates[i], 1, rates[i] / 2);
        }
        x = conv(x, "enc.5", 1, 1, 3, true);
        ggml_tensor * codes[3];
        for (int i = 0; i < 3; ++i) {
            auto * pooled = i < 2 ? ggml_pool_1d(ctx, x, GGML_OP_POOL_AVG, 4 >> i, 4 >> i, 0) : x;
            auto * z      = conv(pooled, "q." + std::to_string(i) + ".in_proj");
            z             = ggml_cont(ctx, ggml_transpose(ctx, z));
            auto * norm   = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, z)));
            z             = ggml_div(ctx, z, ggml_clamp(ctx, norm, 1e-12f, INFINITY));
            auto * scores = ggml_mul_mat(ctx, weight("q." + std::to_string(i) + ".codebook_norm.weight"), z);
            codes[i]      = ggml_argmax(ctx, scores);
            if (i < 2) {
                x = ggml_sub(ctx, x, reconstruct(codes[i], i));
            }
        }
        auto row = [&](int level, int index) {
            return ggml_cont(ctx, ggml_view_2d(ctx, codes[level], 1, n_frames, (1 << level) * sizeof(int32_t),
                                               index * sizeof(int32_t)));
        };
        auto * packed = row(0, 0);
        for (const auto & slot : {
                 std::pair<int, int>{ 1, 0 },
                  { 2, 0 },
                  { 2, 1 },
                  { 1, 1 },
                  { 2, 2 },
                  { 2, 3 }
        }) {
            packed = ggml_concat(ctx, packed, row(slot.first, slot.second), 0);
        }
        return packed;
    }

    ggml_tensor * decode(int n_frames) {
        ggml_tensor * x = nullptr;
        for (int i = 0; i < 3; ++i) {
            auto * codes = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_frames << i);
            ggml_set_name(codes, ("inp_codes_" + std::to_string(i)).c_str());
            ggml_set_input(codes);
            auto * z = reconstruct(codes, i);
            x        = x ? ggml_add(ctx, x, z) : z;
        }
        x                 = conv(x, "dec.0", 1, 1, 3, true);
        x                 = conv(x, "dec.1");
        const int rates[] = { 8, 8, 4, 2 }, dilations[] = { 1, 3, 9 };
        for (int i = 0; i < 4; ++i) {
            const std::string name = "dec." + std::to_string(i + 2);
            x                      = snake(x, name + ".0");
            x                      = ggml_conv_transpose_1d(ctx, weight(name + ".1.weight"), x, rates[i], 0, 1);
            x = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0] - rates[i], x->ne[1], x->nb[1], rates[i] / 2 * x->nb[0]));
            x = ggml_add(ctx, x, weight(name + ".1.bias"));
            auto * noise = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], 1);
            ggml_set_name(noise, ("inp_noise_" + std::to_string(i)).c_str());
            ggml_set_input(noise);
            x = ggml_add(ctx, x, ggml_mul(ctx, conv(x, name + ".2.linear", 1, 1, 0, false, false), noise));
            for (int j = 0; j < 3; ++j) {
                x = residual(x, name + "." + std::to_string(j + 3), dilations[j]);
            }
        }
        return ggml_tanh(ctx, conv(snake(x, "dec.6"), "dec.7", 1, 1, 3));
    }
};
}  // namespace

ggml_cgraph * clip_graph_snac::build() {
    snac_graph graph{ ctx0, model };
    auto *     output = encode ? graph.encode(n_frames) : graph.decode(n_frames);
    ggml_set_name(output, encode ? "out_codes" : "out_audio");
    ggml_build_forward_expand(gf, output);
    return gf;
}
