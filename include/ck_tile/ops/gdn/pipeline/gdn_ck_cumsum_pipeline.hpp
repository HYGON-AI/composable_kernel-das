// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_cumsum_policy.hpp"

namespace ck_tile {

template <typename Policy>
struct GdnCumsumPipeline
{
    static constexpr int kChunkSize = Policy::kChunkSize;
    static constexpr int kSBlock = Policy::kSBlock;
    using DataType = typename Policy::DataType;

    CK_TILE_DEVICE auto make_scalar_view(const DataType* __restrict__ g,
                                         int T,
                                         int H,
                                         int ih) const
    {
        return make_naive_tensor_view<address_space_enum::global>(
            g + ih,
            make_tuple(number<1>{}, T),
            make_tuple(number<0>{}, H),
            number<1>{},
            number<1>{});
    }

    CK_TILE_DEVICE auto make_scalar_view(float* __restrict__ o,
                                         int T,
                                         int H,
                                         int ih) const
    {
        return make_naive_tensor_view<address_space_enum::global>(
            o + ih,
            make_tuple(number<1>{}, T),
            make_tuple(number<0>{}, H),
            number<1>{},
            number<1>{});
    }

    CK_TILE_DEVICE auto make_vector_view(const DataType* __restrict__ g,
                                         int T,
                                         int H,
                                         int S,
                                         int ih,
                                         int is) const
    {
        return make_naive_tensor_view<address_space_enum::global>(
            g + ih * S,
            make_tuple(S, T),
            make_tuple(number<1>{}, H * S),
            number<1>{},
            number<1>{});
    }

    CK_TILE_DEVICE auto make_vector_view(float* __restrict__ o,
                                         int T,
                                         int H,
                                         int S,
                                         int ih,
                                         int is) const
    {
        return make_naive_tensor_view<address_space_enum::global>(
            o + ih * S,
            make_tuple(S, T),
            make_tuple(number<1>{}, H * S),
            number<1>{},
            number<1>{});
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ g,
                                   const float* __restrict__ A_log,
                                   const float* __restrict__ dt_bias,
                                   float* __restrict__ o,
                                   int tc,
                                   int ih,
                                   int is,
                                   int T,
                                   int H,
                                   int S,
                                   float scale,
                                   bool is_vector,
                                   bool use_gate,
                                   bool has_bias) const
    {
        auto in = [&]() {
            if constexpr(kSBlock > 1)
            {
                auto g_view = make_vector_view(g, T, H, S, ih, is);
                auto g_win = make_tile_window(
                    g_view,
                    make_tuple(number<kSBlock>{}, number<kChunkSize>{}),
                    multi_index<2>{is, tc},
                    Policy::MakeTileDistribution());
                return load_tile(g_win);
            }
            else
            {
                auto g_view = make_scalar_view(g, T, H, ih);
                auto g_win = make_tile_window(
                    g_view,
                    make_tuple(number<kSBlock>{}, number<kChunkSize>{}),
                    multi_index<2>{0, tc},
                    Policy::MakeTileDistribution());
                return load_tile(g_win);
            }
        }();
        auto out = make_static_distributed_tensor<float>(in.get_tile_distribution());

        const float neg_exp_A = use_gate ? -expf(A_log[ih]) : 1.0f;
        const float bias = (use_gate && has_bias) ? dt_bias[ih] : 0.0f;
        constexpr auto spans = decltype(in)::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(in.get_tile_distribution(), dstr_idx);
                const int col = tile_idx.at(number<1>{});
                float x = type_convert<float>(in[dstr_idx]);
                if(use_gate)
                {
                    x += bias;
                    x = neg_exp_A * (x < 20.0f ? log1pf(expf(x)) : x);
                }
                static_for<0, 6, 1>{}([&](auto i) {
                    constexpr uint32_t offset = 1u << i;
                    const float y = warp_shuffle_up(x, offset);
                    if(col >= static_cast<int>(offset))
                        x += y;
                });
                out(dstr_idx) = x * scale;
            });
        });

        if constexpr(kSBlock > 1)
        {
            auto o_view = make_vector_view(o, T, H, S, ih, is);
            auto o_win = make_tile_window(
                o_view,
                make_tuple(number<kSBlock>{}, number<kChunkSize>{}),
                multi_index<2>{is, tc},
                Policy::MakeTileDistribution());
            store_tile(o_win, out);
        }
        else
        {
            auto o_view = make_scalar_view(o, T, H, ih);
            auto o_win = make_tile_window(
                o_view,
                make_tuple(number<kSBlock>{}, number<kChunkSize>{}),
                multi_index<2>{0, tc},
                Policy::MakeTileDistribution());
            store_tile(o_win, out);
        }
    }
};

} // namespace ck_tile
