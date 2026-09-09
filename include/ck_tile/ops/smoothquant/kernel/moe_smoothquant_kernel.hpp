// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

struct MoeSmoothquantHostArgs
{
    const void* p_x;
    const void* p_smscale;
    const void* p_topk_ids;
    void* p_yscale;
    void* p_qy;
    index_t tokens;
    index_t hidden_size;
    index_t experts;
    index_t topk;
    index_t x_stride;
    index_t y_stride;
};

template <typename Pipeline_>
struct MoeSmoothquant
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = typename Pipeline::Problem;

    using XDataType           = remove_cvref_t<typename Problem::XDataType>;
    using SmoothScaleDataType = remove_cvref_t<typename Problem::XScaleDataType>;
    using YScaleDataType      = remove_cvref_t<typename Problem::YScaleDataType>;
    using QYDataType          = remove_cvref_t<typename Problem::QYDataType>;

    static constexpr index_t Block_M   = Problem::BlockShape::Block_M;
    static constexpr index_t Block_N   = Problem::BlockShape::Block_N;
    static constexpr index_t Vector_N  = Problem::BlockShape::Vector_N;
    static constexpr index_t BlockSize = Problem::BlockShape::BlockSize;
    static constexpr bool kPadM        = false;
    static constexpr bool kPadN        = Problem::kPadN;

    struct Kargs
    {
        const void* p_x;
        const void* p_smscale;
        const void* p_topk_ids;
        void* p_yscale;
        void* p_qy;
        index_t tokens;
        index_t hidden_size;
        index_t experts;
        index_t topk;
        index_t x_stride;
        index_t y_stride;
    };

    using Hargs = MoeSmoothquantHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        return Kargs{hargs.p_x,
                     hargs.p_smscale,
                     hargs.p_topk_ids,
                     hargs.p_yscale,
                     hargs.p_qy,
                     hargs.tokens,
                     hargs.hidden_size,
                     hargs.experts,
                     hargs.topk,
                     hargs.x_stride,
                     hargs.y_stride};
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        return dim3(hargs.topk, integer_divide_ceil(hargs.tokens, Block_M), 1);
    }

    CK_TILE_HOST static constexpr auto BlockSizeValue() { return BlockSize; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t i_topk  = blockIdx.x;
        const index_t i_token = blockIdx.y * Block_M;
        const index_t i_token_in_thread =
            amd_wave_read_first_lane(threadIdx.x / Problem::BlockShape::ThreadPerBlock_N);
        const index_t token = i_token + i_token_in_thread;
        const index_t i_expert =
            static_cast<const index_t*>(kargs.p_topk_ids)[token * kargs.topk + i_topk];

        const auto x_window = [&]() {
            const auto view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.tokens, kargs.hidden_size),
                make_tuple(kargs.x_stride, 1),
                number<Vector_N>{},
                number<1>{});
            const auto padded = pad_tensor_view(
                view, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                padded, make_tuple(number<Block_M>{}, number<Block_N>{}), {i_token, 0});
        }();

        const auto smscale_window = [&]() {
            const auto view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const SmoothScaleDataType*>(kargs.p_smscale) +
                    i_expert * kargs.hidden_size,
                make_tuple(kargs.hidden_size),
                make_tuple(1),
                number<Vector_N>{},
                number<1>{});
            const auto padded =
                pad_tensor_view(view, make_tuple(number<Block_N>{}), sequence<kPadN>{});
            return make_tile_window(padded, make_tuple(number<Block_N>{}), {0});
        }();

        auto yscale_window = [&]() {
            const auto view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<YScaleDataType*>(kargs.p_yscale) + i_topk * kargs.tokens,
                make_tuple(kargs.tokens),
                make_tuple(1),
                number<1>{});
            const auto padded =
                pad_tensor_view(view, make_tuple(number<Block_M>{}), sequence<kPadM>{});
            return make_tile_window(padded, make_tuple(number<Block_M>{}), {i_token});
        }();

        auto qy_window = [&]() {
            const auto view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<QYDataType*>(kargs.p_qy) +
                    i_topk * kargs.tokens * kargs.y_stride,
                make_tuple(kargs.tokens, kargs.hidden_size),
                make_tuple(kargs.y_stride, 1),
                number<Vector_N>{},
                number<1>{});
            const auto padded = pad_tensor_view(
                view, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                padded, make_tuple(number<Block_M>{}, number<Block_N>{}), {i_token, 0});
        }();

        __shared__ char smem[GetSmemSize()];
        Pipeline{}(x_window, smscale_window, yscale_window, qy_window, kargs.hidden_size, smem);
    }
};

} // namespace ck_tile
