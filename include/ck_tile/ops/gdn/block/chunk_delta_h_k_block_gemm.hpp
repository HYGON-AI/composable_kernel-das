// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include <ck_tile/core.hpp>
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"

namespace ck_tile {

template <typename DataType, typename Mmac>
struct ChunkDeltaHKUpdateBlockGemm
{
    using WarpAttribute = WarpGemmAttributeMmacIterateK<Mmac, 1, 1, 1, 1, 1>;
    using WarpGemm = WarpGemmImpl<WarpAttribute>;
    using AVec = typename Mmac::AVecType;
    using CVec = typename Mmac::CVecType;

    static constexpr index_t kOutputK = 64;
    static constexpr index_t kTokens = 64;
    static constexpr index_t kTile = 16;
    static constexpr index_t kOutputIterations = kOutputK / kTile;
    static constexpr index_t kTokenIterations = kTokens / kTile;

    static_assert(kOutputIterations == 4);
    static_assert(kTokenIterations == 4);

    template <typename ALdsView>
    CK_TILE_DEVICE void operator()(CVec* state,
                                   const AVec* residual_t,
                                   const ALdsView& k_lds) const
    {
        static_for<0, kTokenIterations, 1>{}([&](auto t_iter) {
            typename WarpGemm::AWarpTensor a;
            a.get_thread_buffer().template set_as<AVec>(
                number<0>{}, residual_t[t_iter]);

            static_for<0, kOutputIterations, 1>{}([&](auto k_iter) {
                const auto b_window = make_tile_window(
                    k_lds,
                    make_tuple(number<kTile>{}, number<kTile>{}),
                    multi_index<2>{k_iter * kTile, t_iter * kTile},
                    make_static_tile_distribution(
                        typename WarpGemm::BWarpDstrEncoding{}));
                const auto b = load_tile(b_window);

                typename WarpGemm::CWarpTensor c;
                c.get_thread_buffer().template set_as<CVec>(
                    number<0>{}, state[k_iter]);
                WarpGemm{}(c, a, b);
                state[k_iter] =
                    c.get_thread_buffer().template get_as<CVec>()[number<0>{}];
            });
        });
    }
};

} // namespace ck_tile
