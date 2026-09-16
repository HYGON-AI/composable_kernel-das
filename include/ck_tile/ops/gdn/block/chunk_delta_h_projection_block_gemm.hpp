// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

// One wave computes a 16x64 transposed projection stage:
//   C[value, token] += S^T[value, key] * W^T[key, token]
// State BReg and W AReg already have the physical layouts of the transposed
// A/B operands.  The caller permutes the logical token coordinate in the W
// LDS descriptor so native CReg is also the K-update AReg layout.
template <typename DataType, typename Mmac>
struct ChunkDeltaHProjectionBlockGemm
{
    using WarpAttribute = WarpGemmAttributeMmacIterateK<Mmac, 1, 1, 1, 1, 1>;
    using AVec = typename Mmac::AVecType;
    using BVec = typename Mmac::BVecType;
    using CVec = typename Mmac::CVecType;

    static constexpr index_t kTokens = 64;
    static constexpr index_t kKeys = 64;
    static constexpr index_t kTile = 16;
    static constexpr index_t kTokenIterations = kTokens / kTile;
    static constexpr index_t kKeyIterations = kKeys / kTile;

    static_assert(kTokenIterations == 4);
    static_assert(kKeyIterations == 4);

    template <typename WLdsView>
    CK_TILE_DEVICE void operator()(CVec* projection,
                                   const BVec* state,
                                   const WLdsView& w_lds) const
    {
        static_for<0, kKeyIterations, 1>{}([&](auto k_iter) {
            const BVec s = state[k_iter];
            static_for<0, kTokenIterations, 1>{}([&](auto t_iter) {
                const auto a_window = make_tile_window(
                    w_lds,
                    make_tuple(number<kTile>{}, number<kTile>{}),
                    multi_index<2>{t_iter * kTile, k_iter * kTile},
                    make_static_tile_distribution(
                        typename WarpAttribute::AWarpDstrEncoding{}));
                const AVec w = load_tile(a_window)
                                   .get_thread_buffer()
                                   .template get_as<AVec>()[number<0>{}];
                Mmac{}(projection[t_iter], bit_cast<AVec>(s), bit_cast<BVec>(w));
            });
        });
    }
};

} // namespace ck_tile
