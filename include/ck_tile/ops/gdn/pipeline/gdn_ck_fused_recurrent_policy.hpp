// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_fused_recurrent_problem.hpp"

template <typename Problem_>
struct GdnFusedRecurrentDefaultPolicy
{
    using Problem = Problem_;
    static constexpr ck_tile::index_t kStateVector = 1;
    static constexpr ck_tile::index_t kQKVector = 1;

    static constexpr ck_tile::index_t kWaveSize = Problem::kWaveSize;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;
    static constexpr ck_tile::index_t kValueTile = Problem::kValueTile;
    static constexpr ck_tile::index_t kKeyTile = Problem::kKeyTile;
    static constexpr ck_tile::index_t kKeyValuesPerLane =
        Problem::kKeyValuesPerLane;
    static constexpr ck_tile::index_t kLaunchMinBlocks = 1;

    // Logical [V, K] tile. V and K/64 are per-thread repeats; K%64 is
    // mapped to the lane id. Each lane owns V * (K/64) FP32 state values.
    CK_TILE_DEVICE static constexpr auto MakeStateDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<
                    ck_tile::sequence<kValueTile, 1, 1, 1>,
                    ck_tile::sequence<kKeyValuesPerLane, 1, kWaveSize, 1>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>,
                               ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>,
                               ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<0, 3, 0, 3>>{});
    }

    // Logical [1, K] tile with the same K distribution as the state tile.
    CK_TILE_DEVICE static constexpr auto MakeQKDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<
                    ck_tile::sequence<1, 1, 1, 1>,
                    ck_tile::sequence<kKeyValuesPerLane, 1, kWaveSize, 1>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>,
                               ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>,
                               ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<0, 3, 0, 3>>{});
    }

    CK_TILE_DEVICE static constexpr auto MakeValueDistribution()
    {
        constexpr auto state_distribution = MakeStateDistribution();
        return ck_tile::make_static_tile_distribution(
            ck_tile::detail::make_reduce_tile_distribution_encoding(
                state_distribution
                    .get_static_tile_distribution_encoding(),
                ck_tile::sequence<1>{}));
    }
};

// Four adjacent FP32 keys per lane, 16 lanes per value row. Four waves
// process independent value rows; reductions stay inside a 16-lane group.
// The host dispatch requires [V,K] state and K=V=128.
template <typename Problem_, int ValueRepeats = 1, int NumWarps = 4, bool Normalize = false>
struct GdnFusedRecurrentVectorPolicy
{
    using Problem = Problem_;
    static constexpr ck_tile::index_t kWaveSize = 64;
    static constexpr ck_tile::index_t kBlockSize = NumWarps * 64;
    static constexpr ck_tile::index_t kValueTile = ValueRepeats * NumWarps * 4;
    static constexpr ck_tile::index_t kKeyTile = 128;
    static constexpr ck_tile::index_t kLaunchMinBlocks = 2;
    static constexpr bool kNormalize = Normalize;
    static constexpr ck_tile::index_t kStateVector = 4;
    static constexpr ck_tile::index_t kQKVector = 4;

    CK_TILE_DEVICE static constexpr auto MakeStateDistribution()
    {
        using namespace ck_tile;
        return make_static_tile_distribution(tile_distribution_encoding<
            sequence<>, tuple<sequence<ValueRepeats,NumWarps,4>,sequence<2,16,4>>,
            tuple<sequence<1>,sequence<1,2>>,
            tuple<sequence<1>,sequence<2,1>>,
            sequence<1,2,2>,sequence<0,0,2>>{});
    }
    CK_TILE_DEVICE static constexpr auto MakeQKDistribution()
    {
        using namespace ck_tile;
        return make_static_tile_distribution(tile_distribution_encoding<
            sequence<NumWarps,4>, tuple<sequence<1>,sequence<2,16,4>>,
            tuple<sequence<0>,sequence<0,2>>,
            tuple<sequence<0>,sequence<1,1>>,
            sequence<1,2,2>,sequence<0,0,2>>{});
    }
    CK_TILE_DEVICE static constexpr auto MakeValueDistribution()
    {
        using namespace ck_tile;
        constexpr auto d=MakeStateDistribution();
        return make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
            d.get_static_tile_distribution_encoding(),sequence<1>{}));
    }
};
