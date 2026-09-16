// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_fused_recurrent_problem.hpp"

template <typename Problem_>
struct GdnFusedRecurrentDefaultPolicy
{
    using Problem = Problem_;

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
