// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_cumsum_problem.hpp"

template <typename Problem>
struct GdnCumsumDefaultPolicy
{
    using DataType = typename Problem::DataType;
    static constexpr int kChunkSize = Problem::kChunkSize;
    static constexpr int kSBlock = Problem::kSBlock;
    static constexpr int kBlockSize = Problem::kBlockSize;
    static constexpr int kLaunchMinBlocks = 1;

    CK_TILE_DEVICE static constexpr auto MakeTileDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<1, kSBlock, 1, 1>,
                               ck_tile::sequence<1, 1, kChunkSize, 1>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<0, 3, 0, 3>>{});
    }
};
