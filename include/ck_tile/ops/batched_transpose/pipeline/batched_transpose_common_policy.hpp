// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct BatchedTransposeCommonPolicy
{
    CK_TILE_DEVICE static constexpr auto TileAccessPattern =
        tile_distribution_pattern::thread_raked;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeInputDistribution()
    {
        constexpr index_t block_size   = Problem::kBlockSize;
        constexpr index_t leading_dim  = Problem::kNPerBlock;
        constexpr index_t secondary_dim = Problem::kMPerBlock;
        constexpr index_t vector_size   = Problem::VectorSizeInput;

        using Pattern = tile_distribution_encoding_pattern_2d<block_size,
                                                               secondary_dim,
                                                               leading_dim,
                                                               vector_size,
                                                               TileAccessPattern>;
        return Pattern::make_2d_static_tile_distribution();
    }
};

} // namespace ck_tile
