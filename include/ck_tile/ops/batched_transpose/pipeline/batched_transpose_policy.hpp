// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_transpose/pipeline/batched_transpose_common_policy.hpp"

namespace ck_tile {

struct BatchedTransposePolicy : public BatchedTransposeCommonPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOutputDistribution()
    {
        constexpr index_t block_size  = Problem::kBlockSize;
        constexpr index_t m_per_block = Problem::kMPerBlock;
        constexpr index_t n_per_block = Problem::kNPerBlock;
        constexpr index_t vector_size = Problem::VectorSizeOutput;

        using Pattern = tile_distribution_encoding_pattern_2d<block_size,
                                                               m_per_block,
                                                               n_per_block,
                                                               vector_size,
                                                               TileAccessPattern>;
        return Pattern::make_shuffled_2d_static_tile_distribution();
    }
};

} // namespace ck_tile
