// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_l2norm_problem.hpp"

template <typename Problem_>
struct GdnL2NormFwdDefaultPolicy
{
    using Problem = Problem_;

    static constexpr ck_tile::index_t kWaveSize = 64;
    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kNumWaves = kBlockSize / kWaveSize;
    static constexpr ck_tile::index_t kRowsPerBlock = kNumWaves;
    static constexpr ck_tile::index_t kSmallDLimit = 512;
    static constexpr ck_tile::index_t kSmallValuesPerLane =
        kSmallDLimit / kWaveSize;
};
