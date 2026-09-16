// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_sparse_map_problem.hpp"

namespace sla {

template <typename Problem_>
struct SlaSparseMapPolicy
{
    using Problem = Problem_;
    using DataType = typename Problem::DataType;

    static constexpr ck_tile::index_t kHeadDim = Problem::kHeadDim;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;
    static constexpr ck_tile::index_t kWaveSize = Problem::kWaveSize;
};

} // namespace sla
