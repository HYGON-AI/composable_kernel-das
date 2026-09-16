// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <ck_tile/core.hpp>

namespace sla {

template <typename DataType_>
struct SlaSparseMapKargs
{
    const DataType_* q;
    const DataType_* k;
    DataType_* k_mean;
    DataType_* pooled_q;
    DataType_* pooled_k;
    ck_tile::index_t Lq;
    ck_tile::index_t Lk;
    ck_tile::index_t BLKQ;
    ck_tile::index_t BLKK;
    ck_tile::index_t Lq_blocks;
    ck_tile::index_t Lk_blocks;
};

template <typename DataType_>
struct SlaSparseMapProblem
{
    using DataType = DataType_;
    using Kargs = SlaSparseMapKargs<DataType>;

    static constexpr ck_tile::index_t kHeadDim = 128;
    static constexpr ck_tile::index_t kBlockSize = 128;
    static constexpr ck_tile::index_t kWaveSize = 64;
};

} // namespace sla
