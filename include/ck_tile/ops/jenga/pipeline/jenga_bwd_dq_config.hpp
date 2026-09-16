// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <string>

namespace ck_tile {
namespace jenga_dq {

struct jenga_bwd_dq_traits
{
    std::string data_type = "bf16";
    index_t block_m      = 64;
    index_t block_n      = 64;
    index_t head_dim     = 128;
    index_t max_nnz      = 84;
    int kv_stage_count   = 1;
};

} // namespace jenga_dq
} // namespace ck_tile

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename OGradDataType_,
          typename DDataType_,
          typename LSEDataType_,
          typename QGradDataType_,
          ck_tile::index_t BlockM_,
          ck_tile::index_t BlockN_,
          ck_tile::index_t HeadDim_,
          ck_tile::index_t MaxNnz_,
          ck_tile::index_t BlockSize_  = 256,
          ck_tile::index_t BlockPerCu_ = 1>
struct JengaBwdDqProblem
{
    using QDataType     = ck_tile::remove_cvref_t<QDataType_>;
    using KDataType     = ck_tile::remove_cvref_t<KDataType_>;
    using VDataType     = ck_tile::remove_cvref_t<VDataType_>;
    using OGradDataType = ck_tile::remove_cvref_t<OGradDataType_>;
    using DDataType     = ck_tile::remove_cvref_t<DDataType_>;
    using LSEDataType   = ck_tile::remove_cvref_t<LSEDataType_>;
    using QGradDataType = ck_tile::remove_cvref_t<QGradDataType_>;

    static constexpr ck_tile::index_t kBlockM     = BlockM_;
    static constexpr ck_tile::index_t kBlockN     = BlockN_;
    static constexpr ck_tile::index_t kHeadDim    = HeadDim_;
    static constexpr ck_tile::index_t kMaxNnz     = MaxNnz_;
    static constexpr ck_tile::index_t kBlockSize  = BlockSize_;
    static constexpr ck_tile::index_t kBlockPerCu = BlockPerCu_;
};

} // namespace ck_tile
