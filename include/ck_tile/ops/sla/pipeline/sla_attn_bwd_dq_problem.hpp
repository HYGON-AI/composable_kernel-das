// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename OGradDataType_,
          typename DDataType_,
          typename LSEDataType_,
          typename QGradDataType_,
          index_t BlockM_,
          index_t BlockN_,
          index_t HeadDim_,
          index_t BlockSize_  = 256,
          index_t BlockPerCu_ = 1>
struct SlaAttnBwdDqProblem
{
    using QDataType     = remove_cvref_t<QDataType_>;
    using KDataType     = remove_cvref_t<KDataType_>;
    using VDataType     = remove_cvref_t<VDataType_>;
    using OGradDataType = remove_cvref_t<OGradDataType_>;
    using DDataType     = remove_cvref_t<DDataType_>;
    using LSEDataType   = remove_cvref_t<LSEDataType_>;
    using QGradDataType = remove_cvref_t<QGradDataType_>;

    static constexpr index_t kBlockM     = BlockM_;
    static constexpr index_t kBlockN     = BlockN_;
    static constexpr index_t kHeadDim    = HeadDim_;
    static constexpr index_t kBlockSize  = BlockSize_;
    static constexpr index_t kBlockPerCu = BlockPerCu_;
};

template <typename DataType>
using SlaAttnBwdDqAttentionProblem = SlaAttnBwdDqProblem<DataType,
                                                         DataType,
                                                         DataType,
                                                         DataType,
                                                         float,
                                                         float,
                                                         DataType,
                                                         64,
                                                         64,
                                                         128,
                                                         256,
                                                         2>;

} // namespace ck_tile
