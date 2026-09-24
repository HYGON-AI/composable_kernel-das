// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

#include <cstdint>

namespace ck_tile {
namespace example {
namespace sla {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename OGradDataType_,
          typename KGradDataType_,
          typename VGradDataType_,
          typename AccDataType_,
          typename LSEDataType_,
          index_t BlockM_,
          index_t BlockN_,
          index_t HeadDim_,
          index_t MaxNnz_,
          index_t ThreadsPerBlock_>
struct SlaAttnBwdDkdvProblem
{
    using QDataType     = remove_cvref_t<QDataType_>;
    using KDataType     = remove_cvref_t<KDataType_>;
    using VDataType     = remove_cvref_t<VDataType_>;
    using OGradDataType = remove_cvref_t<OGradDataType_>;
    using KGradDataType = remove_cvref_t<KGradDataType_>;
    using VGradDataType = remove_cvref_t<VGradDataType_>;
    using AccDataType   = remove_cvref_t<AccDataType_>;
    using LSEDataType   = remove_cvref_t<LSEDataType_>;

    static constexpr index_t BlockM          = BlockM_;
    static constexpr index_t BlockN          = BlockN_;
    static constexpr index_t HeadDim         = HeadDim_;
    static constexpr index_t MaxNnz          = MaxNnz_;
    static constexpr index_t ThreadsPerBlock = ThreadsPerBlock_;
};

template <typename DataType>
using SlaAttnBwdDkdvAttentionProblem =
    SlaAttnBwdDkdvProblem<DataType,
                          DataType,
                          DataType,
                          DataType,
                          DataType,
                          DataType,
                          float,
                          float,
                          64,
                          64,
                          128,
                          56,
                          256>;



} // namespace sla
} // namespace example
} // namespace ck_tile
