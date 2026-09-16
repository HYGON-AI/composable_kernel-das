// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once
// GDN split recompute_w_u problem definition.
//
// Processes one (chunk=64, value_head):
//   A, k, v, g_cum, beta -> w, u

#include "ck_tile/core.hpp"

template <typename DataType_ = ck_tile::bf16_t>
struct GdnRecomputeWUProblem
{
    using QKDataType   = DataType_;
    using VDataType    = DataType_;
    using ADataType    = DataType_;
    using AccDataType  = float;
    using GateDataType = float;

    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kChunkSize = 64;
    static constexpr ck_tile::index_t kSubChunk  = 16;
    static constexpr ck_tile::index_t kHeadDim   = 128;
};

using GdnRecomputeWUProblemBf16 = GdnRecomputeWUProblem<ck_tile::bf16_t>;
using GdnRecomputeWUProblemFp16 = GdnRecomputeWUProblem<ck_tile::half_t>;
