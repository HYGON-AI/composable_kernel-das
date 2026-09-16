// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once
// GDN split kkt_solve problem definition.
//
// Processes one (chunk=64, value_head):
//   k, g_cum, beta -> A = solve_tril(I + gated KK^T)

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

template <typename DataType_ = ck_tile::bf16_t>
struct GdnKktSolveProblem
{
    using QKDataType   = DataType_;
    using AccDataType  = float;
    using GateDataType = float;

    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kChunkSize = 64;
    static constexpr ck_tile::index_t kSubChunk  = 16;
    static constexpr ck_tile::index_t kHeadDim   = 128;
};

using GdnKktSolveProblemBf16 = GdnKktSolveProblem<ck_tile::bf16_t>;
using GdnKktSolveProblemF16  = GdnKktSolveProblem<ck_tile::half_t>;
