// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

static constexpr ck_tile::index_t kBlockM = 64;
static constexpr ck_tile::index_t kBlockN = 64;

template <typename DataType>
struct UnifiedAttentionD256AttentionProblemT
{
    using QDataType = DataType;
    using KDataType = DataType;
    using VDataType = DataType;
    using AccDataType = float;

    static constexpr ck_tile::index_t kBlockM =
        ::kBlockM;
    static constexpr ck_tile::index_t kBlockN =
        ::kBlockN;
    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kQkDTile = 64;
};

using UnifiedAttentionD256AttentionProblem = UnifiedAttentionD256AttentionProblemT<ck_tile::bf16_t>;
using UnifiedAttentionD256AttentionProblemFp16 = UnifiedAttentionD256AttentionProblemT<ck_tile::half_t>;

template <int M, typename DataType = ck_tile::bf16_t>
struct UnifiedAttentionD256AttentionProblemM
{
    using QDataType = DataType;
    using KDataType = DataType;
    using VDataType = DataType;
    using AccDataType = float;

    static constexpr ck_tile::index_t kBlockM = M;
    static constexpr ck_tile::index_t kBlockN = ::kBlockN;
    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kQkDTile = 64;
};

template <typename Problem, int D, typename PipelinePolicy>
struct UnifiedAttentionD256QkBlockGemmProblem
{
    using ADataType      = typename Problem::QDataType;
    using BDataType      = typename Problem::KDataType;
    using CDataType      = typename Problem::AccDataType;
    using BlockGemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Problem::kBlockM, Problem::kBlockN, D>,
        typename PipelinePolicy::template QkBlockWarps<D>,
        ck_tile::sequence<16, 64, 32>>;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;
};


template <typename Problem, int NKey, int D, typename PipelinePolicy>
struct UnifiedAttentionD256QkBlockGemmProblemN
{
    using ADataType      = typename Problem::QDataType;
    using BDataType      = typename Problem::KDataType;
    using CDataType      = typename Problem::AccDataType;
    using BlockGemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Problem::kBlockM, NKey, D>,
        typename PipelinePolicy::template QkBlockWarpsN<NKey>,
        ck_tile::sequence<16, NKey, 32>>;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;
};

template <typename Problem, int N, typename PipelinePolicy>
struct UnifiedAttentionD256PvBlockGemmProblem
{
    using ADataType      = typename Problem::QDataType;
    using BDataType      = typename Problem::VDataType;
    using CDataType      = typename Problem::AccDataType;
    using BlockGemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Problem::kBlockM, N, Problem::kBlockN>,
        typename PipelinePolicy::template PvBlockWarps<N>,
        ck_tile::sequence<16, 64, 32>>;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;
};

template <typename Problem, int N, int K, typename PipelinePolicy>
struct UnifiedAttentionD256PvBlockGemmProblemK
{
    using ADataType      = typename Problem::QDataType;
    using BDataType      = typename Problem::VDataType;
    using CDataType      = typename Problem::AccDataType;
    using BlockGemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Problem::kBlockM, N, K>,
        typename PipelinePolicy::template PvBlockWarps<N>,
        ck_tile::sequence<16, 64, 32>>;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;
};
