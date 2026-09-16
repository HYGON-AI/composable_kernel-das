// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_kkt_solve_problem.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"

template <typename DataType>
struct GdnMmacImpl;

template <>
struct GdnMmacImpl<ck_tile::bf16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <>
struct GdnMmacImpl<ck_tile::half_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <typename DataType>
struct GdnMmacTransCImpl;

template <>
struct GdnMmacTransCImpl<ck_tile::bf16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplBF16BF16F32M16N16K16TransC;
};

template <>
struct GdnMmacTransCImpl<ck_tile::half_t>
{
    // The hand-written KKT register shuffles use the operand-swap C layout.
    // gfx938 TransC uses LTS and has a different accumulator distribution.
    using Type = ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16TransCLegacy;
};

template <typename DataType>
using GdnWarpGemm16x16x16 = ck_tile::WarpGemmImpl<
    ck_tile::WarpGemmAttributeMmacIterateK<
        typename GdnMmacImpl<DataType>::Type,
        1, 1, 1, 1, 1>>;

template <typename DataType>
using GdnWarpGemm16x64x32 = ck_tile::WarpGemmMmacDispatcher<
    DataType, DataType, float,
    16, 64, 32, false, 1, 4, 1, 1>;

template <typename Problem_ = GdnKktSolveProblemBf16>
struct GdnKktSolveDefaultPolicy
{
    using Problem  = Problem_;
    using DataType = typename Problem::QKDataType;

    static constexpr int kBlockSize = Problem::kBlockSize;
    static constexpr int kChunkSize = Problem::kChunkSize;
    static constexpr int kSubChunk  = Problem::kSubChunk;
    static constexpr int kHeadDim   = Problem::kHeadDim;
    static constexpr int kLaunchMinBlocks = 1;

    using KktWarpGemm = GdnWarpGemm16x64x32<DataType>;
    using SmallWarpGemm = GdnWarpGemm16x16x16<DataType>;
};
