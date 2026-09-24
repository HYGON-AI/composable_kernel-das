// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_recompute_w_u_problem.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"

template <typename DataType>
struct GdnRecomputeWUMmacImpl;

template <>
struct GdnRecomputeWUMmacImpl<ck_tile::bf16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <>
struct GdnRecomputeWUMmacImpl<ck_tile::half_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <typename DataType>
using GdnRecomputeWUWarpGemm16x64x32 = ck_tile::WarpGemmMmacDispatcher<
    DataType, DataType, float,
    16, 64, 32, false, 1, 4, 1, 1>;

template <typename DataType>
struct GdnRecomputeWUWarpGemmSelector
{
    using Type = GdnRecomputeWUWarpGemm16x64x32<DataType>;
};

template <>
struct GdnRecomputeWUWarpGemmSelector<ck_tile::bf16_t>
{
    using Type = ck_tile::WarpGemmImpl<
        ck_tile::WarpGemmAttributeMmacIterateK<
            ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16,
            1, 4, 1, 1, 1>>;
};

template <typename Problem_ = GdnRecomputeWUProblemBf16>
struct GdnRecomputeWUDefaultPolicy
{
    using Problem  = Problem_;
    using DataType = typename Problem::QKDataType;

    static constexpr int kBlockSize = Problem::kBlockSize;
    static constexpr int kChunkSize = Problem::kChunkSize;
    static constexpr int kSubChunk  = Problem::kSubChunk;
    static constexpr int kHeadDim   = Problem::kHeadDim;
    static constexpr int kLaunchMinBlocks = 1;

    using WarpGemm = typename GdnRecomputeWUWarpGemmSelector<DataType>::Type;
};
