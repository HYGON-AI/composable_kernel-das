// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_pre_process_problem.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"

template <typename DataType>
struct GdnPreProcessMmacImpl;

template <>
struct GdnPreProcessMmacImpl<ck_tile::bf16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <>
struct GdnPreProcessMmacImpl<ck_tile::half_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <typename DataType>
using GdnPreProcessWarpGemm16x64x32 = ck_tile::WarpGemmMmacDispatcher<
    DataType, DataType, float,
    16, 64, 32, false, 1, 4, 1, 1>;

template <typename Problem_ = GdnPreProcessProblemBf16>
struct GdnPreProcessDefaultPolicy
{
    using Problem  = Problem_;
    using DataType = typename Problem::DataType;

    static constexpr int kBlockSize = Problem::kBlockSize;
    static constexpr int kChunkSize = Problem::kChunkSize;
    static constexpr int kHeadDim   = Problem::kHeadDim;
    static constexpr int kLaunchMinBlocks = 1;

    using WarpGemm = GdnPreProcessWarpGemm16x64x32<DataType>;
};
