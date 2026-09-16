// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_attn_fwd_problem.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"
#include <type_traits>

template <typename DataType>
struct SlaAttnFwdMmacImpl;

template <>
struct SlaAttnFwdMmacImpl<ck_tile::bf16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <>
struct SlaAttnFwdMmacImpl<ck_tile::fp16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <typename DataType>
using SlaAttnFwdWarpGemm16x32x32 = ck_tile::WarpGemmImpl<
    ck_tile::WarpGemmAttributeMmacIterateK<
        typename SlaAttnFwdMmacImpl<DataType>::Type,
        1, 2, 1, 1, 2>>;
template <typename DataType>
using SlaAttnFwdWarpGemm16x16x32 = ck_tile::WarpGemmImpl<
    ck_tile::WarpGemmAttributeMmacIterateK<
        typename SlaAttnFwdMmacImpl<DataType>::Type,
        1, 1, 1, 1, 2>>;
template <typename DataType>
using SlaAttnFwdWarpGemm16x16x16 = ck_tile::WarpGemmImpl<
    ck_tile::WarpGemmAttributeMmacIterateK<
        typename SlaAttnFwdMmacImpl<DataType>::Type,
        1, 1, 1, 1, 1>>;

template <typename Problem_ = SlaAttnFwdAttentionProblem>
struct SlaAttnFwdDefaultPipelinePolicyT
{
    using Problem = Problem_;
    using DataType = typename Problem::QDataType;

    static constexpr int kBlockSize = Problem::kBlockSize;
    static constexpr int kLaunchMinBlocks = 1;
    static constexpr bool kUseActiveIndexList = true;
    static constexpr bool kAliasKvWhenPacked = true;
    static constexpr bool kUseRegisterQ = true;
    static constexpr bool kUseAsyncKLoad = true;
    static constexpr bool kUseSplitK32 = true;
    static constexpr bool kUseInterleavedV = true;

    template <int NKey>
    using QkBlockWarpsN = ck_tile::sequence<4, 1, 1>;
    template <int N>
    using PvBlockWarps = std::conditional_t<N == 128,
                                            ck_tile::sequence<1, 4, 1>,
                                            ck_tile::sequence<4, 1, 1>>;
    using PvWarpGemm = SlaAttnFwdWarpGemm16x16x16<DataType>;
    using QkWarpGemm32 = SlaAttnFwdWarpGemm16x32x32<DataType>;
    using PvWarpGemm32 = SlaAttnFwdWarpGemm16x32x32<DataType>;

    CK_TILE_HOST_DEVICE static constexpr bool CanAliasKv(int head_dim, int padded_dim)
    {
        return kAliasKvWhenPacked && head_dim == padded_dim;
    }

    CK_TILE_HOST_DEVICE static constexpr int GetOutputTile(int head_dim)
    {
        return head_dim > 64 ? 128 : 64;
    }

    template <int N_TILE>
    CK_TILE_HOST_DEVICE static constexpr int GetQkDChunks()
    {
        return N_TILE > 64 ? 2 : 1;
    }
};

using SlaAttnFwdDefaultPipelinePolicy = SlaAttnFwdDefaultPipelinePolicyT<SlaAttnFwdAttentionProblem>;
using SlaAttnFwdM128PipelinePolicy =
    SlaAttnFwdDefaultPipelinePolicyT<SlaAttnFwdAttentionProblemM<128>>;
