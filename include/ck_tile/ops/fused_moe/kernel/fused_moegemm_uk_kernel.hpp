// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/fused_moe/kernel/fused_moegemm_kernel.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Partitioner_, typename Pipeline_, typename Epilogue_>
struct FusedMoeGemmUKKernel
{
    using Partitioner = remove_cvref_t<Partitioner_>;
    using Pipeline    = remove_cvref_t<Pipeline_>;
    using Epilogue    = remove_cvref_t<Epilogue_>; // TODO: not used

    using BlockShape        = typename Pipeline::BlockShape;
    static constexpr index_t BlockSize_ = BlockShape::BlockSize;

    using ADataType            = typename Pipeline::Problem::ADataType;
    using GDataType            = typename Pipeline::Problem::GDataType;
    using DDataType            = typename Pipeline::Problem::DDataType;
    using AccDataType          = typename Pipeline::Problem::AccDataType;
    using ODataType            = typename Pipeline::Problem::ODataType;
    using AScaleDataType       = typename Pipeline::Problem::AScaleDataType;
    using GScaleDataType       = typename Pipeline::Problem::GScaleDataType;
    using DScaleDataType       = typename Pipeline::Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Pipeline::Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Pipeline::Problem::TopkWeightDataType;
    using IndexDataType        = typename Pipeline::Problem::IndexDataType;
    using YDataType            = typename Pipeline::Problem::YDataType;

    using Traits                = typename Pipeline::Problem::Traits;
    static constexpr bool UseUK = true;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<bf16_t> { static constexpr const char * name = "bf16"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
#define _SS_ std::string
#define _TS_ std::to_string
        using S_ = BlockShape;

        return _SS_("fused_moe_") + _SS_(t2s<ADataType>::name) + "_uk_" +
               _TS_(S_::Block_M0) + "x" + _TS_(S_::Block_N0) + "x" + _TS_(S_::Block_K0) + "x" +
               _TS_(S_::Block_N1) + "_" + _TS_(S_::WarpPerBlock_M0) + "x" +
               _TS_(S_::WarpPerBlock_N0) + "x" + _TS_(S_::WarpPerBlock_K0) + "_" + _TS_(S_::Warp_M0) +
               "x" + _TS_(S_::Warp_N0) + "x" + _TS_(S_::Warp_K0) + "_" + _SS_(Pipeline::name);
#undef _SS_
#undef _TS_
    }

    struct FusedMoeGemmKargs
    {
        const void* a_ptr;              // [m, k], input token
        const void* a_scale_ptr;        // [m, 1], token scale
        const void* g_ptr;              // [e, n, k]/[e, 2*n, k], pre-shuffled
        const void* d_ptr;              // [e, n, k], pre-shuffled
        const void* g_scale_ptr;        // [e, 1, n], gate(up) scale
        const void* d_scale_ptr;        // [e, 1, k], down scale
        const void* g_zp_ptr;           // [e, 2*n, k/group], gate(up) zero-point
        const void* d_zp_ptr;           // [e, k, n/group], down zero-point
        const void* y_smooth_scale_ptr; // [e, 1, n], smooth-quant-scale
        void* o_ptr;                    // [m, k], output token
        void* deterministic_tmp_ptr;    // reserved
        void* deterministic_counter_ptr;

        const void* sorted_token_ids_ptr;
        const void* sorted_weight_ptr;
        const void* sorted_expert_ids_ptr;
        const void* num_sorted_tiles_ptr;

        index_t hidden_size;       // k
        index_t intermediate_size; // n / TP, for Gate/Up/Down
        index_t num_tokens;        // input number of tokens
        index_t num_experts;       // number of groups
        index_t topk;

        index_t stride_token; // for input/output, stride per row
        index_t block_shape_n;
        index_t block_shape_k;
    };

    using Kargs = FusedMoeGemmKargs;
    using Hargs = FusedMoeGemmHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        return bit_cast<Kargs>(hargs);
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        constexpr index_t block_m = BlockShape::Block_M0;
        int max_num_tokens_padded =
            hargs.topk * hargs.num_tokens + hargs.num_experts * block_m - hargs.topk;
        return Partitioner::GridSize(max_num_tokens_padded, hargs.intermediate_size);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(BlockSize_); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // GetSmemSize() is an element count, not bytes — allocating a char
        // array of that size truncates the LDS buffer to half and pushes the
        // warps-2..3 N-slices of the gemm1 D tile out of bounds.
        __shared__ CK_TILE_LDS_ADDR char smem[Pipeline::GetSmemSizeBytes()];

        IndexDataType num_sorted_tiles = __builtin_amdgcn_readfirstlane(
            *reinterpret_cast<const IndexDataType*>(kargs.num_sorted_tiles_ptr));

        num_sorted_tiles = num_sorted_tiles / BlockShape::Block_M0;

        const auto [sorted_tile_id, intermediate_tile_id] =
            Partitioner{}(num_sorted_tiles, kargs.intermediate_size);
        if(sorted_tile_id >= num_sorted_tiles)
            return;

        Pipeline{}(kargs, smem, sorted_tile_id, intermediate_tile_id);
    }
};

} // namespace ck_tile
