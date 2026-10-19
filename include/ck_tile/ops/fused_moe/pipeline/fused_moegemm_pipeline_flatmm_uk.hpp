// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_flatmm_uk_engine.hpp"

namespace ck_tile {

template <typename Problem_,
          typename Policy_       = FusedMoeGemmPipelineFlatmmPolicy,
          typename EnginePolicy_ = FusedMoeGemmPipelineFlatmmPolicy>
struct FusedMoeGemmPipeline_FlatmmUk
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockShape = typename Problem::BlockShape; 

    using ADataType            = typename Problem::ADataType;
    using GDataType            = typename Problem::GDataType;
    using DDataType            = typename Problem::DDataType;
    using AccDataType          = typename Problem::AccDataType;
    using ODataType            = typename Problem::ODataType;
    using AScaleDataType       = typename Problem::AScaleDataType;
    using GScaleDataType       = typename Problem::GScaleDataType;
    using DScaleDataType       = typename Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Problem::TopkWeightDataType;
    using IndexDataType        = typename Problem::IndexDataType;
    using YDataType            = typename Problem::YDataType;

    using Traits = typename Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    static_assert(!IsGateOnly, "flatmm_uk(HCU): gate+up only, gate-only not supported by engine");

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();

    static constexpr index_t kBlockPerCu = 2;

    static constexpr const char* name = "flatmm_uk";

    // compute engine: HCU MMAC implementation of the 2-gemm dataflow
    // (faithful copy of the fork's flatmm_ex pipeline)
    using Engine = FusedMoeGemmPipeline_FlatmmUkEngine<Problem, EnginePolicy_>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Engine::GetSmemSize();
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeBytes()
    {
        constexpr index_t kMaxElemSize =
            (sizeof(ADataType) > sizeof(DDataType))
                ? (sizeof(ADataType) > sizeof(YDataType) ? sizeof(ADataType) : sizeof(YDataType))
                : (sizeof(DDataType) > sizeof(YDataType) ? sizeof(DDataType) : sizeof(YDataType));
        return Engine::GetSmemSize() * kMaxElemSize;
    }

    CK_TILE_HOST_DEVICE static auto GetACoord()
    {
        constexpr auto a_dist = Policy::template MakeGlobalTileDistribution_A<Problem>();
        const auto a_coord    = a_dist.calculate_index();
        return a_coord;
    }

    CK_TILE_HOST_DEVICE static auto GetOCoord()
    {
        constexpr auto o_dist = Policy::template MakeGlobalTileDistribution_O<Problem>();
        const auto o_coord    = o_dist.calculate_index();
        return o_coord;
    }

    template <typename Kargs>
    CK_TILE_DEVICE auto operator()(const Kargs& kargs,
                                   CK_TILE_LDS_ADDR void* smem,
                                   index_t sorted_tile_id,
                                   index_t intermediate_tile_id)
    {
        static_assert(BlockShape::Block_K1 == BlockShape::Block_N0,
                      "flatmm_uk(HCU): gemm1 K tile must match gemm0 N tile");

        constexpr index_t hidden_radio_0 = IsGateOnly ? 1 : 2; // gate_up = 2

        index_t expert_stride_0 = kargs.intermediate_size * hidden_radio_0 * kargs.hidden_size;
        index_t expert_stride_1 = kargs.hidden_size * kargs.intermediate_size;

        const IndexDataType expert_id = __builtin_amdgcn_readfirstlane(
            reinterpret_cast<const IndexDataType*>(kargs.sorted_expert_ids_ptr)[sorted_tile_id]);

        index_t hidden_idx = __builtin_amdgcn_readfirstlane(
            intermediate_tile_id * BlockShape::Block_N0);

        const auto a_coord = GetACoord(); 
        const auto sorted_token_id =
            a_coord[number<0>{}] + sorted_tile_id * BlockShape::Block_M0;

        index_t token_id =
            reinterpret_cast<const index_t*>(kargs.sorted_token_ids_ptr)[sorted_token_id];
#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID || CK_TILE_REFERENCE_MOE_SORTING_MOCK_EXPERT_ID
        // same decode contract as FusedMoeGemmKernel (A-gather side): strip the
        // packed topk/expert tag before using the id as a token row
        token_id &= 0xffffff;
#endif
        
        const auto a_window = [&]() {
            const ADataType* a_ptr = reinterpret_cast<const ADataType*>(kargs.a_ptr);
            const auto a_view_     = make_naive_tensor_view<address_space_enum::global>(
                a_ptr,
                make_tuple(kargs.num_tokens, kargs.hidden_size),
                make_tuple(kargs.stride_token, 1),
                number<kAlignmentA>{},
                number<1>{});

            const auto a_gather_view_ = transform_tensor_view(
                a_view_,
                make_tuple(make_indexing_transform(kargs.num_tokens, token_id),
                           make_pass_through_transform(kargs.hidden_size)),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            const auto a_gather_view_1_ = pad_tensor_view(
                a_gather_view_,
                make_tuple(number<BlockShape::Block_M0>{}, number<2 * BlockShape::Block_K0>{}),
                sequence<false, PadHiddenSize>{});

            const auto a_window_ = make_tile_window(
                a_gather_view_1_,
                make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
                {0, 0});
            return a_window_;
        }();

        const auto g_window = [&]() {
            const GDataType* g_ptr = reinterpret_cast<const GDataType*>(kargs.g_ptr) +
                                     static_cast<long_index_t>(expert_id) * expert_stride_0 +
                                     hidden_idx * kargs.hidden_size;

            // real gate rows left in this tile; rows beyond intermediate_size
            // belong to the up half / next expert (or past W1 for the last
            // expert), so declare the true extent and let the pad transform
            // zero-fill the tail of a partial tile (gemm1 sees zero columns
            // through the matching guard in the engine's manual D load).
            const index_t g_rows_valid = kargs.intermediate_size - hidden_idx;

            const auto g_view_ = make_naive_tensor_view<address_space_enum::global>(
                g_ptr,
                make_tuple(g_rows_valid, kargs.hidden_size),
                make_tuple(kargs.hidden_size, 1),
                number<kAlignmentG>{},
                number<1>{});

            const auto g_view_1_ = pad_tensor_view(
                g_view_,
                make_tuple(number<BlockShape::Block_N0>{}, number<2 * BlockShape::Block_K0>{}),
                sequence<true, PadHiddenSize>{});

            const auto g_window_ = make_tile_window(
                g_view_1_,
                make_tuple(number<BlockShape::Block_N0>{}, number<BlockShape::Block_K0>{}),
                {0, 0});
            return g_window_;
        }();

        const auto d_window = [&]() {
            const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                     static_cast<long_index_t>(expert_id) * expert_stride_1 +
                                     hidden_idx;

            const auto d_view_ = make_naive_tensor_view<address_space_enum::global>(
                d_ptr,
                make_tuple(kargs.hidden_size, BlockShape::Block_K1),
                make_tuple(kargs.intermediate_size, 1),
                number<kAlignmentD>{},
                number<1>{});

            const auto d_view_1_ = pad_tensor_view(
                d_view_,
                make_tuple(number<2 * BlockShape::Block_N1>{}, number<BlockShape::Block_K1>{}),
                sequence<PadHiddenSize, PadIntermediateSize>{});

            const auto d_window_ = make_tile_window(
                d_view_1_,
                make_tuple(number<BlockShape::Block_N1>{}, number<BlockShape::Block_K1>{}),
                {0, 0});
            return d_window_;
        }();

        struct MoeEpilogue
        {
            ODataType* o_ptr; // [num_tokens, hidden] fp32 atomic accumulation buffer
            const IndexDataType* sorted_ids;
            const TopkWeightDataType* sorted_weights;
            index_t row_base; // sorted_tile_id * Block_M0
            index_t stride_token;
            index_t num_tokens;
            index_t hidden_size;
        };

        Engine{}(a_window,
                 g_window,
                 d_window,
                 MoeEpilogue{reinterpret_cast<ODataType*>(kargs.o_ptr),
                             reinterpret_cast<const IndexDataType*>(kargs.sorted_token_ids_ptr),
                             reinterpret_cast<const TopkWeightDataType*>(kargs.sorted_weight_ptr),
                             sorted_tile_id * BlockShape::Block_M0,
                             kargs.stride_token,
                             kargs.num_tokens,
                             kargs.hidden_size},
                 smem,
                 kargs.hidden_size,
                 kargs.intermediate_size,
                 hidden_idx);
    }
};

} // namespace ck_tile
