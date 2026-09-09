// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

struct GemmABQuantPipelineAgBgCrDefaultPolicy
    : public UniversalGemmBasePolicy<GemmABQuantPipelineAgBgCrDefaultPolicy>
{
    using Base = UniversalGemmBasePolicy<GemmABQuantPipelineAgBgCrDefaultPolicy>;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAQ()
    {
        return GemmAQuantPipelineAgBgCrDefaultPolicy::GetVectorSizeAQ<Problem>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAQDramTileDistribution()
    {
        return GemmAQuantPipelineAgBgCrDefaultPolicy::MakeAQDramTileDistribution<Problem>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::GetVectorSizeBQ<Problem>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::MakeBQDramTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;
        using ADataType  = remove_cvref_t<typename Problem::ADataType>;
        using BDataType  = remove_cvref_t<typename Problem::BDataType>;
        using AComputeDataType =
            std::conditional_t<std::is_same_v<ADataType, fp8_t> ||
                                   std::is_same_v<ADataType, bf8_t>,
                               ADataType,
                               typename Problem::ComputeDataType>;
        using BComputeDataType =
            std::conditional_t<std::is_same_v<BDataType, fp8_t> ||
                                   std::is_same_v<BDataType, bf8_t>,
                               BDataType,
                               typename Problem::ComputeDataType>;

        static_assert(Problem::BQuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize::kK!");

#if CK_TILE_HCU_QUANT_GEMM_TARGET
        using WarpGemm =
            WarpGemmMmacDispatcher<AComputeDataType,
                                   BComputeDataType,
                                   typename Problem::CDataType,
                                   WarpTile::at(I0),
                                   WarpTile::at(I1),
                                   WarpTile::at(I2),
                                   Problem::TransposeC,
                                   Problem::kMRepeat,
                                   Problem::kNRepeat,
                                   Problem::kMInterleave,
                                   Problem::kNInterleave>;
#else
        using WarpGemm = WarpGemmDispatcher<AComputeDataType,
                                            BComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC>;
#endif
        static_assert(std::is_same_v<AComputeDataType, fp8_t> ||
                      std::is_same_v<AComputeDataType, bf8_t>);
        static_assert(std::is_same_v<BComputeDataType, fp8_t> ||
                      std::is_same_v<BComputeDataType, bf8_t>);
        static_assert(std::is_same_v<typename Problem::CDataType, float>);

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return ABQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
