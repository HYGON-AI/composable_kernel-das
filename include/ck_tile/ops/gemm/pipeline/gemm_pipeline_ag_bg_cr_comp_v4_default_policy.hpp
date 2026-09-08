// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_new.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAGmemBGmemCregComputeV4, except the block gemm method, it shares
// the same vector size implementation, SmemSize, Global memory tile distiribution as the
// UniversalGemm Pipeline Policy.
// Default policy class should not be templated, put template on
// member functions instead.
struct GemmPipelineAgBgCrCompV4DefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV4DefaultPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        // using AccDataType     = float;
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;    // <2, 2, 1>
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;      // <32, 32, 16>

        using WarpGemm = WarpGemmMmacDispatcher<typename Problem::ADataType,  // half
                                                typename Problem::BDataType,  // half
                                                typename Problem::CDataType,  // AccDataType     float
                                                WarpTile::at(I0),    // 32
                                                WarpTile::at(I1),    // 32
                                                WarpTile::at(I2),    // 16
                                                Problem::TransposeC, // false
                                                WarpTile::at(I0) / 16, // Mrepeat
                                                WarpTile::at(I1) / 16, // Nrepeat
                                                1, // Minterleave
                                                1, // Ninterleave
                                                false, //SwizzleA
                                                Problem::UseABScale>;   //UseABScale
        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

// gfx936 grouped-GEMM V3 policy. It changes only the global A/B tile
// distribution feeding LDS; the block GEMM and LDS descriptors remain the
// proven Compute-V3 implementation. Keep the default policy untouched.
struct GemmPipelineAgBgCrCompV3WarpRakedPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV3WarpRakedPolicy>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        using WarpGemm = WarpGemmMmacDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                typename Problem::CDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                Problem::TransposeC,
                                                WarpTile::at(I0) / 16,
                                                WarpTile::at(I1) / 16,
                                                1,
                                                1,
                                                false,
                                                Problem::UseABScale>;
        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                typename Problem::CDataType,
                                                BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem,
                                       BlockGemmPolicy,
                                       Problem::TransposeC>{};
    }
};

// The in-thread transpose helper requires the input/output vector dimensions
// produced by the thread-raked access pattern. Keep this policy limited to the
// non-K-contiguous NN/TN producers; the proven RC path remains warp-raked.
struct GemmPipelineAgBgCrCompV3OperandTransposePolicy
    : public UniversalGemmBasePolicy<
          GemmPipelineAgBgCrCompV3OperandTransposePolicy>
{
};

// Exact-layout experiment for gfx936 grouped GEMM. Besides warp-raked global
// copies it asks BlockGemm to update the embedded 32x64 C vectors in place,
// avoiding a temporary wide CWarpTensor that otherwise spills heavily.
struct GemmPipelineAgBgCrCompV3TritonOperandPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV3TritonOperandPolicy>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr bool DirectCBufferMmac  = true;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeTritonDramTileDistribution()
    {
        static_assert(Problem::kBlockSize == 512 &&
                          Problem::BlockGemmShape::kM == 256 &&
                          Problem::BlockGemmShape::kN == 256 &&
                          Problem::BlockGemmShape::kK == 64,
                      "Triton producer mapping is specialized for 256x256x64/W8");
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<4, 8, 8>, sequence<8, 8>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ALayout = remove_cvref_t<typename Problem::ALayout>;
        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
        {
            // Load the physically contiguous M vectors with CK's validated
            // warp-raked distribution. The pipeline transposes this tile and
            // scatters it into the exact gfx936 LDS operand layout.
            return GemmPipelineAgBgCrCompV3OperandTransposePolicy::
                template MakeADramTileDistribution<Problem>();
        }
        else
        {
            return MakeTritonDramTileDistribution<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BLayout = remove_cvref_t<typename Problem::BLayout>;
        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            return GemmPipelineAgBgCrCompV3OperandTransposePolicy::
                template MakeBDramTileDistribution<Problem>();
        }
        else
        {
            return MakeTritonDramTileDistribution<Problem>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledARegTileDistribution()
    {
        return GemmPipelineAgBgCrCompV3OperandTransposePolicy::
            template MakeShuffledARegTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledBRegTileDistribution()
    {
        return GemmPipelineAgBgCrCompV3OperandTransposePolicy::
            template MakeShuffledBRegTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm =
            WarpGemmMmacDispatcher<typename Problem::ADataType,
                                   typename Problem::BDataType,
                                   typename Problem::CDataType,
                                   WarpTile::at(I0),
                                   WarpTile::at(I1),
                                   WarpTile::at(I2),
                                   Problem::TransposeC,
                                   WarpTile::at(I0) / 16,
                                   WarpTile::at(I1) / 16,
                                   1,
                                   1,
                                   false,
                                   Problem::UseABScale>;
        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                typename Problem::CDataType,
                                                BlockWarps,
                                                WarpGemm,
                                                true>;
        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
