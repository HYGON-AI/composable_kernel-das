// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_fused_linear_attn_problem.hpp"
#include "ck_tile/ops/sla/block/sla_attn_fwd_block_gemm.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/mmac_gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"

namespace sla {

// Keep the MMAC configuration owned by linear attention.  The block-GEMM
// implementation is shared, but sparse-attention policy tuning must not alter
// the generated linear-attention kernels.
template <int M_, typename DataType_>
struct SlaLinearMmacProblem
{
    using QDataType = DataType_;
    using KDataType = DataType_;
    using VDataType = DataType_;
    using AccDataType = float;

    static constexpr ck_tile::index_t kBlockM = M_;
    static constexpr ck_tile::index_t kBlockN = 64;
    static constexpr ck_tile::index_t kBlockSize = 256;
};

template <typename DataType>
struct SlaLinearMmacImpl;

template <>
struct SlaLinearMmacImpl<ck_tile::bf16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <>
struct SlaLinearMmacImpl<ck_tile::fp16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <typename DataType>
using SlaLinearWarpGemm16x16x16 = ck_tile::WarpGemmImpl<
    ck_tile::WarpGemmAttributeMmacIterateK<
        typename SlaLinearMmacImpl<DataType>::Type,
        1, 1, 1, 1, 1>>;

template <typename Problem_>
struct SlaLinearMmacPolicy
{
    using Problem = Problem_;
    using DataType = typename Problem::QDataType;

    static constexpr bool kUseInterleavedV = true;

    template <int N>
    using PvBlockWarps = std::conditional_t<N == 128,
                                            ck_tile::sequence<1, 4, 1>,
                                            ck_tile::sequence<4, 1, 1>>;
    using PvWarpGemm = SlaLinearWarpGemm16x16x16<DataType>;
};

template <typename Problem_ = SlaFusedLinearAttnProblem<ck_tile::bf16_t>, int QRows_ = 64>
struct SlaLinearFusedPolicy
{
    using Problem = Problem_;
    using KMmacProblem = SlaLinearMmacProblem<64, typename Problem::DataType>;
    using QMmacProblem = SlaLinearMmacProblem<QRows_, typename Problem::DataType>;
    using KMmacPolicy = SlaLinearMmacPolicy<KMmacProblem>;
    using QMmacPolicy = SlaLinearMmacPolicy<QMmacProblem>;

    static constexpr int kBlockSize = Problem::kBlockSize;
    static constexpr int kHeadDim = Problem::kHeadDim;
    static constexpr int kKRows = 32;
    static constexpr int kQRows = QRows_;
    static constexpr int kKOutputTile = 64;
    static constexpr int kQOutputTile = 128;
    static constexpr int kKTile = 32;
    static constexpr int kKvSplitL = 8;
    static constexpr int kFeatureTilesPerBlock = 16;

    using KBlockGemm = Phase2PvARegBRegBlockGemmK<kKOutputTile, kKTile, KMmacPolicy>;
    using QBlockGemm = Phase2PvARegBRegBlockGemmK<kQOutputTile, kKTile, QMmacPolicy>;
};

struct SlaLinearUniversalBlockPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        using WarpGemm = SlaLinearWarpGemm16x16x16<typename Problem::ADataType>;
        return ck_tile::make_tuple(WarpGemm{}, 4, 1);
    }
};

struct SlaLinearUniversalPipelinePolicy
    : ck_tile::MmacGemmPipelineAGmemBGmemCRegV1DefaultPolicy
{
    using Base = ck_tile::MmacGemmPipelineAGmemBGmemCRegV1DefaultPolicy;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return ck_tile::MmacBlockGemmASmemBSmemCRegV1<Problem,
                                                       SlaLinearUniversalBlockPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using namespace ck_tile;
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(KPerBlock != 64 ||
                     !std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
        {
            return Base::template MakeADramTileDistribution<Problem>();
        }
        else
        {
            constexpr index_t BlockSize = Problem::kBlockSize;
            constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t M1 = Problem::VectorLoadSize / sizeof(ADataType);
            constexpr index_t M0 = MPerBlock / M1;
            constexpr index_t total_pixels = MPerBlock * KPerBlock / BlockSize;
            static_assert(total_pixels % M1 == 0);
            constexpr index_t K3 = total_pixels / M1;
            constexpr index_t KPack = Base::template GetSmemPackA<Problem>();
            static_assert(KPack % K3 == 0);
            constexpr index_t K2 = KPack / K3;

            if constexpr(get_warp_size() % (K2 * M0) == 0)
            {
                constexpr index_t K1 = get_warp_size() / (K2 * M0);
                constexpr index_t K0 = BlockSize / get_warp_size();
                static_assert(KPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1>,
                                                     sequence<K0, K1, K2, K3>>,
                                               tuple<sequence<2>, sequence<2, 1, 2>>,
                                               tuple<sequence<0>, sequence<1, 0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1 = (K2 * M0) / get_warp_size();
                constexpr index_t K2M = K2 / K1;
                constexpr index_t K0 = BlockSize / get_warp_size() / K1;
                static_assert(KPerBlock == K0 * K1 * K2M * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1>,
                                                     sequence<K0, K1, K2M, K3>>,
                                               tuple<sequence<2, 2>, sequence<1, 2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
        }
    }
};

template <typename DataType_, typename ALayout_, typename BLayout_, int KPerBlock_ = 32>
struct SlaLinearUniversalGemmPolicy
{
    using DataType = DataType_;
    using AccType = float;
    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BlockShape = ck_tile::TileGemmShape<ck_tile::sequence<64, 64, KPerBlock_>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 16>>;
    using Traits = ck_tile::TileGemmTraits<false, false, true, ALayout, BLayout, CLayout>;
    using Problem = ck_tile::GemmPipelineProblem<DataType,
                                                 DataType,
                                                 AccType,
                                                 BlockShape,
                                                 Traits>;
    using Pipeline = ck_tile::MmacGemmPipelineAGmemBGmemCRegV1<
        Problem, SlaLinearUniversalPipelinePolicy>;
    using Partitioner = ck_tile::GemmTilePartitioner<BlockShape>;
    using EpilogueProblem = ck_tile::Default2DEpilogueProblem<AccType,
                                                              DataType,
                                                              false,
                                                              false>;
    using Epilogue = ck_tile::Default2DEpilogue<EpilogueProblem>;
    using Kernel = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;
};

template <typename DataType>
using SlaLinearKvGemmPolicy = SlaLinearUniversalGemmPolicy<
    DataType,
    ck_tile::tensor_layout::gemm::ColumnMajor,
    ck_tile::tensor_layout::gemm::RowMajor,
    64>;

template <typename DataType>
using SlaLinearKvProjectionGemmPolicy = SlaLinearUniversalGemmPolicy<
    DataType,
    ck_tile::tensor_layout::gemm::RowMajor,
    ck_tile::tensor_layout::gemm::ColumnMajor>;

using SlaLinearDefaultPolicy = SlaLinearFusedPolicy<>;
using SlaLinearQ32Policy = SlaLinearFusedPolicy<SlaFusedLinearAttnProblem<ck_tile::bf16_t>, 32>;

} // namespace sla
