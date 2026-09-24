// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/mmac_block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"
#include "ck_tile/ops/jenga/block/jenga_bwd_dq_breg_gemm.hpp"
#include "ck_tile/ops/jenga/block/jenga_block_gemm_areg_bsmem_creg_v2r1.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dq_tile_shape.hpp"
#include <type_traits>
namespace ck_tile {
CK_TILE_DEVICE void JengaBwdBlockSyncLdsLight()
{
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}
CK_TILE_DEVICE float JengaBwdFastExp2(float x)
{
    return __builtin_amdgcn_exp2f(x);
}
template <typename DataType, index_t K_>
struct JengaWarpGemmSelector
{
    using Type = std::conditional_t<
        std::is_same_v<DataType, bf16_t>,
        WarpGemmImpl<WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, K_ / 16>>,
        WarpGemmImpl<WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 1, K_ / 16>>>;
};
template <typename DataType, index_t K_>
struct JengaQKChunkWarpGemmSelector
{
    using Type = std::conditional_t<
        std::is_same_v<DataType, bf16_t>,
        WarpGemmImpl<WarpGemmAttributeMmacIterateK<
            WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 2, 1, 1, K_ / 16>>,
        WarpGemmImpl<WarpGemmAttributeMmacIterateK<
            WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 2, 1, 1, K_ / 16>>>;
};
template <ck_tile::index_t M, ck_tile::index_t K>
CK_TILE_HOST_DEVICE constexpr auto MakeSimpleLdsDescriptor()
{
    return ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(ck_tile::number<M>{}, ck_tile::number<K>{}),
        ck_tile::make_tuple(ck_tile::number<K>{}, ck_tile::number<1>{}),
        ck_tile::number<8>{},
        ck_tile::number<1>{});
}
template <ck_tile::index_t K, ck_tile::index_t M>
CK_TILE_HOST_DEVICE constexpr auto MakeTransposedLdsDescriptor()
{
    const auto desc_raw = ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(ck_tile::number<M>{}, ck_tile::number<K>{}),
        ck_tile::make_tuple(ck_tile::number<K>{}, ck_tile::number<1>{}),
        ck_tile::number<8>{},
        ck_tile::number<1>{});
    return ck_tile::transform_tensor_descriptor(
        desc_raw,
        ck_tile::make_tuple(ck_tile::make_pass_through_transform(ck_tile::number<M>{}),
                            ck_tile::make_pass_through_transform(ck_tile::number<K>{})),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}),
        ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0>{}));
}
template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t Stride>
CK_TILE_HOST_DEVICE constexpr auto MakePaddedRowMajorLdsDescriptor()
{
    static_assert(Stride >= N, "padded LDS stride must cover row length");
    return ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(ck_tile::number<M>{}, ck_tile::number<N>{}),
        ck_tile::make_tuple(ck_tile::number<Stride>{}, ck_tile::number<1>{}),
        ck_tile::number<N>{},
        ck_tile::number<1>{});
}
static constexpr ck_tile::index_t kBwdAsyncQVector   = 2;
static constexpr ck_tile::index_t kBwdAsyncQPack     = 8;
static constexpr ck_tile::index_t kBwdAsyncQPad      = 16;
static constexpr ck_tile::index_t kBwdAsyncQNumWarps = 4;
static constexpr ck_tile::index_t kBwdAsyncQWarpSize = 64;
template <ck_tile::index_t M_, ck_tile::index_t K_>
struct JengaBwdAsyncQGeometry
{
    static_assert(M_ == 64 || M_ == 32, "async Q supports 64 or 32 query rows");
    static_assert((K_ == 32 || K_ == 64), "async Q supports 32/64 head chunks");
    static constexpr ck_tile::index_t kRows       = M_;
    static constexpr ck_tile::index_t kCols       = K_;
    static constexpr ck_tile::index_t kLanesPerD  = kCols / kBwdAsyncQVector;
    static constexpr ck_tile::index_t kLaneGroups = kBwdAsyncQWarpSize / kLanesPerD;
    static constexpr ck_tile::index_t kNumIssues  = kRows / (kLaneGroups * kBwdAsyncQNumWarps);
    static constexpr ck_tile::index_t kSmemElements =
        kNumIssues * kBwdAsyncQNumWarps *
        (kBwdAsyncQWarpSize * kBwdAsyncQVector + kBwdAsyncQPad);
    static constexpr ck_tile::index_t kDenseElements = kRows * kCols;
    static_assert(kSmemElements >= kDenseElements, "unexpected async Q LDS geometry");
};
static_assert(JengaBwdAsyncQGeometry<64, 32>::kSmemElements == 2304,
              "unexpected async Q32 LDS geometry");
static constexpr ck_tile::index_t kBwdAsyncDoQHalf    = 32;
static constexpr ck_tile::index_t kBwdAsyncDoVector   = 2;
static constexpr ck_tile::index_t kBwdAsyncDoPack     = 8;
static constexpr ck_tile::index_t kBwdAsyncDoPad      = 12;
static constexpr ck_tile::index_t kBwdAsyncDoNumWarps = 4;
static constexpr ck_tile::index_t kBwdAsyncDoWarpSize = 64;
template <ck_tile::index_t K_>
struct JengaBwdAsyncDoQHalfGeometry
{
    static_assert(K_ == 32, "async dO qhalf supports 32-wide head chunks");
    static constexpr ck_tile::index_t kRows       = kBwdAsyncDoQHalf;
    static constexpr ck_tile::index_t kCols       = K_;
    static constexpr ck_tile::index_t kLanesPerD  = kCols / kBwdAsyncDoVector;
    static constexpr ck_tile::index_t kLaneGroups = kBwdAsyncDoWarpSize / kLanesPerD;
    static constexpr ck_tile::index_t kNumIssues =
        kRows / (kLaneGroups * kBwdAsyncDoNumWarps);
    static constexpr ck_tile::index_t kSmemElements =
        kNumIssues * kBwdAsyncDoNumWarps *
        (kBwdAsyncDoWarpSize * kBwdAsyncDoVector + kBwdAsyncDoPad);
    static constexpr ck_tile::index_t kDenseElements = kRows * kCols;
    static_assert(kSmemElements >= kDenseElements, "unexpected async dO LDS geometry");
};
static_assert(JengaBwdAsyncDoQHalfGeometry<32>::kNumIssues == 2,
              "unexpected async dO qhalf issue count");
static_assert(JengaBwdAsyncDoQHalfGeometry<32>::kSmemElements == 1120,
              "unexpected async dO qhalf chunk LDS geometry");
template <ck_tile::index_t M_, ck_tile::index_t K_>
CK_TILE_HOST_DEVICE constexpr auto MakeBwdDoAsyncDramDistribution()
{
    static_assert(M_ == kBwdAsyncDoQHalf, "async dO supports 32 query rows");
    using G = JengaBwdAsyncDoQHalfGeometry<K_>;
    return ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<1>,
            ck_tile::tuple<ck_tile::sequence<G::kNumIssues,
                                             G::kLaneGroups,
                                             kBwdAsyncDoNumWarps>,
                           ck_tile::sequence<G::kLanesPerD, kBwdAsyncDoVector>>,
            ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
            ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<1, 0>>,
            ck_tile::sequence<1, 2>,
            ck_tile::sequence<0, 1>>{});
}
template <ck_tile::index_t K_, ck_tile::index_t BaseOffset, ck_tile::index_t Chunk = 0>
CK_TILE_HOST_DEVICE constexpr auto MakeBwdDoAsyncLdsStoreDescriptor()
{
    using G = JengaBwdAsyncDoQHalfGeometry<K_>;
    constexpr auto desc0 = ck_tile::make_naive_tensor_descriptor_with_offset(
        ck_tile::make_tuple(ck_tile::number<G::kNumIssues>{},
                            ck_tile::number<G::kLaneGroups>{},
                            ck_tile::number<kBwdAsyncDoNumWarps>{},
                            ck_tile::number<G::kLanesPerD>{},
                            ck_tile::number<kBwdAsyncDoVector>{}),
        ck_tile::make_tuple(ck_tile::number<kBwdAsyncDoNumWarps *
                                             (kBwdAsyncDoWarpSize * kBwdAsyncDoVector +
                                              kBwdAsyncDoPad)>{},
                            ck_tile::number<kBwdAsyncDoQHalf>{},
                            ck_tile::number<kBwdAsyncDoWarpSize * kBwdAsyncDoVector +
                                            kBwdAsyncDoPad>{},
                            ck_tile::number<kBwdAsyncDoVector>{},
                            ck_tile::number<1>{}),
        ck_tile::number<BaseOffset + Chunk * G::kSmemElements>{},
        ck_tile::number<kBwdAsyncDoVector>{},
        ck_tile::number<1>{});
    return ck_tile::transform_tensor_descriptor(
        desc0,
        ck_tile::make_tuple(
            ck_tile::make_pass_through_transform(ck_tile::number<G::kNumIssues>{}),
            ck_tile::make_pass_through_transform(ck_tile::number<kBwdAsyncDoNumWarps>{}),
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<G::kLaneGroups>{},
                                    ck_tile::number<G::kLanesPerD>{},
                                    ck_tile::number<kBwdAsyncDoVector>{}))),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<2>{},
                            ck_tile::sequence<1, 3, 4>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                            ck_tile::sequence<2>{}));
}
template <ck_tile::index_t K_, ck_tile::index_t BaseOffset, ck_tile::index_t Chunk = 0>
CK_TILE_HOST_DEVICE constexpr auto MakeBwdDoAsyncLdsLoadDescriptor()
{
    using G = JengaBwdAsyncDoQHalfGeometry<K_>;
    constexpr auto desc0 = ck_tile::make_naive_tensor_descriptor_with_offset(
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<G::kNumIssues>{},
                            ck_tile::number<kBwdAsyncDoNumWarps>{},
                            ck_tile::number<G::kLaneGroups>{},
                            ck_tile::number<G::kCols / kBwdAsyncDoPack>{},
                            ck_tile::number<kBwdAsyncDoPack>{}),
        ck_tile::make_tuple(ck_tile::number<G::kSmemElements>{},
                            ck_tile::number<kBwdAsyncDoNumWarps *
                                             (kBwdAsyncDoWarpSize * kBwdAsyncDoVector +
                                              kBwdAsyncDoPad)>{},
                            ck_tile::number<kBwdAsyncDoWarpSize * kBwdAsyncDoVector +
                                            kBwdAsyncDoPad>{},
                            ck_tile::number<G::kCols>{},
                            ck_tile::number<kBwdAsyncDoPack>{},
                            ck_tile::number<1>{}),
        ck_tile::number<BaseOffset + Chunk * G::kSmemElements>{},
        ck_tile::number<kBwdAsyncDoPack>{},
        ck_tile::number<1>{});
    return ck_tile::transform_tensor_descriptor(
        desc0,
        ck_tile::make_tuple(
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<G::kNumIssues>{},
                                    ck_tile::number<G::kLaneGroups>{},
                                    ck_tile::number<kBwdAsyncDoNumWarps>{})),
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<G::kCols / kBwdAsyncDoPack>{},
                                    ck_tile::number<kBwdAsyncDoPack>{}))),
        ck_tile::make_tuple(ck_tile::sequence<0, 1, 3, 2>{}, ck_tile::sequence<4, 5>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
}
template <ck_tile::index_t M_, ck_tile::index_t K_>
CK_TILE_HOST_DEVICE constexpr auto MakeBwdQAsyncDramDistribution()
{
    using G = JengaBwdAsyncQGeometry<M_, K_>;
    return ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<1>,
            ck_tile::tuple<ck_tile::sequence<G::kNumIssues, G::kLaneGroups, kBwdAsyncQNumWarps>,
                           ck_tile::sequence<G::kLanesPerD, kBwdAsyncQVector>>,
            ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
            ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<1, 0>>,
            ck_tile::sequence<1, 2>,
            ck_tile::sequence<0, 1>>{});
}
template <ck_tile::index_t M_, ck_tile::index_t K_, ck_tile::index_t Buf = 0>
CK_TILE_HOST_DEVICE constexpr auto MakeBwdQAsyncLdsStoreDescriptor()
{
    using G = JengaBwdAsyncQGeometry<M_, K_>;
    constexpr auto desc0 = ck_tile::make_naive_tensor_descriptor_with_offset(
        ck_tile::make_tuple(ck_tile::number<G::kNumIssues>{},
                            ck_tile::number<G::kLaneGroups>{},
                            ck_tile::number<kBwdAsyncQNumWarps>{},
                            ck_tile::number<G::kLanesPerD>{},
                            ck_tile::number<kBwdAsyncQVector>{}),
        ck_tile::make_tuple(ck_tile::number<kBwdAsyncQNumWarps *
                                            (kBwdAsyncQWarpSize * kBwdAsyncQVector +
                                             kBwdAsyncQPad)>{},
                            ck_tile::number<M_>{},
                            ck_tile::number<kBwdAsyncQWarpSize * kBwdAsyncQVector +
                                            kBwdAsyncQPad>{},
                            ck_tile::number<kBwdAsyncQVector>{},
                            ck_tile::number<1>{}),
        ck_tile::number<Buf * G::kSmemElements>{},
        ck_tile::number<kBwdAsyncQVector>{},
        ck_tile::number<1>{});
    return ck_tile::transform_tensor_descriptor(
        desc0,
        ck_tile::make_tuple(
            ck_tile::make_pass_through_transform(ck_tile::number<G::kNumIssues>{}),
            ck_tile::make_pass_through_transform(ck_tile::number<kBwdAsyncQNumWarps>{}),
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<G::kLaneGroups>{},
                                    ck_tile::number<G::kLanesPerD>{},
                                    ck_tile::number<kBwdAsyncQVector>{}))),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<2>{},
                            ck_tile::sequence<1, 3, 4>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{},
                            ck_tile::sequence<2>{}));
}
template <ck_tile::index_t M_, ck_tile::index_t K_>
CK_TILE_HOST_DEVICE constexpr auto MakeBwdQAsyncLdsLoadDescriptor()
{
    using G = JengaBwdAsyncQGeometry<M_, K_>;
    constexpr auto desc0 = ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<G::kNumIssues>{},
                            ck_tile::number<kBwdAsyncQNumWarps>{},
                            ck_tile::number<G::kLaneGroups>{},
                            ck_tile::number<G::kCols / kBwdAsyncQPack>{},
                            ck_tile::number<kBwdAsyncQPack>{}),
        ck_tile::make_tuple(ck_tile::number<G::kSmemElements>{},
                            ck_tile::number<kBwdAsyncQNumWarps *
                                             (kBwdAsyncQWarpSize * kBwdAsyncQVector +
                                              kBwdAsyncQPad)>{},
                            ck_tile::number<kBwdAsyncQWarpSize * kBwdAsyncQVector +
                                            kBwdAsyncQPad>{},
                            ck_tile::number<G::kCols>{},
                            ck_tile::number<kBwdAsyncQPack>{},
                            ck_tile::number<1>{}),
        ck_tile::number<kBwdAsyncQPack>{},
        ck_tile::number<1>{});
    return ck_tile::transform_tensor_descriptor(
        desc0,
        ck_tile::make_tuple(
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<G::kNumIssues>{},
                                    ck_tile::number<G::kLaneGroups>{},
                                    ck_tile::number<kBwdAsyncQNumWarps>{})),
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<G::kCols / kBwdAsyncQPack>{},
                                    ck_tile::number<kBwdAsyncQPack>{}))),
        ck_tile::make_tuple(ck_tile::sequence<0, 1, 3, 2>{}, ck_tile::sequence<4, 5>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
}
template <ck_tile::index_t M, ck_tile::index_t N, typename BlockGemm, typename AccTile, typename AccDataType>
CK_TILE_DEVICE void StoreMmacOutputTileToLdsRowMajor(const BlockGemm& block_gemm,
                                                     const AccTile& acc_tile,
                                                     AccDataType* out_smem)
{
    const auto out_tile = block_gemm.MakeOuputLayout(acc_tile);
    constexpr auto spans = decltype(out_tile)::get_distributed_spans();
    ck_tile::sweep_tile_span(spans[ck_tile::number<0>{}], [&](auto idx0) {
        ck_tile::sweep_tile_span(spans[ck_tile::number<1>{}], [&](auto idx1) {
            constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
            const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                out_tile.get_tile_distribution(), tile_idx);
            const ck_tile::index_t m = x_idx.at(ck_tile::number<0>{});
            const ck_tile::index_t n = x_idx.at(ck_tile::number<1>{});
            if(m < M && n < N)
            {
                out_smem[static_cast<ck_tile::long_index_t>(m) * N + n] = out_tile[tile_idx];
            }
        });
    });
}
template <typename Problem>
struct JengaBwdDqDefaultPolicy
{
    static constexpr bool LeanFusedStorage = true;
    static constexpr ck_tile::index_t PTransposeLdsStride = Problem::kBlockN + 12;
    static constexpr ck_tile::index_t DoQHalfLdsBaseOffset =
        Problem::kBlockM * PTransposeLdsStride;
    using DataType = typename Problem::QDataType;
    using QKWarpGemm = typename JengaWarpGemmSelector<DataType, 128>::Type;
    using DPK32WarpGemm = typename JengaWarpGemmSelector<DataType, 32>::Type;
    using QKChunkWarpGemm =
        typename JengaQKChunkWarpGemmSelector<DataType, JengaMmacQKChunkConfig::kK>::Type;
    using QKHalfWarpGemm =
        typename JengaWarpGemmSelector<DataType, JengaMmacQKHalfConfig::kK>::Type;
    using DVWarpGemm = typename JengaWarpGemmSelector<DataType, 16>::Type;
    using DKWarpGemm = typename JengaWarpGemmSelector<DataType, 64>::Type;
    struct QKPolicy
    {
        template <typename P>
        CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
        {
            return ck_tile::make_tuple(QKWarpGemm{},
                                       JengaMmacQKConfig::BlockWarps::at(ck_tile::number<0>{}),
                                       JengaMmacQKConfig::BlockWarps::at(ck_tile::number<1>{}));
        }
    };
    struct DPK32Policy
    {
        template <typename P>
        CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
        {
            return ck_tile::make_tuple(DPK32WarpGemm{},
                                       JengaMmacDPK32Config::BlockWarps::at(ck_tile::number<0>{}),
                                       JengaMmacDPK32Config::BlockWarps::at(ck_tile::number<1>{}));
        }
    };
    struct QKChunkPolicy
    {
        template <typename P>
        CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
        {
            return ck_tile::make_tuple(QKChunkWarpGemm{},
                                       JengaMmacQKChunkConfig::BlockWarps::at(ck_tile::number<0>{}),
                                       JengaMmacQKChunkConfig::BlockWarps::at(ck_tile::number<1>{}));
        }
    };
    struct DVPolicy
    {
        template <typename P>
        CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
        {
            return ck_tile::make_tuple(DVWarpGemm{},
                                       JengaMmacDVConfig::BlockWarps::at(ck_tile::number<0>{}),
                                       JengaMmacDVConfig::BlockWarps::at(ck_tile::number<1>{}));
        }
    };
    struct DKPolicy
    {
        template <typename P>
        CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
        {
            return ck_tile::make_tuple(DKWarpGemm{},
                                       JengaMmacDKConfig::BlockWarps::at(ck_tile::number<0>{}),
                                       JengaMmacDKConfig::BlockWarps::at(ck_tile::number<1>{}));
        }
    };
    struct DPPolicy
    {
        template <typename P>
        CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
        {
            return ck_tile::make_tuple(QKWarpGemm{},
                                       JengaMmacDPConfig::BlockWarps::at(ck_tile::number<0>{}),
                                       JengaMmacDPConfig::BlockWarps::at(ck_tile::number<1>{}));
        }
    };
    using QKBlockGemm = ck_tile::MmacBlockGemmASmemBSmemCRegV1<JengaMmacQKGemmProblem<Problem>, QKPolicy>;
    using QKARegBSmemBlockGemm = ck_tile::JengaLocalBlockGemmARegBSmemCReg<
        JengaMmacQKGemmProblem<Problem>,
        ck_tile::JengaLocalARegBSmemPolicy<typename JengaMmacQKConfig::BlockWarps, QKWarpGemm>>;
    using QKARegBRegBlockGemm = ck_tile::JengaLocalBlockGemmARegBRegCReg<
        JengaMmacQKGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::KDataType,
                                                     typename Problem::QDataType,
                                                     float,
                                                     typename JengaMmacQKConfig::BlockWarps,
                                                     QKWarpGemm>>;
    using QKASmemBRegBlockGemm = ck_tile::JengaLocalBlockGemmASmemBRegCReg<
        JengaMmacQKGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::KDataType,
                                                     typename Problem::QDataType,
                                                     float,
                                                     typename JengaMmacQKConfig::BlockWarps,
                                                     QKWarpGemm>>;
    using QKChunkASmemBRegBlockGemm = ck_tile::JengaLocalBlockGemmASmemBRegCReg<
        JengaMmacQKChunkGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::KDataType,
                                                     typename Problem::QDataType,
                                                     float,
                                                     typename JengaMmacQKChunkConfig::BlockWarps,
                                                     QKChunkWarpGemm>>;
    using QKHalfASmemBRegBlockGemm = ck_tile::JengaLocalBlockGemmASmemBRegCReg<
        JengaMmacQKHalfGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::KDataType,
                                                     typename Problem::QDataType,
                                                     float,
                                                     typename JengaMmacQKHalfConfig::BlockWarps,
                                                     QKHalfWarpGemm>>;
    using DVBlockGemm = ck_tile::MmacBlockGemmASmemBSmemCRegV1<JengaMmacDVGemmProblem<Problem>, DVPolicy>;
    using DVBRegBlockGemm = ck_tile::JengaLocalBlockGemmARegBRegCReg<
        JengaMmacDVGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::OGradDataType,
                                                     float,
                                                     typename JengaMmacDVConfig::BlockWarps,
                                                     DVWarpGemm>>;
    using DVK32BRegBlockGemm = ck_tile::JengaLocalBlockGemmARegBRegCReg<
        JengaMmacDVK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::OGradDataType,
                                                     float,
                                                     typename JengaMmacDVConfig::BlockWarps,
                                                     DVWarpGemm>>;
    using DPBlockGemm = ck_tile::MmacBlockGemmASmemBSmemCRegV1<JengaMmacDPGemmProblem<Problem>, DPPolicy>;
    using DPK32BlockGemm = ck_tile::JengaLocalBlockGemmASmemBRegCReg<
        JengaMmacDPK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::VDataType,
                                                     typename Problem::OGradDataType,
                                                     float,
                                                     typename JengaMmacDPK32Config::BlockWarps,
                                                     DPK32WarpGemm>>;
    using DPK32ARegBRegBlockGemm = ck_tile::JengaLocalBlockGemmARegBRegCReg<
        JengaMmacDPK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::VDataType,
                                                     typename Problem::OGradDataType,
                                                     float,
                                                     typename JengaMmacDPK32Config::BlockWarps,
                                                     DPK32WarpGemm>>;
    using DKBlockGemm = ck_tile::MmacBlockGemmASmemBSmemCRegV1<JengaMmacDKGemmProblem<Problem>, DKPolicy>;
    using DQK32BRegBlockGemm = ck_tile::JengaLocalBlockGemmARegBRegCReg<
        JengaMmacDQK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::KDataType,
                                                     float,
                                                     typename JengaMmacDVConfig::BlockWarps,
                                                     DVWarpGemm>>;
    template <typename ProblemType>
    static constexpr ck_tile::index_t GetSmemSize()
    {
        return sizeof(LdsStorage);
    }
    union LdsStorage
    {
        typename Problem::QDataType q_async[2 * JengaBwdAsyncQGeometry<Problem::kBlockM,
                                                                       JengaMmacQKChunkConfig::kK>::kSmemElements];
        struct
        {
            typename Problem::QDataType p_t[Problem::kBlockM * PTransposeLdsStride];
            typename Problem::OGradDataType do_qhalf[JengaBwdAsyncDoQHalfGeometry<
                32>::kSmemElements *
                                                     (Problem::kHeadDim / 32)];
        } dv_input;
    };
};
} // namespace ck_tile
