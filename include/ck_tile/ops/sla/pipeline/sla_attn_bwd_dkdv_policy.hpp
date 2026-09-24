// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"
#include "ck_tile/ops/sla/block/sla_attn_bwd_dkdv_block_gemm.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dkdv_problem.hpp"

namespace ck_tile {
namespace example {
namespace sla {

CK_TILE_DEVICE void SlaBwdBlockSyncLdsLight()
{
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}

CK_TILE_DEVICE float SlaBwdFastExp2(float x)
{
    return __builtin_amdgcn_exp2f(x);
}

using SlaWarpGemmMmacBF16BF16F32_WT16x16x64_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 4>>;

using SlaWarpGemmMmacBF16BF16F32_WT16x16x16_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 1>>;

using SlaWarpGemmMmacBF16BF16F32_WT16x16x32_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 2>>;

using SlaWarpGemmMmacBF16BF16F32_WT16x32x32_MR1NR2MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 2, 1, 1, 2>>;

template <typename DataType>
using SlaDkdvMmacImpl = std::conditional_t<
    std::is_same_v<DataType, ck_tile::fp16_t>,
    WarpGemmAttributeMmacImplF16F16F32M16N16K16,
    WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16>;

template <typename DataType, int NIter, int KIter>
using SlaDkdvWarpGemm = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<SlaDkdvMmacImpl<DataType>, 1, NIter, 1, 1, KIter>>;

template <typename T> using SlaDkdvWarp16x16x128 = SlaDkdvWarpGemm<T, 1, 8>;
template <typename T> using SlaDkdvWarp16x16x64  = SlaDkdvWarpGemm<T, 1, 4>;
template <typename T> using SlaDkdvWarp16x16x32  = SlaDkdvWarpGemm<T, 1, 2>;
template <typename T> using SlaDkdvWarp16x16x16  = SlaDkdvWarpGemm<T, 1, 1>;
template <typename T> using SlaDkdvWarp16x32x32  = SlaDkdvWarpGemm<T, 2, 2>;

struct SlaMmacQKConfig
{
    static constexpr ck_tile::index_t kN         = 64;
    static constexpr ck_tile::index_t kK         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 128>;
    using WarpGemm   = ck_tile::WarpGemmMmacBF16BF16F32_WT16x16x128_MR1NR1MI1NI1;
};

struct SlaMmacDPK32Config
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 32;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 32>;
    using WarpGemm   = SlaWarpGemmMmacBF16BF16F32_WT16x16x32_MR1NR1MI1NI1;
};

struct SlaMmacDPK32Policy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(SlaDkdvWarp16x16x32<typename Problem::QDataType>{},
                                   SlaMmacDPK32Config::BlockWarps::at(ck_tile::number<0>{}),
                                   SlaMmacDPK32Config::BlockWarps::at(ck_tile::number<1>{}));
    }
};

struct SlaMmacQKPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(SlaDkdvWarp16x16x128<typename Problem::QDataType>{},
                                   SlaMmacQKConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   SlaMmacQKConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

struct SlaMmacQKChunkConfig
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 32;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<4, 1, 1>;
    using WarpTile   = ck_tile::sequence<16, 32, 32>;
    using WarpGemm   = SlaWarpGemmMmacBF16BF16F32_WT16x32x32_MR1NR2MI1NI1;
};

struct SlaMmacQKChunkPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(SlaDkdvWarp16x32x32<typename Problem::QDataType>{},
                                   SlaMmacQKChunkConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   SlaMmacQKChunkConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

template <typename Problem>
using SlaMmacQKGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::BlockM,
                                             SlaMmacQKConfig::kN,
                                             SlaMmacQKConfig::kK>,
                           typename SlaMmacQKConfig::BlockWarps,
                           typename SlaMmacQKConfig::WarpTile>;

template <typename Problem>
using SlaMmacQKGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::KDataType,
                              typename Problem::AccDataType,
                              256,
                              SlaMmacQKGemmShape<Problem>>;

template <typename Problem>
using SlaMmacQKChunkGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::BlockM,
                                             SlaMmacQKChunkConfig::kN,
                                             SlaMmacQKChunkConfig::kK>,
                           typename SlaMmacQKChunkConfig::BlockWarps,
                           typename SlaMmacQKChunkConfig::WarpTile>;

template <typename Problem>
using SlaMmacQKChunkGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::KDataType,
                              typename Problem::AccDataType,
                              256,
                              SlaMmacQKChunkGemmShape<Problem>>;

template <typename Problem>
using SlaMmacQKChunkASmemBRegBlockGemm =
    ck_tile::SlaLocalBlockGemmASmemBRegCReg<
        SlaMmacQKChunkGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::KDataType,
                                                     typename Problem::AccDataType,
                                                     typename SlaMmacQKChunkConfig::BlockWarps,
                                                     SlaDkdvWarp16x32x32<typename Problem::QDataType>>>;

template <typename Problem>
struct SlaMmacDVConfigFor
{
    static constexpr ck_tile::index_t kM         = Problem::BlockN;
    static constexpr ck_tile::index_t kN         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = std::conditional_t<Problem::BlockN == 32,
                                          ck_tile::sequence<2, 2, 1>,
                                          ck_tile::sequence<1, 4, 1>>;
    using WarpTile   = ck_tile::sequence<16, 64, 32>;
    using WarpGemm   = SlaWarpGemmMmacBF16BF16F32_WT16x16x16_MR1NR1MI1NI1;
};

struct SlaMmacDVPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        using Config = SlaMmacDVConfigFor<Problem>;
        return ck_tile::make_tuple(SlaDkdvWarp16x16x16<typename Problem::QDataType>{},
                                   Config::BlockWarps::at(ck_tile::number<0>{}),
                                   Config::BlockWarps::at(ck_tile::number<1>{}));
    }
};

template <typename Problem>
using SlaMmacDVGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<SlaMmacDVConfigFor<Problem>::kM,
                                             SlaMmacDVConfigFor<Problem>::kN,
                                             Problem::BlockM>,
                           typename SlaMmacDVConfigFor<Problem>::BlockWarps,
                           typename SlaMmacDVConfigFor<Problem>::WarpTile>;

template <typename Problem>
using SlaMmacDVGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::OGradDataType,
                              typename Problem::AccDataType,
                              256,
                              SlaMmacDVGemmShape<Problem>>;

template <typename Problem>
using SlaMmacDVBRegBlockGemm =
    ck_tile::SlaLocalBlockGemmARegBRegCReg<
        SlaMmacDVGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::OGradDataType,
                                                     typename Problem::AccDataType,
                                                     typename SlaMmacDVConfigFor<Problem>::BlockWarps,
                                                     SlaDkdvWarp16x16x16<typename Problem::QDataType>>>;

template <typename Problem>
using SlaMmacDVK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<SlaMmacDVConfigFor<Problem>::kM,
                                             SlaMmacDVConfigFor<Problem>::kN,
                                             32>,
                           typename SlaMmacDVConfigFor<Problem>::BlockWarps,
                           typename SlaMmacDVConfigFor<Problem>::WarpTile>;

template <typename Problem>
using SlaMmacDVK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::OGradDataType,
                              typename Problem::AccDataType,
                              256,
                              SlaMmacDVK32GemmShape<Problem>>;

template <typename Problem>
using SlaMmacDVK32BRegBlockGemm =
    ck_tile::SlaLocalBlockGemmARegBRegCRegInterleavedB<
        SlaMmacDVK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::OGradDataType,
                                                     typename Problem::AccDataType,
                                                     typename SlaMmacDVConfigFor<Problem>::BlockWarps,
                                                     SlaDkdvWarp16x16x16<typename Problem::QDataType>>>;


struct SlaMmacDPConfig
{
    static constexpr ck_tile::index_t kN         = 64;
    static constexpr ck_tile::index_t kK         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 128>;
    using WarpGemm   = ck_tile::WarpGemmMmacBF16BF16F32_WT16x16x128_MR1NR1MI1NI1;
};

struct SlaMmacDPPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(SlaDkdvWarp16x16x128<typename Problem::QDataType>{},
                                   SlaMmacDPConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   SlaMmacDPConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

template <typename Problem>
using SlaMmacDPGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::BlockM,
                                             SlaMmacDPConfig::kN,
                                             SlaMmacDPConfig::kK>,
                           typename SlaMmacDPConfig::BlockWarps,
                           typename SlaMmacDPConfig::WarpTile>;

template <typename Problem>
using SlaMmacDPGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::OGradDataType,
                              typename Problem::VDataType,
                              typename Problem::AccDataType,
                              256,
                              SlaMmacDPGemmShape<Problem>>;

template <typename Problem>
using SlaMmacDPK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::BlockN,
                                             SlaMmacDPK32Config::kN,
                                             SlaMmacDPK32Config::kK>,
                           typename SlaMmacDPK32Config::BlockWarps,
                           typename SlaMmacDPK32Config::WarpTile>;

template <typename Problem>
using SlaMmacDPK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::OGradDataType,
                              typename Problem::VDataType,
                              typename Problem::AccDataType,
                              256,
                              SlaMmacDPK32GemmShape<Problem>>;

template <typename Problem>
using SlaMmacDPK32BlockGemm =
    ck_tile::SlaLocalBlockGemmASmemBRegCReg<
        SlaMmacDPK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::VDataType,
                                                     typename Problem::OGradDataType,
                                                     typename Problem::AccDataType,
                                                     typename SlaMmacDPK32Config::BlockWarps,
                                                     SlaDkdvWarp16x16x32<typename Problem::QDataType>>>;

template <typename Problem>
using SlaMmacDPK32ARegBRegBlockGemm =
    ck_tile::SlaLocalBlockGemmARegBRegCReg<
        SlaMmacDPK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::VDataType,
                                                     typename Problem::OGradDataType,
                                                     typename Problem::AccDataType,
                                                     typename SlaMmacDPK32Config::BlockWarps,
                                                     SlaDkdvWarp16x16x32<typename Problem::QDataType>>>;

struct SlaMmacDKConfig
{
    static constexpr ck_tile::index_t kM         = 64;
    static constexpr ck_tile::index_t kN         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 64>;
    using WarpGemm   = SlaWarpGemmMmacBF16BF16F32_WT16x16x64_MR1NR1MI1NI1;
};

struct SlaMmacDKPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(SlaDkdvWarp16x16x64<typename Problem::QDataType>{},
                                   SlaMmacDKConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   SlaMmacDKConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

template <typename Problem>
using SlaMmacDKGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<SlaMmacDKConfig::kM,
                                             SlaMmacDKConfig::kN,
                                             Problem::BlockM>,
                           typename SlaMmacDKConfig::BlockWarps,
                           typename SlaMmacDKConfig::WarpTile>;

template <typename Problem>
using SlaMmacDKGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::QDataType,
                              typename Problem::AccDataType,
                              256,
                              SlaMmacDKGemmShape<Problem>>;

template <typename Problem>
using SlaMmacDKK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<SlaMmacDVConfigFor<Problem>::kM,
                                             SlaMmacDVConfigFor<Problem>::kN,
                                             32>,
                           typename SlaMmacDVConfigFor<Problem>::BlockWarps,
                           typename SlaMmacDVConfigFor<Problem>::WarpTile>;

template <typename Problem>
using SlaMmacDKK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::QDataType,
                              typename Problem::AccDataType,
                              256,
                              SlaMmacDKK32GemmShape<Problem>>;

template <typename Problem>
using SlaMmacDKK32BRegBlockGemm =
    ck_tile::SlaLocalBlockGemmARegBRegCRegInterleavedB<
        SlaMmacDKK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::QDataType,
                                                     typename Problem::AccDataType,
                                                     typename SlaMmacDVConfigFor<Problem>::BlockWarps,
                                                     SlaDkdvWarp16x16x16<typename Problem::QDataType>>>;

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

static constexpr ck_tile::index_t kBwdAsyncQVector   = 8;
static constexpr ck_tile::index_t kBwdAsyncQPack     = 8;
static constexpr ck_tile::index_t kBwdAsyncQPad      = 16;
static constexpr ck_tile::index_t kBwdAsyncQNumWarps = 4;
static constexpr ck_tile::index_t kBwdAsyncQWarpSize = 64;

template <ck_tile::index_t M_, ck_tile::index_t K_>
struct SlaBwdAsyncQGeometry
{
    static_assert(M_ == 64, "async Q supports 64 query rows");
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

static_assert(SlaBwdAsyncQGeometry<64, 32>::kNumIssues == 1,
              "unexpected async Q32 issue count");
static_assert(SlaBwdAsyncQGeometry<64, 32>::kSmemElements == 2112,
              "unexpected async Q32 LDS geometry");

static constexpr ck_tile::index_t kBwdAsyncDoQHalf    = 32;
static constexpr ck_tile::index_t kBwdAsyncDoVector   = 4;
static constexpr ck_tile::index_t kBwdAsyncDoPack     = 8;
static constexpr ck_tile::index_t kBwdAsyncDoPad      = 12;
static constexpr ck_tile::index_t kBwdAsyncDoNumWarps = 4;
static constexpr ck_tile::index_t kBwdAsyncDoWarpSize = 64;

template <ck_tile::index_t K_>
struct SlaBwdAsyncDoQHalfGeometry
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

static_assert(SlaBwdAsyncDoQHalfGeometry<32>::kNumIssues == 1,
              "unexpected async dO qhalf issue count");
static_assert(SlaBwdAsyncDoQHalfGeometry<32>::kSmemElements == 1072,
              "unexpected async dO qhalf chunk LDS geometry");

template <ck_tile::index_t M_, ck_tile::index_t K_>
CK_TILE_HOST_DEVICE constexpr auto MakeBwdDoAsyncDramDistribution()
{
    static_assert(M_ == kBwdAsyncDoQHalf, "async dO supports 32 query rows");
    using G = SlaBwdAsyncDoQHalfGeometry<K_>;
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
    using G = SlaBwdAsyncDoQHalfGeometry<K_>;
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
    using G = SlaBwdAsyncDoQHalfGeometry<K_>;
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
    using G = SlaBwdAsyncQGeometry<M_, K_>;
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
    using G = SlaBwdAsyncQGeometry<M_, K_>;
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
    using G = SlaBwdAsyncQGeometry<M_, K_>;
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
struct SlaAttnBwdDkdvDefaultPolicy
{
    static constexpr bool LeanFusedStorage = true;
    static constexpr ck_tile::index_t PTransposeLdsStride = Problem::BlockM + 4;
    static constexpr ck_tile::index_t DoQHalfLdsBaseOffset =
        Problem::BlockN * PTransposeLdsStride;

    using QKChunkASmemBRegBlockGemm = SlaMmacQKChunkASmemBRegBlockGemm<Problem>;
    using DVBRegBlockGemm = SlaMmacDVBRegBlockGemm<Problem>;
    using DVK32BRegBlockGemm = SlaMmacDVK32BRegBlockGemm<Problem>;
    using DPK32BlockGemm = SlaMmacDPK32BlockGemm<Problem>;
    using DPK32ARegBRegBlockGemm = SlaMmacDPK32ARegBRegBlockGemm<Problem>;
    using DKK32BRegBlockGemm = SlaMmacDKK32BRegBlockGemm<Problem>;

    union LdsStorage
    {
        typename Problem::QDataType q_async[4 * SlaBwdAsyncQGeometry<Problem::BlockM,
                                                                       SlaMmacQKChunkConfig::kK>::kSmemElements];
        struct
        {
            typename Problem::QDataType p_t[Problem::BlockN * PTransposeLdsStride];
            typename Problem::OGradDataType do_qhalf[SlaBwdAsyncDoQHalfGeometry<
                SlaMmacQKChunkConfig::kK>::kSmemElements *
                                                     (Problem::HeadDim / SlaMmacQKChunkConfig::kK)];
        } dv_input;
    };
};

} // namespace sla
} // namespace example
} // namespace ck_tile
