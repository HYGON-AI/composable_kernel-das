// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"
#include "ck_tile/ops/jenga/block/jenga_bwd_dkdv_breg_gemm.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dkdv_config.hpp"

namespace ck_tile {
namespace example {
namespace jenga {

CK_TILE_DEVICE void JengaBwdBlockSyncLdsLight()
{
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}

CK_TILE_DEVICE float JengaBwdFastExp2(float x)
{
    return __builtin_amdgcn_exp2f(x);
}

using JengaWarpGemmMmacBF16BF16F32_WT16x16x64_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 4>>;

using JengaWarpGemmMmacBF16BF16F32_WT16x16x16_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 1>>;

using JengaWarpGemmMmacBF16BF16F32_WT16x16x32_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 2>>;

using JengaWarpGemmMmacBF16BF16F32_WT16x32x32_MR1NR2MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 2, 1, 1, 2>>;

template <typename DataType>
using JengaDkdvMmacImpl = std::conditional_t<
    std::is_same_v<DataType, ck_tile::fp16_t>,
    WarpGemmAttributeMmacImplF16F16F32M16N16K16,
    WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16>;

template <typename DataType, int NIter, int KIter>
using JengaDkdvWarpGemm = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<JengaDkdvMmacImpl<DataType>, 1, NIter, 1, 1, KIter>>;

template <typename T> using JengaDkdvWarp16x16x128 = JengaDkdvWarpGemm<T, 1, 8>;
template <typename T> using JengaDkdvWarp16x16x64  = JengaDkdvWarpGemm<T, 1, 4>;
template <typename T> using JengaDkdvWarp16x16x32  = JengaDkdvWarpGemm<T, 1, 2>;
template <typename T> using JengaDkdvWarp16x16x16  = JengaDkdvWarpGemm<T, 1, 1>;
template <typename T> using JengaDkdvWarp16x32x32  = JengaDkdvWarpGemm<T, 2, 2>;

struct JengaMmacQKConfig
{
    static constexpr ck_tile::index_t kN         = 64;
    static constexpr ck_tile::index_t kK         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 128>;
    using WarpGemm   = ck_tile::WarpGemmMmacBF16BF16F32_WT16x16x128_MR1NR1MI1NI1;
};

struct JengaMmacDPK32Config
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 32;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 32>;
    using WarpGemm   = JengaWarpGemmMmacBF16BF16F32_WT16x16x32_MR1NR1MI1NI1;
};

struct JengaMmacDPK32Policy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(JengaDkdvWarp16x16x32<typename Problem::QDataType>{},
                                   JengaMmacDPK32Config::BlockWarps::at(ck_tile::number<0>{}),
                                   JengaMmacDPK32Config::BlockWarps::at(ck_tile::number<1>{}));
    }
};

struct JengaMmacQKPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(JengaDkdvWarp16x16x128<typename Problem::QDataType>{},
                                   JengaMmacQKConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   JengaMmacQKConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

struct JengaMmacQKChunkConfig
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 32;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<4, 1, 1>;
    using WarpTile   = ck_tile::sequence<16, 32, 32>;
    using WarpGemm   = JengaWarpGemmMmacBF16BF16F32_WT16x32x32_MR1NR2MI1NI1;
};

struct JengaMmacQKChunkPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(JengaDkdvWarp16x32x32<typename Problem::QDataType>{},
                                   JengaMmacQKChunkConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   JengaMmacQKChunkConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

template <typename Problem>
using JengaMmacQKGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::BlockM,
                                             JengaMmacQKConfig::kN,
                                             JengaMmacQKConfig::kK>,
                           typename JengaMmacQKConfig::BlockWarps,
                           typename JengaMmacQKConfig::WarpTile>;

template <typename Problem>
using JengaMmacQKGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::KDataType,
                              typename Problem::AccDataType,
                              Problem::ThreadsPerBlock,
                              JengaMmacQKGemmShape<Problem>>;

template <typename Problem>
using JengaMmacQKChunkGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::BlockM,
                                             JengaMmacQKChunkConfig::kN,
                                             JengaMmacQKChunkConfig::kK>,
                           typename JengaMmacQKChunkConfig::BlockWarps,
                           typename JengaMmacQKChunkConfig::WarpTile>;

template <typename Problem>
using JengaMmacQKChunkGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::KDataType,
                              typename Problem::AccDataType,
                              Problem::ThreadsPerBlock,
                              JengaMmacQKChunkGemmShape<Problem>>;

template <typename Problem>
using JengaMmacQKChunkASmemBRegBlockGemm =
    ck_tile::JengaDkdvLocalBlockGemmASmemBRegCReg<
        JengaMmacQKChunkGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::KDataType,
                                                     typename Problem::AccDataType,
                                                     typename JengaMmacQKChunkConfig::BlockWarps,
                                                     JengaDkdvWarp16x32x32<typename Problem::QDataType>>>;

struct JengaMmacDVConfig
{
    static constexpr ck_tile::index_t kM         = 64;
    static constexpr ck_tile::index_t kN         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<1, 4, 1>;
    using WarpTile   = ck_tile::sequence<16, 64, 32>;
    using WarpGemm   = JengaWarpGemmMmacBF16BF16F32_WT16x16x16_MR1NR1MI1NI1;
};

struct JengaMmacDVPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(JengaDkdvWarp16x16x16<typename Problem::QDataType>{},
                                   JengaMmacDVConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   JengaMmacDVConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

template <typename Problem>
using JengaMmacDVGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<JengaMmacDVConfig::kM,
                                             JengaMmacDVConfig::kN,
                                             Problem::BlockM>,
                           typename JengaMmacDVConfig::BlockWarps,
                           typename JengaMmacDVConfig::WarpTile>;

template <typename Problem>
using JengaMmacDVGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::OGradDataType,
                              typename Problem::AccDataType,
                              Problem::ThreadsPerBlock,
                              JengaMmacDVGemmShape<Problem>>;

template <typename Problem>
using JengaMmacDVBRegBlockGemm =
    ck_tile::JengaDkdvLocalBlockGemmARegBRegCReg<
        JengaMmacDVGemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::OGradDataType,
                                                     typename Problem::AccDataType,
                                                     typename JengaMmacDVConfig::BlockWarps,
                                                     JengaDkdvWarp16x16x16<typename Problem::QDataType>>>;

template <typename Problem>
using JengaMmacDVK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<JengaMmacDVConfig::kM,
                                             JengaMmacDVConfig::kN,
                                             32>,
                           typename JengaMmacDVConfig::BlockWarps,
                           typename JengaMmacDVConfig::WarpTile>;

template <typename Problem>
using JengaMmacDVK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::OGradDataType,
                              typename Problem::AccDataType,
                              Problem::ThreadsPerBlock,
                              JengaMmacDVK32GemmShape<Problem>>;

template <typename Problem>
using JengaMmacDVK32BRegBlockGemm =
    ck_tile::JengaDkdvLocalBlockGemmARegBRegCReg<
        JengaMmacDVK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::OGradDataType,
                                                     typename Problem::AccDataType,
                                                     typename JengaMmacDVConfig::BlockWarps,
                                                     JengaDkdvWarp16x16x16<typename Problem::QDataType>>>;


struct JengaMmacDPConfig
{
    static constexpr ck_tile::index_t kN         = 64;
    static constexpr ck_tile::index_t kK         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 128>;
    using WarpGemm   = ck_tile::WarpGemmMmacBF16BF16F32_WT16x16x128_MR1NR1MI1NI1;
};

struct JengaMmacDPPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(JengaDkdvWarp16x16x128<typename Problem::QDataType>{},
                                   JengaMmacDPConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   JengaMmacDPConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

template <typename Problem>
using JengaMmacDPGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::BlockM,
                                             JengaMmacDPConfig::kN,
                                             JengaMmacDPConfig::kK>,
                           typename JengaMmacDPConfig::BlockWarps,
                           typename JengaMmacDPConfig::WarpTile>;

template <typename Problem>
using JengaMmacDPGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::OGradDataType,
                              typename Problem::VDataType,
                              typename Problem::AccDataType,
                              Problem::ThreadsPerBlock,
                              JengaMmacDPGemmShape<Problem>>;

template <typename Problem>
using JengaMmacDPK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::BlockM,
                                             JengaMmacDPK32Config::kN,
                                             JengaMmacDPK32Config::kK>,
                           typename JengaMmacDPK32Config::BlockWarps,
                           typename JengaMmacDPK32Config::WarpTile>;

template <typename Problem>
using JengaMmacDPK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::OGradDataType,
                              typename Problem::VDataType,
                              typename Problem::AccDataType,
                              Problem::ThreadsPerBlock,
                              JengaMmacDPK32GemmShape<Problem>>;

template <typename Problem>
using JengaMmacDPK32BlockGemm =
    ck_tile::JengaDkdvLocalBlockGemmASmemBRegCReg<
        JengaMmacDPK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::VDataType,
                                                     typename Problem::OGradDataType,
                                                     typename Problem::AccDataType,
                                                     typename JengaMmacDPK32Config::BlockWarps,
                                                     JengaDkdvWarp16x16x32<typename Problem::QDataType>>>;

template <typename Problem>
using JengaMmacDPK32ARegBRegBlockGemm =
    ck_tile::JengaDkdvLocalBlockGemmARegBRegCReg<
        JengaMmacDPK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::VDataType,
                                                     typename Problem::OGradDataType,
                                                     typename Problem::AccDataType,
                                                     typename JengaMmacDPK32Config::BlockWarps,
                                                     JengaDkdvWarp16x16x32<typename Problem::QDataType>>>;

struct JengaMmacDKConfig
{
    static constexpr ck_tile::index_t kM         = 64;
    static constexpr ck_tile::index_t kN         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;

    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 64>;
    using WarpGemm   = JengaWarpGemmMmacBF16BF16F32_WT16x16x64_MR1NR1MI1NI1;
};

struct JengaMmacDKPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(JengaDkdvWarp16x16x64<typename Problem::QDataType>{},
                                   JengaMmacDKConfig::BlockWarps::at(ck_tile::number<0>{}),
                                   JengaMmacDKConfig::BlockWarps::at(ck_tile::number<1>{}));
    }
};

template <typename Problem>
using JengaMmacDKGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<JengaMmacDKConfig::kM,
                                             JengaMmacDKConfig::kN,
                                             Problem::BlockM>,
                           typename JengaMmacDKConfig::BlockWarps,
                           typename JengaMmacDKConfig::WarpTile>;

template <typename Problem>
using JengaMmacDKGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::QDataType,
                              typename Problem::AccDataType,
                              Problem::ThreadsPerBlock,
                              JengaMmacDKGemmShape<Problem>>;

template <typename Problem>
using JengaMmacDKK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<JengaMmacDVConfig::kM,
                                             JengaMmacDVConfig::kN,
                                             32>,
                           typename JengaMmacDVConfig::BlockWarps,
                           typename JengaMmacDVConfig::WarpTile>;

template <typename Problem>
using JengaMmacDKK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::QDataType,
                              typename Problem::AccDataType,
                              Problem::ThreadsPerBlock,
                              JengaMmacDKK32GemmShape<Problem>>;

template <typename Problem>
using JengaMmacDKK32BRegBlockGemm =
    ck_tile::JengaDkdvLocalBlockGemmARegBRegCReg<
        JengaMmacDKK32GemmProblem<Problem>,
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                     typename Problem::QDataType,
                                                     typename Problem::AccDataType,
                                                     typename JengaMmacDVConfig::BlockWarps,
                                                     JengaDkdvWarp16x16x16<typename Problem::QDataType>>>;

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
struct JengaBwdDkdvDefaultPolicy
{
    static constexpr bool LeanFusedStorage = true;
    static constexpr ck_tile::index_t PTransposeLdsStride = Problem::BlockM + 12;
    static constexpr ck_tile::index_t DoQHalfLdsBaseOffset =
        Problem::BlockN * PTransposeLdsStride;

    using QKChunkASmemBRegBlockGemm = JengaMmacQKChunkASmemBRegBlockGemm<Problem>;
    using DVBRegBlockGemm = JengaMmacDVBRegBlockGemm<Problem>;
    using DVK32BRegBlockGemm = JengaMmacDVK32BRegBlockGemm<Problem>;
    using DPK32BlockGemm = JengaMmacDPK32BlockGemm<Problem>;
    using DPK32ARegBRegBlockGemm = JengaMmacDPK32ARegBRegBlockGemm<Problem>;
    using DKK32BRegBlockGemm = JengaMmacDKK32BRegBlockGemm<Problem>;

    union LdsStorage
    {
        typename Problem::QDataType q_async[2 * JengaBwdAsyncQGeometry<Problem::BlockM,
                                                                       JengaMmacQKChunkConfig::kK>::kSmemElements];
        struct
        {
            typename Problem::QDataType p_t[Problem::BlockN * PTransposeLdsStride];
            typename Problem::OGradDataType do_qhalf[JengaBwdAsyncDoQHalfGeometry<
                JengaMmacQKChunkConfig::kK>::kSmemElements *
                                                     (Problem::HeadDim / JengaMmacQKChunkConfig::kK)];
        } dv_input;
    };
};

} // namespace jenga
} // namespace example
} // namespace ck_tile
