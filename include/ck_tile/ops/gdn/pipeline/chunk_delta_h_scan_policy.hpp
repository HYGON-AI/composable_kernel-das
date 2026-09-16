// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include <ck_tile/core.hpp>

#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/block/mmac_block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"

namespace ck_tile {

using GdnWarpGemmF16_16x16x64 = WarpGemmImpl<WarpGemmAttributeMmacIterateK<
    WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 1, 4>>;
using GdnWarpGemmBf16_16x16x64 = WarpGemmImpl<WarpGemmAttributeMmacIterateK<
    WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 4>>;
using GdnWarpGemmF16_16x32x64 = WarpGemmImpl<WarpGemmAttributeMmacIterateK<
    WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 2, 1, 1, 4>>;
using GdnWarpGemmBf16_16x32x64 = WarpGemmImpl<WarpGemmAttributeMmacIterateK<
    WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 2, 1, 1, 4>>;
using GdnWarpGemmF16_16x64x64 = WarpGemmImpl<WarpGemmAttributeMmacIterateK<
    WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 4, 1, 1, 4>>;
using GdnWarpGemmBf16_16x64x64 = WarpGemmImpl<WarpGemmAttributeMmacIterateK<
    WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 4, 1, 1, 4>>;

template <typename DataType>
using GdnWarpGemm16x16x64 = std::conditional_t<
    std::is_same_v<DataType, bf16_t>, GdnWarpGemmBf16_16x16x64, GdnWarpGemmF16_16x16x64>;
template <typename DataType>
using GdnWarpGemm16x32x64 = std::conditional_t<
    std::is_same_v<DataType, bf16_t>, GdnWarpGemmBf16_16x32x64, GdnWarpGemmF16_16x32x64>;
template <typename DataType>
using GdnWarpGemm16x64x64 = std::conditional_t<
    std::is_same_v<DataType, bf16_t>, GdnWarpGemmBf16_16x64x64, GdnWarpGemmF16_16x64x64>;

template <typename DataType>
struct GdnChunkMmacPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        if constexpr(Problem::BlockGemmShape::kN == 64)
            return make_tuple(GdnWarpGemm16x64x64<DataType>{}, number<4>{}, number<1>{});
        else if constexpr(Problem::BlockGemmShape::kN == 32)
            return make_tuple(GdnWarpGemm16x32x64<DataType>{}, number<4>{}, number<1>{});
        else
            return make_tuple(GdnWarpGemm16x16x64<DataType>{}, number<4>{}, number<1>{});
    }
};

template <typename DataType, index_t M, index_t N, index_t K>
using GdnChunkMmacShape =
    TileGemmShape<sequence<M, N, K>, sequence<4, 1, 1>, sequence<16, N, 64>>;

template <typename DataType, index_t M, index_t N, index_t K, index_t BlockSize>
using GdnChunkMmacProblem = BlockGemmProblem<
    DataType, DataType, float, BlockSize, GdnChunkMmacShape<DataType, M, N, K>>;

template <typename DataType, index_t M, index_t N, index_t K, index_t BlockSize>
using GdnChunkMmacBlockGemm = MmacBlockGemmASmemBSmemCRegV1<
    GdnChunkMmacProblem<DataType, M, N, K, BlockSize>, GdnChunkMmacPolicy<DataType>>;

template <index_t M, index_t K, index_t Pad = 4>
CK_TILE_HOST_DEVICE constexpr auto MakeGdnSimpleLdsDescriptor()
{
    return make_naive_tensor_descriptor(make_tuple(number<M>{}, number<K>{}),
                                        make_tuple(number<K + Pad>{}, number<1>{}),
                                        number<8>{},
                                        number<1>{});
}

template <index_t Rows, index_t Cols, index_t Vector = 8>
struct GdnAsyncCopyGeometry
{
    static constexpr index_t kWarpSize   = 64;
    static constexpr index_t kVector     = Vector;
    static constexpr index_t kNumWarps   = 4;
    static constexpr index_t kLanesPerRow = Cols / kVector;
    static constexpr index_t kRowGroups  = kWarpSize / kLanesPerRow;
    static constexpr index_t kNumIssues  = Rows / (kRowGroups * kNumWarps);

    static_assert(Cols % kVector == 0);
    static_assert(Rows % (kRowGroups * kNumWarps) == 0);
};

template <index_t Rows, index_t Cols, index_t Vector = 8>
CK_TILE_HOST_DEVICE constexpr auto MakeGdnAsyncDramDistribution()
{
    using G = GdnAsyncCopyGeometry<Rows, Cols, Vector>;
    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>,
            tuple<sequence<G::kNumIssues, G::kRowGroups, G::kNumWarps>,
                  sequence<G::kLanesPerRow, G::kVector>>,
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<2>, sequence<1, 0>>,
            sequence<1, 2>,
            sequence<0, 1>>{});
}

template <index_t VTile, bool UseMainPipeline, bool GuardTail = false>
struct ChunkDeltaHScanPolicy
{
    static_assert(VTile == 16 || VTile == 32 || VTile == 64 ||
                  VTile == 128);

    static constexpr index_t kChunkSize = 64;
    static constexpr index_t kHeadDim   = 128;
    static constexpr index_t kValueDim  = 128;
    static constexpr index_t kVTile     = VTile;
    static constexpr index_t kBlockSize =
        UseMainPipeline ? (VTile / 16 + 4) * 64 : 256;
    // Only Replay128 uses four independent LDS tiles (~39 KiB).  Smaller
    // VTiles retain the two-tile layout and two-block launch bound.
    static constexpr index_t kBlockPerCu = VTile == 128 ? 1 : 2;
    static constexpr index_t kSPT = kHeadDim * kVTile / kBlockSize;
    static constexpr bool kSpecializeCommonFlags = UseMainPipeline;
    static constexpr bool kGuardTail = GuardTail;

    template <typename DataType>
    using ProjectionBlockGemm =
        GdnChunkMmacBlockGemm<DataType, kChunkSize, kVTile, kHeadDim, kBlockSize>;

    template <typename DataType>
    using UpdateBlockGemm =
        GdnChunkMmacBlockGemm<DataType, kHeadDim, kVTile, kChunkSize, kBlockSize>;
};

} // namespace ck_tile
