// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

template <index_t VTile, index_t BlockSize>
struct ChunkDeltaHCpSummaryPolicy
{
    static constexpr index_t kChunkSize = 64;
    static constexpr index_t kHeadDim = 128;
    static constexpr index_t kValueDim = 128;
    static constexpr index_t kVTile = VTile;
    static constexpr index_t kBlockSize = BlockSize;
    static constexpr index_t kBlockPerCu = 2;
    static constexpr bool kSpecializeCommonFlags = true;
};

template <index_t Rows,
          index_t Cols,
          index_t Vector = 8,
          index_t Pad = 4,
          index_t NumWarps = 4>
CK_TILE_HOST_DEVICE constexpr auto MakeGdnAsyncLdsStoreDescriptor()
{
    constexpr index_t kWarpSize    = 64;
    constexpr index_t kLanesPerRow = Cols / Vector;
    constexpr index_t kRowGroups   = kWarpSize / kLanesPerRow;
    constexpr index_t kNumIssues   = Rows / (kRowGroups * NumWarps);

    static_assert(Cols % Vector == 0);
    static_assert(Rows % (kRowGroups * NumWarps) == 0);

    constexpr auto desc = make_naive_tensor_descriptor(
        make_tuple(number<kNumIssues>{},
                   number<kRowGroups>{},
                   number<NumWarps>{},
                   number<kLanesPerRow>{},
                   number<Vector>{}),
        make_tuple(number<NumWarps * (kWarpSize * Vector + Pad)>{},
                   number<Rows>{},
                   number<kWarpSize * Vector + Pad>{},
                   number<Vector>{},
                   number<1>{}),
        number<Vector>{},
        number<1>{});
    return transform_tensor_descriptor(
        desc,
        make_tuple(
            make_pass_through_transform(number<kNumIssues>{}),
            make_pass_through_transform(number<NumWarps>{}),
            make_merge_transform(make_tuple(number<kRowGroups>{},
                                            number<kLanesPerRow>{},
                                            number<Vector>{}))),
        make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
}

template <index_t Rows,
          index_t Cols,
          index_t Vector = 8,
          index_t NumWarps = 4>
CK_TILE_HOST_DEVICE constexpr auto MakeGdnCpAsyncDramDistribution()
{
    constexpr index_t kWarpSize    = 64;
    constexpr index_t kLanesPerRow = Cols / Vector;
    constexpr index_t kRowGroups   = kWarpSize / kLanesPerRow;
    constexpr index_t kNumIssues   = Rows / (kRowGroups * NumWarps);

    static_assert(Cols % Vector == 0);
    static_assert(Rows % (kRowGroups * NumWarps) == 0);

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>,
            tuple<sequence<kNumIssues, kRowGroups, NumWarps>,
                  sequence<kLanesPerRow, Vector>>,
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<2>, sequence<1, 0>>,
            sequence<1, 2>,
            sequence<0, 1>>{});
}

template <typename DataType, index_t VTile>
struct GdnCpPrefixMmacPolicy
{
    static_assert(VTile == 16 || VTile == 32 || VTile == 64);

    using WarpGemm = std::conditional_t<
        VTile == 64,
        GdnWarpGemm16x64x64<DataType>,
        std::conditional_t<VTile == 32,
                           GdnWarpGemm16x32x64<DataType>,
                           GdnWarpGemm16x16x64<DataType>>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemm{}, number<8>{}, number<1>{});
    }
};

template <typename DataType, index_t VTile, index_t BlockSize>
using GdnCpPrefixMmacProblem = BlockGemmProblem<
    DataType,
    DataType,
    float,
    BlockSize,
    TileGemmShape<sequence<128, VTile, 128>,
                  sequence<8, 1, 1>,
                  sequence<16, VTile, 64>>>;

template <typename DataType, index_t VTile, index_t BlockSize>
using GdnCpPrefixMmacBlockGemm = MmacBlockGemmASmemBSmemCRegV1<
    GdnCpPrefixMmacProblem<DataType, VTile, BlockSize>,
    GdnCpPrefixMmacPolicy<DataType, VTile>>;

} // namespace ck_tile
