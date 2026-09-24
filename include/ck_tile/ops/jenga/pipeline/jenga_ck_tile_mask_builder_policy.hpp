// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/jenga/pipeline/jenga_ck_tile_mask_builder_problem.hpp"
#include "ck_tile/core.hpp"
#include <algorithm>

template <typename Problem_>
struct JengaMaskBuilderPolicy
{
    using Problem = Problem_;
    using DataType = typename Problem::DataType;

    static constexpr int kBlockSize = Problem::kBlockSize;
    static constexpr int kBlockM = Problem::kBlockM;
    static constexpr int kBlockN = Problem::kBlockN;
    static constexpr int kHeadDim = Problem::kHeadDim;
    static constexpr int kScoreTile = Problem::kScoreTile;
    static constexpr int kLaunchMinBlocks = 2;

    CK_TILE_HOST_DEVICE static constexpr auto MakeQPoolInputDistribution()
    {
        using TileEncodingPattern =
            ck_tile::tile_distribution_encoding_pattern_2d<kBlockSize,
                                                           kBlockM,
                                                           kHeadDim,
                                                           8,
                                                           ck_tile::tile_distribution_pattern::thread_raked,
                                                           1>;
        return TileEncodingPattern::make_2d_static_tile_distribution();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKPoolInputDistribution()
    {
        using TileEncodingPattern =
            ck_tile::tile_distribution_encoding_pattern_2d<kBlockSize,
                                                           kBlockN,
                                                           kHeadDim,
                                                           8,
                                                           ck_tile::tile_distribution_pattern::thread_raked,
                                                           1>;
        return TileEncodingPattern::make_2d_static_tile_distribution();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeScoreInputDistribution()
    {
        using TileEncodingPattern =
            ck_tile::tile_distribution_encoding_pattern_2d<kBlockSize,
                                                           kScoreTile,
                                                           kHeadDim,
                                                           8,
                                                           ck_tile::tile_distribution_pattern::thread_raked,
                                                           1>;
        return TileEncodingPattern::make_2d_static_tile_distribution();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeScoreRowDistribution()
    {
        using TileEncodingPattern =
            ck_tile::tile_distribution_encoding_pattern_2d<kBlockSize,
                                                           4,
                                                           kScoreTile,
                                                           1,
                                                           ck_tile::tile_distribution_pattern::thread_raked,
                                                           1>;
        return TileEncodingPattern::make_2d_static_tile_distribution();
    }

    CK_TILE_HOST static int GetCandidateK(int top_k, int text_start_block, double prob_threshold)
    {
        if (top_k * 3 >= text_start_block) {
            return text_start_block;
        }
        return std::min<int>(
            text_start_block,
            std::max<int>(top_k * 4,
                          static_cast<int>(text_start_block * (1.0 - prob_threshold) * 3) +
                              top_k));
    }
};
