// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core.hpp"
#include <cstdint>

struct JengaMaskBuilderArgument
{
    const uint16_t* query;
    const uint16_t* key;
    const bool* neighbor_mask;
    bool* out;
    int B;
    int H;
    int N_Q;
    int N_K;
    int D;
    int num_query_blocks;
    int num_blocks;
    int text_start_block;
    int top_k;
    float prob_threshold;
    int text_blocks;
    int first_frame_blocks;
};

template <typename DataType_, int BlockM_, int BlockN_, int HeadDim_>
struct JengaMaskBuilderProblem
{
    using DataType = DataType_;
    using AccDataType = float;

    static constexpr ck_tile::index_t kBlockM = BlockM_;
    static constexpr ck_tile::index_t kBlockN = BlockN_;
    static constexpr ck_tile::index_t kHeadDim = HeadDim_;
    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kScoreTile = 64;
};
