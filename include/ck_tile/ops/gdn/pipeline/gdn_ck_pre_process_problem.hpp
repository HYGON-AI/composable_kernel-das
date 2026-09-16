// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core.hpp"

template <typename DataType_ = ck_tile::bf16_t>
struct GdnPreProcessProblem
{
    using DataType = DataType_;
    using AccType  = float;

    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kChunkSize = 64;
    static constexpr ck_tile::index_t kHeadDim   = 128;
};

using GdnPreProcessProblemBf16 = GdnPreProcessProblem<ck_tile::bf16_t>;
using GdnPreProcessProblemFp16 = GdnPreProcessProblem<ck_tile::half_t>;
