// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core.hpp"

struct GdnCumsumProblemBf16
{
    using DataType = ck_tile::bf16_t;
    static constexpr int kChunkSize = 64;
    static constexpr int kSBlock = 1;
    static constexpr int kBlockSize = 64;
};

struct GdnCumsumVectorProblemBf16
{
    using DataType = ck_tile::bf16_t;
    static constexpr int kChunkSize = 64;
    static constexpr int kSBlock = 16;
    static constexpr int kBlockSize = 1024;
};


struct GdnCumsumProblemF32
{
    using DataType = float;
    static constexpr int kChunkSize = 64;
    static constexpr int kSBlock = 1;
    static constexpr int kBlockSize = 64;
};

struct GdnCumsumVectorProblemF32
{
    using DataType = float;
    static constexpr int kChunkSize = 64;
    static constexpr int kSBlock = 16;
    static constexpr int kBlockSize = 1024;
};
