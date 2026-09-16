// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

template <typename QKDataType_,
          typename VDataType_,
          ck_tile::index_t KeyTile_>
struct GdnFusedRecurrentProblem
{
    using QKDataType = QKDataType_;
    using VDataType = VDataType_;
    using AccDataType = float;

    static constexpr ck_tile::index_t kWaveSize = 64;
    static constexpr ck_tile::index_t kBlockSize = 64;
    static constexpr ck_tile::index_t kValueTile = 8;
    static constexpr ck_tile::index_t kKeyTile = KeyTile_;
    static constexpr ck_tile::index_t kKeyValuesPerLane =
        kKeyTile / kWaveSize;
};
