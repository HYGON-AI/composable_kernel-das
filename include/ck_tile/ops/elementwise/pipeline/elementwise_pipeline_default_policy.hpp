// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct ElementWiseDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<S::kRepeatM,
                                                      S::kWarpPerBlockM,
                                                      S::kThreadPerWarpM,
                                                      S::kVectorM>>,
                                       tuple<sequence<1>, sequence<1>>,
                                       tuple<sequence<1>, sequence<2>>,
                                       sequence<1, 1>,
                                       sequence<0, 3>>{});
    }
};

} // namespace ck_tile
