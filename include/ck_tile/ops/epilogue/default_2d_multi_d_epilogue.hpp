// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename AccDataType_,
          typename D0DataType_,
          typename D1DataType_,
          typename ODataType_>
struct Default2DMultiDEpilogueProblem
{
    using AccDataType = remove_cvref_t<AccDataType_>;
    using D0DataType  = remove_cvref_t<D0DataType_>;
    using D1DataType  = remove_cvref_t<D1DataType_>;
    using ODataType   = remove_cvref_t<ODataType_>;
};

template <typename Problem_>
struct Default2DMultiDEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using D0DataType  = remove_cvref_t<typename Problem::D0DataType>;
    using D1DataType  = remove_cvref_t<typename Problem::D1DataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    template <typename ODramWindow,
              typename D0DramWindow,
              typename D1DramWindow,
              typename OAccTile>
    CK_TILE_DEVICE void operator()(ODramWindow& o_dram_window,
                                   D0DramWindow& d0_dram_window,
                                   D1DramWindow& d1_dram_window,
                                   const OAccTile& o_acc_tile) const
    {
        const auto d0_tile =
            load_tile(make_tile_window(d0_dram_window, o_acc_tile.get_tile_distribution()));
        const auto d1_tile =
            load_tile(make_tile_window(d1_dram_window, o_acc_tile.get_tile_distribution()));
        auto fused_tile =
            make_static_distributed_tensor<AccDataType>(o_acc_tile.get_tile_distribution());

        tile_elementwise_inout(
            [](auto& o, const auto& acc, const auto& d0, const auto& d1) {
                o = acc * type_convert<AccDataType>(d0) * type_convert<AccDataType>(d1);
            },
            fused_tile,
            o_acc_tile,
            d0_tile,
            d1_tile);
        store_tile(o_dram_window, cast_tile<ODataType>(fused_tile));
    }
};

} // namespace ck_tile
