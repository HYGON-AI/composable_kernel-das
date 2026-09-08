// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_transpose/pipeline/batched_transpose_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BatchedTransposePolicy>
struct BatchedTransposePipeline
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    template <typename InputWindow, typename OutputWindow>
    CK_TILE_DEVICE void operator()(const InputWindow& input_window, OutputWindow& output_window)
    {
        auto input_window_distributed =
            make_tile_window(input_window, Policy::template MakeInputDistribution<Problem>());
        auto input_tile = load_tile(input_window_distributed);
        auto output_tile = make_static_distributed_tensor<typename Problem::DataType>(
            Policy::template MakeOutputDistribution<Problem>());
        transpose_tile2d(output_tile, input_tile);
        auto output_window_distributed =
            make_tile_window(output_window, Policy::template MakeOutputDistribution<Problem>());
        store_tile(output_window_distributed, output_tile);
    }
};

} // namespace ck_tile
