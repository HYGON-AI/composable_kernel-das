// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_>
struct ElementWiseKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using ComputeDataType      = remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType            = remove_cvref_t<typename Problem::YDataType>;
    using ElementWiseOperation = remove_cvref_t<typename Problem::ElementWiseOperation>;

    template <typename... XDataTypes, typename Dims>
    CK_TILE_DEVICE void operator()(Dims lens,
                                   Dims input_strides,
                                   Dims output_strides,
                                   const tuple<XDataTypes...>& input_tensors,
                                   YDataType* p_y) const
    {
        using S = typename Problem::BlockShape;

        const index_t iM           = get_block_id() * S::kBlockM;
        const auto merge_transform = make_merge_transform(lens);

        const auto x_tiles = generate_tuple(
            [&](auto i) {
                const auto tensor_view = make_naive_tensor_view<address_space_enum::global>(
                    input_tensors.get(i), lens, input_strides, number<S::kVectorM>{}, number<1>{});

                const auto transformed_tensor = pad_tensor_view(
                    transform_tensor_view(tensor_view,
                                          make_tuple(merge_transform),
                                          make_tuple(make_index_sequence<Dims::size()>{}),
                                          make_tuple(sequence<0>{})),
                    make_tuple(number<S::kBlockM>{}),
                    sequence<Problem::kPad>{});

                const auto x_window =
                    make_tile_window(transformed_tensor,
                                     make_tuple(number<S::kBlockM>{}),
                                     {iM},
                                     Policy::template MakeXBlockTileDistribution<Problem>());
                return load_tile(x_window);
            },
            number<sizeof...(XDataTypes)>{});

        const auto& x_tile0 = x_tiles.get(number<0>{});
        auto y_tile = make_static_distributed_tensor<YDataType>(x_tile0.get_tile_distribution());

        const auto spans = x_tile0.get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx) {
            const auto tile_idx = make_tuple(idx);
            apply(
                [&](auto&&... tiles) {
                    ElementWiseOperation{}(y_tile(tile_idx),
                                           type_convert<ComputeDataType>(tiles[tile_idx])...);
                },
                x_tiles);
        });

        const auto y_view = make_naive_tensor_view<address_space_enum::global>(
            p_y, lens, output_strides, number<S::kVectorM>{});
        const auto transformed_y = pad_tensor_view(
            transform_tensor_view(y_view,
                                  make_tuple(merge_transform),
                                  make_tuple(make_index_sequence<Dims::size()>{}),
                                  make_tuple(sequence<0>{})),
            make_tuple(number<S::kBlockM>{}),
            sequence<Problem::kPad>{});
        auto y_window = make_tile_window(transformed_y,
                                         make_tuple(number<S::kBlockM>{}),
                                         {iM},
                                         y_tile.get_tile_distribution());
        store_tile(y_window, cast_tile<YDataType>(y_tile));
    }

    template <typename... Ints>
    CK_TILE_HOST static bool IsSupportedArgument(const tuple<Ints...>& input_sizes)
    {
        int total_elements        = 1;
        constexpr auto kVectorM = Problem::BlockShape::kVectorM;
        apply([&](auto&&... values) { ((total_elements *= values), ...); }, input_sizes);
        return total_elements > 0 && total_elements % kVectorM == 0;
    }
};

} // namespace ck_tile
