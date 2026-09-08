// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

struct BatchedTransposeHostArgs
{
    const void* p_input;
    void* p_output;
    index_t batch;
    index_t height;
    index_t width;
    index_t dim_stride;
    index_t dim_block_h;
    index_t dim_block_w;
};

template <typename Pipeline_>
struct BatchedTransposeKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = remove_cvref_t<typename Pipeline::Problem>;
    using Type     = typename Problem::DataType;

    struct Kargs
    {
        const void* p_input;
        void* p_output;
        index_t batch;
        index_t height;
        index_t width;
        index_t dim_stride;
    };

    using Hargs = BatchedTransposeHostArgs;

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& args)
    {
        return dim3(integer_divide_ceil(args.height, args.dim_block_h),
                    integer_divide_ceil(args.width, args.dim_block_w),
                    args.batch);
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& args)
    {
        return Kargs{args.p_input,
                     args.p_output,
                     args.batch,
                     args.height,
                     args.width,
                     args.dim_stride};
    }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::kBlockSize; }

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        constexpr index_t m_per_block = Problem::kMPerBlock;
        constexpr index_t n_per_block = Problem::kNPerBlock;
        constexpr index_t vector_input = Problem::VectorSizeInput;
        constexpr index_t vector_output = Problem::VectorSizeOutput;

        const index_t m_offset =
            amd_wave_read_first_lane(static_cast<index_t>(blockIdx.x) * m_per_block);
        const index_t n_offset =
            amd_wave_read_first_lane(static_cast<index_t>(blockIdx.y) * n_per_block);
        const index_t batch_offset =
            amd_wave_read_first_lane(static_cast<index_t>(blockIdx.z) * args.dim_stride);

        const auto input_view = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                static_cast<const Type*>(args.p_input) + batch_offset,
                make_tuple(args.height, args.width),
                make_tuple(args.width, 1),
                number<vector_input>{},
                number<1>{}),
            make_tuple(number<m_per_block>{}, number<n_per_block>{}),
            sequence<Problem::kPadM, Problem::kPadN>{});

        const auto output_view = pad_tensor_view(
            make_naive_tensor_view<address_space_enum::global>(
                static_cast<Type*>(args.p_output) + batch_offset,
                make_tuple(args.width, args.height),
                make_tuple(args.height, 1),
                number<vector_output>{},
                number<1>{}),
            make_tuple(number<n_per_block>{}, number<m_per_block>{}),
            sequence<Problem::kPadN, Problem::kPadM>{});

        auto input_window = make_tile_window(input_view,
                                             make_tuple(number<m_per_block>{},
                                                        number<n_per_block>{}),
                                             {m_offset, n_offset});
        auto output_window = make_tile_window(output_view,
                                              make_tuple(number<n_per_block>{},
                                                         number<m_per_block>{}),
                                              {n_offset, m_offset});
        Pipeline{}(input_window, output_window);
    }
};

} // namespace ck_tile
