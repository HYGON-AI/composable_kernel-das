// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

struct GemmMultiDHostArgs
{
    const void* p_a;
    const void* p_b;
    const void* p_d0;
    const void* p_d1;
    void* p_e;
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_D0;
    index_t stride_D1;
    index_t stride_E;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GemmMultiDKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout          = remove_cvref_t<typename GemmPipeline::CLayout>;
    using ADataType        = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType        = remove_cvref_t<typename GemmPipeline::BDataType>;
    using D0DataType       = remove_cvref_t<typename EpiloguePipeline::D0DataType>;
    using D1DataType       = remove_cvref_t<typename EpiloguePipeline::D1DataType>;
    using EDataType        = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    struct Kargs
    {
        const void* p_a;
        const void* p_b;
        const void* p_d0;
        const void* p_d1;
        void* p_e;
        index_t M;
        index_t N;
        index_t K;
        index_t stride_A;
        index_t stride_B;
        index_t stride_D0;
        index_t stride_D1;
        index_t stride_E;
    };

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N)
    {
        return TilePartitioner::GridSize(M, N, 1);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr Kargs MakeKargs(const GemmMultiDHostArgs& args)
    {
        return Kargs{args.p_a,
                     args.p_b,
                     args.p_d0,
                     args.p_d1,
                     args.p_e,
                     args.M,
                     args.N,
                     args.K,
                     args.stride_A,
                     args.stride_B,
                     args.stride_D0,
                     args.stride_D1,
                     args.stride_E};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto [i_m, i_n] = TilePartitioner{}();
        const auto* a_start    = static_cast<const ADataType*>(kargs.p_a);
        const auto* b_start    = static_cast<const BDataType*>(kargs.p_b);

        const auto a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::VectorSizeA>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(1, kargs.stride_A),
                    number<1>{},
                    number<1>{});
            }
        }();
        const auto b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(1, kargs.stride_B),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(kargs.stride_B, 1),
                    number<GemmPipeline::VectorSizeB>{},
                    number<1>{});
            }
        }();

        const auto a_pad_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(
                    a_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
                    sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(
                    a_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
                    sequence<GemmPipeline::kPadM, false>{});
            }
        }();
        const auto b_pad_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return pad_tensor_view(
                    b_tensor_view,
                    make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
                    sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(
                    b_tensor_view,
                    make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
                    sequence<GemmPipeline::kPadN, false>{});
            }
        }();
        auto a_block_window = make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
            {i_m, 0});
        auto b_block_window = make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
            {i_n, 0});

        __shared__ char smem_ptr[GetSmemSize()];
        const index_t num_loop = TilePartitioner::GetLoopNum(kargs.K);
        const auto c_block_tile =
            GemmPipeline{}(a_block_window, b_block_window, num_loop, smem_ptr);

        const auto make_m_n_view = [&](auto* p, index_t stride) {
            return make_naive_tensor_view<address_space_enum::global>(
                p,
                make_tuple(kargs.M, kargs.N),
                make_tuple(stride, 1),
                number<GemmPipeline::VectorSizeC>{},
                number<1>{});
        };
        const auto d0_pad_view = pad_tensor_view(
            make_m_n_view(static_cast<const D0DataType*>(kargs.p_d0), kargs.stride_D0),
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            sequence<GemmPipeline::kPadM, GemmPipeline::kPadN>{});
        const auto d1_pad_view = pad_tensor_view(
            make_m_n_view(static_cast<const D1DataType*>(kargs.p_d1), kargs.stride_D1),
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            sequence<GemmPipeline::kPadM, GemmPipeline::kPadN>{});
        const auto e_pad_view = pad_tensor_view(
            make_m_n_view(static_cast<EDataType*>(kargs.p_e), kargs.stride_E),
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            sequence<GemmPipeline::kPadM, GemmPipeline::kPadN>{});

        auto d0_block_window = make_tile_window(
            d0_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            {i_m, i_n});
        auto d1_block_window = make_tile_window(
            d1_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            {i_m, i_n});
        auto e_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            {i_m, i_n});
        EpiloguePipeline{}(e_block_window, d0_block_window, d1_block_window, c_block_tile);
    }
};

} // namespace ck_tile
