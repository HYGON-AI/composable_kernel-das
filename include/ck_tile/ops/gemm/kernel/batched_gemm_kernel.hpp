// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

struct BatchedGemmHostArgs
{
    const void* p_a;
    const void* p_b;
    void* p_c;
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
    index_t batch_stride_A;
    index_t batch_stride_B;
    index_t batch_stride_C;
    index_t batch_count;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct BatchedGemmKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout          = remove_cvref_t<typename GemmPipeline::CLayout>;
    using ADataType        = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType        = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType        = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    struct Kargs
    {
        const void* p_a;
        const void* p_b;
        void* p_c;
        index_t M;
        index_t N;
        index_t K;
        index_t stride_A;
        index_t stride_B;
        index_t stride_C;
        index_t batch_stride_A;
        index_t batch_stride_B;
        index_t batch_stride_C;
    };

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t batch_count)
    {
        return TilePartitioner::GridSize(M, N, batch_count);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr Kargs MakeKargs(const BatchedGemmHostArgs& args)
    {
        return Kargs{args.p_a,
                     args.p_b,
                     args.p_c,
                     args.M,
                     args.N,
                     args.K,
                     args.stride_A,
                     args.stride_B,
                     args.stride_C,
                     args.batch_stride_A,
                     args.batch_stride_B,
                     args.batch_stride_C};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto [i_m, i_n] = TilePartitioner{}();
        const index_t batch   = __builtin_amdgcn_readfirstlane(blockIdx.z);
        const ADataType* a_start = static_cast<const ADataType*>(kargs.p_a) +
                                   batch * kargs.batch_stride_A;
        const BDataType* b_start = static_cast<const BDataType*>(kargs.p_b) +
                                   batch * kargs.batch_stride_B;
        CDataType* c_start =
            static_cast<CDataType*>(kargs.p_c) + batch * kargs.batch_stride_C;

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
        auto a_block_window = make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
            {i_m, 0});

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
        auto b_block_window = make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
            {i_n, 0});

        __shared__ char smem_ptr[GetSmemSize()];
        const index_t num_loop = TilePartitioner::GetLoopNum(kargs.K);
        auto c_block_tile =
            GemmPipeline{}(a_block_window, b_block_window, num_loop, smem_ptr);

        const auto c_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<GemmPipeline::VectorSizeC>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        const auto c_pad_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(
                    c_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                    sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(
                    c_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                    sequence<GemmPipeline::kPadM, false>{});
            }
        }();
        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            {i_m, i_n});
        EpiloguePipeline{}(c_block_window, c_block_tile);
    }
};

} // namespace ck_tile
