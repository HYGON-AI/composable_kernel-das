// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

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

    // Target device limits: wave64, 256 VGPRs per thread, 64KB LDS per workgroup.
    static constexpr index_t kTargetMaxVgprPerThread = 256;
    static constexpr index_t kTargetMaxSmemBytes     = 64 * 1024;

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

    
    CK_TILE_HOST_DEVICE static constexpr index_t GetEstimatedVgprPerThread()
    {
        using AccDataType = remove_cvref_t<typename GemmPipeline::CDataType>;

        constexpr index_t acc_bytes = TilePartitioner::kM * TilePartitioner::kN *
                                      static_cast<index_t>(sizeof(AccDataType));
        constexpr index_t a_bytes = TilePartitioner::kM * TilePartitioner::kK *
                                    static_cast<index_t>(sizeof(ADataType)) /
                                    numeric_traits<ADataType>::PackedSize;
        constexpr index_t b_bytes = TilePartitioner::kN * TilePartitioner::kK *
                                    static_cast<index_t>(sizeof(BDataType)) /
                                    numeric_traits<BDataType>::PackedSize;

        return integer_divide_ceil(acc_bytes + a_bytes + b_bytes, KernelBlockSize * 4);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Kargs& kargs)
    {
        if constexpr(GetSmemSize() > kTargetMaxSmemBytes)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("LDS usage ",
                              GetSmemSize(),
                              " bytes exceeds the device limit of ",
                              kTargetMaxSmemBytes,
                              " bytes!");
            }
            return false;
        }

        if constexpr(GetEstimatedVgprPerThread() > kTargetMaxVgprPerThread)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Estimated VGPR usage ",
                              GetEstimatedVgprPerThread(),
                              " per thread exceeds the device limit of ",
                              kTargetMaxVgprPerThread,
                              "!");
            }
            return false;
        }

        // Vector-load divisibility is only required on the paths this kernel actually vectorizes
        // (A RowMajor along K, B ColumnMajor along K, C RowMajor along N). Each batch starts at
        // p + batch * batch_stride, so the batch strides must keep that alignment as well.
        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % TilePartitioner::kK != 0 && GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support K that is not a multiple of KPerBlock without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::VectorSizeA != 0 ||
               kargs.stride_A % GemmPipeline::VectorSizeA != 0 ||
               kargs.batch_stride_A % GemmPipeline::VectorSizeA != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K, stride_A or batch_stride_A is not a multiple of vector load "
                                  "size for A tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::kM != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::kN != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.K % TilePartitioner::kK != 0 && GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support K that is not a multiple of KPerBlock without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::VectorSizeB != 0 ||
               kargs.stride_B % GemmPipeline::VectorSizeB != 0 ||
               kargs.batch_stride_B % GemmPipeline::VectorSizeB != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K, stride_B or batch_stride_B is not a multiple of vector load "
                                  "size for B tensor!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::kN != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % GemmPipeline::VectorSizeC != 0 ||
               kargs.stride_C % GemmPipeline::VectorSizeC != 0 ||
               kargs.batch_stride_C % GemmPipeline::VectorSizeC != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N, stride_C or batch_stride_C is not a multiple of vector store "
                                  "size for C tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::kM != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
        }

        return true;
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
