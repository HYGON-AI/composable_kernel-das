// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/sla.hpp"
#include "sla_config.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>




namespace sla {

template <typename DataType, bool UseD128>
inline void launch_sla_attn_bwd_preprocess_kernel(
    const DataType* os,
    const DataType* dos,
    float* deltas,
    int total_tokens,
    int head_dim,
    hipStream_t stream)
{
    using Pipeline = ck_tile::example::sla::SlaAttnBwdPreprocessPipeline<
        DataType, UseD128>;
    using Kernel = ck_tile::example::sla::SlaAttnBwdPreprocessKernel<Pipeline>;
    const typename Kernel::Kargs args{os, dos, deltas, total_tokens, head_dim};
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<Kernel::kBlockSize, 1>(
            Kernel{}, Kernel::GridSize(args), Kernel::BlockSize(), 0, args));
}

template <typename DataType>
inline void launch_sla_attn_bwd_preprocess(const DataType* os,
                                           const DataType* dos,
                                           float* deltas,
                                           int total_tokens,
                                           int head_dim = ck_tile::example::sla::kHeadDim,
                                           hipStream_t stream = nullptr)
{
    if(head_dim == ck_tile::example::sla::kHeadDim)
    {
        launch_sla_attn_bwd_preprocess_kernel<DataType, true>(
            os, dos, deltas, total_tokens, head_dim, stream);
    }
    else
    {
        launch_sla_attn_bwd_preprocess_kernel<DataType, false>(
            os, dos, deltas, total_tokens, head_dim, stream);
    }
}

} // namespace sla

namespace ck_tile {
namespace example {
namespace sla {

using ::sla::launch_sla_attn_bwd_preprocess;

} // namespace sla
} // namespace example
} // namespace ck_tile




namespace ck_tile {
namespace example {
namespace sla {

template <int KvStageCount>
inline void launch_sla_attn_bwd_dq_kv_partition(
    const SlaAttnBwdDqKvPartitionKargs& arg,
    hipStream_t stream)
{
    if(arg.topk <= ck_tile::example::sla::kDqLutStageCapacity)
    {
        using Pipeline = SlaAttnBwdDqKvPartitionFastPipeline<KvStageCount>;
        using Kernel   = SlaAttnBwdDqKvPartitionKernel<Pipeline>;
        ck_tile::launch_kernel(
            ck_tile::stream_config{stream},
            ck_tile::make_kernel<Kernel::kBlockSize, 1>(
                Kernel{}, Kernel::GridSize(arg), Kernel::BlockSize(), 0, arg));
    }
    else
    {
        using Pipeline = SlaAttnBwdDqKvPartitionGenericPipeline<KvStageCount>;
        using Kernel   = SlaAttnBwdDqKvPartitionKernel<Pipeline>;
        ck_tile::launch_kernel(
            ck_tile::stream_config{stream},
            ck_tile::make_kernel<Kernel::kBlockSize, 1>(
                Kernel{}, Kernel::GridSize(arg), Kernel::BlockSize(), 0, arg));
    }
}

template <typename DataType>
inline void launch_sla_attn_bwd_dq_reduce(const DataType* workspace,
                                          DataType* output,
                                          int stage_count,
                                          int64_t stage_stride,
                                          int64_t element_count,
                                          hipStream_t stream)
{
    constexpr int kValuesPerVector = 8;
    using Pipeline = SlaAttnBwdDqReducePipeline<DataType>;
    using Kernel   = SlaAttnBwdDqReduceKernel<Pipeline>;
    const typename Kernel::Kargs args{
        reinterpret_cast<const float4*>(workspace),
        reinterpret_cast<float4*>(output),
        stage_count,
        stage_stride / kValuesPerVector,
        element_count / kValuesPerVector};
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<Kernel::kBlockSize, 1>(
            Kernel{}, Kernel::GridSize(args), Kernel::BlockSize(), 0, args));
}

} // namespace sla
} // namespace example
} // namespace ck_tile

// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.


namespace ck_tile {
namespace example {
namespace sla {

template <typename DataType>
inline void launch_sla_attn_bwd_dkdv(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const void* do_ptr,
    const float* lse_ptr,
    const float* delta_ptr,
    const int8_t* sparse_map_ptr,
    void* dk_ptr,
    void* dv_ptr,
    int32_t* rlut_ptr,
    int32_t* rsize_ptr,
    int32_t* kv_perm_ptr,
    int b,
    int h,
    int seqlen,
    int d,
    int block_m,
    int /*block_n*/,
    float qk_scale,
    hipStream_t stream)
{
    using Problem = SlaAttnBwdDkdvAttentionProblem<DataType>;
    const int64_t public_kv_blocks = seqlen / ck_tile::example::sla::kBlockN;
    const int64_t internal_kv_blocks = seqlen / Problem::BlockN;

    const int32_t NB = static_cast<int32_t>(public_kv_blocks);
    const int32_t Q = static_cast<int32_t>(seqlen / block_m);
    const int32_t q_slices = block_m / ck_tile::example::sla::kQSliceBlockM;
    const int32_t rlut_stride_n = Q * q_slices;

    using BucketKernel = SlaBwdDkdvBucketLutKernel;
    const BucketKernel::Kargs bucket_args{sparse_map_ptr,
                                           Q,
                                           NB,
                                           static_cast<int32_t>(block_m),
                                           rlut_ptr,
                                           rsize_ptr,
                                           kv_perm_ptr};
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<BucketKernel::kBlockSize, 1>(
            BucketKernel{},
            BucketKernel::GridSize(b * h),
            BucketKernel::BlockSize(),
            BucketKernel::LdsSize(),
            bucket_args));

    using GreedyKernel = SlaBwdDkdvClusterGreedyKernel;
    const size_t greedy_smem_bytes = GreedyKernel::LdsSize(Q);
    const GreedyKernel::Kargs greedy_args{rlut_ptr,
                                          rsize_ptr,
                                          Q,
                                          NB,
                                          static_cast<int32_t>(block_m),
                                          kv_perm_ptr};
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<GreedyKernel::kBlockSize, 1>(
            GreedyKernel{},
            GreedyKernel::GridSize(b * h, NB),
            GreedyKernel::BlockSize(),
            greedy_smem_bytes,
            greedy_args));

    // Prepare Arguments for SlaAttnBwdDkdvKernel
    SlaAttnBwdDkdvKernelArgument args{};
    args.q_ptr              = q_ptr;
    args.k_ptr              = k_ptr;
    args.v_ptr              = v_ptr;
    args.do_ptr             = do_ptr;
    args.delta_ptr          = delta_ptr;
    args.lse_ptr            = lse_ptr;
    args.lse_text_ptr       = lse_ptr;
    args.rlut_ptr           = rlut_ptr;
    args.rlut_size_ptr      = rsize_ptr;
    args.kv_perm_ptr        = kv_perm_ptr;
    args.seqlens_ptr        = nullptr;
    args.dk_ptr             = dk_ptr;
    args.dv_ptr             = dv_ptr;

    args.H                  = h;
    args.N_Q                = seqlen;
    args.D                  = d;
    args.text_block_start    = static_cast<int>(public_kv_blocks);
    args.sm_scale            = qk_scale;

    args.stride_qz           = h * seqlen * d;
    args.stride_qh           = seqlen * d;
    args.stride_qm           = d;
    args.stride_qd           = 1;

    args.stride_kz           = h * seqlen * d;
    args.stride_kh           = seqlen * d;
    args.stride_kn           = d;
    args.stride_kd           = 1;

    args.stride_vz           = h * seqlen * d;
    args.stride_vh           = seqlen * d;
    args.stride_vn           = d;
    args.stride_vd           = 1;

    args.stride_doz          = h * seqlen * d;
    args.stride_doh          = seqlen * d;
    args.stride_dom          = d;
    args.stride_dod          = 1;

    args.stride_delta_z      = seqlen;
    args.stride_delta_m      = 1;

    args.stride_dkz          = h * seqlen * d;
    args.stride_dkh          = seqlen * d;
    args.stride_dkn          = d;
    args.stride_dkd          = 1;

    args.stride_dvz          = h * seqlen * d;
    args.stride_dvh          = seqlen * d;
    args.stride_dvn          = d;
    args.stride_dvd          = 1;

    args.stride_lz           = seqlen;
    args.stride_lm           = 1;

    args.stride_rlutz        = NB * rlut_stride_n;
    args.stride_rlutn        = rlut_stride_n;
    args.stride_rlutk        = 1;

    args.stride_rlut_size_z  = NB;
    args.stride_rlut_size_n  = 1;

    using Policy   = SlaAttnBwdDkdvDefaultPolicy<Problem>;
    using Pipeline = SlaAttnBwdDkdvPipeline<Problem, Policy>;
    using Kernel   = SlaAttnBwdDkdvKernel<Pipeline>;

    const auto grid = Kernel::GridSize(
        static_cast<int>(internal_kv_blocks), static_cast<int>(b * h));
    constexpr auto block = Kernel::BlockSize();
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<Kernel::ThreadsPerBlock, 1>(
            Kernel{}, grid, block, 0, args));
}

} // namespace sla
} // namespace example
} // namespace ck_tile
