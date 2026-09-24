// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "jenga_bwd.hpp"
#include "jenga_config.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>

#ifdef JENGA_BWD_BUILD_DQ
#include "ck_tile/ops/jenga/kernel/jenga_bwd_dq_kernel.hpp"
#include "ck_tile/ops/jenga/kernel/jenga_bwd_reduce.hpp"

namespace jenga_dq = ck_tile::jenga_dq;

template <typename DataType>
float launch_jenga_bwd_dq(jenga_dq::jenga_bwd_dq_traits t,
                          jenga_dq::jenga_bwd_dq_args a,
                          ck_tile::stream_config s,
                          float* reduce_ms)
{
    using problem = ck_tile::JengaBwdDqProblem<DataType,
                                               DataType,
                                               DataType,
                                               DataType,
                                               float,
                                               float,
                                               DataType,
                                               ck_tile::example::jenga::kBlockM,
                                               ck_tile::example::jenga::kBlockN,
                                               ck_tile::example::jenga::kHeadDim,
                                               ck_tile::example::jenga::kDqMaxNnzCap,
                                               ck_tile::example::jenga::kThreadsPerBlock,
                                               ck_tile::example::jenga::kBlocksPerCu>;
    using kernel = ck_tile::JengaBwdDqKernel<problem>;
    const auto num_q_blocks = ck_tile::integer_divide_ceil(a.seqlen_q, a.block_m);
    auto kargs = kernel::MakeKargs(a.q,
                                   a.k,
                                   a.v,
                                   a.dout,
                                   a.deltas,
                                   a.lse,
                                   a.dq,
                                   a.dq_workspace,
                                   a.kv_stage_count,
                                   a.lut,
                                   a.lut_size,
                                   a.seqlens,
                                   a.sm_scale,
                                   a.text_amp,
                                   a.text_block_start,
                                   a.nhead,
                                   a.seqlen_q,
                                   a.seqlen_q,
                                   num_q_blocks,
                                   t.head_dim,
                                   a.max_nnz,
                                   a.stride_qz,
                                   a.stride_qm,
                                   a.stride_qk,
                                   a.stride_kz,
                                   a.stride_kn,
                                   a.stride_kk,
                                   a.stride_vz,
                                   a.stride_vn,
                                   a.stride_vk,
                                   a.stride_doz,
                                   a.stride_dom,
                                   a.stride_dok,
                                   a.stride_dqz,
                                   a.stride_dqm,
                                   a.stride_dqk,
                                   a.stride_dz,
                                   a.stride_dm,
                                   a.stride_lz,
                                   a.stride_lm,
                                   a.stride_lutz,
                                   a.stride_lutm,
                                   a.stride_lutk);
    dim3 grids = kernel::GridSize(a.batch, a.nhead, a.seqlen_q);
    grids.z = a.kv_stage_count;
    const dim3 blocks = kernel::BlockSize();
    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel: " << kernel::GetName() << " with args:"
                  << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                  << "}" << std::endl;
    }
    float elapsed = ck_tile::launch_kernel(
        s,
        ck_tile::make_kernel<kernel::kBlockSize, kernel::kBlockPerCu>(
            kernel{}, grids, blocks, 0, kargs));
    if(a.kv_stage_count > 1)
    {
        long long total_elements = static_cast<long long>(a.batch) * a.nhead * a.stride_dqz;
        const float ms = jenga_bwd_reduce_workspace(
            reinterpret_cast<const DataType*>(a.dq_workspace),
            reinterpret_cast<DataType*>(a.dq),
            a.kv_stage_count,
            total_elements,
            total_elements,
            s);
        if(reduce_ms != nullptr)
            *reduce_ms = ms;
        elapsed += ms;
    }
    else if(reduce_ms != nullptr)
        *reduce_ms = 0.0f;
    return elapsed;
}

float jenga_bwd_dq(jenga_dq::jenga_bwd_dq_traits t,
                   jenga_dq::jenga_bwd_dq_args a,
                   ck_tile::stream_config s,
                   float* reduce_ms)
{
    if(t.block_m == ck_tile::example::jenga::kBlockM &&
       t.block_n == ck_tile::example::jenga::kBlockN &&
       t.head_dim == ck_tile::example::jenga::kHeadDim && t.max_nnz == a.max_nnz &&
       a.max_nnz <= ck_tile::example::jenga::kDqMaxNnzCap)
    {
        if(t.data_type == "fp16")
        {
            return launch_jenga_bwd_dq<ck_tile::fp16_t>(t, a, s, reduce_ms);
        }
        if(t.data_type == "bf16")
        {
            return launch_jenga_bwd_dq<ck_tile::bf16_t>(t, a, s, reduce_ms);
        }
    }

    std::cerr << "unsupported jenga_bwd_dq config: dtype=" << t.data_type
              << " block_m=" << t.block_m << " block_n=" << t.block_n
              << " head_dim=" << t.head_dim << " max_nnz=" << t.max_nnz
              << " max_nnz_cap=" << ck_tile::example::jenga::kDqMaxNnzCap << std::endl;
    return -1.0f;
}
#endif // JENGA_BWD_BUILD_DQ

#ifdef JENGA_BWD_BUILD_DKDV
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dkdv_config.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dkdv_policy.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dkdv_pipeline.hpp"
#include "ck_tile/ops/jenga/kernel/jenga_bwd_dkdv_kernel.hpp"

template <typename DataType>
float launch_jenga_bwd_dkdv_pipeline(const ck_tile::example::jenga::jenga_bwd_dkdv_args& args,
                                     const ck_tile::stream_config& s)
{
    using Problem = ck_tile::example::jenga::JengaBwdDkdvProblem<
        DataType, DataType, DataType, DataType, DataType, DataType, float, float,
        ck_tile::example::jenga::kBlockM,
        ck_tile::example::jenga::kBlockN,
        ck_tile::example::jenga::kHeadDim,
        ck_tile::example::jenga::kThreadsPerBlock
    >;

    using Policy = ck_tile::example::jenga::JengaBwdDkdvDefaultPolicy<Problem>;
    using Pipeline = ck_tile::example::jenga::JengaBwdDkdvPipeline<Problem, Policy>;
    using Kernel = ck_tile::example::jenga::JengaBwdDkdvKernel<Pipeline>;

    if(args.M0 != Kernel::BlockM || args.N0 != Kernel::BlockN ||
       args.D != Kernel::HeadDim || args.N_Q_BLOCKS <= 0 ||
       args.N_KV_BLOCKS <= 0)
    {
        throw std::runtime_error(
            "jenga_bwd_dkdv_pipeline requires the supported head dimension "
            "and non-empty grids");
    }

    const dim3 grids      = Kernel::GridSize(args.N_KV_BLOCKS, args.B * args.H, args.split_kv);
    constexpr dim3 blocks = Kernel::BlockSize();

    if(s.log_level_ > 0)
    {
        std::cout << "Launching fused jenga_bwd_dkdv_pipeline grid: {" << grids.x << ", " << grids.y
                  << ", " << grids.z << "}, block: {" << blocks.x << ", " << blocks.y << ", "
                  << blocks.z << "}" << std::endl;
    }

    return ck_tile::launch_kernel(
        s,
        ck_tile::make_kernel<Kernel::ThreadsPerBlock,
                             ck_tile::example::jenga::kDkdvLaunchMinBlocks>(
            Kernel{}, grids, blocks, 0, args));
}

float jenga_bwd_dkdv_pipeline_calc(
    const ck_tile::example::jenga::jenga_bwd_dkdv_traits& traits,
    const ck_tile::example::jenga::jenga_bwd_dkdv_args& args,
    const ck_tile::stream_config& s)
{
    if(traits.block_m != ck_tile::example::jenga::kBlockM ||
       traits.block_n != ck_tile::example::jenga::kBlockN ||
       traits.head_dim != ck_tile::example::jenga::kHeadDim ||
       traits.max_nnz != args.max_nnz_r)
    {
        throw std::invalid_argument(
            "jenga dK/dV traits do not match the selected kernel or runtime arguments");
    }
    if(traits.data_type == "fp16")
        return launch_jenga_bwd_dkdv_pipeline<ck_tile::fp16_t>(args, s);
    if(traits.data_type == "bf16")
        return launch_jenga_bwd_dkdv_pipeline<ck_tile::bf16_t>(args, s);
    throw std::invalid_argument("unsupported jenga dK/dV precision: " + traits.data_type);
}
#endif // JENGA_BWD_BUILD_DKDV
