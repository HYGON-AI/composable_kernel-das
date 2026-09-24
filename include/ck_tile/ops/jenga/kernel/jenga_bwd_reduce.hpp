// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host.hpp"

struct JengaBwdReducePolicy
{
    static constexpr int kBlockSize       = 256;
    static constexpr int kVectorSize      = 8;
    static constexpr int kLaunchMinBlocks = 1;
};

template <typename DataType, typename Policy = JengaBwdReducePolicy>
struct JengaBwdReducePipeline
{
    CK_TILE_DEVICE void operator()(const DataType* workspace,
                                   DataType* output,
                                   int stage_count,
                                   int64_t stage_stride,
                                   int64_t num_elements) const
    {
        const int64_t begin =
            (static_cast<int64_t>(blockIdx.x) * Policy::kBlockSize + threadIdx.x) *
            Policy::kVectorSize;
        float sums[Policy::kVectorSize] = {};
        for(int stage = 0; stage < stage_count; ++stage)
        {
            const DataType* src = workspace + static_cast<int64_t>(stage) * stage_stride + begin;
#pragma unroll
            for(int i = 0; i < Policy::kVectorSize; ++i)
                if(begin + i < num_elements)
                    sums[i] += ck_tile::type_convert<float>(src[i]);
        }
#pragma unroll
        for(int i = 0; i < Policy::kVectorSize; ++i)
            if(begin + i < num_elements)
                output[begin + i] = ck_tile::type_convert<DataType>(sums[i]);
    }
};

template <typename DataType, typename Policy = JengaBwdReducePolicy>
struct JengaBwdReduceKernel
{
    using Pipeline = JengaBwdReducePipeline<DataType, Policy>;

    CK_TILE_HOST static dim3 GridSize(int64_t num_elements)
    {
        constexpr int elements_per_block = Policy::kBlockSize * Policy::kVectorSize;
        return dim3(static_cast<unsigned int>(
            (num_elements + elements_per_block - 1) / elements_per_block));
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(Policy::kBlockSize); }

    CK_TILE_DEVICE void operator()(const DataType* workspace,
                                   DataType* output,
                                   int stage_count,
                                   int64_t stage_stride,
                                   int64_t num_elements) const
    {
        Pipeline{}(workspace, output, stage_count, stage_stride, num_elements);
    }
};

template <typename DataType, typename Policy = JengaBwdReducePolicy>
float jenga_bwd_reduce_workspace(const DataType* workspace,
                                 DataType* output,
                                 int stage_count,
                                 int64_t stage_stride,
                                 int64_t num_elements,
                                 const ck_tile::stream_config& stream_config)
{
    using Kernel = JengaBwdReduceKernel<DataType, Policy>;
    constexpr auto block = Kernel::BlockSize();
    return ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(Kernel{},
                                                                Kernel::GridSize(num_elements),
                                                                block,
                                                                0,
                                                                workspace,
                                                                output,
                                                                stage_count,
                                                                stage_stride,
                                                                num_elements));
}
