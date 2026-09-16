// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host.hpp"

struct JengaOnehotToLutPolicy
{
    static constexpr int kBlockSize       = 256;
    static constexpr int kLaunchMinBlocks = 1;
};

template <typename Policy>
struct JengaOnehotToLutPipeline
{
    CK_TILE_DEVICE void operator()(const bool* onehot,
                                   int32_t* active_indices,
                                   int32_t* active_counts,
                                   int rows,
                                   int num_blocks,
                                   int active_capacity) const
    {
        const int row = blockIdx.x * Policy::kBlockSize + threadIdx.x;
        if(row >= rows)
            return;

        const bool* src = onehot + static_cast<int64_t>(row) * num_blocks;
        int32_t* dst = active_indices + static_cast<int64_t>(row) * active_capacity;
        int count = 0;
        for(int kb = 0; kb < num_blocks; ++kb)
        {
            if(src[kb])
            {
                if(count < active_capacity)
                    dst[count] = kb;
                ++count;
            }
        }
        active_counts[row] = ck_tile::min(count, active_capacity);
    }
};

template <typename Policy = JengaOnehotToLutPolicy>
struct JengaOnehotToLutKernel
{
    using Pipeline = JengaOnehotToLutPipeline<Policy>;

    CK_TILE_HOST static dim3 GridSize(int rows)
    {
        return dim3((rows + Policy::kBlockSize - 1) / Policy::kBlockSize);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(Policy::kBlockSize); }

    CK_TILE_DEVICE void operator()(const bool* onehot,
                                   int32_t* active_indices,
                                   int32_t* active_counts,
                                   int rows,
                                   int num_blocks,
                                   int active_capacity) const
    {
        Pipeline{}(onehot, active_indices, active_counts, rows, num_blocks, active_capacity);
    }
};

template <typename Policy = JengaOnehotToLutPolicy>
void jenga_onehot_to_lut(const bool* onehot,
                         int32_t* active_indices,
                         int32_t* active_counts,
                         int rows,
                         int num_blocks,
                         int active_capacity,
                         hipStream_t stream = nullptr)
{
    using Kernel = JengaOnehotToLutKernel<Policy>;
    constexpr auto block = Kernel::BlockSize();
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(Kernel{},
                                                                Kernel::GridSize(rows),
                                                                block,
                                                                0,
                                                                onehot,
                                                                active_indices,
                                                                active_counts,
                                                                rows,
                                                                num_blocks,
                                                                active_capacity));
}

template <typename Policy>
struct JengaOnehotToReverseLutPipeline
{
    CK_TILE_DEVICE void operator()(const bool* onehot,
                                   int32_t* reverse_indices,
                                   int32_t* reverse_counts,
                                   int batch_heads,
                                   int num_q_blocks,
                                   int num_k_blocks,
                                   int reverse_capacity) const
    {
        const int row = blockIdx.x * Policy::kBlockSize + threadIdx.x;
        const int rows = batch_heads * num_k_blocks;
        if(row >= rows)
            return;

        const int bh = row / num_k_blocks;
        const int kb = row % num_k_blocks;
        int32_t* dst = reverse_indices + static_cast<int64_t>(row) * reverse_capacity;
        int count = 0;
        for(int qb = 0; qb < num_q_blocks; ++qb)
        {
            const bool selected =
                onehot[(static_cast<int64_t>(bh) * num_q_blocks + qb) * num_k_blocks + kb];
            if(selected)
            {
                if(count < reverse_capacity)
                    dst[count] = qb;
                ++count;
            }
        }
        reverse_counts[row] = ck_tile::min(count, reverse_capacity);
    }
};

template <typename Policy = JengaOnehotToLutPolicy>
struct JengaOnehotToReverseLutKernel
{
    using Pipeline = JengaOnehotToReverseLutPipeline<Policy>;

    CK_TILE_HOST static dim3 GridSize(int rows)
    {
        return dim3((rows + Policy::kBlockSize - 1) / Policy::kBlockSize);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(Policy::kBlockSize); }

    CK_TILE_DEVICE void operator()(const bool* onehot,
                                   int32_t* reverse_indices,
                                   int32_t* reverse_counts,
                                   int batch_heads,
                                   int num_q_blocks,
                                   int num_k_blocks,
                                   int reverse_capacity) const
    {
        Pipeline{}(onehot,
                   reverse_indices,
                   reverse_counts,
                   batch_heads,
                   num_q_blocks,
                   num_k_blocks,
                   reverse_capacity);
    }
};

template <typename Policy = JengaOnehotToLutPolicy>
void jenga_onehot_to_reverse_lut(const bool* onehot,
                                 int32_t* reverse_indices,
                                 int32_t* reverse_counts,
                                 int batch_heads,
                                 int num_q_blocks,
                                 int num_k_blocks,
                                 int reverse_capacity,
                                 hipStream_t stream = nullptr)
{
    using Kernel = JengaOnehotToReverseLutKernel<Policy>;
    constexpr auto block = Kernel::BlockSize();
    const int rows = batch_heads * num_k_blocks;
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(Kernel{},
                                                                Kernel::GridSize(rows),
                                                                block,
                                                                0,
                                                                onehot,
                                                                reverse_indices,
                                                                reverse_counts,
                                                                batch_heads,
                                                                num_q_blocks,
                                                                num_k_blocks,
                                                                reverse_capacity));
}
