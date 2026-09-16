// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile {

constexpr int kCpDim       = 128;
constexpr int kCpValueTile = 64;
constexpr int kCpBlockSize = 256;

struct PopulateCpKargs
{
    const float* local_hm;
    float* ag_hm;
    int value_heads;
    int world_size;
};

struct MergeCpKargs
{
    const float* ag_hm;
    float* state;
    int value_heads;
    int rank;
};

struct PopulateCpKernel
{
    using Kargs = PopulateCpKargs;

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        const size_t summary_elements =
            static_cast<size_t>(args.value_heads) * kCpDim * (kCpDim + kCpDim);
        const size_t index =
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total =
            summary_elements * static_cast<size_t>(args.world_size);
        if(index >= total)
            return;

        const int rank = static_cast<int>(index / summary_elements);
        const size_t local_index = index - rank * summary_elements;
        const int column = static_cast<int>(local_index % (kCpDim + kCpDim));

        // Keep rank 0 identical to the actual preprocess output. Other ranks
        // remain valid affine [H, M] summaries but are deterministically
        // different and non-zero, so the multi-rank merge is fully exercised.
        const float rank_delta = static_cast<float>(rank) * (1.0f / 128.0f);
        const float scale =
            column < kCpDim ? 1.0f + rank_delta : 1.0f - 0.5f * rank_delta;
        args.ag_hm[index] = args.local_hm[local_index] * scale;
    }
};

struct MergeCpKernel
{
    using Kargs = MergeCpKargs;

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        const int value_tile = static_cast<int>(blockIdx.x);
        const int head       = static_cast<int>(blockIdx.y);
        const int tid        = static_cast<int>(threadIdx.x);

        __shared__ float current[kCpDim * kCpValueTile];
        float next_values[kCpDim * kCpValueTile / kCpBlockSize];
        const int rank_stride =
            args.value_heads * kCpDim * (kCpDim + kCpDim);
        const int head_stride = kCpDim * (kCpDim + kCpDim);

        for(int elem = tid; elem < kCpDim * kCpValueTile; elem += kCpBlockSize)
            current[elem] = 0.0f;
        __syncthreads();

        for(int current_rank = 0; current_rank < args.rank; ++current_rank)
        {
            const float* summary =
                args.ag_hm + current_rank * rank_stride + head * head_stride;
            int slot = 0;
            for(int elem = tid; elem < kCpDim * kCpValueTile;
                elem += kCpBlockSize, ++slot)
            {
                const int row = elem / kCpValueTile;
                const int col = elem - row * kCpValueTile;
                const int value_col = value_tile * kCpValueTile + col;
                float next =
                    summary[row * (kCpDim + kCpDim) + value_col];
                for(int inner = 0; inner < kCpDim; ++inner)
                {
                    next +=
                        summary[row * (kCpDim + kCpDim) + kCpDim + inner] *
                        current[inner * kCpValueTile + col];
                }
                next_values[slot] = next;
            }
            __syncthreads();
            slot = 0;
            for(int elem = tid; elem < kCpDim * kCpValueTile;
                elem += kCpBlockSize, ++slot)
                current[elem] = next_values[slot];
            __syncthreads();
        }

        float* state_head = args.state + head * kCpDim * kCpDim;
        for(int elem = tid; elem < kCpDim * kCpValueTile; elem += kCpBlockSize)
        {
            const int row = elem / kCpValueTile;
            const int col = elem - row * kCpValueTile;
            const int value_col = value_tile * kCpValueTile + col;
            state_head[row * kCpDim + value_col] = current[elem];
        }
    }
};

} // namespace ck_tile
