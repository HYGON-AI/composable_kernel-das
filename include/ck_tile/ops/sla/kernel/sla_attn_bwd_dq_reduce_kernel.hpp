// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core.hpp"

#include <cstddef>
#include <cstdint>

namespace ck_tile {
namespace example {
namespace sla {

struct SlaAttnBwdDqReduceKargs
{
    const float4* workspace;
    float4* output;
    int stage_count;
    int64_t stage_stride_float4;
    int64_t vector_count;
};

template <typename DataType>
struct SlaAttnBwdDqReducePipeline
{
    using Kargs = SlaAttnBwdDqReduceKargs;

    CK_TILE_DEVICE void operator()(const Kargs& arg) const
    {
        const int64_t idx = static_cast<int64_t>(ck_tile::get_block_1d_id()) * 256 +
                            ck_tile::get_thread_local_1d_id();
        if(idx >= arg.vector_count)
        {
            return;
        }

        union Packed8
        {
            float4 packed;
            DataType values[8];
        };
        static_assert(sizeof(Packed8) == sizeof(float4));

        float sums[8] = {};
        if(arg.stage_count == 2)
        {
            Packed8 input0;
            Packed8 input1;
            input0.packed = arg.workspace[idx];
            input1.packed = arg.workspace[arg.stage_stride_float4 + idx];
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                sums[i] = ck_tile::type_convert<float>(input0.values[i]) +
                          ck_tile::type_convert<float>(input1.values[i]);
            }
        }
        else
        {
            for(int stage = 0; stage < arg.stage_count; ++stage)
            {
                Packed8 input;
                input.packed = arg.workspace[
                    static_cast<int64_t>(stage) * arg.stage_stride_float4 + idx];
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    sums[i] += ck_tile::type_convert<float>(input.values[i]);
                }
            }
        }
        Packed8 result;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            result.values[i] = ck_tile::type_convert<DataType>(sums[i]);
        }
        arg.output[idx] = result.packed;
    }
};

template <typename Pipeline>
struct SlaAttnBwdDqReduceKernel
{
    using Kargs = typename Pipeline::Kargs;
    static constexpr int kBlockSize = 256;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(
            (arg.vector_count + kBlockSize - 1) / kBlockSize));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }
    CK_TILE_DEVICE void operator()(const Kargs& arg) const { Pipeline{}(arg); }
};



} // namespace sla
} // namespace example
} // namespace ck_tile
