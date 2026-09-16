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

template <typename DataType>
struct SlaAttnBwdPreprocessKargs
{
    const DataType* os;
    const DataType* dos;
    float* deltas;
    int total_tokens;
    int head_dim;
};

template <typename DataType, bool UseD128>
struct SlaAttnBwdPreprocessPipeline
{
    using Kargs = SlaAttnBwdPreprocessKargs<DataType>;
    static constexpr bool kUseD128 = UseD128;

    CK_TILE_DEVICE void operator()(const Kargs& arg) const
    {
        if constexpr(UseD128)
        {
            constexpr int kLanesPerToken  = 4;
            constexpr int kVectorsPerLane = 4;
            const int tid                 = ck_tile::get_thread_local_1d_id();
            const int lane_in_token       = tid % kLanesPerToken;
            const int token_in_block      = tid / kLanesPerToken;
            const int tokens_per_block    = 256 / kLanesPerToken;
            const int token_idx =
                ck_tile::get_block_1d_id() * tokens_per_block + token_in_block;
            if(token_idx >= arg.total_tokens)
            {
                return;
            }

            const float4* os_vec = reinterpret_cast<const float4*>(
                arg.os + static_cast<int64_t>(token_idx) * 128);
            const float4* dos_vec = reinterpret_cast<const float4*>(
                arg.dos + static_cast<int64_t>(token_idx) * 128);
            float sum_delta = 0.0f;
#pragma unroll
            for(int i = 0; i < kVectorsPerLane; ++i)
            {
                const int vec_idx    = lane_in_token + i * kLanesPerToken;
                const float4 os_chunk = os_vec[vec_idx];
                const float4 dos_chunk = dos_vec[vec_idx];
                const DataType* os_sub = reinterpret_cast<const DataType*>(&os_chunk);
                const DataType* dos_sub = reinterpret_cast<const DataType*>(&dos_chunk);
#pragma unroll
                for(int s = 0; s < 8; ++s)
                {
                    sum_delta += ck_tile::type_convert<float>(os_sub[s]) *
                                 ck_tile::type_convert<float>(dos_sub[s]);
                }
            }
            sum_delta += __shfl_down(sum_delta, 2, kLanesPerToken);
            sum_delta += __shfl_down(sum_delta, 1, kLanesPerToken);
            if(lane_in_token == 0)
            {
                arg.deltas[token_idx] = sum_delta;
            }
        }
        else
        {
            const int token_idx = ck_tile::get_block_1d_id() * 256 +
                                  ck_tile::get_thread_local_1d_id();
            if(token_idx >= arg.total_tokens)
            {
                return;
            }

            const float4* os_vec = reinterpret_cast<const float4*>(
                arg.os + static_cast<int64_t>(token_idx) * arg.head_dim);
            const float4* dos_vec = reinterpret_cast<const float4*>(
                arg.dos + static_cast<int64_t>(token_idx) * arg.head_dim);
            float sum_delta = 0.0f;
#pragma unroll
            for(int i = 0; i < 16; ++i)
            {
                const float4 os_chunk  = os_vec[i];
                const float4 dos_chunk = dos_vec[i];
                const DataType* os_sub = reinterpret_cast<const DataType*>(&os_chunk);
                const DataType* dos_sub = reinterpret_cast<const DataType*>(&dos_chunk);
#pragma unroll
                for(int s = 0; s < 8; ++s)
                {
                    sum_delta += ck_tile::type_convert<float>(os_sub[s]) *
                                 ck_tile::type_convert<float>(dos_sub[s]);
                }
            }
            arg.deltas[token_idx] = sum_delta;
        }
    }
};

} // namespace sla
} // namespace example
} // namespace ck_tile
