// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_l2norm_policy.hpp"

namespace ck_tile {

template <typename Policy>
struct GdnL2NormFwdPipeline
{
    using Problem = typename Policy::Problem;
    using XDataType = typename Problem::XDataType;
    using YDataType = typename Problem::YDataType;

    CK_TILE_DEVICE static float wave_reduce_sum(float value, index_t lane)
    {
        float remote = warp_shuffle_down(value, 32);
        if(lane < 32)
            value += remote;
        remote = warp_shuffle_down(value, 16);
        if(lane < 16)
            value += remote;
        remote = warp_shuffle_down(value, 8);
        if(lane < 8)
            value += remote;
        remote = warp_shuffle_down(value, 4);
        if(lane < 4)
            value += remote;
        remote = warp_shuffle_down(value, 2);
        if(lane < 2)
            value += remote;
        remote = warp_shuffle_down(value, 1);
        if(lane < 1)
            value += remote;
        return value;
    }

    CK_TILE_DEVICE void RunSmall(const XDataType* x,
                                 YDataType* y,
                                 float* rstd,
                                 long_index_t rows,
                                 index_t dim,
                                 float epsilon) const
    {
        const index_t tid = get_thread_local_1d_id();
        const index_t wave = tid / Policy::kWaveSize;
        const index_t lane = tid % Policy::kWaveSize;
        const long_index_t row =
            static_cast<long_index_t>(blockIdx.x) * Policy::kRowsPerBlock + wave;
        const bool valid_row = row < rows;
        const long_index_t row_offset = row * static_cast<long_index_t>(dim);

        thread_buffer<float, Policy::kSmallValuesPerLane> values{};
        float square_sum = 0.0f;
        static_for<0, Policy::kSmallValuesPerLane, 1>{}([&](auto i) {
            const index_t col = lane + i * Policy::kWaveSize;
            float value = 0.0f;
            if(valid_row && col < dim)
                value = type_convert<float>(x[row_offset + col]);
            values(i) = value;
            square_sum += value * value;
        });

        square_sum = wave_reduce_sum(square_sum, lane);
        float inv_norm = 0.0f;
        if(lane == 0)
        {
            inv_norm = 1.0f / ck_tile::sqrt(square_sum + epsilon);
            if(valid_row)
                rstd[row] = inv_norm;
        }
        inv_norm = warp_shuffle(inv_norm, 0);

        static_for<0, Policy::kSmallValuesPerLane, 1>{}([&](auto i) {
            const index_t col = lane + i * Policy::kWaveSize;
            if(valid_row && col < dim)
                y[row_offset + col] = type_convert<YDataType>(values(i) * inv_norm);
        });
    }

    CK_TILE_DEVICE void RunLarge(const XDataType* x,
                                 YDataType* y,
                                 float* rstd,
                                 long_index_t rows,
                                 index_t dim,
                                 float epsilon,
                                 float* shared) const
    {
        const index_t tid = get_thread_local_1d_id();
        const index_t wave = tid / Policy::kWaveSize;
        const index_t lane = tid % Policy::kWaveSize;
        const long_index_t row = static_cast<long_index_t>(blockIdx.x);
        if(row >= rows)
            return;

        const long_index_t row_offset = row * static_cast<long_index_t>(dim);
        float square_sum = 0.0f;
        for(index_t col = tid; col < dim; col += Policy::kBlockSize)
        {
            const float value = type_convert<float>(x[row_offset + col]);
            square_sum += value * value;
        }

        square_sum = wave_reduce_sum(square_sum, lane);
        if(lane == 0)
            shared[wave] = square_sum;
        __syncthreads();

        if(wave == 0)
        {
            float block_sum = lane < Policy::kNumWaves ? shared[lane] : 0.0f;
            block_sum = wave_reduce_sum(block_sum, lane);
            if(lane == 0)
            {
                const float inv_norm = 1.0f / ck_tile::sqrt(block_sum + epsilon);
                shared[0] = inv_norm;
                rstd[row] = inv_norm;
            }
        }
        __syncthreads();

        const float inv_norm = shared[0];
        for(index_t col = tid; col < dim; col += Policy::kBlockSize)
        {
            const float value = type_convert<float>(x[row_offset + col]);
            y[row_offset + col] = type_convert<YDataType>(value * inv_norm);
        }
    }
};

} // namespace ck_tile
