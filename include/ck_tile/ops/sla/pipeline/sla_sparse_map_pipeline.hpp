// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_sparse_map_policy.hpp"

namespace sla {

template <typename Policy>
struct SlaSparseMapMeanKPipeline
{
    using Problem = typename Policy::Problem;
    using DataType = typename Problem::DataType;

    CK_TILE_DEVICE void operator()(const DataType* k,
                                   DataType* k_mean,
                                   ck_tile::index_t Lk,
                                   ck_tile::index_t bh,
                                   ck_tile::index_t vec_idx,
                                   ck_tile::index_t tid,
                                   float* scratch) const
    {
        float sum[8] = {0.0f};
        const DataType* k_bh = k + bh * Lk * Policy::kHeadDim;

        for(ck_tile::index_t l = tid; l < Lk; l += Policy::kBlockSize)
        {
            const float4 chunk = reinterpret_cast<const float4*>(
                k_bh + l * Policy::kHeadDim)[vec_idx];
            const DataType* values = reinterpret_cast<const DataType*>(&chunk);
#pragma unroll
            for(ck_tile::index_t i = 0; i < 8; ++i)
                sum[i] += ck_tile::type_convert<float>(values[i]);
        }

#pragma unroll
        for(ck_tile::index_t i = 0; i < 8; ++i)
            scratch[i * Policy::kBlockSize + tid] = sum[i];
        __syncthreads();

        for(ck_tile::index_t stride = Policy::kBlockSize / 2; stride > 0; stride >>= 1)
        {
            if(tid < stride)
            {
#pragma unroll
                for(ck_tile::index_t i = 0; i < 8; ++i)
                    scratch[i * Policy::kBlockSize + tid] +=
                        scratch[i * Policy::kBlockSize + tid + stride];
            }
            __syncthreads();
        }

        if(tid == 0)
        {
            float4 output;
            DataType* values = reinterpret_cast<DataType*>(&output);
            const float inv_l = 1.0f / static_cast<float>(Lk);
#pragma unroll
            for(ck_tile::index_t i = 0; i < 8; ++i)
                values[i] = ck_tile::type_convert<DataType>(scratch[i * Policy::kBlockSize] * inv_l);
            reinterpret_cast<float4*>(k_mean + bh * Policy::kHeadDim)[vec_idx] = output;
        }
    }
};

template <typename Policy>
struct SlaSparseMapPoolQPipeline
{
    using Problem = typename Policy::Problem;
    using DataType = typename Problem::DataType;

    template <typename QView, typename PooledQView>
    CK_TILE_DEVICE void operator()(
        const QView& q_view,
        const PooledQView& pooled_q_view,
        ck_tile::index_t l_start,
        ck_tile::index_t l_end,
        ck_tile::index_t tid) const
    {
        const DataType* q_ptr = q_view.get_buffer_view().p_data_;
        DataType* pooled_q_ptr = pooled_q_view.get_buffer_view().p_data_;

        const ck_tile::index_t len = l_end - l_start;
        if (len <= 0) return;

        constexpr ck_tile::index_t kCols = 16;
        constexpr ck_tile::index_t kLanes = Policy::kBlockSize / kCols;
        const ck_tile::index_t col = tid & 15;
        const ck_tile::index_t r_lane = tid >> 4;

        float sum_val[8] = {0.0f};

        for (ck_tile::index_t l = l_start + r_lane; l < l_end; l += kLanes) {
            const float4* row_vec = reinterpret_cast<const float4*>(
                q_ptr + l * Policy::kHeadDim);
            float4 chunk = row_vec[col];
            const DataType* sub_in = reinterpret_cast<const DataType*>(&chunk);

            #pragma unroll
            for (ck_tile::index_t s = 0; s < 8; ++s) {
                sum_val[s] += ck_tile::type_convert<float>(sub_in[s]);
            }
        }

        __shared__ float s_sum[Policy::kBlockSize][8];
        #pragma unroll
        for (ck_tile::index_t s = 0; s < 8; ++s) {
            s_sum[tid][s] = sum_val[s];
        }
        __syncthreads();

        if (r_lane == 0) {
            #pragma unroll
            for (ck_tile::index_t s = 0; s < 8; ++s) {
                float total = s_sum[col][s];
                #pragma unroll
                for (ck_tile::index_t lane = 1; lane < kLanes; ++lane) {
                    total += s_sum[lane * kCols + col][s];
                }
                sum_val[s] = total;
            }

            const float inv_len = 1.0f / static_cast<float>(len);
            float4 out_chunk;
            DataType* sub_out = reinterpret_cast<DataType*>(&out_chunk);

            #pragma unroll
            for (ck_tile::index_t s = 0; s < 8; ++s) {
                sub_out[s] = ck_tile::type_convert<DataType>(sum_val[s] * inv_len);
            }
            reinterpret_cast<float4*>(pooled_q_ptr)[col] = out_chunk;
        }
    }
};

template <typename Policy>
struct SlaSparseMapPoolKPipeline
{
    using Problem = typename Policy::Problem;
    using DataType = typename Problem::DataType;

    template <typename KView, typename PooledKView>
    CK_TILE_DEVICE void operator()(
        const KView& k_view,
        const PooledKView& pooled_k_view,
        ck_tile::index_t l_start,
        ck_tile::index_t l_end,
        ck_tile::index_t tid) const
    {
        const DataType* k_ptr = k_view.get_buffer_view().p_data_;
        DataType* pooled_k_ptr = pooled_k_view.get_buffer_view().p_data_;

        const ck_tile::index_t len = l_end - l_start;
        if (len <= 0) return;

        constexpr ck_tile::index_t kCols = 16;
        constexpr ck_tile::index_t kLanes = Policy::kBlockSize / kCols;
        const ck_tile::index_t col = tid & 15;
        const ck_tile::index_t r_lane = tid >> 4;

        float sum_val[8] = {0.0f};

        for (ck_tile::index_t l = l_start + r_lane; l < l_end; l += kLanes) {
            const float4* row_vec = reinterpret_cast<const float4*>(
                k_ptr + l * Policy::kHeadDim);
            float4 chunk = row_vec[col];
            const DataType* sub_in = reinterpret_cast<const DataType*>(&chunk);

            #pragma unroll
            for (ck_tile::index_t s = 0; s < 8; ++s) {
                sum_val[s] += ck_tile::type_convert<float>(sub_in[s]);
            }
        }

        __shared__ float s_sum[Policy::kBlockSize][8];
        #pragma unroll
        for (ck_tile::index_t s = 0; s < 8; ++s) {
            s_sum[tid][s] = sum_val[s];
        }
        __syncthreads();

        if (r_lane == 0) {
            #pragma unroll
            for (ck_tile::index_t s = 0; s < 8; ++s) {
                float total = s_sum[col][s];
                #pragma unroll
                for (ck_tile::index_t lane = 1; lane < kLanes; ++lane) {
                    total += s_sum[lane * kCols + col][s];
                }
                sum_val[s] = total;
            }

            const float inv_len = 1.0f / static_cast<float>(len);
            float4 out_chunk;
            DataType* sub_out = reinterpret_cast<DataType*>(&out_chunk);

            #pragma unroll
            for (ck_tile::index_t s = 0; s < 8; ++s) {
                sub_out[s] = ck_tile::type_convert<DataType>(sum_val[s] * inv_len);
            }
            reinterpret_cast<float4*>(pooled_k_ptr)[col] = out_chunk;
        }
    }
};

template <typename Policy>
struct SlaSparseMapScorePipeline
{
    using Problem = typename Policy::Problem;
    using DataType = typename Problem::DataType;

    CK_TILE_DEVICE void operator()(
        const DataType* pooled_q,
        const DataType* pooled_k,
        float* pooled_score,
        ck_tile::index_t Lq_blocks,
        ck_tile::index_t Lk_blocks,
        ck_tile::index_t bh,
        ck_tile::index_t q_blk,
        ck_tile::index_t k_blk,
        ck_tile::index_t tid) const
    {
        if (tid >= 16) return;

        const DataType* q_vec_ptr = pooled_q + (bh * Lq_blocks + q_blk) * Policy::kHeadDim;
        const DataType* k_vec_ptr = pooled_k + (bh * Lk_blocks + k_blk) * Policy::kHeadDim;

        const float4 q_chunk = reinterpret_cast<const float4*>(q_vec_ptr)[tid];
        const float4 k_chunk = reinterpret_cast<const float4*>(k_vec_ptr)[tid];

        const DataType* q_sub = reinterpret_cast<const DataType*>(&q_chunk);
        const DataType* k_sub = reinterpret_cast<const DataType*>(&k_chunk);

        float dot = 0.0f;
        #pragma unroll
        for (ck_tile::index_t s = 0; s < 8; ++s) {
            dot += ck_tile::type_convert<float>(q_sub[s]) * ck_tile::type_convert<float>(k_sub[s]);
        }

        #pragma unroll
        for (int mask = 8; mask > 0; mask /= 2) {
            dot += __shfl_xor(dot, mask, 64);
        }

        if (tid == 0) {
            pooled_score[(bh * Lq_blocks + q_blk) * Lk_blocks + k_blk] = dot;
        }
    }
};

template <typename Policy>
struct SlaSparseMapTopkPipeline
{
    CK_TILE_DEVICE void operator()(
        const float* pooled_score,
        int64_t* lut,
        int8_t* sparse_map,
        ck_tile::index_t Lq_blocks,
        ck_tile::index_t Lk_blocks,
        ck_tile::index_t topk,
        ck_tile::index_t bh,
        ck_tile::index_t q_blk,
        ck_tile::index_t tid) const
    {
        const int64_t score_row_offset = (bh * Lq_blocks + q_blk) * Lk_blocks;
        const float* score_row = pooled_score + score_row_offset;
        int64_t* lut_row = lut + (bh * Lq_blocks + q_blk) * topk;
        int8_t* map_row = sparse_map + score_row_offset;

        for (ck_tile::index_t k = tid; k < Lk_blocks; k += blockDim.x) {
            map_row[k] = 0;
        }
        __syncthreads();

        if (tid == 0) {
            for (ck_tile::index_t t = 0; t < topk; ++t) {
                float max_val = -1e30f;
                int best_k = 0;
                for (ck_tile::index_t k = 0; k < Lk_blocks; ++k) {
                    bool already_selected = false;
                    for (ck_tile::index_t prev = 0; prev < t; ++prev) {
                        if (lut_row[prev] == k) {
                            already_selected = true;
                            break;
                        }
                    }
                    if (!already_selected && score_row[k] > max_val) {
                        max_val = score_row[k];
                        best_k = k;
                    }
                }
                lut_row[t] = static_cast<int64_t>(best_k);
                map_row[best_k] = 1;
            }
        }
    }
};

template <typename Policy>
struct SlaSparseMapScoreRowPipeline
{
    using Problem = typename Policy::Problem;
    using DataType = typename Problem::DataType;

    CK_TILE_DEVICE void operator()(const DataType* pooled_q,
                                   const DataType* pooled_k,
                                   DataType* pooled_score,
                                   ck_tile::index_t Lq_blocks,
                                   ck_tile::index_t Lk_blocks,
                                   ck_tile::index_t bh,
                                   ck_tile::index_t q_blk,
                                   ck_tile::index_t tid) const
    {
        constexpr ck_tile::index_t kDotLanes = 16;
        constexpr ck_tile::index_t kDotsPerBlock = Policy::kBlockSize / kDotLanes;
        const ck_tile::index_t dot_id = tid / kDotLanes;
        const ck_tile::index_t dot_lane = tid % kDotLanes;

        const DataType* q_ptr = pooled_q +
            (bh * Lq_blocks + q_blk) * Policy::kHeadDim;
        const float4 q_chunk = reinterpret_cast<const float4*>(q_ptr)[dot_lane];
        const DataType* q_values = reinterpret_cast<const DataType*>(&q_chunk);
        float q_reg[8];
#pragma unroll
        for(ck_tile::index_t i = 0; i < 8; ++i)
            q_reg[i] = ck_tile::type_convert<float>(q_values[i]);

        for(ck_tile::index_t k_blk = dot_id; k_blk < Lk_blocks; k_blk += kDotsPerBlock)
        {
            const DataType* k_ptr = pooled_k +
                (bh * Lk_blocks + k_blk) * Policy::kHeadDim;
            const float4 k_chunk = reinterpret_cast<const float4*>(k_ptr)[dot_lane];
            const DataType* k_values = reinterpret_cast<const DataType*>(&k_chunk);
            float dot = 0.0f;
#pragma unroll
            for(ck_tile::index_t i = 0; i < 8; ++i)
                dot += q_reg[i] * ck_tile::type_convert<float>(k_values[i]);
#pragma unroll
            for(ck_tile::index_t offset = kDotLanes / 2; offset > 0; offset >>= 1)
                dot += __shfl_down(dot, offset, kDotLanes);
            if(dot_lane == 0)
            {
                pooled_score[(bh * Lq_blocks + q_blk) * Lk_blocks + k_blk] =
                    ck_tile::type_convert<DataType>(dot);
            }
        }
    }
};

template <typename Policy>
struct SlaSparseMapScatterPipeline
{
    CK_TILE_DEVICE void operator()(const int32_t* lut_i32,
                                   int64_t* lut,
                                   int8_t* sparse_map,
                                   ck_tile::index_t Lq_blocks,
                                   ck_tile::index_t Lk_blocks,
                                   ck_tile::index_t topk,
                                   ck_tile::index_t bh,
                                   ck_tile::index_t q_blk,
                                   ck_tile::index_t tid) const
    {
        int8_t* map_row = sparse_map + (bh * Lq_blocks + q_blk) * Lk_blocks;
        const int32_t* lut_i32_row = lut_i32 + (bh * Lq_blocks + q_blk) * topk;
        int64_t* lut_row = lut + (bh * Lq_blocks + q_blk) * topk;
        for(ck_tile::index_t k_blk = tid; k_blk < Lk_blocks;
            k_blk += Policy::kBlockSize)
            map_row[k_blk] = 0;
        __syncthreads();
        for(ck_tile::index_t i = tid; i < topk; i += Policy::kBlockSize)
        {
            const int32_t index = lut_i32_row[i];
            lut_row[i] = static_cast<int64_t>(index);
            map_row[index] = 1;
        }
    }
};

template <typename Policy, ck_tile::index_t PaddedCandidates>
struct SlaSparseMapScoreRadixPipeline
{
    using Problem = typename Policy::Problem;
    using DataType = typename Problem::DataType;

    CK_TILE_DEVICE void operator()(const DataType* pooled_q,
                                   const DataType* pooled_k,
                                   int64_t* lut,
                                   int8_t* sparse_map,
                                   ck_tile::index_t Lq_blocks,
                                   ck_tile::index_t Lk_blocks,
                                   ck_tile::index_t topk,
                                   ck_tile::index_t bh,
                                   ck_tile::index_t q_blk,
                                   ck_tile::index_t tid,
                                   uint16_t* keys,
                                   int32_t* histogram,
                                   int32_t* metadata) const
    {
        constexpr ck_tile::index_t kDotLanes = 16;
        constexpr ck_tile::index_t kDotsPerBlock = Policy::kBlockSize / kDotLanes;
        const ck_tile::index_t dot_id = tid / kDotLanes;
        const ck_tile::index_t dot_lane = tid % kDotLanes;

        for(ck_tile::index_t i = tid; i < PaddedCandidates; i += Policy::kBlockSize)
            keys[i] = 0;

        const DataType* q_ptr = pooled_q +
            (bh * Lq_blocks + q_blk) * Policy::kHeadDim;
        const float4 q_chunk = reinterpret_cast<const float4*>(q_ptr)[dot_lane];
        const DataType* q_values = reinterpret_cast<const DataType*>(&q_chunk);
        float q_reg[8];
#pragma unroll
        for(ck_tile::index_t i = 0; i < 8; ++i)
            q_reg[i] = ck_tile::type_convert<float>(q_values[i]);

        ck_tile::index_t k_blk = dot_id;
        for(; k_blk + kDotsPerBlock < Lk_blocks; k_blk += kDotsPerBlock * 2)
        {
            const DataType* k_ptr1 = pooled_k +
                (bh * Lk_blocks + k_blk) * Policy::kHeadDim;
            const float4 k_chunk1 = reinterpret_cast<const float4*>(k_ptr1)[dot_lane];
            const DataType* k_values1 = reinterpret_cast<const DataType*>(&k_chunk1);

            const DataType* k_ptr2 = pooled_k +
                (bh * Lk_blocks + (k_blk + kDotsPerBlock)) * Policy::kHeadDim;
            const float4 k_chunk2 = reinterpret_cast<const float4*>(k_ptr2)[dot_lane];
            const DataType* k_values2 = reinterpret_cast<const DataType*>(&k_chunk2);

            float dot1 = 0.0f;
            float dot2 = 0.0f;
#pragma unroll
            for(ck_tile::index_t i = 0; i < 8; ++i)
            {
                dot1 += q_reg[i] * ck_tile::type_convert<float>(k_values1[i]);
                dot2 += q_reg[i] * ck_tile::type_convert<float>(k_values2[i]);
            }

#pragma unroll
            for(ck_tile::index_t offset = kDotLanes / 2; offset > 0; offset >>= 1)
            {
                dot1 += __shfl_down(dot1, offset, kDotLanes);
                dot2 += __shfl_down(dot2, offset, kDotLanes);
            }

            if(dot_lane == 0)
            {
                const DataType rounded1 = ck_tile::type_convert<DataType>(dot1);
                const uint16_t raw1 = ck_tile::bit_cast<uint16_t>(rounded1);
                keys[k_blk] = raw1 ^ ((raw1 & 0x8000u) ? 0xffffu : 0x8000u);

                const DataType rounded2 = ck_tile::type_convert<DataType>(dot2);
                const uint16_t raw2 = ck_tile::bit_cast<uint16_t>(rounded2);
                keys[k_blk + kDotsPerBlock] = raw2 ^ ((raw2 & 0x8000u) ? 0xffffu : 0x8000u);
            }
        }

        for(; k_blk < Lk_blocks; k_blk += kDotsPerBlock)
        {
            const DataType* k_ptr = pooled_k +
                (bh * Lk_blocks + k_blk) * Policy::kHeadDim;
            const float4 k_chunk = reinterpret_cast<const float4*>(k_ptr)[dot_lane];
            const DataType* k_values = reinterpret_cast<const DataType*>(&k_chunk);
            float dot = 0.0f;
#pragma unroll
            for(ck_tile::index_t i = 0; i < 8; ++i)
                dot += q_reg[i] * ck_tile::type_convert<float>(k_values[i]);
#pragma unroll
            for(ck_tile::index_t offset = kDotLanes / 2; offset > 0; offset >>= 1)
                dot += __shfl_down(dot, offset, kDotLanes);
            if(dot_lane == 0)
            {
                const DataType rounded = ck_tile::type_convert<DataType>(dot);
                const uint16_t raw = ck_tile::bit_cast<uint16_t>(rounded);
                keys[k_blk] = raw ^ ((raw & 0x8000u) ? 0xffffu : 0x8000u);
            }
        }
        __syncthreads();

        for(ck_tile::index_t i = tid; i < 256; i += Policy::kBlockSize)
            histogram[i] = 0;
        __syncthreads();
        for(ck_tile::index_t i = tid; i < Lk_blocks; i += Policy::kBlockSize)
            atomicAdd(histogram + (keys[i] >> 8), 1);
        __syncthreads();

        if(tid == 0)
        {
            int remaining = topk;
            for(int bucket = 255; bucket >= 0; --bucket)
            {
                if(remaining > histogram[bucket])
                    remaining -= histogram[bucket];
                else
                {
                    metadata[0] = bucket;
                    metadata[1] = remaining;
                    break;
                }
            }
        }
        __syncthreads();

        for(ck_tile::index_t i = tid; i < 256; i += Policy::kBlockSize)
            histogram[i] = 0;
        __syncthreads();
        for(ck_tile::index_t i = tid; i < Lk_blocks; i += Policy::kBlockSize)
            if((keys[i] >> 8) == metadata[0])
                atomicAdd(histogram + (keys[i] & 0xff), 1);
        __syncthreads();

        if(tid == 0)
        {
            int remaining = metadata[1];
            for(int bucket = 255; bucket >= 0; --bucket)
            {
                if(remaining > histogram[bucket])
                    remaining -= histogram[bucket];
                else
                {
                    metadata[2] = (metadata[0] << 8) | bucket;
                    metadata[3] = remaining;
                    metadata[4] = topk - remaining;
                    metadata[5] = 0;
                    metadata[6] = 0;
                    break;
                }
            }
        }
        __syncthreads();

        int8_t* map_row = sparse_map + (bh * Lq_blocks + q_blk) * Lk_blocks;
        int64_t* lut_row = lut + (bh * Lq_blocks + q_blk) * topk;
        for(ck_tile::index_t i = tid; i < Lk_blocks; i += Policy::kBlockSize)
            map_row[i] = 0;
        __syncthreads();

        for(ck_tile::index_t i = tid; i < Lk_blocks; i += Policy::kBlockSize)
        {
            if(keys[i] > metadata[2])
            {
                const int position = atomicAdd(metadata + 5, 1);
                lut_row[position] = static_cast<int64_t>(i);
                map_row[i] = 1;
            }
        }
        __syncthreads();

        for(ck_tile::index_t i = tid; i < Lk_blocks; i += Policy::kBlockSize)
        {
            if(keys[i] == metadata[2])
            {
                const int rank = atomicAdd(metadata + 6, 1);
                if(rank < metadata[3])
                {
                    const int position = metadata[4] + rank;
                    lut_row[position] = static_cast<int64_t>(i);
                    map_row[i] = 1;
                }
            }
        }
    }
};

} // namespace sla
