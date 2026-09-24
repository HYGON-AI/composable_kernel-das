// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>
#include <ck_tile/core.hpp>
#include <cmath>
#include <cstdint>

namespace sla_reference {

inline constexpr int kReferenceBlockSize = 128;

template <typename DataType>
CK_TILE_DEVICE float to_float(uint16_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<DataType>(x));
}

template <typename DataType>
CK_TILE_DEVICE uint16_t from_float(float x)
{
    return ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(x));
}

template <typename DataType>
CK_TILE_DEVICE float quantize_to_data_type(float x)
{
    return ck_tile::type_convert<float>(ck_tile::type_convert<DataType>(x));
}

CK_TILE_DEVICE float block_reduce_128(float value, float* wave_sums)
{
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
#pragma unroll
    for(int offset = 32; offset > 0; offset >>= 1)
        value += __shfl_down(value, offset, 64);
    if(lane == 0)
        wave_sums[wave] = value;
    __syncthreads();
    return wave_sums[0] + wave_sums[1];
}

template <typename DataType>
__global__ void fwd_ref(const uint16_t* q,
                        const uint16_t* k,
                        const uint16_t* v,
                        const int64_t* lut,
                        const int32_t* lut_size,
                        uint16_t* out,
                        float* lse,
                        int rows,
                        int n,
                        int d,
                        int block_m,
                        int max_nnz,
                        float scale)
{
    const int row = blockIdx.x;
    const int x   = threadIdx.x;
    if(row >= rows)
        return;
    __shared__ float wave_sums[2];
    __shared__ float row_max;
    __shared__ float row_denom;
    __shared__ float probability;

    const int row_in_head = row % n;
    const int head = row / n;
    const int qb = row_in_head / block_m;
    const int q_blocks = n / block_m;
    const int count = max_nnz;
    (void)lut_size;
    const int64_t* lut_head = lut + static_cast<size_t>(head * q_blocks + qb) * max_nnz;

    if(x == 0)
        row_max = -INFINITY;
    __syncthreads();

    for(int a = 0; a < count; ++a)
    {
        const int key_block = static_cast<int>(lut_head[a]);
        for(int key_offset = 0; key_offset < 64; ++key_offset)
        {
            const int col = key_block * 64 + key_offset;
            if(col >= n)
                continue;
            const int global_col = head * n + col;
            const float product = x < d ? to_float<DataType>(q[row * d + x]) *
                                              to_float<DataType>(k[global_col * d + x])
                                        : 0.0f;
            const float dot = block_reduce_128(product, wave_sums);
            if(x == 0)
                row_max = fmaxf(row_max, dot * scale * ck_tile::log2e_v<float>);
            __syncthreads();
        }
    }
    if(x == 0)
        row_denom = 0.0f;
    float out_acc = 0.0f;
    __syncthreads();

    for(int a = 0; a < count; ++a)
    {
        const int key_block = static_cast<int>(lut_head[a]);
        for(int key_offset = 0; key_offset < 64; ++key_offset)
        {
            const int col = key_block * 64 + key_offset;
            if(col >= n)
                continue;
            const int global_col = head * n + col;
            const float product = x < d ? to_float<DataType>(q[row * d + x]) *
                                              to_float<DataType>(k[global_col * d + x])
                                        : 0.0f;
            const float dot = block_reduce_128(product, wave_sums);
            if(x == 0)
            {
                probability = exp2f(dot * scale * ck_tile::log2e_v<float> - row_max);
                row_denom += probability;
            }
            __syncthreads();
            if(x < d)
                out_acc += probability * to_float<DataType>(v[global_col * d + x]);
        }
    }
    if(x == 0)
        lse[row] = row_max + log2f(row_denom);
    if(x < d)
        out[row * d + x] = from_float<DataType>(out_acc / (row_denom + 1e-10f));
}

template <typename DataType>
__global__ void preprocess_ref(const uint16_t* out,
                               const uint16_t* dout,
                               float* delta,
                               int rows,
                               int d)
{
    const int row = blockIdx.x;
    const int x   = threadIdx.x;
    if(row >= rows)
        return;
    __shared__ float wave_sums[2];
    const float product = x < d ? to_float<DataType>(out[row * d + x]) *
                                      to_float<DataType>(dout[row * d + x])
                                : 0.0f;
    const float sum = block_reduce_128(product, wave_sums);
    if(x == 0)
        delta[row] = sum;
}

template <typename DataType>
__global__ void bwd_dq_ref(const uint16_t* q,
                           const uint16_t* k,
                           const uint16_t* v,
                           const uint16_t* dout,
                           const float* lse,
                           const float* delta,
                           const int64_t* lut,
                           float* dq,
                           int n,
                           int d,
                           int block_m,
                           int block_n,
                           int max_nnz,
                           int stage_count,
                           float scale_log2)
{
    const int row = blockIdx.x;
    const int x   = threadIdx.x;
    if(row >= n)
        return;
    __shared__ float score_reduce[kReferenceBlockSize];
    __shared__ float dp_reduce[kReferenceBlockSize];
    __shared__ float probability;
    __shared__ float ds;

    const float q_value    = x < d ? to_float<DataType>(q[row * d + x]) : 0.0f;
    const float dout_value = x < d ? to_float<DataType>(dout[row * d + x]) : 0.0f;
    const int qb = row / block_m;
    float dq_reduced = 0.0f;
    for(int stage = 0; stage < stage_count; ++stage)
    {
        float dq_stage_acc = 0.0f;
        const int active_begin = max_nnz * stage / stage_count;
        const int active_end = max_nnz * (stage + 1) / stage_count;
        for(int a = active_begin; a < active_end; ++a)
        {
            const int kb = static_cast<int>(lut[qb * max_nnz + a]);
            for(int col = kb * block_n; col < min(n, (kb + 1) * block_n); ++col)
            {
                const float k_value = x < d ? to_float<DataType>(k[col * d + x]) : 0.0f;
                const float v_value = x < d ? to_float<DataType>(v[col * d + x]) : 0.0f;
                score_reduce[x] = q_value * k_value;
                dp_reduce[x]    = dout_value * v_value;
                __syncthreads();
                for(int offset = 64; offset > 0; offset >>= 1)
                {
                    if(x < offset)
                    {
                        score_reduce[x] += score_reduce[x + offset];
                        dp_reduce[x] += dp_reduce[x + offset];
                    }
                    __syncthreads();
                }
                if(x == 0)
                {
                    probability = quantize_to_data_type<DataType>(
                        exp2f(score_reduce[0] * scale_log2 - lse[row]));
                    ds = quantize_to_data_type<DataType>(
                        probability * (dp_reduce[0] - delta[row]));
                }
                __syncthreads();
                if(x < d)
                    dq_stage_acc +=
                        scale_log2 * (1.0f / ck_tile::log2e_v<float>) * ds * k_value;
            }
        }
        if(x < d)
            dq_reduced += quantize_to_data_type<DataType>(dq_stage_acc);
    }
    if(x < d)
        dq[row * d + x] = quantize_to_data_type<DataType>(dq_reduced);
}

template <typename DataType>
__global__ void bwd_dkdv_ref(const uint16_t* q,
                             const uint16_t* k,
                             const uint16_t* v,
                             const uint16_t* dout,
                             const float* lse,
                             const float* delta,
                             const int32_t* reverse_lut,
                             const int32_t* reverse_lut_size,
                             float* dk,
                             float* dv,
                             int n,
                             int d,
                             int block_n,
                             int reverse_lut_stride,
                             float scale_log2)
{
    const int col = blockIdx.x;
    const int x   = threadIdx.x;
    if(col >= n)
        return;
    __shared__ float score_reduce[kReferenceBlockSize];
    __shared__ float dp_reduce[kReferenceBlockSize];
    __shared__ float probability;
    __shared__ float ds;

    const int kb = col / block_n;
    const float k_value = x < d ? to_float<DataType>(k[col * d + x]) : 0.0f;
    const float v_value = x < d ? to_float<DataType>(v[col * d + x]) : 0.0f;
    float dk_acc = 0.0f;
    float dv_acc = 0.0f;
    for(int a = 0; a < reverse_lut_size[kb]; ++a)
    {
        const int qb = reverse_lut[kb * reverse_lut_stride + a];
        for(int row = qb * 64; row < min(n, (qb + 1) * 64); ++row)
        {
            const float q_value = x < d ? to_float<DataType>(q[row * d + x]) : 0.0f;
            const float dout_value = x < d ? to_float<DataType>(dout[row * d + x]) : 0.0f;
            score_reduce[x] = q_value * k_value;
            dp_reduce[x]    = dout_value * v_value;
            __syncthreads();
            for(int offset = 64; offset > 0; offset >>= 1)
            {
                if(x < offset)
                {
                    score_reduce[x] += score_reduce[x + offset];
                    dp_reduce[x] += dp_reduce[x + offset];
                }
                __syncthreads();
            }
            if(x == 0)
            {
                probability = quantize_to_data_type<DataType>(
                    exp2f(score_reduce[0] * scale_log2 - lse[row]));
                ds = quantize_to_data_type<DataType>(
                    probability * (dp_reduce[0] - delta[row]));
            }
            __syncthreads();
            if(x < d)
            {
                dk_acc += scale_log2 * (1.0f / ck_tile::log2e_v<float>) * ds * q_value;
                dv_acc += probability * dout_value;
            }
        }
    }
    if(x < d)
    {
        dk[col * d + x] = dk_acc;
        dv[col * d + x] = dv_acc;
    }
}

template <typename DataType>
__global__ void linear_k_feature_ref(const DataType* k,
                                     DataType* k_feature,
                                     int rows,
                                     int d)
{
    const int row = blockIdx.x;
    const int x = threadIdx.x;
    if(row >= rows || x >= d)
        return;

    __shared__ float values[kReferenceBlockSize];
    const float value = ck_tile::type_convert<float>(k[row * d + x]);
    values[x] = value;
    __syncthreads();

    for(int stride = d / 2; stride > 0; stride >>= 1)
    {
        if(x < stride)
            values[x] = fmaxf(values[x], values[x + stride]);
        __syncthreads();
    }
    const float row_max = values[0];
    values[x] = expf(value - row_max);
    __syncthreads();
    for(int stride = d / 2; stride > 0; stride >>= 1)
    {
        if(x < stride)
            values[x] += values[x + stride];
        __syncthreads();
    }
    k_feature[row * d + x] = ck_tile::type_convert<DataType>(
        expf(value - row_max) / values[0]);
}

template <typename DataType>
__global__ void linear_ksum_ref(const DataType* k_feature,
                                float* ksum,
                                int bh,
                                int n,
                                int d)
{
    const int index = blockIdx.x;
    const int head = index / d;
    const int feature = index % d;
    if(head >= bh)
        return;

    __shared__ float values[kReferenceBlockSize];
    float sum = 0.0f;
    const int64_t base = static_cast<int64_t>(head) * n * d;
    for(int row = threadIdx.x; row < n; row += blockDim.x)
        sum += ck_tile::type_convert<float>(k_feature[base + row * d + feature]);
    values[threadIdx.x] = sum;
    __syncthreads();
    for(int stride = kReferenceBlockSize / 2; stride > 0; stride >>= 1)
    {
        if(static_cast<int>(threadIdx.x) < stride)
            values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if(threadIdx.x == 0)
        ksum[index] = values[0];
}

template <typename DataType>
__global__ void linear_kv_ref(const DataType* k_feature,
                              const DataType* v,
                              DataType* kv,
                              int bh,
                              int n,
                              int d)
{
    const int matrix_index = blockIdx.x;
    const int head = matrix_index / (d * d);
    const int rem = matrix_index % (d * d);
    const int feature = rem / d;
    const int output = rem % d;
    if(head >= bh)
        return;

    __shared__ float values[kReferenceBlockSize];
    float sum = 0.0f;
    const int64_t base = static_cast<int64_t>(head) * n * d;
    for(int row = threadIdx.x; row < n; row += blockDim.x)
        sum += ck_tile::type_convert<float>(k_feature[base + row * d + feature]) *
               ck_tile::type_convert<float>(v[base + row * d + output]);
    values[threadIdx.x] = sum;
    __syncthreads();
    for(int stride = kReferenceBlockSize / 2; stride > 0; stride >>= 1)
    {
        if(static_cast<int>(threadIdx.x) < stride)
            values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if(threadIdx.x == 0)
        kv[matrix_index] = ck_tile::type_convert<DataType>(values[0]);
}

template <typename DataType>
__global__ void linear_projection_ref(const DataType* kv,
                                      const DataType* weight,
                                      DataType* projected,
                                      int bh,
                                      int d)
{
    const int matrix_index = blockIdx.x;
    const int head = matrix_index / (d * d);
    const int rem = matrix_index % (d * d);
    const int feature = rem / d;
    const int output = rem % d;
    if(head >= bh)
        return;

    __shared__ float values[kReferenceBlockSize];
    float sum = 0.0f;
    for(int k = threadIdx.x; k < d; k += blockDim.x)
        sum += ck_tile::type_convert<float>(
                   kv[(static_cast<int64_t>(head) * d + feature) * d + k]) *
               ck_tile::type_convert<float>(weight[output * d + k]);
    values[threadIdx.x] = sum;
    __syncthreads();
    for(int stride = kReferenceBlockSize / 2; stride > 0; stride >>= 1)
    {
        if(static_cast<int>(threadIdx.x) < stride)
            values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if(threadIdx.x == 0)
        projected[matrix_index] = ck_tile::type_convert<DataType>(values[0]);
}

template <typename DataType>
__global__ void linear_final_ref(const DataType* q,
                                 const DataType* projected,
                                 const float* ksum,
                                 const DataType* bias,
                                 const DataType* sparse_out,
                                 DataType* output,
                                 int rows,
                                 int n,
                                 int d)
{
    const int row = blockIdx.x;
    const int x = threadIdx.x;
    if(row >= rows || x >= d)
        return;

    const int head = row / n;
    __shared__ float values[kReferenceBlockSize];
    __shared__ DataType q_feature[kReferenceBlockSize];
    const float q_value = ck_tile::type_convert<float>(q[row * d + x]);
    values[x] = q_value;
    __syncthreads();
    for(int stride = d / 2; stride > 0; stride >>= 1)
    {
        if(x < stride)
            values[x] = fmaxf(values[x], values[x + stride]);
        __syncthreads();
    }
    const float row_max = values[0];
    values[x] = expf(q_value - row_max);
    __syncthreads();
    for(int stride = d / 2; stride > 0; stride >>= 1)
    {
        if(x < stride)
            values[x] += values[x + stride];
        __syncthreads();
    }
    q_feature[x] = ck_tile::type_convert<DataType>(
        expf(q_value - row_max) / values[0]);
    __syncthreads();

    values[x] = ck_tile::type_convert<float>(q_feature[x]) * ksum[head * d + x];
    __syncthreads();
    for(int stride = d / 2; stride > 0; stride >>= 1)
    {
        if(x < stride)
            values[x] += values[x + stride];
        __syncthreads();
    }
    const float inv_denom = 1.0f / (values[0] + 1.0e-5f);

    float numerator = 0.0f;
    for(int feature = 0; feature < d; ++feature)
        numerator += ck_tile::type_convert<float>(q_feature[feature]) *
                     ck_tile::type_convert<float>(
                         projected[(static_cast<int64_t>(head) * d + feature) * d + x]);
    const DataType normalized = ck_tile::type_convert<DataType>(numerator * inv_denom);
    const float result = ck_tile::type_convert<float>(normalized) +
                         ck_tile::type_convert<float>(bias[x]) +
                         ck_tile::type_convert<float>(sparse_out[row * d + x]);
    output[row * d + x] = ck_tile::type_convert<DataType>(result);
}

} // namespace sla_reference

namespace ck_tile {
namespace sla_reference = ::sla_reference;
} // namespace ck_tile
