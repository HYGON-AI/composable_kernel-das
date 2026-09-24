// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <ck_tile/core.hpp>
#include <cmath>
#include <cstdint>

namespace jenga_reference {

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

// The validation kernels use exactly two gfx9 wave64s per block.
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

// Independent GPU reference. One 128-thread block owns a query row. The
// threads cooperate on QK reductions, so every attention score is evaluated
// only once per softmax pass instead of once per output element.
template <typename DataType>
__global__ void fwd(const uint16_t* q,
                         const uint16_t* k,
                         const uint16_t* v,
                         const int32_t* lut,
                         const int32_t* lut_size,
                         uint16_t* out,
                         float* lse,
                         int rows,
                         int n,
                         int d,
                         int block_n,
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
    const int blocks_per_head = (n + block_n - 1) / block_n;
    const int qb = head * blocks_per_head + row_in_head / block_n;
    const int count = lut_size[qb];
    if(x == 0)
        row_max = -INFINITY;
    __syncthreads();

    for(int a = 0; a < count; ++a)
    {
        const int kb = lut[qb * max_nnz + a];
        for(int col = kb * block_n; col < min(n, (kb + 1) * block_n); ++col)
        {
            const int global_col = head * n + col;
            const float product = x < d ? to_float<DataType>(q[row * d + x]) *
                                              to_float<DataType>(k[global_col * d + x])
                                        : 0.0f;
            const float dot = block_reduce_128(product, wave_sums);
            if(x == 0)
                row_max = fmaxf(row_max, dot * scale);
            __syncthreads();
        }
    }
    if(x == 0)
        row_denom = 0.0f;
    float out_acc = 0.0f;
    __syncthreads();

    for(int a = 0; a < count; ++a)
    {
        const int kb = lut[qb * max_nnz + a];
        for(int col = kb * block_n; col < min(n, (kb + 1) * block_n); ++col)
        {
            const int global_col = head * n + col;
            const float product = x < d ? to_float<DataType>(q[row * d + x]) *
                                              to_float<DataType>(k[global_col * d + x])
                                        : 0.0f;
            const float dot = block_reduce_128(product, wave_sums);
            if(x == 0)
            {
                probability = exp2f(dot * scale - row_max);
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
        out[row * d + x] = from_float<DataType>(out_acc / row_denom);
}

template <typename DataType>
__global__ void preprocess(const uint16_t* out,
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

// One block owns a query row and produces dQ without atomics.
template <typename DataType>
__global__ void bwd_dq(const uint16_t* q,
                       const uint16_t* k,
                       const uint16_t* v,
                       const uint16_t* dout,
                       const float* lse,
                       const float* delta,
                       const int32_t* lut,
                       const int32_t* lut_size,
                       float* dq,
                       int n,
                       int d,
                       int block_n,
                       int max_nnz,
                       float scale)
{
    const int row = blockIdx.x;
    const int x   = threadIdx.x;
    if(row >= n)
        return;
    __shared__ float score_reduce[kReferenceBlockSize];
    __shared__ float dp_reduce[kReferenceBlockSize];
    __shared__ float probability;
    __shared__ float ds;

    float dq_acc = 0.0f;
    const float q_value    = x < d ? to_float<DataType>(q[row * d + x]) : 0.0f;
    const float dout_value = x < d ? to_float<DataType>(dout[row * d + x]) : 0.0f;
    const int qb = row / block_n;
    for(int a = 0; a < lut_size[qb]; ++a)
    {
        const int kb = lut[qb * max_nnz + a];
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
                probability = exp2f(score_reduce[0] * scale - lse[row]);
                ds = probability * (dp_reduce[0] - delta[row]);
            }
            __syncthreads();
            constexpr float ln2 = 1.0f / ck_tile::log2e_v<float>;
            const float grad_scale = scale * ln2;
            if(x < d)
                dq_acc += grad_scale * ds * k_value;
        }
    }
    if(x < d)
        dq[row * d + x] = dq_acc;
}

// One block owns a key row. The reverse LUT enumerates only query blocks that
// selected this key block, allowing dK/dV to be accumulated without atomics.
template <typename DataType>
__global__ void bwd_dkdv(const uint16_t* q,
                         const uint16_t* k,
                         const uint16_t* v,
                         const uint16_t* dout,
                         const float* lse,
                         const float* delta,
                         const int32_t* rlut,
                         const int32_t* rlut_size,
                         float* dk,
                         float* dv,
                         int n,
                         int d,
                         int block_n,
                         int max_nnz_r,
                         float scale)
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
    for(int a = 0; a < rlut_size[kb]; ++a)
    {
        const int qb = rlut[kb * max_nnz_r + a];
        for(int row = qb * block_n; row < min(n, (qb + 1) * block_n); ++row)
        {
            const float q_value = x < d ? to_float<DataType>(q[row * d + x]) : 0.0f;
            const float dout_value =
                x < d ? to_float<DataType>(dout[row * d + x]) : 0.0f;
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
                probability = exp2f(score_reduce[0] * scale - lse[row]);
                ds = probability * (dp_reduce[0] - delta[row]);
            }
            __syncthreads();
            constexpr float ln2 = 1.0f / ck_tile::log2e_v<float>;
            const float grad_scale = scale * ln2;
            if(x < d)
            {
                dk_acc += grad_scale * ds * q_value;
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
__global__ void cast_from_float(const float* input, uint16_t* output, long long count)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < count)
        output[i] = from_float<DataType>(input[i]);
}

} // namespace jenga_reference

namespace ck_tile {
namespace jenga_reference = ::jenga_reference;
}
