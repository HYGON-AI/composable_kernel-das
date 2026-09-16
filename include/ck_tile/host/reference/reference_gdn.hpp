// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include <hip/hip_runtime.h>

namespace gdn_reference {

template <typename DataType>
CK_TILE_DEVICE float load_value(const uint16_t* p, size_t i)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<DataType>(p[i]));
}

template <typename DataType>
CK_TILE_DEVICE uint16_t store_value(float x)
{
    return ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(x));
}

template <typename DataType>
CK_TILE_DEVICE float load_beta(const void* beta, size_t i, bool beta_is_data_type)
{
    return beta_is_data_type
               ? load_value<DataType>(static_cast<const uint16_t*>(beta), i)
               : static_cast<const float*>(beta)[i];
}

// Independent recurrent GPU reference. One block owns one value head and each
// thread owns one value column of the FP32 state. Q/K are common to every value
// column, so the block loads them once per token instead of issuing 128
// identical global loads for every dimension.
template <typename DataType>
__global__ void prefill(const uint16_t* q,
                        const uint16_t* k,
                        const uint16_t* v,
                        const float* g,
                        const float* gk,
                        const float* gv,
                        const float* a_log,
                        const float* dt_bias,
                        const void* beta,
                        const float* initial_state,
                        const int64_t* cu_seqlens,
                        uint16_t* output,
                        float* final_state,
                        int t,
                        int h_qk,
                        int h_v,
                        int num_sequences,
                        bool is_varlen,
                        float scale,
                        bool qk_l2norm,
                        bool beta_is_data_type,
                        bool gate_in_kernel,
                        bool has_dt_bias,
                        bool use_g,
                        bool use_gk,
                        bool use_gv,
                        bool beta_headwise,
                        bool use_exp2,
                        bool transpose_state)
{
    const int sequence   = blockIdx.x / h_v;
    const int value_head = blockIdx.x % h_v;
    const int value_col  = threadIdx.x;
    if(sequence >= num_sequences || value_col >= 128)
        return;

    const int begin = is_varlen
        ? static_cast<int>(cu_seqlens[sequence])
        : sequence * t;
    const int end = is_varlen
        ? static_cast<int>(cu_seqlens[sequence + 1])
        : begin + t;
    const int qk_head = value_head / (h_v / h_qk);
    __shared__ float q_shared[128];
    __shared__ float k_shared[128];
    __shared__ float key_decay_shared[128];
    __shared__ float decay_shared;
    __shared__ float q_inv_norm;
    __shared__ float k_inv_norm;
    __shared__ float q_norm_shared[128];
    __shared__ float k_norm_shared[128];
    float state[128];
#pragma unroll
    for(int d = 0; d < 128; ++d)
    {
        const size_t head_offset =
            (static_cast<size_t>(sequence) * h_v + value_head) * 128 * 128;
        const size_t state_offset = transpose_state
            ? head_offset + static_cast<size_t>(value_col) * 128 + d
            : head_offset + static_cast<size_t>(d) * 128 + value_col;
        state[d] = initial_state == nullptr ? 0.0f : initial_state[state_offset];
    }

    for(int token = begin; token < end; ++token)
    {
        const size_t qk_offset =
            (static_cast<size_t>(token) * h_qk + qk_head) * 128 + threadIdx.x;
        q_shared[threadIdx.x] = load_value<DataType>(q, qk_offset);
        k_shared[threadIdx.x] = load_value<DataType>(k, qk_offset);
        const size_t key_gate_offset =
            (static_cast<size_t>(token) * h_v + value_head) * 128 + threadIdx.x;
        key_decay_shared[threadIdx.x] =
            use_gk ? (use_exp2 ? exp2f(gk[key_gate_offset])
                               : expf(gk[key_gate_offset]))
                   : 1.0f;
        if(qk_l2norm)
        {
            q_norm_shared[threadIdx.x] =
                q_shared[threadIdx.x] * q_shared[threadIdx.x];
            k_norm_shared[threadIdx.x] =
                k_shared[threadIdx.x] * k_shared[threadIdx.x];
        }
        __syncthreads();
        if(qk_l2norm)
        {
            for(int offset = 64; offset > 0; offset >>= 1)
            {
                if(static_cast<int>(threadIdx.x) < offset)
                {
                    q_norm_shared[threadIdx.x] +=
                        q_norm_shared[threadIdx.x + offset];
                    k_norm_shared[threadIdx.x] +=
                        k_norm_shared[threadIdx.x + offset];
                }
                __syncthreads();
            }
        }
        if(threadIdx.x == 0)
        {
            const size_t gate_offset =
                static_cast<size_t>(token) * h_v + value_head;
            if(!use_g)
            {
                decay_shared = 1.0f;
            }
            else if(gate_in_kernel)
            {
                float gate = g[gate_offset];
                if(has_dt_bias)
                    gate += dt_bias[value_head];
                const float softplus =
                    gate < 20.0f ? log1pf(expf(gate)) : gate;
                decay_shared =
                    expf(-expf(a_log[value_head]) * softplus);
            }
            else
            {
                decay_shared = use_exp2 ? exp2f(g[gate_offset])
                                        : expf(g[gate_offset]);
            }
            if(qk_l2norm)
            {
                q_inv_norm = rsqrtf(q_norm_shared[0] + 1.0e-6f);
                k_inv_norm = rsqrtf(k_norm_shared[0] + 1.0e-6f);
            }
        }
        __syncthreads();
        if(qk_l2norm)
        {
            q_shared[threadIdx.x] *= q_inv_norm;
            k_shared[threadIdx.x] *= k_inv_norm;
        }
        __syncthreads();

        const size_t value_gate_offset =
            (static_cast<size_t>(token) * h_v + value_head) * 128 + value_col;
        const float value_decay =
            use_gv ? (use_exp2 ? exp2f(gv[value_gate_offset])
                               : expf(gv[value_gate_offset]))
                   : 1.0f;
        float projection = 0.0f;
#pragma unroll
        for(int d = 0; d < 128; ++d)
        {
            state[d] *= decay_shared * key_decay_shared[d] * value_decay;
            projection += k_shared[d] * state[d];
        }

        const size_t v_offset =
            (static_cast<size_t>(token) * h_v + value_head) * 128 + value_col;
        const size_t beta_offset =
            beta_headwise
                ? static_cast<size_t>(token) * h_v + value_head
                : v_offset;
        const float residual =
            (load_value<DataType>(v, v_offset) - projection) *
            load_beta<DataType>(beta, beta_offset, beta_is_data_type);

        float out = 0.0f;
#pragma unroll
        for(int d = 0; d < 128; ++d)
        {
            state[d] += k_shared[d] * residual;
            out += q_shared[d] * state[d];
        }
        if(output != nullptr)
            output[v_offset] = store_value<DataType>(out * scale);
        __syncthreads();
    }

    if(final_state != nullptr)
    {
#pragma unroll
        for(int d = 0; d < 128; ++d)
        {
            const size_t head_offset =
                (static_cast<size_t>(sequence) * h_v + value_head) * 128 * 128;
            const size_t state_offset = transpose_state
                ? head_offset + static_cast<size_t>(value_col) * 128 + d
                : head_offset + static_cast<size_t>(d) * 128 + value_col;
            final_state[state_offset] = state[d];
        }
    }
}

template <typename DataType>
void launch(const uint16_t* q,
            const uint16_t* k,
            const uint16_t* v,
            const float* g,
            const float* gk,
            const float* gv,
            const float* a_log,
            const float* dt_bias,
            const void* beta,
            const float* initial_state,
            const int64_t* cu_seqlens,
            uint16_t* output,
            float* final_state,
            int t,
            int h_qk,
            int h_v,
            int num_sequences,
            bool is_varlen,
            float scale,
            bool qk_l2norm = false,
            bool beta_is_data_type = false,
            bool gate_in_kernel = false,
            bool has_dt_bias = false,
            bool use_g = true,
            bool use_gk = false,
            bool use_gv = false,
            bool beta_headwise = true,
            bool use_exp2 = false,
            bool transpose_state = false,
            hipStream_t stream = nullptr)
{
    prefill<DataType><<<dim3(num_sequences * h_v), dim3(128), 0, stream>>>(
        q,
        k,
        v,
        g,
        gk,
        gv,
        a_log,
        dt_bias,
        beta,
        initial_state,
        cu_seqlens,
        output,
        final_state,
        t,
        h_qk,
        h_v,
        num_sequences,
        is_varlen,
        scale,
        qk_l2norm,
        beta_is_data_type,
        gate_in_kernel,
        has_dt_bias,
        use_g,
        use_gk,
        use_gv,
        beta_headwise,
        use_exp2,
        transpose_state);
}

} // namespace gdn_reference

namespace ck_tile {
namespace gdn_reference = ::gdn_reference;
}
