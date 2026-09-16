// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/host.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ck_tile {

struct reference_unified_attention_arg
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const int32_t* block_table;
    const int32_t* q_offsets;
    const int32_t* n_kvs;
    const float* sinks;
    const float* alibi;
    const float* qq_bias;
    const int32_t* mm_ranges;
    uint16_t* out;
    int batch;
    int total_nq;
    int h_q;
    int h_kv;
    int d;
    int page_size;
    int block_table_stride;
    int qq_bias_stride;
    int max_mm_ranges;
    int sliding_window;
    int q_token_stride;
    int q_head_stride;
    int64_t k_page_stride;
    int64_t k_token_stride;
    int64_t k_head_stride;
    int64_t v_page_stride;
    int64_t v_token_stride;
    int64_t v_head_stride;
    float scale;
    float softcap;
    bool alibi_sqrt;
    bool qq_bias_fp32;
};

using UnifiedAttentionReferenceArgument = reference_unified_attention_arg;

template <typename DataType>
struct reference_unified_attention_kernel
{
    using Kargs = reference_unified_attention_arg;
    static constexpr int kBlockSize = 256;
    static constexpr int kKeyTile = 64;

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }
    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.total_nq * arg.h_q));
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        __shared__ float scores[kKeyTile];
        __shared__ float weights[kKeyTile];
        __shared__ float dot_partials[kBlockSize];
        __shared__ float old_weight;
        __shared__ float shared_denom;

        const int tid = static_cast<int>(threadIdx.x);
        const int q_global = static_cast<int>(blockIdx.x) / arg.h_q;
        const int q_head = static_cast<int>(blockIdx.x) % arg.h_q;
        int seq = 0;
        while(seq + 1 < arg.batch && q_global >= arg.q_offsets[seq + 1]) ++seq;
        const int q_local = q_global - arg.q_offsets[seq];
        const int n_q = arg.q_offsets[seq + 1] - arg.q_offsets[seq];
        const int n_kv = arg.n_kvs[seq];
        const int context = n_kv - n_q;
        const int q_abs = context + q_local;
        const int kv_head = q_head / (arg.h_q / arg.h_kv);

        float acc = 0.0f;
        float running_max = arg.sinks == nullptr ? -INFINITY : arg.sinks[q_head];
        if(tid == 0) shared_denom = arg.sinks == nullptr ? 0.0f : 1.0f;
        __syncthreads();

        for(int key_begin = 0; key_begin < n_kv; key_begin += kKeyTile)
        {
            const int key_slot = tid / 4;
            const int d_lane = tid & 3;
            const int key = key_begin + key_slot;
            float dot = 0.0f;
            if(key < n_kv)
            {
                const int physical_page =
                    arg.block_table[static_cast<int64_t>(seq) *
                                        arg.block_table_stride +
                                    key / arg.page_size];
                const int page_offset = key % arg.page_size;
                for(int x = d_lane; x < arg.d; x += 4)
                {
                    const auto q_word =
                        arg.q[static_cast<int64_t>(q_global) * arg.q_token_stride +
                              static_cast<int64_t>(q_head) * arg.q_head_stride + x];
                    const auto k_word =
                        arg.k[static_cast<int64_t>(physical_page) *
                                  arg.k_page_stride +
                              static_cast<int64_t>(page_offset) *
                                  arg.k_token_stride +
                              static_cast<int64_t>(kv_head) *
                                  arg.k_head_stride +
                              x];
                    dot += ck_tile::type_convert<float>(
                               ck_tile::bit_cast<DataType>(q_word)) *
                           ck_tile::type_convert<float>(
                               ck_tile::bit_cast<DataType>(k_word));
                }
            }
            dot_partials[tid] = dot;
            __syncthreads();
            if(d_lane == 0)
            {
                dot = dot_partials[tid] + dot_partials[tid + 1] +
                      dot_partials[tid + 2] + dot_partials[tid + 3];
                const int causal_begin =
                    arg.sliding_window > 0
                        ? ck_tile::max(0, q_abs - arg.sliding_window + 1)
                        : 0;
                const int causal_end = q_abs;
                bool visible =
                    key < n_kv && key >= causal_begin && key <= causal_end;
                if(arg.mm_ranges != nullptr)
                {
                    const int32_t* ranges =
                        arg.mm_ranges + static_cast<int64_t>(seq) * arg.max_mm_ranges * 2;
                    for(int i = 0; i < arg.max_mm_ranges; ++i)
                    {
                        const int begin = ranges[i * 2];
                        const int end = ranges[i * 2 + 1];
                        if(begin < end && q_abs >= begin && q_abs <= end &&
                           key >= begin && key <= end)
                            visible = true;
                    }
                }
                float score = dot * arg.scale;
                if(arg.softcap > 0.0f)
                    score = arg.softcap * tanhf(score / arg.softcap);
                if(key < n_kv && arg.alibi != nullptr)
                {
                    const int rel = key - q_abs;
                    const float offset =
                        arg.alibi_sqrt
                            ? (rel <= 0
                                   ? -sqrtf(static_cast<float>(-rel))
                                   : 0.0f)
                            : static_cast<float>(key - context);
                    score += arg.alibi[q_head] * offset;
                }
                if(key < n_kv && arg.qq_bias != nullptr)
                {
                    const int key_rel = key - context;
                    if(key_rel >= 0 && key_rel < arg.qq_bias_stride)
                    {
                        const int64_t bias_offset =
                            static_cast<int64_t>(q_local) * arg.qq_bias_stride +
                            key_rel;
                        score +=
                            arg.qq_bias_fp32
                                ? arg.qq_bias[bias_offset]
                                : ck_tile::type_convert<float>(
                                      reinterpret_cast<const DataType*>(
                                          arg.qq_bias)[bias_offset]);
                    }
                }
                scores[key_slot] = visible ? score : -INFINITY;
            }
            __syncthreads();

            if(tid == 0)
            {
                float next_max = running_max;
                for(int i = 0; i < kKeyTile; ++i)
                    next_max = ck_tile::max(next_max, scores[i]);
                old_weight =
                    isfinite(running_max) ? expf(running_max - next_max) : 0.0f;
                shared_denom *= old_weight;
                for(int i = 0; i < kKeyTile; ++i)
                {
                    weights[i] =
                        isfinite(scores[i]) ? expf(scores[i] - next_max) : 0.0f;
                    shared_denom += weights[i];
                }
                running_max = next_max;
            }
            __syncthreads();

            if(tid < arg.d)
            {
                acc *= old_weight;
                for(int key_slot_i = 0; key_slot_i < kKeyTile; ++key_slot_i)
                {
                    const int key_i = key_begin + key_slot_i;
                    if(key_i >= n_kv || weights[key_slot_i] == 0.0f) continue;
                    const int physical_page =
                        arg.block_table[static_cast<int64_t>(seq) *
                                            arg.block_table_stride +
                                        key_i / arg.page_size];
                    const int page_offset = key_i % arg.page_size;
                    const auto v_word =
                        arg.v[static_cast<int64_t>(physical_page) *
                                  arg.v_page_stride +
                              static_cast<int64_t>(page_offset) *
                                  arg.v_token_stride +
                              static_cast<int64_t>(kv_head) *
                                  arg.v_head_stride +
                              tid];
                    acc += weights[key_slot_i] *
                           ck_tile::type_convert<float>(
                               ck_tile::bit_cast<DataType>(v_word));
                }
            }
            __syncthreads();
        }

        if(tid < arg.d)
            arg.out[(static_cast<int64_t>(q_global) * arg.h_q + q_head) *
                        arg.d +
                    tid] =
                ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(
                    shared_denom > 0.0f ? acc / shared_denom : 0.0f));
    }
};

template <typename DataType>
using UnifiedAttentionReferenceKernel = reference_unified_attention_kernel<DataType>;

template <typename DataType>
void reference_unified_attention(
    const reference_unified_attention_arg& reference_arg,
    hipStream_t stream = nullptr)
{
    using Kernel = reference_unified_attention_kernel<DataType>;
    constexpr auto block = Kernel::BlockSize();
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<block.x, 1>(
            Kernel{}, Kernel::GridSize(reference_arg), block, 0, reference_arg));
}

template <typename DataType>
void run_unified_attention_reference(
    const reference_unified_attention_arg& reference_arg,
    hipStream_t stream = nullptr)
{
    reference_unified_attention<DataType>(reference_arg, stream);
}

} // namespace ck_tile

using ck_tile::reference_unified_attention_arg;
using ck_tile::reference_unified_attention_kernel;
using ck_tile::UnifiedAttentionReferenceArgument;
using ck_tile::UnifiedAttentionReferenceKernel;
using ck_tile::reference_unified_attention;
using ck_tile::run_unified_attention_reference;
