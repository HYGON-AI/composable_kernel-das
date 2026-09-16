// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/ops/unified_attention/kernel/unified_attention_d256_mmac_kernel.hpp"

struct UnifiedAttentionD256TinyBatchArgument
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    uint16_t* out;
    const int32_t* block_table;
    const float* sinks;
    const float* alibi;
    const float* qq_bias;
    const int32_t* mm_ranges;
    const int32_t* q_offsets_device;
    const int32_t* n_kvs_device;
    int q_offsets[kMaxBatchSeqs + 1];
    int n_kvs[kMaxBatchSeqs];
    int num_seqs;
    int h_q;
    int h_kv;
    int head_dim;
    int page_size;
    int max_mm_ranges;
    int qq_bias_stride;
    int sliding_window;
    int total_q_rows;
    int64_t q_stride_t;
    int64_t q_stride_h;
    int64_t k_stride_page;
    int64_t k_stride_token;
    int64_t k_stride_head;
    int64_t v_stride_page;
    int64_t v_stride_token;
    int64_t v_stride_head;
    int64_t out_stride_t;
    int64_t out_stride_h;
    int64_t block_table_stride;
    float qk_scale_log2;
    float softcap;
    bool alibi_sqrt;
};

template <typename DataType, bool HasSoftcap, bool HasQqBias>
struct UnifiedAttentionD256TinyBatchKernel
{
    using Kargs = UnifiedAttentionD256TinyBatchArgument;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.h_q),
                    static_cast<unsigned int>(arg.total_q_rows / arg.h_q));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(64); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int kWaveSize = ck_tile::get_warp_size();
        constexpr int kValuesPerLane = 4;
        constexpr float kLog2e = ck_tile::log2e_v<float>;

        const int lane = static_cast<int>(threadIdx.x);
        const int qhead = static_cast<int>(blockIdx.x);
        const int global_q = static_cast<int>(blockIdx.y);
        int seq = 0;
        while(seq + 1 < arg.num_seqs &&
              global_q >= (arg.q_offsets_device != nullptr
                                ? arg.q_offsets_device[seq + 1]
                                : arg.q_offsets[seq + 1]))
            ++seq;
        const int q_begin = arg.q_offsets_device != nullptr
                                ? arg.q_offsets_device[seq]
                                : arg.q_offsets[seq];
        const int q_end = arg.q_offsets_device != nullptr
                              ? arg.q_offsets_device[seq + 1]
                              : arg.q_offsets[seq + 1];
        const int q_pos = global_q - q_begin;
        const int n_q = q_end - q_begin;
        const int n_kv = arg.n_kvs_device != nullptr ? arg.n_kvs_device[seq]
                                                     : arg.n_kvs[seq];
        const int kv_head = qhead * arg.h_kv / arg.h_q;
        const int context = n_kv - n_q;

        float o_values[kValuesPerLane] = {0.0f, 0.0f, 0.0f, 0.0f};

        const int q_abs = context + q_pos;
        const int32_t* seq_ranges =
            arg.mm_ranges == nullptr
                ? nullptr
                : arg.mm_ranges + static_cast<int64_t>(seq) * arg.max_mm_ranges * 2;
        const int causal_begin =
            arg.sliding_window > 0 ? ck_tile::max(0, q_abs - arg.sliding_window + 1) : 0;
        const int causal_end = ck_tile::min(n_kv - 1, q_abs);

        auto is_key_allowed = [&](int k_pos) {
            bool allowed = k_pos >= causal_begin && k_pos <= causal_end;
            if(seq_ranges != nullptr)
            {
                for(int i = 0; i < arg.max_mm_ranges; ++i)
                {
                    const int begin = seq_ranges[i * 2];
                    const int end = seq_ranges[i * 2 + 1];
                    allowed = allowed ||
                              (begin < end && q_abs >= begin && q_abs <= end &&
                               k_pos >= begin && k_pos <= end);
                }
            }
            return allowed;
        };

        auto apply_score_modifiers = [&](float dot, int k_pos) {
            float score = dot * arg.qk_scale_log2;
            if constexpr(HasSoftcap)
            {
                const float softcap_log2 = arg.softcap * kLog2e;
                score = softcap_log2 * tanhf(score / softcap_log2);
            }
            if(arg.alibi != nullptr)
            {
                const int relative = k_pos - q_abs;
                const float offset = arg.alibi_sqrt
                                         ? (relative <= 0
                                                ? -sqrtf(static_cast<float>(-relative))
                                                : 0.0f)
                                         : static_cast<float>(k_pos - context);
                score += arg.alibi[qhead] * offset * kLog2e;
            }
            if constexpr(HasQqBias)
            {
                const int key_rel = k_pos - context;
                if(key_rel >= 0 && key_rel < arg.qq_bias_stride)
                    score += arg.qq_bias[static_cast<int64_t>(q_pos) *
                                             arg.qq_bias_stride +
                                         key_rel] *
                             kLog2e;
            }
            return score;
        };

        // A tiny row has at most one score per lane. For three or more keys,
        // split the wave into four 16-lane dot-product groups so four QK scores
        // can make progress together. Group g computes key 4*b+g and stores it
        // in lane 16*g+b. The extra Q registers are intentional: tiny launches
        // are limited by total waves and latency, not by resident occupancy.
        float lane_score = -ck_tile::numeric<float>::infinity();
        const int64_t table_base = static_cast<int64_t>(seq) * arg.block_table_stride;
        const int causal_key_count =
            causal_end >= causal_begin ? causal_end - causal_begin + 1 : 0;
        const bool use_grouped_qk =
            seq_ranges != nullptr ? n_kv >= 3 : causal_key_count >= 3;
        if(use_grouped_qk)
        {
            constexpr int kQkGroups = 4;
            constexpr int kQkGroupSize = kWaveSize / kQkGroups;
            const int key_group = lane / kQkGroupSize;
            const int d_lane = lane - key_group * kQkGroupSize;
            float q_group_values[kQkGroupSize];
            ck_tile::static_for<0, kQkGroupSize, 1>{}([&](auto i_num) {
                constexpr int i = decltype(i_num)::value;
                const int d = d_lane + i * kQkGroupSize;
                q_group_values[i] = ck_tile::type_convert<float>(
                    ck_tile::bit_cast<DataType>(
                        arg.q[static_cast<int64_t>(global_q) * arg.q_stride_t +
                              static_cast<int64_t>(qhead) * arg.q_stride_h + d]));
            });

            const bool single_page = n_kv <= arg.page_size;
            const int32_t first_physical_page = arg.block_table[table_base];
            const int key_batches = (n_kv + kQkGroups - 1) / kQkGroups;
            for(int key_batch = 0; key_batch < key_batches; ++key_batch)
            {
                const int k_pos = key_batch * kQkGroups + key_group;
                if(k_pos < n_kv && is_key_allowed(k_pos))
                {
                    int32_t key_physical_page = first_physical_page;
                    int key_page_offset = k_pos;
                    if(!single_page)
                    {
                        key_physical_page =
                            arg.block_table[table_base + k_pos / arg.page_size];
                        key_page_offset = k_pos % arg.page_size;
                    }
                    const int64_t grouped_k_base =
                        static_cast<int64_t>(key_physical_page) * arg.k_stride_page +
                        static_cast<int64_t>(key_page_offset) * arg.k_stride_token +
                        static_cast<int64_t>(kv_head) * arg.k_stride_head;
                    float dot = 0.0f;
                    ck_tile::static_for<0, kQkGroupSize, 1>{}([&](auto i_num) {
                        constexpr int i = decltype(i_num)::value;
                        const int d = d_lane + i * kQkGroupSize;
                        const float k_value = ck_tile::type_convert<float>(
                            ck_tile::bit_cast<DataType>(arg.k[grouped_k_base + d]));
                        dot += q_group_values[i] * k_value;
                    });
                    ck_tile::static_for<0, 4, 1>{}([&](auto step_num) {
                        constexpr int offset =
                            (kQkGroupSize / 2) >> decltype(step_num)::value;
                        dot += ck_tile::warp_shuffle(dot, lane ^ offset);
                    });

                    if(d_lane == key_batch)
                        lane_score = apply_score_modifiers(dot, k_pos);
                }
            }
        }
        else
        {
            float q_values[kValuesPerLane];
            ck_tile::static_for<0, kValuesPerLane, 1>{}([&](auto i_num) {
                constexpr int i = decltype(i_num)::value;
                const int d = lane + i * kWaveSize;
                q_values[i] = ck_tile::type_convert<float>(
                    ck_tile::bit_cast<DataType>(
                        arg.q[static_cast<int64_t>(global_q) * arg.q_stride_t +
                              static_cast<int64_t>(qhead) * arg.q_stride_h + d]));
            });
            int page_index = 0;
            int page_offset = 0;
            int32_t physical_page = n_kv > 0 ? arg.block_table[table_base] : 0;
            int64_t k_base = static_cast<int64_t>(physical_page) * arg.k_stride_page +
                             static_cast<int64_t>(kv_head) * arg.k_stride_head;
            for(int k_pos = 0; k_pos < n_kv; ++k_pos)
            {
                if(is_key_allowed(k_pos))
                {
                    float dot = 0.0f;
                    ck_tile::static_for<0, kValuesPerLane, 1>{}([&](auto i_num) {
                        constexpr int i = decltype(i_num)::value;
                        const int d = lane + i * kWaveSize;
                        const float k_value = ck_tile::type_convert<float>(
                            ck_tile::bit_cast<DataType>(arg.k[k_base + d]));
                        dot += q_values[i] * k_value;
                    });
                    ck_tile::static_for<0, 6, 1>{}([&](auto step_num) {
                        constexpr int offset =
                            (kWaveSize / 2) >> decltype(step_num)::value;
                        dot += ck_tile::warp_shuffle(dot, lane ^ offset);
                    });

                    if(lane == k_pos)
                        lane_score = apply_score_modifiers(dot, k_pos);
                }

                ++page_offset;
                if(page_offset == arg.page_size)
                {
                    page_offset = 0;
                    ++page_index;
                    if(k_pos + 1 < n_kv)
                    {
                        physical_page = arg.block_table[table_base + page_index];
                        k_base = static_cast<int64_t>(physical_page) * arg.k_stride_page +
                                 static_cast<int64_t>(kv_head) * arg.k_stride_head;
                    }
                }
                else
                    k_base += arg.k_stride_token;
            }
        }

        const float sink_score = arg.sinks != nullptr
                                     ? arg.sinks[qhead] * kLog2e
                                     : -ck_tile::numeric<float>::infinity();
        float row_max = ck_tile::max(lane_score, sink_score);
        ck_tile::static_for<0, 6, 1>{}([&](auto step_num) {
            constexpr int offset = (kWaveSize / 2) >> decltype(step_num)::value;
            row_max = ck_tile::max(row_max,
                                   ck_tile::warp_shuffle(row_max, lane ^ offset));
        });

        const float lane_weight =
            lane_score == -ck_tile::numeric<float>::infinity()
                ? 0.0f
                : unified_attention_d256_fast_exp2(lane_score - row_max);
        float row_sum = lane_weight;
        ck_tile::static_for<0, 6, 1>{}([&](auto step_num) {
            constexpr int offset = (kWaveSize / 2) >> decltype(step_num)::value;
            row_sum += ck_tile::warp_shuffle(row_sum, lane ^ offset);
        });
        if(arg.sinks != nullptr)
            row_sum += unified_attention_d256_fast_exp2(sink_score - row_max);

        int v_page_index = 0;
        int v_page_offset = 0;
        int32_t v_physical_page = n_kv > 0 ? arg.block_table[table_base] : 0;
        int64_t v_base = static_cast<int64_t>(v_physical_page) * arg.v_stride_page +
                         static_cast<int64_t>(kv_head) * arg.v_stride_head;
        for(int k_pos = 0; k_pos < n_kv; ++k_pos)
        {
            if(is_key_allowed(k_pos))
            {
                const int weight_lane =
                    use_grouped_qk ? (k_pos & 3) * 16 + (k_pos >> 2) : k_pos;
                const float weight = ck_tile::warp_shuffle(lane_weight, weight_lane);
                ck_tile::static_for<0, kValuesPerLane, 1>{}([&](auto i_num) {
                    constexpr int i = decltype(i_num)::value;
                    const int d = lane + i * kWaveSize;
                    const float v_value = ck_tile::type_convert<float>(
                        ck_tile::bit_cast<DataType>(arg.v[v_base + d]));
                    o_values[i] += weight * v_value;
                });
            }

            ++v_page_offset;
            if(v_page_offset == arg.page_size)
            {
                v_page_offset = 0;
                ++v_page_index;
                if(k_pos + 1 < n_kv)
                {
                    v_physical_page = arg.block_table[table_base + v_page_index];
                    v_base = static_cast<int64_t>(v_physical_page) * arg.v_stride_page +
                             static_cast<int64_t>(kv_head) * arg.v_stride_head;
                }
            }
            else
                v_base += arg.v_stride_token;
        }

        const float inv_sum = row_sum > 1.0e-10f ? 1.0f / row_sum : 0.0f;
        ck_tile::static_for<0, kValuesPerLane, 1>{}([&](auto i_num) {
            constexpr int i = decltype(i_num)::value;
            const int d = lane + i * kWaveSize;
            const DataType value = ck_tile::type_convert<DataType>(o_values[i] * inv_sum);
            arg.out[static_cast<int64_t>(global_q) * arg.out_stride_t +
                    static_cast<int64_t>(qhead) * arg.out_stride_h + d] =
                ck_tile::bit_cast<uint16_t>(value);
        });
    }
};
