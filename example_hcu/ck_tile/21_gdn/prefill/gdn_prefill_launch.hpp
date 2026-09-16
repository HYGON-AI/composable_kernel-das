// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace gdn_example {

inline constexpr int kSupportedHeadDim = 128;
inline constexpr int kChunkSize = 64;

struct PrefillArguments
{
    const void* q;
    const void* k;
    const void* v;
    const void* q_input;
    const void* k_input;
    void* q_norm;
    void* k_norm;
    float* q_rstd;
    float* k_rstd;
    const float* g;
    const float* gk;
    const float* a_log;
    const float* dt_bias;
    const float* beta;
    const float* beta_input;
    float* beta_processed;
    const float* initial_state;
    const int64_t* cu_seqlens;
    int64_t* chunk_offsets;
    int64_t* chunk_indices;
    float* g_cum;
    uint16_t* a;
    uint16_t* w;
    uint16_t* u;
    uint16_t* h;
    uint16_t* v_new;
    float* final_state;
    float* state_input_workspace;
    float* state_final_workspace;
    uint16_t* state_cp_group_a;
    uint16_t* state_cp_group_b;
    float* state_cp_group_start;
    int state_cp_groups;
    uint16_t* output;
    float* cp_hm;
    float* cp_ag_hm;
    float* cp_state;
    int cp_world_size;
    int cp_rank;
    int t;
    int h_qk;
    int h_v;
    int head_dim;
    int num_sequences;
    int num_chunks;
    float scale;
    bool output_final_state;
    bool is_varlen;
    bool use_qk_l2norm;
    bool cp_context;
    bool is_bf16;
    bool state_has_initial_state;
    bool state_store_final_state;
    bool state_save_new_value;
    bool state_use_g;
    bool state_use_gk;
    bool state_use_exp2;
    bool state_transpose_state;
    bool state_is_varlen;
    bool gate_in_kernel;
    bool has_dt_bias;
    bool beta_sigmoid_in_kernel;
};

void launch_beta_sigmoid(const PrefillArguments&, hipStream_t);
void launch_l2norm_bf16(const PrefillArguments&, hipStream_t);
void launch_l2norm_fp16(const PrefillArguments&, hipStream_t);
void launch_prepare_chunk_indices(const PrefillArguments&, hipStream_t);
void launch_cumsum(const PrefillArguments&, hipStream_t);
void launch_kkt_bf16(const PrefillArguments&, hipStream_t);
void launch_kkt_fp16(const PrefillArguments&, hipStream_t);
void launch_recompute_bf16(const PrefillArguments&, hipStream_t);
void launch_recompute_fp16(const PrefillArguments&, hipStream_t);
void launch_state_bf16(const PrefillArguments&, hipStream_t);
void launch_state_fp16(const PrefillArguments&, hipStream_t);
int select_state_cp_groups(int total_tokens, int value_heads);
void launch_output_bf16(const PrefillArguments&, hipStream_t);
void launch_output_fp16(const PrefillArguments&, hipStream_t);

void launch_prefill_bf16(const PrefillArguments&, hipStream_t);
void launch_prefill_fp16(const PrefillArguments&, hipStream_t);

bool launch_cp_context(const PrefillArguments&, hipStream_t = nullptr);

} // namespace gdn_example
