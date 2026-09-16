// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include <cstdint>
#include <type_traits>

namespace unified_attention::mmac_kargs {

// Give every disabled feature a distinct empty base. This is the same layout
// technique used by CK-Tile FMHA to keep inactive fields out of the kernarg.
template <int I>
struct Empty
{
};

struct Common
{
    const uint16_t* q_scaled;
    const uint16_t* k;
    const uint16_t* v;
    const int32_t* seqlens;
    const float* sinks;
    const float* alibi;
    const bool* block_mask;
    const int32_t* active_indices;
    const int32_t* active_counts;
    uint16_t* o;
    float* lse;
    int H;
    int H_KV;
    int N_Q;
    int N_KV;
    int head_dim;
    int padded_dim;
    int num_q_blocks;
    int num_blocks;
    int nqb;
    int active_capacity;
    int kv_stage_count;
    int tail_kv_stage_count;
    int tail_lse_stage_offset;
    int64_t stride_os;
    int64_t stride_ls;
    float qk_scale;
    bool causal;
    int sliding_window;
    bool alibi_sqrt;
    int64_t stride_qh;
    int64_t stride_kh;
    int64_t stride_vh;
    int64_t stride_oz;
    int64_t stride_oh;
    int64_t stride_om;
    int64_t stride_ok;
    int64_t stride_lz;
    int64_t stride_lm;
};

struct Text
{
    float text_amp;
    int text_block_start;
};

struct Softcap
{
    float softcap;
};

struct QqBias
{
    const float* qq_bias;
    int qq_bias_stride;
};

struct MmPrefix
{
    const int32_t* mm_ranges;
    int max_mm_ranges;
};

template <bool HasText, bool HasSoftcap, bool HasQqBias, bool HasMmPrefix>
struct Fused
    : Common,
      std::conditional_t<HasText, Text, Empty<0>>,
      std::conditional_t<HasSoftcap, Softcap, Empty<1>>,
      std::conditional_t<HasQqBias, QqBias, Empty<2>>,
      std::conditional_t<HasMmPrefix, MmPrefix, Empty<3>>
{
};

} // namespace unified_attention::mmac_kargs
