// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_vector_types.h>
#include "ck_tile/ops/unified_attention/kernel/unified_attention_mmac_kargs.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_mmac_pipeline.hpp"
#include "ck_tile/host.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

struct UnifiedAttentionFusedKernelLaunchConfig
{
    dim3 grid;
    hipStream_t stream;
};


template <int N_TILE,
          bool HasPadding,
          bool HasText,
          bool HasSoftcap,
          bool HasQqBias,
          typename PipelinePolicy,
          bool FuseD192Tail = false,
          bool HasMmPrefix = false,
          bool LsePerD = false>
struct UnifiedAttentionFusedKernel
{
    using Kargs = unified_attention::mmac_kargs::Fused<
        HasText, HasSoftcap, HasQqBias, HasMmPrefix>;
    static_assert(std::is_trivially_copyable_v<Kargs>);
    using Pipeline = UnifiedAttentionFusedQkSoftmaxPvPipeline<
        N_TILE, HasPadding, HasText, HasSoftcap, HasQqBias, PipelinePolicy,
        64, FuseD192Tail, HasMmPrefix, false, true, 1, LsePerD>;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const uint16_t* q_scaled,
                                                  const uint16_t* k,
                                                  const uint16_t* v,
                                                  const int32_t* seqlens,
                                                  const float* sinks,
                                                  const float* alibi,
                                                  const float* qq_bias,
                                                  const int32_t* mm_ranges,
                                                  const bool* block_mask,
                                                  const int32_t* active_indices,
                                                  const int32_t* active_counts,
                                                  uint16_t* o,
                                                  float* lse,
                                                  int H,
                                                  int H_KV,
                                                  int N_Q,
                                                  int N_KV,
                                                  int head_dim,
                                                  int padded_dim,
                                                  int num_q_blocks,
                                                  int num_blocks,
                                                  int nqb,
                                                  int active_capacity,
                                                  int kv_stage_count,
                                                  int tail_kv_stage_count,
                                                  int tail_lse_stage_offset,
                                                  int64_t stride_os,
                                                  int64_t stride_ls,
                                                  float text_amp,
                                                  float qk_scale,
                                                  float softcap,
                                                  int qq_bias_stride,
                                                  int text_block_start,
                                                  bool causal,
                                                  int max_mm_ranges,
                                                  int sliding_window,
                                                  bool alibi_sqrt,
                                                  int64_t stride_qh,
                                                  int64_t stride_kh,
                                                  int64_t stride_vh,
                                                  int64_t stride_oz,
                                                  int64_t stride_oh,
                                                  int64_t stride_om,
                                                  int64_t stride_ok,
                                                  int64_t stride_lz,
                                                  int64_t stride_lm)
    {
        Kargs args{};
        args.q_scaled = q_scaled;
        args.k = k;
        args.v = v;
        args.seqlens = seqlens;
        args.sinks = sinks;
        args.alibi = alibi;
        args.block_mask = block_mask;
        args.active_indices = active_indices;
        args.active_counts = active_counts;
        args.o = o;
        args.lse = lse;
        args.H = H;
        args.H_KV = H_KV;
        args.N_Q = N_Q;
        args.N_KV = N_KV;
        args.head_dim = head_dim;
        args.padded_dim = padded_dim;
        args.num_q_blocks = num_q_blocks;
        args.num_blocks = num_blocks;
        args.nqb = nqb;
        args.active_capacity = active_capacity;
        args.kv_stage_count = kv_stage_count;
        args.tail_kv_stage_count = tail_kv_stage_count;
        args.tail_lse_stage_offset = tail_lse_stage_offset;
        args.stride_os = stride_os;
        args.stride_ls = stride_ls;
        args.qk_scale = qk_scale;
        args.causal = causal;
        args.sliding_window = sliding_window;
        args.alibi_sqrt = alibi_sqrt;
        args.stride_qh = stride_qh;
        args.stride_kh = stride_kh;
        args.stride_vh = stride_vh;
        args.stride_oz = stride_oz;
        args.stride_oh = stride_oh;
        args.stride_om = stride_om;
        args.stride_ok = stride_ok;
        args.stride_lz = stride_lz;
        args.stride_lm = stride_lm;
        if constexpr(HasText)
        {
            args.text_amp = text_amp;
            args.text_block_start = text_block_start;
        }
        if constexpr(HasSoftcap)
            args.softcap = softcap;
        if constexpr(HasQqBias)
        {
            args.qq_bias = qq_bias;
            args.qq_bias_stride = qq_bias_stride;
        }
        if constexpr(HasMmPrefix)
        {
            args.mm_ranges = mm_ranges;
            args.max_mm_ranges = max_mm_ranges;
        }
        return args;
    }

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.num_q_blocks),
                    static_cast<unsigned int>(arg.kv_stage_count),
                    static_cast<unsigned int>(FuseD192Tail
                                                  ? 1
                                                  : (arg.head_dim + N_TILE - 1) / N_TILE));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_HOST static constexpr bool IsSupportedArgument(const Kargs& arg)
    {
        return arg.head_dim <= 2 * N_TILE && arg.padded_dim >= arg.head_dim;
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int g = blockIdx.x;
        const int kv_stage = blockIdx.y;
        const int d_tile = blockIdx.z;
        const bool use_tail_schedule = d_tile > 0 && arg.tail_kv_stage_count > 0;
        const int effective_stage_count =
            use_tail_schedule ? arg.tail_kv_stage_count : arg.kv_stage_count;
        if(kv_stage >= effective_stage_count) return;
        const int out_d_base = d_tile * N_TILE;
        const int bh = g / arg.nqb;
        const int qb = g - bh * arg.nqb;
        const int b_idx = bh / arg.H;
        const int h_idx = bh - b_idx * arg.H;
        const int kv_h_idx = h_idx * arg.H_KV / arg.H;
        const int kv_bh = b_idx * arg.H_KV + kv_h_idx;
        const int seqlen_val = arg.seqlens[b_idx];
        const int start_m = qb * kBlockM;

        const int32_t* active_q_base = arg.active_indices +
                                       static_cast<int64_t>(g) * arg.active_capacity;
        const int active_total = arg.active_counts[g];
        const int32_t* active_q = active_q_base;
        int active_count = active_total;
        if (effective_stage_count > 1) {
            const int active_begin = active_total * kv_stage / effective_stage_count;
            const int active_end = active_total * (kv_stage + 1) / effective_stage_count;
            active_q = active_q_base + active_begin;
            active_count = active_end - active_begin;
        }
        const uint16_t* q_bh = arg.q_scaled + static_cast<int64_t>(bh) * arg.stride_qh;
        const uint16_t* k_bh = arg.k + static_cast<int64_t>(kv_bh) * arg.stride_kh;
        const uint16_t* v_bh = arg.v + static_cast<int64_t>(kv_bh) * arg.stride_vh;
        uint16_t* o_bh = arg.o + static_cast<int64_t>(kv_stage) * arg.stride_os +
                         static_cast<int64_t>(b_idx) * arg.stride_oz +
                         static_cast<int64_t>(h_idx) * arg.stride_oh;
        const int lse_stage_index = use_tail_schedule
            ? arg.tail_lse_stage_offset + kv_stage
            : kv_stage;
        float* lse_stage =
            arg.lse + static_cast<int64_t>(lse_stage_index) * arg.stride_ls;

        // Causal pruning can leave an early query block with fewer active KV
        // blocks than split-K stages.  Materialize the softmax identity for an
        // empty stage so the stage reducer never consumes uninitialized/NaN
        // partial output.
        if (active_count == 0) {
            constexpr int output_columns = FuseD192Tail ? N_TILE + 64 : N_TILE;
            for (int linear = static_cast<int>(threadIdx.x);
                 linear < kBlockM * output_columns;
                 linear += static_cast<int>(blockDim.x)) {
                const int row = linear / output_columns;
                const int col = linear - row * output_columns;
                const int q_pos = start_m + row;
                const int d_pos = FuseD192Tail ? col : out_d_base + col;
                if (q_pos < arg.N_Q && d_pos < arg.head_dim) {
                    o_bh[static_cast<int64_t>(q_pos) * arg.stride_om +
                         static_cast<int64_t>(d_pos) * arg.stride_ok] = 0;
                }
            }
            if (d_tile == 0 || LsePerD) {
                for (int row = static_cast<int>(threadIdx.x);
                     row < kBlockM;
                     row += static_cast<int>(blockDim.x)) {
                    const int q_pos = start_m + row;
                    if (q_pos < arg.N_Q) {
                        lse_stage[static_cast<int64_t>(bh) * arg.stride_lz +
                                  static_cast<int64_t>(q_pos) * arg.stride_lm] =
                            -ck_tile::numeric<float>::infinity();
                    }
                }
            }
            return;
        }

        float text_amp = 0.0f;
        int text_block_start = 0;
        float softcap = 0.0f;
        const float* qq_bias = nullptr;
        int qq_bias_stride = 0;
        const int32_t* mm_ranges = nullptr;
        int max_mm_ranges = 0;
        if constexpr(HasText)
        {
            text_amp = arg.text_amp;
            text_block_start = arg.text_block_start;
        }
        if constexpr(HasSoftcap)
            softcap = arg.softcap;
        if constexpr(HasQqBias)
        {
            qq_bias = arg.qq_bias;
            qq_bias_stride = arg.qq_bias_stride;
        }
        if constexpr(HasMmPrefix)
        {
            mm_ranges = arg.mm_ranges;
            max_mm_ranges = arg.max_mm_ranges;
        }

        __shared__ typename Pipeline::SharedStorage scratch;
        Pipeline{}(q_bh,
                   k_bh,
                   v_bh,
                   active_q,
                   active_count,
                   o_bh,
                   lse_stage,
                   bh,
                   qb,
                   d_tile,
                   out_d_base,
                   start_m,
                   seqlen_val,
                   arg.N_Q,
                   arg.N_KV,
                   arg.head_dim,
                   arg.padded_dim,
                   arg.padded_dim,
                   text_block_start,
                   text_amp,
                   arg.qk_scale,
                   softcap,
                   arg.causal,
                   h_idx,
                   kv_stage == 0 ? arg.sinks : nullptr,
                   arg.alibi,
                   qq_bias,
                   qq_bias_stride,
                   mm_ranges,
                   max_mm_ranges,
                   arg.sliding_window,
                   arg.alibi_sqrt,
                   arg.stride_om,
                   arg.stride_ok,
                   arg.stride_lz,
                   arg.stride_lm,
                   scratch);
    }
};

struct UnifiedAttentionSwaBatchedKernelArgument
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const int32_t* cu_q;
    const int32_t* seqlens_k;
    const int32_t* active_indices;
    const int32_t* active_counts;
    const float* sinks;
    uint16_t* out;
    float* lse;
    int B;
    int H;
    int H_KV;
    int total_nq_pad;
    int total_nqb;
    int max_nq_pad;
    int max_nqb;
    int n_kv_pad;
    int head_dim;
    int padded_dim;
    int active_capacity;
    int sliding_window;
    float qk_scale;
    int64_t stride_q_m;
    int64_t stride_q_h;
    int64_t stride_out_m;
    int64_t stride_out_h;
    int64_t stride_out_k;
};

// Compact multi-sequence dense-prefill wrapper. Query blocks are packed per
// sequence but partial outputs use the original concatenated token numbering,
// allowing all sequences to share one five-stage main launch and one reducer.
struct UnifiedAttentionDenseBatchedKernelArgument
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const int32_t* cu_q;
    const int32_t* seqlens_k;
    const int32_t* active_indices;
    const int32_t* active_counts;
    uint16_t* partial_out;
    float* lse;
    int B;
    int H;
    int H_KV;
    int total_nq;
    int total_nq_pad;
    int total_nqb;
    int max_nq_pad;
    int max_nqb;
    int n_kv_pad;
    int head_dim;
    int padded_dim;
    int active_capacity;
    int kv_stage_count;
    float qk_scale;
    int64_t stride_q_m;
    int64_t stride_q_h;
};

struct UnifiedAttentionDenseBatchedFeatureKernelArgument
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const int32_t* cu_q;
    const int32_t* seqlens_k;
    const int32_t* active_indices;
    const int32_t* active_counts;
    const float* sinks;
    const float* alibi;
    const float* qq_bias;
    const int32_t* mm_ranges;
    uint16_t* partial_out;
    float* lse;
    int B;
    int H;
    int H_KV;
    int total_nq;
    int total_nq_pad;
    int total_nqb;
    int max_nq_pad;
    int max_nqb;
    int n_kv_pad;
    int head_dim;
    int padded_dim;
    int active_capacity;
    int kv_stage_count;
    int sliding_window;
    int max_mm_ranges;
    int qq_bias_stride;
    float qk_scale;
    float softcap;
    bool alibi_sqrt;
    int64_t stride_q_m;
    int64_t stride_q_h;
};

template <typename PipelinePolicy, bool RuntimeFeatures = false>
struct UnifiedAttentionDenseBatchedKernel
{
    using Kargs = std::conditional_t<RuntimeFeatures,
                                     UnifiedAttentionDenseBatchedFeatureKernelArgument,
                                     UnifiedAttentionDenseBatchedKernelArgument>;
    using Pipeline = UnifiedAttentionFusedQkSoftmaxPvPipeline<
        128, true, false, false, false, PipelinePolicy,
        64, true, false, false, true, 1, false, RuntimeFeatures>;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.H * arg.total_nqb),
                    static_cast<unsigned int>(arg.kv_stage_count), 1);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int g = static_cast<int>(blockIdx.x);
        const int h = g % arg.H;
        int compact_qb = g / arg.H;
        int b = arg.B - 1;
        int qb = 0;
        for(; b >= 0; --b)
        {
            const int q_len_b = arg.cu_q[b + 1] - arg.cu_q[b];
            const int nqb_b = (q_len_b + kBlockM - 1) / kBlockM;
            if(compact_qb < nqb_b)
            {
                qb = nqb_b - 1 - compact_qb;
                break;
            }
            compact_qb -= nqb_b;
        }

        const int q_begin = arg.cu_q[b];
        const int q_len = arg.cu_q[b + 1] - q_begin;
        const int start_m = qb * kBlockM;
        const int kv_stage = static_cast<int>(blockIdx.y);
        const int kv_h = h * arg.H_KV / arg.H;
        const int64_t kv_seq_stride =
            static_cast<int64_t>(arg.H_KV) * arg.n_kv_pad * arg.padded_dim;
        const int64_t kv_head_stride =
            static_cast<int64_t>(arg.n_kv_pad) * arg.padded_dim;
        const int64_t partial_stage_stride =
            static_cast<int64_t>(arg.H) * arg.total_nq * arg.head_dim;
        const int64_t lse_stage_stride =
            static_cast<int64_t>(arg.H) * arg.total_nq;

        const uint16_t* q_bh;
        int64_t q_row_stride;
        if constexpr(RuntimeFeatures)
        {
            int padded_q_begin = 0;
            for(int seq = 0; seq < b; ++seq)
            {
                const int seq_q_len = arg.cu_q[seq + 1] - arg.cu_q[seq];
                padded_q_begin += ((seq_q_len + kBlockM - 1) / kBlockM) * kBlockM;
            }
            q_bh = arg.q + static_cast<int64_t>(h) * arg.total_nq_pad * arg.padded_dim +
                   static_cast<int64_t>(padded_q_begin) * arg.padded_dim;
            q_row_stride = arg.padded_dim;
        }
        else
        {
            q_bh = arg.q + static_cast<int64_t>(q_begin) * arg.stride_q_m +
                   static_cast<int64_t>(h) * arg.stride_q_h;
            q_row_stride = arg.stride_q_m;
        }
        const uint16_t* k_bh = arg.k + static_cast<int64_t>(b) * kv_seq_stride +
                               static_cast<int64_t>(kv_h) * kv_head_stride;
        const uint16_t* v_bh = arg.v + static_cast<int64_t>(b) * kv_seq_stride +
                               static_cast<int64_t>(kv_h) * kv_head_stride;
        uint16_t* o_bh = arg.partial_out +
                         static_cast<int64_t>(kv_stage) * partial_stage_stride +
                         static_cast<int64_t>(h) * arg.total_nq * arg.head_dim +
                         static_cast<int64_t>(q_begin) * arg.head_dim;
        float* lse_stage = arg.lse + static_cast<int64_t>(kv_stage) * lse_stage_stride +
                           static_cast<int64_t>(q_begin);

        const int active_g = (b * arg.H + h) * arg.max_nqb + qb;
        const int32_t* active_base = arg.active_indices +
                                     static_cast<int64_t>(active_g) * arg.active_capacity;
        const int active_total = arg.active_counts[active_g];
        const int active_begin = active_total * kv_stage / arg.kv_stage_count;
        const int active_end = active_total * (kv_stage + 1) / arg.kv_stage_count;
        const int active_count = active_end - active_begin;

        if(active_count == 0)
        {
            for(int linear = static_cast<int>(threadIdx.x);
                linear < kBlockM * arg.head_dim;
                linear += static_cast<int>(blockDim.x))
            {
                const int row = linear / arg.head_dim;
                const int col = linear - row * arg.head_dim;
                if(start_m + row < q_len)
                    o_bh[static_cast<int64_t>(start_m + row) * arg.head_dim + col] = 0;
            }
            for(int row = static_cast<int>(threadIdx.x); row < kBlockM;
                row += static_cast<int>(blockDim.x))
                if(start_m + row < q_len)
                    lse_stage[static_cast<int64_t>(h) * arg.total_nq + start_m + row] =
                        -ck_tile::numeric<float>::infinity();
            return;
        }

        const float* sinks = nullptr;
        const float* alibi = nullptr;
        const float* qq_bias = nullptr;
        const int32_t* mm_ranges_b = nullptr;
        int qq_bias_stride = 0;
        int max_mm_ranges = 0;
        int sliding_window = 0;
        float softcap = 0.0f;
        bool alibi_sqrt = false;
        if constexpr(RuntimeFeatures)
        {
            sinks = arg.sinks;
            alibi = arg.alibi;
            qq_bias = arg.qq_bias;
            max_mm_ranges = arg.max_mm_ranges;
            qq_bias_stride = arg.qq_bias_stride;
            sliding_window = arg.sliding_window;
            softcap = arg.softcap;
            alibi_sqrt = arg.alibi_sqrt;
            mm_ranges_b = arg.mm_ranges == nullptr
                ? nullptr
                : arg.mm_ranges + static_cast<int64_t>(b) * max_mm_ranges * 2;
        }

        __shared__ typename Pipeline::SharedStorage scratch;
        Pipeline{}(q_bh, k_bh, v_bh, active_base + active_begin, active_count,
                   o_bh, lse_stage, h, qb, 0, 0, start_m, arg.seqlens_k[b],
                   q_len, arg.seqlens_k[b], arg.head_dim, arg.padded_dim,
                   q_row_stride,
                   0, 0.0f, arg.qk_scale, softcap, true, h,
                   active_begin == 0 ? sinks : nullptr, alibi, qq_bias,
                   qq_bias_stride, mm_ranges_b, max_mm_ranges,
                   sliding_window, alibi_sqrt,
                   arg.head_dim, 1, arg.total_nq, 1, scratch);
    }
};

template <typename PipelinePolicy, bool FuseD192Tail = false>
struct UnifiedAttentionSwaBatchedKernel
{
    using Kargs = UnifiedAttentionSwaBatchedKernelArgument;
    using Pipeline = UnifiedAttentionFusedQkSoftmaxPvPipeline<
        128, true, false, false, false, PipelinePolicy, 64, FuseD192Tail>;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.H * arg.total_nqb),
                    1,
                    FuseD192Tail ? 1 : 2);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int g = static_cast<int>(blockIdx.x);
        const int h = g % arg.H;
        int compact_qb = g / arg.H;
        // Schedule the longest/later causal query blocks first.  Besides
        // avoiding padded grid entries, this keeps the last wave of CTAs from
        // containing only the most expensive lower-triangular work.
        int b = arg.B - 1;
        int qb = 0;
        for(; b >= 0; --b)
        {
            const int q_len_b = arg.cu_q[b + 1] - arg.cu_q[b];
            const int nqb_b = (q_len_b + kBlockM - 1) / kBlockM;
            if(compact_qb < nqb_b)
            {
                qb = nqb_b - 1 - compact_qb;
                break;
            }
            compact_qb -= nqb_b;
        }
        const int q_begin = arg.cu_q[b];
        const int q_len = arg.cu_q[b + 1] - q_begin;
        const int start_m = qb * kBlockM;

        const int d_tile = static_cast<int>(blockIdx.z);
        const int out_d_base = d_tile * 128;
        const int kv_h = h * arg.H_KV / arg.H;
        const int64_t kv_seq_stride =
            static_cast<int64_t>(arg.H_KV) * arg.n_kv_pad * arg.padded_dim;
        const int64_t kv_head_stride =
            static_cast<int64_t>(arg.n_kv_pad) * arg.padded_dim;

        const uint16_t* q_bh = arg.q + static_cast<int64_t>(q_begin) * arg.stride_q_m +
                               static_cast<int64_t>(h) * arg.stride_q_h;
        const uint16_t* k_bh = arg.k + static_cast<int64_t>(b) * kv_seq_stride +
                               static_cast<int64_t>(kv_h) * kv_head_stride;
        const uint16_t* v_bh = arg.v + static_cast<int64_t>(b) * kv_seq_stride +
                               static_cast<int64_t>(kv_h) * kv_head_stride;
        uint16_t* o_bh = arg.out + static_cast<int64_t>(q_begin) * arg.stride_out_m +
                         static_cast<int64_t>(h) * arg.stride_out_h;
        const int active_g = (b * arg.H + h) * arg.max_nqb + qb;
        const int32_t* active = arg.active_indices +
                                static_cast<int64_t>(active_g) * arg.active_capacity;
        const int active_count = arg.active_counts[active_g];

        __shared__ typename Pipeline::SharedStorage scratch;
        Pipeline{}(q_bh,
                   k_bh,
                   v_bh,
                   active,
                   active_count,
                   o_bh,
                   arg.lse,
                   b * arg.H + h,
                   qb,
                   d_tile,
                   out_d_base,
                   start_m,
                   arg.seqlens_k[b],
                   q_len,
                   arg.seqlens_k[b],
                   arg.head_dim,
                   arg.padded_dim,
                   arg.stride_q_m,
                   0,
                   0.0f,
                   arg.qk_scale,
                   0.0f,
                   true,
                   h,
                   arg.sinks,
                   nullptr,
                   nullptr,
                   0,
                   nullptr,
                   0,
                   arg.sliding_window,
                   false,
                   arg.stride_out_m,
                   arg.stride_out_k,
                   arg.max_nq_pad,
                   1,
                   scratch);
    }
};

template <typename PipelinePolicy,
          int N_TILE,
          bool HasPadding,
          bool HasText,
          bool HasSoftcap = false,
          bool HasQqBias = false,
          bool HasMmPrefix = false>
struct UnifiedAttentionFusedKernelTraits
{
    using Kernel = UnifiedAttentionFusedKernel<
        N_TILE, HasPadding, HasText, HasSoftcap, HasQqBias, PipelinePolicy,
        false, HasMmPrefix>;
    using Argument = typename Kernel::Kargs;

    static constexpr int kBlockSize = PipelinePolicy::kBlockSize;
    static constexpr int kNPerTile = N_TILE;
    static constexpr bool kHasPadding = HasPadding;
    static constexpr bool kHasText = HasText;

    static dim3 GridSize(const Argument& arg) { return Kernel::GridSize(arg); }

    static bool IsSupportedArgument(const Argument& arg) { return Kernel::IsSupportedArgument(arg); }

    static void Run(const Argument& arg, const UnifiedAttentionFusedKernelLaunchConfig& cfg)
    {
        const auto grids = Kernel::GridSize(arg);
        constexpr auto blocks = Kernel::BlockSize();
        ck_tile::stream_config stream_cfg{cfg.stream};
        ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<blocks.x, PipelinePolicy::kLaunchMinBlocks>(
                Kernel{}, grids, blocks, 0, arg));
    }
};


struct UnifiedAttentionKvStageReduceKernelArgument
{
    const uint16_t* partial_o;
    const float* partial_lse;
    const int32_t* seqlens;
    uint16_t* o;
    float* lse;
    int B;
    int H;
    int N_Q;
    int head_dim;
    int kv_stage_count;
    int64_t partial_o_stage_stride;
    int64_t partial_lse_stage_stride;
    int64_t stride_oz;
    int64_t stride_oh;
    int64_t stride_om;
    int64_t stride_ok;
    int64_t stride_lz;
    int64_t stride_lm;
};

struct UnifiedAttentionD192AsymmetricReduceKernelArgument
{
    const uint16_t* partial_o;
    const float* partial_lse;
    uint16_t* o;
    float* lse;
    int H;
    int N_Q;
    int tail_kv_stage_count;
    int64_t partial_o_stage_stride;
    int64_t partial_lse_stage_stride;
    int64_t stride_oh;
    int64_t stride_om;
    int64_t stride_ok;
    int64_t stride_lz;
    int64_t stride_lm;
};

// The short D192 ALiBi cases launch five split-K stages for D[0:128], but
// only three stages for D[128:192].  The two output tiles therefore have
// different softmax normalizers.  Merge both schedules in one reduction
// launch while publishing the full-stage LSE as the result LSE.
template <typename PipelinePolicy>
struct UnifiedAttentionD192AsymmetricReduceKernel
{
    using Kargs = UnifiedAttentionD192AsymmetricReduceKernelArgument;
    using DataType = typename PipelinePolicy::Problem::QDataType;
    static constexpr int kMainStageCount = 5;
    static constexpr int kTailStageCapacity = 5;
    static constexpr int kTailLseStageOffset = 5;
    static constexpr int kHeadDim = 192;
    static constexpr int kRowsPerBlock = 8;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        const int64_t rows = static_cast<int64_t>(arg.H) * arg.N_Q;
        return dim3(static_cast<unsigned int>(
            (rows + kRowsPerBlock - 1) / kRowsPerBlock));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        __shared__ float row_main_denom[kRowsPerBlock];
        __shared__ float row_tail_denom[kRowsPerBlock];
        __shared__ float row_main_weight[kRowsPerBlock][kMainStageCount];
        __shared__ float row_tail_weight[kRowsPerBlock][kTailStageCapacity];

        const int tid = static_cast<int>(threadIdx.x);
        const int row_group = tid >> 5;
        const int d_lane = tid & 31;
        const int64_t row_linear =
            static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + row_group;
        const int64_t total_rows = static_cast<int64_t>(arg.H) * arg.N_Q;
        const bool row_valid = row_group < kRowsPerBlock && row_linear < total_rows;
        const int q_pos = row_valid ? static_cast<int>(row_linear % arg.N_Q) : 0;
        const int h_idx = row_valid ? static_cast<int>(row_linear / arg.N_Q) : 0;
        const int64_t lse_row = static_cast<int64_t>(h_idx) * arg.N_Q + q_pos;

        if(row_valid && d_lane == 0)
        {
            float main_m = -ck_tile::numeric<float>::infinity();
            for(int stage = 0; stage < kMainStageCount; ++stage)
            {
                const float x = arg.partial_lse[
                    static_cast<int64_t>(stage) * arg.partial_lse_stage_stride + lse_row];
                if(x > -ck_tile::numeric<float>::infinity() &&
                   x < ck_tile::numeric<float>::infinity())
                    main_m = ck_tile::max(main_m, x);
            }
            float main_denom = 0.0f;
            for(int stage = 0; stage < kMainStageCount; ++stage)
            {
                const float x = arg.partial_lse[
                    static_cast<int64_t>(stage) * arg.partial_lse_stage_stride + lse_row];
                const bool valid = x > -ck_tile::numeric<float>::infinity() &&
                                   x < ck_tile::numeric<float>::infinity();
                const float w = valid ? unified_attention_fast_exp2(x - main_m) : 0.0f;
                row_main_weight[row_group][stage] = w;
                main_denom += w;
            }
            row_main_denom[row_group] = main_denom;

            float tail_m = -ck_tile::numeric<float>::infinity();
            for(int stage = 0; stage < arg.tail_kv_stage_count; ++stage)
            {
                const float x = arg.partial_lse[
                    static_cast<int64_t>(kTailLseStageOffset + stage) *
                        arg.partial_lse_stage_stride + lse_row];
                if(x > -ck_tile::numeric<float>::infinity() &&
                   x < ck_tile::numeric<float>::infinity())
                    tail_m = ck_tile::max(tail_m, x);
            }
            float tail_denom = 0.0f;
            for(int stage = 0; stage < kTailStageCapacity; ++stage)
            {
                float w = 0.0f;
                if(stage < arg.tail_kv_stage_count)
                {
                    const float x = arg.partial_lse[
                        static_cast<int64_t>(kTailLseStageOffset + stage) *
                            arg.partial_lse_stage_stride + lse_row];
                    const bool valid = x > -ck_tile::numeric<float>::infinity() &&
                                       x < ck_tile::numeric<float>::infinity();
                    w = valid ? unified_attention_fast_exp2(x - tail_m) : 0.0f;
                }
                row_tail_weight[row_group][stage] = w;
                tail_denom += w;
            }
            row_tail_denom[row_group] = tail_denom;
            arg.lse[static_cast<int64_t>(h_idx) * arg.stride_lz +
                    static_cast<int64_t>(q_pos) * arg.stride_lm] =
                main_denom > 1.0e-10f
                    ? main_m + log2f(main_denom)
                    : -ck_tile::numeric<float>::infinity();
        }
        __syncthreads();

        if(!row_valid) return;
        for(int d = d_lane; d < kHeadDim; d += 32)
        {
            const bool tail = d >= 128;
            const int stage_count = tail ? arg.tail_kv_stage_count : kMainStageCount;
            const float denom = tail ? row_tail_denom[row_group] : row_main_denom[row_group];
            const float inv = denom > 1.0e-10f ? 1.0f / denom : 0.0f;
            const int64_t o_off = static_cast<int64_t>(h_idx) * arg.N_Q * kHeadDim +
                                  static_cast<int64_t>(q_pos) * kHeadDim + d;
            float acc = 0.0f;
            for(int stage = 0; stage < stage_count; ++stage)
            {
                const float w = tail ? row_tail_weight[row_group][stage]
                                     : row_main_weight[row_group][stage];
                if(w != 0.0f)
                {
                    const auto x = ck_tile::bit_cast<DataType>(
                        arg.partial_o[static_cast<int64_t>(stage) *
                                          arg.partial_o_stage_stride + o_off]);
                    acc += ck_tile::type_convert<float>(x) * w;
                }
            }
            arg.o[static_cast<int64_t>(q_pos) * arg.stride_om +
                  static_cast<int64_t>(h_idx) * arg.stride_oh +
                  static_cast<int64_t>(d) * arg.stride_ok] =
                ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(acc * inv));
        }
    }
};

template <typename PipelinePolicy, int ReduceKvStageCount = PipelinePolicy::kKvStageCount>
struct UnifiedAttentionKvStageReduceKernel
{
    using Kargs = UnifiedAttentionKvStageReduceKernelArgument;
    using DataType = typename PipelinePolicy::Problem::QDataType;
    static constexpr int kRowsPerBlock = 8;
    static constexpr int kVectorSize = 8;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const uint16_t* partial_o,
                                                  const float* partial_lse,
                                                  const int32_t* seqlens,
                                                  uint16_t* o,
                                                  float* lse,
                                                  int B,
                                                  int H,
                                                  int N_Q,
                                                  int head_dim,
                                                  int kv_stage_count,
                                                  int64_t partial_o_stage_stride,
                                                  int64_t partial_lse_stage_stride,
                                                  int64_t stride_oz,
                                                  int64_t stride_oh,
                                                  int64_t stride_om,
                                                  int64_t stride_ok,
                                                  int64_t stride_lz,
                                                  int64_t stride_lm)
    {
        return Kargs{partial_o, partial_lse, seqlens, o, lse,
                     B, H, N_Q, head_dim, kv_stage_count,
                     partial_o_stage_stride, partial_lse_stage_stride,
                     stride_oz, stride_oh, stride_om, stride_ok, stride_lz, stride_lm};
    }

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        if (arg.head_dim <= 256) {
            const int64_t rows = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q;
            return dim3(static_cast<unsigned int>(
                (rows + kRowsPerBlock - 1) / kRowsPerBlock));
        }
        const int64_t total = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q * arg.head_dim;
        return dim3(static_cast<unsigned int>((total + PipelinePolicy::kBlockSize - 1) /
                                              PipelinePolicy::kBlockSize));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        if (arg.head_dim <= 256) {
            __shared__ float row_m[kRowsPerBlock];
            __shared__ float row_denom[kRowsPerBlock];
            __shared__ float row_weight[kRowsPerBlock][ReduceKvStageCount];
            const int tid = static_cast<int>(threadIdx.x);
            const int row_group = tid >> 5;
            const int d_lane = tid & 31;
            const int64_t row_linear =
                static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + row_group;
            const int64_t total_rows = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q;
            const bool row_valid = row_group < kRowsPerBlock && row_linear < total_rows;

            int q_pos = 0;
            int h_idx = 0;
            int b_idx = 0;
            int bh = 0;
            int seqlen_val = 0;
            if (row_valid) {
                q_pos = row_linear % arg.N_Q;
                const int64_t bh_linear = row_linear / arg.N_Q;
                h_idx = bh_linear % arg.H;
                b_idx = bh_linear / arg.H;
                bh = b_idx * arg.H + h_idx;
                seqlen_val = arg.seqlens[b_idx];
            }

            if (row_valid && q_pos < seqlen_val && d_lane == 0) {
                float m = -ck_tile::numeric<float>::infinity();
                for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                    const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                        static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                    if (lse_s > -ck_tile::numeric<float>::infinity() &&
                        lse_s < ck_tile::numeric<float>::infinity()) {
                        m = ck_tile::max(m, lse_s);
                    }
                }
                float denom = 0.0f;
                for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                    const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                        static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                    const bool valid_lse =
                        lse_s > -ck_tile::numeric<float>::infinity() &&
                        lse_s < ck_tile::numeric<float>::infinity();
                    const float w = valid_lse ? unified_attention_fast_exp2(lse_s - m) : 0.0f;
                    row_weight[row_group][stage] = w;
                    denom += w;
                }
                row_m[row_group] = m;
                row_denom[row_group] = denom;
                arg.lse[static_cast<int64_t>(bh) * arg.stride_lz +
                        static_cast<int64_t>(q_pos) * arg.stride_lm] =
                    denom > 1.0e-10f ? (m + log2f(denom)) : -ck_tile::numeric<float>::infinity();
            }
            __syncthreads();

            if (row_valid) {
                uint16_t* o_bh = arg.o + static_cast<int64_t>(b_idx) * arg.stride_oz +
                                 static_cast<int64_t>(h_idx) * arg.stride_oh;
                for (int d = d_lane * kVectorSize; d < arg.head_dim;
                     d += PipelinePolicy::kBlockSize) {
                    if (q_pos >= seqlen_val) {
                        if (d + kVectorSize - 1 < arg.head_dim && arg.stride_ok == 1) {
                            reinterpret_cast<uint4*>(&o_bh[static_cast<int64_t>(q_pos) * arg.stride_om + d])[0] = make_uint4(0, 0, 0, 0);
                        } else {
                            for (int dd = d;
                                 dd < arg.head_dim && dd < d + kVectorSize;
                                 ++dd) {
                                o_bh[static_cast<int64_t>(q_pos) * arg.stride_om + dd * arg.stride_ok] = 0;
                            }
                        }
                        if (d == 0) {
                            arg.lse[static_cast<int64_t>(bh) * arg.stride_lz +
                                    static_cast<int64_t>(q_pos) * arg.stride_lm] =
                                -ck_tile::numeric<float>::infinity();
                        }
                    } else {
                        const float denom = row_denom[row_group];
                        const float inv = denom > 1.0e-10f ? 1.0f / denom : 0.0f;
                        const int64_t o_off = static_cast<int64_t>(b_idx) * arg.H * arg.N_Q * arg.head_dim +
                                              static_cast<int64_t>(h_idx) * arg.N_Q * arg.head_dim +
                                              static_cast<int64_t>(q_pos) * arg.head_dim + d;
                        if (d + kVectorSize - 1 < arg.head_dim && arg.stride_ok == 1) {
                            float acc[kVectorSize] = {0.0f};
                            for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                                const float w = row_weight[row_group][stage];
                                if (w != 0.0f) {
                                    uint4 packed = *reinterpret_cast<const uint4*>(&arg.partial_o[static_cast<int64_t>(stage) * arg.partial_o_stage_stride + o_off]);
                                    uint16_t* packed_ptr = reinterpret_cast<uint16_t*>(&packed);
                                    #pragma unroll
                                    for(int i = 0; i < kVectorSize; ++i) {
                                        const auto o_s = ck_tile::bit_cast<DataType>(packed_ptr[i]);
                                        acc[i] += ck_tile::type_convert<float>(o_s) * w;
                                    }
                                }
                            }
                            uint4 out_packed;
                            uint16_t* out_ptr = reinterpret_cast<uint16_t*>(&out_packed);
                            #pragma unroll
                            for(int i = 0; i < kVectorSize; ++i) {
                                out_ptr[i] = ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(acc[i] * inv));
                            }
                            *reinterpret_cast<uint4*>(&o_bh[static_cast<int64_t>(q_pos) * arg.stride_om + d]) = out_packed;
                        } else {
                            for (int dd = d;
                                 dd < arg.head_dim && dd < d + kVectorSize;
                                 ++dd) {
                                float acc_tail = 0.0f;
                                for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                                    const float w = row_weight[row_group][stage];
                                    if (w != 0.0f) {
                                        const auto o_s = ck_tile::bit_cast<DataType>(
                                            arg.partial_o[static_cast<int64_t>(stage) * arg.partial_o_stage_stride + o_off + (dd - d)]);
                                        acc_tail += ck_tile::type_convert<float>(o_s) * w;
                                    }
                                }
                                o_bh[static_cast<int64_t>(q_pos) * arg.stride_om + dd * arg.stride_ok] =
                                    ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(acc_tail * inv));
                            }
                        }
                    }
                }
            }
            return;
        }

        int64_t total = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q * arg.head_dim;
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        for (; idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
            int d = idx % arg.head_dim;
            int q_pos = (idx / arg.head_dim) % arg.N_Q;
            int h_idx = (idx / (static_cast<int64_t>(arg.head_dim) * arg.N_Q)) % arg.H;
            int b_idx = idx / (static_cast<int64_t>(arg.head_dim) * arg.N_Q * arg.H);
            int bh = b_idx * arg.H + h_idx;
            int seqlen_val = arg.seqlens[b_idx];

            uint16_t* o_bh = arg.o + static_cast<int64_t>(b_idx) * arg.stride_oz +
                             static_cast<int64_t>(h_idx) * arg.stride_oh;
            if (q_pos >= seqlen_val) {
                o_bh[static_cast<int64_t>(q_pos) * arg.stride_om +
                     static_cast<int64_t>(d) * arg.stride_ok] = 0;
                if (d == 0) {
                    arg.lse[static_cast<int64_t>(bh) * arg.stride_lz +
                            static_cast<int64_t>(q_pos) * arg.stride_lm] =
                        -ck_tile::numeric<float>::infinity();
                }
                continue;
            }

            float m = -ck_tile::numeric<float>::infinity();
            for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                    static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                if (lse_s > -ck_tile::numeric<float>::infinity() &&
                    lse_s < ck_tile::numeric<float>::infinity()) {
                    m = ck_tile::max(m, lse_s);
                }
            }

            const int64_t o_off = static_cast<int64_t>(b_idx) * arg.H * arg.N_Q * arg.head_dim +
                                  static_cast<int64_t>(h_idx) * arg.N_Q * arg.head_dim +
                                  static_cast<int64_t>(q_pos) * arg.head_dim + d;
            float denom = 0.0f;
            float acc = 0.0f;
            for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                    static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                const bool valid_lse =
                    lse_s > -ck_tile::numeric<float>::infinity() &&
                    lse_s < ck_tile::numeric<float>::infinity();
                const float w = valid_lse ? unified_attention_fast_exp2(lse_s - m) : 0.0f;
                denom += w;
                if (w != 0.0f) {
                    const auto o_s = ck_tile::bit_cast<DataType>(
                        arg.partial_o[static_cast<int64_t>(stage) * arg.partial_o_stage_stride + o_off]);
                    acc += ck_tile::type_convert<float>(o_s) * w;
                }
            }
            const float inv = denom > 1.0e-10f ? 1.0f / denom : 0.0f;
            o_bh[static_cast<int64_t>(q_pos) * arg.stride_om +
                 static_cast<int64_t>(d) * arg.stride_ok] =
                ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(acc * inv));
            if (d == 0) {
                arg.lse[static_cast<int64_t>(bh) * arg.stride_lz +
                        static_cast<int64_t>(q_pos) * arg.stride_lm] =
                    denom > 1.0e-10f ? (m + log2f(denom)) : -ck_tile::numeric<float>::infinity();
            }
        }
    }
};

template <typename PipelinePolicy>
struct UnifiedAttentionKvStageReduceKernelInvoker
{
    using Argument = UnifiedAttentionKvStageReduceKernelArgument;

    template <int ReduceKvStageCount>
    static void RunSpecialized(const Argument& arg, hipStream_t stream)
    {
        using Kernel = UnifiedAttentionKvStageReduceKernel<PipelinePolicy, ReduceKvStageCount>;
        const auto grids = Kernel::GridSize(arg);
        constexpr auto blocks = Kernel::BlockSize();
        ck_tile::stream_config stream_cfg{stream};
        ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<blocks.x, PipelinePolicy::kLaunchMinBlocks>(
                Kernel{}, grids, blocks, 0, arg));
    }

    static void Run(const Argument& arg, hipStream_t stream)
    {
        switch (arg.kv_stage_count) {
        case 1: RunSpecialized<1>(arg, stream); break;
        case 2: RunSpecialized<2>(arg, stream); break;
        case 3: RunSpecialized<3>(arg, stream); break;
        case 4: RunSpecialized<4>(arg, stream); break;
        case 5: RunSpecialized<5>(arg, stream); break;
        case 6: RunSpecialized<6>(arg, stream); break;
        case 8: RunSpecialized<8>(arg, stream); break;
        case 10: RunSpecialized<10>(arg, stream); break;
        default: RunSpecialized<PipelinePolicy::kKvStageCount>(arg, stream); break;
        }
    }
};
