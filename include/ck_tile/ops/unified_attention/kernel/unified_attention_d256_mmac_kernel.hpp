// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/unified_attention/pipeline/unified_attention_d256_mmac_pipeline.hpp"
#include "ck_tile/host.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

static constexpr int kMaxBatchSeqs = 8;

struct UnifiedAttentionD256FusedKernelArgument
{
    const uint16_t* q_scaled;
    const uint16_t* k;
    const uint16_t* v;
    const int32_t* seqlens;
    const float* sinks;
    const float* alibi;
    const float* qq_bias;
    const int32_t* mm_ranges;
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
    int64_t stride_os;
    int64_t stride_ls;
    float text_amp;
    float qk_scale;
    float softcap;
    int qq_bias_stride;
    int text_block_start;
    bool causal;
    int max_mm_ranges;
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

struct UnifiedAttentionD256FusedBatchKernelArgument
{
    const uint16_t* q_bases[kMaxBatchSeqs];
    const uint16_t* k_bases[kMaxBatchSeqs];
    const uint16_t* v_bases[kMaxBatchSeqs];
    uint16_t* o_bases[kMaxBatchSeqs];
    float* lse_bases[kMaxBatchSeqs];
    const int32_t* active_indices_bases[kMaxBatchSeqs];
    const int32_t* active_counts_bases[kMaxBatchSeqs];
    const int32_t* mm_ranges_bases[kMaxBatchSeqs];
    const int32_t* block_table_bases[kMaxBatchSeqs];
    const int32_t* q_offsets;
    const int32_t* n_kvs_device;
    const float* sinks;
    const float* alibi;
    const float* qq_bias;
    int num_seqs;
    int total_nqb;
    int total_q_tasks;
    int split_qb_begin;
    int H;
    int H_KV;
    int head_dim;
    int padded_dim;
    int active_capacity;
    int kv_stage_count;
    int max_mm_ranges;
    int text_block_start;
    int sliding_window;
    bool causal;
    bool alibi_sqrt;
    float text_amp;
    float qk_scale;
    float softcap;
    int qq_bias_stride;
    int64_t stride_qh[kMaxBatchSeqs];
    int64_t stride_kh[kMaxBatchSeqs];
    int64_t stride_vh[kMaxBatchSeqs];
    int64_t stride_os[kMaxBatchSeqs];
    int64_t stride_oz[kMaxBatchSeqs];
    int64_t stride_oh[kMaxBatchSeqs];
    int64_t stride_om[kMaxBatchSeqs];
    int64_t stride_ok[kMaxBatchSeqs];
    int64_t stride_ls[kMaxBatchSeqs];
    int64_t stride_lz[kMaxBatchSeqs];
    int64_t stride_lm[kMaxBatchSeqs];
    int64_t k_page_stride;
    int64_t k_token_stride;
    int64_t k_head_stride;
    int64_t v_page_stride;
    int64_t v_token_stride;
    int64_t v_head_stride;
    int64_t q_token_stride;
    int64_t q_head_stride;
};

struct UnifiedAttentionD256FusedKernelLaunchConfig
{
    dim3 grid;
    hipStream_t stream;
};


template <int N_TILE,
          bool HasPadding,
          bool HasText,
          bool HasSoftcap,
          bool HasQqBias,
          bool DirectCausal,
          bool DirectPage,
          typename PipelinePolicy>
struct UnifiedAttentionD256FusedKernel
{
    using Kargs = UnifiedAttentionD256FusedKernelArgument;
    using Pipeline = UnifiedAttentionD256FusedQkSoftmaxPvPipeline<
        N_TILE, HasPadding, HasText, HasSoftcap, HasQqBias, DirectCausal,
        DirectPage, true, UnifiedAttentionD256AlibiMode::Runtime, PipelinePolicy>;

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
        return Kargs{q_scaled,
                     k,
                     v,
                     seqlens,
                     sinks,
                     alibi,
                     qq_bias,
                     mm_ranges,
                     block_mask,
                     active_indices,
                     active_counts,
                     o,
                     lse,
                     H,
                     H_KV,
                     N_Q,
                     N_KV,
                     head_dim,
                     padded_dim,
                     num_q_blocks,
                     num_blocks,
                     nqb,
                     active_capacity,
                     kv_stage_count,
                     stride_os,
                     stride_ls,
                     text_amp,
                     qk_scale,
                     softcap,
                     qq_bias_stride,
                     text_block_start,
                     causal,
                     max_mm_ranges,
                     sliding_window,
                     alibi_sqrt,
                     stride_qh,
                     stride_kh,
                     stride_vh,
                     stride_oz,
                     stride_oh,
                     stride_om,
                     stride_ok,
                     stride_lz,
                     stride_lm};
    }

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.num_q_blocks),
                    static_cast<unsigned int>(arg.kv_stage_count));
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
        const int bh = g / arg.nqb;
        const int qb = g - bh * arg.nqb;
        const int b_idx = bh / arg.H;
        const int h_idx = bh - b_idx * arg.H;
        const int kv_h_idx = h_idx * arg.H_KV / arg.H;
        const int kv_bh = b_idx * arg.H_KV + kv_h_idx;
        const int seqlen_val = arg.seqlens[b_idx];
        const int start_m = qb * kBlockM;

        const int32_t* active_q_base = arg.active_indices + static_cast<int64_t>(g) * arg.active_capacity;
        const int active_total = arg.active_counts[g];
        const int32_t* active_q = active_q_base;
        int active_count = active_total;
        if (arg.kv_stage_count > 1) {
            // Put remainder blocks in earlier stages. The HIP grid dispatches
            // lower stage indices first, so late stages become no heavier than
            // early ones and do not form a long launch tail.
            const int active_begin =
                (active_total * kv_stage + arg.kv_stage_count - 1) /
                arg.kv_stage_count;
            const int active_end =
                (active_total * (kv_stage + 1) + arg.kv_stage_count - 1) /
                arg.kv_stage_count;
            active_q = active_q_base + active_begin;
            active_count = active_end - active_begin;
        }
        const uint16_t* q_bh = arg.q_scaled + static_cast<int64_t>(bh) * arg.stride_qh;
        const uint16_t* k_bh = arg.k + static_cast<int64_t>(kv_bh) * arg.stride_kh;
        const uint16_t* v_bh = arg.v + static_cast<int64_t>(kv_bh) * arg.stride_vh;
        uint16_t* o_bh = arg.o + static_cast<int64_t>(kv_stage) * arg.stride_os +
                         static_cast<int64_t>(b_idx) * arg.stride_oz +
                         static_cast<int64_t>(h_idx) * arg.stride_oh;
        float* lse_stage = arg.lse + static_cast<int64_t>(kv_stage) * arg.stride_ls;

        // Causal pruning can leave an early query block with fewer active KV
        // blocks than split-K stages.  Materialize the softmax identity for an
        // empty stage so the stage reducer never consumes uninitialized/NaN
        // partial output.
        if (active_count == 0) {
            ck_tile::static_for<0, 2, 1>{}([&](auto half_num) {
                constexpr int half = decltype(half_num)::value;
                const int out_d_base = half * N_TILE;
                for (int linear = static_cast<int>(threadIdx.x);
                     linear < kBlockM * N_TILE;
                     linear += static_cast<int>(blockDim.x)) {
                    const int row = linear / N_TILE;
                    const int col = linear - row * N_TILE;
                    const int q_pos = start_m + row;
                    const int d_pos = out_d_base + col;
                    if (q_pos < arg.N_Q && d_pos < arg.head_dim) {
                        o_bh[static_cast<int64_t>(q_pos) * arg.stride_om +
                             static_cast<int64_t>(d_pos) * arg.stride_ok] = 0;
                    }
                }
            });
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
            return;
        }

        __shared__ typename Pipeline::SharedStorage scratch;
        Pipeline{}(q_bh,
                   k_bh,
                   v_bh,
                   active_q,
                   active_count,
                   0,
                   o_bh,
                   lse_stage,
                   bh,
                   qb,
                   start_m,
                   seqlen_val,
                   arg.N_Q,
                   arg.N_KV,
                   arg.head_dim,
                   arg.padded_dim,
                   arg.padded_dim,
                   arg.padded_dim,
                   arg.padded_dim,
                   arg.text_block_start,
                   arg.text_amp,
                   arg.qk_scale,
                   arg.softcap,
                   arg.causal,
                   h_idx,
                   kv_stage == 0 ? arg.sinks : nullptr,
                   arg.alibi,
                   arg.qq_bias,
                   arg.qq_bias_stride,
                   arg.mm_ranges,
                   arg.max_mm_ranges,
                   arg.sliding_window,
                   arg.alibi_sqrt,
                   arg.stride_om,
                   arg.stride_ok,
                   arg.stride_lz,
                   arg.stride_lm,
                   scratch);
    }
};

// Feature-minimal Kargs for the plain causal, single-page, single-stage path.
// Like FmhaFwdKernel's conditional Kargs, this type is selected by the kernel
// traits at compile time; fields for inactive features do not enter kernarg.
struct UnifiedAttentionD256DirectPageBatchKernelArgument
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    uint16_t* o;
    const int32_t* q_offsets;
    const int32_t* n_kvs;
    const int32_t* block_table;
    int num_seqs;
    int total_nqb;
    int h_q;
    int h_kv;
    int head_dim;
    int padded_dim;
    int64_t q_token_stride;
    int64_t q_head_stride;
    int64_t k_page_stride;
    int64_t k_token_stride;
    int64_t k_head_stride;
    int64_t v_page_stride;
    int64_t v_token_stride;
    int64_t v_head_stride;
    int64_t o_token_stride;
    int64_t o_head_stride;
    int64_t o_dim_stride;
    int64_t block_table_stride;
    float qk_scale;
    const float* alibi;
};

template <int N_TILE,
          bool HasPadding,
          bool HasText,
          bool HasSoftcap,
          bool HasQqBias,
          bool DirectCausal,
          bool DirectPage,
          bool StoreLse,
          UnifiedAttentionD256AlibiMode AlibiMode,
          bool CompactKargs,
          typename PipelinePolicy>
struct UnifiedAttentionD256FusedBatchKernel
{
    using Kargs = std::conditional_t<CompactKargs,
                                     UnifiedAttentionD256DirectPageBatchKernelArgument,
                                     UnifiedAttentionD256FusedBatchKernelArgument>;
    using Pipeline = UnifiedAttentionD256FusedQkSoftmaxPvPipeline<
        N_TILE, HasPadding, HasText, HasSoftcap, HasQqBias, DirectCausal,
        DirectPage, StoreLse, AlibiMode, PipelinePolicy>;
    static_assert(!CompactKargs ||
                  (N_TILE == 128 && HasPadding && !HasText && !HasSoftcap &&
                   !HasQqBias && DirectCausal && DirectPage && !StoreLse));
    static_assert(!CompactKargs ||
                  (AlibiMode == UnifiedAttentionD256AlibiMode::None ||
                   AlibiMode == UnifiedAttentionD256AlibiMode::Linear ||
                   AlibiMode == UnifiedAttentionD256AlibiMode::Sqrt));

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        if constexpr(CompactKargs)
        {
            return dim3(static_cast<unsigned int>(arg.total_nqb * arg.h_q));
        }
        else
        {
            if(arg.split_qb_begin >= 0)
                return dim3(static_cast<unsigned int>(arg.total_q_tasks * arg.H));
            return dim3(static_cast<unsigned int>(arg.total_nqb * arg.H),
                        static_cast<unsigned int>(arg.kv_stage_count));
        }
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        if constexpr(CompactKargs)
        {
            const int g = static_cast<int>(blockIdx.x);
            const int schedule_idx = g / arg.h_q;
            const int h_idx = g - schedule_idx * arg.h_q;
            const int global_qb = arg.total_nqb - 1 - schedule_idx;

            int seq = 0;
            while(seq + 1 < arg.num_seqs &&
                  global_qb >= arg.q_offsets[seq + 1] / kBlockM + seq + 1)
                ++seq;
            const int q_begin = arg.q_offsets[seq];
            const int q_end = arg.q_offsets[seq + 1];
            const int n_q = q_end - q_begin;
            const int qb = global_qb - (q_begin / kBlockM + seq);
            const int nqb = (n_q + kBlockM - 1) / kBlockM;
            if(qb < 0 || qb >= nqb)
                return;

            const int n_kv = arg.n_kvs[seq];
            const int start_m = qb * kBlockM;
            const int context = n_kv - n_q;
            const int last_q = ck_tile::min(start_m + kBlockM - 1, n_q - 1);
            const int active_count = (context + last_q) / kBlockN + 1;
            const int kv_h_idx = h_idx * arg.h_kv / arg.h_q;
            const int physical_page =
                arg.block_table[static_cast<int64_t>(seq) * arg.block_table_stride];

            const uint16_t* q_bh =
                arg.q + static_cast<int64_t>(q_begin) * arg.q_token_stride +
                static_cast<int64_t>(h_idx) * arg.q_head_stride;
            const uint16_t* k_bh =
                arg.k + static_cast<int64_t>(physical_page) * arg.k_page_stride +
                static_cast<int64_t>(kv_h_idx) * arg.k_head_stride;
            const uint16_t* v_bh =
                arg.v + static_cast<int64_t>(physical_page) * arg.v_page_stride +
                static_cast<int64_t>(kv_h_idx) * arg.v_head_stride;
            uint16_t* o_bh =
                arg.o + static_cast<int64_t>(q_begin) * arg.o_token_stride +
                static_cast<int64_t>(h_idx) * arg.o_head_stride;

            __shared__ typename Pipeline::SharedStorage scratch;
            Pipeline{}(q_bh,
                       k_bh,
                       v_bh,
                       nullptr,
                       active_count,
                       0,
                       o_bh,
                       nullptr,
                       h_idx,
                       qb,
                       start_m,
                       n_kv,
                       n_q,
                       n_kv,
                       arg.head_dim,
                       arg.padded_dim,
                       arg.q_token_stride,
                       arg.k_token_stride,
                       arg.v_token_stride,
                       0,
                       0.0f,
                       arg.qk_scale,
                       0.0f,
                       true,
                       h_idx,
                       nullptr,
                       AlibiMode == UnifiedAttentionD256AlibiMode::None
                           ? nullptr
                           : arg.alibi,
                       nullptr,
                       0,
                       nullptr,
                       0,
                       0,
                       false,
                       arg.o_token_stride,
                       arg.o_dim_stride,
                       0,
                       0,
                       scratch);
        }
        else
        {
        const int g = blockIdx.x;
        const int schedule_idx = g / arg.H;
        const int h_idx = g - schedule_idx * arg.H;
        int kv_stage = static_cast<int>(blockIdx.y);
        int global_qb = arg.total_nqb - 1 - schedule_idx;

        // Selective split-K flattens only the expensive suffix into the x
        // dimension. Unsplit query blocks therefore launch one CTA rather
        // than an empty second stage. Keep the reverse order so expensive
        // causal blocks are still dispatched first.
        if(arg.split_qb_begin >= 0)
        {
            const int task = arg.total_q_tasks - 1 - schedule_idx;
            if(task < arg.split_qb_begin)
            {
                global_qb = task;
                kv_stage = 0;
            }
            else
            {
                const int split_task = task - arg.split_qb_begin;
                global_qb = arg.split_qb_begin + split_task / 2;
                kv_stage = split_task & 1;
            }
        }

        int seq = 0;
        while(seq + 1 < arg.num_seqs &&
              global_qb >= arg.q_offsets[seq + 1] / kBlockM + seq + 1)
            ++seq;
        const int q_begin = arg.q_offsets[seq];
        const int q_end = arg.q_offsets[seq + 1];
        const int n_q = q_end - q_begin;
        const int qb = global_qb - (q_begin / kBlockM + seq);
        const int nqb = (n_q + kBlockM - 1) / kBlockM;
        if(qb < 0 || qb >= nqb)
            return;
        const int active_row = h_idx * nqb + qb;
        const int32_t* active_q_base = nullptr;
        int active_total = 0;
        if constexpr(DirectCausal)
        {
            const int context = arg.n_kvs_device[seq] - n_q;
            const int last_q = ck_tile::min(qb * kBlockM + kBlockM - 1, n_q - 1);
            active_total = (context + last_q) / kBlockN + 1;
        }
        else
        {
            active_q_base = arg.active_indices_bases[seq] +
                static_cast<int64_t>(active_row) * arg.active_capacity;
            active_total = arg.active_counts_bases[seq][active_row];
        }
        const int32_t* active_q = active_q_base;
        int active_count = active_total;
        int active_begin = 0;
        const int my_kv_stages =
            arg.split_qb_begin >= 0 && qb < arg.split_qb_begin
                ? 1
                : arg.kv_stage_count;
        if(my_kv_stages > 1)
        {
            // Front-load remainder blocks so late-dispatched stages cannot
            // become the critical launch tail.
            active_begin =
                (active_total * kv_stage + my_kv_stages - 1) / my_kv_stages;
            const int active_end =
                (active_total * (kv_stage + 1) + my_kv_stages - 1) /
                my_kv_stages;
            if constexpr(!DirectCausal)
                active_q = active_q_base + active_begin;
            active_count = active_end - active_begin;
        }

        const int kv_h_idx = h_idx * arg.H_KV / arg.H;
        const uint16_t* q_bh = DirectPage
            ? arg.q_bases[seq] + static_cast<int64_t>(q_begin) * arg.q_token_stride +
                  static_cast<int64_t>(h_idx) * arg.q_head_stride
            : arg.q_bases[seq] + static_cast<int64_t>(h_idx) * arg.stride_qh[seq];
        const int physical_page = DirectPage ? arg.block_table_bases[seq][0] : 0;
        const uint16_t* k_bh = DirectPage
            ? arg.k_bases[seq] + static_cast<int64_t>(physical_page) * arg.k_page_stride +
                  static_cast<int64_t>(kv_h_idx) * arg.k_head_stride
            : arg.k_bases[seq] + static_cast<int64_t>(kv_h_idx) * arg.stride_kh[seq];
        const uint16_t* v_bh = DirectPage
            ? arg.v_bases[seq] + static_cast<int64_t>(physical_page) * arg.v_page_stride +
                  static_cast<int64_t>(kv_h_idx) * arg.v_head_stride
            : arg.v_bases[seq] + static_cast<int64_t>(kv_h_idx) * arg.stride_vh[seq];
        uint16_t* o_bh =
            arg.o_bases[seq] +
            static_cast<int64_t>(kv_stage) * arg.stride_os[seq] +
            static_cast<int64_t>(q_begin) * arg.stride_om[seq] +
            static_cast<int64_t>(h_idx) * arg.stride_oh[seq];
        float* lse_stage =
            arg.lse_bases[seq] + static_cast<int64_t>(kv_stage) * arg.stride_ls[seq];
        const int start_m = qb * kBlockM;

        // The reducer keeps its original fixed-stage fast path. For an
        // unsplit query block, stage 0 publishes the missing stage's softmax
        // identity; its partial O is never read because this LSE has zero
        // weight.
        if(arg.split_qb_begin >= 0 && qb < arg.split_qb_begin)
        {
            float* unused_lse_stage = arg.lse_bases[seq] + arg.stride_ls[seq];
            for(int row = static_cast<int>(threadIdx.x);
                row < kBlockM;
                row += static_cast<int>(blockDim.x))
            {
                const int q_pos = start_m + row;
                if(q_pos < n_q)
                {
                    unused_lse_stage[
                        static_cast<int64_t>(h_idx) * arg.stride_lz[seq] +
                        static_cast<int64_t>(q_pos) * arg.stride_lm[seq]] =
                        -ck_tile::numeric<float>::infinity();
                }
            }
        }

        const int seqlen_val = arg.n_kvs_device[seq];

        if(kv_stage >= my_kv_stages || active_count == 0)
        {
            ck_tile::static_for<0, 2, 1>{}([&](auto half_num) {
                constexpr int half = decltype(half_num)::value;
                const int out_d_base = half * N_TILE;
                for(int linear = static_cast<int>(threadIdx.x);
                    linear < kBlockM * N_TILE;
                    linear += static_cast<int>(blockDim.x))
                {
                    const int row = linear / N_TILE;
                    const int col = linear - row * N_TILE;
                    const int q_pos = start_m + row;
                    const int d_pos = out_d_base + col;
                    if(q_pos < n_q && d_pos < arg.head_dim)
                    {
                        o_bh[static_cast<int64_t>(q_pos) * arg.stride_om[seq] +
                             static_cast<int64_t>(d_pos) * arg.stride_ok[seq]] = 0;
                    }
                }
            });
            for(int row = static_cast<int>(threadIdx.x);
                row < kBlockM;
                row += static_cast<int>(blockDim.x))
            {
                const int q_pos = start_m + row;
                if(q_pos < n_q)
                {
                    lse_stage[static_cast<int64_t>(h_idx) * arg.stride_lz[seq] +
                              static_cast<int64_t>(q_pos) * arg.stride_lm[seq]] =
                        -ck_tile::numeric<float>::infinity();
                }
            }
            return;
        }

        __shared__ typename Pipeline::SharedStorage scratch;
        Pipeline{}(q_bh,
                   k_bh,
                   v_bh,
                   active_q,
                   active_count,
                   active_begin,
                   o_bh,
                   lse_stage,
                   h_idx,
                   qb,
                   start_m,
                   seqlen_val,
                   n_q,
                   seqlen_val,
                   arg.head_dim,
                   arg.padded_dim,
                   DirectPage ? arg.q_token_stride : arg.padded_dim,
                   DirectPage ? arg.k_token_stride : arg.padded_dim,
                   DirectPage ? arg.v_token_stride : arg.padded_dim,
                   arg.text_block_start,
                   arg.text_amp,
                   arg.qk_scale,
                   arg.softcap,
                   arg.causal,
                   h_idx,
                   kv_stage == 0 ? arg.sinks : nullptr,
                   arg.alibi,
                   arg.qq_bias,
                   arg.qq_bias_stride,
                   arg.mm_ranges_bases[seq],
                   arg.max_mm_ranges,
                   arg.sliding_window,
                   arg.alibi_sqrt,
                   arg.stride_om[seq],
                   arg.stride_ok[seq],
                   arg.stride_lz[seq],
                   arg.stride_lm[seq],
                   scratch);
        }
    }
};

template <typename PipelinePolicy,
          int N_TILE,
          bool HasPadding,
          bool HasText,
          bool HasSoftcap = false,
          bool HasQqBias = false,
          bool DirectCausal = false,
          bool DirectPage = false>
struct UnifiedAttentionD256FusedKernelTraits
{
    using Argument = UnifiedAttentionD256FusedKernelArgument;
    using Kernel = UnifiedAttentionD256FusedKernel<
        N_TILE, HasPadding, HasText, HasSoftcap, HasQqBias, DirectCausal,
        DirectPage, PipelinePolicy>;

    static constexpr int kBlockSize = PipelinePolicy::kBlockSize;
    static constexpr int kNPerTile = N_TILE;
    static constexpr bool kHasPadding = HasPadding;
    static constexpr bool kHasText = HasText;

    static dim3 GridSize(const Argument& arg) { return Kernel::GridSize(arg); }

    static bool IsSupportedArgument(const Argument& arg) { return Kernel::IsSupportedArgument(arg); }

    static void Run(const Argument& arg, const UnifiedAttentionD256FusedKernelLaunchConfig& cfg)
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


struct UnifiedAttentionD256KvStageReduceKernelArgument
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

template <typename PipelinePolicy, int ReduceKvStageCount = PipelinePolicy::kKvStageCount>
struct UnifiedAttentionD256KvStageReduceKernel
{
    using Kargs = UnifiedAttentionD256KvStageReduceKernelArgument;
    using DataType = typename PipelinePolicy::Problem::QDataType;
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
        if (CanUseVectorRowReduce(arg)) {
            const int64_t rows = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q;
            return dim3(static_cast<unsigned int>((rows + 15) / 16));
        }
        const int64_t total = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q * arg.head_dim;
        return dim3(static_cast<unsigned int>((total + PipelinePolicy::kBlockSize - 1) /
                                              PipelinePolicy::kBlockSize));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_HOST_DEVICE static constexpr bool CanUseVectorRowReduce(const Kargs& arg)
    {
        return arg.head_dim % kVectorSize == 0 &&
               arg.partial_o_stage_stride % kVectorSize == 0 &&
               arg.stride_ok == 1 && arg.stride_om % kVectorSize == 0 &&
               arg.stride_oh % kVectorSize == 0 &&
               arg.stride_oz % kVectorSize == 0;
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        if (CanUseVectorRowReduce(arg)) {
            static_assert(PipelinePolicy::kBlockSize == 256,
                          "vector row reduction requires a 256-thread block");
            const int tid = static_cast<int>(threadIdx.x);
            const int row_group = tid >> 4;
            const int d_lane = tid & 15;
            const int64_t row_linear =
                static_cast<int64_t>(blockIdx.x) * 16 + row_group;
            const int64_t total_rows = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q;
            const bool row_valid = row_linear < total_rows;

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

            const bool valid_query = row_valid && q_pos < seqlen_val;
            float m = -ck_tile::numeric<float>::infinity();
            if (valid_query && d_lane == 0) {
                ck_tile::static_for<0, ReduceKvStageCount, 1>{}([&](auto stage_num) {
                    constexpr int stage = decltype(stage_num)::value;
                    const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                        static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                    if (lse_s > -ck_tile::numeric<float>::infinity() &&
                        lse_s < ck_tile::numeric<float>::infinity()) {
                        m = ck_tile::max(m, lse_s);
                    }
                });
            }
            m = __shfl(m, 0, 16);

            float weights[ReduceKvStageCount] = {};
            float denom = 0.0f;
            if (valid_query && d_lane == 0) {
                ck_tile::static_for<0, ReduceKvStageCount, 1>{}([&](auto stage_num) {
                    constexpr int stage = decltype(stage_num)::value;
                    const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                        static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                    const bool valid_lse =
                        lse_s > -ck_tile::numeric<float>::infinity() &&
                        lse_s < ck_tile::numeric<float>::infinity();
                    const float w = valid_lse ? unified_attention_d256_fast_exp2(lse_s - m) : 0.0f;
                    weights[stage] = w;
                    denom += w;
                });
                arg.lse[static_cast<int64_t>(bh) * arg.stride_lz +
                        static_cast<int64_t>(q_pos) * arg.stride_lm] =
                    denom > 1.0e-10f ? (m + log2f(denom)) : -ck_tile::numeric<float>::infinity();
            }
            denom = __shfl(denom, 0, 16);
            ck_tile::static_for<0, ReduceKvStageCount, 1>{}([&](auto stage_num) {
                constexpr int stage = decltype(stage_num)::value;
                weights[stage] = __shfl(weights[stage], 0, 16);
            });

            if (row_valid) {
                union Vector8 {
                    float4 packed;
                    DataType values[kVectorSize];
                };
                uint16_t* o_bh = arg.o + static_cast<int64_t>(b_idx) * arg.stride_oz +
                                 static_cast<int64_t>(h_idx) * arg.stride_oh;
                const int64_t partial_row_offset =
                    (static_cast<int64_t>(b_idx) * arg.H + h_idx) *
                        arg.N_Q * arg.head_dim +
                    static_cast<int64_t>(q_pos) * arg.head_dim;
                ck_tile::static_for<0, 2, 1>{}([&](auto d_chunk_num) {
                    const int d =
                        (d_lane + decltype(d_chunk_num)::value * 16) * kVectorSize;
                    if (q_pos >= seqlen_val) {
                        Vector8 zero{};
                        reinterpret_cast<float4*>(
                            o_bh + static_cast<int64_t>(q_pos) * arg.stride_om)
                            [d / kVectorSize] =
                            zero.packed;
                        if (d == 0) {
                            arg.lse[static_cast<int64_t>(bh) * arg.stride_lz +
                                    static_cast<int64_t>(q_pos) * arg.stride_lm] =
                                -ck_tile::numeric<float>::infinity();
                        }
                    } else {
                        const float inv = denom > 1.0e-10f ? 1.0f / denom : 0.0f;
                        float acc[kVectorSize] = {};
                        ck_tile::static_for<0, ReduceKvStageCount, 1>{}([&](auto stage_num) {
                            constexpr int stage = decltype(stage_num)::value;
                            const float w = weights[stage];
                            if (w != 0.0f) {
                                Vector8 stage_o;
                                stage_o.packed = reinterpret_cast<const float4*>(arg.partial_o)[
                                    (static_cast<int64_t>(stage) * arg.partial_o_stage_stride +
                                     partial_row_offset + d) /
                                    kVectorSize];
                                ck_tile::static_for<0, kVectorSize, 1>{}([&](auto i_num) {
                                    constexpr int i = decltype(i_num)::value;
                                    acc[i] += ck_tile::type_convert<float>(stage_o.values[i]) * w;
                                });
                            }
                        });
                        Vector8 result;
                        ck_tile::static_for<0, kVectorSize, 1>{}([&](auto i_num) {
                            constexpr int i = decltype(i_num)::value;
                            result.values[i] = ck_tile::type_convert<DataType>(acc[i] * inv);
                        });
                        reinterpret_cast<float4*>(
                            o_bh + static_cast<int64_t>(q_pos) * arg.stride_om)
                            [d / kVectorSize] =
                            result.packed;
                    }
                });
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
            ck_tile::static_for<0, ReduceKvStageCount, 1>{}([&](auto stage_num) {
                constexpr int stage = decltype(stage_num)::value;
                const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                    static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                if (lse_s > -ck_tile::numeric<float>::infinity() &&
                    lse_s < ck_tile::numeric<float>::infinity()) {
                    m = ck_tile::max(m, lse_s);
                }
            });

            const int64_t o_off = static_cast<int64_t>(b_idx) * arg.H * arg.N_Q * arg.head_dim +
                                  static_cast<int64_t>(h_idx) * arg.N_Q * arg.head_dim +
                                  static_cast<int64_t>(q_pos) * arg.head_dim + d;
            float denom = 0.0f;
            float acc = 0.0f;
            ck_tile::static_for<0, ReduceKvStageCount, 1>{}([&](auto stage_num) {
                constexpr int stage = decltype(stage_num)::value;
                const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                    static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                const bool valid_lse =
                    lse_s > -ck_tile::numeric<float>::infinity() &&
                    lse_s < ck_tile::numeric<float>::infinity();
                const float w = valid_lse ? unified_attention_d256_fast_exp2(lse_s - m) : 0.0f;
                denom += w;
                if (w != 0.0f) {
                    const auto o_s = ck_tile::bit_cast<DataType>(
                        arg.partial_o[static_cast<int64_t>(stage) * arg.partial_o_stage_stride + o_off]);
                    acc += ck_tile::type_convert<float>(o_s) * w;
                }
            });
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
struct UnifiedAttentionD256KvStageReduceKernelInvoker
{
    using Argument = UnifiedAttentionD256KvStageReduceKernelArgument;

    template <int ReduceKvStageCount>
    static void RunSpecialized(const Argument& arg, hipStream_t stream)
    {
        using Kernel = UnifiedAttentionD256KvStageReduceKernel<PipelinePolicy, ReduceKvStageCount>;
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
        default: RunSpecialized<PipelinePolicy::kKvStageCount>(arg, stream); break;
        }
    }
};
