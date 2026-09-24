// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/jenga/pipeline/jenga_ck_tile_multi_64_pipeline.hpp"
#include "ck_tile/host.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>

struct Jenga64FusedKernelArgument
{
    const uint16_t* q_scaled;
    const uint16_t* k;
    const uint16_t* v;
    const int32_t* seqlens;
    const bool* block_mask;
    const int32_t* active_indices;
    const int32_t* active_counts;
    uint16_t* o;
    float* lse;
    int H;
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
    int text_block_start;
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

struct Jenga64FusedKernelLaunchConfig
{
    dim3 grid;
    hipStream_t stream;
};


template <int N_TILE, bool HasPadding, bool HasText, typename PipelinePolicy>
struct Jenga64FusedKernel
{
    using Kargs = Jenga64FusedKernelArgument;
    using Pipeline = Jenga64FusedQkSoftmaxPvPipeline<N_TILE, HasPadding, HasText, PipelinePolicy>;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const uint16_t* q_scaled,
                                                  const uint16_t* k,
                                                  const uint16_t* v,
                                                  const int32_t* seqlens,
                                                  const bool* block_mask,
                                                  const int32_t* active_indices,
                                                  const int32_t* active_counts,
                                                  uint16_t* o,
                                                  float* lse,
                                                  int H,
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
                                                  int text_block_start,
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
                     block_mask,
                     active_indices,
                     active_counts,
                     o,
                     lse,
                     H,
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
                     text_block_start,
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
                    static_cast<unsigned int>(arg.kv_stage_count),
                    static_cast<unsigned int>((arg.head_dim + N_TILE - 1) / N_TILE));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_HOST static constexpr bool IsSupportedArgument(const Kargs& arg)
    {
        return arg.head_dim <= N_TILE && arg.padded_dim >= N_TILE;
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int g = blockIdx.x;
        const int kv_stage = blockIdx.y;
        const int d_tile = blockIdx.z;
        const int out_d_base = d_tile * N_TILE;
        const int bh = g / arg.nqb;
        const int qb = g - bh * arg.nqb;
        const int b_idx = bh / arg.H;
        const int h_idx = bh - b_idx * arg.H;
        const int seqlen_val = arg.seqlens[b_idx];
        const int start_m = qb * kBlockM;

        const int32_t* active_q_base = arg.active_indices + static_cast<int64_t>(g) * arg.active_capacity;
        const int active_total = arg.active_counts[g];
        const int32_t* active_q = active_q_base;
        int active_count = active_total;
        if (arg.kv_stage_count > 1) {
            const int active_begin = active_total * kv_stage / arg.kv_stage_count;
            const int active_end = active_total * (kv_stage + 1) / arg.kv_stage_count;
            active_q = active_q_base + active_begin;
            active_count = active_end - active_begin;
        }
        const uint16_t* q_bh = arg.q_scaled + static_cast<int64_t>(bh) * arg.stride_qh;
        const uint16_t* k_bh = arg.k + static_cast<int64_t>(bh) * arg.stride_kh;
        const uint16_t* v_bh = arg.v + static_cast<int64_t>(bh) * arg.stride_vh;
        uint16_t* o_bh = arg.o + static_cast<int64_t>(kv_stage) * arg.stride_os +
                         static_cast<int64_t>(b_idx) * arg.stride_oz +
                         static_cast<int64_t>(h_idx) * arg.stride_oh;
        float* lse_stage = arg.lse + static_cast<int64_t>(kv_stage) * arg.stride_ls;

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
                   arg.text_block_start,
                   arg.text_amp,
                   arg.qk_scale,
                   arg.stride_om,
                   arg.stride_ok,
                   arg.stride_lz,
                   arg.stride_lm,
                   scratch);
    }
};

template <typename PipelinePolicy, int N_TILE, bool HasPadding, bool HasText>
struct Jenga64FusedKernelTraits
{
    using Argument = Jenga64FusedKernelArgument;
    using Kernel = Jenga64FusedKernel<N_TILE, HasPadding, HasText, PipelinePolicy>;

    static constexpr int kBlockSize = PipelinePolicy::kBlockSize;
    static constexpr int kNPerTile = N_TILE;
    static constexpr bool kHasPadding = HasPadding;
    static constexpr bool kHasText = HasText;

    static dim3 GridSize(const Argument& arg) { return Kernel::GridSize(arg); }

    static bool IsSupportedArgument(const Argument& arg) { return Kernel::IsSupportedArgument(arg); }

    static void Run(const Argument& arg, const Jenga64FusedKernelLaunchConfig& cfg)
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


struct Jenga64KvStageReduceKernelArgument
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
struct Jenga64KvStageReduceKernel
{
    using Kargs = Jenga64KvStageReduceKernelArgument;
    using DataType = typename PipelinePolicy::Problem::QDataType;

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
        if (arg.head_dim <= 128) {
            const int64_t rows = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q;
            return dim3(static_cast<unsigned int>((rows + 7) / 8));
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
        if (arg.head_dim <= 128) {
            __shared__ float row_m[8];
            __shared__ float row_denom[8];
            __shared__ float row_weight[8][ReduceKvStageCount];
            const int tid = static_cast<int>(threadIdx.x);
            const int row_group = tid >> 5;
            const int d_lane = tid & 31;
            const int64_t row_linear = static_cast<int64_t>(blockIdx.x) * 8 + row_group;
            const int64_t total_rows = static_cast<int64_t>(arg.B) * arg.H * arg.N_Q;
            const bool row_valid = row_group < 8 && row_linear < total_rows;

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
                    m = ck_tile::max(m, lse_s);
                }
                float denom = 0.0f;
                for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                    const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                        static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                    const float w = (lse_s == -ck_tile::numeric<float>::infinity()) ? 0.0f : jenga_fast_exp2(lse_s - m);
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
                for (int d = d_lane; d < arg.head_dim; d += 32) {
                    if (q_pos >= seqlen_val) {
                        o_bh[static_cast<int64_t>(q_pos) * arg.stride_om +
                             static_cast<int64_t>(d) * arg.stride_ok] = 0;
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
                        float acc = 0.0f;
                        for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                            const float w = row_weight[row_group][stage];
                            const auto o_s = ck_tile::bit_cast<DataType>(
                                arg.partial_o[static_cast<int64_t>(stage) * arg.partial_o_stage_stride + o_off]);
                            acc += ck_tile::type_convert<float>(o_s) * w;
                        }
                        o_bh[static_cast<int64_t>(q_pos) * arg.stride_om +
                             static_cast<int64_t>(d) * arg.stride_ok] =
                            ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(acc * inv));
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
                m = ck_tile::max(m, lse_s);
            }

            const int64_t o_off = static_cast<int64_t>(b_idx) * arg.H * arg.N_Q * arg.head_dim +
                                  static_cast<int64_t>(h_idx) * arg.N_Q * arg.head_dim +
                                  static_cast<int64_t>(q_pos) * arg.head_dim + d;
            float denom = 0.0f;
            float acc = 0.0f;
            for (int stage = 0; stage < ReduceKvStageCount; ++stage) {
                const float lse_s = arg.partial_lse[static_cast<int64_t>(stage) * arg.partial_lse_stage_stride +
                                                    static_cast<int64_t>(bh) * arg.N_Q + q_pos];
                const float w = (lse_s == -ck_tile::numeric<float>::infinity()) ? 0.0f : jenga_fast_exp2(lse_s - m);
                denom += w;
                const auto o_s = ck_tile::bit_cast<DataType>(
                    arg.partial_o[static_cast<int64_t>(stage) * arg.partial_o_stage_stride + o_off]);
                acc += ck_tile::type_convert<float>(o_s) * w;
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
struct Jenga64KvStageReduceKernelInvoker
{
    using Argument = Jenga64KvStageReduceKernelArgument;

    template <int ReduceKvStageCount>
    static void RunSpecialized(const Argument& arg, hipStream_t stream)
    {
        using Kernel = Jenga64KvStageReduceKernel<PipelinePolicy, ReduceKvStageCount>;
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
        case 2: RunSpecialized<2>(arg, stream); break;
        case 3: RunSpecialized<3>(arg, stream); break;
        case 4: RunSpecialized<4>(arg, stream); break;
        default: RunSpecialized<PipelinePolicy::kKvStageCount>(arg, stream); break;
        }
    }
};

template <typename PipelinePolicy>
struct Jenga64FusedKernelInvoker
{
    using Argument = Jenga64FusedKernelArgument;

    static dim3 MakeGrid(const Argument& arg, int out_tile)
    {
        if (out_tile == 128) {
            return Jenga64FusedKernelTraits<PipelinePolicy, 128, false, false>::GridSize(arg);
        }
        return Jenga64FusedKernelTraits<PipelinePolicy, 64, false, false>::GridSize(arg);
    }

    template <int N_TILE, bool HasPadding, bool HasText>
    static void RunSpecialized(const Argument& arg, hipStream_t stream)
    {
        using Traits = Jenga64FusedKernelTraits<PipelinePolicy, N_TILE, HasPadding, HasText>;
        if(!Traits::IsSupportedArgument(arg))
            throw std::invalid_argument("unsupported fused 64 kernel argument");
        Traits::Run(arg, Jenga64FusedKernelLaunchConfig{Traits::GridSize(arg), stream});
    }

    static void Run(const Argument& arg, int out_tile, bool has_padding, bool has_text, hipStream_t stream)
    {
        if (out_tile == 128) {
            if (has_padding) {
                has_text ? RunSpecialized<128, true, true>(arg, stream)
                         : RunSpecialized<128, true, false>(arg, stream);
            } else {
                has_text ? RunSpecialized<128, false, true>(arg, stream)
                         : RunSpecialized<128, false, false>(arg, stream);
            }
        } else {
            if (has_padding) {
                has_text ? RunSpecialized<64, true, true>(arg, stream)
                         : RunSpecialized<64, true, false>(arg, stream);
            } else {
                has_text ? RunSpecialized<64, false, true>(arg, stream)
                         : RunSpecialized<64, false, false>(arg, stream);
            }
        }
    }
};
