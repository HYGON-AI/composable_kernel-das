// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_cumsum_pipeline.hpp"
#include "ck_tile/host.hpp"

struct GdnCumsumFwdKernelArg
{
    const void* g;
    const float* A_log;
    const float* dt_bias;
    float* o;
    int T;
    int H;
    int S;
    int BT;
    float scale;
    bool is_vector;
    bool use_gate;
    bool has_bias;
    const int64_t* cu_seqlens;
    const int64_t* chunk_indices;
    int num_chunks;
    bool is_varlen;
};

template <typename Policy>
struct GdnCumsumFwdKernel
{
    using Kargs = GdnCumsumFwdKernelArg;
    using Pipeline = ck_tile::GdnCumsumPipeline<Policy>;

    CK_TILE_HOST static constexpr Kargs MakeKargs(
        const void* g,
        const float* A_log,
        const float* dt_bias,
        float* o,
        int T,
        int H,
        int S,
        float scale,
        bool is_vector,
        bool use_gate,
        bool has_bias,
        const int64_t* cu_seqlens,
        const int64_t* chunk_indices,
        int num_chunks,
        bool is_varlen)
    {
        constexpr int BT = Policy::kChunkSize;
        return Kargs{g, A_log, dt_bias, o, T, H, S, BT, scale,
                     is_vector, use_gate, has_bias, cu_seqlens,
                     chunk_indices, num_chunks, is_varlen};
    }

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(
                        arg.is_varlen ? arg.num_chunks
                                      : (arg.T + arg.BT - 1) / arg.BT),
                    static_cast<unsigned int>(arg.H),
                    static_cast<unsigned int>((arg.S + Policy::kSBlock - 1) /
                                              Policy::kSBlock));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(Policy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int i_chunk = blockIdx.x;
        const int ih = blockIdx.y;
        const int is = blockIdx.z * Policy::kSBlock;
        int tc = i_chunk * arg.BT;
        int token_end = arg.T;
        if(arg.is_varlen)
        {
            const int sequence =
                static_cast<int>(arg.chunk_indices[i_chunk * 2]);
            const int local_chunk =
                static_cast<int>(arg.chunk_indices[i_chunk * 2 + 1]);
            tc = static_cast<int>(arg.cu_seqlens[sequence]) +
                 local_chunk * arg.BT;
            token_end = static_cast<int>(arg.cu_seqlens[sequence + 1]);
        }
        if(tc >= token_end)
            return;

        using DataType = typename Policy::DataType;
        Pipeline{}(
            reinterpret_cast<const DataType*>(arg.g),
            arg.A_log,
            arg.dt_bias,
            arg.o,
            tc,
            ih,
            is,
            token_end,
            arg.H,
            arg.S,
            arg.scale,
            arg.is_vector,
            arg.use_gate,
            arg.has_bias);
    }
};

template <typename Policy>
struct GdnCumsumFwdInvoker
{
    using Kernel = GdnCumsumFwdKernel<Policy>;

    static void Run(hipStream_t stream,
                    const void* g,
                    const float* A_log,
                    const float* dt_bias,
                    float* o,
                    int T,
                    int H,
                    int S,
                    float scale,
                    bool is_vector,
                    bool use_gate,
                    bool has_bias,
                    const int64_t* cu_seqlens = nullptr,
                    const int64_t* chunk_indices = nullptr,
                    int num_chunks = 0,
                    bool is_varlen = false)
    {
        auto kargs = Kernel::MakeKargs(g, A_log, dt_bias, o, T, H, S,
                                       scale, is_vector, use_gate, has_bias,
                                       cu_seqlens, chunk_indices, num_chunks,
                                       is_varlen);
        auto grid = Kernel::GridSize(kargs);
        constexpr auto block = Kernel::BlockSize();

        ck_tile::stream_config stream_cfg{stream};
        ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                Kernel{}, grid, block, 0, kargs));
    }
};
