// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_kkt_solve_pipeline.hpp"
#include "ck_tile/host.hpp"

struct GdnKktSolveFwdKernelArg
{
    const uint16_t* k;
    const float*    g_cum;
    const float*    beta;
    uint16_t*       A;
    int T, H, HV, K_dim, BT;
    int stride_k_t;
    int stride_g_t;
    int stride_A_t;
    bool use_exp2;
    const int64_t* cu_seqlens;
    const int64_t* chunk_indices;
    int num_chunks;
    bool is_varlen;
};

template <typename Policy, bool MaskKRows, int NumWarps = 4, bool SplitHeads = false>
struct GdnKktSolveFwdKernel
{
    using Kargs    = GdnKktSolveFwdKernelArg;
    using Pipeline = ck_tile::GdnKktSolvePipeline<Policy, MaskKRows, NumWarps>;

    CK_TILE_HOST static constexpr Kargs MakeKargs(
        const uint16_t* k,
        const float* g_cum,
        const float* beta,
        uint16_t* A,
        int T, int H, int HV, int K_dim,
        bool use_exp2,
        const int64_t* cu_seqlens,
        const int64_t* chunk_indices,
        int num_chunks,
        bool is_varlen)
    {
        constexpr int BT = Policy::kChunkSize;
        return Kargs{k, g_cum, beta, A, T, H, HV, K_dim, BT,
                     H * K_dim, HV, HV * 64, use_exp2, cu_seqlens,
                     chunk_indices, num_chunks, is_varlen};
    }

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(SplitHeads ? arg.HV : arg.H),
                    static_cast<unsigned int>(
                        arg.is_varlen ? arg.num_chunks
                                      : (arg.T + arg.BT - 1) / arg.BT),
                    1);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(NumWarps * 64);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        int i_chunk = blockIdx.y;
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
        if (tc >= token_end)
            return;

        __shared__ typename Pipeline::SharedStorage scratch;
        using DataType = typename Policy::DataType;
        auto* k_typed = reinterpret_cast<const DataType*>(arg.k);
        auto* A_typed = reinterpret_cast<DataType*>(arg.A);

        const int ratio = arg.HV / arg.H;
        const int ih = SplitHeads ? blockIdx.x / ratio : blockIdx.x;
        const int vh_offset = SplitHeads ? blockIdx.x % ratio : 0;
        Pipeline{}(
            k_typed, arg.g_cum, arg.beta, A_typed,
            tc, ih, vh_offset,
            token_end, arg.H, arg.HV,
            arg.stride_k_t, arg.stride_g_t, arg.stride_A_t,
            arg.use_exp2,
            scratch);
    }
};

template <typename Policy, int NumWarps, bool SplitHeads = false>
struct GdnKktSolveFwdInvoker
{
    static_assert(NumWarps == 1 || NumWarps == 2 || NumWarps == 4);

    static void Run(
        hipStream_t stream,
        const uint16_t* k,
        const float* g_cum,
        const float* beta,
        uint16_t* A,
        int T, int H, int HV, int K_dim,
        bool use_exp2,
        const int64_t* cu_seqlens = nullptr,
        const int64_t* chunk_indices = nullptr,
        int num_chunks = 0,
        bool is_varlen = false)
    {
        using Kernel = GdnKktSolveFwdKernel<Policy, true, NumWarps, SplitHeads>;
        auto kargs = Kernel::MakeKargs(k,
                                      g_cum,
                                      beta,
                                      A,
                                      T,
                                      H,
                                      HV,
                                      K_dim,
                                      use_exp2,
                                      cu_seqlens,
                                      chunk_indices,
                                      num_chunks,
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
