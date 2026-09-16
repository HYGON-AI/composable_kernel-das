// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_recompute_w_u_pipeline.hpp"
#include "ck_tile/host.hpp"

struct GdnRecomputeWUFwdKernelArg
{
    const uint16_t* k;
    const uint16_t* v;
    const float*    beta;
    const uint16_t* A;
    const float*    g_cum;
    uint16_t*       w;
    uint16_t*       u;
    int T, H, HV, K_dim, V_dim, BT;
    int stride_k_t;
    int stride_v_t;
    int stride_g_t;
    int stride_A_t;
    int stride_w_t;
    int stride_u_t;
    bool use_exp2;
    const int64_t* cu_seqlens;
    const int64_t* chunk_indices;
    int num_chunks;
    bool is_varlen;
};
template <typename Policy, bool Split4, bool FixedTp1 = false, bool MaskedTail = false>
struct GdnRecomputeWUFwdKernel
{
    using Kargs    = GdnRecomputeWUFwdKernelArg;
    using Pipeline = ck_tile::GdnRecomputeWUPipeline<Policy>;

    CK_TILE_HOST static constexpr Kargs MakeKargs(
        const uint16_t* k,
        const uint16_t* v,
        const float* beta,
        const uint16_t* A,
        const float* g_cum,
        uint16_t* w,
        uint16_t* u,
        int T, int H, int HV, int K_dim, int V_dim,
        bool use_exp2,
        const int64_t* cu_seqlens = nullptr,
        const int64_t* chunk_indices = nullptr,
        int num_chunks = 0,
        bool is_varlen = false)
    {
        constexpr int BT = Policy::kChunkSize;
        return Kargs{k, v, beta, A, g_cum, w, u,
                     T, H, HV, K_dim, V_dim, BT,
                     H * K_dim,
                     HV * V_dim,
                     HV,
                     HV * BT,
                     HV * K_dim,
                     HV * V_dim,
                     use_exp2,
                     cu_seqlens,
                     chunk_indices,
                     num_chunks,
                     is_varlen};
    }

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(FixedTp1 ? 64 : arg.HV),
                    static_cast<unsigned int>((arg.is_varlen
                                                   ? arg.num_chunks
                                                   : (arg.T + arg.BT - 1) /
                                                         arg.BT) *
                                              (Split4 ? 4 : 1)),
                    1);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(Policy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        int i_chunk = blockIdx.y;
        int task = -1;
        if constexpr(Split4)
        {
            task = i_chunk & 3;
            i_chunk >>= 2;
        }
        int ivh = blockIdx.x;
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

        int ih;
        if constexpr(FixedTp1)
        {
            ih = ivh >> 2;
        }
        else
        {
            const int ratio_hv = arg.HV / arg.H;
            ih = ivh / ratio_hv;
        }

        __shared__ typename Pipeline::SharedStorage scratch;
        using DataType = typename Policy::DataType;
        // The masked MMAC/LDS path below is BF16-specific. FP16 partial
        // chunks are handled by GdnRecomputeWUFp16TailKernel instead.
        if constexpr(MaskedTail && std::is_same_v<DataType, ck_tile::half_t>)
        {
            if(token_end - tc < arg.BT)
                return;
        }
        auto* k_typed = reinterpret_cast<const DataType*>(arg.k);
        auto* v_typed = reinterpret_cast<const DataType*>(arg.v);
        auto* A_typed = reinterpret_cast<const DataType*>(arg.A);
        auto* w_typed = reinterpret_cast<DataType*>(arg.w);
        auto* u_typed = reinterpret_cast<DataType*>(arg.u);

        if constexpr(Split4)
        {
            if constexpr(MaskedTail)
            {
                const int valid_rows = token_end - tc;
                if(valid_rows < arg.BT)
                {
                    Pipeline{}.run_split4_task_masked(
                        k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                        w_typed, u_typed,
                        tc, ih, ivh,
                        arg.stride_k_t, arg.stride_v_t, arg.stride_g_t, arg.stride_A_t,
                        arg.stride_w_t, arg.stride_u_t,
                        valid_rows,
                        arg.use_exp2,
                        task,
                        scratch);
                }
                else
                {
                    Pipeline{}.run_split4_task(
                        k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                        w_typed, u_typed,
                        tc, ih, ivh,
                        arg.stride_k_t, arg.stride_v_t, arg.stride_g_t, arg.stride_A_t,
                        arg.stride_w_t, arg.stride_u_t,
                        arg.use_exp2,
                        task,
                        scratch);
                }
            }
            else
            {
                Pipeline{}.run_split4_task(
                    k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                    w_typed, u_typed,
                    tc, ih, ivh,
                    arg.stride_k_t, arg.stride_v_t, arg.stride_g_t, arg.stride_A_t,
                    arg.stride_w_t, arg.stride_u_t,
                    arg.use_exp2,
                    task,
                    scratch);
            }
        }
        else
        {
            const int valid_rows = token_end - tc;
            if constexpr(MaskedTail)
            {
                if(valid_rows < arg.BT)
                {
                    if constexpr(FixedTp1)
                    {
                        Pipeline{}.run_masked(
                            k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                            w_typed, u_typed,
                            tc, ih, ivh,
                            2048, 8192, 64, 4096, 8192, 8192,
                            valid_rows,
                            arg.use_exp2,
                            scratch);
                    }
                    else
                    {
                        Pipeline{}.run_masked(
                            k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                            w_typed, u_typed,
                            tc, ih, ivh,
                            arg.stride_k_t, arg.stride_v_t, arg.stride_g_t,
                            arg.stride_A_t, arg.stride_w_t, arg.stride_u_t,
                            valid_rows,
                            arg.use_exp2,
                            scratch);
                    }
                }
                else if constexpr(FixedTp1)
                {
                    Pipeline{}(
                        k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                        w_typed, u_typed,
                        tc, ih, ivh,
                        0, 0, 0,
                        2048, 8192, 64, 4096, 8192, 8192,
                        arg.use_exp2,
                        scratch);
                }
                else
                {
                    Pipeline{}(
                        k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                        w_typed, u_typed,
                        tc, ih, ivh,
                        arg.T, arg.H, arg.HV,
                        arg.stride_k_t, arg.stride_v_t, arg.stride_g_t,
                        arg.stride_A_t, arg.stride_w_t, arg.stride_u_t,
                        arg.use_exp2,
                        scratch);
                }
            }
            else if constexpr(FixedTp1)
            {
                Pipeline{}(
                    k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                    w_typed, u_typed,
                    tc, ih, ivh,
                    0, 0, 0,
                    2048, 8192, 64, 4096, 8192, 8192,
                    arg.use_exp2,
                    scratch);
            }
            else
            {
                Pipeline{}(
                    k_typed, v_typed, arg.beta, A_typed, arg.g_cum,
                    w_typed, u_typed,
                    tc, ih, ivh,
                    arg.T, arg.H, arg.HV,
                    arg.stride_k_t, arg.stride_v_t, arg.stride_g_t, arg.stride_A_t,
                    arg.stride_w_t, arg.stride_u_t,
                    arg.use_exp2,
                    scratch);
            }
        }
    }
};

// The optimized masked LDS path is BF16-specific. Repair only the partial
// FP16 sequence chunks with a simple independent calculation; full chunks
// remain on the optimized pipeline above.
struct GdnRecomputeWUFp16TailKernel
{
    using Kargs = GdnRecomputeWUFwdKernelArg;

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int ivh = blockIdx.x;
        const int global_chunk = blockIdx.y;
        int tc = global_chunk * arg.BT;
        int token_end = arg.T;
        if(arg.is_varlen)
        {
            const int sequence =
                static_cast<int>(arg.chunk_indices[global_chunk * 2]);
            const int local_chunk =
                static_cast<int>(arg.chunk_indices[global_chunk * 2 + 1]);
            tc = static_cast<int>(arg.cu_seqlens[sequence]) + local_chunk * arg.BT;
            token_end = static_cast<int>(arg.cu_seqlens[sequence + 1]);
        }
        const int valid_rows = token_end - tc;
        if(valid_rows <= 0 || valid_rows >= arg.BT)
            return;

        const int ih = ivh / (arg.HV / arg.H);
        const auto* k = reinterpret_cast<const ck_tile::half_t*>(arg.k);
        const auto* v = reinterpret_cast<const ck_tile::half_t*>(arg.v);
        const auto* a = reinterpret_cast<const ck_tile::half_t*>(arg.A);
        auto* w = reinterpret_cast<ck_tile::half_t*>(arg.w);
        auto* u = reinterpret_cast<ck_tile::half_t*>(arg.u);

        for(int index = threadIdx.x; index < valid_rows * 128;
            index += blockDim.x)
        {
            const int row = index / 128;
            const int col = index % 128;
            float w_acc = 0.0f;
            float u_acc = 0.0f;
            for(int j = 0; j < valid_rows; ++j)
            {
                const size_t a_offset =
                    (static_cast<size_t>(tc + row) * arg.HV + ivh) * 64 + j;
                const float a_value = ck_tile::type_convert<float>(a[a_offset]);
                const float beta =
                    arg.beta[static_cast<size_t>(tc + j) * arg.HV + ivh];
                const float gate = arg.use_exp2
                    ? exp2f(arg.g_cum[static_cast<size_t>(tc + j) * arg.HV + ivh])
                    : expf(arg.g_cum[static_cast<size_t>(tc + j) * arg.HV + ivh]);
                const size_t v_offset =
                    (static_cast<size_t>(tc + j) * arg.HV + ivh) * 128 + col;
                const size_t k_offset =
                    (static_cast<size_t>(tc + j) * arg.H + ih) * 128 + col;
                u_acc += a_value * beta * ck_tile::type_convert<float>(v[v_offset]);
                w_acc += a_value * beta * gate *
                         ck_tile::type_convert<float>(k[k_offset]);
            }
            const size_t out_offset =
                (static_cast<size_t>(tc + row) * arg.HV + ivh) * 128 + col;
            u[out_offset] = ck_tile::type_convert<ck_tile::half_t>(u_acc);
            w[out_offset] = ck_tile::type_convert<ck_tile::half_t>(w_acc);
        }
    }
};

template <typename Policy>
struct GdnRecomputeWUFwdInvoker
{
    static void Run(
        hipStream_t stream,
        const uint16_t* k,
        const uint16_t* v,
        const float* beta,
        const uint16_t* A,
        const float* g_cum,
        uint16_t* w,
        uint16_t* u,
        int T, int H, int HV, int K_dim, int V_dim,
        bool use_exp2,
        const int64_t* cu_seqlens = nullptr,
        const int64_t* chunk_indices = nullptr,
        int num_chunks = 0,
        bool is_varlen = false)
    {
        ck_tile::stream_config stream_cfg{stream};
        if constexpr(std::is_same_v<typename Policy::DataType, ck_tile::half_t>)
        {
            if(!is_varlen && T % Policy::kChunkSize != 0)
            {
                using Kernel = GdnRecomputeWUFwdKernel<Policy, false, false, true>;
                auto kargs = Kernel::MakeKargs(k, v, beta, A, g_cum, w, u,
                                               T, H, HV, K_dim, V_dim, use_exp2);
                const auto grid = Kernel::GridSize(kargs);
                constexpr auto block = Kernel::BlockSize();
                ck_tile::launch_kernel(
                    stream_cfg,
                    ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                        Kernel{}, grid, block, 0, kargs));
                ck_tile::launch_kernel(
                    stream_cfg,
                    ck_tile::make_kernel<256, 1>(
                        GdnRecomputeWUFp16TailKernel{}, grid, dim3(256), 0, kargs));
                return;
            }
        }
        if(is_varlen)
        {
            using Kernel = GdnRecomputeWUFwdKernel<Policy, false, false, true>;
            auto kargs = Kernel::MakeKargs(k,
                                           v,
                                           beta,
                                           A,
                                           g_cum,
                                           w,
                                           u,
                                           T,
                                           H,
                                           HV,
                                           K_dim,
                                           V_dim,
                                           use_exp2,
                                           cu_seqlens,
                                           chunk_indices,
                                           num_chunks,
                                           true);
            auto grid = Kernel::GridSize(kargs);
            constexpr auto block = Kernel::BlockSize();
            ck_tile::launch_kernel(
                stream_cfg,
                ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                    Kernel{}, grid, block, 0, kargs));
            if constexpr(std::is_same_v<typename Policy::DataType, ck_tile::half_t>)
            {
                ck_tile::launch_kernel(
                    stream_cfg,
                    ck_tile::make_kernel<256, 1>(
                        GdnRecomputeWUFp16TailKernel{},
                        dim3(HV, num_chunks, 1),
                        dim3(256, 1, 1),
                        0,
                        kargs));
            }
            return;
        }
        if(T < 1024 && T % Policy::kChunkSize != 0)
        {
            if constexpr(std::is_same_v<typename Policy::DataType, ck_tile::bf16_t>)
            {
                using Kernel = GdnRecomputeWUFwdKernel<Policy, true, false, true>;
                auto kargs = Kernel::MakeKargs(k, v, beta, A, g_cum, w, u,
                                               T, H, HV, K_dim, V_dim, use_exp2);
                auto grid = Kernel::GridSize(kargs);
                constexpr auto block = Kernel::BlockSize();
                ck_tile::launch_kernel(
                    stream_cfg,
                    ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                        Kernel{}, grid, block, 0, kargs));
                return;
            }
        }
        if(T < 1024)
        {
            using Kernel = GdnRecomputeWUFwdKernel<Policy, true>;
            auto kargs = Kernel::MakeKargs(k, v, beta, A, g_cum, w, u,
                                           T, H, HV, K_dim, V_dim, use_exp2);
            auto grid = Kernel::GridSize(kargs);
            constexpr auto block = Kernel::BlockSize();
            ck_tile::launch_kernel(
                stream_cfg,
                ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                    Kernel{}, grid, block, 0, kargs));
        }
        else
        {
            if constexpr(std::is_same_v<typename Policy::DataType, ck_tile::bf16_t>)
            {
                if(H == 16 && HV == 64 && K_dim == 128 && V_dim == 128)
                {
                    if(T % Policy::kChunkSize != 0)
                    {
                        using Kernel = GdnRecomputeWUFwdKernel<Policy, false, true, true>;
                        auto kargs = Kernel::MakeKargs(k, v, beta, A, g_cum, w, u,
                                                       T, H, HV, K_dim, V_dim, use_exp2);
                        auto grid = Kernel::GridSize(kargs);
                        constexpr auto block = Kernel::BlockSize();
                        ck_tile::launch_kernel(
                            stream_cfg,
                            ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                                Kernel{}, grid, block, 0, kargs));
                    }
                    else
                    {
                        using Kernel = GdnRecomputeWUFwdKernel<Policy, false, true>;
                        auto kargs = Kernel::MakeKargs(k, v, beta, A, g_cum, w, u,
                                                       T, H, HV, K_dim, V_dim, use_exp2);
                        auto grid = Kernel::GridSize(kargs);
                        constexpr auto block = Kernel::BlockSize();
                        ck_tile::launch_kernel(
                            stream_cfg,
                            ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                                Kernel{}, grid, block, 0, kargs));
                    }
                    return;
                }

                if(T % Policy::kChunkSize != 0)
                {
                    using Kernel = GdnRecomputeWUFwdKernel<Policy, false, false, true>;
                    auto kargs = Kernel::MakeKargs(k, v, beta, A, g_cum, w, u,
                                                   T, H, HV, K_dim, V_dim, use_exp2);
                    auto grid = Kernel::GridSize(kargs);
                    constexpr auto block = Kernel::BlockSize();
                    ck_tile::launch_kernel(
                        stream_cfg,
                        ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                            Kernel{}, grid, block, 0, kargs));
                    return;
                }
            }

            using Kernel = GdnRecomputeWUFwdKernel<Policy, false, false>;
            auto kargs = Kernel::MakeKargs(k, v, beta, A, g_cum, w, u,
                                           T, H, HV, K_dim, V_dim, use_exp2);
            auto grid = Kernel::GridSize(kargs);
            constexpr auto block = Kernel::BlockSize();
            ck_tile::launch_kernel(
                stream_cfg,
                ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                    Kernel{}, grid, block, 0, kargs));
        }
    }
};
