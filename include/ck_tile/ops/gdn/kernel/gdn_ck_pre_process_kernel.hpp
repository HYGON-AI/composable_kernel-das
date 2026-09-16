// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_pre_process_pipeline.hpp"
#include "ck_tile/host.hpp"

struct GdnPreProcessFwdKernelArg
{
    const uint16_t* k;
    const uint16_t* w;
    const uint16_t* u;
    const float*    g_cum;
    float*          hm;
    int T, H, HV, K_dim, V_dim, BT;
    int stride_k_t;
    int stride_w_t;
    int stride_u_t;
    int stride_g_t;
    bool use_exp2;
};

template <typename Policy>
struct GdnPreProcessFwdKernel
{
    using Kargs    = GdnPreProcessFwdKernelArg;
    using Pipeline = ck_tile::GdnPreProcessPipeline<Policy>;

    CK_TILE_HOST static constexpr Kargs MakeKargs(
        const uint16_t* k,
        const uint16_t* w,
        const uint16_t* u,
        const float* g_cum,
        float* hm,
        int T, int H, int HV, int K_dim, int V_dim,
        bool use_exp2)
    {
        constexpr int BT = Policy::kChunkSize;
        return Kargs{k, w, u, g_cum, hm,
                     T, H, HV, K_dim, V_dim, BT,
                     H * K_dim,
                     HV * K_dim,
                     HV * V_dim,
                     HV,
                     use_exp2};
    }

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(4, static_cast<unsigned int>(arg.HV), 1);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(Policy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int i_col = blockIdx.x;
        const int ivh = blockIdx.y;
        const int ratio_hv = arg.HV / arg.H;
        const int ih = ivh / ratio_hv;

        __shared__ typename Pipeline::SharedStorage scratch;
        using DataType = typename Policy::DataType;
        Pipeline{}(
            reinterpret_cast<const DataType*>(arg.k),
            reinterpret_cast<const DataType*>(arg.w),
            reinterpret_cast<const DataType*>(arg.u),
            arg.g_cum,
            arg.hm,
            i_col, ih, ivh,
            arg.T,
            arg.stride_k_t,
            arg.stride_w_t,
            arg.stride_u_t,
            arg.stride_g_t,
            arg.use_exp2,
            scratch);
    }
};

template <typename Policy>
struct GdnPreProcessFwdInvoker
{
    using Kernel = GdnPreProcessFwdKernel<Policy>;

    static void Run(
        hipStream_t stream,
        const uint16_t* k,
        const uint16_t* w,
        const uint16_t* u,
        const float* g_cum,
        float* hm,
        int T, int H, int HV, int K_dim, int V_dim,
        bool use_exp2)
    {
        auto kargs = Kernel::MakeKargs(k, w, u, g_cum, hm,
                                       T, H, HV, K_dim, V_dim, use_exp2);
        auto grid = Kernel::GridSize(kargs);
        constexpr auto block = Kernel::BlockSize();

        ck_tile::stream_config stream_cfg{stream};
        ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                Kernel{}, grid, block, 0, kargs));
    }
};
