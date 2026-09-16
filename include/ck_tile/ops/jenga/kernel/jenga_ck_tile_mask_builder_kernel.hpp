// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/jenga/pipeline/jenga_ck_tile_mask_builder_pipeline.hpp"
#include "ck_tile/host.hpp"
#include <hip/hip_runtime.h>

struct JengaMaskBuilderLaunchConfig
{
    hipStream_t stream;
};

template <typename PipelinePolicy, bool IsQuery>
struct JengaMaskPoolKernel
{
    using Kargs = JengaMaskBuilderArgument;
    using Pipeline = JengaMaskPoolPipeline<PipelinePolicy, IsQuery>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(
            arg.B * arg.H * (IsQuery ? arg.num_query_blocks : arg.num_blocks)));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg, float* q_pool, float* k_pool) const
    {
        if constexpr (IsQuery) {
            Pipeline{}(arg.query, q_pool, arg.H, arg.N_Q, arg.num_query_blocks);
        } else {
            Pipeline{}(arg.key, k_pool, arg.H, arg.N_K, arg.num_blocks);
        }
    }
};

template <typename PipelinePolicy>
struct JengaMaskScoreKernel
{
    using Kargs = JengaMaskBuilderArgument;
    using Pipeline = JengaMaskScorePipeline<PipelinePolicy>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.B * arg.H * arg.num_query_blocks),
                    static_cast<unsigned int>((arg.text_start_block + PipelinePolicy::kScoreTile - 1) /
                                              PipelinePolicy::kScoreTile));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg, const float* q_pool, const float* k_pool, float* scores) const
    {
        Pipeline{}(q_pool, k_pool, scores, arg.H, arg.num_query_blocks, arg.num_blocks, arg.text_start_block);
    }
};

template <typename PipelinePolicy>
struct JengaMaskSelectKernel
{
    using Kargs = JengaMaskBuilderArgument;
    using Pipeline = JengaMaskSelectPipeline<PipelinePolicy>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.B * arg.H * arg.num_query_blocks));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg, float* scores, int candidate_k) const
    {
        Pipeline{}(scores,
                   arg.neighbor_mask,
                   arg.out,
                   arg.B * arg.H * arg.num_query_blocks,
                   arg.num_query_blocks,
                   arg.num_blocks,
                   arg.text_start_block,
                   arg.top_k,
                   arg.prob_threshold,
                   candidate_k,
                   arg.text_blocks,
                   arg.first_frame_blocks,
                   arg.neighbor_mask != nullptr);
    }
};

template <typename PipelinePolicy>
struct JengaMaskBuilderInvoker
{
    using Kargs = JengaMaskBuilderArgument;

    static void RunPoolQ(const Kargs& arg, float* q_pool, float* k_pool, hipStream_t stream)
    {
        using Kernel = JengaMaskPoolKernel<PipelinePolicy, true>;
        constexpr auto blocks = Kernel::BlockSize();
        ck_tile::stream_config stream_cfg{stream};
        ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<blocks.x, PipelinePolicy::kLaunchMinBlocks>(
                Kernel{}, Kernel::GridSize(arg), blocks, 0, arg, q_pool, k_pool));
    }

    static void RunPoolK(const Kargs& arg, float* q_pool, float* k_pool, hipStream_t stream)
    {
        using Kernel = JengaMaskPoolKernel<PipelinePolicy, false>;
        constexpr auto blocks = Kernel::BlockSize();
        ck_tile::stream_config stream_cfg{stream};
        ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<blocks.x, PipelinePolicy::kLaunchMinBlocks>(
                Kernel{}, Kernel::GridSize(arg), blocks, 0, arg, q_pool, k_pool));
    }

    static void RunScore(const Kargs& arg,
                         const float* q_pool,
                         const float* k_pool,
                         float* scores,
                         hipStream_t stream)
    {
        using Kernel = JengaMaskScoreKernel<PipelinePolicy>;
        constexpr auto blocks = Kernel::BlockSize();
        ck_tile::stream_config stream_cfg{stream};
        ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<blocks.x, PipelinePolicy::kLaunchMinBlocks>(
                Kernel{}, Kernel::GridSize(arg), blocks, 0, arg, q_pool, k_pool, scores));
    }

    static void RunSelect(const Kargs& arg, float* scores, int candidate_k, hipStream_t stream)
    {
        using Kernel = JengaMaskSelectKernel<PipelinePolicy>;
        constexpr auto blocks = Kernel::BlockSize();
        ck_tile::stream_config stream_cfg{stream};
        ck_tile::launch_kernel(
            stream_cfg,
            ck_tile::make_kernel<blocks.x, PipelinePolicy::kLaunchMinBlocks>(
                Kernel{}, Kernel::GridSize(arg), blocks, 0, arg, scores, candidate_k));
    }
};
