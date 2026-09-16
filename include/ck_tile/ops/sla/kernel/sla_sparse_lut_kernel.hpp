// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_sparse_lut_pipeline.hpp"
#include "ck_tile/host.hpp"

#include <cstddef>
#include <cstdint>

namespace ck_tile {
namespace example {
namespace sla {

template <typename Pipeline>
struct SlaAttnBwdDqKvPartitionKernel
{
    using Kargs = typename Pipeline::Kargs;
    static constexpr int kBlockSize = 64;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.row_count));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }
    CK_TILE_DEVICE void operator()(const Kargs& arg) const { Pipeline{}(arg); }
};

struct SlaBwdDkdvBucketLutKernel
{
    using Kargs = SlaBwdDkdvBucketLutKargs;
    static constexpr int kBlockSize = SlaBwdDkdvBucketLutPipeline::kBlockSize;

    CK_TILE_HOST static dim3 GridSize(int bh_count) { return dim3(bh_count); }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }
    CK_TILE_HOST static constexpr size_t LdsSize()
    {
        return 2 * SlaBwdDkdvBucketLutPipeline::kBucketCount * sizeof(int32_t);
    }
    CK_TILE_DEVICE void operator()(const Kargs& arg) const
    {
        SlaBwdDkdvBucketLutPipeline{}(arg);
    }
};

struct SlaBwdDkdvClusterGreedyKernel
{
    using Kargs = SlaBwdDkdvClusterGreedyKargs;
    static constexpr int kBlockSize = SlaBwdDkdvClusterGreedyPipeline::kBlockSize;

    CK_TILE_HOST static dim3 GridSize(int bh_count, int kv_blocks)
    {
        const int segments =
            (kv_blocks + SlaBwdDkdvClusterGreedyPipeline::kSegmentSize - 1) /
            SlaBwdDkdvClusterGreedyPipeline::kSegmentSize;
        return dim3(bh_count * segments);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }
    CK_TILE_HOST static size_t LdsSize(int q_blocks)
    {
        const size_t scalar_count = static_cast<size_t>(
            q_blocks + SlaBwdDkdvClusterGreedyPipeline::kSegmentSize);
        const size_t aligned_scalar_count = (scalar_count + 1) & ~size_t{1};
        const size_t q_words = static_cast<size_t>((q_blocks + 63) / 64);
        return aligned_scalar_count * sizeof(int32_t) +
               q_words * SlaBwdDkdvClusterGreedyPipeline::kSegmentSize * sizeof(uint64_t);
    }
    CK_TILE_DEVICE void operator()(const Kargs& arg) const
    {
        SlaBwdDkdvClusterGreedyPipeline{}(arg);
    }
};

} // namespace sla
} // namespace example
} // namespace ck_tile

template <int MaxKvStageCount>
struct SlaAttnFwdKvStagePartitionKernel
{
    using Pipeline = SlaAttnFwdKvStagePartitionFastPipeline<MaxKvStageCount>;
    using Kargs    = typename Pipeline::Kargs;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return Pipeline::GridSize(arg);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return Pipeline::BlockSize(); }
    CK_TILE_DEVICE void operator()(const Kargs& arg) const { Pipeline{}(arg); }
};

template <int MaxKvStageCount>
struct SlaAttnFwdKvStagePartitionGenericKernel
{
    using Pipeline = SlaAttnFwdKvStagePartitionGenericPipeline<MaxKvStageCount>;
    using Kargs    = typename Pipeline::Kargs;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return Pipeline::GridSize(arg);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return Pipeline::BlockSize(); }
    CK_TILE_DEVICE void operator()(const Kargs& arg) const { Pipeline{}(arg); }
};

template <int MaxKvStageCount>
inline void launch_sla_attn_fwd_kv_stage_partition(
    const SlaAttnFwdKvStagePartitionArgument& arg,
    hipStream_t stream)
{
    ck_tile::stream_config stream_config{stream};
    if(arg.topk <= 128)
    {
        using Kernel = SlaAttnFwdKvStagePartitionKernel<MaxKvStageCount>;
        const auto grid = Kernel::GridSize(arg);
        constexpr auto block = Kernel::BlockSize();
        ck_tile::launch_kernel(
            stream_config,
            ck_tile::make_kernel<block.x, 1>(Kernel{}, grid, block, 0, arg));
    }
    else
    {
        using Kernel = SlaAttnFwdKvStagePartitionGenericKernel<MaxKvStageCount>;
        const auto grid = Kernel::GridSize(arg);
        constexpr auto block = Kernel::BlockSize();
        ck_tile::launch_kernel(
            stream_config,
            ck_tile::make_kernel<block.x, 1>(Kernel{}, grid, block, 0, arg));
    }
}
