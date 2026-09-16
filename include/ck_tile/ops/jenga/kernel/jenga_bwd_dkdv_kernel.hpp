// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dkdv_pipeline.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile {
namespace example {
namespace jenga {

template <typename Pipeline_>
struct JengaBwdDkdvTilePartitioner
{
    using Pipeline = remove_cvref_t<Pipeline_>;

    CK_TILE_HOST static dim3 GridSize(ck_tile::index_t n_kv_blocks, ck_tile::index_t batch_heads, ck_tile::index_t split_kv = 1)
    {
        return dim3(n_kv_blocks, batch_heads, split_kv);
    }

    CK_TILE_DEVICE auto operator()() const
    {
        return make_tuple(static_cast<index_t>(blockIdx.x), static_cast<index_t>(blockIdx.y), static_cast<index_t>(blockIdx.z));
    }
};

template <typename Pipeline_>
struct JengaBwdDkdvKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using TilePartitioner = JengaBwdDkdvTilePartitioner<Pipeline>;

    static constexpr index_t BlockM          = Pipeline::BlockM;
    static constexpr index_t BlockN          = Pipeline::BlockN;
    static constexpr index_t HeadDim         = Pipeline::HeadDim;
    static constexpr index_t ThreadsPerBlock = Pipeline::ThreadsPerBlock;
    static constexpr index_t KernelBlockSize = ThreadsPerBlock;

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t n_kv_blocks, ck_tile::index_t batch_heads, ck_tile::index_t split_kv = 1)
    {
        return TilePartitioner::GridSize(n_kv_blocks, batch_heads, split_kv);
    }

    CK_TILE_DEVICE void operator()(jenga_bwd_dkdv_args args) const
    {
        const auto [kv_block, off_hz, split_id] = TilePartitioner{}();
        const ck_tile::index_t batch  = off_hz / args.H;
        const ck_tile::index_t head   = off_hz - batch * args.H;
        const ck_tile::index_t seqlen = args.seqlens_ptr[batch];

        if(kv_block * BlockN >= seqlen)
        {
            return;
        }

        Pipeline::run(args, kv_block, off_hz, batch, head, seqlen, split_id);
    }
};

} // namespace jenga
} // namespace example
} // namespace ck_tile
