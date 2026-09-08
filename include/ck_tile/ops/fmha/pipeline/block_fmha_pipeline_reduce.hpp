// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// [HCU移植] 本文件为 FMHA 移植新增（上游 github CK 没有）。
// 背景：HCU MMAC 的 C 矩阵把 M 维放在 16 条 lane 上（wavefront=64），softmax 的行内
// 规约（rowmax/rowsum）必须按这套 16-lane 分布跨 lane 做 shuffle，与上游针对 AMD MFMA
// 布局的 block_tile_reduce_sync 不同。各 fwd 流水线里的 block_tile_reduce_sync 已替换为本函数。
//   - gfx938/gfx946（lit_lts 布局）：在 16 lane 组内按 delta=8/4/2/1 做 __shfl_xor。
//   - 其它 basic MMAC 架构：跨 64 lane 用 delta=16/32 做 __shfl_xor。
template <typename Tile, typename ReduceOp>
CK_TILE_HOST_DEVICE void fmha_reduce_sync_16lane(Tile& tile, ReduceOp reduce_op)
{
    auto& buf = tile.get_thread_buffer();
    static_for<0, Tile::get_thread_buffer_size(), 1>{}([&](auto i) {
        auto v = buf[i];
#if defined(__gfx938__) || defined(__gfx946__)
        for(int delta = 8; delta > 0; delta >>= 1)
        {
            v = reduce_op(v, __shfl_xor(v, delta, 16));
        }
#else
        v = reduce_op(v, __shfl_xor(v, 16, 64));
        v = reduce_op(v, __shfl_xor(v, 32, 64));
#endif
        buf(i) = v;
    });
}

} // namespace ck_tile
