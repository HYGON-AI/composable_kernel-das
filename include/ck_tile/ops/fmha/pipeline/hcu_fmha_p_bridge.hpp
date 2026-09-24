// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// The HCU QK accumulator and PV operand use different lane/register layouts.
// Match the bridge used by the HCU asynchronous forward-attention pipeline.
template <typename Problem, typename Policy, typename PTile>
CK_TILE_DEVICE void hcu_fmha_p_bridge(PTile& p, void* smem_ptr)
{
#if defined(__gfx936__) || defined(__gfx938__)
    static_assert(Problem::BlockFmhaShape::kM0 == 16 && Problem::BlockFmhaShape::kN0 == 64);
    using PDataType = typename Problem::PDataType;
    auto* scratch = reinterpret_cast<PDataType*>(
        static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeKVWithoutPBridge<Problem>());
    const index_t lane = get_lane_id();
    static_for<0, 4, 1>{}([&](auto r) {
        static_for<0, 4, 1>{}([&](auto c) {
#if defined(__gfx938__)
            const index_t row = 4 * (lane / 16) + r;
            const index_t col = c * 16 + lane % 16;
#else
            const index_t row = lane % 16;
            const index_t col = c * 16 + r * 4 + lane / 16;
#endif
            scratch[row * 64 + col] = p.get_thread_buffer()[c * 4 + r];
        });
    });
    block_sync_lds();
    static_for<0, 16, 1>{}([&](auto k) {
        p.get_thread_buffer()(k) = scratch[(lane % 16) * 64 + (lane / 16) * 16 + k];
    });
#else
    (void)p;
    (void)smem_ptr;
#endif
}

} // namespace ck_tile
