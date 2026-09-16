// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/unified_attention/pipeline/unified_attention_mmac_policy.hpp"

namespace unified_attention {

template <typename Problem, int QkDChunks = 4>
struct UnifiedAttentionMmacDefaultPolicy : UnifiedAttentionMmacPipelinePolicyT<Problem>
{
    static constexpr bool kUseSplitK32 = false;

    CK_TILE_HOST_DEVICE static constexpr int GetOutputTile(int) { return 128; }

    template <int>
    CK_TILE_HOST_DEVICE static constexpr int GetQkDChunks()
    {
        return QkDChunks;
    }
};

} // namespace unified_attention
