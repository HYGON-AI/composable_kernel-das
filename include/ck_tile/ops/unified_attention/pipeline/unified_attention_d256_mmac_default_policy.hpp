// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/unified_attention/pipeline/unified_attention_d256_mmac_policy.hpp"

template <typename Problem>
struct UnifiedAttentionD256MmacDefaultPolicy
    : UnifiedAttentionD256DefaultPipelinePolicyT<Problem>
{
    static constexpr bool kUseSplitK32 = true;

    CK_TILE_HOST_DEVICE static constexpr int GetOutputTile(int) { return 128; }

    template <int>
    CK_TILE_HOST_DEVICE static constexpr int GetQkDChunks()
    {
        return 4;
    }
};
