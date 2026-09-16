// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_preprocess_pipeline.hpp"
#include "ck_tile/core.hpp"

namespace ck_tile {
namespace example {
namespace sla {

template <typename Pipeline>
struct SlaAttnBwdPreprocessKernel
{
    using Kargs = typename Pipeline::Kargs;
    static constexpr int kBlockSize = 256;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        constexpr int kTokensPerBlock = Pipeline::kUseD128 ? kBlockSize / 4 : kBlockSize;
        return dim3((arg.total_tokens + kTokensPerBlock - 1) / kTokensPerBlock);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_DEVICE void operator()(const Kargs& arg) const { Pipeline{}(arg); }
};

} // namespace sla
} // namespace example
} // namespace ck_tile
