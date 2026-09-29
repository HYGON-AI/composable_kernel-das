// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

#include <cstddef>
#include <limits>

namespace ck_tile::example::sla {

inline constexpr index_t kBlockN                   = 64;
inline constexpr index_t kDefaultBlockM            = 64;
inline constexpr index_t kLargeBlockM              = 128;
inline constexpr index_t kLargeBlockMSequenceLength = 18048;
inline constexpr index_t kHeadDim                   = 128;
inline constexpr index_t kDqLutStageCapacity        = 128;
inline constexpr index_t kQSliceBlockM              = 64;
inline constexpr index_t kMaxKvBlocks               = 2048;
inline constexpr index_t kMaxKvStageCount           = 16;
inline constexpr index_t kDefaultStageCount         = 1;
inline constexpr index_t kDefaultFwdKvStageCount    = 4;
inline constexpr index_t kLargeSequenceDqStageCount = 2;
inline constexpr index_t kTunedFwdKvStageCount       = 4;
inline constexpr index_t kTunedBwdDqStageCount       = 2;
inline constexpr index_t kTunedSequenceLength        = 75648;
inline constexpr index_t kLargeSequenceThreshold     = 64 * 1024;
inline constexpr std::size_t kWorkspaceAlignment     = 256;

// Pooling and sparse-map kernels still compute flattened offsets in int.
// Bound both [B*H, S, D] tensors and [B*H, D, D] linear-attention matrices
// before allocating anything. Division avoids overflow even for extreme CLI inputs.
inline constexpr bool is_sla_shape_indexable(int b, int h, int s, int d)
{
    return b > 0 && h > 0 && s > 0 && d > 0 &&
           b <= std::numeric_limits<int>::max() / h / (s > d ? s : d) / d;
}

} // namespace ck_tile::example::sla
