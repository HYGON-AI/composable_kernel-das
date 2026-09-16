// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

#include <cstddef>

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

} // namespace ck_tile::example::sla
