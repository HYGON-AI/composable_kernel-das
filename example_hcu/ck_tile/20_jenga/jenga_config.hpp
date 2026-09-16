// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile::example::jenga {

inline constexpr index_t kBlockM          = 64;
inline constexpr index_t kBlockN          = 64;
inline constexpr index_t kHeadDim         = 128;
inline constexpr index_t kThreadsPerBlock = 256;
inline constexpr index_t kBlocksPerCu     = 2;
inline constexpr index_t kDqMaxNnzCap     = 512;
inline constexpr index_t kDkdvLaunchMinBlocks = 1;

inline constexpr index_t kTargetActiveBlocksPerStage = 24;
inline constexpr index_t kSelectionCapacityMargin     = 2;
inline constexpr index_t kMinSplitStageCount          = 2;
inline constexpr index_t kLargeSequenceThreshold      = 65536;
inline constexpr index_t kTunedSequenceLength         = 75648;
inline constexpr index_t kDefaultStageCount           = 1;
inline constexpr index_t kMaxDqStageCount              = 8;
inline constexpr index_t kTunedDqStageCount            = 5;
inline constexpr index_t kMaxDkdvSplitCount            = 8;
inline constexpr index_t kTunedDkdvSplitCount          = 4;

} // namespace ck_tile::example::jenga
