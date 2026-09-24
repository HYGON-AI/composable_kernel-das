// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <ck_tile/core.hpp>

namespace gdn {

// Compile-time storage and copy configuration shared by every fwd_output
// problem specialization.  Layout construction belongs to the policy; this
// config only names hardware/tile constants.
struct GdnOutputFwdConfig
{
    static constexpr ck_tile::index_t kChunkSize = 64;
    static constexpr ck_tile::index_t kHeadDim = 128;
    static constexpr ck_tile::index_t kValueDim = 128;
    static constexpr ck_tile::index_t kRowTile = 32;
    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kNumWarps = 4;
    static constexpr ck_tile::index_t kWarpSize = 64;
    static constexpr ck_tile::index_t kGlobalVector = 8;
    static constexpr ck_tile::index_t kLdsPadding = 8;
    static constexpr ck_tile::index_t kScorePadding = 8;

};

} // namespace gdn
