// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
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

    // Experiment switch: the paired GroupSize=2 epilogue requires an extra
    // 4 KiB LDS allocation.  Disable it to let both row tiles reuse the
    // 16 KiB main workspace and test the 4-CTA/CU occupancy threshold.
    static constexpr bool kUsePairedEpilogue = false;

    // Experiment: GroupSize=2 normally exchanges H through LDS to obtain the
    // MMAC B layout. Load the logical [N,K] view directly into BReg instead,
    // matching the established GroupSize=4 path.
    static constexpr bool kDirectLogicalHForGroup2 = true;

    // Experiment: V is physically [K, value_dim].  The legacy path performs
    // GMEM -> copy registers -> padded LDS -> PV BReg to transpose it to the
    // logical [N,K] MMAC-B view.  For GroupSize=2, load that logical view
    // directly into the native PV BReg distribution and bypass the exchange.
    static constexpr bool kDirectLogicalVForGroup2 = true;
    static constexpr bool kPackedRawHPrototype = false;
};

} // namespace gdn
