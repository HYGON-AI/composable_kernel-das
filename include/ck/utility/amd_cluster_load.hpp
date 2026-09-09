// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modified by Hygon Information Technology Co., Ltd.
//
// Hygon HCU: cluster_multicast_load is NOT available.
// Upstream uses __builtin_amdgcn_cluster_load_b32/b64/b128 (gfx1250+ only).
// HCU architectures (gfx928/gfx936/gfx938) have no cluster multicast equivalent.

#pragma once

namespace ck {

// cluster_multicast_load: not supported on HCU.
// Requires gfx1250+ cluster multicast hardware.

} // namespace ck
