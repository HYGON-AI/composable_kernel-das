// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modified by Hygon Information Technology Co., Ltd.
//
// Hygon HCU: global_load_tr (transpose load) is NOT available.
// Upstream uses __builtin_amdgcn_global_load_tr_b128_v8f16 (gfx12x only).
// HCU architectures (gfx928/gfx936/gfx938) have no transpose load builtin.

#pragma once

namespace ck {

// amd_global_load_transpose_to_vgpr: not supported on HCU.
// Requires gfx12x transpose load hardware.

} // namespace ck
