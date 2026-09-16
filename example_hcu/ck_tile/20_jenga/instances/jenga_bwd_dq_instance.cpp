// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Keep dQ in its own translation unit: gfx936 needs the extended VGPR mode for
// this kernel, while applying it to dK/dV reduces occupancy significantly.
#define JENGA_BWD_BUILD_DQ
#include "jenga_bwd_launch.hpp"
