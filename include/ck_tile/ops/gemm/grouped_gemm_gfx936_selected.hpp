// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

// Compatibility include for callers built against the original gfx936 name.
// New code should include grouped_gemm_bw_family_selected.hpp and call the
// canonical bw_family symbols.
#include "ck_tile/ops/gemm/grouped_gemm_bw_family_selected.hpp"
