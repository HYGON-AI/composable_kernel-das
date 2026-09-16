// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/kernel/gdn_ck_fused_recurrent_kernel.hpp"
#include <hip/hip_runtime.h>

namespace gdn_decode_example {

inline constexpr int kSupportedHeadDim = 128;

void launch_bf16(const GdnFusedRecurrentKargs&, hipStream_t);
void launch_fp16(const GdnFusedRecurrentKargs&, hipStream_t);

} // namespace gdn_decode_example
