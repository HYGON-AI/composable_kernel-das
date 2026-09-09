// 实例化条件：gfx936 或 gfx938、BF16、TN、K 为 device-lengths capacity，且
// enable_bf16_tn_logical_k_tail policy 开启；effective-M、N/K 和 N 对齐仍须满足
// BW-family 大块门。该实例按每组逻辑 K 处理尾部，不把 capacity 当作单组 K。
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX: SPDX-License-Identifier: MIT
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_ONLY
#include "instances/grouped_gemm_gfx936_bf16_tn_blas_transposed_logical_k_tail.cpp"
