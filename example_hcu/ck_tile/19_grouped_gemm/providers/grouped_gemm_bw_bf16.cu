// 实例化条件（编译期）：本文件由 CK provider manifest 纳入 gfx936/gfx938
// device-args 构建，生成 BW-family BF16 的通用预编译入口。
// 当前 selector 在 gfx936 BF16 NT/NN 且 effective-M、N/K、N/K 对齐满足
// BW-family 大块门时选择它；两个被提升的 gfx936 BF16 NN 精确 shape 会先选择
// fixed/fast 专用 provider，BF16 TN 则由 transposed/logical-tail provider 承担。
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX: SPDX-License-Identifier: MIT
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_ONLY
#include "instances/grouped_gemm_gfx936_bf16.cpp"
