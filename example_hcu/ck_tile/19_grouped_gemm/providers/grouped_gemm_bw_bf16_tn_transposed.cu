// 实例化条件：gfx936 或 gfx938、BF16、TN，effective-M 达到 1920（device
// lengths）或 2048（common）、N/K>=2048、N%256==0、K%64==0；K 不是启用
// logical-tail 的 variable dimension 时，selector 选择 blas_transposed 实例。
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX: SPDX-License-Identifier: MIT
#include "instances/grouped_gemm_gfx936_bf16_tn_blas_transposed.cpp"
