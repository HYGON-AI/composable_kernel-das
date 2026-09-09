// 实例化条件：gfx936、BF16、NN、group_count=16、M 为 device-lengths capacity，
// effective-M 在 [1920,4096]，且 N=2048、K=7168；同时还须满足 BW-family
// 公共大块/对齐门和 enable_gfx936_bf16_nn_promoted policy。
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX: SPDX-License-Identifier: MIT
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_ONLY
#include "instances/grouped_gemm_gfx936_bf16_nn_rr_fast_vmem3_early_final_barrier.cpp"
