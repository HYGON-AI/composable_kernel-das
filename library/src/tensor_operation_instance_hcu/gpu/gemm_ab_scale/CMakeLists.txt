# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd.

# GEMM ab_scale (per-block A/B dequantization scales) instances for HCU
# (gfx936 family, 16x16 XDL tiles only).
set(GEMM_AB_SCALE_HCU_INSTANCES
    device_gemm_ab_scale_xdl_bf16_bf16_bf16_mk_nk_mn_instance.cpp
    )

add_instance_library(device_gemm_ab_scale_instance ${GEMM_AB_SCALE_HCU_INSTANCES})
