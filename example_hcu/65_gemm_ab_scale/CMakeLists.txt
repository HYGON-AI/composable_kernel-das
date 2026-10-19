# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.
# hcu gemm ab_scale (per-block A/B dequantization scales) example
add_example_executable(example_hcu_gemm_ab_scale_xdl_bf16
                       gemm_ab_scale_xdl_bf16.cpp)

add_test(NAME example_hcu_gemm_ab_scale_xdl_bf16_smoke
    COMMAND $<TARGET_FILE:example_hcu_gemm_ab_scale_xdl_bf16> 1 1 0 256 256 512 0
)
set_tests_properties(example_hcu_gemm_ab_scale_xdl_bf16_smoke
    PROPERTIES LABELS "hcu;gemm;ab_scale;smoke"
)
