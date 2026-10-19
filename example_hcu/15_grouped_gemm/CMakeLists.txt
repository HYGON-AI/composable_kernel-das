# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.
# hcu grouped gemm tile loop examples
add_example_executable(example_hcu_grouped_gemm_multiple_d_xdl_fp16
                       grouped_gemm_multiple_d_xdl_fp16.cpp)

add_test(NAME example_hcu_grouped_gemm_multiple_d_xdl_fp16_smoke
    COMMAND $<TARGET_FILE:example_hcu_grouped_gemm_multiple_d_xdl_fp16>
            1 1 0 64,127,255 252,252,252 4608,4608,4608 4608,4608,4608 4608,4608,4608 252,252,252
)
set_tests_properties(example_hcu_grouped_gemm_multiple_d_xdl_fp16_smoke
    PROPERTIES LABELS "hcu;grouped_gemm;multi_d;smoke"
)
