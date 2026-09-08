# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

# hcu batched gemm examples
add_custom_target(example_hcu_batched_gemm_xdl)

add_example_executable(example_hcu_batched_gemm_xdl_bf16_v3 batched_gemm_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_batched_gemm_multi_d_xdl_bf16_v3
                       batched_gemm_multi_d_xdl_bf16_v3.cpp)

add_custom_target(example_hcu_batched_gemm_multi_d_xdl_bf16_v3_smoke
    COMMAND $<TARGET_FILE:example_hcu_batched_gemm_multi_d_xdl_bf16_v3>
            1 1 0 128 128 128 2
    DEPENDS example_hcu_batched_gemm_multi_d_xdl_bf16_v3
    COMMENT "Running HCU batched GEMM MultiD BF16 Add smoke test"
    VERBATIM
)

add_test(NAME example_hcu_batched_gemm_multi_d_xdl_bf16_v3_smoke
    COMMAND $<TARGET_FILE:example_hcu_batched_gemm_multi_d_xdl_bf16_v3>
            1 1 0 128 128 128 2
)
set_tests_properties(example_hcu_batched_gemm_multi_d_xdl_bf16_v3_smoke
    PROPERTIES LABELS "hcu;batched_gemm;multi_d;smoke"
)

add_dependencies(example_hcu_batched_gemm_xdl
                 example_hcu_batched_gemm_xdl_bf16_v3
                 example_hcu_batched_gemm_multi_d_xdl_bf16_v3)
