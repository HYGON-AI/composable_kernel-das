# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

# hcu GEMM XDL CShuffle v3 examples
add_custom_target(example_hcu_gemm_xdl_v3)

add_example_executable(example_hcu_gemm_xdl_bf16_v3 gemm_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_multi_d_xdl_bf16_v3 gemm_multi_d_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_multi_abd_xdl_bf16_v3 gemm_multi_abd_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_add_xdl_bf16_v3 gemm_add_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_add_relu_xdl_bf16_v3 gemm_add_relu_xdl_bf16_v3.cpp)

add_dependencies(example_hcu_gemm_xdl_v3
                 example_hcu_gemm_xdl_bf16_v3
                 example_hcu_gemm_multi_d_xdl_bf16_v3
                 example_hcu_gemm_multi_abd_xdl_bf16_v3
                 example_hcu_gemm_add_xdl_bf16_v3
                 example_hcu_gemm_add_relu_xdl_bf16_v3)
