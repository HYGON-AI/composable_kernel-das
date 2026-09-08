# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_example_executable(example_batched_gemm_bias_e_permute_xdl_fp16 batched_gemm_bias_e_permute_xdl_fp16.cpp)

if(GPU_TARGETS MATCHES "gfx1100" OR GPU_TARGETS MATCHES "gfx1101" OR GPU_TARGETS MATCHES "gfx1102")
    add_example_executable(example_batched_gemm_bias_e_permute_wmma_fp16 batched_gemm_bias_e_permute_wmma_fp16.cpp)
endif()
