# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

add_executable(tile_example_unified_attention_2d
    unified_attention_2d.cpp
    instances/unified_attention_2d_d192_fp16.cpp
    instances/unified_attention_2d_d192_bf16.cpp
    instances/unified_attention_2d_d256_fp16.cpp
    instances/unified_attention_2d_d256_bf16.cpp)
add_executable(tile_example_unified_attention_3d unified_attention_3d.cpp)

foreach(target tile_example_unified_attention_2d tile_example_unified_attention_3d)
    # Masked rows and empty softmax segments use infinity sentinels. Preserve
    # their comparisons and isfinite checks in both kernels and the reference.
    target_compile_options(${target} PRIVATE -O3 -ffast-math -fno-finite-math-only
        -w -U__HIP_NO_HALF_CONVERSIONS__)
endforeach()

# The imported 2D MMAC implementation uses the experimental MMAC attribute family.
target_compile_definitions(tile_example_unified_attention_2d
    PRIVATE CK_EXPERIMENTAL_BIT_INT_EXTENSION)
