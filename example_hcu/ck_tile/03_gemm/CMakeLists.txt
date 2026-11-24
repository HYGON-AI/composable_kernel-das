
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX: SPDX-License-Identifier: MIT
add_example_executable(tile_example_gemm_basic  gemm_basic.cpp)
if("gfx928" IN_LIST GPU_TARGETS OR "gfx92a" IN_LIST GPU_TARGETS OR
   "gfx936" IN_LIST GPU_TARGETS OR "gfx938" IN_LIST GPU_TARGETS)
    add_example_executable(tile_example_gemm_universal universal_gemm.cpp
        -m=256 -n=256 -k=128 -v=1 -warmup=0 -repeat=0)
    target_sources(tile_example_gemm_universal PRIVATE
        universal_gemm_fp16_rc.cpp universal_gemm_fp16_rr.cpp
        universal_gemm_bf16_rc.cpp universal_gemm_bf16_rr.cpp)
endif()
