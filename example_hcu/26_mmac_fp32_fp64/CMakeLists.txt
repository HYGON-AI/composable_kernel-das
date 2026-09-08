# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

# FP64 is exposed by CK on these two targets. Do not run a no-op fallback elsewhere.
if(NOT GPU_TARGETS)
    return()
endif()
foreach(arch IN LISTS GPU_TARGETS)
    if(NOT arch MATCHES "^gfx93[68](:.*)?$")
        return()
    endif()
endforeach()

add_example_executable(example_mmac_fp32_fp64 mmac_fp32_fp64.cpp)
