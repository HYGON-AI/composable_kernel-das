# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd.
add_executable(tile_example_moe_sorting EXCLUDE_FROM_ALL moe_sorting.cpp moe_sorting_api.cpp)
target_include_directories(tile_example_moe_sorting PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

set(EXAMPLE_MOE_SORTING_COMPILE_OPTIONS)
# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_MOE_SORTING_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
# list(APPEND EXAMPLE_MOE_SORTING_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)
target_compile_options(tile_example_moe_sorting PRIVATE ${EXAMPLE_MOE_SORTING_COMPILE_OPTIONS})

# Match AITER JIT; runtime device properties choose ordinary or multi-die launch.
if("gfx946" IN_LIST GPU_TARGETS OR "gfx946" IN_LIST CMAKE_HIP_ARCHITECTURES)
  target_compile_definitions(tile_example_moe_sorting PRIVATE CK_ENABLE_MULTIDIE=1)
endif()

