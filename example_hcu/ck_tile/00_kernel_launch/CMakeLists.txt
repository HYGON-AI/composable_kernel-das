# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
add_executable(tile_test_multidie_launch EXCLUDE_FROM_ALL test_multidie_launch.cpp)
add_test(NAME tile_test_multidie_launch COMMAND $<TARGET_FILE:tile_test_multidie_launch>)
if("gfx946" IN_LIST GPU_TARGETS OR "gfx946" IN_LIST CMAKE_HIP_ARCHITECTURES)
  target_compile_definitions(tile_test_multidie_launch PRIVATE CK_ENABLE_MULTIDIE=1)
endif()
