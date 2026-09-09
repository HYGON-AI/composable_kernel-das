# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd.

add_gtest_executable(test_reduce_no_index reduce_no_index.cpp)
add_gtest_executable(test_reduce_with_index reduce_with_index.cpp)
target_link_libraries(test_reduce_no_index PRIVATE utility device_reduce_gfx936_instance)
target_link_libraries(test_reduce_with_index PRIVATE utility device_reduce_gfx936_instance)