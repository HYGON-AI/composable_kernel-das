# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_grouped_conv2d_fwd_mmac grouped_conv2d_fwd_mmac.cpp)
target_link_libraries(
  test_grouped_conv2d_fwd_mmac PRIVATE
  utility
  device_grouped_conv2d_fwd_gfx928_instance
  device_grouped_conv2d_fwd_gfx936_instance
)
