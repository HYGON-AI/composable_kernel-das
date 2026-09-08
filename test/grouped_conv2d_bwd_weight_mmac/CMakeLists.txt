# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_grouped_conv2d_bwd_weight_mmac grouped_conv2d_bwd_weight_mmac.cpp)
target_link_libraries(
  test_grouped_conv2d_bwd_weight_mmac PRIVATE
  utility
  device_grouped_conv2d_wrw_gfx928_instance
  device_grouped_conv2d_wrw_gfx936_instance
)
