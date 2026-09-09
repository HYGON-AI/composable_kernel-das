# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

add_gtest_executable(test_convnd_fwd convnd_fwd.cpp)
target_link_libraries(test_convnd_fwd PRIVATE utility device_conv2d_fwd_instance)
