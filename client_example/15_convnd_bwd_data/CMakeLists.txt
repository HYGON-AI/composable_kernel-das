# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

add_executable(client_conv3d_bwd_data_fp16 conv3d_bwd_data_fp16.cpp)
add_executable(client_conv3d_bwd_data_fp32 conv3d_bwd_data_fp32.cpp)

target_link_libraries(client_conv3d_bwd_data_fp16 PRIVATE composable_kernel::device_operations)
target_link_libraries(client_conv3d_bwd_data_fp32 PRIVATE composable_kernel::device_operations)
