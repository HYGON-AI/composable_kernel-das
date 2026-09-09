# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

add_executable(client_fused_attention fused_attention.cpp)
target_link_libraries(client_fused_attention PRIVATE composable_kernel::device_operations)

add_executable(client_fused_attention_bias fused_attention_bias.cpp)
target_link_libraries(client_fused_attention_bias PRIVATE composable_kernel::device_operations)
