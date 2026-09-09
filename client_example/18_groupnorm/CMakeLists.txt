# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

add_executable(client_groupnorm_swish groupnorm_swish.cpp)
target_link_libraries(client_groupnorm_swish PRIVATE composable_kernel::device_operations)
