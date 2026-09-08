# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(client_layernorm2d layernorm2d.cpp)
target_link_libraries(client_layernorm2d PRIVATE composable_kernel::device_operations)
