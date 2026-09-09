# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

add_executable(client_gemm gemm.cpp)
target_link_libraries(client_gemm PRIVATE composable_kernel::device_operations)
