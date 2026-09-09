# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

add_test_executable(test_gemm_split_k gemm_split_k.cpp)
target_link_libraries(test_gemm_split_k PRIVATE utility)
target_link_libraries(test_gemm_split_k PRIVATE device_gemm_splitk_instance)
