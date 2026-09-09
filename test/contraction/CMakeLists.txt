# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd.

# TODO: Test for gfx936 Only
if((DTYPES MATCHES "fp32" OR DTYPES MATCHES "fp64") OR NOT DEFINED DTYPES)
    add_gtest_executable(test_contraction test_contraction_xdl.cpp)
    target_link_libraries(test_contraction PRIVATE utility device_contraction_bilinear_gfx936_instance device_contraction_scale_gfx936_instance)
    add_gtest_executable(test_contraction_interface test_contraction_interface_xdl.cpp)
    target_link_libraries(test_contraction_interface PRIVATE utility device_contraction_bilinear_gfx936_instance device_contraction_scale_gfx936_instance)
endif()
