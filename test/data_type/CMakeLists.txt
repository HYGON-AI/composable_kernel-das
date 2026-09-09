# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

if (USE_BITINT_EXTENSION_INT4)
  add_gtest_executable(test_int4 int4.cpp)
  target_link_libraries(test_int4 PRIVATE utility)
endif()
