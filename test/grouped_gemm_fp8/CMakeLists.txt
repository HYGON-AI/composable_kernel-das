# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Can also be configured alone without the full CK suite's GTest dependency.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(ck_grouped_gemm_fp8 LANGUAGES CXX)
  enable_testing()
  add_executable(test_grouped_gemm_fp8 grouped_gemm_fp8.cpp)
  target_compile_features(test_grouped_gemm_fp8 PRIVATE cxx_std_17)
  target_compile_options(test_grouped_gemm_fp8 PRIVATE -x hip --offload-arch=gfx938 -O2)
  target_link_options(test_grouped_gemm_fp8 PRIVATE --hip-link)
  target_include_directories(test_grouped_gemm_fp8 PRIVATE ../../include ../../library/include)
  add_test(NAME test_grouped_gemm_fp8 COMMAND test_grouped_gemm_fp8)
elseif(GPU_TARGETS STREQUAL "gfx938")
  # This regression configuration is currently validated only on gfx938.
  add_test_executable(test_grouped_gemm_fp8 grouped_gemm_fp8.cpp)
endif()
