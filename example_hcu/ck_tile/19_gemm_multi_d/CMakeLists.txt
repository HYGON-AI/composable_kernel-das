# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_executable(tile_example_gemm_multi_d EXCLUDE_FROM_ALL gemm_multi_d.cpp)
target_link_libraries(tile_example_gemm_multi_d PRIVATE utility)
