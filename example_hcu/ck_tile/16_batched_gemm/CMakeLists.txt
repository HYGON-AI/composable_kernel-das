# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_executable(tile_example_batched_gemm EXCLUDE_FROM_ALL batched_gemm.cpp)
target_link_libraries(tile_example_batched_gemm PRIVATE utility)
