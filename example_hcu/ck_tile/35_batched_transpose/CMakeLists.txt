
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX: SPDX-License-Identifier: MIT
add_executable(tile_example_batched_transpose EXCLUDE_FROM_ALL batched_transpose.cpp)
target_link_libraries(tile_example_batched_transpose PRIVATE utility)
