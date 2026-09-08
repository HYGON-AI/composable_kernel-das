# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_executable(tile_example_permute EXCLUDE_FROM_ALL permute.cpp)
target_link_libraries(tile_example_permute PRIVATE utility)
