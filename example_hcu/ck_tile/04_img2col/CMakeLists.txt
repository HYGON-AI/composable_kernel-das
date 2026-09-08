# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

# Keep this focused acceptance target out of the default all/install/check graph.
add_executable(tile_example_img2col EXCLUDE_FROM_ALL image_to_column.cpp)
target_link_libraries(tile_example_img2col PRIVATE utility)
