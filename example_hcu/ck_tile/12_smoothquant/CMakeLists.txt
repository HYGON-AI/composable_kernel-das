# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_executable(tile_example_smoothquant EXCLUDE_FROM_ALL example_smoothquant.cpp)
target_compile_options(tile_example_smoothquant PRIVATE -Wno-undefined-func-template
                                                         -Wno-float-equal)
target_link_libraries(tile_example_smoothquant PRIVATE utility)
