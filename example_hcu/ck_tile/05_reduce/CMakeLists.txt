# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_executable(tile_example_reduce EXCLUDE_FROM_ALL reduce.cpp)
target_include_directories(tile_example_reduce PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(tile_example_reduce PRIVATE -Wno-undefined-func-template -Wno-float-equal)
target_link_libraries(tile_example_reduce PRIVATE utility)
