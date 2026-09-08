# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_executable(tile_example_moe_smoothquant EXCLUDE_FROM_ALL moe_smoothquant.cpp)
target_compile_options(tile_example_moe_smoothquant PRIVATE -Wno-undefined-func-template
                                                             -Wno-float-equal)
target_link_libraries(tile_example_moe_smoothquant PRIVATE utility)

add_executable(tile_example_moe_smoothquant_tail EXCLUDE_FROM_ALL moe_smoothquant_tail.cpp)
target_compile_options(tile_example_moe_smoothquant_tail
                       PRIVATE -Wno-undefined-func-template -Wno-float-equal)
target_link_libraries(tile_example_moe_smoothquant_tail PRIVATE utility)
