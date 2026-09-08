# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

add_example_executable(tile_example_conv_fwd  conv_fwd.cpp)
add_example_executable(tile_example_conv_fwd_dsreadm  conv_fwd_dsreadm.cpp)
set_tests_properties(tile_example_conv_fwd_dsreadm PROPERTIES SKIP_RETURN_CODE 77)
add_example_executable(tile_example_conv_fwd_lds_direct  conv_fwd_lds_direct.cpp)
