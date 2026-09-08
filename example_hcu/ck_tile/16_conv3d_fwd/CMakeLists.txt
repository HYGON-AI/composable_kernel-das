# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

ck_add_arch_vgpr_compile_options()
add_compile_options("SHELL: -Rpass-analysis=kernel-resource-usage")
add_example_executable(tile_example_conv3d_fwd  conv3d_fwd.cpp)
