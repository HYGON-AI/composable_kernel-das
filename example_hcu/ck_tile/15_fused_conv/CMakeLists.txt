# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

# add_compile_options("SHELL: -mllvm -support-512-vgprs=true -mllvm -stream-unfolded-args-in-metadata -Rpass-analysis=kernel-resource-usage")
ck_add_arch_vgpr_compile_options()
add_compile_options("SHELL: -mllvm -disable-machine-licm -mllvm -stream-unfolded-args-in-metadata -Rpass-analysis=kernel-resource-usage")
# add_example_executable(tile_example_fused_conv fused_conv.cpp)
# add_example_executable(tile_example_fused_conv fused_conv_tls.cpp)
add_example_executable(tile_example_fused_conv_mls fused_conv_mls.cpp)
add_executable(tile_example_fused_conv_mls_wasp_probe EXCLUDE_FROM_ALL fused_conv_mls_wasp.cpp)
target_link_libraries(tile_example_fused_conv_mls_wasp_probe PRIVATE utility)
# The perf-only grouped-conv mapping probes were removed after tuning. The
# group-local persistent traversal was made correct, but remained slower than
# the accepted grouped-GEMM MLS kernel for the equivalent 1x1 workload. Keep
# only the general correctness/resource probes here.
add_executable(tile_example_fused_conv_mls_wasp_bf16_probe EXCLUDE_FROM_ALL fused_conv_mls_wasp.cpp)
target_compile_definitions(tile_example_fused_conv_mls_wasp_bf16_probe PRIVATE CK_TILE_MLS_PROBE_BF16=1)
target_link_libraries(tile_example_fused_conv_mls_wasp_bf16_probe PRIVATE utility)
# add_example_executable(tile_example_fused_conv fused_conv_cshuffle.cpp)
# add_example_executable(tile_example_fused_conv fused_conv_tls_wasp.cpp)
if(GPU_TARGETS MATCHES "gfx946")
    add_example_executable(tile_example_fused_conv_tls_wasp_persistant
                           fused_conv_tls_wasp_persistant.cpp)
endif()
# add_example_executable(tile_example_fused_conv fused_conv_mls_wasp.cpp)
# add_example_executable(tile_example_fused_conv fused_conv_mls_wasp_persistant.cpp)
if(CK_BUILD_UNSTABLE_HCU_EXAMPLES)
add_example_executable(tile_example_fused_conv fused_conv_wasp_persistant.cpp)
endif()
