# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

set(TILE_EXAMPLE_GROUPED_GEMM "tile_example_grouped_gemm")
message("adding ${TILE_EXAMPLE_GROUPED_GEMM}")

# Keep the development executable deterministic. Shared implementation bodies
# and archived tuning probes must not be discovered as standalone TUs.
set(INSTANCE_SRCS
    instances/grouped_gemm_bf16.cpp
    instances/grouped_gemm_bf8.cpp
    instances/grouped_gemm_fp16.cpp
    instances/grouped_gemm_fp8.cpp
    instances/grouped_gemm_int4.cpp
    instances/grouped_gemm_int8.cpp
    # The generic FP16/BF16 dispatchers contain runtime-gated references to
    # these providers on every architecture, so both TUs are link providers
    # even when the compiled target is not gfx936.
    instances/grouped_gemm_gfx936_bf16.cpp
    instances/grouped_gemm_gfx936_fp16.cpp)


ck_add_arch_vgpr_compile_options()
add_compile_options("SHELL: -Rpass-analysis=kernel-resource-usage")

add_executable(${TILE_EXAMPLE_GROUPED_GEMM}
    grouped_gemm.cpp
    ${INSTANCE_SRCS}
)
target_include_directories(${TILE_EXAMPLE_GROUPED_GEMM} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
option(CK_TILE_GROUPED_GEMM_RESOURCE_REPORT "Print grouped_gemm kernel resource usage during compile" OFF)
option(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX938_DISABLE
    "Disable the default gfx938 FP16 NN ragged-M Family-A V3 single-LDS route" OFF)
option(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX936_DISABLE
    "Disable the default gfx936 FP16 NN ragged-M Family-A V3 single-LDS route" OFF)
option(CK_TILE_GROUPED_GEMM_P1_FAMILY_B_V6_M32_MLE8_GFX936_DISABLE
    "Disable the default gfx936 FP16 NN Family-B selector-effective-M<=8 V6 route" OFF)
option(CK_TILE_GROUPED_GEMM_P2_BF16_G8_V4_64_M01_4_GFX936_DISABLE
    "Disable the default gfx936 BF16 G8 fixed-M 1918..1921 V4 64x128 M01=4 route"
    OFF)
option(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX936_FC2_NN_FAST_VMEM3_DISABLE
    "Disable the default gfx936 BF16 G8 common-M 2048..2221 FC2 NN fast-vmem3 route"
    OFF)
option(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX938_V5_FC1_ROUTE_DISABLE
    "Disable the default gfx938 BF16 G8 fixed-M 1918..1921 NT N4096 K7168 V4 route"
    OFF)
option(CK_TILE_GROUPED_GEMM_P3_FP16_TN_M1024_K4096_ROUTE_DISABLE
    "Disable the default dual-architecture FP16 TN homogeneous M1024 N4096 K4096 selector exchange"
    OFF)

set(HCU_GROUPED_GEMM_DEFAULT_ROUTE_DISABLE_OPTIONS
    CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX938_DISABLE
    CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX936_DISABLE
    CK_TILE_GROUPED_GEMM_P1_FAMILY_B_V6_M32_MLE8_GFX936_DISABLE
    CK_TILE_GROUPED_GEMM_P2_BF16_G8_V4_64_M01_4_GFX936_DISABLE
    CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX936_FC2_NN_FAST_VMEM3_DISABLE
    CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX938_V5_FC1_ROUTE_DISABLE
    CK_TILE_GROUPED_GEMM_P3_FP16_TN_M1024_K4096_ROUTE_DISABLE)

function(ck_tile_hcu_grouped_gemm_apply_default_route_disables target_name)
    foreach(route_disable IN LISTS HCU_GROUPED_GEMM_DEFAULT_ROUTE_DISABLE_OPTIONS)
        if(${route_disable})
            target_compile_definitions(${target_name} PRIVATE ${route_disable})
        endif()
    endforeach()
endfunction()
if(CK_TILE_GROUPED_GEMM_RESOURCE_REPORT)
    target_compile_options(${TILE_EXAMPLE_GROUPED_GEMM} PRIVATE "-Rpass-analysis=kernel-resource-usage")
endif()

function(add_grouped_gemm_fast_target target_name fast_prec_define instance_src)
    add_executable(${target_name} EXCLUDE_FROM_ALL
        grouped_gemm.cpp
        instances/${instance_src}
    )
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    ck_add_arch_vgpr_compile_options()
    add_compile_options("SHELL: -Rpass-analysis=kernel-resource-usage")
    target_compile_definitions(${target_name}
        PRIVATE
            CK_TILE_GROUPED_GEMM_FAST_BUILD
            CK_TILE_GROUPED_GEMM_FAST_RC_ONLY
            ${fast_prec_define}
            ${ARGN})
    if(CK_TILE_GROUPED_GEMM_RESOURCE_REPORT)
        target_compile_options(${target_name} PRIVATE "-Rpass-analysis=kernel-resource-usage")
    endif()
endfunction()

add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_fast CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_fast CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp)
set(HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES)
set(HCU_GROUPED_GEMM_BW_TN_DEVICE_ARGS_SOURCES)
# Keep only the two promoted gfx936 BF16-NN instances in normal and production targets.
if(GPU_TARGETS MATCHES "gfx936" AND NOT GPU_TARGETS MATCHES "gfx938|gfx946")
    list(APPEND HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES
        instances/grouped_gemm_gfx936_bf16_nn_fixed_srd_lds8_boundary_lds0_vmem0.cpp
        instances/grouped_gemm_gfx936_bf16_nn_rr_fast_vmem3_early_final_barrier.cpp)
endif()

# The BF16 TN device-argument family is validated on both gfx936 and gfx938.
# Keep its providers in the selected CK library even though the old
# host-descriptor default only references the promoted logical-tail entry on gfx936.
if(GPU_TARGETS MATCHES "gfx936|gfx938")
    list(APPEND HCU_GROUPED_GEMM_BW_TN_DEVICE_ARGS_SOURCES
        instances/grouped_gemm_gfx936_bf16_tn_blas_transposed.cpp
        instances/grouped_gemm_gfx936_bf16_tn_blas_transposed_logical_k_tail.cpp)
endif()

if(HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES)
    target_sources(${TILE_EXAMPLE_GROUPED_GEMM}
        PRIVATE
            ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES}
            ${HCU_GROUPED_GEMM_BW_TN_DEVICE_ARGS_SOURCES})
    target_compile_definitions(${TILE_EXAMPLE_GROUPED_GEMM}
        PRIVATE CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED)
    target_sources(tile_example_grouped_gemm_bf16_fast
        PRIVATE
            instances/grouped_gemm_gfx936_bf16.cpp
            ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES}
            ${HCU_GROUPED_GEMM_BW_TN_DEVICE_ARGS_SOURCES})
    target_compile_definitions(tile_example_grouped_gemm_bf16_fast
        PRIVATE CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED)
endif()
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_baseline CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_baseline CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
# BW-family large aligned RC: retain only the selected V3
# 256x256x64/W16/GroupNum=8 reproduction targets. Groups 1/2/3/5/8 are
# correct; groups 1/3/4/5/8 performance scales with the same selected config.
# Removed probes: W8 K64 spilled and took 21.88-24.97 ms; 128x256 K64/K32
# reached 6.80/5.92 ms; W16 K32, M8xN2 and explicit overlap reached
# 6.52/5.87/5.42 ms; direct-store was neutral (5.1952 vs 5.1949 ms CShuffle).
# The selected warp-raked V3 policy reduced LDS bank conflicts by 68.09% and
# FP16 latency from 5.1980 to 5.1634 ms without changing V3/V4 implementations.
# The normal targets exercise the bounded default while V4 remains the fallback
# for other shapes, layouts and targets. Group-count tests cover 1/2/3/5/8.
add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_v3_regprefetch_256_w16_group8 CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_REGPREFETCH_256_W16_GROUP8)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_v3_regprefetch_256_w16_group8 CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_TUNE_V3_REGPREFETCH_256_W16_GROUP8)
# Historical tuning targets are intentionally absent from the production
# build graph. The normal targets above exercise the selected gfx936 defaults;
# the two explicit targets below reproduce the accepted gfx938 MLS policy.
# gfx938 tuning summary for group_count=4, M/N/K=4096/7168/4096:
# keep only the accepted 256x256x32, eight-wave, K32 single-stage MLS target.
# It overlaps the next global load with MMAC, skips the redundant post-MMAC
# sync, retains CShuffle, uses 32 KiB LDS and permits two blocks/CU. Standalone
# FP16/BF16 measured 4.77439/4.77946 ms and passed CPU reference validation.
# Removed candidates covered Compute-V3 register prefetch, K64/split-register,
# sequential/fused pairs, direct store, alternative CShuffle/warp/tile shapes,
# priority/sync/DS-addressing variants, Lit/Lts and non-persistent traversal.
# They were slower, neutral, unsupported, superseded, or failed Primus
# fwd/grad-A/grad-B correctness. The normal fp16/bf16 targets also exercise
# this accepted configuration through the bounded default dispatch.
if(GPU_TARGETS STREQUAL "gfx938" OR GPU_TARGETS STREQUAL "gfx946")
    add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp16_mls_256_k32_single_stage CK_TILE_GROUPED_GEMM_FAST_FP16 grouped_gemm_fp16.cpp CK_TILE_GROUPED_GEMM_TUNE_MLS_256 CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE CK_TILE_GROUPED_GEMM_MLS_OVERLAP_GLOBAL_LOAD CK_TILE_GROUPED_GEMM_MLS_SKIP_POST_MMAC_SYNC)
endif()
if(GPU_TARGETS STREQUAL "gfx938" OR GPU_TARGETS STREQUAL "gfx946")
    add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf16_mls_256_k32_single_stage CK_TILE_GROUPED_GEMM_FAST_BF16 grouped_gemm_bf16.cpp CK_TILE_GROUPED_GEMM_TUNE_MLS_256 CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE CK_TILE_GROUPED_GEMM_MLS_OVERLAP_GLOBAL_LOAD CK_TILE_GROUPED_GEMM_MLS_SKIP_POST_MMAC_SYNC)
endif()
if(GPU_TARGETS STREQUAL "gfx938" OR GPU_TARGETS STREQUAL "gfx946")
    add_grouped_gemm_fast_target(tile_example_grouped_gemm_fp8_fast CK_TILE_GROUPED_GEMM_FAST_FP8 grouped_gemm_fp8.cpp)
endif()
if(GPU_TARGETS STREQUAL "gfx938" OR GPU_TARGETS STREQUAL "gfx946")
    add_grouped_gemm_fast_target(tile_example_grouped_gemm_bf8_fast CK_TILE_GROUPED_GEMM_FAST_BF8 grouped_gemm_bf8.cpp)
endif()
add_grouped_gemm_fast_target(tile_example_grouped_gemm_int8_fast CK_TILE_GROUPED_GEMM_FAST_INT8 grouped_gemm_int8.cpp)
add_grouped_gemm_fast_target(tile_example_grouped_gemm_int4_fast CK_TILE_GROUPED_GEMM_FAST_INT4 grouped_gemm_int4.cpp)

if(GPU_TARGETS MATCHES "gfx938|gfx946")
    set(HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS)
    if(GPU_TARGETS MATCHES "gfx938")
        list(APPEND HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS
             CK_TILE_HCU_QUANT_GEMM_GFX938=1)
    endif()
    if(GPU_TARGETS MATCHES "gfx946")
        list(APPEND HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS
             CK_TILE_HCU_QUANT_GEMM_GFX946=1)
    endif()

    add_executable(tile_example_grouped_gemm_quant_tensor EXCLUDE_FROM_ALL
        quant_grouped_gemm_tensor.cpp)
    target_compile_definitions(tile_example_grouped_gemm_quant_tensor
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})

    add_executable(tile_example_grouped_gemm_quant_rowcol EXCLUDE_FROM_ALL
        quant_grouped_gemm_rowcol.cpp)
    target_compile_definitions(tile_example_grouped_gemm_quant_rowcol
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})

    add_executable(tile_example_grouped_gemm_quant_blockwise EXCLUDE_FROM_ALL
        quant_grouped_gemm_blockwise.cpp)
    target_compile_definitions(tile_example_grouped_gemm_quant_blockwise
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})

    add_custom_target(tile_example_grouped_gemm_quant_smoke
        COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_tensor>
        COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_rowcol>
        COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_blockwise>
        DEPENDS
            tile_example_grouped_gemm_quant_tensor
            tile_example_grouped_gemm_quant_rowcol
            tile_example_grouped_gemm_quant_blockwise
        COMMENT "Running HCU ck_tile FP8/BF8 quant grouped GEMM smoke tests"
        VERBATIM)

    foreach(quant_mode tensor rowcol blockwise)
        add_test(NAME tile_example_grouped_gemm_quant_${quant_mode}_smoke
            COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_${quant_mode}>)
        set_tests_properties(tile_example_grouped_gemm_quant_${quant_mode}_smoke
            PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;quant;smoke")
    endforeach()
endif()

# gfx938/gfx946 quant grouped-GEMM instances and the versioned device-args C ABI
# are CK-owned. Keep their heavy gemm_quant.hpp closure isolated from the
# existing non-quant provider archive.
if(GPU_TARGETS MATCHES "gfx938|gfx946")
    set(HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_MANIFEST
        ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_quant_device_args_provider_sources.txt)
    file(STRINGS ${HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_MANIFEST}
        HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_SOURCES
        REGEX "^[^#].+")
    set_source_files_properties(${HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_SOURCES}
        PROPERTIES LANGUAGE CXX)

    add_library(ck_tile_hcu_quant_grouped_gemm_device_args_providers STATIC EXCLUDE_FROM_ALL
        ${HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_SOURCES})
    target_include_directories(ck_tile_hcu_quant_grouped_gemm_device_args_providers
        PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    target_compile_definitions(ck_tile_hcu_quant_grouped_gemm_device_args_providers
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})
    target_compile_options(ck_tile_hcu_quant_grouped_gemm_device_args_providers
        PRIVATE "-Wno-unused-template" "-Wno-unneeded-internal-declaration")
    set_target_properties(ck_tile_hcu_quant_grouped_gemm_device_args_providers
        PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(ck_tile_hcu_quant_grouped_gemm_device_args_providers
        PRIVATE
            "-fno-offload-uniform-block"
            "SHELL: -mllvm --lsr-drop-solution=1"
            "SHELL: -mllvm -enable-post-misched=0"
            "SHELL: -mllvm -amdgpu-early-inline-all=true"
            "-fno-gpu-rdc")

    add_library(ck_tile_hcu_quant_grouped_gemm STATIC EXCLUDE_FROM_ALL
        grouped_gemm_quant_device_args.cpp)
    target_include_directories(ck_tile_hcu_quant_grouped_gemm
        PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    target_compile_definitions(ck_tile_hcu_quant_grouped_gemm
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})
    set_target_properties(ck_tile_hcu_quant_grouped_gemm
        PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(ck_tile_hcu_quant_grouped_gemm
        PRIVATE
            "-fno-offload-uniform-block"
            "SHELL: -mllvm --lsr-drop-solution=1"
            "SHELL: -mllvm -enable-post-misched=0"
            "SHELL: -mllvm -amdgpu-early-inline-all=true"
            "-fno-gpu-rdc")
    target_link_libraries(ck_tile_hcu_quant_grouped_gemm
        PUBLIC ck_tile_hcu_quant_grouped_gemm_device_args_providers)

    add_executable(tile_example_grouped_gemm_quant_device_args_smoke EXCLUDE_FROM_ALL
        grouped_gemm_quant_device_args_smoke.cpp)
    target_compile_definitions(tile_example_grouped_gemm_quant_device_args_smoke
        PRIVATE ${HCU_QUANT_GROUPED_GEMM_ARCH_DEFINITIONS})
    target_link_libraries(tile_example_grouped_gemm_quant_device_args_smoke
        PRIVATE ck_tile_hcu_quant_grouped_gemm)
    add_test(NAME tile_example_grouped_gemm_quant_device_args_smoke
        COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_device_args_smoke>)
    set_tests_properties(tile_example_grouped_gemm_quant_device_args_smoke
        PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;quant;device_args;smoke")
endif()

add_library(ck_tile_hcu_grouped_gemm STATIC EXCLUDE_FROM_ALL
    grouped_gemm.cpp
    grouped_gemm_device_args.cpp
    instances/grouped_gemm_fp16.cpp
    instances/grouped_gemm_bf16.cpp
    instances/grouped_gemm_gfx936_fp16.cpp
    instances/grouped_gemm_gfx936_bf16.cpp
    ${HCU_GROUPED_GEMM_BW_TN_DEVICE_ARGS_SOURCES}
    ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES}
)
set(HCU_GROUPED_GEMM_RDC_BUILD OFF)
if(CMAKE_CXX_FLAGS MATCHES "(^| )-fgpu-rdc( |$)")
    set(HCU_GROUPED_GEMM_RDC_BUILD ON)
endif()

# Match the HCU production code-generation contract used by the existing
# grouped-GEMM consumer.  These options are target-local: they affect only the
# public grouped-GEMM archives and do not change CK-wide compilation policy.
# Keep RDC an explicit opt-in because the validated non-RDC provider objects
# avoid the register spill observed at the integration boundary.
function(ck_tile_hcu_grouped_gemm_apply_codegen target_name)
    target_compile_options(${target_name}
        PRIVATE
            "-fno-offload-uniform-block"
            "SHELL: -mllvm --lsr-drop-solution=1"
            "SHELL: -mllvm -enable-post-misched=0"
            "SHELL: -mllvm -amdgpu-early-inline-all=true")
    if(NOT HCU_GROUPED_GEMM_RDC_BUILD)
        target_compile_options(${target_name} PRIVATE "-fno-gpu-rdc")
    endif()
endfunction()

# Compile the exact CK-owned provider manifest as its own archive. Consumers
# such as Primus compile these same sources, whose DEVICE_ARGS_ONLY and
# REGISTRY_ONLY wrappers keep the new device-args closure independent from the
# retained host-descriptor ABI translation units below.
set(HCU_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_MANIFEST
    ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_device_args_provider_sources.txt)
file(STRINGS ${HCU_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_MANIFEST}
    HCU_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_SOURCES
    REGEX "^[^#].+")
set_source_files_properties(${HCU_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_SOURCES}
    PROPERTIES LANGUAGE CXX)
add_library(ck_tile_hcu_grouped_gemm_device_args_providers STATIC EXCLUDE_FROM_ALL
    ${HCU_GROUPED_GEMM_DEVICE_ARGS_PROVIDER_SOURCES})
target_include_directories(ck_tile_hcu_grouped_gemm_device_args_providers
    PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ck_tile_hcu_grouped_gemm_device_args_providers
    PRIVATE
        CK_TILE_GROUPED_GEMM_FAST_BUILD
        CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY)
if(GPU_TARGETS MATCHES "gfx936")
    target_compile_definitions(ck_tile_hcu_grouped_gemm_device_args_providers
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_GFX936)
endif()
if(GPU_TARGETS MATCHES "gfx938")
    target_compile_definitions(ck_tile_hcu_grouped_gemm_device_args_providers
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_GFX938)
endif()
target_compile_options(ck_tile_hcu_grouped_gemm_device_args_providers
    PRIVATE "-Wno-unused-template" "-Wno-unneeded-internal-declaration")
set_target_properties(ck_tile_hcu_grouped_gemm_device_args_providers
    PROPERTIES POSITION_INDEPENDENT_CODE ON)
ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_hcu_grouped_gemm_device_args_providers)
set_source_files_properties(
    ${HCU_GROUPED_GEMM_BW_TN_DEVICE_ARGS_SOURCES}
    ${HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES}
    PROPERTIES
        COMPILE_OPTIONS "-Wno-unused-template;-Wno-unneeded-internal-declaration")
target_include_directories(ck_tile_hcu_grouped_gemm PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ck_tile_hcu_grouped_gemm
    PRIVATE
        CK_TILE_EXAMPLE_NO_MAIN
        CK_TILE_GROUPED_GEMM_FAST_BUILD
        CK_TILE_GROUPED_GEMM_FAST_RC_ONLY
        CK_TILE_GROUPED_GEMM_FAST_FP16
        CK_TILE_GROUPED_GEMM_FAST_BF16
        CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY)
if(HCU_GROUPED_GEMM_RDC_BUILD)
    target_compile_definitions(ck_tile_hcu_grouped_gemm
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
endif()
ck_tile_hcu_grouped_gemm_apply_default_route_disables(ck_tile_hcu_grouped_gemm)
if(GPU_TARGETS MATCHES "gfx936")
    target_compile_definitions(ck_tile_hcu_grouped_gemm
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_GFX936)
    # Device-args providers always compile the gfx936 BF16-NN promoted
    # kernels. Open the matching C ABI dispatch even in gfx936;gfx938 fat
    # binaries so selector InstanceId 1001/1002 is not left undispatchable.
    target_compile_definitions(ck_tile_hcu_grouped_gemm
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED)
endif()
if(GPU_TARGETS MATCHES "gfx938")
    target_compile_definitions(ck_tile_hcu_grouped_gemm
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_GFX938)
endif()
if(HCU_GROUPED_GEMM_GFX936_SELECTED_SOURCES)
    # Keep gfx936-only host-descriptor declarations and link providers paired.
    # In particular, a gfx938-only build has an empty selected source list and
    # must not make grouped_gemm.cpp reference the gfx936 selected entry points.
    target_compile_definitions(ck_tile_hcu_grouped_gemm
        PRIVATE CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED)
endif()
if(HCU_GROUPED_GEMM_BW_TN_DEVICE_ARGS_SOURCES)
    target_compile_definitions(ck_tile_hcu_grouped_gemm
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_BW_TN)
endif()
set_target_properties(ck_tile_hcu_grouped_gemm PROPERTIES POSITION_INDEPENDENT_CODE ON)
ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_hcu_grouped_gemm)
target_link_libraries(ck_tile_hcu_grouped_gemm
    PUBLIC ck_tile_hcu_grouped_gemm_device_args_providers)

# Minimal consumer-independent library used by the public C ABI matrix. It
# deliberately excludes all default-off Phase316/317 candidate TUs so generic
# NT/NN/TN fallback coverage cannot accidentally resolve an experiment symbol.
add_library(ck_tile_hcu_grouped_gemm_generic STATIC EXCLUDE_FROM_ALL
    grouped_gemm.cpp
    instances/grouped_gemm_fp16.cpp
    instances/grouped_gemm_bf16.cpp
    instances/grouped_gemm_gfx936_fp16.cpp
    instances/grouped_gemm_gfx936_bf16.cpp
)
target_include_directories(ck_tile_hcu_grouped_gemm_generic PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ck_tile_hcu_grouped_gemm_generic
    PRIVATE
        CK_TILE_EXAMPLE_NO_MAIN
        CK_TILE_GROUPED_GEMM_FAST_BUILD
        CK_TILE_GROUPED_GEMM_FAST_RC_ONLY
        CK_TILE_GROUPED_GEMM_FAST_FP16
        CK_TILE_GROUPED_GEMM_FAST_BF16)
set_target_properties(ck_tile_hcu_grouped_gemm_generic
    PROPERTIES POSITION_INDEPENDENT_CODE ON)
ck_tile_hcu_grouped_gemm_apply_codegen(ck_tile_hcu_grouped_gemm_generic)

add_executable(tile_example_grouped_gemm_c_api_smoke EXCLUDE_FROM_ALL
    grouped_gemm_c_api_smoke.cpp
)
target_link_libraries(tile_example_grouped_gemm_c_api_smoke
    PRIVATE ck_tile_hcu_grouped_gemm_generic)

# Production-selector variant of the exact same 60-case public C ABI matrix.
# On gfx936 it links the promoted selected instances; on gfx938 the selected
# source list is empty and the architecture-specific generic portfolio stays
# unchanged.
add_executable(tile_example_grouped_gemm_c_api_selected_smoke EXCLUDE_FROM_ALL
    grouped_gemm_c_api_smoke.cpp
)
target_link_libraries(tile_example_grouped_gemm_c_api_selected_smoke
    PRIVATE ck_tile_hcu_grouped_gemm)

# This smoke validates the gfx936-only host-descriptor selected path and its
# same-ELF rollback. Other architectures have no selected path to exercise.
if(GPU_TARGETS MATCHES "gfx936")
    add_executable(tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke EXCLUDE_FROM_ALL
        grouped_gemm_c_api_tn_logical_k_tail_smoke.cpp
    )
    target_link_libraries(tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke
        PRIVATE ck_tile_hcu_grouped_gemm)
endif()

# Host-only Phase-1 selector/contract test. This target deliberately does not
# link a grouped-GEMM provider, so adding it cannot alter a device launch.
add_executable(tile_example_grouped_gemm_selector_test EXCLUDE_FROM_ALL
    grouped_gemm_selector_test.cpp
)
add_test(NAME tile_example_grouped_gemm_selector_test
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_selector_test>)
set_tests_properties(tile_example_grouped_gemm_selector_test
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;selector")

# Host-only quant contract/selector test. It covers the reported gfx938
# group-count-sensitive cases without linking or launching a device provider.
add_executable(tile_example_grouped_gemm_quant_selector_test EXCLUDE_FROM_ALL
    grouped_gemm_quant_selector_test.cpp)
add_test(NAME tile_example_grouped_gemm_quant_selector_test
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_quant_selector_test>)
set_tests_properties(tile_example_grouped_gemm_quant_selector_test
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;quant;selector")

add_custom_target(tile_example_grouped_gemm_quant_device_args_c_header_smoke
    COMMAND ${CMAKE_C_COMPILER}
            -std=c11
            -Wall
            -Wextra
            -Werror
            -I${PROJECT_SOURCE_DIR}/include
            -c
            ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_quant_device_args_c_header_smoke.c
            -o
            ${CMAKE_CURRENT_BINARY_DIR}/grouped_gemm_quant_device_args_c_header_smoke.o
    COMMENT "Compiling quant grouped-GEMM device-args public header as C11"
    VERBATIM)

# Compile the installed device-args header as strict C11, independently of
# every C++ implementation header and the directory-level HIP compile flags.
add_custom_target(tile_example_grouped_gemm_device_args_c_header_smoke
    COMMAND ${CMAKE_C_COMPILER}
            -std=c11
            -Wall
            -Wextra
            -Werror
            -I${PROJECT_SOURCE_DIR}/include
            -fsyntax-only
            ${CMAKE_CURRENT_LIST_DIR}/grouped_gemm_device_args_c_header_smoke.c
    COMMENT "Compiling grouped-GEMM device-args v1 header as pure C11"
    VERBATIM)

add_executable(tile_example_grouped_gemm_device_args_query_test EXCLUDE_FROM_ALL
    grouped_gemm_device_args_query_test.cpp)
ck_tile_hcu_grouped_gemm_apply_default_route_disables(
    tile_example_grouped_gemm_device_args_query_test)
target_link_libraries(tile_example_grouped_gemm_device_args_query_test
    PRIVATE ck_tile_hcu_grouped_gemm)
if(HCU_GROUPED_GEMM_RDC_BUILD)
    target_compile_definitions(tile_example_grouped_gemm_device_args_query_test
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
endif()
add_test(NAME tile_example_grouped_gemm_device_args_query_test
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_device_args_query_test>)
set_tests_properties(tile_example_grouped_gemm_device_args_query_test
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;device_args;selector")

add_executable(tile_example_grouped_gemm_device_args_smoke EXCLUDE_FROM_ALL
    grouped_gemm_device_args_smoke.cpp)
target_link_libraries(tile_example_grouped_gemm_device_args_smoke
    PRIVATE ck_tile_hcu_grouped_gemm)
if(HCU_GROUPED_GEMM_RDC_BUILD)
    target_compile_definitions(tile_example_grouped_gemm_device_args_smoke
        PRIVATE CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
endif()
add_test(NAME tile_example_grouped_gemm_device_args_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_device_args_smoke>)
set_tests_properties(tile_example_grouped_gemm_device_args_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;device_args;correctness;smoke")

# 可参数化的生产调用示例：复用 Primus 使用的 versioned device-args ABI，
# 运行时只选择 CK 已经预编译的 stable InstanceId。BLAS 通过 dlopen 加载，
# 可指定 hipBLASLt 安装根目录或单个 ELF，不把某个 BLAS 安装路径写死进 CK。
add_executable(tile_example_grouped_gemm_device_args_benchmark EXCLUDE_FROM_ALL
    grouped_gemm_device_args_benchmark.cpp)
target_link_libraries(tile_example_grouped_gemm_device_args_benchmark
    PRIVATE ck_tile_hcu_grouped_gemm ${CMAKE_DL_LIBS})

# 若需要让编译期头文件也严格跟随 Primus 的 hipBLASLt 安装，可在配置时设置
# -DCK_GROUPED_GEMM_HIPBLASLT_ROOT=...；未设置时继续使用工具链自带头文件。
set(CK_GROUPED_GEMM_HIPBLASLT_ROOT "" CACHE PATH
    "hipBLASLt install root used only by the grouped GEMM benchmark example")
if(NOT CK_GROUPED_GEMM_HIPBLASLT_ROOT AND DEFINED ENV{HIPBLASLT_ROOT})
    set(CK_GROUPED_GEMM_HIPBLASLT_ROOT "$ENV{HIPBLASLT_ROOT}")
endif()
if(CK_GROUPED_GEMM_HIPBLASLT_ROOT)
    target_include_directories(tile_example_grouped_gemm_device_args_benchmark SYSTEM
        PRIVATE "${CK_GROUPED_GEMM_HIPBLASLT_ROOT}/include")
endif()

# CTest 仅覆盖 CK device-args 小 shape；系统 BLAS 路径和性能矩阵由用户显式运行，
# 避免常规 correctness 测试依赖某个特定 BLAS 安装位置。
add_test(NAME tile_example_grouped_gemm_device_args_benchmark_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_device_args_benchmark>
            --m=128 --n=128 --k=128 --groups=2
            --dtype=fp16 --layout=NT --backend=ck
            --time=0 --correctness-repeats=1)
set_tests_properties(tile_example_grouped_gemm_device_args_benchmark_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;device_args;example;smoke")

# 通用 ragged frontend smoke：不包含任何 consumer workload 名称。NT/NN 将
# group lengths 解释为 M，TN 将其解释为 K；capacity 与 logical lengths 分离。
add_test(NAME tile_example_grouped_gemm_device_args_benchmark_ragged_m_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_device_args_benchmark>
            --m=129 --n=129 --k=129
            --group-lengths=65,0,129 --variable-capacity=256
            --allow-empty-groups=1
            --dtype=bf16 --layout=NN --backend=ck
            --time=0 --correctness-repeats=1)
set_tests_properties(tile_example_grouped_gemm_device_args_benchmark_ragged_m_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;device_args;ragged;example;smoke")

add_test(NAME tile_example_grouped_gemm_device_args_benchmark_ragged_k_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_device_args_benchmark>
            --m=128 --n=128 --k=129
            --group-lengths=65,0,129 --variable-capacity=256
            --allow-empty-groups=1
            --dtype=bf16 --layout=TN --backend=ck
            --time=0 --correctness-repeats=1)
set_tests_properties(tile_example_grouped_gemm_device_args_benchmark_ragged_k_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;device_args;ragged;example;smoke")

set(GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS
    -group_count=2
    -Ms=128,256
    -Ns=128,128
    -Ks=128,128
    -stride_As=128,128
    -stride_Bs=128,128
    -stride_Cs=128,128
    -validate=1
    -warmup=1
    -repeat=1
)

set(GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS
    -group_count=3
    -Ms=128,256,192
    -Ns=128,128,256
    -Ks=128,192,256
    -stride_As=128,192,256
    -stride_Bs=128,192,256
    -stride_Cs=128,128,256
    -validate=1
    -warmup=1
    -repeat=1
)

set(GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS
    ${GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS}
)
list(REMOVE_ITEM GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS -stride_Bs=128,192,256)
list(APPEND GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS -stride_Bs=128,128,256)

# TN 布局的对齐 variable-K 测试参数：A 为 ColumnMajor，B/C 为 RowMajor。
# ColumnMajor A 的 leading stride 是 M，所以 stride_As 与各组 Ms 相同。
set(GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS
    -group_count=3
    -Ms=128,256,192
    -Ns=128,128,256
    -Ks=128,192,256
    -stride_As=128,256,192
    -stride_Bs=128,128,256
    -stride_Cs=128,128,256
    -validate=1
    -warmup=1
    -repeat=1
)

# 全维 padding 测试故意选用 M/N/K 都不满足 tile 对齐的 shape，用于验证
# kPadM=true、kPadN=true、kPadK=true 时的边界 load/store 正确性。
set(GROUPED_GEMM_FAST_PADDED_COMMON_SHAPE_ARGS
    -group_count=3
    -Ms=65,129,191
    -Ns=129,130,257
    -Ks=129,193,255
    -stride_Cs=129,130,257
    -validate=1
    -warmup=1
    -repeat=1
)

# 按 NT、NN、TN 的物理布局分别设置 A/B 的 leading stride；调用方仍只分配
# 逻辑尺寸的 A/B/C buffer，不需要为 padded shape 额外申请显存。
set(GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS
    ${GROUPED_GEMM_FAST_PADDED_COMMON_SHAPE_ARGS}
    -stride_As=129,193,255
    -stride_Bs=129,193,255
)
set(GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS
    ${GROUPED_GEMM_FAST_PADDED_COMMON_SHAPE_ARGS}
    -stride_As=129,193,255
    -stride_Bs=129,130,257
)
set(GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS
    ${GROUPED_GEMM_FAST_PADDED_COMMON_SHAPE_ARGS}
    -stride_As=65,129,191
    -stride_Bs=129,130,257
)

# 聚合 smoke target：构建后顺序运行 FP16/BF16 的 NT/NN/TN、variable-K、
# 全维 padding 以及 C ABI 正确性用例。它属于 example 测试，不是 ck4inductor 测试。
add_custom_target(tile_example_grouped_gemm_fp16_bf16_fast_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=C -b_layout=R
            ${GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=C -b_layout=R
            ${GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=C -b_layout=R
            ${GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=R
            ${GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=C -b_layout=R
            ${GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS}
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_c_api_smoke>
    DEPENDS
            tile_example_grouped_gemm_fp16_fast
            tile_example_grouped_gemm_bf16_fast
            tile_example_grouped_gemm_c_api_smoke
    COMMENT "Running HCU ck_tile grouped GEMM FP16/BF16 fast smoke tests"
    VERBATIM
)

add_test(NAME tile_example_grouped_gemm_fp16_fast_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_fp16_fast>
            -prec=fp16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
)
set_tests_properties(tile_example_grouped_gemm_fp16_fast_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;smoke"
)

add_test(NAME tile_example_grouped_gemm_bf16_fast_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_bf16_fast>
            -prec=bf16
            -a_layout=R -b_layout=C
            ${GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS}
)
set_tests_properties(tile_example_grouped_gemm_bf16_fast_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;smoke"
)

# 将一个 example 命令注册成独立 CTest，便于通过 ctest -R 精确执行。
# a_layout/b_layout 使用 R/C 表示 RowMajor/ColumnMajor。
function(add_grouped_gemm_fast_smoke_test test_name target_name precision a_layout b_layout shape_args)
    add_test(NAME ${test_name}
        COMMAND $<TARGET_FILE:${target_name}>
                -prec=${precision}
                -a_layout=${a_layout}
                -b_layout=${b_layout}
                ${${shape_args}}
    )
    set_tests_properties(${test_name}
        PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;smoke"
    )
endfunction()

add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nn_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R R GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nn_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R R GROUPED_GEMM_FAST_FIXED_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nt_variable_k_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R C GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nt_variable_k_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R C GROUPED_GEMM_FAST_NT_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nn_variable_k_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R R GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nn_variable_k_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R R GROUPED_GEMM_FAST_NN_VARIABLE_K_SHAPE_ARGS)

# TN variable-K：验证新增加的 ColumnMajor-A + RowMajor-B layout。
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_tn_variable_k_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 C R GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_tn_variable_k_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 C R GROUPED_GEMM_FAST_TN_VARIABLE_K_SHAPE_ARGS)

# NT/NN/TN 全维 padding：分别覆盖 FP16 和 BF16 的非对齐 M/N/K。
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nt_padded_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R C GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nt_padded_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R C GROUPED_GEMM_FAST_PADDED_NT_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_nn_padded_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 R R GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_nn_padded_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 R R GROUPED_GEMM_FAST_PADDED_NN_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_fp16_tn_padded_fast_smoke
    tile_example_grouped_gemm_fp16_fast fp16 C R GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS)
add_grouped_gemm_fast_smoke_test(tile_example_grouped_gemm_bf16_tn_padded_fast_smoke
    tile_example_grouped_gemm_bf16_fast bf16 C R GROUPED_GEMM_FAST_PADDED_TN_SHAPE_ARGS)

# C ABI smoke 绕过 example CLI，直接验证公开 grouped GEMM C 接口的 layout/padding。
add_test(NAME tile_example_grouped_gemm_c_api_layout_padding_smoke
    COMMAND $<TARGET_FILE:tile_example_grouped_gemm_c_api_smoke>
)
set_tests_properties(tile_example_grouped_gemm_c_api_layout_padding_smoke
    PROPERTIES LABELS "hcu;ck_tile;grouped_gemm;c_api;smoke"
)
