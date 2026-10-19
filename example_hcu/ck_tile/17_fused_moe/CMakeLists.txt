
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX: SPDX-License-Identifier: MIT

add_example_executable(example_hcu_ck_tile_fused_moe_flatmm_uk
                       fused_moe_flatmm_uk_fp16.cpp)

add_test(NAME example_hcu_ck_tile_fused_moe_flatmm_uk_smoke
    COMMAND $<TARGET_FILE:example_hcu_ck_tile_fused_moe_flatmm_uk> 64 4 2 256 256 1 2 2 0)

add_test(NAME example_hcu_ck_tile_fused_moe_flatmm_uk_partial_interm
    COMMAND $<TARGET_FILE:example_hcu_ck_tile_fused_moe_flatmm_uk> 64 4 2 256 1232 1 2 2 0)

add_test(NAME example_hcu_ck_tile_fused_moe_flatmm_uk_mock_id
    COMMAND $<TARGET_FILE:example_hcu_ck_tile_fused_moe_flatmm_uk> 64 4 2 256 256 1 2 2 0 11939 2)

add_test(NAME example_hcu_ck_tile_fused_moe_flatmm_uk_tiny_hidden
    COMMAND $<TARGET_FILE:example_hcu_ck_tile_fused_moe_flatmm_uk> 8192 128 64 1 1 1 2 2 0)
add_test(NAME example_hcu_ck_tile_fused_moe_flatmm_uk_ragged_hidden
    COMMAND $<TARGET_FILE:example_hcu_ck_tile_fused_moe_flatmm_uk> 64 4 2 3 256 1 2 2 0)

set_tests_properties(example_hcu_ck_tile_fused_moe_flatmm_uk_smoke
                     example_hcu_ck_tile_fused_moe_flatmm_uk_partial_interm
                     example_hcu_ck_tile_fused_moe_flatmm_uk_mock_id
                     example_hcu_ck_tile_fused_moe_flatmm_uk_tiny_hidden
                     example_hcu_ck_tile_fused_moe_flatmm_uk_ragged_hidden
                     PROPERTIES LABELS "hcu;ck_tile;fused_moe;smoke")

set(TILE_EXAPMLE_FUSED_MOE "tile_example_fused_moe")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding ${TILE_EXAPMLE_FUSED_MOE}")
file(GLOB INSTANCE_SRCS instances/*.cpp)
ck_add_arch_vgpr_compile_options()
add_compile_options("SHELL: -Rpass-analysis=kernel-resource-usage")
add_executable(${TILE_EXAPMLE_FUSED_MOE} EXCLUDE_FROM_ALL main.cpp)
target_include_directories(${TILE_EXAPMLE_FUSED_MOE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${TILE_EXAPMLE_FUSED_MOE} PRIVATE ${INSTANCE_SRCS})

set(TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS -DCK_TILE_BUFFER_LOAD_AGPR=1) # TODO: enable load to a
list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=4) # rta

# list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS  -mllvm -greedy-reverse-local-assignment=1)
# list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)

target_compile_options(${TILE_EXAPMLE_FUSED_MOE} PRIVATE ${TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS})
