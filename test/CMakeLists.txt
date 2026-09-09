# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/profiler/include
)

add_custom_target(tests)

function(add_test_executable TEST_NAME)
    message("adding test ${TEST_NAME}")
    add_executable(${TEST_NAME} ${ARGN})
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
    rocm_install(TARGETS ${TEST_NAME} COMPONENT tests)
endfunction(add_test_executable TEST_NAME)

find_package(GTest REQUIRED)

function(add_gtest_executable TEST_NAME)
    message("adding gtest ${TEST_NAME}")
    add_executable(${TEST_NAME} ${ARGN})
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})

    # suppress gtest warnings
    target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_libraries(${TEST_NAME} PRIVATE ${GTEST_MAIN_LIBRARIES})
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    rocm_install(TARGETS ${TEST_NAME} COMPONENT tests)
endfunction(add_gtest_executable TEST_NAME)

add_subdirectory(grouped_conv2d_fwd_mmac)
add_subdirectory(grouped_conv2d_bwd_data_mmac)
add_subdirectory(grouped_conv2d_bwd_weight_mmac)
add_subdirectory(contraction)
add_subdirectory(reduce)
add_subdirectory(softmax)
add_subdirectory(permute_scale)