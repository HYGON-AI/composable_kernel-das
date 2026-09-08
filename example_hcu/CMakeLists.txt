# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

add_custom_target(examples)
option(CK_BUILD_UNSTABLE_HCU_EXAMPLES "Build unstable HCU examples" OFF)
option(CK_ENABLE_ARCH_VGPR_HINT "Enable arch-based support-xxx-vgprs flags" ON)

function(ck_add_arch_vgpr_compile_options)
    if(NOT CK_ENABLE_ARCH_VGPR_HINT)
        return()
    endif()

    if(NOT DEFINED GPU_TARGETS OR "${GPU_TARGETS}" STREQUAL "")
        return()
    endif()

    string(TOLOWER "${GPU_TARGETS}" _gpu_targets_lc)

    set(_enable_768 OFF)
    set(_enable_512 OFF)

    if(_gpu_targets_lc MATCHES "gfx936" OR _gpu_targets_lc MATCHES "gfx938")
        set(_enable_768 ON)
    endif()

    if(_gpu_targets_lc MATCHES "gfx928" OR _gpu_targets_lc MATCHES "gfx92a" OR
       _gpu_targets_lc MATCHES "gfx946")
        set(_enable_512 ON)
    endif()

    if(_enable_768 AND _enable_512)
        message(WARNING "GPU_TARGETS includes both 768-VGPR and 512-VGPR target families; skip support-xxx-vgprs hint.")
        return()
    endif()

    if(_enable_768)
        add_compile_options("SHELL: -mllvm -support-768-vgprs=true")
    elseif(_enable_512)
        add_compile_options("SHELL: -mllvm -support-512-vgprs=true")
    endif()
endfunction()

function(add_example_executable EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${FILE_NAME})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE utility)
    add_test(NAME ${EXAMPLE_NAME} COMMAND $<TARGET_FILE:${EXAMPLE_NAME}> ${ARGN})
    add_dependencies(examples ${EXAMPLE_NAME})
    add_dependencies(check ${EXAMPLE_NAME})
    rocm_install(TARGETS ${EXAMPLE_NAME} COMPONENT examples)
endfunction(add_example_executable EXAMPLE_NAME)

function(add_example_executable_no_testing EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${FILE_NAME})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE utility)
    add_dependencies(examples ${EXAMPLE_NAME})
    rocm_install(TARGETS ${EXAMPLE_NAME} COMPONENT examples)
endfunction(add_example_executable_no_testing EXAMPLE_NAME)

# add all example subdir
file(GLOB dir_list LIST_DIRECTORIES true *)
FOREACH(subdir ${dir_list})
    IF(IS_DIRECTORY "${subdir}")
        add_subdirectory(${subdir})
    ENDIF()
ENDFOREACH()
