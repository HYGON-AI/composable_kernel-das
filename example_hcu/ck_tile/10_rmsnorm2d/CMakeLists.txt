# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set(RMSNORM2D_FWD_KNOWN_APIS "fwd;bwd")
set(RMSNORM2D_FWD_ENABLE_APIS  "fwd" CACHE STRING
    "semicolon-separated list of APIs to generate (${RMSNORM2D_FWD_KNOWN_APIS}) & link, or \"all\".")
set(Python3_EXECUTABLE "python3")
if(RMSNORM2D_FWD_ENABLE_APIS  STREQUAL "all")
  set(RMSNORM2D_FWD_ENABLE_APIS  ${RMSNORM2D_FWD_KNOWN_APIS})
endif()

# generate a list of kernels, but not actually emit files at config sta
set(RMSNORM2D_FWD_CODEGEN_OPTIONS)
# Match AITER's FP8 policy; this is independent of gfx946 multi-die launch.
foreach(fp8_arch IN ITEMS gfx938 gfx92a gfx946)
  if(fp8_arch IN_LIST GPU_TARGETS OR fp8_arch IN_LIST CMAKE_HIP_ARCHITECTURES)
    list(APPEND RMSNORM2D_FWD_CODEGEN_OPTIONS --enable-fp8)
    break()
  endif()
endforeach()
if(NOT RMSNORM2D_FWD_CODEGEN_OPTIONS AND "$ENV{AITER_ENABLE_FP8}" GREATER 0)
  list(APPEND RMSNORM2D_FWD_CODEGEN_OPTIONS --enable-fp8)
endif()
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api ${RMSNORM2D_FWD_ENABLE_APIS} --working_path ${CMAKE_CURRENT_BINARY_DIR} --list_blobs ${RMSNORM2D_FWD_CODEGEN_OPTIONS}
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "Fail to generate kernels via Python. ${ret}")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/rmsnorm2d_fwd_blobs.txt RMSNORM2D_FWD_GEN_BLOBS)

add_custom_command(
  OUTPUT ${RMSNORM2D_FWD_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api ${RMSNORM2D_FWD_ENABLE_APIS} --working_path ${CMAKE_CURRENT_BINARY_DIR} --gen_blobs ${RMSNORM2D_FWD_CODEGEN_OPTIONS}
)

set(TILE_RMSNORM2D_FWD "tile_rmsnorm2d_fwd")

message("adding ${TILE_RMSNORM2D_FWD}")
add_executable(${TILE_RMSNORM2D_FWD} EXCLUDE_FROM_ALL rmsnorm2d_fwd.cpp)
target_include_directories(${TILE_RMSNORM2D_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${TILE_RMSNORM2D_FWD} PRIVATE ${RMSNORM2D_FWD_GEN_BLOBS})

set(TILE_RMSNORM2D_FWD_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
# list(APPEND TILE_RMSNORM2D_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal --offload-compress)
#Note: Disable offload-compress
list(APPEND TILE_RMSNORM2D_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)


target_compile_options(${TILE_RMSNORM2D_FWD} PRIVATE ${TILE_RMSNORM2D_FWD_COMPILE_OPTIONS})

if(RMSNORM2D_FWD_CODEGEN_OPTIONS)
  target_compile_definitions(${TILE_RMSNORM2D_FWD} PRIVATE GPU_ENABLE_FP8)
endif()

# Match AITER JIT: compile both paths for gfx946; select by runtime HIP topology.
if("gfx946" IN_LIST GPU_TARGETS OR "gfx946" IN_LIST CMAKE_HIP_ARCHITECTURES)
  target_compile_definitions(${TILE_RMSNORM2D_FWD} PRIVATE CK_TILE_RMSNORM_ENABLE_MULTIDIE=1)
endif()

set(EXAMPLE_RMSNORM2D_FWD "tile_example_rmsnorm2d_fwd")
add_executable(${EXAMPLE_RMSNORM2D_FWD} EXCLUDE_FROM_ALL example_rmsnorm2d_fwd.cpp)
target_compile_options(${EXAMPLE_RMSNORM2D_FWD} PRIVATE ${TILE_RMSNORM2D_FWD_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
