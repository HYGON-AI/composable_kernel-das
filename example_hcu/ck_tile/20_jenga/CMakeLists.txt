# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.
add_example_executable(tile_example_jenga_fwd jenga_fwd.cpp)
target_sources(tile_example_jenga_fwd PRIVATE
  instances/jenga_fwd_bf16_instance.cpp
  instances/jenga_fwd_fp16_instance.cpp)

target_include_directories(tile_example_jenga_fwd PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})

add_example_executable(tile_example_jenga_bwd jenga_bwd.cpp)
target_sources(tile_example_jenga_bwd PRIVATE
  instances/jenga_bwd_dq_instance.cpp
  instances/jenga_bwd_dkdv_instance.cpp)

target_include_directories(tile_example_jenga_bwd PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})

set(_jenga_extension_flags_default OFF)
if(CMAKE_CXX_COMPILER MATCHES "/opt/dtk/bin/hipcc$")
  set(_jenga_extension_flags_default ON)
endif()
option(JENGA_MATCH_EXTENSION_FLAGS
  "Compile Jenga with the HIP/PyTorch extension device-side defines"
  ${_jenga_extension_flags_default})
if(JENGA_MATCH_EXTENSION_FLAGS)
  target_compile_definitions(tile_example_jenga_bwd PRIVATE
    USE_ROCM=1
    HIPBLAS_V2
    CUDA_HAS_FP16=1
    HIP_ENABLE_WARP_SYNC_BUILTINS=1)
  target_compile_options(tile_example_jenga_bwd PRIVATE
    --rocm-device-lib-path=/opt/dtk/amdgcn/bitcode
    -fno-gpu-rdc
    -w)
endif()

# The migrated experimental kernels predate the repository-wide -Weverything
# policy and intentionally compare softmax sentinels with infinity.
target_compile_options(tile_example_jenga_fwd PRIVATE
  -O3 -DNDEBUG
  -Wno-shadow -Wno-float-equal -Wno-unused-variable -Wno-unused-local-typedef)
target_compile_options(tile_example_jenga_bwd PRIVATE
  -O3 -DNDEBUG
  -Wno-shadow -Wno-float-equal -Wno-unused-variable -Wno-unused-local-typedef
  -Wno-pass-failed
  "SHELL:-mllvm -amdgpu-spill-vgpr-to-agpr=true")

# Match the extension build: only dQ is compiled with the 768-VGPR mode.
set_property(SOURCE instances/jenga_bwd_dq_instance.cpp APPEND PROPERTY COMPILE_OPTIONS
  -mllvm
  -enable-num-vgprs-768=true)
