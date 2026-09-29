# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
add_example_executable(tile_example_sla_fwd sla_fwd.cpp)
target_sources(tile_example_sla_fwd PRIVATE
  instances/sla_fwd_bf16_instance.cpp
  instances/sla_fwd_fp16_instance.cpp)

target_include_directories(tile_example_sla_fwd PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})

add_example_executable(tile_example_sla_bwd sla_bwd.cpp)
add_example_executable(tile_example_sla_sizes test_sla_sizes.cpp)
# The size-only test includes the runner without instantiating its GPU launchers.
target_compile_options(tile_example_sla_sizes PRIVATE -Wno-unused-template)
target_sources(tile_example_sla_bwd PRIVATE
  instances/sla_bwd_bf16_instance.cpp
  instances/sla_bwd_fp16_instance.cpp)

target_include_directories(tile_example_sla_bwd PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})

set(_sla_extension_flags_default OFF)
if(CMAKE_CXX_COMPILER MATCHES "/opt/dtk/bin/hipcc$")
  set(_sla_extension_flags_default ON)
endif()

option(SLA_MATCH_EXTENSION_FLAGS
  "Compile SLA with HIP device-side defines"
  ${_sla_extension_flags_default})

set(_sla_targets
  tile_example_sla_fwd
  tile_example_sla_bwd
  tile_example_sla_sizes)

foreach(_sla_target IN LISTS _sla_targets)
  target_compile_options(${_sla_target} PRIVATE
    -O3 -DNDEBUG -g -fwrapv -fPIC
    -Wno-shadow -Wno-float-equal -Wno-unused-variable -Wno-unused-local-typedef
    -Wno-disabled-macro-expansion -Wno-unused-macros -Wno-pass-failed)

  if(SLA_MATCH_EXTENSION_FLAGS)
    target_compile_definitions(${_sla_target} PRIVATE
      USE_ROCM=1
      HIPBLAS_V2
      CUDA_HAS_FP16=1
      HIP_ENABLE_WARP_SYNC_BUILTINS=1
      CK_EXPERIMENTAL_BIT_INT_EXTENSION
      SLA_FWD_HEAD_DIM=128)
    target_compile_options(${_sla_target} PRIVATE
      --rocm-device-lib-path=/opt/dtk/amdgcn/bitcode
      -I/opt/dtk/llvm/lib/clang/17.0.0/include
      -ffast-math
      -fno-gpu-rdc
      -w
      -U__HIP_NO_HALF_OPERATORS__
      -U__HIP_NO_HALF_CONVERSIONS__
      "SHELL:-mllvm -amdgpu-spill-vgpr-to-agpr=true")
  endif()
endforeach()

# setup.py adds this device register limit only to the dQ and dK/dV sources.
# The example keeps those kernels in the BWD translation unit.
target_compile_options(tile_example_sla_bwd PRIVATE
  -Xarch_gfx936
  -mllvm=-enable-num-vgprs-768=true)
