# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.
add_example_executable(tile_example_gdn_prefill gdn_prefill.cpp)
target_sources(tile_example_gdn_prefill PRIVATE
  prefill/gdn_prefill_launch.cpp
  prefill/gdn_preprocess_instance.cpp
  prefill/gdn_cumsum_instance.cpp
  prefill/gdn_kkt_instance.cpp
  prefill/gdn_recompute_instance.cpp
  prefill/gdn_state_instance.cpp
  prefill/gdn_output_instance.cpp)

target_compile_options(tile_example_gdn_prefill PRIVATE
  -O3
  -DNDEBUG
  -Wno-shadow
  -Wno-float-equal
  -Wno-unused-variable
  -Wno-unused-parameter
  -Wno-duplicate-decl-specifier
  -Wno-unused-local-typedef
  -Wno-error=unused-parameter
  -Wno-error=unused-macros
  -Wno-error=disabled-macro-expansion
  -Wno-error=undefined-reinterpret-cast
  -Wno-pass-failed
  "SHELL:-mllvm -enable-num-vgprs-768=true")

add_example_executable(tile_example_gdn_decode gdn_decode.cpp)
target_sources(tile_example_gdn_decode PRIVATE
  decode/gdn_decode_instance.cpp)

target_compile_options(tile_example_gdn_decode PRIVATE
  -O3
  -DNDEBUG
  -Wno-shadow
  -Wno-float-equal
  -Wno-unused-variable
  -Wno-unused-parameter
  -Wno-unused-local-typedef
  -Wno-error=unused-parameter
  -Wno-error=unused-macros
  -Wno-error=disabled-macro-expansion
  -Wno-error=undefined-reinterpret-cast
  -Wno-pass-failed
  "SHELL:-mllvm -enable-num-vgprs-768=true")
