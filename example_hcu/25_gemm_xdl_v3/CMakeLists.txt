# Copyright (c) 2026 Hygon Info Technologies Ltd.
# SPDX-License-Identifier: MIT

# hcu GEMM XDL CShuffle v3 examples
add_custom_target(example_hcu_gemm_xdl_v3)

add_example_executable(example_hcu_gemm_xdl_bf16_v3 gemm_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_multi_d_xdl_bf16_v3 gemm_multi_d_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_multi_abd_xdl_bf16_v3 gemm_multi_abd_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_add_xdl_bf16_v3 gemm_add_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_add_relu_xdl_bf16_v3 gemm_add_relu_xdl_bf16_v3.cpp)
add_example_executable(example_hcu_gemm_xdl_bf16_mk_nk_mn_v3 gemm_xdl_bf16_mk_nk_mn_v3.cpp)
add_example_executable(example_hcu_gemm_xdl_bf16_mk_kn_mn_v3 gemm_xdl_bf16_mk_kn_mn_v3.cpp)
add_example_executable(example_hcu_gemm_xdl_bf16_km_nk_mn_v3 gemm_xdl_bf16_km_nk_mn_v3.cpp)
add_example_executable(example_hcu_gemm_xdl_bf16_km_kn_mn_v3 gemm_xdl_bf16_km_kn_mn_v3.cpp)

add_dependencies(example_hcu_gemm_xdl_v3
                 example_hcu_gemm_xdl_bf16_v3
                 example_hcu_gemm_multi_d_xdl_bf16_v3
                 example_hcu_gemm_multi_abd_xdl_bf16_v3
                 example_hcu_gemm_add_xdl_bf16_v3
                 example_hcu_gemm_add_relu_xdl_bf16_v3
                 example_hcu_gemm_xdl_bf16_mk_nk_mn_v3
                 example_hcu_gemm_xdl_bf16_mk_kn_mn_v3
                 example_hcu_gemm_xdl_bf16_km_nk_mn_v3
                 example_hcu_gemm_xdl_bf16_km_kn_mn_v3)

add_custom_target(example_hcu_gemm_multi_d_f16_v3)
foreach(layout mk_kn_mn mk_nk_mn km_kn_mn km_nk_mn)
    set(target example_hcu_gemm_multi_d_xdl_f16_${layout}_v3)
    add_example_executable(${target} gemm_multi_d_xdl_f16_${layout}_v3.cpp --test)
    add_dependencies(example_hcu_gemm_multi_d_f16_v3 ${target})
    add_dependencies(example_hcu_gemm_xdl_v3 ${target})
endforeach()
