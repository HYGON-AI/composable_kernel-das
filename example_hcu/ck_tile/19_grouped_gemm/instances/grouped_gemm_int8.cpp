// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "grouped_gemm_impl.hpp"

// --- C ABI (uses GemmConfigComputeV7 per current code) ---
int grouped_gemm_c_run_int8(const ck_tile_hcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream)
{
    return dispatch_grouped_gemm_c_layout<GemmConfigComputeV6<ck_tile::int8_t>, ck_tile::int8_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

// --- Example Runner (two config variants) ---
int grouped_gemm_example_run_int8_32x32(std::string a_layout, std::string b_layout, int argc,
                                        char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeV6<ck_tile::int8_t>, ck_tile::int8_t>(
        a_layout, b_layout, argc, argv);
}

int grouped_gemm_example_run_int8(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeV7<ck_tile::int8_t>, ck_tile::int8_t>(
        a_layout, b_layout, argc, argv);
}
