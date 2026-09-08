// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "grouped_gemm_impl.hpp"

// --- C ABI (uses GemmConfigComputeBf8K64 per current code) ---
int grouped_gemm_c_run_bf8(const ck_tile_hcu_grouped_gemm_desc* descs,
                           int group_count,
                           char a_layout,
                           char b_layout,
                           void* workspace,
                           hipStream_t stream)
{
    return dispatch_grouped_gemm_c_layout<GemmConfigComputeBf8K64<ck_tile::bf8_t>, ck_tile::bf8_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_c_run_bf8_mpad(const ck_tile_hcu_grouped_gemm_desc* descs,
                                 int group_count,
                                 char a_layout,
                                 char b_layout,
                                 void* workspace,
                                 hipStream_t stream)
{
    return dispatch_grouped_gemm_c_layout<GemmConfigComputeBf8K64Mpad<ck_tile::bf8_t>, ck_tile::bf8_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

// --- Example Runner (multiple config variants) ---
int grouped_gemm_example_run_bf8_k64(std::string a_layout, std::string b_layout, int argc,
                                     char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeBf8K64<ck_tile::bf8_t>, ck_tile::bf8_t>(
        a_layout, b_layout, argc, argv);
}

int grouped_gemm_example_run_bf8_128x64(std::string a_layout, std::string b_layout, int argc,
                                        char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeBf8<ck_tile::bf8_t>, ck_tile::bf8_t>(
        a_layout, b_layout, argc, argv);
}

int grouped_gemm_example_run_bf8_128x128(std::string a_layout, std::string b_layout, int argc,
                                         char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeBf8V2<ck_tile::bf8_t>, ck_tile::bf8_t>(
        a_layout, b_layout, argc, argv);
}
