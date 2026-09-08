// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "grouped_gemm_impl.hpp"

int grouped_gemm_c_run_fp8(const ck_tile_hcu_grouped_gemm_desc* descs,
                           int group_count,
                           char a_layout,
                           char b_layout,
                           void* workspace,
                           hipStream_t stream)
{
    return dispatch_grouped_gemm_c_layout<GemmConfigComputeV5<ck_tile::fp8_t>, ck_tile::fp8_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_c_run_fp8_mpad(const ck_tile_hcu_grouped_gemm_desc* descs,
                                 int group_count,
                                 char a_layout,
                                 char b_layout,
                                 void* workspace,
                                 hipStream_t stream)
{
    return dispatch_grouped_gemm_c_layout<GemmConfigComputeV5Mpad<ck_tile::fp8_t>, ck_tile::fp8_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_example_run_fp8(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeV5<ck_tile::fp8_t>, ck_tile::fp8_t>(
        a_layout, b_layout, argc, argv);
}

int grouped_gemm_example_run_fp8_128x64(std::string a_layout,
                                        std::string b_layout,
                                        int argc,
                                        char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeFp8N64K64<ck_tile::fp8_t>, ck_tile::fp8_t>(
        a_layout, b_layout, argc, argv);
}

int grouped_gemm_example_run_fp8_128x128_k32(std::string a_layout,
                                             std::string b_layout,
                                             int argc,
                                             char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeBf8V2<ck_tile::fp8_t>, ck_tile::fp8_t>(
        a_layout, b_layout, argc, argv);
}
