// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"

std::size_t ck_tile_hcu_grouped_gemm_workspace_size(int group_count, int num_d_tensors)
{
    if(group_count <= 0)
        return 0;
    if(num_d_tensors <= 0)
        return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArg);
    if(num_d_tensors == 1)
        return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArgImpl<1>);
    return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArgImpl<2>);
}

namespace {

// For MOE with dynamic M per group, use the kPadM instance when any group
// has M not aligned to the tile boundary.
//   fp16/bf16 V4: MPerBlock=64 → check M % 64
//   fp8/bf8 V5/Bf8K64: MPerBlock=128 → check M % 128
[[maybe_unused]] bool grouped_gemm_need_mpad(const ck_tile_hcu_grouped_gemm_desc* descs, int group_count, int align_m = 64)
{
    for(int i = 0; i < group_count; ++i)
    {
        if(descs[i].M % align_m != 0)
            return true;
    }
    return false;
}

// Keep aligned FP16/BF16 problems on the original vectorized fast config. Select the
// all-dimension padded config only when a group has an M/N tile tail or a per-split K tail.
[[maybe_unused]] bool grouped_gemm_need_v4_padding(
    const ck_tile_hcu_grouped_gemm_desc* descs, int group_count)
{
    constexpr int align_m = 64;
    constexpr int align_n = 128;
    constexpr int align_k = 64;
    for(int i = 0; i < group_count; ++i)
    {
        const int kbatch     = descs[i].k_batch > 0 ? descs[i].k_batch : 1;
        const int k_alignment = align_k * kbatch;
        if(descs[i].M % align_m != 0 || descs[i].N % align_n != 0 ||
           descs[i].K % k_alignment != 0)
        {
            return true;
        }
    }
    return false;
}

} // namespace

int ck_tile_hcu_grouped_gemm_run(const ck_tile_hcu_grouped_gemm_desc* descs,
                                 int group_count,
                                 int dtype,
                                 char a_layout,
                                 char b_layout,
                                 void* workspace,
                                 hipStream_t stream)
{
    try
    {
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_FP16)
        if(dtype == CK_TILE_HCU_GROUPED_GEMM_FP16)
        {
            if(grouped_gemm_need_v4_padding(descs, group_count))
                return grouped_gemm_c_run_fp16_mpad(descs, group_count,
                    a_layout, b_layout, workspace, stream);
            return grouped_gemm_c_run_fp16(descs, group_count,
                a_layout, b_layout, workspace, stream);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_FP8)
        if(dtype == CK_TILE_HCU_GROUPED_GEMM_FP8)
        {
            if(grouped_gemm_need_mpad(descs, group_count, 128))
                return grouped_gemm_c_run_fp8_mpad(descs, group_count,
                    a_layout, b_layout, workspace, stream);
            return grouped_gemm_c_run_fp8(descs, group_count,
                a_layout, b_layout, workspace, stream);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_BF16)
        if(dtype == CK_TILE_HCU_GROUPED_GEMM_BF16)
        {
#if defined(CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED)
            if(ck_tile::get_hcu_target_enum() ==
                   ck_tile::hcu_target_enum::gfx936 &&
               a_layout == 'C' && b_layout == 'R' &&
               grouped_gemm_prefers_gfx936_bf16_tn_logical_k_tail(
                   descs, group_count))
            {
                return grouped_gemm_c_run_gfx936_bf16_tn_blas_transposed_logical_k_tail(
                    descs, group_count, a_layout, b_layout, workspace, stream);
            }
            if(ck_tile::get_hcu_target_enum() ==
                   ck_tile::hcu_target_enum::gfx936 &&
               a_layout == 'R' && b_layout == 'R')
            {
                if(grouped_gemm_prefers_gfx936_bf16_nn_rr_fast_vmem3_early_final_barrier(
                       descs, group_count))
                {
                    return grouped_gemm_c_run_gfx936_bf16_nn_rr_fast_vmem3_early_final_barrier(
                        descs, group_count, a_layout, b_layout, workspace, stream);
                }
                if(grouped_gemm_prefers_gfx936_bf16_nn_fixed_srd_lds8(
                       descs, group_count))
                {
                    return grouped_gemm_c_run_gfx936_bf16_nn_fixed_srd_lds8_boundary_lds0_vmem0(
                        descs, group_count, a_layout, b_layout, workspace, stream);
                }
            }
#endif
            if(grouped_gemm_need_v4_padding(descs, group_count))
                return grouped_gemm_c_run_bf16_mpad(descs, group_count,
                    a_layout, b_layout, workspace, stream);
            return grouped_gemm_c_run_bf16(descs, group_count,
                a_layout, b_layout, workspace, stream);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_BF8)
        if(dtype == CK_TILE_HCU_GROUPED_GEMM_BF8)
        {
            if(grouped_gemm_need_mpad(descs, group_count, 128))
                return grouped_gemm_c_run_bf8_mpad(descs, group_count,
                    a_layout, b_layout, workspace, stream);
            return grouped_gemm_c_run_bf8(descs, group_count,
                a_layout, b_layout, workspace, stream);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_INT8)
        if(dtype == CK_TILE_HCU_GROUPED_GEMM_INT8)
        {
            return grouped_gemm_c_run_int8(descs, group_count,
                a_layout, b_layout, workspace, stream);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_INT4)
        if(dtype == CK_TILE_HCU_GROUPED_GEMM_INT4)
        {
            return grouped_gemm_c_run_int4(descs, group_count,
                a_layout, b_layout, workspace, stream);
        }
#endif
        return -5;
    }
    catch(...)
    {
        return -100;
    }
}

// 不要删除此宏定义，其用于在编译时禁用 main 函数，避免与其他模块的 main 冲突。
#ifndef CK_TILE_EXAMPLE_NO_MAIN
int run_grouped_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return 0;
    }

    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string data_type = arg_parser.get_str("prec");
    const std::string config    = arg_parser.get_str("config");

#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_FP16)
    if(data_type == "fp16")
    {
        return grouped_gemm_example_run_fp16(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_FP8)
    if(data_type == "fp8")
    {
        if(config == "fp8_128x64")
            return grouped_gemm_example_run_fp8_128x64(a_layout, b_layout, argc, argv);
        if(config == "fp8_128x128_k32")
            return grouped_gemm_example_run_fp8_128x128_k32(a_layout, b_layout, argc, argv);
        return grouped_gemm_example_run_fp8(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_BF16)
    if(data_type == "bf16")
    {
        return grouped_gemm_example_run_bf16(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_BF8)
    if(data_type == "bf8")
    {
        if(config == "bf8_128x64")
            return grouped_gemm_example_run_bf8_128x64(a_layout, b_layout, argc, argv);
        if(config == "bf8_k64")
            return grouped_gemm_example_run_bf8_k64(a_layout, b_layout, argc, argv);
        if(config == "bf8_128x128")
            return grouped_gemm_example_run_bf8_128x128(a_layout, b_layout, argc, argv);
        return grouped_gemm_example_run_bf8_k64(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_INT8)
    if(data_type == "int8")
    {
        if(config == "int8_32x32")
            return grouped_gemm_example_run_int8_32x32(a_layout, b_layout, argc, argv);
        return grouped_gemm_example_run_int8(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_INT4)
    if(data_type == "int4")
    {
        return grouped_gemm_example_run_int4(a_layout, b_layout, argc, argv);
    }
    else
#endif
    {
        std::cerr << "Unsupported data type configuration: " << data_type << std::endl;
        return 0;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        return run_grouped_gemm_example(argc, argv) ? 0 : 1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
#endif
