// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdlib>
#include <hip/hip_runtime.h>

// CK-owned default for the validated gfx936 BF16 NN descriptor families.
// Keep one explicit rollback switch for production and same-process A/B.
inline bool ck_tile_hcu_grouped_gemm_bf16_nn_fixed_srd_lds8_enabled()
{
    const char* disable =
        std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_GFX936_BF16_NN_FIXED_SRD_LDS8");
    return !(disable != nullptr && disable[0] == '1' && disable[1] == '\0');
}

// CK-owned gfx936 BF16 TN logical-K-tail default. The candidate has an
// explicit rollback switch so downstream integrations can perform same-
// process A/B without changing the aligned/generic fallback portfolio.
inline bool ck_tile_hcu_grouped_gemm_bf16_tn_logical_k_tail_enabled()
{
    const char* disable = std::getenv(
        "CK_TILE_GROUPED_GEMM_DISABLE_GFX936_BF16_TN_LOGICAL_K_TAIL");
    return !(disable != nullptr && disable[0] == '1' && disable[1] == '\0');
}

// Launch the selected BW-family FP16/BF16 grouped-GEMM kernels from an
// existing device array of ck_tile::GemmTransKernelArg. The validated family
// currently covers gfx936 and gfx938 under the same aligned-shape contract.
#define CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(suffix)                        \
    extern "C" int ck_tile_hcu_grouped_gemm_bw_family_##suffix##_device_args(       \
        void* device_args,                                                           \
        int group_count,                                                             \
        std::uint32_t num_cu,                                                        \
        hipStream_t stream);                                                         \
    extern "C" int ck_tile_hcu_grouped_gemm_gfx936_##suffix##_device_args(           \
        void* device_args,                                                           \
        int group_count,                                                             \
        std::uint32_t num_cu,                                                        \
        hipStream_t stream)

CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(fp16);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(fp16_nn);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16_nn);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16_tn_blas_transposed);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(
    bf16_tn_blas_transposed_logical_k_tail);

#undef CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY
