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

// v7 BF16 NN / TN instances. These routes are OPT-IN (the selector policy
// fields default to false, see grouped_gemm_selector.hpp), so unlike the two
// switches above the enable side is the one that has to be set explicitly. The
// disable side still wins, which is what makes a same-process A/B able to pin
// either state:
//   CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN=1  -> route BF16 NN to instance 5000
//   CK_TILE_GROUPED_GEMM_ENABLE_BF16_V8_TN=1  -> route BF16 TN to instance 5001
//   CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_V8*=1  -> force the route off
inline bool ck_tile_hcu_grouped_gemm_bf16_v7_nn_enabled()
{
    const char* disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN");
    if(disable != nullptr && disable[0] == '1' && disable[1] == '\0')
        return false;
    const char* enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN");
    return enable != nullptr && enable[0] == '1' && enable[1] == '\0';
}

inline bool ck_tile_hcu_grouped_gemm_bf16_v8_tn_enabled()
{
    const char* disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V8_TN");
    if(disable != nullptr && disable[0] == '1' && disable[1] == '\0')
        return false;
    const char* enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V8_TN");
    return enable != nullptr && enable[0] == '1' && enable[1] == '\0';
}

// The v7 device-args entries. Declared unconditionally (with C linkage) so the
// dispatch can name them; whether they exist at link time is what
// CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_V7_V8_BF16 guards.
//
// `tile_slots` is an upper bound on the (group, m tile, n tile) tile count,
// rounded up to a whole number of CTAs. The v7 configs are NON-persistent --
// one tile per CTA, which is exactly what keeps their register budget at the
// reference kernels' level (a wrapping tile loop costs 40 VGPRs and ~20%) -- so
// the launcher has to size a flat grid, and per-group M is device-resident. The
// bound comes from the host-visible problem dimensions and is never read back.
extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream,
    std::uint64_t tile_slots);
extern "C" int ck_tile_hcu_grouped_gemm_v8_bf16_tn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream,
    std::uint64_t tile_slots);

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
