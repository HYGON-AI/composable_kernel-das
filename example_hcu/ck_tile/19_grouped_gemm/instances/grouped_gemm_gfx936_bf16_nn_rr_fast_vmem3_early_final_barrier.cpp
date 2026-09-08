// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// Promoted gfx936 BF16 NN production specialization. The early source barrier
// aligns all eight waves, and the final inline barrier prevents a wave from
// refilling circular LDS while another wave still consumes the current slot.
#define CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG 93714
#define CK_TILE_GROUPED_GEMM_GRID_MULTIPLIER_FIXED 2
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FULL_LDS_WAIT
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_VMEM_WAIT_COUNT 3
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_EARLY_AND_FINAL_INLINE_BARRIER
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SETPRIO_AFTER_FIRST
#define CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_HOST_ENTRY \
    grouped_gemm_c_run_gfx936_bf16_nn_rr_fast_vmem3_early_final_barrier
#define CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_ENTRY \
    ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_rr_fast_vmem3_early_final_barrier_ck_device_args
#define CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_COMPAT_ENTRY \
    ck_tile_hcu_grouped_gemm_gfx936_bf16_nn_rr_fast_vmem3_early_final_barrier_ck_device_args
#include "grouped_gemm_gfx936_bf16_nn_rr_blas.cpp"
