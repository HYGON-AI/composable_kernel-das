// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// Promoted gfx936 BF16 NN fixed-SRD production specialization. It keeps a
// complete LDS/VMEM drain at the K64 circular-slot boundary while steady K16
// intervals retain lgkmcnt(8)/vmcnt(4).
#define CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG 93701
#define CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LDS_WAIT_COUNT 8
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_LDS_WAIT_COUNT 0
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_VMEM_WAIT_COUNT 0
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SETPRIO_AFTER_FIRST
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD
#define CK_TILE_GROUPED_GEMM_BF16_NN_RR_BOUNDARY_HOST_ENTRY \
    grouped_gemm_c_run_gfx936_bf16_nn_fixed_srd_lds8_boundary_lds0_vmem0
#define CK_TILE_GROUPED_GEMM_BF16_NN_RR_BOUNDARY_ENTRY \
    ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_rr_fixed_srd_lds8_boundary_lds0_vmem0_device_args
#define CK_TILE_GROUPED_GEMM_BF16_NN_RR_BOUNDARY_COMPAT_ENTRY \
    ck_tile_hcu_grouped_gemm_gfx936_bf16_nn_rr_fixed_srd_lds8_boundary_lds0_vmem0_device_args
#include "grouped_gemm_gfx936_bf16_nn_rr_boundary_vmem.cpp"
