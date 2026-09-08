// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// BF16 grad-W is exposed as row-major C = A^T B.  This selected instance
// consumes device arguments rewritten to the mathematically equivalent
// column-major C^T = B^T A, so its scalar stores are contiguous along M just
// like the matching hipBLASLt solution.
#ifndef CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG
#define CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG 93602
#endif
#ifndef CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
#endif
#ifndef CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
#endif
#ifndef CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
#endif
#ifndef CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE
#define CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE
#endif
#ifndef CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
#define CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
#endif
#ifndef CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
#define CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
#endif
#ifndef CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
#define CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
#endif
#ifndef CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
#define CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
#endif
#ifndef CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
#define CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
#endif
#ifndef CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
#endif
#include "grouped_gemm_impl.hpp"

#ifndef CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_ENTRY
#define CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_ENTRY \
    ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_device_args
#endif

#ifndef CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_COMPAT_ENTRY
#define CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_COMPAT_ENTRY \
    ck_tile_hcu_grouped_gemm_gfx936_bf16_tn_blas_transposed_device_args
#endif

extern "C" int CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_ENTRY(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    if(device_args == nullptr || group_count <= 0)
    {
        return -1;
    }

    using Config = GemmConfigComputeV3DsreadmStage256W8<ck_tile::bf16_t>;
    using Row    = ck_tile::tensor_layout::gemm::RowMajor;
    using Col    = ck_tile::tensor_layout::gemm::ColumnMajor;

    grouped_gemm_tileloop<Config,
                          Col,
                          Row,
                          Col,
                          ck_tile::bf16_t,
                          ck_tile::bf16_t,
                          float,
                          ck_tile::bf16_t,
#if defined(CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG)
                          CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG
#else
                          0
#endif
                          >(ck_tile::stream_config{stream},
                            group_count,
                            device_args,
                            {},
                            false,
                            num_cu);
    return 0;
}

extern "C" int CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_COMPAT_ENTRY(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_ENTRY(
        device_args, group_count, num_cu, stream);
}
