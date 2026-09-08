// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// gfx936 BF16 TN logical-K-tail production instance. Full K64 tiles keep the
// submitted BLAS-transposed DSReadM instruction stream; only the physically
// peeled final K64 tile applies the vendor voffset=-1 tail convention.
#define CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG 93607
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
#define CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE
#define CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED
#define CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT
#define CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM
#define CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_DIRECT_SCALAR_SHORT
#define CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
#define CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED
#define CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_ENTRY \
    ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_logical_k_tail_device_args
#define CK_TILE_GROUPED_GEMM_BF16_TN_BLAS_TRANSPOSED_COMPAT_ENTRY \
    ck_tile_hcu_grouped_gemm_gfx936_bf16_tn_blas_transposed_logical_k_tail_device_args
#include "grouped_gemm_gfx936_bf16_tn_blas_transposed.cpp"

#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_ONLY)
int grouped_gemm_c_run_gfx936_bf16_tn_blas_transposed_logical_k_tail(
    const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count,
    char a_layout,
    char b_layout,
    void* workspace,
    hipStream_t stream)
{
    if(descs == nullptr || group_count <= 0 || a_layout != 'C' || b_layout != 'R')
    {
        return -1;
    }

    std::vector<ck_tile_hcu_grouped_gemm_desc> transposed_descs;
    transposed_descs.reserve(group_count);
    for(int i = 0; i < group_count; ++i)
    {
        const auto& arg = descs[i];
        if(arg.a_ptr == nullptr || arg.b_ptr == nullptr || arg.c_ptr == nullptr ||
           arg.k_batch != 1 || arg.num_d_tensors != 0 || arg.M <= 0 ||
           arg.N <= 0 || arg.K <= 0)
        {
            return -3;
        }

        // Row-major C[M,N] = A^T[M,K] * B[K,N] is byte-identical to
        // column-major C^T[N,M] = B^T[N,K] * A[K,M]. Swap only descriptor
        // roles; the caller-owned tensor pointers and output bytes stay put.
        transposed_descs.push_back({arg.b_ptr,
                                    arg.a_ptr,
                                    arg.c_ptr,
                                    arg.k_batch,
                                    arg.N,
                                    arg.M,
                                    arg.K,
                                    arg.stride_B,
                                    arg.stride_A,
                                    arg.stride_C,
                                    0,
                                    nullptr,
                                    nullptr});
    }

    using Config = GemmConfigComputeV3DsreadmStage256W8<ck_tile::bf16_t>;
    return dispatch_grouped_gemm_c_layout<Config, ck_tile::bf16_t, 93607>(
        transposed_descs.data(), group_count, 'C', 'R', workspace, stream);
}
#endif
