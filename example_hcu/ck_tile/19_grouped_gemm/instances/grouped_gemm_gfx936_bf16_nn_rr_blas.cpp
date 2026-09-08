// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// Isolate the validated BW-family BF16 R,R grad-X specialization from the
// existing R,C forward translation unit. The four-stage circular LDS
// producer/consumer follows the selected custom hipBLASLt schedule while the
// scalar-m0 variant remains the fastest correct CK control.
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
#define CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL
#define CK_TILE_GROUPED_GEMM_TRITON_TILE_M01 2
#define CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C
#define CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
#define CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
#define CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT
#define CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0

#include "grouped_gemm_impl.hpp"

#ifndef CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_ENTRY
#define CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_ENTRY \
    ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_rr_blas_device_args
#endif
#ifndef CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_COMPAT_ENTRY
#define CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_COMPAT_ENTRY \
    ck_tile_hcu_grouped_gemm_gfx936_bf16_nn_rr_blas_device_args
#endif

using Bf16NnRrBlasConfig =
    GemmConfigComputeV3TritonOperand256ExactTraversalTransposeC<
        ck_tile::bf16_t>;

#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_ONLY) && \
    defined(CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_HOST_ENTRY)
int CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_HOST_ENTRY(
    const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count,
    char a_layout,
    char b_layout,
    void* workspace,
    hipStream_t stream)
{
    if(a_layout != 'R' || b_layout != 'R')
    {
        return -4;
    }
    return dispatch_grouped_gemm_c_layout<Bf16NnRrBlasConfig,
                                           ck_tile::bf16_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}
#endif

extern "C" int CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_ENTRY(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    if(device_args == nullptr || group_count <= 0)
    {
        return -1;
    }

    (void)num_cu;
    using Config = Bf16NnRrBlasConfig;
    using Row = ck_tile::tensor_layout::gemm::RowMajor;

    grouped_gemm_tileloop<Config,
                          Row,
                          Row,
                          Row,
                          ck_tile::bf16_t,
                          ck_tile::bf16_t,
                          float,
                          ck_tile::bf16_t,
                          CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG>(
                                           ck_tile::stream_config{stream},
                                           group_count,
                                           device_args,
                                           {},
                                           false);
    return 0;
}

extern "C" int CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_COMPAT_ENTRY(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return CK_TILE_GROUPED_GEMM_BF16_NN_RR_BLAS_ENTRY(
        device_args, group_count, num_cu, stream);
}
