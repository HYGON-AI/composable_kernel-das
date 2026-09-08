// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// Isolate the BW-family large aligned RC specialization so the normal FP16
// translation unit keeps every existing V4, padded, Split-K and multi-D path.
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_SPLIT_GLOBAL
#define CK_TILE_GROUPED_GEMM_TRITON_TILE_M01 8
#define CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE
#define CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC
#define CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE
#define CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D

#include "grouped_gemm_impl.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_bw_family_selected.hpp"

#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_ONLY)
int grouped_gemm_c_run_gfx936_fp16(
    const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count,
    char a_layout,
    char b_layout,
    void* workspace,
    hipStream_t stream)
{
    return dispatch_grouped_gemm_c_layout<
        GemmConfigComputeV3TritonOperand256ExactTraversal<ck_tile::half_t>,
        ck_tile::half_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_example_run_gfx936_fp16(std::string a_layout,
                                          std::string b_layout,
                                          int argc,
                                          char* argv[])
{
    return run_gemm_example_prec_type<
        GemmConfigComputeV3TritonOperand256ExactTraversal<ck_tile::half_t>,
        ck_tile::half_t>(a_layout, b_layout, argc, argv);
}
#endif

template <typename ALayout, typename BLayout>
static int run_bw_family_fp16_device_args(
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
    using Config = GemmConfigComputeV3TritonOperand256ExactTraversal<ck_tile::half_t>;
    using Row    = ck_tile::tensor_layout::gemm::RowMajor;

    grouped_gemm_tileloop<Config,
                          ALayout,
                          BLayout,
                          Row,
                          ck_tile::half_t,
                          ck_tile::half_t,
                          float,
                          ck_tile::half_t>(ck_tile::stream_config{stream},
                                           group_count,
                                           device_args,
                                           {},
                                           false);
    return 0;
}

extern "C" int ck_tile_hcu_grouped_gemm_bw_family_fp16_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
    return run_bw_family_fp16_device_args<Row, Col>(
        device_args, group_count, num_cu, stream);
}

extern "C" int ck_tile_hcu_grouped_gemm_bw_family_fp16_nn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    return run_bw_family_fp16_device_args<Row, Row>(
        device_args, group_count, num_cu, stream);
}

extern "C" int ck_tile_hcu_grouped_gemm_bw_family_fp16_tn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
    return run_bw_family_fp16_device_args<Col, Row>(
        device_args, group_count, num_cu, stream);
}

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_fp16_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return ck_tile_hcu_grouped_gemm_bw_family_fp16_device_args(
        device_args, group_count, num_cu, stream);
}

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_fp16_nn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return ck_tile_hcu_grouped_gemm_bw_family_fp16_nn_device_args(
        device_args, group_count, num_cu, stream);
}

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_fp16_tn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return ck_tile_hcu_grouped_gemm_bw_family_fp16_tn_device_args(
        device_args, group_count, num_cu, stream);
}
