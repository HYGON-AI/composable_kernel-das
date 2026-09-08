// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// Compile only the C,C operand lane used by the zero-copy transposed grad-X
// experiment. Keeping this in an independent translation unit makes each
// LDS/wave/epilogue ablation cheap and prevents unrelated layout
// instantiations from obscuring its resource report.
#include "grouped_gemm_impl.hpp"

namespace {

using Config = GemmConfigComputeV3DsreadmStage256W8<ck_tile::bf16_t>;
using Col    = ck_tile::tensor_layout::gemm::ColumnMajor;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT)
using Out = ck_tile::tensor_layout::gemm::ColumnMajor;
#else
using Out = ck_tile::tensor_layout::gemm::RowMajor;
#endif

} // namespace

int grouped_gemm_c_run_bf16(const ck_tile_hcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream)
{
    if(a_layout != 'C' || b_layout != 'C')
    {
        return -4;
    }
    return run_grouped_gemm_c_impl<Config,
                                   Col,
                                   Col,
                                   Out,
                                   ck_tile::bf16_t,
                                   ck_tile::bf16_t,
                                   float,
                                   ck_tile::bf16_t>(
        descs, group_count, workspace, stream);
}

int grouped_gemm_c_run_bf16_mpad(const ck_tile_hcu_grouped_gemm_desc* descs,
                                 int group_count,
                                 char a_layout,
                                 char b_layout,
                                 void* workspace,
                                 hipStream_t stream)
{
    return grouped_gemm_c_run_bf16(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_example_run_bf16(std::string a_layout,
                                  std::string b_layout,
                                  int argc,
                                  char* argv[])
{
    if(a_layout != "C" || b_layout != "C")
    {
        throw std::runtime_error(
            "gfx936 BF16 C,C control accepts only column-major operands.");
    }
    return run_grouped_gemm_example_with_layouts<Config,
                                                 ck_tile::bf16_t,
                                                 ck_tile::bf16_t,
                                                 ck_tile::bf16_t,
                                                 float>(
        argc, argv, Col{}, Col{}, Out{});
}
