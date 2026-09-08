// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm_quant/grouped_gemm_quant_device_args.h"

int ck_tile_hcu_quant_grouped_gemm_device_args_c_header_smoke(void)
{
    ck_tile_hcu_quant_grouped_gemm_problem_v1 problem = {0};
    problem.struct_size = (uint32_t)sizeof(problem);
    problem.abi_version = CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    return problem.struct_size == 0u;
}
