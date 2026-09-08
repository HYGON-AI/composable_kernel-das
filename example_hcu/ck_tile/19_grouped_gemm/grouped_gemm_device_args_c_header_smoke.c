// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"

int main(void)
{
    ck_tile_hcu_grouped_gemm_problem_v1 problem = {0};
    ck_tile_hcu_grouped_gemm_selection_v1 selection = {0};
    ck_tile_hcu_grouped_gemm_launch_v1 launch = {0};
    ck_tile_hcu_grouped_gemm_candidate_info_v1 candidate = {0};
    problem.struct_size = (uint32_t)sizeof(problem);
    selection.struct_size = (uint32_t)sizeof(selection);
    launch.struct_size = (uint32_t)sizeof(launch);
    candidate.struct_size = (uint32_t)sizeof(candidate);
    candidate.instance_id =
        CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V3_128_M4_FULL_PADDING_V1;
    candidate.family = CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1;
    candidate.architecture_mask = CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1;
    candidate.data_type_mask = CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_FP16_V1;
    candidate.layout_mask = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1;
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION == 1u &&
                   candidate.instance_id == 4009u
               ? 0
               : 1;
}
