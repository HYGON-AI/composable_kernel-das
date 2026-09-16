// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gdn/block/gdn_ck_pre_process_block_gemm.hpp"
#include "ck_tile/ops/gdn/block/gdn_ck_kkt_solve_block_gemm.hpp"
#include "ck_tile/ops/gdn/block/gdn_ck_recompute_w_u_block_gemm.hpp"
#include "ck_tile/ops/gdn/block/chunk_delta_h_k_block_gemm.hpp"
#include "ck_tile/ops/gdn/block/chunk_delta_h_projection_block_gemm.hpp"

#include "ck_tile/ops/gdn/pipeline/gdn_ck_fused_recurrent_problem.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_fused_recurrent_policy.hpp"

#include "ck_tile/ops/gdn/pipeline/gdn_ck_cumsum_problem.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_cumsum_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_cumsum_pipeline.hpp"

#include "ck_tile/ops/gdn/pipeline/gdn_ck_pre_process_problem.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_pre_process_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_pre_process_pipeline.hpp"

#include "ck_tile/ops/gdn/pipeline/gdn_ck_kkt_solve_problem.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_kkt_solve_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_kkt_solve_pipeline.hpp"

#include "ck_tile/ops/gdn/pipeline/gdn_ck_recompute_w_u_problem.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_recompute_w_u_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_recompute_w_u_pipeline.hpp"

#include "ck_tile/ops/gdn/pipeline/gdn_ck_l2norm_problem.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_l2norm_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_ck_l2norm_pipeline.hpp"

#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_scan_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/cp/chunk_delta_h_grouped_scan.hpp"
#include "ck_tile/ops/gdn/pipeline/cp/chunk_delta_h_scan_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/cp/chunk_delta_h_wave_reg_pipeline.hpp"

#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_config.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_problem.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_pipeline.hpp"

#include "ck_tile/ops/gdn/kernel/gdn_ck_fused_recurrent_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_cumsum_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_pre_process_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_kkt_solve_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_recompute_w_u_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_l2norm_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/chunk_delta_h_scan_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_fwd_output_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_beta_sigmoid_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_cp_context_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_prepare_indices_kernel.hpp"
