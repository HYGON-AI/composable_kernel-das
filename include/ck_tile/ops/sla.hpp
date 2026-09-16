// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/sla/block/sla_attn_fwd_block_gemm.hpp"
#include "ck_tile/ops/sla/block/sla_attn_bwd_dq_block_gemm.hpp"
#include "ck_tile/ops/sla/block/sla_attn_bwd_dq_asmem_block_gemm.hpp"
#include "ck_tile/ops/sla/block/sla_attn_bwd_dkdv_block_gemm.hpp"

#include "ck_tile/ops/sla/pipeline/sla_attn_fwd_problem.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_fwd_policy.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_fwd_pipeline.hpp"
#include "ck_tile/ops/sla/pipeline/sla_sparse_map_problem.hpp"
#include "ck_tile/ops/sla/pipeline/sla_sparse_map_policy.hpp"
#include "ck_tile/ops/sla/pipeline/sla_sparse_map_pipeline.hpp"
#include "ck_tile/ops/sla/pipeline/sla_fused_linear_attn_problem.hpp"
#include "ck_tile/ops/sla/pipeline/sla_fused_linear_policy.hpp"
#include "ck_tile/ops/sla/pipeline/sla_fused_linear_pipeline.hpp"
#include "ck_tile/ops/sla/pipeline/sla_sparse_lut_pipeline.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_preprocess_pipeline.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dq_tile_shape.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dq_problem.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dq_policy.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dq_pipeline.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dkdv_problem.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dkdv_policy.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dkdv_pipeline.hpp"

#include "ck_tile/ops/sla/kernel/sla_attn_fwd_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_attn_fwd_reduce_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_sparse_lut_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_sparse_map_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_fused_linear_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_attn_bwd_preprocess_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_attn_bwd_dq_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_attn_bwd_dq_reduce_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_attn_bwd_dkdv_kernel.hpp"
