// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core/config.hpp"

#if CK_TILE_ARCH_SUPPORTS_FP4
#include "ck_tile/core/numeric/pk_fp4.hpp"
#endif
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"
#include "ck_tile/ops/common/load_and_convert_tile.hpp"
#include "ck_tile/ops/common/streamk_common.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/common/utils.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp"
#include "ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_bquant_cr.hpp"
#if !CK_TILE_HCU_QUANT_GEMM_TARGET
#include "ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp"
#include "ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp"
#endif
#include "ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp"
#include "ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp"
#if !CK_TILE_HCU_QUANT_GEMM_TARGET
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp"
#endif
#include "ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"
