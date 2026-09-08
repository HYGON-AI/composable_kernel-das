// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kv_load_mode_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding.hpp"
#include "ck_tile/ops/fmha/block/page_block_navigator.hpp"
// [HCU移植] FMHA 移植中 BWD 与扩展 FWD（appendkv/splitkv 等）头文件尚未在 HCU 上打通，
// 用 CK_TILE_FMHA_ENABLE_BWD_HEADERS / CK_TILE_FMHA_ENABLE_EXTENDED_FWD_HEADERS 做编译门禁，
// 仅在对应 example/实例目标里按需打开（见 01_fmha/CMakeLists.txt）。
#if defined(CK_TILE_FMHA_ENABLE_BWD_HEADERS)
#include "ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp"
#endif
#if defined(CK_TILE_FMHA_ENABLE_EXTENDED_FWD_HEADERS)
#include "ck_tile/ops/fmha/kernel/fmha_fwd_appendkv_kernel.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_appendkv_tile_partitioner.hpp"
#endif
#include "ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp"
#if defined(CK_TILE_FMHA_ENABLE_EXTENDED_FWD_HEADERS)
#include "ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_tile_partitioner.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_tile_partitioner.hpp"
#endif
#include "ck_tile/ops/fmha/kernel/fmha_fwd_tile_partitioner.hpp"
#if defined(CK_TILE_FMHA_ENABLE_BWD_HEADERS)
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_convert_dq.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dot_do_o.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_problem.hpp"
#endif
#if defined(CK_TILE_FMHA_ENABLE_EXTENDED_FWD_HEADERS)
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs_default_policy.hpp"
#endif
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs_default_policy.hpp"
#if defined(CK_TILE_FMHA_ENABLE_EXTENDED_FWD_HEADERS)
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"
#endif
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
