// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs_default_policy.hpp"

namespace ck_tile {

// Paged prefill uses the same HCU QK/PV MMAC layouts as split-KV attention.
struct BlockFmhaFwdPagedKVPipelineQRKSVSDefaultPolicy
    : BlockFmhaFwdSplitKVPipelineQRKSVSDefaultPolicy
{
};

} // namespace ck_tile
