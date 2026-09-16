// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/jenga/kernel/jenga_bwd_dq_kernel.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dq_config.hpp"
#include "ck_tile/ops/jenga/kernel/jenga_bwd_dkdv_kernel.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dkdv_config.hpp"

struct JengaBwdDkdvTypeConfig
{
    using QDataType     = ck_tile::bf16_t;
    using KDataType     = ck_tile::bf16_t;
    using VDataType     = ck_tile::bf16_t;
    using ODataType     = ck_tile::bf16_t;
    using OGradDataType = ck_tile::bf16_t;
    using KGradDataType = ck_tile::bf16_t;
    using VGradDataType = ck_tile::bf16_t;
    using AccDataType   = float;
    using LSEDataType   = float;
};

using QDataType     = JengaBwdDkdvTypeConfig::QDataType;
using KDataType     = JengaBwdDkdvTypeConfig::KDataType;
using VDataType     = JengaBwdDkdvTypeConfig::VDataType;
using ODataType     = JengaBwdDkdvTypeConfig::ODataType;
using OGradDataType = JengaBwdDkdvTypeConfig::OGradDataType;
using KGradDataType = JengaBwdDkdvTypeConfig::KGradDataType;
using VGradDataType = JengaBwdDkdvTypeConfig::VGradDataType;
using AccDataType   = JengaBwdDkdvTypeConfig::AccDataType;
using LSEDataType   = JengaBwdDkdvTypeConfig::LSEDataType;

float jenga_bwd_dq(ck_tile::jenga_dq::jenga_bwd_dq_traits t,
                   ck_tile::jenga_dq::jenga_bwd_dq_args a,
                   ck_tile::stream_config s,
                   float* reduce_ms = nullptr);

float jenga_bwd_dkdv_pipeline_calc(
    const ck_tile::example::jenga::jenga_bwd_dkdv_traits& traits,
    const ck_tile::example::jenga::jenga_bwd_dkdv_args& args,
    const ck_tile::stream_config& s);
