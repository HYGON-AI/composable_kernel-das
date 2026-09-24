// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// Single public entry point. D192, D256 and 3D currently contain translation-
// unit-local helper names inherited from their extension implementations, so a
// translation unit selects one implementation family before including this
// header.
#if defined(CK_TILE_UNIFIED_ATTENTION_D256)
#include "ck_tile/ops/unified_attention/kernel/unified_attention_d256_mmac_kernel.hpp"
#include "ck_tile/ops/unified_attention/kernel/unified_attention_d256_prepare_kernel.hpp"
#include "ck_tile/ops/unified_attention/kernel/unified_attention_d256_tiny_kernel.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_d256_mmac_default_policy.hpp"
#elif defined(CK_TILE_UNIFIED_ATTENTION_3D)
#include "ck_tile/ops/unified_attention/kernel/unified_attention_3d_kernel.hpp"
#else
#include "ck_tile/ops/unified_attention/kernel/unified_attention_mmac_kernel.hpp"
#include "ck_tile/ops/unified_attention/kernel/unified_attention_pack_kernel.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_mmac_default_policy.hpp"
#endif
