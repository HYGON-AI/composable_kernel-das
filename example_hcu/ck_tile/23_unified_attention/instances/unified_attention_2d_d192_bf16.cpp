// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#include "unified_attention_2d_d192_impl.hpp"

template int run_unified_attention_2d_d192<ck_tile::bf16_t>(
    const ck_tile::ArgParser&, const std::vector<int>&);
