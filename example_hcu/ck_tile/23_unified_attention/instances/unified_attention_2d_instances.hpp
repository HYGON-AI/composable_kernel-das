// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"

#include <vector>

template <typename DataType>
int run_unified_attention_2d_d192(
    const ck_tile::ArgParser& parser,
    const std::vector<int>& query_lens = {});

template <typename DataType>
int run_unified_attention_2d_d256(const ck_tile::ArgParser& parser);

extern template int run_unified_attention_2d_d192<ck_tile::fp16_t>(
    const ck_tile::ArgParser&, const std::vector<int>&);
extern template int run_unified_attention_2d_d192<ck_tile::bf16_t>(
    const ck_tile::ArgParser&, const std::vector<int>&);
extern template int run_unified_attention_2d_d256<ck_tile::fp16_t>(
    const ck_tile::ArgParser&);
extern template int run_unified_attention_2d_d256<ck_tile::bf16_t>(
    const ck_tile::ArgParser&);
