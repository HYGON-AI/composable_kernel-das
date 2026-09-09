// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <ck_tile/core.hpp>
#include "fused_moegemm.hpp"
#include "fused_moegemm_api_traits.hpp"
#include "fused_moegemm_api_internal.hpp"

// clang-format off
// template float fused_moegemm_<
//     fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, float, int, S<32, 256, 32>, S<1, 4, 1>, S<32, 64, 32>, S<32, 64, 256>, S<32, 16, 256>, S<2, 4, 4, 4>,  S<4>, S<8, 8, 8, 4>, 1, 4, 1, 1, 0, 0>
// >(const ck_tile::stream_config& s, fused_moegemm_args a);

// template float fused_moegemm_<
//     fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 0, 1, 0>
// >(const ck_tile::stream_config& s, fused_moegemm_args a);

// template float fused_moegemm_<
//     fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 1, 0, 0>
// >(const ck_tile::stream_config& s, fused_moegemm_args a);

// template float fused_moegemm_<
//     fmoe_<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float, S<32, 512, 128, 128>, S<1, 4, 1>, S<16, 16, 32>, 1, 1, 0>
// >(const ck_tile::stream_config& s, fused_moegemm_args a);

// clang-format on
