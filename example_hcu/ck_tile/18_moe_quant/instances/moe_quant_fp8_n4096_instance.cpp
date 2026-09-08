// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "moe_quant_instance_common.hpp"

//                                               rm rn tm tn   vm vn pad
template float moe_quant_<
    trait_<ck_tile::fp16_t, ck_tile::fp8_t, 1, 4, 1, 1024, 1, 1, true>>(const S&, A);
template float moe_quant_<
    trait_<ck_tile::bf16_t, ck_tile::fp8_t, 1, 4, 1, 1024, 1, 1, true>>(const S&, A);
