
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "moe_quant_instance_common.hpp"

// clang-format off
//                                                  rm rn  tm  tn  vn   pd    2p
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  1,  8, 32, 1, 2,  true >>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  1,  4, 64, 1, 1,  true >>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  1,  4, 64, 1, 2,  true >>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  2,  4, 64, 1, 1,  true >>(const S&, A);

template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  1,  4, 64, 1, 4,  true >>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  2,  4, 64, 1, 2,  true >>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t, 1,  4,  4, 64, 1, 1,  true >>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 1,  4,  64, 1, 8,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 2,  4,  64, 1, 4,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 4,  4,  64, 1, 2,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 8,  4,  64, 1, 1,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 3,  4,  64, 1, 4,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 6,  4,  64, 1, 2,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1,12,  4,  64, 1, 1,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 1, 2,  128, 1, 8,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 2, 2,  128, 1, 4,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 4, 2,  128, 1, 2,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 4, 1,  256, 1, 1,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 3, 4,   64, 1, 8,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 3, 2,  128, 1, 4,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 3, 1,  256, 1, 2,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 6, 1,  256, 1, 1,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 1, 1,  256, 1, 8,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 2, 1,  256, 1, 4,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 4, 1,  256, 1, 2,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 8, 1,  256, 1, 1,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 3, 1,  128, 1, 8,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 3, 1,  256, 1, 4,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 6, 1,  256, 1, 2,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 3, 1, 1024, 1, 1,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 2, 1,  256, 1, 8,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 4, 1,  256, 1, 4,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 2, 1, 1024, 1, 2,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 4, 1, 1024, 1, 1,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 4, 1,  256, 1, 8,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 8, 1,  256, 1, 4,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 4, 1, 1024, 1, 2,  true>>(const S&, A);
template float moe_quant_<trait_<ck_tile::bf16_t, ck_tile::int8_t,  1, 8, 1, 1024, 1, 1,  true>>(const S&, A);
// clang-format on
