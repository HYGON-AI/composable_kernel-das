// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include <iostream>
#include <ck_tile/core.hpp>
#include "moe_quant.hpp"

template <typename InType,
          typename OutType,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_M_,
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_>
using trait_ = moe_quant_traits_<InType,
                                       OutType,
                                       Repeat_M_,
                                       Repeat_N_,
                                       ThreadPerBlock_M_,
                                       ThreadPerBlock_N_,
                                       Vector_M_,
                                       Vector_N_,
                                       kPadN_>;

template <typename in_type, typename out_type>
float moe_quant_dispatch(moe_quant_traits /*t*/,
                               moe_quant_args a,
                               const ck_tile::stream_config& s)
{
    float r = -1;
    // clang-format off
    //                                                    rm  rn  tm  tn  vn   pd    2p
    if(a.hidden_size <= 64) {
            r = moe_quant_<trait_<in_type, out_type, 1,  1,  4,  64, 1, 1,  true>>(s, a);
            //r = moe_quant_<trait_<in_type, out_type, 1,  1,  8,  32, 2,  true>>(s, a);
    }
    else if(a.hidden_size <= 128) {
        if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type, 1,  1,  4,  64, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type, 1,  2,  4,  64, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 256) {
        if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 1,  4,  64, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 2,  4,  64, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1, 4,  4,  64, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 512) {
        if (a.hidden_size % 8 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 1,  4,  64, 1, 8,  true>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 2,  4,  64, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 4,  4,  64, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1, 8,  4,  64, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 768) {
        if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 3,  4,  64, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 6,  4,  64, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1,12,  4,  64, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 1024) {
        if (a.hidden_size % 8 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 1, 2,  128, 1, 8,  true>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 2, 2,  128, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 4, 2,  128, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1, 4, 1,  256, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 1536) {
        if (a.hidden_size % 8 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 3, 4,   64, 1, 8,  true>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 3, 2,  128, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 3, 1,  256, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1, 6, 1,  256, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 2048) {
        if (a.hidden_size % 8 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 1, 1,  256, 1, 8,  true>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 2, 1,  256, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 4, 1,  256, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1, 8, 1,  256, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 3072) {
        if (a.hidden_size % 8 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 3, 1,  128, 1, 8,  true>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 3, 1,  256, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 6, 1,  256, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1, 3, 1, 1024, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 4096) {
        if (a.hidden_size % 8 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 2, 1,  256, 1, 8,  true>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 4, 1,  256, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 2, 1, 1024, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1, 4, 1, 1024, 1, 1,  true>>(s, a);
    }
    else if(a.hidden_size <= 8192) {
        if (a.hidden_size % 8 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 4, 1,  256, 1, 8,  true>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 8, 1,  256, 1, 4,  true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_quant_<trait_<in_type, out_type,  1, 4, 1, 1024, 1, 2,  true>>(s, a);
        else
            r = moe_quant_<trait_<in_type, out_type,  1, 8, 1, 1024, 1, 1,  true>>(s, a);
    }
    else {
	    throw std::runtime_error("[ERROR] the hidden_size of input tensor exceeds the 8192,this is not supported");
    }
    return r;
    // clang-format on
}

template <typename in_type>
float moe_quant_fp8_dispatch(moe_quant_args a, const ck_tile::stream_config& s)
{
    if(a.hidden_size <= 64)
    {
        return moe_quant_<
            trait_<in_type, ck_tile::fp8_t, 1, 1, 4, 64, 1, 1, true>>(s, a);
    }
    else if(a.hidden_size <= 128)
    {
        return moe_quant_<
            trait_<in_type, ck_tile::fp8_t, 1, 2, 4, 64, 1, 1, true>>(s, a);
    }
    else if(a.hidden_size <= 256)
    {
        return moe_quant_<
            trait_<in_type, ck_tile::fp8_t, 1, 4, 4, 64, 1, 1, true>>(s, a);
    }
    else if(a.hidden_size <= 512)
    {
        return moe_quant_<
            trait_<in_type, ck_tile::fp8_t, 1, 8, 4, 64, 1, 1, true>>(s, a);
    }
    else if(a.hidden_size <= 1024)
    {
        return moe_quant_<
            trait_<in_type, ck_tile::fp8_t, 1, 4, 1, 256, 1, 1, true>>(s, a);
    }
    else if(a.hidden_size <= 2048)
    {
        return moe_quant_<
            trait_<in_type, ck_tile::fp8_t, 1, 8, 1, 256, 1, 1, true>>(s, a);
    }
    else if(a.hidden_size <= 4096)
    {
        return moe_quant_<
            trait_<in_type, ck_tile::fp8_t, 1, 4, 1, 1024, 1, 1, true>>(s, a);
    }
    else if(a.hidden_size <= 8192)
    {
        return moe_quant_<
            trait_<in_type, ck_tile::fp8_t, 1, 8, 1, 1024, 1, 1, true>>(s, a);
    }
    throw std::runtime_error("FP8 output hidden_size exceeds 8192");
}

float moe_quant(moe_quant_traits t,
                      moe_quant_args a,
                      const ck_tile::stream_config& s)
{
    if(t.in_type.compare("fp16") == 0 && t.out_type == "int8")
    {
        return moe_quant_dispatch<ck_tile::fp16_t, ck_tile::int8_t>(t, a, s);
    }
    else if(t.in_type.compare("fp16") == 0 && t.out_type == "fp8")
    {
        return moe_quant_fp8_dispatch<ck_tile::fp16_t>(a, s);
    }
    else if(t.in_type.compare("bf16") == 0 && t.out_type == "int8")
    {
        return moe_quant_dispatch<ck_tile::bf16_t, ck_tile::int8_t>(t, a, s);
    }
    else if(t.in_type.compare("bf16") == 0 && t.out_type == "fp8")
    {
        return moe_quant_fp8_dispatch<ck_tile::bf16_t>(a, s);
    }
    else
        throw std::runtime_error("Without supported instances!");
}
