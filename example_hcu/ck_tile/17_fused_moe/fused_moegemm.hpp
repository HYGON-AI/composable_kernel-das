// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fused_moe.hpp"
#include <string>

// this is only a convenient structure for creating an example
// this is not part of the host API
template <typename I, typename W, typename O, typename ST, typename SW, typename SQ, typename KW, typename ZP>
struct FusedMoeGemmTypeConfig;

template <typename ST, typename SW, typename SQ, typename KW, typename ZP>
struct FusedMoeGemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, ST, SW, SQ, KW, ZP>
{
    using ADataType            = ck_tile::bf16_t;
    using GDataType            = ck_tile::bf16_t;
    using DDataType            = ck_tile::bf16_t;
    using AccDataType          = float;
    using ODataType            = ck_tile::bf16_t;
    using AScaleDataType       = ck_tile::remove_cvref_t<ST>;
    using GScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using DScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<SQ>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<KW>;
    using IndexDataType        = ck_tile::index_t;
};

template <typename ST, typename SW, typename SQ, typename KW, typename ZP>
struct FusedMoeGemmTypeConfig<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, ST, SW, SQ, KW, ZP>
{
    using ADataType            = ck_tile::fp16_t;
    using GDataType            = ck_tile::fp16_t;
    using DDataType            = ck_tile::fp16_t;
    using AccDataType          = float;
    using ODataType            = ck_tile::fp16_t;
    using AScaleDataType       = ck_tile::remove_cvref_t<ST>;
    using GScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using DScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<SQ>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<KW>;
    using IndexDataType        = ck_tile::index_t;
};

template <typename ST, typename SW, typename SQ, typename KW, typename ZP>
struct FusedMoeGemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, ck_tile::bf16_t, ST, SW, SQ, KW, ZP>
{
    using ADataType            = ck_tile::int8_t;
    using GDataType            = ck_tile::int8_t;
    using DDataType            = ck_tile::int8_t;
    using AccDataType          = int32_t;
    using ODataType            = ck_tile::bf16_t;
    using AScaleDataType       = ck_tile::remove_cvref_t<ST>;
    using GScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using DScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<SQ>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<KW>;
    using IndexDataType        = ck_tile::index_t;
};

template <typename ST, typename SW, typename SQ, typename KW, typename ZP>
struct FusedMoeGemmTypeConfig<ck_tile::fp16_t, ck_tile::fp16_t, float, ST, SW, SQ, KW, ZP>
{
    using ADataType            = ck_tile::fp16_t;
    using GDataType            = ck_tile::fp16_t;
    using DDataType            = ck_tile::fp16_t;
    using AccDataType          = float;
    using ODataType            = float;
    using AScaleDataType       = ck_tile::remove_cvref_t<ST>;
    using GScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using DScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<SQ>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<KW>;
    using IndexDataType        = ck_tile::index_t;
};

template <typename ST, typename SW, typename SQ, typename KW, typename ZP>
struct FusedMoeGemmTypeConfig<ck_tile::fp16_t, ck_tile::int8_t, float, ST, SW, SQ, KW, ZP>
{
    using ADataType            = ck_tile::fp16_t;
    using GDataType            = ck_tile::int8_t;
    using DDataType            = ck_tile::int8_t;
    using AccDataType          = float;
    using ODataType            = float;
    using AScaleDataType       = ck_tile::remove_cvref_t<ST>;
    using GScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using DScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<SQ>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<KW>;
    using IndexDataType        = ck_tile::index_t;
};

template <typename ST, typename SW, typename SQ, typename KW, typename ZP>
struct FusedMoeGemmTypeConfig<ck_tile::bf16_t, ck_tile::int8_t, float, ST, SW, SQ, KW, ZP>
{
    using ADataType            = ck_tile::bf16_t;
    using GDataType            = ck_tile::int8_t;
    using DDataType            = ck_tile::int8_t;
    using AccDataType          = float;
    using ODataType            = float;
    using AScaleDataType       = ck_tile::remove_cvref_t<ST>;
    using GScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using DScaleDataType       = ck_tile::remove_cvref_t<SW>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<ZP>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<SQ>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<KW>;
    using IndexDataType        = ck_tile::index_t;
};

template <typename ST, typename SQ, typename KW>
struct FusedMoeGemmTypeConfig<ck_tile::fp16_t, ck_tile::int8_t, float, ST, ck_tile::fp16_t, SQ, KW, ck_tile::int8_t>
{
    using ADataType            = ck_tile::fp16_t;
    using GDataType            = ck_tile::int8_t;
    using DDataType            = ck_tile::int8_t;
    using AccDataType          = float;
    using ODataType            = float;
    using AScaleDataType       = ck_tile::remove_cvref_t<ST>;
    using GScaleDataType       = ck_tile::remove_cvref_t<ck_tile::fp16_t>;
    using DScaleDataType       = ck_tile::remove_cvref_t<ck_tile::fp16_t>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<ck_tile::int8_t>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<ck_tile::int8_t>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<SQ>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<KW>;
    using IndexDataType        = ck_tile::index_t;
};

template <typename ST, typename SQ, typename KW>
struct FusedMoeGemmTypeConfig<ck_tile::bf16_t, ck_tile::int8_t, float, ST, ck_tile::bf16_t, SQ, KW, ck_tile::int8_t>
{
    using ADataType            = ck_tile::bf16_t;
    using GDataType            = ck_tile::int8_t;
    using DDataType            = ck_tile::int8_t;
    using AccDataType          = float;
    using ODataType            = float;
    using AScaleDataType       = ck_tile::remove_cvref_t<ST>;
    using GScaleDataType       = ck_tile::remove_cvref_t<ck_tile::bf16_t>;
    using DScaleDataType       = ck_tile::remove_cvref_t<ck_tile::bf16_t>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<ck_tile::int8_t>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<ck_tile::int8_t>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<SQ>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<KW>;
    using IndexDataType        = ck_tile::index_t;
};

// runtime args
struct fused_moegemm_args : public ck_tile::FusedMoeGemmHostArgs
{
};

// This is the public API, will be generated by script
struct fused_moegemm_traits
{
    std::string prec_i;  // input precision
    std::string prec_w;  // weight precision
    std::string prec_o;  // output precision
    std::string prec_st; // token scale data type
    std::string prec_sw; // weight scale data type
    std::string prec_sq; // smooth quant scale
    std::string prec_kw; // topk-weight data type
    std::string prec_zp; // zero points data type
    int block_m;
    int activation;  // 0:gelu, 1:silu
    int gate_only;   // 0:g1u0, 1:g1u1
    int fused_quant; // 0:no-sweep, 1:smooth-dynamic-quant, 2:int8_w8a16, 3:int4_w4a16, 4:use_int8_w8a8_block, 5:use_int4_w4a8_block, 10:dynamic-quant
    int solution_id;
    bool use_wt_shuffle;
};

float fused_moegemm(fused_moegemm_traits, fused_moegemm_args, const ck_tile::stream_config&);

void fused_moe_find_solutions(fused_moegemm_traits t, fused_moegemm_args a, int* sol_data, int* sol_size);
