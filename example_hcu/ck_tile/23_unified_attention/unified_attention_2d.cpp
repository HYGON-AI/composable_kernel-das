// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#include "ck_tile/host.hpp"
#include "instances/unified_attention_2d_instances.hpp"
#include "unified_attention_helper.hpp"

#include <iostream>
#include <string>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser parser;
    parser.insert("nq", "64", "number of query tokens")
        .insert("nkv", "64", "number of KV tokens")
        .insert("h", "16", "number of query heads")
        .insert("hkv", "2", "number of KV heads")
        .insert("d", "192", "head dimension: 192 or 256")
        .insert("prec", "fp16", "data type: fp16 or bf16")
        .insert("query_lens", "", "comma-separated query lengths; enables multibatch")
        .insert("kv_lens", "", "comma-separated KV lengths; empty uses nkv for every sequence")
        .insert("block_size", "64", "number of tokens in each physical KV-cache page")
        .insert("kv_cache_blocks", "0", "physical KV-cache pages; 0 uses the required page count")
        .insert("block_table_width", "0", "logical pages per sequence; 0 uses ceil(max(kv_lens)/block_size)")
        .insert("causal", "1", "causal attention; Triton-compatible path requires 1")
        .insert("mask", "causal", "mask mode: causal, sliding, prefix, sliding_prefix")
        .insert("sliding_window", "0", "sliding-window length; 0 disables it")
        .insert("sinks", "0", "enable per-head attention sinks")
        .insert("sink_value", "0", "value used to initialize every sink")
        .insert("alibi", "0", "enable per-head ALiBi slopes")
        .insert("alibi_sqrt", "0", "use sqrt ALiBi distance")
        .insert("qq_bias", "0", "enable float32 query-query bias")
        .insert("qq_bias_value", "0.01", "constant query-query bias value")
        .insert("qq_bias_dtype", "fp32", "query-query bias data type: fp32 or same as prec")
        .insert("softcap", "0", "positive logits softcap; 0 disables it")
        .insert("mm_prefix", "0", "enable PrefixLM ranges")
        .insert("prefix_begin", "0", "PrefixLM inclusive range begin")
        .insert("prefix_end", "-1", "PrefixLM inclusive range end; -1 uses nq-1")
        .insert("warmup", "10", "warmup launches")
        .insert("repeat", "100", "timed launches")
        .insert("v", "1", "run full independent GPU reference validation");
    return std::make_tuple(parser.parse(argc, argv), parser);
}

int main(int argc, char* argv[])
{
    const auto [ok, parser] = create_args(argc, argv);
    if(!ok) return 0;
    const int d = parser.get_int("d");
    const std::string prec = parser.get_str("prec");
    const auto query_lens =
        ck_tile::parse_query_lens_helper(parser.get_str("query_lens"));
    if(d != 192 && d != 256)
    {
        std::cerr << "d must be 192 or 256\n";
        return 2;
    }
    if(!parser.get_str("query_lens").empty() && query_lens.size() < 2)
    {
        std::cerr << "query_lens must contain at least two positive lengths\n";
        return 2;
    }
    // Validate lengths and page size before D192 builds metadata: its helper
    // indexes kv_lens and divides by block_size while constructing the problem.
    const auto kv_lens = ck_tile::parse_query_lens_helper(parser.get_str("kv_lens"));
    const auto lengths = query_lens.empty() ? std::vector<int>{parser.get_int("nq")}
                                           : query_lens;
    if(parser.get_int("block_size") <= 0 ||
       (!kv_lens.empty() && kv_lens.size() != lengths.size()))
    {
        std::cerr << "block_size must be positive and kv_lens must match query_lens\n";
        return 2;
    }
    for(std::size_t i = 0; i < lengths.size(); ++i)
    {
        const int nkv = kv_lens.empty() ? parser.get_int("nkv") : kv_lens[i];
        if(lengths[i] <= 0 || nkv < lengths[i])
        {
            std::cerr << "require 0 < query length <= KV length for each sequence\n";
            return 2;
        }
    }
    if(prec == "fp16")
        return d == 192
                   ? run_unified_attention_2d_d192<ck_tile::fp16_t>(parser, query_lens)
                   : run_unified_attention_2d_d256<ck_tile::fp16_t>(parser);
    if(prec == "bf16")
        return d == 192
                   ? run_unified_attention_2d_d192<ck_tile::bf16_t>(parser, query_lens)
                   : run_unified_attention_2d_d256<ck_tile::bf16_t>(parser);
    std::cerr << "prec must be fp16 or bf16\n";
    return 2;
}
