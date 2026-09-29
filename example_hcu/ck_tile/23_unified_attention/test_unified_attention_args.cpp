// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "unified_attention_helper.hpp"

int main()
{
    using ck_tile::parse_query_lens_helper;
    if(!parse_query_lens_helper("").empty() ||
       parse_query_lens_helper("17,65") != std::vector<int>({17, 65}) ||
       parse_query_lens_helper(" +17 , 65 ") != std::vector<int>({17, 65}))
        return 1;
    for(const auto* text : {"0", "-1", "17,0", "17,-1", "abc", "17x", "1.5",
                            "2147483648", "99999999999999999999999999999",
                            ",", ",17", "17,", "17,,65", " "})
    {
        try
        {
            (void)parse_query_lens_helper(text, "kv_lens");
            std::cerr << "accepted invalid lengths: " << text << '\n';
            return 1;
        }
        catch(const std::invalid_argument& error)
        {
            if(std::string(error.what()).find("kv_lens") == std::string::npos) return 1;
        }
    }

    // Call the metadata constructor directly, bypassing main's validation.
    auto make_parser = [](const char* kv_lens) {
        ck_tile::ArgParser parser;
        parser.insert("h", "8", "").insert("hkv", "2", "").insert("d", "192", "")
            .insert("block_size", "64", "").insert("mask", "causal", "")
            .insert("sliding_window", "0", "").insert("sinks", "0", "")
            .insert("alibi", "0", "").insert("alibi_sqrt", "0", "")
            .insert("qq_bias", "0", "").insert("qq_bias_dtype", "fp32", "")
            .insert("softcap", "0", "").insert("mm_prefix", "0", "")
            .insert("prefix_begin", "0", "").insert("prefix_end", "-1", "")
            .insert("nq", "64", "").insert("nkv", "128", "")
            .insert("query_lens", "17,65", "").insert("kv_lens", kv_lens, "")
            .insert("block_table_width", "0", "").insert("kv_cache_blocks", "0", "");
        return parser;
    };
    for(const auto* text : {"128", "128,128,128", "128,0", "128,-1", "16,128"})
    {
        try
        {
            (void)ck_tile::UnifiedAttentionProblem::create(make_parser(text));
            std::cerr << "accepted invalid metadata lengths: " << text << '\n';
            return 1;
        }
        catch(const std::invalid_argument&) {}
    }
    const auto defaults = ck_tile::UnifiedAttentionProblem::create(make_parser(""));
    const auto explicit_lens = ck_tile::UnifiedAttentionProblem::create(make_parser("63,129"));
    if(defaults.seqlens != std::vector<int32_t>({128, 128}) ||
       explicit_lens.seqlens != std::vector<int32_t>({63, 129}))
        return 1;
    std::cout << "Unified attention parsing and metadata checks passed\n";
    return 0;
}
