// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "ck_tile/host.hpp"
#include "jenga_fwd.hpp"
#include <iostream>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "1", "batch size")
        .insert("h", "1", "number of heads")
        .insert("s", "1024", "sequence length (multiple of 64)")
        .insert("d", "128", "head dimension (currently 128)")
        .insert("prec", "bf16", "data type: bf16 or fp16")
        .insert("topk", "8", "minimum visual KV blocks selected per Q block")
        .insert("prob_threshold", "0.3", "cumulative probability threshold")
        .insert("text_start_block", "-1", "first text block; -1 means all visual blocks")
        .insert("text_blocks", "0", "number of text blocks always selected")
        .insert("first_frame_blocks", "0", "first-frame blocks always selected")
        .insert("store_lse", "1", "copy and validate LSE output")
        .insert("kv_stages", "0", "KV stages: 0=auto, 1=off, 2-5=force")
        .insert("warmup", "10", "warmup launches")
        .insert("repeat", "100", "timed launches")
        .insert("v", "1", "run the GPU reference")
        .insert("seed", "1", "deterministic input seed");
    return std::make_tuple(arg_parser.parse(argc, argv), arg_parser);
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return 1;
    const auto precision = arg_parser.get_str("prec");
    if(precision == "bf16")
        return run_jenga_fwd_bf16(arg_parser);
    if(precision == "fp16")
        return run_jenga_fwd_fp16(arg_parser);
    std::cerr << "unsupported precision: " << precision << std::endl;
    return 1;
}
