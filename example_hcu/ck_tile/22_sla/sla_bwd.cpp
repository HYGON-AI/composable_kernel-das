// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "ck_tile/host.hpp"
#include "sla_bwd.hpp"
#include <iostream>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "1", "batch size")
        .insert("h", "1", "number of heads")
        .insert("s", "1024", "sequence length (multiple of 64)")
        .insert("d", "128", "head dimension (currently 128)")
        .insert("prec", "bf16", "data type: bf16 or fp16")
        .insert("topk_ratio", "0.1", "topk ratio for sparse_map (0.0 - 1.0)")
        .insert("kv_stages", "0", "dQ KV stages: 0=auto, otherwise 1-16")
        .insert("warmup", "10", "warmup launches")
        .insert("repeat", "100", "timed launches")
        .insert("v", "1", "run verification")
        .insert("seed", "1", "random seed");
    return std::make_tuple(arg_parser.parse(argc, argv), arg_parser);
}

int main(int argc, char* argv[])
{
    auto [parsed, arg_parser] = create_args(argc, argv);
    if(!parsed)
        return 1;
    const auto precision = arg_parser.get_str("prec");
    if(precision == "bf16")
        return run_sla_bwd_bf16(arg_parser);
    if(precision == "fp16")
        return run_sla_bwd_fp16(arg_parser);
    std::cerr << "unsupported precision: " << precision << std::endl;
    return 1;
}
