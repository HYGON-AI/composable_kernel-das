// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "topk_softmax_api.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string>
#include <tuple>
#include <vector>

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
void reference_topk_softmax(const ck_tile::HostTensor<InputType>& x,
                            ck_tile::HostTensor<WeightType>& y_values,
                            ck_tile::HostTensor<IndexType>& y_indices,
                            ck_tile::index_t k)
{
    auto y = ck_tile::reference_softmax<InputType, WeightType, WeightType>(x, -1);
    ck_tile::reference_topk(y, y_values, y_indices, k, -1, true, true);
}

template <typename DataType>
auto get_elimit()
{
    return ck_tile::make_tuple(1e-3, 1e-3);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    return ck_tile::make_tuple(1e-2, 1e-2);
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "whether to perform CPU validation")
        .insert("pr_i", "fp16", "input type: fp16 or bf16")
        .insert("pr_w", "fp32", "output weight type; only fp32 is supported")
        .insert("t", "32", "number of input tokens")
        .insert("e", "8", "number of experts")
        .insert("k", "2", "top-k")
        .insert("st_i", "-1", "input row stride; -1 means experts")
        .insert("st_o", "-1", "output/index row stride; -1 means top-k")
        .insert("seed", "-1", "random seed; -1 uses current time")
        .insert("kname", "0", "print kernel name")
        .insert("time", "1", "time kernel execution")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "timed iterations");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
bool test_topk_softmax(ck_tile::ArgParser args)
{
    const bool validate           = args.get_bool("v");
    const std::string input_prec  = args.get_str("pr_i");
    const std::string weight_prec = args.get_str("pr_w");
    const int tokens              = args.get_int("t");
    const int experts             = args.get_int("e");
    const int topk                = args.get_int("k");
    int seed                      = args.get_int("seed");
    int stride_input              = args.get_int("st_i");
    int stride_output             = args.get_int("st_o");
    const bool kname              = args.get_bool("kname");
    const bool time_kernel        = args.get_bool("time");
    const int warmup              = time_kernel ? args.get_int("warmup") : 0;
    const int repeat              = time_kernel ? args.get_int("repeat") : 1;

    if(stride_input < 0)
    {
        stride_input = experts;
    }
    if(stride_output < 0)
    {
        stride_output = topk;
    }
    if(tokens <= 0 || experts <= 0 || topk <= 0 || topk > experts || stride_input < experts ||
       stride_output < topk)
    {
        std::printf("invalid arguments\n");
        return false;
    }
    if(seed < 0)
    {
        seed = static_cast<int>(std::time(nullptr));
    }

    ck_tile::HostTensor<InputType> x_host({tokens, experts}, {stride_input, 1});
    ck_tile::HostTensor<WeightType> value_host({tokens, topk}, {stride_output, 1});
    ck_tile::HostTensor<IndexType> index_host({tokens, topk}, {stride_output, 1});

    auto rand_gen = ck_tile::FillUniformDistribution_Unique<InputType>{
        -5.0f, 5.0f, static_cast<std::uint32_t>(seed)};
    for(int token = 0; token < tokens; ++token)
    {
        ck_tile::HostTensor<InputType> x_row({experts});
        rand_gen(x_row);
        std::copy(x_row.begin(), x_row.end(), x_host.begin() + token * stride_input);
        rand_gen.clear();
    }

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem value_dev(value_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem index_dev(index_host.get_element_space_size_in_bytes());
    x_dev.ToDevice(x_host.data());

    topk_softmax_trait trait{input_prec, weight_prec, experts};
    topk_softmax_kargs karg{x_dev.GetDeviceBuffer(),
                            value_dev.GetDeviceBuffer(),
                            index_dev.GetDeviceBuffer(),
                            tokens,
                            experts,
                            topk,
                            stride_input,
                            stride_output};

    ck_tile::stream_config stream_config{nullptr, time_kernel, kname ? 1 : 0, warmup, repeat};
    const float ms = topk_softmax(trait, karg, stream_config);

    std::printf("[%s|%s]tokens:%d, experts:%d, topk:%d, st_i:%d, st_o:%d, ",
                input_prec.c_str(),
                weight_prec.c_str(),
                tokens,
                experts,
                topk,
                stride_input,
                stride_output);
    if(time_kernel)
    {
        std::printf("ms:%f, ", ms);
    }
    else
    {
        std::printf("timing:off, ");
    }
    if(ms < 0)
    {
        std::printf("not supported\n");
        return false;
    }

    value_dev.FromDevice(value_host.data());
    index_dev.FromDevice(index_host.data());

    bool valid = true;
    if(validate)
    {
        ck_tile::HostTensor<WeightType> value_ref({tokens, topk}, {stride_output, 1});
        ck_tile::HostTensor<IndexType> index_ref({tokens, topk}, {stride_output, 1});
        reference_topk_softmax<InputType, WeightType, IndexType>(
            x_host, value_ref, index_ref, topk);

        const auto [rtol, atol] = get_elimit<InputType>();
        for(int token = 0; token < tokens; ++token)
        {
            const auto begin =
                std::vector<size_t>{static_cast<size_t>(token), static_cast<size_t>(0)};
            const auto end = std::vector<size_t>{static_cast<size_t>(token + 1),
                                                 static_cast<size_t>(topk)};
            valid &= ck_tile::check_err(value_host.slice(begin, end),
                                        value_ref.slice(begin, end),
                                        "[" + std::to_string(token) + "] Value Error:",
                                        rtol,
                                        atol);
            valid &= ck_tile::check_err(index_host.slice(begin, end),
                                        index_ref.slice(begin, end),
                                        "[" + std::to_string(token) + "] Index Error:",
                                        rtol,
                                        atol);
        }
    }

    std::printf("valid:%s\n", valid ? "y" : "n");
    return valid;
}

int main(int argc, char** argv)
{
    auto [parsed, args] = create_args(argc, argv);
    if(!parsed)
    {
        return -1;
    }

    const std::string input_prec  = args.get_str("pr_i");
    const std::string weight_prec = args.get_str("pr_w");
    if(input_prec == "fp16" && weight_prec == "fp32")
    {
        return test_topk_softmax<ck_tile::fp16_t, float>(args) ? 0 : -1;
    }
    if(input_prec == "bf16" && weight_prec == "fp32")
    {
        return test_topk_softmax<ck_tile::bf16_t, float>(args) ? 0 : -1;
    }

    std::printf("unsupported type combination: %s/%s\n",
                input_prec.c_str(),
                weight_prec.c_str());
    return -1;
}
