// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("d0", "2", "dimension 0")
        .insert("d1", "4", "dimension 1")
        .insert("d2", "8", "dimension 2")
        .insert("d3", "32", "dimension 3")
        .insert("v", "1", "perform CPU validation")
        .insert("prec", "fp16", "input/output precision")
        .insert("time", "1", "time kernel execution")
        .insert("warmup", "10", "warmup iterations")
        .insert("repeat", "50", "timed iterations");

    const bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& args)
{
    const ck_tile::index_t d0 = args.get_int("d0");
    const ck_tile::index_t d1 = args.get_int("d1");
    const ck_tile::index_t d2 = args.get_int("d2");
    const ck_tile::index_t d3 = args.get_int("d3");
    if(d0 <= 0 || d1 <= 0 || d2 <= 0 || d3 <= 0)
    {
        throw std::runtime_error("all dimensions must be positive");
    }

    const bool validate    = args.get_bool("v");
    const bool time_kernel = args.get_bool("time");
    const int warmup       = time_kernel ? args.get_int("warmup") : 0;
    const int repeat       = time_kernel ? args.get_int("repeat") : 1;

    using XDataType            = DataType;
    using ComputeDataType      = DataType;
    using YDataType            = DataType;
    using ElementWiseOperation = ck_tile::element_wise::Add;
    using BlockTile            = ck_tile::sequence<2048>;
    using BlockWarps           = ck_tile::sequence<8>;
    using WarpTile             = ck_tile::sequence<64>;
    using Shape = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, ComputeDataType>;
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        ElementWiseOperation>;
    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    const std::vector<ck_tile::index_t> lengths_vector{d0, d1, d2, d3};
    const std::vector<ck_tile::index_t> strides_vector{d1 * d2 * d3, d2 * d3, d3, 1};
    ck_tile::HostTensor<XDataType> x0_host(lengths_vector, strides_vector);
    ck_tile::HostTensor<XDataType> x1_host(lengths_vector, strides_vector);
    ck_tile::HostTensor<YDataType> y_host(lengths_vector, strides_vector);
    ck_tile::HostTensor<YDataType> y_ref(lengths_vector, strides_vector);
    ck_tile::FillUniformDistribution<XDataType>{-5.0f, 5.0f, 11939}(x0_host);
    ck_tile::FillUniformDistribution<XDataType>{-5.0f, 5.0f, 11940}(x1_host);

    ck_tile::DeviceMem x0_device(x0_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x1_device(x1_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_device(y_host.get_element_space_size_in_bytes());
    x0_device.ToDevice(x0_host.data());
    x1_device.ToDevice(x1_host.data());

    const auto lengths = ck_tile::make_tuple(d0, d1, d2, d3);
    const auto strides = ck_tile::make_tuple(
        strides_vector[0], strides_vector[1], strides_vector[2], strides_vector[3]);
    if(!Kernel::IsSupportedArgument(lengths))
    {
        throw std::runtime_error("unsupported element count for vector width");
    }

    constexpr ck_tile::index_t block_size         = Shape::kBlockSize;
    constexpr ck_tile::index_t elements_per_block = Shape::kBlockM;
    constexpr ck_tile::index_t blocks_per_cu       = 1;
    const ck_tile::index_t total_elements          = d0 * d1 * d2 * d3;
    const ck_tile::index_t grid_size =
        (total_elements + elements_per_block - 1) / elements_per_block;
    const auto input_tensors =
        ck_tile::make_tuple(static_cast<XDataType*>(x0_device.GetDeviceBuffer()),
                            static_cast<XDataType*>(x1_device.GetDeviceBuffer()));

    const float ms = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, time_kernel, 0, warmup, repeat},
        ck_tile::make_kernel<block_size, blocks_per_cu>(
            Kernel{},
            grid_size,
            block_size,
            0,
            lengths,
            strides,
            strides,
            input_tensors,
            static_cast<YDataType*>(y_device.GetDeviceBuffer())));

    std::cout << "grid:" << grid_size << ", block:" << block_size << ", ";
    if(time_kernel)
    {
        std::cout << "ms:" << ms << ", ";
    }
    else
    {
        std::cout << "timing:off, ";
    }

    bool valid = ms >= 0;
    if(validate && valid)
    {
        y_device.FromDevice(y_host.data());
        const auto add = [](const auto& x0, const auto& x1) { return x0 + x1; };
        ck_tile::reference_binary_elementwise<XDataType,
                                              XDataType,
                                              YDataType,
                                              ComputeDataType>(x0_host, x1_host, y_ref, add);
        valid = ck_tile::check_err(y_host, y_ref, "4D Elementwise Add Error", 1e-3, 1e-3);
    }

    std::cout << "valid:" << (valid ? "y" : "n") << std::endl;
    return valid;
}

int main(int argc, char* argv[])
{
    auto [parsed, args] = create_args(argc, argv);
    if(!parsed)
    {
        return -1;
    }

    try
    {
        if(args.get_str("prec") == "fp16")
        {
            return run<ck_tile::fp16_t>(args) ? 0 : -2;
        }
        std::cerr << "only fp16 is supported by this acceptance slice" << std::endl;
        return -3;
    }
    catch(const std::exception& error)
    {
        std::cerr << "error: " << error.what() << std::endl;
        return -4;
    }
}
