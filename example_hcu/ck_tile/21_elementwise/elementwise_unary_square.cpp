// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "32", "row count")
        .insert("n", "64", "column count")
        .insert("stride", "-1", "row stride; -1 means n")
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
    const ck_tile::index_t m = args.get_int("m");
    const ck_tile::index_t n = args.get_int("n");
    ck_tile::index_t stride  = args.get_int("stride");
    if(stride < 0)
    {
        stride = n;
    }
    if(m <= 0 || n <= 0 || stride != n)
    {
        throw std::runtime_error("m/n must be positive and this merged kernel requires stride == n");
    }

    const bool validate    = args.get_bool("v");
    const bool time_kernel = args.get_bool("time");
    const int warmup       = time_kernel ? args.get_int("warmup") : 0;
    const int repeat       = time_kernel ? args.get_int("repeat") : 1;

    using XDataType            = DataType;
    using ComputeDataType      = DataType;
    using YDataType            = DataType;
    using ElementWiseOperation = ck_tile::element_wise::UnarySquare;
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

    ck_tile::HostTensor<XDataType> x_host({m, n}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host({m, n}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_ref({m, n}, {stride, 1});
    ck_tile::FillUniformDistribution<XDataType>{-5.0f, 5.0f, 11939}(x_host);

    ck_tile::DeviceMem x_device(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_device(y_host.get_element_space_size_in_bytes());
    x_device.ToDevice(x_host.data());

    const auto lengths = ck_tile::make_tuple(m, n);
    const auto strides = ck_tile::make_tuple(stride, 1);
    if(!Kernel::IsSupportedArgument(lengths))
    {
        throw std::runtime_error("unsupported element count for vector width");
    }

    constexpr ck_tile::index_t block_size         = Shape::kBlockSize;
    constexpr ck_tile::index_t elements_per_block = Shape::kBlockM;
    constexpr ck_tile::index_t blocks_per_cu       = 1;
    const ck_tile::index_t total_elements          = m * n;
    const ck_tile::index_t grid_size =
        (total_elements + elements_per_block - 1) / elements_per_block;
    const auto input_tensors =
        ck_tile::make_tuple(static_cast<XDataType*>(x_device.GetDeviceBuffer()));

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
        const auto square = [](const auto& x) { return x * x; };
        ck_tile::reference_unary_elementwise<XDataType, YDataType, ComputeDataType>(
            x_host, y_ref, square);
        valid = ck_tile::check_err(y_host, y_ref, "Elementwise Square Error", 1e-3, 1e-3);
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
