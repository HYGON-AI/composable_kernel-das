// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/batched_transpose.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("batch", "2", "batch count")
        .insert("height", "32", "matrix height")
        .insert("width", "32", "matrix width")
        .insert("v", "1", "perform CPU validation")
        .insert("prec", "fp16", "input/output precision")
        .insert("time", "1", "time kernel execution")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "timed iterations");

    const bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& args)
{
    const ck_tile::index_t batch  = args.get_int("batch");
    const ck_tile::index_t height = args.get_int("height");
    const ck_tile::index_t width  = args.get_int("width");
    if(batch <= 0 || height <= 0 || width <= 0 || height % 32 != 0 || width % 32 != 0)
    {
        throw std::runtime_error("basic slice requires positive batch and 32-aligned matrices");
    }

    const bool validate    = args.get_bool("v");
    const bool time_kernel = args.get_bool("time");
    const int warmup       = time_kernel ? args.get_int("warmup") : 0;
    const int repeat       = time_kernel ? args.get_int("repeat") : 1;

    ck_tile::HostTensor<DataType> input({batch, height, width});
    ck_tile::HostTensor<DataType> output({batch, width, height});
    ck_tile::HostTensor<DataType> reference({batch, width, height});
    ck_tile::FillUniformDistribution<DataType>{-5.0f, 5.0f, 11939}(input);

    ck_tile::DeviceMem input_device(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_device(output.get_element_space_size_in_bytes());
    input_device.ToDevice(input.data());

    using Problem = ck_tile::BatchedTransposeProblem<DataType,
                                                      ck_tile::sequence<32, 32>,
                                                      ck_tile::sequence<1, 1>,
                                                      false,
                                                      false>;
    using Pipeline = ck_tile::BatchedTransposePipeline<Problem>;
    using Kernel   = ck_tile::BatchedTransposeKernel<Pipeline>;

    ck_tile::BatchedTransposeHostArgs host_args{input_device.GetDeviceBuffer(),
                                                output_device.GetDeviceBuffer(),
                                                batch,
                                                height,
                                                width,
                                                height * width,
                                                32,
                                                32};
    const auto kernel_args = Kernel::MakeKargs(host_args);
    const dim3 grid        = Kernel::GridSize(host_args);
    constexpr dim3 block   = Kernel::BlockSize();
    const float ms         = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, time_kernel, 0, warmup, repeat},
        ck_tile::make_kernel<block.x, 1>(Kernel{}, grid, block, 0, kernel_args));

    std::cout << "grid:(" << grid.x << "," << grid.y << "," << grid.z << "), block:"
              << block.x << ", ";
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
        output_device.FromDevice(output.data());
        for(ck_tile::index_t b = 0; b < batch; ++b)
        {
            for(ck_tile::index_t m = 0; m < height; ++m)
            {
                for(ck_tile::index_t n = 0; n < width; ++n)
                {
                    reference(b, n, m) = input(b, m, n);
                }
            }
        }
        valid = ck_tile::check_err(output, reference, "Batched Transpose Error", 0, 0);
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
