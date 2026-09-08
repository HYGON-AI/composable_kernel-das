// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "reduce.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("time", "1", "time kernel (0=no, 1=yes)")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    const bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = DataType;

    const ck_tile::index_t m = arg_parser.get_int("m");
    const ck_tile::index_t n = arg_parser.get_int("n");
    const int do_validation  = arg_parser.get_int("v");
    const bool time_kernel   = arg_parser.get_bool("time");
    const int warmup         = time_kernel ? arg_parser.get_int("warmup") : 0;
    const int repeat         = time_kernel ? arg_parser.get_int("repeat") : 1;

    using ReduceOp   = ck_tile::ReduceOp::Add;
    using BlockWarps = ck_tile::sequence<4, 1>;
    using BlockTile  = ck_tile::sequence<128, 128>;
    using WarpTile   = ck_tile::sequence<32, 128>;
    using Vector     = ck_tile::sequence<8, 8>;

    if(m <= 0 || n <= 0 || n % Vector::at(ck_tile::number<1>{}) != 0)
    {
        std::cerr << "m and n must be positive, and n must be a multiple of "
                  << Vector::at(ck_tile::number<1>{}) << std::endl;
        return false;
    }

    ck_tile::HostTensor<XDataType> x_host({m, n});
    ck_tile::HostTensor<YDataType> y_host_ref({m});
    ck_tile::HostTensor<YDataType> y_host_dev({m});

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    constexpr ck_tile::index_t kBlockSize  = 256;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    const ck_tile::index_t kGridSize =
        ck_tile::integer_divide_ceil(m, BlockTile::at(ck_tile::number<0>{}));
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape = ck_tile::Reduce2dShape<BlockWarps, BlockTile, WarpTile, Vector>;
    using Problem =
        ck_tile::Reduce2dProblem<XDataType, ComputeDataType, YDataType, Shape, ReduceOp>;

    using Kernel = ck_tile::Reduce<Problem>;

    const float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, time_kernel, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockSize, kBlockPerCu>(Kernel{},
                                                       kGridSize,
                                                       kBlockSize,
                                                       0,
                                                       static_cast<XDataType*>(
                                                           x_buf.GetDeviceBuffer()),
                                                       static_cast<YDataType*>(
                                                           y_buf.GetDeviceBuffer()),
                                                       m,
                                                       n));

    if(time_kernel)
    {
        const std::size_t num_btype =
            sizeof(XDataType) * m * n + sizeof(YDataType) * m;
        const float gb_per_sec = num_btype / 1.E6 / ave_time;
        std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    }
    else
    {
        std::cout << "timing:off" << std::endl;
    }

    bool pass = true;

    if(do_validation)
    {
        ck_tile::reference_reduce<XDataType, ComputeDataType, YDataType>(
            x_host, y_host_ref, ReduceOp{});
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, y_host_ref);

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }

    std::cerr << "unsupported precision: " << data_type << std::endl;
    return -3;
}
