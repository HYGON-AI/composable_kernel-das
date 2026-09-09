// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "permute.hpp"
#include "ck_tile/host.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace detail {
template <int bytes>
struct to_integer_type;

template <>
struct to_integer_type<4>
{
    using type = int32_t;
};

template <>
struct to_integer_type<2>
{
    using type = int16_t;
};

template <>
struct to_integer_type<1>
{
    using type = int8_t;
};
} // namespace detail

template <int bytes>
using to_integer_type = typename detail::to_integer_type<bytes>::type;

float permute(permute_traits traits,
              permute_args args,
              const ck_tile::stream_config& stream_config)
{
    auto launch = [&](auto data_type) {
        using DataType        = decltype(data_type);
        using PipelineProblem = ck_tile::GenericPermuteProblem<DataType>;
        using Kernel          = ck_tile::GenericPermute<PipelineProblem>;

        auto kargs           = Kernel::MakeKargs(args);
        const dim3 grids      = Kernel::GridSize(args);
        constexpr dim3 blocks = Kernel::BlockSize();

        return ck_tile::launch_kernel(
            stream_config,
            ck_tile::make_kernel<blocks.x, 1>(Kernel{}, grids, blocks, 0, kargs));
    };

    if(traits.data_type == "fp8")
        return launch(ck_tile::fp8_t{});
    if(traits.data_type == "fp16")
        return launch(ck_tile::half_t{});
    if(traits.data_type == "fp32")
        return launch(float{});

    return 0;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& values)
{
    os << "[";
    for(std::size_t i = 0; i < values.size(); ++i)
    {
        if(i != 0)
            os << ", ";
        os << values[i];
    }
    return os << "]";
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "do CPU validation")
        .insert("prec", "fp16", "data type: fp8/fp16/fp32")
        .insert("shape", "2,3,4", "input tensor shape")
        .insert("perm", "2,1,0", "permutation")
        .insert("kname", "0", "print kernel name")
        .insert("seed", "11939", "random seed")
        .insert("time", "1", "time kernel (0=no, 1=yes)")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "timed iterations");

    const bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

std::vector<ck_tile::index_t> decode_vec(const std::string& text)
{
    std::string::size_type pos = 0;
    std::vector<ck_tile::index_t> values;

    while(true)
    {
        const auto found = text.find(',', pos);
        values.push_back(static_cast<ck_tile::index_t>(
            std::atoi(text.substr(pos, found == std::string::npos ? found : found - pos).c_str())));
        if(found == std::string::npos)
            break;
        pos = found + 1;
    }

    return values;
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    const std::string data_type = arg_parser.get_str("prec");
    const int do_validation     = arg_parser.get_int("v");
    const auto shape            = decode_vec(arg_parser.get_str("shape"));
    const auto permutation      = decode_vec(arg_parser.get_str("perm"));
    const bool time_kernel      = arg_parser.get_bool("time");
    const int stream_warmup     = time_kernel ? arg_parser.get_int("warmup") : 0;
    const int stream_repeat     = time_kernel ? arg_parser.get_int("repeat") : 1;
    const bool kname            = arg_parser.get_bool("kname");
    const int seed              = arg_parser.get_int("seed");

    if(shape.size() != permutation.size() ||
       permutation.size() > ck_tile::GenericPermuteHostArgs::kMaxRanks)
    {
        std::cerr << "unsupported shape/permutation rank" << std::endl;
        return false;
    }

    const ck_tile::index_t rank = permutation.size();

    ck_tile::HostTensor<DataType> x(shape);
    ck_tile::FillUniformDistributionIntegerValue<DataType>{-15, 15, seed}(x);

    std::vector<ck_tile::index_t> y_shape(rank, 0);
    for(ck_tile::index_t i = 0; i < rank; ++i)
    {
        if(permutation[i] < 0 || permutation[i] >= rank)
        {
            std::cerr << "invalid permutation" << std::endl;
            return false;
        }
        y_shape[i] = shape[permutation[i]];
    }

    ck_tile::HostTensor<DataType> y(y_shape);
    ck_tile::DeviceMem x_buf(x.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y.get_element_space_size_in_bytes());
    x_buf.ToDevice(x.data());

    std::cout << "[" << data_type << "] shape:" << shape << "->" << y_shape
              << ", permute:" << permutation << std::flush;

    const ck_tile::stream_config stream_config{nullptr,
                                                time_kernel,
                                                kname ? 1 : 0,
                                                stream_warmup,
                                                stream_repeat};

    permute_traits traits{data_type};
    permute_args args{};
    args.p_src = x_buf.GetDeviceBuffer();
    args.p_dst = y_buf.GetDeviceBuffer();
    args.rank  = rank;
    std::copy(shape.begin(), shape.end(), args.shape);
    std::copy(permutation.begin(), permutation.end(), args.perm);

    const float ave_time = permute(traits, args, stream_config);
    if(time_kernel)
        std::cout << ", time:" << ave_time << "ms" << std::flush;
    else
        std::cout << ", timing:off" << std::flush;

    bool pass = true;
    if(do_validation)
    {
        reference_permute(x, y, permutation);
        ck_tile::HostTensor<DataType> y_device(y.get_lengths());
        y_buf.FromDevice(y_device.data());

        pass = std::equal(y_device.begin(),
                          y_device.end(),
                          y.begin(),
                          [&](const DataType& device, const DataType& host) {
                              using Integer = to_integer_type<sizeof(DataType)>;
                              return ck_tile::bit_cast<Integer>(device) ==
                                     ck_tile::bit_cast<Integer>(host);
                          });
        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush;
    }

    std::cout << std::endl;
    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp8")
        return run<ck_tile::fp8_t>(arg_parser) ? 0 : -2;
    if(data_type == "fp16")
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    if(data_type == "fp32")
        return run<float>(arg_parser) ? 0 : -2;

    std::cerr << "unsupported precision: " << data_type << std::endl;
    return -3;
}
