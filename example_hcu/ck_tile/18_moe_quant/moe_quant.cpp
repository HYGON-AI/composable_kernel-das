// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "ck_tile/host.hpp"
#include "moe_quant.hpp"
#include <cstring>
#include <set>

// different threshold for different dtype
template <typename DataType>
auto get_elimit()
{
    double rtol = 1e-5;
    double atol = 1e-5;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1e-5;
    double atol = 1e-5;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::int8_t>()
{
    // due to rounding, int8 quantization might have 1 abs error
    double rtol = 1;
    double atol = 1;
    return ck_tile::make_tuple(rtol, atol);
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("t", "128", "tokens dimension")
        .insert("h", "8192", "hidden_size dimension")
        //.insert("g", "4096", "group quant size")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec_i", "fp16", "input precision, fp16/bf16")
        .insert("prec_o", "int8", "output precision, int8 or fp8 (hidden_size<=8192)")
        .insert("time", "1", "time kernel (0=no, 1=yes)")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InputType, typename OutputType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t tokens      = arg_parser.get_int("t");
    ck_tile::index_t hidden_size = arg_parser.get_int("h");
    //ck_tile::index_t group_size = arg_parser.get_int("g");
    std::string prec_i       = arg_parser.get_str("prec_i");
    std::string prec_o       = arg_parser.get_str("prec_o");
    int kname                = arg_parser.get_int("kname");
    int do_validation        = arg_parser.get_int("v");
    bool time_kernel         = arg_parser.get_bool("time");
    int warmup               = arg_parser.get_int("warmup");
    int repeat               = arg_parser.get_int("repeat");

    ck_tile::index_t x_stride = 1;
    ck_tile::index_t y_stride = hidden_size;

    using TypeConfig = MoequantTypeConfig<InputType, OutputType>;

    using InputDataType       = typename TypeConfig::InputDataType;
    using ScaleDataType       = typename TypeConfig::ScaleDataType;
    using OutDataType         = typename TypeConfig::OutDataType;
    using ComputeDataType     = typename TypeConfig::ComputeDataType;

    // host verify
    ck_tile::HostTensor<InputDataType> input_host({tokens, hidden_size}, {hidden_size, 1});

    ck_tile::HostTensor<ScaleDataType> scale_host_ref({tokens}, {1});
    ck_tile::HostTensor<ScaleDataType> scale_host_dev({tokens}, {1});

    ck_tile::HostTensor<OutDataType> q_host_ref({tokens, hidden_size}, {hidden_size, 1});
    ck_tile::HostTensor<OutDataType> q_host_dev({tokens, hidden_size}, {hidden_size, 1});

    ck_tile::FillUniformDistribution<InputDataType>{-.5f, .5f}(input_host);

    ck_tile::DeviceMem input_buf(input_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_buf(scale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem q_buf(q_host_dev.get_element_space_size_in_bytes());

    input_buf.ToDevice(input_host.data());

    std::cout << "[" << prec_i << "-" << prec_o << "]"
              << " tokens:" << tokens << ", hidden_size:" << hidden_size<< std::flush;

    moe_quant_traits traits{prec_i, prec_o};
    moe_quant_args args{input_buf.GetDeviceBuffer(),
                        scale_buf.GetDeviceBuffer(),
                        q_buf.GetDeviceBuffer(),
                        tokens,
                        hidden_size,
                        x_stride,
                        y_stride};

    float ave_time = moe_quant(traits,
                               args,
                               ck_tile::stream_config{nullptr,
                                                      time_kernel,
                                                      kname ? 1 : 0,
                                                      time_kernel ? warmup : 0,
                                                      time_kernel ? repeat : 1});

    if(time_kernel)
    {
        std::size_t num_byte = sizeof(InputDataType) * tokens * hidden_size +
                               sizeof(ScaleDataType) * tokens +
                               sizeof(OutDataType) * tokens * hidden_size;

        float gb_per_sec = num_byte / 1.E6 / ave_time;
        std::cout << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s"
                  << std::flush;
    }
    else
    {
        std::cout << ", timing:off" << std::flush;
    }

    bool pass = true;

    if(do_validation)
    {
        using YDataType = ComputeDataType;
        ck_tile::HostTensor<ComputeDataType> y_host({tokens, hidden_size}, {hidden_size, 1});
        // smooth outlier
        {
            auto f = [&](auto i_token) {
                    for(int i_h = 0; i_h < hidden_size; ++i_h)
                    {
                        y_host(i_token, i_h) = ck_tile::type_convert<ComputeDataType>(input_host(i_token, i_h));
                    }
            };

            ck_tile::make_ParallelTensorFunctor(f, tokens)(std::thread::hardware_concurrency());
        }

        // yscale
        {
            ck_tile::HostTensor<YDataType> rowwise_amax_host({tokens});

            using ReduceAmax = ck_tile::ReduceOp::AbsMax;
            ck_tile::reference_reduce<ComputeDataType, ComputeDataType, YDataType>(
                y_host, rowwise_amax_host, ReduceAmax{});

            auto op = [](const auto& v0) {
                return v0 /
                       ck_tile::type_convert<ComputeDataType>(ck_tile::numeric<OutDataType>::max());
            };
            ck_tile::reference_unary_elementwise<YDataType, ScaleDataType, ComputeDataType>(
                rowwise_amax_host, scale_host_ref, op);

            scale_buf.FromDevice(scale_host_dev.mData.data());

            auto [rtol, atol] = get_elimit<ScaleDataType>();
            pass &= ck_tile::check_err(scale_host_dev,
                                       scale_host_ref,
                                       std::string("yscale Error: Incorrect results!"),
                                       rtol,
                                       atol);
        }

        // rowwise quantization
        {
            ck_tile::reference_rowwise_quantization2d<YDataType, ScaleDataType, OutDataType>(
                y_host, scale_host_ref, q_host_ref);

            q_buf.FromDevice(q_host_dev.data());
            auto [rtol, atol] = get_elimit<OutDataType>();

            {
                pass = ck_tile::check_err(q_host_dev,
                                          q_host_ref,
                                          std::string("qy Error: Incorrect results!"),
                                          rtol,
                                          atol);
            }
        }

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    if(arg_parser.get_int("h") > 8192)
    {
        std::cerr << "[ERROR] hidden_size exceeds 8192" << std::endl;
        return -4;
    }

    const std::string prec_i = arg_parser.get_str("prec_i");
    const std::string prec_o = arg_parser.get_str("prec_o");
    if(prec_i == "fp16" && prec_o == "int8")
    {
        return run<ck_tile::half_t, ck_tile::int8_t>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "fp16" && prec_o == "fp8")
    {
        return run<ck_tile::half_t, ck_tile::fp8_t>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "int8")
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "fp8")
    {
        return run<ck_tile::bf16_t, ck_tile::fp8_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
