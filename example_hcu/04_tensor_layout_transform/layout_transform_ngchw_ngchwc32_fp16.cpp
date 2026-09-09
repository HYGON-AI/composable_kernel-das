// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_layout_transform_impl.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using FP16 = ck::half_t;
using FP32 = float;

using InDataType  = FP16;
using OutDataType = FP16;

using InLayout  = ck::tensor_layout::convolution::NGCHW;
using OutLayout = ck::tensor_layout::convolution::NGCHWc32;

static constexpr ck::index_t NDimSpatial = 2;

using DeviceLayoutTransformInstance =
    ck::tensor_operation::device::DeviceLayoutTransformImpl<InDataType,
                                                            OutDataType,
                                                            InLayout,
                                                            OutLayout,
                                                            256,
                                                            2,
                                                            32,
                                                            32,
                                                            S<1, 32, 8>, // <NG * C_Tildes, C32, HW>
                                                            S<1, 32, 8>, // <NG * C_Tildes, HW, C32>
                                                            4,
                                                            4,
                                                            true>;

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 2;
    bool time_kernel     = true;
};

#define DefaultConvParams                                                                   \
    ck::utils::conv::ConvParam                                                              \
    {                                                                                       \
        NDimSpatial, 1, 256, 2048, 1024, {1, 1}, {14, 14}, {2, 2}, {1, 1}, {0, 0}, { 0, 0 } \
    }

inline void print_help_msg()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ExecutionConfig& config,
                           ck::utils::conv::ConvParam& conv_params)
{
    constexpr int num_execution_config_args =
        3; // arguments for do_verification, init_method, time_kernel
    constexpr int num_conv_param_leading_args = 5; // arguments for num_dim_spatial_, G_, N_, K_, C_

    constexpr int threshold_to_catch_partial_args = 1 + num_execution_config_args;
    constexpr int threshold_to_catch_all_args =
        threshold_to_catch_partial_args + num_conv_param_leading_args;

    if(argc == 1)
    {
        // use default
        config = ExecutionConfig{};
    }
    // catch only ExecutionConfig arguments
    else if(argc == threshold_to_catch_partial_args)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    // catch both ExecutionConfig & ConvParam arguments
    else if(threshold_to_catch_all_args < argc && ((argc - threshold_to_catch_all_args) % 3 == 0))
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        const ck::index_t num_dim_spatial = std::stoi(argv[4]);
        conv_params                       = ck::utils::conv::parse_conv_param(
            num_dim_spatial, threshold_to_catch_partial_args + 1, argv);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

template <typename DeviceOpInstance>
float run(const ExecutionConfig& config,
          const HostTensorDescriptor& tensor_org_desc,
          const HostTensorDescriptor& tensor_trans_desc)
{
    Tensor<InDataType> tensor_org(tensor_org_desc);
    Tensor<OutDataType> tensor_trans(tensor_trans_desc);

    // for result check
    Tensor<float> tensor_host_golden_fp32(tensor_org_desc);
    Tensor<float> tensor_host_result_fp32(tensor_org_desc);

    std::cout << "org desc: " << tensor_org.mDesc << std::endl;
    std::cout << "trans desc: " << tensor_trans.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1: tensor_org.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5}); break;
    default: tensor_org.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * tensor_org.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * tensor_trans.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(tensor_org.mData.data());

    // reset output to zero
    out_device_buf.SetZero();

    std::array<ck::index_t, NDimSpatial + 3> org_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> trans_g_n_c_wis_lengths{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(tensor_org.GetLengths(), org_g_n_c_wis_lengths);
    copy(tensor_trans.GetLengths(), trans_g_n_c_wis_lengths);

    static_assert(std::is_default_constructible_v<DeviceOpInstance>);

    auto trans    = DeviceOpInstance{};
    auto invoker  = trans.MakeInvokerPointer();
    auto argument = trans.MakeArgumentPointer(org_g_n_c_wis_lengths,
                                              trans_g_n_c_wis_lengths,
                                              in_device_buf.GetDeviceBuffer(),
                                              out_device_buf.GetDeviceBuffer());

    if(!trans.IsSupportedArgument(argument.get()))
    {
        std::cerr << "wrong! device_trans with the specified compilation parameters does "
                     "not support this  problem"
                  << std::endl;

        return -1.0f;
    }

    float ave_time = invoker->Run(argument.get(), StreamConfig{nullptr, config.time_kernel});

    std::size_t num_bytes = tensor_org.GetElementSpaceSizeInBytes();

    float gb_per_sec = num_bytes / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    if(config.do_verification)
    {
        out_device_buf.FromDevice(tensor_trans.mData.data());

        tensor_trans.ForEach([&](auto&, auto idx) mutable {
            // idx order: (n, g, c_tildes, hi, wi, c_vect)
            tensor_host_golden_fp32(idx[0], idx[1], idx[2] * 32 + idx[5], idx[3], idx[4]) =
                static_cast<float>(
                    tensor_org(idx[0], idx[1], idx[2] * 32 + idx[5], idx[3], idx[4]));
            tensor_host_result_fp32(idx[0], idx[1], idx[2] * 32 + idx[5], idx[3], idx[4]) =
                static_cast<float>(tensor_trans(idx));
        });

        const bool pass = ck::utils::check_err(tensor_host_result_fp32.mData,
                                               tensor_host_golden_fp32.mData,
                                               "Error: Incorrect results!",
                                               1e-5,
                                               3e-4);
        if(!pass)
        {
            return -1.0f;
        }
    }

    return ave_time;
}

int run_layout_transform_ngchw_to_ngchwc32(int argc, char* argv[])
{
    namespace ctc = ck::tensor_layout::convolution;

    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_params = DefaultConvParams;

    if(!parse_cmd_args(argc, argv, config, conv_params))
    {
        return EXIT_FAILURE;
    }

    if(conv_params.num_dim_spatial_ != NDimSpatial)
    {
        std::cerr << "unsupported # of spatials dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    float total_time = 0.0f;
    // input trans
    const auto in_g_n_c_wis_org_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_params);
    const auto in_g_n_c_wis_trans_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<OutLayout>(conv_params);
    const float input_transform_time =
        run<DeviceLayoutTransformInstance>(config, in_g_n_c_wis_org_desc, in_g_n_c_wis_trans_desc);
    if(input_transform_time < 0)
    {
        return EXIT_FAILURE;
    }
    total_time += input_transform_time;

#if 0
    // filter trans
    if(conv_params.filter_spatial_lengths_[0] * conv_params.filter_spatial_lengths_[1] != 1)
    {
        const auto wei_g_k_c_xs_org_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<InLayout>(
                conv_params);
        const auto wei_g_k_c_xs_trans_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<OutLayout>(
                conv_params);
        total_time += run<DeviceLayoutTransformInstance>(
            config, wei_g_k_c_xs_org_desc, wei_g_k_c_xs_trans_desc);
    }

    // output trans
    const auto out_g_n_k_wos_org_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<InLayout>(conv_params);
    const auto out_g_n_k_wos_trans_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
            conv_params);
    total_time += run<DeviceLayoutTransformInstance>(
        config, out_g_n_k_wos_org_desc, out_g_n_k_wos_trans_desc);
#endif

    std::cout << "total trans time: " << total_time << " ms " << std::endl;

    return 0;
}

int main(int argc, char* argv[]) { return run_layout_transform_ngchw_to_ngchwc32(argc, argv); }
