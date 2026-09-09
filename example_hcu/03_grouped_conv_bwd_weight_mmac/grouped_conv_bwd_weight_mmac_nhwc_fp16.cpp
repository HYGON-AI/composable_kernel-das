// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_mmac.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static inline constexpr ck::index_t NDimSpatial = 2;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

using FP16 = ck::half_t;
using FP32 = float;

using OutDataType = FP16;
using InDataType  = FP16;
using AccDataType = FP32;
using WeiDataType = FP32;

using OutLayout = ck::tensor_layout::convolution::NHWGK;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using InLayout  = ck::tensor_layout::convolution::NHWGC;

using OutElementOp = PassThrough;
using WeiElementOp = PassThrough;
using InElementOp  = PassThrough;

using DeviceGroupedConvInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Mmac<2,
                                                                  OutLayout,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  OutDataType,
                                                                  InDataType,
                                                                  AccDataType,
                                                                  WeiDataType,
                                                                  OutElementOp,
                                                                  InElementOp,
                                                                  WeiElementOp,
                                                                  ConvSpec,
                                                                  256,
                                                                  128,
                                                                  128,
                                                                  4, // K0PerBlock
                                                                  8, // K1
                                                                  16,
                                                                  16,
                                                                  4, // MRepeat
                                                                  4, // NRepeat
                                                                  S<1, 4, 64, 1>,
                                                                  S<0, 3, 1, 2>,
                                                                  S<0, 2, 1, 3>,
                                                                  2, // gemm m
                                                                  2, // src vector on gemm m
                                                                  8, // dst vector on gemm k1
                                                                  true,
                                                                  S<1, 4, 64, 1>,
                                                                  S<0, 3, 1, 2>,
                                                                  S<0, 2, 1, 3>,
                                                                  2, // gemm n
                                                                  2, // src vector on gemm n
                                                                  8, // dst vector on gemm k1
                                                                  true>;

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 2;
    bool time_kernel     = true;
};

#define DefaultConvParams                                                                 \
    ck::utils::conv::ConvParam                                                            \
    {                                                                                     \
        NDimSpatial, 1, 128, 128, 128, {1, 1}, {14, 14}, {1, 1}, {1, 1}, {0, 0}, { 0, 0 } \
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

bool run_conv_bwd_weight(const ExecutionConfig& config,
                         const ck::utils::conv::ConvParam& conv_params,
                         const HostTensorDescriptor& out_g_n_k_wos_desc,
                         const HostTensorDescriptor& in_g_n_c_wis_desc,
                         const HostTensorDescriptor& wei_g_k_c_xs_desc,
                         const OutElementOp& out_element_op,
                         const InElementOp& in_element_op,
                         const WeiElementOp& wei_element_op)
{
    Tensor<OutDataType> out(out_g_n_k_wos_desc);
    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei_host(wei_g_k_c_xs_desc);
    Tensor<WeiDataType> wei_device(wei_g_k_c_xs_desc);

    std::cout << "out: " << out.mDesc << std::endl;
    std::cout << "wei: " << wei_host.mDesc << std::endl;
    std::cout << "in: " << in.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        in.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        in.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_host.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    in_device_buf.ToDevice(in.mData.data());

    // reset input to zero
    wei_device_buf.SetZero();

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> c_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> c_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(out_g_n_k_wos_desc.GetLengths(), a_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), a_g_n_k_wos_strides);
    copy(in_g_n_c_wis_desc.GetLengths(), b_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), b_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), c_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), c_g_k_c_xs_strides);
    copy(conv_params.conv_filter_strides_, conv_filter_strides);
    copy(conv_params.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_params.input_left_pads_, input_left_pads);
    copy(conv_params.input_right_pads_, input_right_pads);

    static_assert(std::is_default_constructible_v<DeviceGroupedConvInstance>);

    // do conv
    auto conv     = DeviceGroupedConvInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(out_device_buf.GetDeviceBuffer(),
                                      in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      a_g_n_k_wos_lengths,
                                      a_g_n_k_wos_strides,
                                      b_g_n_c_wis_lengths,
                                      b_g_n_c_wis_strides,
                                      c_g_k_c_xs_lengths,
                                      c_g_k_c_xs_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      out_element_op,
                                      wei_element_op,
                                      in_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        std::cerr << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;

        return false;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    if(config.do_verification && config.time_kernel)
    {
        // The backward-weight kernel accumulates into the output. Timed runs execute the
        // kernel repeatedly, so regenerate a clean single-run result for verification.
        wei_device_buf.SetZero();
        invoker.Run(argument, StreamConfig{nullptr, false});
    }

    std::size_t flop      = conv_params.GetFlops();
    std::size_t num_btype = conv_params.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(config.do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                           InDataType,
                                                                           WeiDataType,
                                                                           OutDataType,
                                                                           PassThrough,
                                                                           WeiElementOp,
                                                                           OutElementOp>();

        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei_host,
                                                  out,
                                                  conv_params.conv_filter_strides_,
                                                  conv_params.conv_filter_dilations_,
                                                  conv_params.input_left_pads_,
                                                  conv_params.input_right_pads_,
                                                  PassThrough{},
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);

        wei_device_buf.FromDevice(wei_device.mData.data());

        return ck::utils::check_err(
            wei_device.mData, wei_host.mData, "Error: Incorrect results!", 1e-5, 3e-4);
    }

    return true;
}

int run_grouped_conv_bwd_weight_example(int argc, char* argv[])
{
    namespace ctc = ck::tensor_layout::convolution;

    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_params = DefaultConvParams;

    if(!parse_cmd_args(argc, argv, config, conv_params))
    {
        return EXIT_FAILURE;
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(conv_params.num_dim_spatial_ != NDimSpatial)
    {
        std::cerr << "unsupported # of spatials dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    // output image: GNHWK
    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
            conv_params);

    // weight: GKYXC
    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_params);

    // input image: GNHWC
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_params);

    return !run_conv_bwd_weight(config,
                                conv_params,
                                out_g_n_k_wos_desc,
                                in_g_n_c_wis_desc,
                                wei_g_k_c_xs_desc,
                                out_element_op,
                                in_element_op,
                                wei_element_op);
}

int main(int argc, char* argv[]) { return run_grouped_conv_bwd_weight_example(argc, argv); }
