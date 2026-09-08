// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance_hcu/gpu/grouped_convolution_backward_weight.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"

namespace ck {
namespace profiler {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
bool profile_grouped_conv_bwd_weight_impl(int do_verification,
                                          int init_method,
                                          bool do_log,
                                          bool time_kernel,
                                          const ck::utils::conv::ConvParam& conv_param,
                                          bool packed_channel = false,
                                          int iter            = 10,
                                          int splitk          = 1)
{
    std::ignore = splitk;

    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> c_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> c_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    if constexpr(is_same_v<InLayout, ck::tensor_layout::convolution::NGCHWc32> &&
                 is_same_v<WeiLayout, ck::tensor_layout::convolution::GKCYXc32> &&
                 is_same_v<OutLayout, ck::tensor_layout::convolution::NGKHWk32>)
    {
        {
            auto copy = [](const auto& x, auto& y) {
                std::copy(x.begin(), x.end() - 1, y.begin());
            };
            copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
            copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
            copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
            copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
            copy(out_g_n_k_wos_desc.GetLengths(), c_g_n_k_wos_lengths);
            copy(out_g_n_k_wos_desc.GetStrides(), c_g_n_k_wos_strides);
        }

        {
            auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };
            copy(conv_param.conv_filter_strides_, conv_filter_strides);
            copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
            copy(conv_param.input_left_pads_, input_left_pads);
            copy(conv_param.input_right_pads_, input_right_pads);
        }
    }
    else if constexpr(std::is_base_of_v<ck::tensor_layout::convolution::NGCHWcBase, InLayout> &&
                      std::is_base_of_v<ck::tensor_layout::convolution::GKCYXcBase, WeiLayout> &&
                      std::is_base_of_v<ck::tensor_layout::convolution::NGCHWcBase, OutLayout>)
    {
        {
            auto copy = [](const auto& x, auto& y) {
                std::copy(x.begin(), x.end() - 1, y.begin());
            };
            copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
            copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
            copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
            copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
            copy(out_g_n_k_wos_desc.GetLengths(), c_g_n_k_wos_lengths);
            copy(out_g_n_k_wos_desc.GetStrides(), c_g_n_k_wos_strides);
        }

        {
            auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };
            copy(conv_param.conv_filter_strides_, conv_filter_strides);
            copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
            copy(conv_param.input_left_pads_, input_left_pads);
            copy(conv_param.input_right_pads_, input_right_pads);
        }
    }
    else
    {
        auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
        copy(out_g_n_k_wos_desc.GetLengths(), c_g_n_k_wos_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), c_g_n_k_wos_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);
    }

    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight_host_result(wei_g_k_c_xs_desc);
    Tensor<WeiDataType> weight_device_result(wei_g_k_c_xs_desc);
    Tensor<OutDataType> output(out_g_n_k_wos_desc);

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weight: " << weight_host_result.mDesc << std::endl;
    std::cout << "output: " << output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        output.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        output.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.05, 0.05});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) *
                             weight_device_result.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * output.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(input.mData.data());
    out_device_buf.ToDevice(output.mData.data());

    if(do_verification)
    {
        auto ref_conv     = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                               InDataType,
                                                                               WeiDataType,
                                                                               OutDataType,
                                                                               InElementOp,
                                                                               WeiElementOp,
                                                                               OutElementOp>{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weight_host_result,
                                                  output,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op,
                                                  packed_channel);

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device Conv instances
    bool pass = true;

    auto run_impl = [&](auto& op_ptr, auto& argument_ptr, bool profiling = false) {
        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string op_name = op_ptr->GetTypeString();

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            float avg_time = 0.0f;

            if(profiling)
            {
                avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});
            }
            else
            {
                for(int i = 0; i < iter; ++i)
                {
                    // using atomic add, so need to reset input
                    wei_device_buf.SetZero();

                    avg_time +=
                        invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});
                }
                avg_time /= iter;
            }

            std::size_t flop      = conv_param.GetFlops();
            std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                wei_device_buf.FromDevice(weight_device_result.mData.data());

                pass = pass & ck::utils::check_err(weight_device_result,
                                                   weight_host_result,
                                                   "Error: Incorrect results!",
                                                   1e-5,
                                                   2e-3);

                if(!pass)
                {
                    std::cout << "Fail info: " << op_ptr->GetTypeString() << std::endl;
                }

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "output : ", output.mData, ",") << std::endl;
                    ;
                    LogRangeAsType<float>(
                        std::cout << "weight (device): ", weight_device_result.mData, ",")
                        << std::endl;
                    ;
                    LogRangeAsType<float>(
                        std::cout << "weight (host): ", weight_host_result.mData, ",")
                        << std::endl;
                    ;
                    LogRangeAsType<float>(std::cout << "input: ", input.mData, ",") << std::endl;
                    ;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    };

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeightV2<NDimSpatial,
                                                                                OutLayout,
                                                                                InLayout,
                                                                                WeiLayout,
                                                                                OutDataType,
                                                                                InDataType,
                                                                                WeiDataType,
                                                                                OutElementOp,
                                                                                InElementOp,
                                                                                WeiElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    bool profiling = false;

    // only run once for profiling
    profiling = conv_param.GetInstanceName().empty() ? false : true;

    if(profiling)
    {
        auto it = std::find_if(op_ptrs.begin(), op_ptrs.end(), [&](auto& op_ptr) {
            if(op_ptr->GetTypeString() == conv_param.GetInstanceName())
                return true;
            return false;
        });

        if(it == op_ptrs.end())
        {
            std::cout << "instance not found: " << conv_param.GetInstanceName() << std::endl;
            return 1;
        }

        auto argument_ptr =
            (*it)->MakeArgumentPointer(static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                       static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                       static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                       c_g_n_k_wos_lengths,
                                       c_g_n_k_wos_strides,
                                       a_g_n_c_wis_lengths,
                                       a_g_n_c_wis_strides,
                                       b_g_k_c_xs_lengths,
                                       b_g_k_c_xs_strides,
                                       conv_filter_strides,
                                       conv_filter_dilations,
                                       input_left_pads,
                                       input_right_pads,
                                       out_element_op,
                                       in_element_op,
                                       wei_element_op);

        run_impl((*it), argument_ptr, true);
    }
    else
    {
        for(auto& op_ptr : op_ptrs)
        {
            auto argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                c_g_n_k_wos_lengths,
                c_g_n_k_wos_strides,
                a_g_n_c_wis_lengths,
                a_g_n_c_wis_strides,
                b_g_k_c_xs_lengths,
                b_g_k_c_xs_strides,
                conv_filter_strides,
                conv_filter_dilations,
                input_left_pads,
                input_right_pads,
                out_element_op,
                in_element_op,
                wei_element_op);

            run_impl(op_ptr, argument_ptr);
        }
    }

    std::cout << "Best configuration parameters:"
              << "\nname: " << best_op_name << "\navg_time: " << best_avg_time
              << "\ntflops: " << best_tflops << "\nGB/s: " << best_gb_per_sec << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
