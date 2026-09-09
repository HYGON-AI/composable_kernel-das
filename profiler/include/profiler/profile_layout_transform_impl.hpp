// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <iomanip>
#include <iostream>
#include <numeric>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance_hcu/gpu/layout_transform.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

#include "ck/utility/type.hpp"

template <typename Layout>
struct LayoutTraits;

template <>
struct LayoutTraits<ck::tensor_layout::convolution::NGKHW>
{
    using type = ck::tensor_layout::convolution::NGCHW;
};

template <>
struct LayoutTraits<ck::tensor_layout::convolution::GKCYX>
{
    using type = ck::tensor_layout::convolution::NGCHW;
};

template <>
struct LayoutTraits<ck::tensor_layout::convolution::NGKHWk32>
{
    using type = ck::tensor_layout::convolution::NGCHWc32;
};

template <>
struct LayoutTraits<ck::tensor_layout::convolution::GKCYXc32>
{
    using type = ck::tensor_layout::convolution::NGCHWc32;
};

template <>
struct LayoutTraits<ck::tensor_layout::convolution::NGKHWk<16>>
{
    using type = ck::tensor_layout::convolution::NGCHWc<16>;
};

template <>
struct LayoutTraits<ck::tensor_layout::convolution::GKCYXc<16>>
{
    using type = ck::tensor_layout::convolution::NGCHWc<16>;
};

namespace ck {
namespace profiler {

template <ck::index_t NumDim,
          typename InDataType,
          typename OutDataType,
          typename InLayout,
          typename OutLayout>
float run(int do_verification,
          int init_method,
          bool time_kernel,
          const HostTensorDescriptor& tensor_org_desc,
          const HostTensorDescriptor& tensor_trans_desc,
          std::string& best_op_name,
          float& best_ave_time,
          float& best_gb_per_sec,
          int iter = 10)
{
    Tensor<InDataType> tensor_org(tensor_org_desc);
    Tensor<OutDataType> tensor_trans(tensor_trans_desc);

    // for result check
    Tensor<float> tensor_host_golden_fp32(tensor_org_desc);
    Tensor<float> tensor_host_result_fp32(tensor_org_desc);
    tensor_host_golden_fp32.SetZero();
    tensor_host_result_fp32.SetZero();

    // std::cout << "org desc: " << tensor_org.mDesc << std::endl;
    // std::cout << "trans desc: " << tensor_trans.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: tensor_org.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5}); break;
    default: tensor_org.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * tensor_org.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * tensor_trans.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(tensor_org.mData.data());

    std::array<ck::index_t, NumDim> org_g_n_c_wis_lengths{};
    std::array<ck::index_t, NumDim> trans_g_n_c_wis_lengths{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(tensor_org.GetLengths(), org_g_n_c_wis_lengths);
    copy(tensor_trans.GetLengths(), trans_g_n_c_wis_lengths);

    bool pass = true;

    auto run_impl = [&](auto& op_ptr, auto& argument_ptr) {
        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            out_device_buf.SetZero();
            std::string op_name = op_ptr->GetTypeString();

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            float ave_time = 0.0f;

            // warmup
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            for(int i = 0; i < iter; ++i)
            {
                ave_time +=
                    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});
            }
            ave_time /= iter;

            std::size_t transfer_bytes =
                tensor_org.GetElementSpaceSizeInBytes() + tensor_trans.GetElementSpaceSizeInBytes();
            float gb_per_sec = transfer_bytes / 1.E6 / ave_time;
            // std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s, " << op_name
            //           << std::endl;

            if(gb_per_sec > best_gb_per_sec)
            {
                best_op_name    = op_name;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                out_device_buf.FromDevice(tensor_trans.mData.data());

                if constexpr(is_same_v<InLayout, tensor_layout::convolution::NGCHW> &&
                             is_same_v<OutLayout, tensor_layout::convolution::NGCHWc32>)
                {
                    tensor_trans.ForEach([&](auto&, auto idx) mutable {
                        // idx order: (g, n, c_tildes, hi, wi, c_vect)
                        const auto c_idx = idx[2] * 32 + idx[5];
                        if(c_idx < tensor_host_golden_fp32.GetLengths()[2])
                        {
                            tensor_host_golden_fp32(
                                idx[0], idx[1], idx[2] * 32 + idx[5], idx[3], idx[4]) =
                                static_cast<float>(tensor_org(
                                    idx[0], idx[1], idx[2] * 32 + idx[5], idx[3], idx[4]));
                            tensor_host_result_fp32(
                                idx[0], idx[1], idx[2] * 32 + idx[5], idx[3], idx[4]) =
                                static_cast<float>(tensor_trans(idx));
                        }
                    });
                }
                else if constexpr(is_same_v<InLayout, tensor_layout::convolution::NGCHWc32> &&
                                  is_same_v<OutLayout, tensor_layout::convolution::NGCHW>)
                {
                    tensor_trans.ForEach([&](auto&, auto idx) mutable {
                        // idx order: (g, n, c, hi, wi)
                        tensor_host_golden_fp32(
                            idx[0], idx[1], idx[2] / 32, idx[3], idx[4], idx[2] % 32) =
                            static_cast<float>(tensor_org(
                                idx[0], idx[1], idx[2] / 32, idx[3], idx[4], idx[2] % 32));
                        tensor_host_result_fp32(
                            idx[0], idx[1], idx[2] / 32, idx[3], idx[4], idx[2] % 32) =
                            static_cast<float>(tensor_trans(idx));
                    });
                }
                else if constexpr(is_same_v<InLayout, tensor_layout::convolution::NGCHW> &&
                                  is_same_v<OutLayout, tensor_layout::convolution::NGCHWc<16>>)
                {
                    tensor_trans.ForEach([&](auto&, auto idx) mutable {
                        // idx order: (g, n, c_tildes, hi, wi, c_vect)
                        const auto c_idx = idx[2] * 16 + idx[5];
                        if(c_idx < tensor_host_golden_fp32.GetLengths()[2])
                        {
                            tensor_host_golden_fp32(
                                idx[0], idx[1], idx[2] * 16 + idx[5], idx[3], idx[4]) =
                                static_cast<float>(tensor_org(
                                    idx[0], idx[1], idx[2] * 16 + idx[5], idx[3], idx[4]));
                            tensor_host_result_fp32(
                                idx[0], idx[1], idx[2] * 16 + idx[5], idx[3], idx[4]) =
                                static_cast<float>(tensor_trans(idx));
                        }
                    });
                }
                else if constexpr(is_same_v<InLayout, tensor_layout::convolution::NGCHWc<16>> &&
                                  is_same_v<OutLayout, tensor_layout::convolution::NGCHW>)
                {
                    tensor_trans.ForEach([&](auto&, auto idx) mutable {
                        // idx order: (g, n, c, hi, wi)
                        tensor_host_golden_fp32(
                            idx[0], idx[1], idx[2] / 16, idx[3], idx[4], idx[2] % 16) =
                            static_cast<float>(tensor_org(
                                idx[0], idx[1], idx[2] / 16, idx[3], idx[4], idx[2] % 16));
                        tensor_host_result_fp32(
                            idx[0], idx[1], idx[2] / 16, idx[3], idx[4], idx[2] % 16) =
                            static_cast<float>(tensor_trans(idx));
                    });
                }

                pass = pass & ck::utils::check_err(tensor_host_result_fp32.mData,
                                                   tensor_host_golden_fp32.mData,
                                                   "Error: Incorrect results!",
                                                   1e-5,
                                                   3e-4);
            }
        }
        else
        {
            // std::cout << op_ptr->GetTypeString() << " does not support this problem" <<
            // std::endl;
        }
    };

    using DeviceOp = ck::tensor_operation::device::
        DeviceLayoutTransform<NumDim, InDataType, OutDataType, InLayout, OutLayout>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    // std::cout << "trans found " << op_ptrs.size() << " instances" << std::endl;

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(org_g_n_c_wis_lengths,
                                                        trans_g_n_c_wis_lengths,
                                                        in_device_buf.GetDeviceBuffer(),
                                                        out_device_buf.GetDeviceBuffer());
        run_impl(op_ptr, argument_ptr);
    }

    return pass;
}

template <ck::index_t NumDim,
          typename InType,
          typename InTypeTrans,
          typename WeiType,
          typename WeiTypeTrans,
          typename OutType,
          typename OutTypeTrans,
          typename InLayout,
          typename InLayoutTrans,
          typename WeiLayout,
          typename WeiLayoutTrans,
          typename OutLayout,
          typename OutLayoutTrans>
bool profile_layout_transform_impl(int do_verification,
                                   int init_method,
                                   int /*do_log*/,
                                   bool time_kernel,
                                   const ck::utils::conv::ConvParam& conv_param,
                                   int iter = 10)
{
    std::array<std::string, 3> best_op_name = {"", "", ""};
    std::array<float, 3> best_ave_time      = {0, 0, 0};
    std::array<float, 3> best_gb_per_sec    = {0, 0, 0};

    // input trans
    const auto in_g_n_c_wis_org_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto in_g_n_c_wis_trans_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayoutTrans>(
            conv_param);

    run<NumDim, InType, InTypeTrans, InLayout, InLayoutTrans>(do_verification,
                                                              init_method,
                                                              time_kernel,
                                                              in_g_n_c_wis_org_desc,
                                                              in_g_n_c_wis_trans_desc,
                                                              best_op_name[0],
                                                              best_ave_time[0],
                                                              best_gb_per_sec[0],
                                                              iter);

    // filter trans
    const auto wei_g_k_c_xs_org_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto wei_g_k_c_xs_trans_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayoutTrans>(
            conv_param);

    bool skip_filter_trans =
        conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1] == 1;
    if(!skip_filter_trans)
    {
        run<NumDim,
            WeiType,
            WeiTypeTrans,
            typename LayoutTraits<WeiLayout>::type,
            typename LayoutTraits<WeiLayoutTrans>::type>(do_verification,
                                                         init_method,
                                                         time_kernel,
                                                         wei_g_k_c_xs_org_desc,
                                                         wei_g_k_c_xs_trans_desc,
                                                         best_op_name[1],
                                                         best_ave_time[1],
                                                         best_gb_per_sec[1],
                                                         iter);
    }

    // out trans
    const auto out_g_n_k_wos_org_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);
    const auto out_g_n_k_wos_trans_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayoutTrans>(
            conv_param);
    run<NumDim,
        OutType,
        OutTypeTrans,
        typename LayoutTraits<OutLayout>::type,
        typename LayoutTraits<OutLayoutTrans>::type>(do_verification,
                                                     init_method,
                                                     time_kernel,
                                                     out_g_n_k_wos_org_desc,
                                                     out_g_n_k_wos_trans_desc,
                                                     best_op_name[2],
                                                     best_ave_time[2],
                                                     best_gb_per_sec[2],
                                                     iter);

    // std::cout << "Best input trans:"
    //           << "\nname: " << best_op_name[0] << "\navg_time: " << best_ave_time[0]
    //           << "\nGB/s: " << best_gb_per_sec[0] << std::endl;

    // if(!skip_filter_trans)
    // {
    //     std::cout << "Best filter trans:"
    //               << "\nname: " << best_op_name[1] << "\navg_time: " << best_ave_time[1]
    //               << "\nGB/s: " << best_gb_per_sec[1] << std::endl;
    // }

    // std::cout << "Best output trans:"
    //           << "\nname: " << best_op_name[2] << "\navg_time: " << best_ave_time[2]
    //           << "\nGB/s: " << best_gb_per_sec[2] << std::endl;

    // std::cout << "Total trans time: "
    //           << std::accumulate(std::begin(best_ave_time), std::end(best_ave_time), 0.0f) << "
    //           ms"
    //           << std::endl;

    // clang-format off
    std::cout << in_g_n_c_wis_org_desc << " | " 
              << in_g_n_c_wis_trans_desc << " | " 
              << best_op_name[0] << " | "
              << best_ave_time[0] << " ms | " 
              << best_gb_per_sec[0] << " GB/s" << std::endl;

    if(!skip_filter_trans)
    {
        std::cout << wei_g_k_c_xs_org_desc << " | " 
                  << wei_g_k_c_xs_trans_desc << " | " 
                  << best_op_name[1] << " | "
                  << best_ave_time[1] << " ms | " 
                  << best_gb_per_sec[1] << " GB/s" << std::endl;
    }

    std::cout << out_g_n_k_wos_org_desc << " | " 
              << out_g_n_k_wos_trans_desc << " | " 
              << best_op_name[2] << " | "
              << best_ave_time[2] << " ms | " 
              << best_gb_per_sec[2] << " GB/s" << std::endl;

    // clang-format on

    return 0;
}

} // namespace profiler
} // namespace ck
