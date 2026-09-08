// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_grouped_conv_fwd_bias_relu_mmac_impl.hpp"
#include "profiler_operation_registry.hpp"

namespace {

enum struct ConvLayout
{
    NHWGC_GKYXC_NHWGK,          // 0
    NGCHW_GKCYX_NGKHW,          // 1
    NGCHWc32_GKCYXc32_NGKHWk32, // 2
    NGCHWc16_GKCYXc16_NGKHWk16, // 3
    NGCHWc32_GKCYXc32_NGKHW,    // 4
};

enum struct ConvDataType
{
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
};

#define OP_NAME "grouped_conv_fwd_bias_relu"
#define OP_DESC "Grouped Convolution Forward Bias Relu"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
        << "                 1: Input fp16, Weight fp16, Output fp16\n"
        << "                 2: Input bf16, Weight bf16, Output bf16\n"
        << "arg3: tensor layout (0: NHWGC\n"
        << "                     1: NGCHW\n"
        << "                     2: NGCHWc32\n"
        << "                     3: NGCHWc16\n"
        << "                     4: NGCHWc32_NGCHW_MIXED\n"
        << "arg4: verification (0: no, 1: yes)\n"
        << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg6: print tensor value (0: no; 1: yes)\n"
        << "arg7: time kernel (0: no, 1: yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format on
}

} // namespace

int profile_grouped_conv_fwd_bias_relu(int argc, char* argv[])
{

    // 8 for control, 1 for num_dim_spatial
    if(argc < 9)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);
    const int num_dim_spatial  = std::stoi(argv[8]);

    // 8 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial
    if((argc != 8 + 1 + 4 + 6 * num_dim_spatial) && (argc != 8 + 1 + 4 + 6 * num_dim_spatial + 1))
    {
        print_helper_msg();
        return 1;
    }

    const auto params = ck::utils::conv::parse_conv_param(
        num_dim_spatial, 9, argv, argc == 8 + 1 + 4 + 6 * num_dim_spatial + 1 ? true : false);

    using F16 = ck::half_t;

#if 0
    using NGCHW = ck::tensor_layout::convolution::NGCHW;
    using GKCYX = ck::tensor_layout::convolution::GKCYX;
    using NGKHW = ck::tensor_layout::convolution::NGKHW;

    using NGCHWc32 = ck::tensor_layout::convolution::NGCHWc32;
    using GKCYXc32 = ck::tensor_layout::convolution::GKCYXc32;
    using NGKHWk32 = ck::tensor_layout::convolution::NGKHWk32;

    using NGCHWc16 = ck::tensor_layout::convolution::NGCHWc<16>;
    using GKCYXc16 = ck::tensor_layout::convolution::GKCYXc<16>;
    using NGKHWk16 = ck::tensor_layout::convolution::NGKHWk<16>;
#endif

    using NHWGC = ck::tensor_layout::convolution::NHWGC;
    using GKYXC = ck::tensor_layout::convolution::GKYXC;
    using NHWGK = ck::tensor_layout::convolution::NHWGK;

    constexpr auto I2 = ck::Number<2>{};

    auto profile = [&](auto num_dim_spatial_tmp,
                       auto in_layout,
                       auto wei_layout,
                       auto out_layout,
                       auto in_type,
                       auto wei_type,
                       auto out_type,
                       bool packed_channel = false) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using OutLayout = decltype(out_layout);

        using InDataType  = decltype(in_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

        bool pass = ck::profiler::profile_grouped_conv_fwd_bias_relu_impl<NDimSpatial,
                                                                          InLayout,
                                                                          WeiLayout,
                                                                          OutLayout,
                                                                          InDataType,
                                                                          WeiDataType,
                                                                          OutDataType>(
            do_verification, init_method, do_log, time_kernel, params, packed_channel, 1);

        return pass ? 0 : 1;
    };

    // NHWGC_GKYXC_NHWGK
    if(num_dim_spatial == 2 && layout == ConvLayout::NGCHW_GKCYX_NGKHW)
    {

        if(data_type == ConvDataType::F16_F16_F16)
        {
            // return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, F16{}, F16{}, F16{});
        }
    }
#if 0
    else if(num_dim_spatial == 2 & layout == ConvLayout::NGCHWc32_GKCYXc32_NGKHWk32)
    {
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NGCHWc32{}, GKCYXc32{}, NGKHWk32{}, F16{}, F16{}, F16{}, true);
        }
    }
    else if(num_dim_spatial == 2 & layout == ConvLayout::NGCHWc16_GKCYXc16_NGKHWk16)
    {
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NGCHWc16{}, GKCYXc16{}, NGKHWk16{}, F16{}, F16{}, F16{}, true);
        }
    }
    else if(num_dim_spatial == 2 & layout == ConvLayout::NGCHWc32_GKCYXc32_NGKHW)
    {
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NGCHWc32{}, GKCYXc32{}, NGKHW{}, F16{}, F16{}, F16{}, true);
        }
    }
#endif
    else if(num_dim_spatial == 2 & layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F16{}, F16{}, F16{});
        }
    }

    // TODO: add more layout support

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_fwd_bias_relu);
