// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_layout_transform_impl.hpp"
#include "profiler_operation_registry.hpp"

namespace {

enum struct TransLayout
{
    NGCHW_NGCHWc32_fwd, // 0
    NGCHW_NGCHWc32_bwd, // 1
    NGCHW_NGCHWc32_wrw, // 2
    NGCHW_NGCHWc16_fwd, // 3
    NGCHW_NGCHWc16_bwd, // 4
    NGCHW_NGCHWc16_wrw, // 5
};

enum struct TransDtype
{
    F16, // 0
};

#define OP_NAME "layout_transform"
#define OP_DESC "Tensor Layout Transform"

static void print_helper_msg()
{
    std::cout
        // clang-format-off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: trans data type (0: fp16\n"
        << "arg3: tensor layout (0: NGCHW_NGCHWc32 for conv fwd\n"
        << "                     1: NGCHW_NGCHWc32 for conv bwd\n"
        << "                     2: NGCHW_NGCHWc32 for conv wrw)\n"
        << "                     3: NGCHW_NGCHWc16 for conv fwd)\n"
        << "                     4: NGCHW_NGCHWc16 for conv bwd\n"
        << "                     5: NGCHW_NGCHWc16 for conv wrw)\n"
        << "arg4: verification (0: no, 1: yes)\n"
        << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg6: print tensor value (0: no; 1: yes)\n"
        << "arg7: time kernel (0: no, 1: yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format-on
}

} // namespace

int profile_layout_transform(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 9)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<TransDtype>(std::stoi(argv[2]));
    const auto layout          = static_cast<TransLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);
    const int num_dim_spatial  = std::stoi(argv[8]);

    // 8 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial
    if(argc != 8 + 1 + 4 + 6 * num_dim_spatial)
    {
        print_helper_msg();
        return 1;
    }

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 9, argv);

    using F16 = ck::half_t;
    using F32 = float;

    using NGCHW    = ck::tensor_layout::convolution::NGCHW;
    using GKCYX    = ck::tensor_layout::convolution::GKCYX;
    using NGKHW    = ck::tensor_layout::convolution::NGKHW;

    using NGCHWc32 = ck::tensor_layout::convolution::NGCHWc32;
    using GKCYXc32 = ck::tensor_layout::convolution::GKCYXc32;
    using NGKHWk32 = ck::tensor_layout::convolution::NGKHWk32;

    using NGCHWc16 = ck::tensor_layout::convolution::NGCHWc<16>;
    using GKCYXc16 = ck::tensor_layout::convolution::GKCYXc<16>;
    using NGKHWk16 = ck::tensor_layout::convolution::NGKHWk<16>;

    constexpr auto I2 = ck::Number<2>{};

    auto profile =
        [&](auto num_dim_spatial_tmp,
            auto in_type,
            auto in_type_trans,
            auto wei_type,
            auto wei_type_trans,
            auto out_type,
            auto out_type_trans,
            auto in_layout,
            auto in_layout_trans,
            auto wei_layout,
            auto wei_layout_trans,
            auto out_layout,
            auto out_layout_trans) {
            constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

            using InType       = decltype(in_type);
            using InTypeTrans  = decltype(in_type_trans);
            using WeiType      = decltype(wei_type);
            using WeiTypeTrans = decltype(wei_type_trans);
            using OutType      = decltype(out_type);
            using OutTypeTrans = decltype(out_type_trans);

            using InLayout       = decltype(in_layout);
            using InLayoutTrans  = decltype(in_layout_trans);
            using WeiLayout      = decltype(wei_layout);
            using WeiLayoutTrans = decltype(wei_layout_trans);
            using OutLayout      = decltype(out_layout);
            using OutLayoutTrans = decltype(out_layout_trans);

            bool pass = ck::profiler::profile_layout_transform_impl<NDimSpatial + 3,
                                                                    InType,
                                                                    InTypeTrans,
                                                                    WeiType,
                                                                    WeiTypeTrans,
                                                                    OutType,
                                                                    OutTypeTrans,
                                                                    InLayout,
                                                                    InLayoutTrans,
                                                                    WeiLayout,
                                                                    WeiLayoutTrans,
                                                                    OutLayout,
                                                                    OutLayoutTrans>(
                do_verification, init_method, do_log, time_kernel, params);

            return pass ? 0 : 1;
        };

    if(layout == TransLayout::NGCHW_NGCHWc32_fwd)
    {
        if(data_type == TransDtype::F16)
        {
            return profile(I2,
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           NGCHW{},
                           NGCHWc32{},
                           GKCYX{},
                           GKCYXc32{},
                           NGKHWk32{},
                           NGKHW{});
        }
    }
    else if(layout == TransLayout::NGCHW_NGCHWc32_bwd)
    {
        if(data_type == TransDtype::F16)
        {
            return profile(I2,
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           NGCHWc32{},
                           NGCHW{},
                           GKCYX{},
                           GKCYXc32{},
                           NGKHW{},
                           NGKHWk32{});
        }
    }
    else if(layout == TransLayout::NGCHW_NGCHWc32_wrw)
    {
        if(data_type == TransDtype::F16)
        {
            return profile(I2,
                           F16{},
                           F16{},
                           F32{},
                           F16{},
                           F16{},
                           F16{},
                           NGCHW{},
                           NGCHWc32{},
                           GKCYXc32{},
                           GKCYX{},
                           NGKHW{},
                           NGKHWk32{});
        }
    }
    else if(layout == TransLayout::NGCHW_NGCHWc16_fwd)
    {
        if(data_type == TransDtype::F16)
        {
            return profile(I2,
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           NGCHW{},
                           NGCHWc16{},
                           GKCYX{},
                           GKCYXc16{},
                           NGKHWk16{},
                           NGKHW{});
        }
    }
    else if(layout == TransLayout::NGCHW_NGCHWc16_bwd)
    {
        if(data_type == TransDtype::F16)
        {
            return profile(I2,
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           F16{},
                           NGCHWc16{},
                           NGCHW{},
                           GKCYX{},
                           GKCYXc16{},
                           NGKHW{},
                           NGKHWk16{});
        }
    }
    else if(layout == TransLayout::NGCHW_NGCHWc16_wrw)
    {
        if(data_type == TransDtype::F16)
        {
            return profile(I2,
                           F16{},
                           F16{},
                           F32{},
                           F16{},
                           F16{},
                           F16{},
                           NGCHW{},
                           NGCHWc16{},
                           GKCYXc16{},
                           GKCYX{},
                           NGKHW{},
                           NGKHWk16{});
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_layout_transform);
