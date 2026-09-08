// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_fwd_mmac_impl.hpp"

class TestGroupedConvNdFwd : public ::testing::Test
{
    public:
    TestGroupedConvNdFwd() : ::testing::Test()
    {
        conv_params.clear();

        // clang-format off

        // conv4_block1_2_conv
        conv_params.push_back({2, 1, 32, 256,  256,  {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
        // conv5_block1_0_conv
        conv_params.push_back({2, 1, 32, 2048, 1024, {1, 1}, {14, 14}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
        // conv3_block2_1_conv
        conv_params.push_back({2, 1, 32, 128,  512,  {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
        // grouped conv4_block1_2_conv
        conv_params.push_back({2, 2, 32, 256,  256,  {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
        // grouped conv5_block1_0_conv
        conv_params.push_back({2, 2, 32, 2048, 1024, {1, 1}, {14, 14}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
        // grouped conv3_block2_1_conv
        conv_params.push_back({2, 2, 32, 128,  512,  {1, 1}, {28, 28}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

        // clang-format on
    }

    protected:
    std::vector<ck::utils::conv::ConvParam> conv_params;
};

// 2d NHWGC/GKYXC/NHWGK
TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNHWGC_F16)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::NHWGC,
                                                           ck::tensor_layout::convolution::GKYXC,
                                                           ck::tensor_layout::convolution::NHWGK,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t>(true,  // do_verification
                                                                       2,     // init_method
                                                                       false, // do_log
                                                                       true,  // time_kernel
                                                                       param,
                                                                       false,
                                                                       1);

        EXPECT_TRUE(pass);
    }
}

TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNHWGC_BF16)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::NHWGC,
                                                           ck::tensor_layout::convolution::GKYXC,
                                                           ck::tensor_layout::convolution::NHWGK,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t>(true,  // do_verification
                                                                        2,     // init_method
                                                                        false, // do_log
                                                                        true,  // time_kernel
                                                                        param,
                                                                        false,
                                                                        1);

        EXPECT_TRUE(pass);
    }
}

TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNHWGC_F32)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::NHWGC,
                                                           ck::tensor_layout::convolution::GKYXC,
                                                           ck::tensor_layout::convolution::NHWGK,
                                                           float,
                                                           float,
                                                           float>(true,  // do_verification
                                                                  2,     // init_method
                                                                  false, // do_log
                                                                  true,  // time_kernel
                                                                  param,
                                                                  false,
                                                                  1);

        EXPECT_TRUE(pass);
    }
}

// 2d NGCHW/GKCYX/NGKHW
TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNGCHW_F16)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::NGCHW,
                                                           ck::tensor_layout::convolution::GKCYX,
                                                           ck::tensor_layout::convolution::NGKHW,
                                                           ck::half_t,
                                                           ck::half_t,
                                                           ck::half_t>(true,  // do_verification
                                                                       2,     // init_method
                                                                       false, // do_log
                                                                       true,  // time_kernel
                                                                       param,
                                                                       false,
                                                                       1);

        EXPECT_TRUE(pass);
    }
}

TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNGCHW_BF16)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::NGCHW,
                                                           ck::tensor_layout::convolution::GKCYX,
                                                           ck::tensor_layout::convolution::NGKHW,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t,
                                                           ck::bhalf_t>(true,  // do_verification
                                                                        2,     // init_method
                                                                        false, // do_log
                                                                        true,  // time_kernel
                                                                        param,
                                                                        false,
                                                                        1);

        EXPECT_TRUE(pass);
    }
}

TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNGCHW_F32)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass = ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                           ck::tensor_layout::convolution::NGCHW,
                                                           ck::tensor_layout::convolution::GKCYX,
                                                           ck::tensor_layout::convolution::NGKHW,
                                                           float,
                                                           float,
                                                           float>(true,  // do_verification
                                                                  2,     // init_method
                                                                  false, // do_log
                                                                  true,  // time_kernel
                                                                  param,
                                                                  false,
                                                                  1);

        EXPECT_TRUE(pass);
    }
}

// 2d NGCHWc32/GKCYXc32/NGKHWk32
TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNGCHWc32_F16)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass =
            ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                        ck::tensor_layout::convolution::NGCHWc<32>,
                                                        ck::tensor_layout::convolution::GKCYXc<32>,
                                                        ck::tensor_layout::convolution::NGKHWk<32>,
                                                        ck::half_t,
                                                        ck::half_t,
                                                        ck::half_t>(true,  // do_verification
                                                                    2,     // init_method
                                                                    false, // do_log
                                                                    true,  // time_kernel
                                                                    param,
                                                                    true,
                                                                    1);

        EXPECT_TRUE(pass);
    }
}

// 2d NGCHWc32/GKCYXc32/NGKHW
TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNGCHWc32_Mixed_F16)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass =
            ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                        ck::tensor_layout::convolution::NGCHWc<32>,
                                                        ck::tensor_layout::convolution::GKCYXc<32>,
                                                        ck::tensor_layout::convolution::NGKHW,
                                                        ck::half_t,
                                                        ck::half_t,
                                                        ck::half_t>(true,  // do_verification
                                                                    2,     // init_method
                                                                    false, // do_log
                                                                    true,  // time_kernel
                                                                    param,
                                                                    true,
                                                                    1);

        EXPECT_TRUE(pass);
    }
}

// 2d NGCHWc16/GKCYXc16/NGKHWk16
TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNGCHWc16_F16)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass =
            ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                        ck::tensor_layout::convolution::NGCHWc<16>,
                                                        ck::tensor_layout::convolution::GKCYXc<16>,
                                                        ck::tensor_layout::convolution::NGKHWk<16>,
                                                        ck::half_t,
                                                        ck::half_t,
                                                        ck::half_t>(true,  // do_verification
                                                                    2,     // init_method
                                                                    false, // do_log
                                                                    true,  // time_kernel
                                                                    param,
                                                                    true,
                                                                    1);

        EXPECT_TRUE(pass);
    }
}

// 2d NGCHWc16/GKCYXc16/NGKHW
TEST_F(TestGroupedConvNdFwd, GroupedConv2dFwdNGCHWc16_Mixed_F16)
{
    for(auto& param : conv_params)
    {
        bool pass;

        // fp16
        pass =
            ck::profiler::profile_grouped_conv_fwd_impl<2,
                                                        ck::tensor_layout::convolution::NGCHWc<16>,
                                                        ck::tensor_layout::convolution::GKCYXc<16>,
                                                        ck::tensor_layout::convolution::NGKHW,
                                                        ck::half_t,
                                                        ck::half_t,
                                                        ck::half_t>(true,  // do_verification
                                                                    2,     // init_method
                                                                    false, // do_log
                                                                    true,  // time_kernel
                                                                    param,
                                                                    true,
                                                                    1);

        EXPECT_TRUE(pass);
    }
}
