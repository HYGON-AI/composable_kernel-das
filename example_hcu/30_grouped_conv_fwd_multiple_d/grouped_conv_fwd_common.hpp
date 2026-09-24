// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

// Reuse the established strided convolution descriptors and CPU reference.
#include "../../example/30_grouped_conv_fwd_multiple_d/common.hpp"

namespace hcu_conv_multi_d {

template <typename T, ck::index_t Dim, bool Fused>
using DeviceConv = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<
    Dim, InputLayout<Dim>, WeightLayout<Dim>,
    std::conditional_t<Fused, ck::Tuple<ctl::G_K, OutputLayout<Dim>>, ck::Tuple<>>,
    OutputLayout<Dim>, T, T, float, float,
    std::conditional_t<Fused, ck::Tuple<T, T>, ck::Tuple<>>, T,
    PassThrough, PassThrough,
    std::conditional_t<Fused, ck::tensor_operation::element_wise::AddReluAdd, PassThrough>,
    ConvSpec, GemmSpec, 1, 256, 128, 128, 32,
    std::is_same_v<T, float> ? 4 : 8, std::is_same_v<T, float> ? 4 : 8,
    16, 16, 4, 4,
    S<4,64,1>, S<1,0,2>, S<1,0,2>, 2,
    std::is_same_v<T, float> ? 4 : 8, std::is_same_v<T, float> ? 4 : 8, 1,
    S<4,64,1>, S<1,0,2>, S<1,0,2>, 2,
    std::is_same_v<T, float> ? 4 : 8, std::is_same_v<T, float> ? 4 : 8, 1,
    1, 1, S<1,32,1,8>, 4>;

template <typename T, ck::index_t Dim, bool Fused>
bool run(const ck::utils::conv::ConvParam& p)
{
    using Conv = DeviceConv<T, Dim, Fused>;
    using Op = std::conditional_t<Fused, ck::tensor_operation::element_wise::AddReluAdd,
                                 PassThrough>;
    Tensor<T> in(make_input_descriptor(p));
    Tensor<T> weight(make_weight_descriptor(p));
    Tensor<T> bias(make_bias_descriptor(p));
    Tensor<T> residual(make_output_descriptor(p));
    Tensor<T> expected(make_output_descriptor(p));
    Tensor<T> actual(expected.mDesc);
    Tensor<float> c(expected.mDesc);
    // Fill physical storage once, including the broadcast bias, with bounded
    // binary fractions. Every group, spatial position and channel varies.
    auto fill = [](auto& tensor, int multiplier, int modulus) {
        for(std::size_t i = 0; i < tensor.mData.size(); ++i)
            tensor.mData[i] = ck::type_convert<T>(
                static_cast<float>(static_cast<int>(i * multiplier % modulus) - modulus / 2) / 16);
    };
    fill(in, 7, 23);
    fill(weight, 5, 19);
    fill(bias, 3, 11);
    fill(residual, 11, 17);
    DeviceMem ad(in.mData.size() * sizeof(T)), bd(weight.mData.size() * sizeof(T));
    DeviceMem d0d(bias.mData.size() * sizeof(T)), d1d(residual.mData.size() * sizeof(T));
    DeviceMem ed(actual.mData.size() * sizeof(T));
    ad.ToDevice(in.mData.data());
    bd.ToDevice(weight.mData.data());
    d0d.ToDevice(bias.mData.data());
    d1d.ToDevice(residual.mData.data());
    ed.SetZero();
    using Shape = std::array<ck::index_t, Dim + 3>;
    auto array = [](const auto& v) {
        Shape a{};
        std::copy(v.begin(), v.end(), a.begin());
        return a;
    };
    auto spatial = [](const auto& v) {
        std::array<ck::index_t, Dim> a{};
        std::copy(v.begin(), v.end(), a.begin());
        return a;
    };
    std::array<const void*, Fused ? 2 : 0> ds{};
    std::array<Shape, Fused ? 2 : 0> ds_lengths{}, ds_strides{};
    if constexpr(Fused)
    {
        ds = {d0d.GetDeviceBuffer(), d1d.GetDeviceBuffer()};
        ds_lengths = {array(bias.mDesc.GetLengths()), array(residual.mDesc.GetLengths())};
        ds_strides = {array(bias.mDesc.GetStrides()), array(residual.mDesc.GetStrides())};
    }
    Conv conv;
    auto arg = conv.MakeArgument(
        ad.GetDeviceBuffer(), bd.GetDeviceBuffer(), ds, ed.GetDeviceBuffer(),
        array(in.mDesc.GetLengths()), array(in.mDesc.GetStrides()),
        array(weight.mDesc.GetLengths()), array(weight.mDesc.GetStrides()),
        ds_lengths, ds_strides, array(actual.mDesc.GetLengths()), array(actual.mDesc.GetStrides()),
        spatial(p.conv_filter_strides_), spatial(p.conv_filter_dilations_),
        spatial(p.input_left_pads_), spatial(p.input_right_pads_),
        PassThrough{}, PassThrough{}, Op{});
    if(!conv.IsSupportedArgument(arg))
    {
        std::cerr << "Unsupported grouped convolution: " << conv.GetTypeString() << '\n';
        return false;
    }
    conv.MakeInvoker().Run(arg, StreamConfig{nullptr, false});
    ed.FromDevice(actual.mData.data());
    using Ref = ck::tensor_operation::host::ReferenceConvFwd<
        Dim, T, T, float, PassThrough, PassThrough, PassThrough>;
    Ref ref;
    auto ref_arg = ref.MakeArgument(in, weight, c, p.conv_filter_strides_,
                                    p.conv_filter_dilations_, p.input_left_pads_, p.input_right_pads_,
                                    PassThrough{}, PassThrough{}, PassThrough{});
    ref.MakeInvoker().Run(ref_arg);
    expected.ForEach([&](auto&, auto idx) {
        float value = c(idx);
        if constexpr(Fused)
            value = std::max(0.0f, value + ck::type_convert<float>(bias(idx))) +
                    ck::type_convert<float>(residual(idx));
        expected(idx) = ck::type_convert<T>(value);
    });
    const double tolerance = std::is_same_v<T, float> ? 1e-5 :
                             std::is_same_v<T, ck::half_t> ? 1e-3 : 1e-2;
    const bool ok = ck::utils::check_err(actual, expected, "Grouped convolution mismatch",
                                         tolerance, tolerance);
    std::cout << (ok ? "PASS " : "FAIL ") << Dim << "D G=" << p.G_ << " N=" << p.N_
              << " K=" << p.K_ << " C=" << p.C_ << (Fused ? " bias_relu_residual" : " conv")
              << '\n';
    return ok;
}

template <typename T>
int run_examples()
{
    using Param = ck::utils::conv::ConvParam;
    const std::array<Param, 6> cases{{
        {1, 1, 2, 20, 8, {3}, {17}, {1}, {1}, {1}, {1}},
        {1, 3, 2, 12, 8, {3}, {19}, {2}, {2}, {0}, {1}},
        {2, 3, 2, 16, 16, {1,1}, {5,7}, {1,1}, {1,1}, {0,0}, {0,0}},
        {2, 2, 2, 132, 8, {3,3}, {9,9}, {1,1}, {1,1}, {1,1}, {1,1}},
        {2, 4, 2, 20, 8, {3,3}, {9,11}, {2,1}, {2,1}, {1,0}, {2,1}},
        {3, 2, 1, 12, 8, {3,1,3}, {5,4,7}, {1,1,2}, {1,1,1}, {1,0,1}, {1,0,0}}
    }};
    bool ok = true;
    for(const auto& p : cases)
    {
        switch(p.num_dim_spatial_)
        {
        case 1:
            ok = run<T, 1, false>(p) && ok;
            ok = run<T, 1, true>(p) && ok;
            break;
        case 2:
            ok = run<T, 2, false>(p) && ok;
            ok = run<T, 2, true>(p) && ok;
            break;
        case 3:
            ok = run<T, 3, false>(p) && ok;
            ok = run<T, 3, true>(p) && ok;
            break;
        }
    }
    return ok ? 0 : 1;
}

} // namespace hcu_conv_multi_d
