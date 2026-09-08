// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_fused_conv.hpp"
#include "ck_tile/ops/conv.hpp"
#include "ck_tile/ops/fused_conv.hpp"
#include "ck_tile/ops/elementwise.hpp"

using InDataType   = ck_tile::fp16_t;
using WeiDataType  = ck_tile::fp16_t;
using OutDataType  = ck_tile::fp16_t;
using AccDataType  = float;
using BiasDataType = ck_tile::fp16_t;

#if 1
using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;
using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;
#else
using InLayout  = ck_tile::tensor_layout::convolution::NGCHWc<32>;
using WeiLayout = ck_tile::tensor_layout::convolution::GKCYXc<32>;
using OutLayout = ck_tile::tensor_layout::convolution::NGKHWk<32>;
#endif

using InElementOp  = ck_tile::element_wise::PassThrough;
using WeiElementOp = ck_tile::element_wise::PassThrough;
using OutElementOp = ck_tile::element_wise::AddAddRelu;

// configs
static constexpr ck_tile::index_t MPerWG    = 64;
static constexpr ck_tile::index_t NPerWG    = 128;
static constexpr ck_tile::index_t KPerBlock = 32;

static constexpr ck_tile::index_t MWGs = 2;
static constexpr ck_tile::index_t NWGs = 1;

static constexpr ck_tile::index_t MWarpsPerWG = 2;
static constexpr ck_tile::index_t NWarpsPerWG = 2;

static constexpr ck_tile::index_t MmmacIter       = 2;
static constexpr ck_tile::index_t NmmacIter       = 2;
static constexpr ck_tile::index_t MmmacInterleave = 1;
static constexpr ck_tile::index_t NmmacInterleave = 1;

static constexpr auto InWeiGmemLoadVecLens           = ck_tile::sequence<8, 8>{};
static constexpr auto InWeiSmemLoadStoreVecLens      = ck_tile::sequence<8, 8>{};
static constexpr ck_tile::index_t OutGmemStoreVecLen = 8;
static constexpr ck_tile::index_t MinBlockPerCU      = 1;

// tile shape
static constexpr auto WGs     = ck_tile::sequence<MWGs, NWGs>{};
static constexpr auto WGTile  = ck_tile::sequence<MPerWG, NPerWG, KPerBlock>{};
static constexpr auto WGWarps = ck_tile::sequence<MWarpsPerWG, NWarpsPerWG>{};
static constexpr auto WarpTile =
    ck_tile::sequence<MmmacIter, NmmacIter, MmmacInterleave, NmmacInterleave>{};

// fuse mode
// static constexpr auto FuseMode = ck_tile::FusedConvModeTraits<OutElementOp>::value;

static constexpr ck_tile::index_t NDimSpatial = 2;
static constexpr ck_tile::index_t NDim        = NDimSpatial + 3;

static constexpr ck_tile::index_t NumLdsStages = 2;

static constexpr auto ConvSpec = ck_tile::ConvFwdSpecEnum::F1x1_S1_D1_P0;

bool run_fused_conv(const ck_tile::conv::ConvParam& conv_param, bool time_kernel)
{
    const auto in_g_n_c_wis_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    ck_tile::HostTensor<InDataType> in(in_g_n_c_wis_desc);
    ck_tile::HostTensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    ck_tile::HostTensor<OutDataType> out_host(out_g_n_k_wos_desc);
    ck_tile::HostTensor<OutDataType> out_device(out_g_n_k_wos_desc);
    ck_tile::HostTensor<BiasDataType> bias({conv_param.G_, conv_param.K_});
    ck_tile::HostTensor<OutDataType> res(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;
    std::cout << "bias: " << bias.mDesc << std::endl;
    std::cout << "res: " << res.mDesc << std::endl;

#if 1
    ck_tile::FillUniformDistribution<InDataType>{0.0f, 1.0f}(in);
    ck_tile::FillUniformDistribution<WeiDataType>{-.0005f, 0.0005f}(wei);
    ck_tile::FillUniformDistribution<BiasDataType>{-.0000f, 0.0000f}(bias);
    ck_tile::FillUniformDistribution<OutDataType>{0.0000f, 0.0000f}(res);
#else
    ck_tile::FillUniformDistribution<InDataType>{1.0f, 1.0f}(in);
    ck_tile::FillUniformDistribution<WeiDataType>{1.0f, 1.0f}(wei);
    ck_tile::FillUniformDistribution<BiasDataType>{-.0000f, 0.0000f}(bias);
    ck_tile::FillUniformDistribution<OutDataType>{0.0000f, 0.0000f}(res);
#endif

    ck_tile::DeviceMem in_device_buf(in.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_device_buf(wei.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_device_buf(out_device.get_element_space_size_in_bytes());
    ck_tile::DeviceMem bias_device_buf(bias.get_element_space_size_in_bytes());
    ck_tile::DeviceMem res_device_buf(res.get_element_space_size_in_bytes());

    in_device_buf.ToDevice(in.data());
    wei_device_buf.ToDevice(wei.data());
    bias_device_buf.ToDevice(bias.data());
    res_device_buf.ToDevice(res.data());
    out_host.SetZero();
    out_device.SetZero();

    std::array<ck_tile::index_t, NDim> in_g_n_c_wis_lengths{};
    std::array<ck_tile::index_t, NDim> in_g_n_c_wis_strides{};
    std::array<ck_tile::index_t, NDim> wei_g_k_c_xs_lengths{};
    std::array<ck_tile::index_t, NDim> wei_g_k_c_xs_strides{};
    std::array<ck_tile::index_t, NDim> out_g_n_k_wos_lengths{};
    std::array<ck_tile::index_t, NDim> out_g_n_k_wos_strides{};
    std::array<ck_tile::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck_tile::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck_tile::index_t, NDimSpatial> input_left_pads{};
    std::array<ck_tile::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };

    copy(in_g_n_c_wis_desc.get_lengths(), in_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.get_strides(), in_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.get_lengths(), wei_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.get_strides(), wei_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.get_lengths(), out_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.get_strides(), out_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // gpu fused conv

    using TileShape = ck_tile::FusedConvWaspTileShapeV2<decltype(WGs),
                                                        decltype(WGTile),
                                                        decltype(WGWarps),
                                                        decltype(WarpTile)>;

    using TileTraits =
        ck_tile::FusedConvTileTraits<ck_tile::tuple<InLayout, WeiLayout, OutLayout>,
                                     ck_tile::tuple<InElementOp, WeiElementOp, OutElementOp>,
                                     decltype(InWeiGmemLoadVecLens),
                                     decltype(InWeiSmemLoadStoreVecLens),
                                     OutGmemStoreVecLen>;

    using Problem = ck_tile::FusedConvWaspProblemV2<
        NDimSpatial,
        ck_tile::tuple<InDataType, WeiDataType, OutDataType, AccDataType, OutDataType, OutDataType>,
        TileShape,
        TileTraits,
        NumLdsStages,
        ConvSpec>;

    using Policy = ck_tile::FusedConvIgemmPipelineWaspPolicyV3;

    using Pipeline = ck_tile::FusedConvIgemmPipelineWaspAgmemBgmemCregV3<Problem, Policy>;

    using TilePartitioner = ck_tile::FusedConvPersistantTilePartitionerV1<TileShape>;

    using BlockwiseGemm = ck_tile::remove_cvref_t<
        decltype(Policy::template GetBlockwiseGemm<Problem>())>;

    using WarpGemm = typename BlockwiseGemm::WarpGemm;

    using Epilogue = ck_tile::FusedConvEpilogueWaspV2<Problem, WarpGemm>;

    using Kernel = ck_tile::FusedConvWaspKernelV3<TilePartitioner, Pipeline, Epilogue>;

    auto host_args = Kernel::MakeHostArgs(in_device_buf.GetDeviceBuffer(),
                                          wei_device_buf.GetDeviceBuffer(),
                                          out_device_buf.GetDeviceBuffer(),
                                          bias_device_buf.GetDeviceBuffer(),
                                          res_device_buf.GetDeviceBuffer(),
                                          in_g_n_c_wis_lengths,
                                          in_g_n_c_wis_strides,
                                          wei_g_k_c_xs_lengths,
                                          wei_g_k_c_xs_strides,
                                          out_g_n_k_wos_lengths,
                                          out_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads);

    auto kernel_args = Kernel::MakeKernelArgs(host_args);

    if(!Kernel::IsSupportedArgument(host_args, kernel_args))
    {
        throw std::runtime_error("wrong! fused_conv with the specified parameters does "
                                 "not support this Conv problem");
    }

    float avg_time = ck_tile::launch_kernel(
        {nullptr, time_kernel, 0, time_kernel ? 1 : 0, time_kernel ? 10 : 1},
        ck_tile::make_kernel<TileShape::BlockSize, MinBlockPerCU>(
            Kernel{},
            kernel_args.tile_partitioner.CalculateGridSize(Kernel::GetMaxOccupancy()),
            TileShape::BlockSize,
            0,
            kernel_args,
            ck_tile::bool_constant<true>{}));

    if(time_kernel)
    {
        std::size_t flop      = conv_param.GetFlops();
        std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

        float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
        float gb_per_sec = num_btype / 1.E6 / avg_time;
        std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << std::endl;
    }
    else
    {
        std::cout << "timing:off" << std::endl;
    }

    // cpu fused conv
    auto ref_fused_conv = ck_tile::ReferenceFusedConv<NDimSpatial,
                                                      InDataType,
                                                      WeiDataType,
                                                      OutDataType,
                                                      BiasDataType,
                                                      InElementOp,
                                                      WeiElementOp,
                                                      OutElementOp>();
    auto ref_invoker    = ref_fused_conv.MakeInvoker();
    auto ref_arg        = ref_fused_conv.MakeArgument(in,
                                               wei,
                                               out_host,
                                               bias,
                                               res,
                                               conv_param.conv_filter_strides_,
                                               conv_param.conv_filter_dilations_,
                                               conv_param.input_left_pads_,
                                               conv_param.input_right_pads_,
                                               InElementOp{},
                                               WeiElementOp{},
                                               OutElementOp{});
    ref_invoker.Run(ref_arg, false); // skip relu

    // check
    out_device_buf.FromDevice(out_device.data());
    return ck_tile::check_err(out_device, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
}

bool run_fused_conv_example(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "2", "spatial dim")
        .insert("g", "1", "group")
        .insert("n", "128", "batch size")
        .insert("k", "1024", "out channels")
        .insert("c", "256", "in channels")
        .insert("r", "1", "filter x")
        .insert("s", "1", "filter y")
        .insert("h", "14", "input x")
        .insert("w", "14", "input y")
        .insert("u", "1", "stride h")
        .insert("v", "1", "stride w")
        .insert("l", "1", "dilation h")
        .insert("j", "1", "dilation w")
        .insert("p", "0", "pad h")
        .insert("q", "0", "pad w")
        .insert("time", "1", "time kernel (0=no, 1=yes)");

    bool result = arg_parser.parse(argc, argv);

    if(!result)
        return false;

    ck_tile::conv::ConvParam conv_param{arg_parser.get_int("m"),
                                        arg_parser.get_int("g"),
                                        arg_parser.get_int("n"),
                                        arg_parser.get_int("k"),
                                        arg_parser.get_int("c"),
                                        {arg_parser.get_int("r"), arg_parser.get_int("s")},
                                        {arg_parser.get_int("h"), arg_parser.get_int("w")},
                                        {arg_parser.get_int("u"), arg_parser.get_int("v")},
                                        {arg_parser.get_int("l"), arg_parser.get_int("j")},
                                        {arg_parser.get_int("p"), arg_parser.get_int("p")},
                                        {arg_parser.get_int("q"), arg_parser.get_int("q")}};

    if(arg_parser.get_int("m") == 2)
    {
        return run_fused_conv(conv_param, arg_parser.get_bool("time"));
    }
    else
    {
        printf("unsupport spatial dim\n");
        return false;
    }
}

int main(int argc, char* argv[]) { return run_fused_conv_example(argc, argv) ? 0 : 1; }
