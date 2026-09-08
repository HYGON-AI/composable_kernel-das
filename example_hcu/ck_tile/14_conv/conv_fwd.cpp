// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_conv_fwd.hpp"
#include "ck_tile/ops/conv.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/epilogue.hpp"

using ADataType   = ck_tile::fp16_t;
using BDataType   = ck_tile::fp16_t;
using CDataType   = ck_tile::fp16_t;
using AccDataType = float;

using ALayout = ck_tile::tensor_layout::convolution::NHWGC;
using BLayout = ck_tile::tensor_layout::convolution::GKYXC;
using CLayout = ck_tile::tensor_layout::convolution::NHWGK;

using AElementOp = ck_tile::element_wise::PassThrough;
using BElementOp = ck_tile::element_wise::PassThrough;
using CElementOp = ck_tile::element_wise::PassThrough;

static constexpr ck_tile::index_t MBlockTile = 64;
static constexpr ck_tile::index_t NBlockTile = 64;
static constexpr ck_tile::index_t KBlockTile = 16;

static constexpr ck_tile::index_t MWarp = 2;
static constexpr ck_tile::index_t NWarp = 2;

static constexpr ck_tile::index_t MWarpIter = 2;
static constexpr ck_tile::index_t NWarpIter = 2;

static constexpr ck_tile::index_t MmmacIter = 1;
static constexpr ck_tile::index_t NmmacIter = 1;

static constexpr ck_tile::index_t MmmacInterleave = 1;
static constexpr ck_tile::index_t NmmacInterleave = 1;

static constexpr ck_tile::index_t NumPrefetch = 2;

static constexpr ck_tile::index_t BlockSize     = MWarp * NWarp * ck_tile::get_warp_size();
static constexpr ck_tile::index_t MinBlockPerCU = 2;

using BlockTile  = ck_tile::sequence<MBlockTile, NBlockTile, KBlockTile>;
using BlockWarps = ck_tile::sequence<MWarp, NWarp>;
using WarpTile = ck_tile::sequence<MmmacIter, NmmacIter, 16, 16, MmmacInterleave, NmmacInterleave>;

using TileShape = ck_tile::ConvIgemmTileShape<BlockTile, BlockWarps, WarpTile>;

using TileTraits =
    ck_tile::ConvIgemmUniversalTileTraits<ck_tile::tuple<ALayout, BLayout, CLayout>,
                                          ck_tile::tuple<AElementOp, BElementOp, CElementOp>,
                                          ck_tile::sequence<4, 4>,
                                          ck_tile::sequence<4, 4>,
                                          ck_tile::sequence<4, 4>,
                                          4>;

using Policy = ck_tile::ConvIgemmFwdPipelineDefaultPolicy;

using TilePartitioner = ck_tile::ConvIgemmTilePartitioner<TileShape>;

static constexpr auto ConvSpec = ck_tile::ConvFwdSpec::Default;

template <ck_tile::index_t NDimSpatial>
bool run_grouped_conv_fwd(const ck_tile::conv::ConvParam& conv_param,
                          bool time_kernel,
                          double rtol,
                          double atol)
{
    static constexpr ck_tile::index_t NDim = NDimSpatial + 3;

    const auto in_g_n_c_wis_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<ALayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<BLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<CLayout>(conv_param);

    ck_tile::HostTensor<ADataType> in(in_g_n_c_wis_desc);
    ck_tile::HostTensor<BDataType> wei(wei_g_k_c_xs_desc);
    ck_tile::HostTensor<CDataType> out_host(out_g_n_k_wos_desc);
    ck_tile::HostTensor<CDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(in);
    ck_tile::FillUniformDistribution<BDataType>{-0.05f, 0.05f}(wei);

    ck_tile::DeviceMem in_device_buf(in.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_device_buf(wei.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_device_buf(out_device.get_element_space_size_in_bytes());

    in_device_buf.ToDevice(in.data());
    wei_device_buf.ToDevice(wei.data());
    out_host.SetZero();
    out_device.SetZero();

    std::array<ck_tile::index_t, NDim> a_g_n_c_wis_lengths{};
    std::array<ck_tile::index_t, NDim> a_g_n_c_wis_strides{};
    std::array<ck_tile::index_t, NDim> b_g_k_c_xs_lengths{};
    std::array<ck_tile::index_t, NDim> b_g_k_c_xs_strides{};
    std::array<ck_tile::index_t, NDim> c_g_n_k_wos_lengths{};
    std::array<ck_tile::index_t, NDim> c_g_n_k_wos_strides{};
    std::array<ck_tile::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck_tile::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck_tile::index_t, NDimSpatial> input_left_pads{};
    std::array<ck_tile::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };

    copy(in_g_n_c_wis_desc.get_lengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.get_strides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.get_lengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.get_strides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.get_lengths(), c_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.get_strides(), c_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // gpu conv
    using Problem = ck_tile::ConvIgemmPipelineProblem<
        NDimSpatial,
        ck_tile::tuple<ADataType, BDataType, CDataType, AccDataType>,
        TileShape,
        TileTraits,
        NumPrefetch,
        true,
        true>;

    using EpilogueProblem = ck_tile::LdsCShuffleEpilogueProblem<
        AccDataType,
        CDataType,
        ck_tile::sequence<MWarpIter, NWarpIter>,
        ck_tile::sequence<MWarp, NWarp>,
        ck_tile::sequence<MmmacIter, NmmacIter, 16, 16, MmmacInterleave, NmmacInterleave>,
        true>;

    using WG = typename ck_tile::conv::WarpGemmMmacDispatcher<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              MmmacIter,
                                                              NmmacIter,
                                                              16,
                                                              16,
                                                              MmmacInterleave,
                                                              NmmacInterleave,
                                                              2,
                                                              true>;

    using Epilogue = ck_tile::LdsCShuffleEpilogue<EpilogueProblem, WG>;

    using Pipeline = ck_tile::ConvIgemmPipelineAGmemBGmemCRegV1<Problem, Policy>;

    using Kernel = ck_tile::ConvIgemmFwdKernel<TilePartitioner, Pipeline, Epilogue, ConvSpec>;

    auto host_args = Kernel::MakeHostArgs(in_device_buf.GetDeviceBuffer(),
                                          wei_device_buf.GetDeviceBuffer(),
                                          out_device_buf.GetDeviceBuffer(),
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          c_g_n_k_wos_lengths,
                                          c_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads);

    auto kernel_args = Kernel::MakeKernelArgs(host_args);

    const auto grid_size =
        kernel_args.tile_partitioner.CalculateGridSize() * conv_param.G_;

    float avg_time = ck_tile::launch_kernel(
        {nullptr, time_kernel},
        ck_tile::make_kernel<BlockSize, MinBlockPerCU>(
            Kernel{}, grid_size, BlockSize, 0, kernel_args));

    if(time_kernel)
    {
        std::size_t flop      = conv_param.GetFlops();
        std::size_t num_btype = conv_param.GetByte<ADataType, BDataType, CDataType>();

        float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
        float gb_per_sec = num_btype / 1.E6 / avg_time;
        std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << std::endl;
    }
    else
    {
        std::cout << "timing:off" << std::endl;
    }

    auto ref_conv = ck_tile::ReferenceConvFwd<NDimSpatial,
                                              ADataType,
                                              BDataType,
                                              CDataType,
                                              AElementOp,
                                              BElementOp,
                                              CElementOp>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in,
                                              wei,
                                              out_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              AElementOp{},
                                              BElementOp{},
                                              CElementOp{});

    ref_invoker.Run(ref_argument);

    out_device_buf.FromDevice(out_device.data());

    return ck_tile::check_err(out_device, out_host, "Error: incorrect results!", rtol, atol);
}

bool run_grouped_conv_fwd_example(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "2", "spatial dim")
        .insert("g", "1", "group")
        .insert("n", "64", "batch size")
        .insert("k", "64", "out channels")
        .insert("c", "64", "in channels")
        .insert("r", "3", "filter x")
        .insert("s", "3", "filter y")
        .insert("h", "4", "input x")
        .insert("w", "4", "input y")
        .insert("u", "1", "stride h")
        .insert("v", "1", "stride w")
        .insert("l", "1", "dilation h")
        .insert("j", "1", "dilation w")
        .insert("p", "1", "pad h")
        .insert("q", "1", "pad w")
        .insert("time", "1", "time kernel (0=no, 1=yes)")
        .insert("rtol", "1e-3", "relative validation tolerance")
        .insert("atol", "1e-3", "absolute validation tolerance");

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
        return run_grouped_conv_fwd<2>(
            conv_param,
            arg_parser.get_bool("time"),
            arg_parser.get_double("rtol"),
            arg_parser.get_double("atol"));
    }
    else
    {
        printf("unsupport spatial dim\n");
        return false;
    }
}

int main(int argc, char* argv[]) { return run_grouped_conv_fwd_example(argc, argv) ? 0 : 1; }
