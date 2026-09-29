// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

// ConvIgemmFwdKernel: bf16 inputs + fp32 output (AccDataType = fp32).
//
// Two input-layout configurations are provided and selected at runtime:
//   -layout 0 (default): channels-last style
//          A = NHWGC, B = GKYXC, C = NHWGK
//          gmem/smem load vector lengths <4, 4>, C gmem store vector length 4
//   -layout 1: C-interleaved blocked style (x = 32)
//          A = NGCHWc<32>, B = GKCYXc<32>, C = NGKHWk<32>
//          gmem/smem load vector lengths <4, 4>, C gmem store vector length 4
//          (host descriptors become 6-D: {G,N,C/x,Hi,Wi,x}; C/K are padded up
//           to a multiple of x by the host tensor descriptor helper)
//
// Vector-length constraints for this kernel (see README_conv_layouts.md):
//   - A/B vector length must divide the innermost physical dimension
//     (C for NHWGC/GKYXC, x for NGCHWc<x>/GKCYXc<x>)
//   - A/B vector length <= MPerBlock * KPerBlock / BlockSize (= 4 for the
//     default 64x64x16 tile with 256 threads)
//   - C vector length must divide the innermost physical dimension of the
//     output (K for NHWGK, x for NGKHWk<x>)
//
// Validation is always done against ReferenceConvFwd on channels-last
// (NHWGC/GKYXC/NHWGK) tensors; for -layout 1 the data is converted
// coordinate-wise between the two physical layouts.

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

using ADataType   = ck_tile::bf16_t;
using BDataType   = ck_tile::bf16_t;
using CDataType   = float;
using AccDataType = float;

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
using WarpTile   = ck_tile::sequence<MmmacIter, NmmacIter, 16, 16, MmmacInterleave, NmmacInterleave>;

using TileShape = ck_tile::ConvIgemmTileShape<BlockTile, BlockWarps, WarpTile>;

using TilePartitioner = ck_tile::ConvIgemmTilePartitioner<TileShape>;

static constexpr auto ConvSpec = ck_tile::ConvFwdSpec::Default;

// bundles the layout types with the vector lengths they require
template <typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          ck_tile::index_t AVec_,
          ck_tile::index_t BVec_,
          ck_tile::index_t CVec_>
struct ConvFwdLayoutConfig
{
    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;

    static constexpr ck_tile::index_t AVectorLength = AVec_;
    static constexpr ck_tile::index_t BVectorLength = BVec_;
    static constexpr ck_tile::index_t CVectorLength = CVec_;
};

// layout 0: channels-last. Innermost dims: A:C, B:C, C:K
using ConvCfgChannelsLast = ConvFwdLayoutConfig<
    ck_tile::tensor_layout::convolution::NHWGC,
    ck_tile::tensor_layout::convolution::GKYXC,
    ck_tile::tensor_layout::convolution::NHWGK,
    4,
    4,
    4>;

// layout 1: C-interleaved blocked with x = 32. Innermost dims are the x block
using ConvCfgInterleaved = ConvFwdLayoutConfig<
    ck_tile::tensor_layout::convolution::NGCHWc<32>,
    ck_tile::tensor_layout::convolution::GKCYXc<32>,
    ck_tile::tensor_layout::convolution::NGKHWk<32>,
    4,
    4,
    4>;

template <typename Layout>
static constexpr bool is_interleaved_in()
{
    return std::is_base_of_v<ck_tile::tensor_layout::convolution::BaseNGCHWc, Layout>;
}

template <typename Layout>
static constexpr bool is_interleaved_wei()
{
    return std::is_base_of_v<ck_tile::tensor_layout::convolution::BaseGKCYXc, Layout>;
}

template <ck_tile::index_t NDimSpatial, typename Cfg>
bool run_grouped_conv_fwd(const ck_tile::conv::ConvParam& conv_param,
                          bool time_kernel,
                          double rtol,
                          double atol)
{
    static constexpr ck_tile::index_t NDim = NDimSpatial + 3;

    using ALayout = typename Cfg::ALayout;
    using BLayout = typename Cfg::BLayout;
    using CLayout = typename Cfg::CLayout;

    // device-side tensors, laid out physically as Cfg requests
    // (interleaved layouts produce 6-D descriptors {G,N,C/x,Hi,Wi,x})
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

    // reference tensors: always channels-last 5-D, independent of Cfg
    const auto ref_in_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
            ck_tile::tensor_layout::convolution::NHWGC>(conv_param);
    const auto ref_wei_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
            ck_tile::tensor_layout::convolution::GKYXC>(conv_param);
    const auto ref_out_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
            ck_tile::tensor_layout::convolution::NHWGK>(conv_param);

    ck_tile::HostTensor<ADataType> ref_in(ref_in_desc);
    ck_tile::HostTensor<BDataType> ref_wei(ref_wei_desc);
    ck_tile::HostTensor<CDataType> ref_out_host(ref_out_desc);
    ck_tile::HostTensor<CDataType> ref_out_device(ref_out_desc);

    std::cout << "layout: " << ALayout::name << " / " << BLayout::name << " / " << CLayout::name
              << std::endl;
    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(ref_in);
    ck_tile::FillUniformDistribution<BDataType>{-0.05f, 0.05f}(ref_wei);

    // move the reference data into the device-layout tensors
    in.SetZero();
    wei.SetZero();
    {
        const auto G  = conv_param.G_;
        const auto N  = conv_param.N_;
        const auto C  = conv_param.C_;
        const auto Hi = conv_param.input_spatial_lengths_[0];
        const auto Wi = conv_param.input_spatial_lengths_[1];
        const auto K  = conv_param.K_;
        const auto Y  = conv_param.filter_spatial_lengths_[0];
        const auto X  = conv_param.filter_spatial_lengths_[1];

        if constexpr(is_interleaved_in<ALayout>())
        {
            constexpr ck_tile::index_t Cx = ALayout::x;
            for(ck_tile::index_t g = 0; g < G; ++g)
                for(ck_tile::index_t n = 0; n < N; ++n)
                    for(ck_tile::index_t c = 0; c < C; ++c)
                        for(ck_tile::index_t hi = 0; hi < Hi; ++hi)
                            for(ck_tile::index_t wi = 0; wi < Wi; ++wi)
                                in(g, n, c / Cx, hi, wi, c % Cx) = ref_in(g, n, c, hi, wi);
        }
        else
        {
            for(ck_tile::index_t g = 0; g < G; ++g)
                for(ck_tile::index_t n = 0; n < N; ++n)
                    for(ck_tile::index_t c = 0; c < C; ++c)
                        for(ck_tile::index_t hi = 0; hi < Hi; ++hi)
                            for(ck_tile::index_t wi = 0; wi < Wi; ++wi)
                                in(g, n, c, hi, wi) = ref_in(g, n, c, hi, wi);
        }

        if constexpr(is_interleaved_wei<BLayout>())
        {
            constexpr ck_tile::index_t Cx = BLayout::x;
            for(ck_tile::index_t g = 0; g < G; ++g)
                for(ck_tile::index_t k = 0; k < K; ++k)
                    for(ck_tile::index_t c = 0; c < C; ++c)
                        for(ck_tile::index_t y = 0; y < Y; ++y)
                            for(ck_tile::index_t x = 0; x < X; ++x)
                                wei(g, k, c / Cx, y, x, c % Cx) = ref_wei(g, k, c, y, x);
        }
        else
        {
            for(ck_tile::index_t g = 0; g < G; ++g)
                for(ck_tile::index_t k = 0; k < K; ++k)
                    for(ck_tile::index_t c = 0; c < C; ++c)
                        for(ck_tile::index_t y = 0; y < Y; ++y)
                            for(ck_tile::index_t x = 0; x < X; ++x)
                                wei(g, k, c, y, x) = ref_wei(g, k, c, y, x);
        }
    }

    ck_tile::DeviceMem in_device_buf(in.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_device_buf(wei.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_device_buf(out_device.get_element_space_size_in_bytes());

    in_device_buf.ToDevice(in.data());
    wei_device_buf.ToDevice(wei.data());
    out_host.SetZero();
    out_device.SetZero();

    // kernel-side lengths/strides keep the g_n_c_wis order; for interleaved
    // layouts only the first NDim entries are passed (the trailing x dim has
    // stride 1 and its extent is recovered from ALayout::x inside the kernel)
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

    auto copy_first = [](const auto& src, auto& dst) { std::copy_n(src.begin(), dst.size(), dst.begin()); };

    copy_first(in_g_n_c_wis_desc.get_lengths(), a_g_n_c_wis_lengths);
    copy_first(in_g_n_c_wis_desc.get_strides(), a_g_n_c_wis_strides);
    copy_first(wei_g_k_c_xs_desc.get_lengths(), b_g_k_c_xs_lengths);
    copy_first(wei_g_k_c_xs_desc.get_strides(), b_g_k_c_xs_strides);
    copy_first(out_g_n_k_wos_desc.get_lengths(), c_g_n_k_wos_lengths);
    copy_first(out_g_n_k_wos_desc.get_strides(), c_g_n_k_wos_strides);
    copy_first(conv_param.conv_filter_strides_, conv_filter_strides);
    copy_first(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy_first(conv_param.input_left_pads_, input_left_pads);
    copy_first(conv_param.input_right_pads_, input_right_pads);

    // gpu conv
    using TileTraits =
        ck_tile::ConvIgemmUniversalTileTraits<ck_tile::tuple<ALayout, BLayout, CLayout>,
                                              ck_tile::tuple<AElementOp, BElementOp, CElementOp>,
                                              ck_tile::sequence<Cfg::AVectorLength, Cfg::BVectorLength>,
                                              ck_tile::sequence<Cfg::AVectorLength, Cfg::BVectorLength>,
                                              ck_tile::sequence<Cfg::AVectorLength, Cfg::BVectorLength>,
                                              Cfg::CVectorLength>;

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

    using Policy = ck_tile::ConvIgemmFwdPipelineDefaultPolicy;

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

    // reference conv fwd on channels-last tensors
    auto ref_conv = ck_tile::ReferenceConvFwd<NDimSpatial,
                                              ADataType,
                                              BDataType,
                                              CDataType,
                                              AElementOp,
                                              BElementOp,
                                              CElementOp>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(ref_in,
                                              ref_wei,
                                              ref_out_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              AElementOp{},
                                              BElementOp{},
                                              CElementOp{});

    ref_invoker.Run(ref_argument);

    out_device_buf.FromDevice(out_device.data());

    // convert device-layout output back to channels-last and compare
    ref_out_device.SetZero();
    {
        const auto G  = conv_param.G_;
        const auto N  = conv_param.N_;
        const auto K  = conv_param.K_;
        const auto Ho = conv_param.output_spatial_lengths_[0];
        const auto Wo = conv_param.output_spatial_lengths_[1];

        if constexpr(is_interleaved_in<CLayout>())
        {
            constexpr ck_tile::index_t Kx = CLayout::x;
            for(ck_tile::index_t g = 0; g < G; ++g)
                for(ck_tile::index_t n = 0; n < N; ++n)
                    for(ck_tile::index_t k = 0; k < K; ++k)
                        for(ck_tile::index_t ho = 0; ho < Ho; ++ho)
                            for(ck_tile::index_t wo = 0; wo < Wo; ++wo)
                                ref_out_device(g, n, k, ho, wo) =
                                    out_device(g, n, k / Kx, ho, wo, k % Kx);
        }
        else
        {
            for(ck_tile::index_t g = 0; g < G; ++g)
                for(ck_tile::index_t n = 0; n < N; ++n)
                    for(ck_tile::index_t k = 0; k < K; ++k)
                        for(ck_tile::index_t ho = 0; ho < Ho; ++ho)
                            for(ck_tile::index_t wo = 0; wo < Wo; ++wo)
                                ref_out_device(g, n, k, ho, wo) = out_device(g, n, k, ho, wo);
        }
    }

    return ck_tile::check_err(ref_out_device, ref_out_host, "Error: incorrect results!", rtol, atol);
}

bool run_grouped_conv_fwd_bf16_f32_example(int argc, char* argv[])
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
        .insert("layout", "0", "input layout: 0=NHWGC/GKYXC/NHWGK, 1=NGCHWc<32>/GKCYXc<32>/NGKHWk<32>")
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
        switch(arg_parser.get_int("layout"))
        {
        case 0:
            return run_grouped_conv_fwd<2, ConvCfgChannelsLast>(
                conv_param,
                arg_parser.get_bool("time"),
                arg_parser.get_double("rtol"),
                arg_parser.get_double("atol"));
        case 1:
            return run_grouped_conv_fwd<2, ConvCfgInterleaved>(
                conv_param,
                arg_parser.get_bool("time"),
                arg_parser.get_double("rtol"),
                arg_parser.get_double("atol"));
        default: printf("unsupport layout\n"); return false;
        }
    }
    else
    {
        printf("unsupport spatial dim\n");
        return false;
    }
}

int main(int argc, char* argv[])
{
    return run_grouped_conv_fwd_bf16_f32_example(argc, argv) ? 0 : 1;
}
