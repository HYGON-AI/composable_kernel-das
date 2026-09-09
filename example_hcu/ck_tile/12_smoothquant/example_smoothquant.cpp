// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/smoothquant.hpp"

template <typename DataType>
auto get_elimit()
{
    return ck_tile::make_tuple(1e-5, 1e-5);
}

template <>
auto get_elimit<ck_tile::int8_t>()
{
    return ck_tile::make_tuple(1.0, 1.0);
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("stride", "-1", "row stride; -1 means n")
        .insert("v", "1", "do CPU validation")
        .insert("prec", "fp16", "precision: fp16 or bf16")
        .insert("time", "1", "time kernel (0=no, 1=yes)")
        .insert("warmup", "0", "warmup iterations")
        .insert("repeat", "1", "timed iterations");

    const bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    const ck_tile::index_t m = arg_parser.get_int("m");
    const ck_tile::index_t n = arg_parser.get_int("n");
    ck_tile::index_t stride   = arg_parser.get_int("stride");
    if(stride < 0)
        stride = n;

    const std::string data_type = arg_parser.get_str("prec");
    const int do_validation     = arg_parser.get_int("v");
    const bool time_kernel      = arg_parser.get_bool("time");
    const int warmup            = time_kernel ? arg_parser.get_int("warmup") : 0;
    const int repeat            = time_kernel ? arg_parser.get_int("repeat") : 1;

    if(stride < n)
    {
        std::cerr << "stride must be at least n" << std::endl;
        return false;
    }

    using XDataType       = DataType;
    using XScaleDataType  = float;
    using YScaleDataType  = float;
    using QYDataType      = ck_tile::int8_t;
    using ComputeDataType = float;

    ck_tile::HostTensor<XDataType> x_host({m, n}, {stride, 1});
    ck_tile::HostTensor<XScaleDataType> xscale_host({n});
    ck_tile::HostTensor<YScaleDataType> yscale_host_ref({m}, {1});
    ck_tile::HostTensor<YScaleDataType> yscale_host_dev({m}, {1});
    ck_tile::HostTensor<QYDataType> qy_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<QYDataType> qy_host_dev({m, n}, {stride, 1});

    ck_tile::FillUniformDistribution<XDataType>{-0.5f, 0.5f}(x_host);
    ck_tile::FillUniformDistribution<XScaleDataType>{1e-3f, 0.5f}(xscale_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem xscale_buf(xscale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem yscale_buf(yscale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem qy_buf(qy_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    xscale_buf.ToDevice(xscale_host.data());

    constexpr bool kTwoPass = true;

    using BlockWarps = ck_tile::sequence<2, 2>;
    using BlockTile  = ck_tile::sequence<2, 128>;
    using WarpTile   = ck_tile::sequence<1, 64>;
    using Vector     = ck_tile::sequence<1, 1>;

    using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;
    using Problem = ck_tile::SmoothquantPipelineProblem<XDataType,
                                                        XScaleDataType,
                                                        ComputeDataType,
                                                        YScaleDataType,
                                                        QYDataType,
                                                        Shape,
                                                        true,
                                                        kTwoPass>;

    using OnePassPipeline = ck_tile::SmoothquantPipelineOnePass<Problem>;
    using TwoPassPipeline = ck_tile::SmoothquantPipelineTwoPass<Problem>;
    using Pipeline        = std::conditional_t<kTwoPass, TwoPassPipeline, OnePassPipeline>;
    using Kernel          = ck_tile::Smoothquant<Pipeline>;

    const ck_tile::SmoothquantHostArgs args{x_buf.GetDeviceBuffer(),
                                             xscale_buf.GetDeviceBuffer(),
                                             yscale_buf.GetDeviceBuffer(),
                                             qy_buf.GetDeviceBuffer(),
                                             m,
                                             n,
                                             stride};

    const auto kargs                         = Kernel::MakeKargs(args);
    const dim3 grids                         = Kernel::GridSize(args);
    constexpr dim3 blocks                    = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu   = 1;
    const ck_tile::stream_config stream_conf =
        {nullptr, time_kernel, 0, warmup, repeat};

    const float ave_time = ck_tile::launch_kernel(
        stream_conf,
        ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    if(time_kernel)
        std::cout << "time:" << ave_time << "ms, ";
    else
        std::cout << "timing:off, ";

    bool pass = true;

    if(do_validation)
    {
        using YDataType = ComputeDataType;
        ck_tile::HostTensor<YDataType> y_host({m, n}, {stride, 1});

        auto smooth_column = [&](auto n_) {
            const auto scale = ck_tile::type_convert<ComputeDataType>(xscale_host(n_));
            for(int m_ = 0; m_ < m; ++m_)
            {
                y_host(m_, n_) =
                    ck_tile::type_convert<ComputeDataType>(x_host(m_, n_)) * scale;
            }
        };
        ck_tile::make_ParallelTensorFunctor(smooth_column, xscale_host.get_element_space_size())(
            std::thread::hardware_concurrency());

        ck_tile::HostTensor<YDataType> y_rowwise_amax_host({m});
        using ReduceAmax = ck_tile::ReduceOp::AbsMax;
        ck_tile::reference_reduce<ComputeDataType, ComputeDataType, YDataType>(
            y_host, y_rowwise_amax_host, ReduceAmax{});

        auto make_scale = [](const auto& value) {
            return value /
                   ck_tile::type_convert<ComputeDataType>(ck_tile::numeric<QYDataType>::max());
        };
        ck_tile::reference_unary_elementwise<YDataType, YScaleDataType, ComputeDataType>(
            y_rowwise_amax_host, yscale_host_ref, make_scale);

        yscale_buf.FromDevice(yscale_host_dev.mData.data());
        auto [scale_rtol, scale_atol] = get_elimit<YScaleDataType>();
        pass &= ck_tile::check_err(yscale_host_dev,
                                   yscale_host_ref,
                                   "yscale Error: Incorrect results!",
                                   scale_rtol,
                                   scale_atol);

        ck_tile::reference_rowwise_quantization2d<YDataType, YScaleDataType, QYDataType>(
            y_host, yscale_host_ref, qy_host_ref);
        qy_buf.FromDevice(qy_host_dev.data());
        auto [qy_rtol, qy_atol] = get_elimit<QYDataType>();
        pass &= ck_tile::check_err(
            qy_host_dev, qy_host_ref, "qy Error: Incorrect results!", qy_rtol, qy_atol);

        std::cout << "[" << data_type << "] m:" << m << ", n:" << n
                  << ", stride:" << stride << ", valid:" << (pass ? "y" : "n")
                  << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    if(arg_parser.get_str("prec") == "fp16")
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;

    if(arg_parser.get_str("prec") == "bf16")
        return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;

    std::cerr << "unsupported precision" << std::endl;
    return -3;
}
