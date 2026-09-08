// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/smoothquant.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser parser;
    parser.insert("tokens", "2", "token count")
        .insert("hidden", "128", "hidden dimension")
        .insert("experts", "4", "expert count")
        .insert("topk", "2", "experts selected per token")
        .insert("v", "1", "perform CPU validation")
        .insert("time", "1", "time kernel execution")
        .insert("warmup", "0", "warmup iterations")
        .insert("repeat", "1", "timed iterations");
    return std::make_tuple(parser.parse(argc, argv), parser);
}

bool run(const ck_tile::ArgParser& parser)
{
    using XDataType           = ck_tile::half_t;
    using SmoothScaleDataType = float;
    using ComputeDataType     = float;
    using YScaleDataType      = float;
    using QYDataType          = ck_tile::int8_t;

    const ck_tile::index_t tokens  = parser.get_int("tokens");
    const ck_tile::index_t hidden  = parser.get_int("hidden");
    const ck_tile::index_t experts = parser.get_int("experts");
    const ck_tile::index_t topk    = parser.get_int("topk");
    const bool validate            = parser.get_bool("v");
    const bool time_kernel         = parser.get_bool("time");
    const int warmup               = time_kernel ? parser.get_int("warmup") : 0;
    const int repeat               = time_kernel ? parser.get_int("repeat") : 1;

    if(tokens != 2 || hidden != 128 || experts != 4 || topk != 2)
    {
        std::cerr << "this acceptance slice requires tokens=2, hidden=128, experts=4, topk=2"
                  << std::endl;
        return false;
    }

    ck_tile::HostTensor<XDataType> x({tokens, hidden});
    ck_tile::HostTensor<SmoothScaleDataType> smooth_scale({experts, hidden});
    ck_tile::HostTensor<ck_tile::index_t> topk_ids({tokens, topk});
    ck_tile::HostTensor<YScaleDataType> yscale({topk, tokens});
    ck_tile::HostTensor<YScaleDataType> yscale_ref({topk, tokens});
    ck_tile::HostTensor<QYDataType> qy({topk, tokens, hidden});
    ck_tile::HostTensor<QYDataType> qy_ref({topk, tokens, hidden});

    ck_tile::FillUniformDistribution<XDataType>{-0.5f, 0.5f, 11939}(x);
    ck_tile::FillUniformDistribution<SmoothScaleDataType>{1e-3f, 0.5f, 11940}(smooth_scale);
    topk_ids(0, 0) = 0;
    topk_ids(0, 1) = 1;
    topk_ids(1, 0) = 2;
    topk_ids(1, 1) = 3;

    ck_tile::DeviceMem x_device(x.get_element_space_size_in_bytes());
    ck_tile::DeviceMem smooth_scale_device(smooth_scale.get_element_space_size_in_bytes());
    ck_tile::DeviceMem topk_ids_device(topk_ids.get_element_space_size_in_bytes());
    ck_tile::DeviceMem yscale_device(yscale.get_element_space_size_in_bytes());
    ck_tile::DeviceMem qy_device(qy.get_element_space_size_in_bytes());
    x_device.ToDevice(x.data());
    smooth_scale_device.ToDevice(smooth_scale.data());
    topk_ids_device.ToDevice(topk_ids.data());

    using BlockWarps = ck_tile::sequence<2, 2>;
    using BlockTile  = ck_tile::sequence<2, 128>;
    using WarpTile   = ck_tile::sequence<1, 64>;
    using Vector     = ck_tile::sequence<1, 1>;
    using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;
    using Problem = ck_tile::SmoothquantPipelineProblem<XDataType,
                                                        SmoothScaleDataType,
                                                        ComputeDataType,
                                                        YScaleDataType,
                                                        QYDataType,
                                                        Shape,
                                                        false,
                                                        true>;
    using Pipeline = ck_tile::SmoothquantPipelineTwoPass<Problem>;
    using Kernel   = ck_tile::MoeSmoothquant<Pipeline>;

    const ck_tile::MoeSmoothquantHostArgs args{x_device.GetDeviceBuffer(),
                                                smooth_scale_device.GetDeviceBuffer(),
                                                topk_ids_device.GetDeviceBuffer(),
                                                yscale_device.GetDeviceBuffer(),
                                                qy_device.GetDeviceBuffer(),
                                                tokens,
                                                hidden,
                                                experts,
                                                topk,
                                                hidden,
                                                hidden};
    const auto kargs       = Kernel::MakeKargs(args);
    const dim3 grid        = Kernel::GridSize(args);
    constexpr dim3 block   = Kernel::BlockSizeValue();
    const float kernel_ms  = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, time_kernel, 0, warmup, repeat},
        ck_tile::make_kernel<block.x, 1>(Kernel{}, grid, block, 0, kargs));

    std::cout << "grid:(" << grid.x << "," << grid.y << "," << grid.z << "), block:"
              << block.x << ", " << (time_kernel ? "timing:on" : "timing:off") << ", ";

    bool pass = kernel_ms >= 0;
    if(validate && pass)
    {
        const float quant_max =
            ck_tile::type_convert<float>(ck_tile::numeric<QYDataType>::max());
        for(ck_tile::index_t k = 0; k < topk; ++k)
        {
            for(ck_tile::index_t t = 0; t < tokens; ++t)
            {
                const ck_tile::index_t expert = topk_ids(t, k);
                float amax                     = 0;
                for(ck_tile::index_t n = 0; n < hidden; ++n)
                {
                    const float value = ck_tile::type_convert<float>(x(t, n)) *
                                        ck_tile::type_convert<float>(smooth_scale(expert, n));
                    amax = std::max(amax, std::abs(value));
                }
                yscale_ref(k, t) = amax / quant_max;
                for(ck_tile::index_t n = 0; n < hidden; ++n)
                {
                    const float value = ck_tile::type_convert<float>(x(t, n)) *
                                        ck_tile::type_convert<float>(smooth_scale(expert, n));
                    qy_ref(k, t, n) = ck_tile::type_convert<QYDataType>(
                        std::nearbyint(value / yscale_ref(k, t)));
                }
            }
        }

        yscale_device.FromDevice(yscale.data());
        qy_device.FromDevice(qy.data());
        pass &= ck_tile::check_err(yscale, yscale_ref, "yscale error", 1e-5, 1e-5);
        pass &= ck_tile::check_err(qy, qy_ref, "qy error", 1, 1);
    }

    std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    return pass;
}

int main(int argc, char* argv[])
{
    auto [parsed, parser] = create_args(argc, argv);
    return parsed && run(parser) ? 0 : -1;
}
