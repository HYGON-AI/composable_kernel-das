// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue/default_2d_multi_d_epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser parser;
    parser.insert("m", "128", "M dimension")
        .insert("n", "128", "N dimension")
        .insert("k", "32", "K dimension")
        .insert("v", "1", "perform CPU validation")
        .insert("time", "1", "time kernel execution")
        .insert("warmup", "0", "warmup iterations")
        .insert("repeat", "1", "timed iterations");
    return std::make_tuple(parser.parse(argc, argv), parser);
}

bool run(const ck_tile::ArgParser& parser)
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using D0DataType  = ck_tile::half_t;
    using D1DataType  = ck_tile::half_t;
    using AccDataType = float;
    using EDataType   = ck_tile::half_t;
    using ALayout     = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout     = ck_tile::tensor_layout::gemm::ColumnMajor;
    using ELayout     = ck_tile::tensor_layout::gemm::RowMajor;

    const ck_tile::index_t M = parser.get_int("m");
    const ck_tile::index_t N = parser.get_int("n");
    const ck_tile::index_t K = parser.get_int("k");
    const bool validate      = parser.get_bool("v");
    const bool time_kernel   = parser.get_bool("time");
    const int warmup         = time_kernel ? parser.get_int("warmup") : 0;
    const int repeat         = time_kernel ? parser.get_int("repeat") : 1;

    if(M <= 0 || N <= 0 || M % 128 != 0 || N % 128 != 0 || K != 32)
    {
        std::cerr << "this acceptance slice requires positive M/N multiples of 128, K=32"
                  << std::endl;
        return false;
    }

    const ck_tile::index_t stride_A = K;
    const ck_tile::index_t stride_B = K;
    const ck_tile::index_t stride_D = N;
    const ck_tile::index_t stride_E = N;
    ck_tile::HostTensor<ADataType> a({M, K}, {stride_A, 1});
    ck_tile::HostTensor<BDataType> b({K, N}, {1, stride_B});
    ck_tile::HostTensor<D0DataType> d0({M, N}, {stride_D, 1});
    ck_tile::HostTensor<D1DataType> d1({M, N}, {stride_D, 1});
    ck_tile::HostTensor<EDataType> e({M, N}, {stride_E, 1});
    ck_tile::HostTensor<EDataType> reference({M, N}, {stride_E, 1});
    ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f, 11939}(a);
    ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f, 11940}(b);
    ck_tile::FillUniformDistribution<D0DataType>{0.5f, 1.0f, 11941}(d0);
    ck_tile::FillUniformDistribution<D1DataType>{0.5f, 1.0f, 11942}(d1);

    ck_tile::DeviceMem a_device(a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_device(b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d0_device(d0.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d1_device(d1.get_element_space_size_in_bytes());
    ck_tile::DeviceMem e_device(e.get_element_space_size_in_bytes());
    a_device.ToDevice(a.data());
    b_device.ToDevice(b.data());
    d0_device.ToDevice(d0.data());
    d1_device.ToDevice(d1.data());
    e_device.SetZero();

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 32>,
                                             ck_tile::sequence<4, 1, 1>,
                                             ck_tile::sequence<32, 64, 32>>;
    using TilePartitioner = ck_tile::GemmTilePartitioner<GemmShape>;
    using Traits =
        ck_tile::MmacTileGemmTraits<false, false, false, ALayout, BLayout, ELayout, 2, 1, 1, 4>;
    using Problem = ck_tile::MmacUniversalGemmPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              GemmShape,
                                                              Traits>;
    using Pipeline = ck_tile::MmacGemmPipelineAGmemBGmemCRegV1<
        Problem,
        ck_tile::MmacUniversalGemmPipelineAgBgCrPolicy>;
    using Epilogue = ck_tile::Default2DMultiDEpilogue<
        ck_tile::Default2DMultiDEpilogueProblem<AccDataType,
                                                D0DataType,
                                                D1DataType,
                                                EDataType>>;
    using Kernel = ck_tile::GemmMultiDKernel<TilePartitioner, Pipeline, Epilogue>;

    const ck_tile::GemmMultiDHostArgs args{a_device.GetDeviceBuffer(),
                                            b_device.GetDeviceBuffer(),
                                            d0_device.GetDeviceBuffer(),
                                            d1_device.GetDeviceBuffer(),
                                            e_device.GetDeviceBuffer(),
                                            M,
                                            N,
                                            K,
                                            stride_A,
                                            stride_B,
                                            stride_D,
                                            stride_D,
                                            stride_E};
    const auto kargs      = Kernel::MakeKargs(args);
    const dim3 grid       = Kernel::GridSize(M, N);
    constexpr dim3 block  = Kernel::BlockSize();
    const float kernel_ms = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, time_kernel, 0, warmup, repeat},
        ck_tile::make_kernel<block.x, 1>(Kernel{}, grid, block, 0, kargs));

    std::cout << "grid:(" << grid.x << "," << grid.y << "," << grid.z << "), block:"
              << block.x << ", " << (time_kernel ? "timing:on" : "timing:off") << ", ";

    bool pass = kernel_ms >= 0;
    if(validate && pass)
    {
        e_device.FromDevice(e.data());
        for(ck_tile::index_t m = 0; m < M; ++m)
        {
            for(ck_tile::index_t n = 0; n < N; ++n)
            {
                float acc = 0;
                for(ck_tile::index_t k = 0; k < K; ++k)
                {
                    acc += ck_tile::type_convert<float>(a(m, k)) *
                           ck_tile::type_convert<float>(b(k, n));
                }
                reference(m, n) = ck_tile::type_convert<EDataType>(
                    acc * ck_tile::type_convert<float>(d0(m, n)) *
                    ck_tile::type_convert<float>(d1(m, n)));
            }
        }
        pass = ck_tile::check_err(e, reference, "gemm multi-d error", 1e-3, 1e-3);
    }

    std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    return pass;
}

int main(int argc, char* argv[])
{
    auto [parsed, parser] = create_args(argc, argv);
    return parsed && run(parser) ? 0 : -1;
}
