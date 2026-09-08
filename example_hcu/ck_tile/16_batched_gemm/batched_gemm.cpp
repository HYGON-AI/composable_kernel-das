// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser parser;
    parser.insert("batch", "2", "batch count")
        .insert("m", "128", "M dimension")
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
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    using ALayout     = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout     = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout     = ck_tile::tensor_layout::gemm::RowMajor;

    const ck_tile::index_t batch = parser.get_int("batch");
    const ck_tile::index_t M     = parser.get_int("m");
    const ck_tile::index_t N     = parser.get_int("n");
    const ck_tile::index_t K     = parser.get_int("k");
    const bool validate          = parser.get_bool("v");
    const bool time_kernel       = parser.get_bool("time");
    const int warmup             = time_kernel ? parser.get_int("warmup") : 0;
    const int repeat             = time_kernel ? parser.get_int("repeat") : 1;

    if(batch <= 0 || M <= 0 || N <= 0 || M % 128 != 0 || N % 128 != 0 || K != 32)
    {
        std::cerr << "this acceptance slice requires positive batch/M/N, M/N multiples of 128, "
                     "K=32"
                  << std::endl;
        return false;
    }

    const ck_tile::index_t stride_A = K;
    const ck_tile::index_t stride_B = K;
    const ck_tile::index_t stride_C = N;
    const ck_tile::index_t batch_stride_A = M * K;
    const ck_tile::index_t batch_stride_B = K * N;
    const ck_tile::index_t batch_stride_C = M * N;

    ck_tile::HostTensor<ADataType> a({batch, M, K}, {batch_stride_A, stride_A, 1});
    ck_tile::HostTensor<BDataType> b({batch, K, N}, {batch_stride_B, 1, stride_B});
    ck_tile::HostTensor<CDataType> c({batch, M, N}, {batch_stride_C, stride_C, 1});
    ck_tile::HostTensor<CDataType> reference(
        {batch, M, N}, {batch_stride_C, stride_C, 1});
    ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f, 11939}(a);
    ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f, 11940}(b);

    ck_tile::DeviceMem a_device(a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_device(b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_device(c.get_element_space_size_in_bytes());
    a_device.ToDevice(a.data());
    b_device.ToDevice(b.data());
    c_device.SetZero();

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 32>,
                                             ck_tile::sequence<4, 1, 1>,
                                             ck_tile::sequence<32, 64, 32>>;
    using TilePartitioner = ck_tile::GemmTilePartitioner<GemmShape>;
    using Epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, false, false>>;
    using Traits =
        ck_tile::MmacTileGemmTraits<false, false, false, ALayout, BLayout, CLayout, 2, 1, 1, 4>;
    using Problem = ck_tile::MmacUniversalGemmPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              GemmShape,
                                                              Traits>;
    using Pipeline = ck_tile::MmacGemmPipelineAGmemBGmemCRegV1<
        Problem,
        ck_tile::MmacUniversalGemmPipelineAgBgCrPolicy>;
    using Kernel = ck_tile::BatchedGemmKernel<TilePartitioner, Pipeline, Epilogue>;

    const ck_tile::BatchedGemmHostArgs args{a_device.GetDeviceBuffer(),
                                             b_device.GetDeviceBuffer(),
                                             c_device.GetDeviceBuffer(),
                                             M,
                                             N,
                                             K,
                                             stride_A,
                                             stride_B,
                                             stride_C,
                                             batch_stride_A,
                                             batch_stride_B,
                                             batch_stride_C,
                                             batch};
    const auto kargs      = Kernel::MakeKargs(args);
    const dim3 grid       = Kernel::GridSize(M, N, batch);
    constexpr dim3 block  = Kernel::BlockSize();
    const float kernel_ms = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, time_kernel, 0, warmup, repeat},
        ck_tile::make_kernel<block.x, 1>(Kernel{}, grid, block, 0, kargs));

    std::cout << "grid:(" << grid.x << "," << grid.y << "," << grid.z << "), block:"
              << block.x << ", " << (time_kernel ? "timing:on" : "timing:off") << ", ";

    bool pass = kernel_ms >= 0;
    if(validate && pass)
    {
        c_device.FromDevice(c.data());
        for(ck_tile::index_t g = 0; g < batch; ++g)
        {
            for(ck_tile::index_t m = 0; m < M; ++m)
            {
                for(ck_tile::index_t n = 0; n < N; ++n)
                {
                    float acc = 0;
                    for(ck_tile::index_t k = 0; k < K; ++k)
                    {
                        acc += ck_tile::type_convert<float>(a(g, m, k)) *
                               ck_tile::type_convert<float>(b(g, k, n));
                    }
                    reference(g, m, n) = ck_tile::type_convert<CDataType>(acc);
                }
            }
        }
        pass = ck_tile::check_err(c, reference, "batched gemm error", 1e-3, 1e-3);
    }

    std::cout << "valid:" << (pass ? "y" : "n") << std::endl;
    return pass;
}

int main(int argc, char* argv[])
{
    auto [parsed, parser] = create_args(argc, argv);
    return parsed && run(parser) ? 0 : -1;
}
