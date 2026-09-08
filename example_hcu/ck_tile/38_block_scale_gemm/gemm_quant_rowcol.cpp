// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

namespace {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

struct RunConfig
{
    bool time_kernel = true;
    int warmup       = 5;
    int repeat       = 20;
};

RunConfig run_config;

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout = Row,
          typename BLayout = Col,
          bool PadM        = false,
          bool PadN        = false,
          bool PadK        = false>
bool run_rowcol_quant_gemm(const char* case_name,
                           ck_tile::index_t M = 128,
                           ck_tile::index_t N = 128,
                           ck_tile::index_t K = 128)
{
    using AccDataType = float;
    using CLayout     = Row;

    constexpr ck_tile::index_t MTile = 128;
    constexpr ck_tile::index_t NTile = 128;
    constexpr ck_tile::index_t KTile = 32;
    constexpr ck_tile::index_t MWarp = 4;
    constexpr ck_tile::index_t NWarp = 1;
    constexpr ck_tile::index_t KWarp = 1;
    constexpr ck_tile::index_t MWarpTile = 32;
    constexpr ck_tile::index_t NWarpTile = 64;
    constexpr ck_tile::index_t KWarpTile = 32;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<MTile, NTile, KTile>,
                               ck_tile::sequence<MWarp, NWarp, KWarp>,
                               ck_tile::sequence<MWarpTile, NWarpTile, KWarpTile>>;
    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    using GemmTraits =
        ck_tile::TileGemmQuantTraits<PadM,
                                     PadN,
                                     PadK,
                                     false,
                                     false,
                                     false,
                                     ALayout,
                                     BLayout,
                                     CLayout,
                                     ck_tile::QuantType::RowColQuant,
                                     Row,
                                     Col,
                                     false,
                                     false,
                                     false>;
    using Problem =
        ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
                                                       BDataType,
                                                       AccDataType,
                                                       AccDataType,
                                                       GemmShape,
                                                       GemmTraits>;
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;

    using EpilogueProblem =
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         BDataType,
                                         ck_tile::tuple<>,
                                         AccDataType,
                                         CDataType,
                                         ck_tile::tuple<>,
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         MWarp,
                                         NWarp,
                                         MWarpTile,
                                         NWarpTile,
                                         KWarpTile,
                                         false,
                                         ck_tile::memory_operation_enum::set>;
    using Epilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
    using Kernel =
        ck_tile::QuantGemmKernel<TilePartitioner,
                                 Pipeline,
                                 Epilogue,
                                 ck_tile::QuantType::RowColQuant>;

    const auto a_desc = ck_tile::HostTensorDescriptor(
        std::vector<std::size_t>{std::size_t(M), std::size_t(K)},
        std::is_same_v<ALayout, Row>
            ? std::vector<std::size_t>{std::size_t(K), std::size_t(1)}
            : std::vector<std::size_t>{std::size_t(1), std::size_t(M)});
    const auto b_desc = ck_tile::HostTensorDescriptor(
        std::vector<std::size_t>{std::size_t(K), std::size_t(N)},
        std::is_same_v<BLayout, Row>
            ? std::vector<std::size_t>{std::size_t(N), std::size_t(1)}
            : std::vector<std::size_t>{std::size_t(1), std::size_t(K)});
    const auto c_desc =
        ck_tile::HostTensorDescriptor({std::size_t(M), std::size_t(N)},
                                      {std::size_t(N), std::size_t(1)});
    const auto aq_desc =
        ck_tile::HostTensorDescriptor({std::size_t(M), std::size_t(1)},
                                      {std::size_t(1), std::size_t(1)});
    const auto bq_desc =
        ck_tile::HostTensorDescriptor({std::size_t(1), std::size_t(N)},
                                      {std::size_t(N), std::size_t(1)});

    ck_tile::HostTensor<ADataType> a(a_desc);
    ck_tile::HostTensor<BDataType> b(b_desc);
    ck_tile::HostTensor<float> aq(aq_desc);
    ck_tile::HostTensor<float> bq(bq_desc);
    ck_tile::HostTensor<CDataType> c_device(c_desc);
    ck_tile::HostTensor<CDataType> c_expected(c_desc);
    ck_tile::HostTensor<float> c_reference(c_desc);

    ck_tile::FillUniformDistribution<ADataType>{-1.0f, 1.0f}(a);
    ck_tile::FillUniformDistribution<BDataType>{-1.0f, 1.0f}(b);
    for(ck_tile::index_t m = 0; m < M; ++m)
    {
        aq(m, 0) = 0.25f + 0.01f * static_cast<float>(m % 7);
    }
    for(ck_tile::index_t n = 0; n < N; ++n)
    {
        bq(0, n) = 0.50f + 0.02f * static_cast<float>(n % 5);
    }

    c_device.SetZero();
    c_expected.SetZero();
    c_reference.SetZero();
    ck_tile::reference_gemm<ADataType, BDataType, AccDataType, float>(a, b, c_reference);
    for(ck_tile::index_t m = 0; m < M; ++m)
    {
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            c_expected(m, n) = ck_tile::type_convert<CDataType>(
                c_reference(m, n) * aq(m, 0) * bq(0, n));
        }
    }

    ck_tile::DeviceMem a_device(a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_device(b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem aq_device(aq.get_element_space_size_in_bytes());
    ck_tile::DeviceMem bq_device(bq.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_device_buffer(c_device.get_element_space_size_in_bytes());

    a_device.ToDevice(a.data());
    b_device.ToDevice(b.data());
    aq_device.ToDevice(aq.data());
    bq_device.ToDevice(bq.data());
    c_device_buffer.SetZero();

    const ck_tile::QuantGemmHostArgs host_args{
        a_device.GetDeviceBuffer(),
        b_device.GetDeviceBuffer(),
        c_device_buffer.GetDeviceBuffer(),
        aq_device.GetDeviceBuffer(),
        bq_device.GetDeviceBuffer(),
        1,
        M,
        N,
        K,
        1,
        1,
        std::is_same_v<ALayout, Row> ? K : M,
        std::is_same_v<BLayout, Row> ? N : K,
        N,
        1,
        1};
    const auto kernel_args = Kernel::MakeKernelArgs(host_args);

    if(!Kernel::IsSupportedArgument(kernel_args))
    {
        std::cerr << case_name << " arguments are not supported" << std::endl;
        return false;
    }

    const auto elapsed_ms = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr,
                               run_config.time_kernel,
                               1,
                               run_config.time_kernel ? run_config.warmup : 0,
                               run_config.time_kernel ? run_config.repeat : 1},
        ck_tile::make_kernel<1>(
            Kernel{}, Kernel::GridSize(M, N, 1), Kernel::BlockSize(), 0, kernel_args));

    c_device_buffer.FromDevice(c_device.data());
    const bool pass =
        ck_tile::check_err(c_device, c_expected, "row-col quant GEMM mismatch", 1e-2, 1e-2);
    std::cout << case_name << " " << (pass ? "PASS" : "FAIL");
    if(run_config.time_kernel)
        std::cout << ", " << elapsed_ms << " ms";
    else
        std::cout << ", timing:off";
    std::cout << std::endl;
    return pass;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("time", "1", "time kernel (0=no, 1=yes)")
        .insert("warmup", "5", "number of warmup iterations")
        .insert("repeat", "20", "number of timed iterations")
        .insert("bf16_output", "1", "run BF16 output cases (0=no, 1=yes)");
    if(!arg_parser.parse(argc, argv))
        return 1;
    run_config = {arg_parser.get_bool("time"),
                  arg_parser.get_int("warmup"),
                  arg_parser.get_int("repeat")};
    const bool run_bf16_output = arg_parser.get_bool("bf16_output");

    bool pass = true;
    pass = run_rowcol_quant_gemm<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>(
               "rowcol FP8xFP8->FP16") &&
           pass;
    if(run_bf16_output)
        pass = run_rowcol_quant_gemm<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bf16_t>(
                   "rowcol FP8xFP8->BF16") &&
               pass;
    pass = run_rowcol_quant_gemm<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(
               "rowcol BF8xBF8->FP16") &&
           pass;
    if(run_bf16_output)
        pass = run_rowcol_quant_gemm<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bf16_t>(
                   "rowcol BF8xBF8->BF16") &&
               pass;
    pass = run_rowcol_quant_gemm<ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t>(
               "rowcol FP8xBF8->FP16") &&
           pass;
    if(run_bf16_output)
        pass = run_rowcol_quant_gemm<ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::bf16_t>(
                   "rowcol FP8xBF8->BF16") &&
               pass;
    pass = run_rowcol_quant_gemm<ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t>(
               "rowcol BF8xFP8->FP16") &&
           pass;
    if(run_bf16_output)
        pass = run_rowcol_quant_gemm<ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::bf16_t>(
                   "rowcol BF8xFP8->BF16") &&
               pass;
    if(!run_bf16_output)
        std::cout << "rowcol BF16 output SKIP-PMD" << std::endl;
    pass = run_rowcol_quant_gemm<ck_tile::fp8_t,
                                 ck_tile::fp8_t,
                                 ck_tile::half_t,
                                 Row,
                                 Row>("rowcol FP8xFP8->FP16 RRR") &&
           pass;
    pass = run_rowcol_quant_gemm<ck_tile::fp8_t,
                                 ck_tile::fp8_t,
                                 ck_tile::half_t,
                                 Col,
                                 Row>("rowcol FP8xFP8->FP16 CRR") &&
           pass;
    pass = run_rowcol_quant_gemm<ck_tile::fp8_t,
                                 ck_tile::fp8_t,
                                 ck_tile::half_t,
                                 Row,
                                 Col,
                                 true,
                                 true,
                                 true>(
               "rowcol FP8xFP8->FP16 RCR padded 160x192x80", 160, 192, 80) &&
           pass;
    return pass ? 0 : 1;
}
