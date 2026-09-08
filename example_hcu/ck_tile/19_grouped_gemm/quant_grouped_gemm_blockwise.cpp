// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <memory>
#include <string>
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

struct QuantRunConfig
{
    bool time_kernel = true;
    int warmup       = 5;
    int repeat       = 20;
};

QuantRunConfig run_config{};

template <typename Layout>
ck_tile::HostTensorDescriptor
make_matrix_descriptor(ck_tile::index_t rows, ck_tile::index_t columns)
{
    if constexpr(std::is_same_v<Layout, Row>)
    {
        return ck_tile::HostTensorDescriptor(
            {std::size_t(rows), std::size_t(columns)},
            {std::size_t(columns), std::size_t(1)});
    }
    else
    {
        return ck_tile::HostTensorDescriptor(
            {std::size_t(rows), std::size_t(columns)},
            {std::size_t(1), std::size_t(rows)});
    }
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout>
struct BlockwiseGroup
{
    static constexpr ck_tile::index_t QuantGroupK = 128;
    static constexpr bool DenseBQuant =
        std::is_same_v<ALayout, Col> && std::is_same_v<BLayout, Row>;
    static constexpr ck_tile::index_t QuantGroupN = DenseBQuant ? 1 : 128;

    BlockwiseGroup(ck_tile::index_t m,
                   ck_tile::index_t n,
                   ck_tile::index_t k,
                   std::size_t group_id)
        : M(m),
          N(n),
          K(k),
          AQK(ck_tile::integer_divide_ceil(K, QuantGroupK)),
          BQK(ck_tile::integer_divide_ceil(K, QuantGroupK)),
          BQN(ck_tile::integer_divide_ceil(N, QuantGroupN)),
          a(make_matrix_descriptor<ALayout>(M, K)),
          b(make_matrix_descriptor<BLayout>(K, N)),
          aq(make_matrix_descriptor<Row>(M, AQK)),
          bq(make_matrix_descriptor<Col>(BQK, BQN)),
          c_device(make_matrix_descriptor<Row>(M, N)),
          c_expected(make_matrix_descriptor<Row>(M, N)),
          a_device(a.get_element_space_size_in_bytes()),
          b_device(b.get_element_space_size_in_bytes()),
          aq_device(aq.get_element_space_size_in_bytes()),
          bq_device(bq.get_element_space_size_in_bytes()),
          c_device_buffer(c_device.get_element_space_size_in_bytes())
    {
        ck_tile::FillUniformDistribution<ADataType>{-1.0f, 1.0f}(a);
        ck_tile::FillUniformDistribution<BDataType>{-1.0f, 1.0f}(b);

        for(ck_tile::index_t row = 0; row < M; ++row)
        {
            for(ck_tile::index_t qk = 0; qk < AQK; ++qk)
            {
                aq(row, qk) = 0.25f + 0.01f * static_cast<float>((row + group_id) % 7) +
                              0.05f * static_cast<float>(qk);
            }
        }
        for(ck_tile::index_t qk = 0; qk < BQK; ++qk)
        {
            for(ck_tile::index_t qn = 0; qn < BQN; ++qn)
            {
                bq(qk, qn) = 0.50f + 0.07f * static_cast<float>(qk) +
                             0.03f * static_cast<float>((qn + group_id) % 5);
            }
        }

        c_device.SetZero();
        c_expected.SetZero();
        for(ck_tile::index_t row = 0; row < M; ++row)
        {
            for(ck_tile::index_t column = 0; column < N; ++column)
            {
                float acc = 0;
                for(ck_tile::index_t k_index = 0; k_index < K; ++k_index)
                {
                    const auto qk = k_index / QuantGroupK;
                    const auto qn = column / QuantGroupN;
                    acc += ck_tile::type_convert<float>(a(row, k_index)) *
                           ck_tile::type_convert<float>(b(k_index, column)) * aq(row, qk) *
                           bq(qk, qn);
                }
                c_expected(row, column) = ck_tile::type_convert<CDataType>(acc);
            }
        }

        a_device.ToDevice(a.data());
        b_device.ToDevice(b.data());
        aq_device.ToDevice(aq.data());
        bq_device.ToDevice(bq.data());
        c_device_buffer.SetZero();
    }

    ck_tile::QuantGroupedGemmHostArgs MakeHostArgs() const
    {
        return {a_device.GetDeviceBuffer(),
                b_device.GetDeviceBuffer(),
                c_device_buffer.GetDeviceBuffer(),
                aq_device.GetDeviceBuffer(),
                bq_device.GetDeviceBuffer(),
                1,
                M,
                N,
                K,
                AQK,
                BQK,
                std::is_same_v<ALayout, Row> ? K : M,
                std::is_same_v<BLayout, Row> ? N : K,
                N,
                AQK,
                BQK};
    }

    bool Check(std::size_t group_id)
    {
        c_device_buffer.FromDevice(c_device.data());
        return ck_tile::check_err(c_device,
                                  c_expected,
                                  "blockwise quant grouped GEMM group " +
                                      std::to_string(group_id) + " mismatch",
                                  1e-2,
                                  1e-2);
    }

    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::index_t AQK;
    ck_tile::index_t BQK;
    ck_tile::index_t BQN;
    ck_tile::HostTensor<ADataType> a;
    ck_tile::HostTensor<BDataType> b;
    ck_tile::HostTensor<float> aq;
    ck_tile::HostTensor<float> bq;
    ck_tile::HostTensor<CDataType> c_device;
    ck_tile::HostTensor<CDataType> c_expected;
    ck_tile::DeviceMem a_device;
    ck_tile::DeviceMem b_device;
    ck_tile::DeviceMem aq_device;
    ck_tile::DeviceMem bq_device;
    ck_tile::DeviceMem c_device_buffer;
};

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout = Row,
          typename BLayout = Col,
          bool PadM        = true,
          bool PadN        = true,
          bool PadK        = true>
bool run_blockwise_grouped_case(const char* case_name,
                                const std::vector<ck_tile::index_t>& ms,
                                const std::vector<ck_tile::index_t>& ns,
                                const std::vector<ck_tile::index_t>& ks)
{
    if(ms.empty() || ms.size() != ns.size() || ms.size() != ks.size())
    {
        std::cerr << case_name << " invalid group shape list" << std::endl;
        return false;
    }

    using AccDataType = float;
    using CLayout     = Row;
    using AQLayout    = Row;
    using BQLayout    = Col;

    constexpr ck_tile::index_t MTile = 128;
    constexpr ck_tile::index_t NTile = 128;
    constexpr ck_tile::index_t KTile = 128;
    constexpr ck_tile::index_t MWarp = 4;
    constexpr ck_tile::index_t NWarp = 1;
    constexpr ck_tile::index_t KWarp = 1;
    constexpr ck_tile::index_t MWarpTile = 32;
    constexpr ck_tile::index_t NWarpTile = 64;
    constexpr ck_tile::index_t KWarpTile = 32;
    constexpr bool DenseBQuant =
        std::is_same_v<ALayout, Col> && std::is_same_v<BLayout, Row>;
    constexpr ck_tile::index_t QuantGroupN = DenseBQuant ? 1 : 128;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<MTile, NTile, KTile>,
                               ck_tile::sequence<MWarp, NWarp, KWarp>,
                               ck_tile::sequence<MWarpTile, NWarpTile, KWarpTile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;
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
                                     ck_tile::QuantType::ABQuantGrouped,
                                     AQLayout,
                                     BQLayout,
                                     false,
                                     false,
                                     false>;
    using AQuantGroupSize =
        ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
    using BQuantGroupSize =
        ck_tile::QuantGroupShape<ck_tile::sequence<1, QuantGroupN, 128>>;
    using Problem =
        ck_tile::GemmABQuantPipelineProblem<ADataType,
                                             AccDataType,
                                             BDataType,
                                             AccDataType,
                                             AccDataType,
                                             GemmShape,
                                             GemmTraits,
                                             AQuantGroupSize,
                                             BQuantGroupSize,
                                             false>;
    using Pipeline = ck_tile::ABQuantGemmPipelineAgBgCrCompV3<Problem>;
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
        ck_tile::QuantGroupedGemmKernel<TilePartitioner,
                                        Pipeline,
                                        Epilogue,
                                        ck_tile::QuantType::ABQuantGrouped>;
    using Group = BlockwiseGroup<ADataType, BDataType, CDataType, ALayout, BLayout>;

    std::vector<std::unique_ptr<Group>> groups;
    std::vector<ck_tile::QuantGroupedGemmHostArgs> host_args;
    groups.reserve(ms.size());
    host_args.reserve(ms.size());

    for(std::size_t i = 0; i < ms.size(); ++i)
    {
        groups.emplace_back(std::make_unique<Group>(ms[i], ns[i], ks[i], i));
        host_args.emplace_back(groups.back()->MakeHostArgs());
    }

    const auto kernel_args = Kernel::MakeKargs(host_args);
    if(kernel_args.size() != groups.size() || !Kernel::IsSupportedArgument(kernel_args))
    {
        std::cerr << case_name << " arguments are not supported" << std::endl;
        return false;
    }

    ck_tile::DeviceMem kernel_args_device(Kernel::GetWorkSpaceSize(host_args));
    kernel_args_device.ToDevice(kernel_args.data());

    const auto elapsed_ms = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr,
                               run_config.time_kernel,
                               1,
                               run_config.time_kernel ? run_config.warmup : 0,
                               run_config.time_kernel ? run_config.repeat : 1},
        ck_tile::make_kernel<Kernel::kBlockSize, 1>(
            Kernel{},
            Kernel::GridSize(host_args),
            Kernel::BlockSize(),
            0,
            ck_tile::cast_pointer_to_constant_address_space(
                kernel_args_device.GetDeviceBuffer()),
            ck_tile::type_convert<ck_tile::index_t>(kernel_args.size())));

    bool pass = true;
    for(std::size_t i = 0; i < groups.size(); ++i)
    {
        pass = groups[i]->Check(i) && pass;
    }

    std::cout << case_name << " " << (pass ? "PASS" : "FAIL");
    if(run_config.time_kernel)
        std::cout << ", " << elapsed_ms << " ms";
    else
        std::cout << ", timing:off";
    std::cout << std::endl;
    return pass;
}

template <typename ALayout, typename BLayout>
bool run_blockwise_dtype_matrix(const char* layout_name,
                                const std::vector<ck_tile::index_t>& ms,
                                const std::vector<ck_tile::index_t>& ns,
                                const std::vector<ck_tile::index_t>& ks,
                                bool run_bf16_output)
{
    const std::string prefix = "blockwise grouped ";
    const std::string suffix = std::string{" "} + layout_name;
    bool pass                = true;

#define RUN_BLOCKWISE_CASE(AType, BType, CType, TypeName)                                  \
    pass = run_blockwise_grouped_case<AType, BType, CType, ALayout, BLayout>(              \
               (prefix + TypeName + suffix).c_str(), ms, ns, ks) &&                        \
           pass

    RUN_BLOCKWISE_CASE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, "FP8xFP8->FP16");
    if(run_bf16_output)
        RUN_BLOCKWISE_CASE(
            ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bf16_t, "FP8xFP8->BF16");
    RUN_BLOCKWISE_CASE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, "BF8xBF8->FP16");
    if(run_bf16_output)
        RUN_BLOCKWISE_CASE(
            ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bf16_t, "BF8xBF8->BF16");
    RUN_BLOCKWISE_CASE(ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, "FP8xBF8->FP16");
    if(run_bf16_output)
        RUN_BLOCKWISE_CASE(
            ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::bf16_t, "FP8xBF8->BF16");
    RUN_BLOCKWISE_CASE(ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, "BF8xFP8->FP16");
    if(run_bf16_output)
        RUN_BLOCKWISE_CASE(
            ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::bf16_t, "BF8xFP8->BF16");
    if(!run_bf16_output)
        std::cout << "blockwise grouped BF16 output " << layout_name << " SKIP-PMD"
                  << std::endl;

#undef RUN_BLOCKWISE_CASE
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

    const std::vector<ck_tile::index_t> fixed_ms{128, 96, 160};
    const std::vector<ck_tile::index_t> fixed_ns{128, 128, 128};
    const std::vector<ck_tile::index_t> fixed_ks{256, 256, 256};
    const std::vector<ck_tile::index_t> padded_ms{127, 160, 65};
    const std::vector<ck_tile::index_t> padded_ns{192, 120, 136};
    const std::vector<ck_tile::index_t> padded_ks{160, 160, 160};
    const std::vector<ck_tile::index_t> variable_ms{128, 128, 128};
    const std::vector<ck_tile::index_t> variable_ns{128, 128, 128};
    const std::vector<ck_tile::index_t> variable_ks{128, 256, 384};

    bool pass = run_blockwise_dtype_matrix<Row, Col>(
        "NT fixed-K", fixed_ms, fixed_ns, fixed_ks, run_bf16_output);
    pass = run_blockwise_dtype_matrix<Row, Row>(
               "NN fixed-K", fixed_ms, fixed_ns, fixed_ks, run_bf16_output) &&
           pass;
    pass = run_blockwise_dtype_matrix<Col, Row>(
               "TN variable-K dense-BQ",
               variable_ms,
               variable_ns,
               variable_ks,
               run_bf16_output) &&
           pass;
    pass = run_blockwise_grouped_case<ck_tile::fp8_t,
                                      ck_tile::fp8_t,
                                      ck_tile::half_t,
                                      Row,
                                      Col>("blockwise grouped FP8xFP8->FP16 NT M/N/K padded",
                                           padded_ms,
                                           padded_ns,
                                           padded_ks) &&
           pass;
    return pass ? 0 : 1;
}
