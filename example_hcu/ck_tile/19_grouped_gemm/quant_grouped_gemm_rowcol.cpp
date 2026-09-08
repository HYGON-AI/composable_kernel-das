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
struct RowColGroup
{
    RowColGroup(ck_tile::index_t m, ck_tile::index_t n, ck_tile::index_t k, std::size_t group_id)
        : M(m),
          N(n),
          K(k),
          a(make_matrix_descriptor<ALayout>(M, K)),
          b(make_matrix_descriptor<BLayout>(K, N)),
          aq(make_matrix_descriptor<Row>(M, 1)),
          bq(make_matrix_descriptor<Row>(1, N)),
          c_device(make_matrix_descriptor<Row>(M, N)),
          c_expected(make_matrix_descriptor<Row>(M, N)),
          c_reference(make_matrix_descriptor<Row>(M, N)),
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
            aq(row, 0) = 0.30f + 0.02f * static_cast<float>((row + group_id) % 7);
        }
        for(ck_tile::index_t column = 0; column < N; ++column)
        {
            bq(0, column) =
                0.55f + 0.03f * static_cast<float>((column + 2 * group_id) % 5);
        }

        c_device.SetZero();
        c_expected.SetZero();
        c_reference.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, float, float>(a, b, c_reference);
        for(ck_tile::index_t row = 0; row < M; ++row)
        {
            for(ck_tile::index_t column = 0; column < N; ++column)
            {
                c_expected(row, column) = ck_tile::type_convert<CDataType>(
                    c_reference(row, column) * aq(row, 0) * bq(0, column));
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
                1,
                1,
                std::is_same_v<ALayout, Row> ? K : M,
                std::is_same_v<BLayout, Row> ? N : K,
                N,
                1,
                1};
    }

    bool Check(std::size_t group_id)
    {
        c_device_buffer.FromDevice(c_device.data());
        return ck_tile::check_err(c_device,
                                  c_expected,
                                  "rowwise quant grouped GEMM group " +
                                      std::to_string(group_id) + " mismatch",
                                  1e-2,
                                  1e-2);
    }

    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::HostTensor<ADataType> a;
    ck_tile::HostTensor<BDataType> b;
    ck_tile::HostTensor<float> aq;
    ck_tile::HostTensor<float> bq;
    ck_tile::HostTensor<CDataType> c_device;
    ck_tile::HostTensor<CDataType> c_expected;
    ck_tile::HostTensor<float> c_reference;
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
bool run_rowcol_grouped_case(const char* case_name,
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
        ck_tile::QuantGroupedGemmKernel<TilePartitioner,
                                        Pipeline,
                                        Epilogue,
                                        ck_tile::QuantType::RowColQuant>;
    using Group = RowColGroup<ADataType, BDataType, CDataType, ALayout, BLayout>;

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
bool run_rowcol_dtype_matrix(const char* layout_name,
                             const std::vector<ck_tile::index_t>& ms,
                             const std::vector<ck_tile::index_t>& ns,
                             const std::vector<ck_tile::index_t>& ks,
                             bool run_bf16_output)
{
    const std::string prefix = "rowwise grouped ";
    const std::string suffix = std::string{" "} + layout_name;
    bool pass                = true;

#define RUN_ROWCOL_CASE(AType, BType, CType, TypeName)                                  \
    pass = run_rowcol_grouped_case<AType, BType, CType, ALayout, BLayout>(              \
               (prefix + TypeName + suffix).c_str(), ms, ns, ks) &&                     \
           pass

    RUN_ROWCOL_CASE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, "FP8xFP8->FP16");
    if(run_bf16_output)
        RUN_ROWCOL_CASE(ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bf16_t, "FP8xFP8->BF16");
    RUN_ROWCOL_CASE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, "BF8xBF8->FP16");
    if(run_bf16_output)
        RUN_ROWCOL_CASE(ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bf16_t, "BF8xBF8->BF16");
    RUN_ROWCOL_CASE(ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::half_t, "FP8xBF8->FP16");
    if(run_bf16_output)
        RUN_ROWCOL_CASE(ck_tile::fp8_t, ck_tile::bf8_t, ck_tile::bf16_t, "FP8xBF8->BF16");
    RUN_ROWCOL_CASE(ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::half_t, "BF8xFP8->FP16");
    if(run_bf16_output)
        RUN_ROWCOL_CASE(ck_tile::bf8_t, ck_tile::fp8_t, ck_tile::bf16_t, "BF8xFP8->BF16");
    if(!run_bf16_output)
        std::cout << "rowwise grouped BF16 output " << layout_name << " SKIP-PMD"
                  << std::endl;

#undef RUN_ROWCOL_CASE
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
    const std::vector<ck_tile::index_t> fixed_ks{128, 128, 128};
    const std::vector<ck_tile::index_t> padded_ms{127, 160, 65};
    const std::vector<ck_tile::index_t> padded_ns{192, 120, 136};
    const std::vector<ck_tile::index_t> padded_ks{80, 80, 80};
    const std::vector<ck_tile::index_t> variable_ms{128, 128, 128};
    const std::vector<ck_tile::index_t> variable_ns{128, 128, 128};
    const std::vector<ck_tile::index_t> variable_ks{128, 160, 256};

    bool pass = run_rowcol_dtype_matrix<Row, Col>(
        "NT fixed-K", fixed_ms, fixed_ns, fixed_ks, run_bf16_output);
    pass = run_rowcol_dtype_matrix<Row, Row>(
               "NN fixed-K", fixed_ms, fixed_ns, fixed_ks, run_bf16_output) &&
           pass;
    pass = run_rowcol_dtype_matrix<Col, Row>(
               "TN variable-K", variable_ms, variable_ns, variable_ks, run_bf16_output) &&
           pass;
    pass = run_rowcol_grouped_case<ck_tile::fp8_t,
                                   ck_tile::fp8_t,
                                   ck_tile::half_t,
                                   Row,
                                   Col>("rowwise grouped FP8xFP8->FP16 NT M/N/K padded",
                                        padded_ms,
                                        padded_ns,
                                        padded_ks) &&
           pass;
    return pass ? 0 : 1;
}
