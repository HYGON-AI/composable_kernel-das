// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstdlib>
#include <iostream>
#include <random>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

using ::HostTensorDescriptor;
using ::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using ADataType        = BF16;
using BDataType        = BF16;
using DDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using EDataType        = BF16;

using ALayout  = Row;
using BLayout  = Col;
using DLayout  = Row;
using ELayout  = Row;
using DsLayout = ck::Tuple<DLayout>;

using DsDataType = ck::Tuple<DDataType>;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    ADataType,
    BDataType,
    DsDataType,
    EDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    GemmDefault,
    256,
    128,
    128,
    32,
    8,
    8,
    16,
    16,
    4,
    4,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    0,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    0,
    1,
    1,
    S<1, 32, 1, 8>,
    S<4>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v3>;

using ReferenceBatchedGemmInstance =
    ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                     BDataType,
                                                     CShuffleDataType,
                                                     AccDataType,
                                                     AElementOp,
                                                     BElementOp,
                                                     PassThrough>;

struct ProblemSize final
{
    ck::index_t M = 128;
    ck::index_t N = 128;
    ck::index_t K = 128;

    ck::index_t stride_A = K;
    ck::index_t stride_B = K;
    ck::index_t stride_D = N;
    ck::index_t stride_E = N;

    ck::index_t batch_stride_A = M * K;
    ck::index_t batch_stride_B = K * N;
    ck::index_t batch_stride_D = M * N;
    ck::index_t batch_stride_E = M * N;

    ck::index_t batch_count = 2;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

template <typename Layout>
static auto make_host_tensor_descriptor(std::size_t batch_count,
                                        std::size_t row,
                                        std::size_t col,
                                        std::size_t stride,
                                        std::size_t batch_stride,
                                        Layout)
{
    using namespace ck::literals;

    if constexpr(std::is_same_v<Layout, Row>)
    {
        return HostTensorDescriptor({batch_count, row, col}, {batch_stride, stride, 1_uz});
    }
    else
    {
        return HostTensorDescriptor({batch_count, row, col}, {batch_stride, 1_uz, stride});
    }
}

static void refresh_default_strides(ProblemSize& problem_size)
{
    problem_size.stride_A       = problem_size.K;
    problem_size.stride_B       = problem_size.K;
    problem_size.stride_D       = problem_size.N;
    problem_size.stride_E       = problem_size.N;
    problem_size.batch_stride_A = problem_size.M * problem_size.K;
    problem_size.batch_stride_B = problem_size.K * problem_size.N;
    problem_size.batch_stride_D = problem_size.M * problem_size.N;
    problem_size.batch_stride_E = problem_size.M * problem_size.N;
}

static bool parse_cmd_args(int argc, char* argv[], ProblemSize& problem_size, ExecutionConfig& config)
{
    if(argc == 1)
    {
        return true;
    }

    if(argc == 4 || argc == 8)
    {
        config.do_verification = std::stoi(argv[1]) != 0;
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]) != 0;

        if(argc == 8)
        {
            problem_size.M           = std::stoi(argv[4]);
            problem_size.N           = std::stoi(argv[5]);
            problem_size.K           = std::stoi(argv[6]);
            problem_size.batch_count = std::stoi(argv[7]);
            refresh_default_strides(problem_size);
        }

        return true;
    }

    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << "arg4 to 7: M, N, K, Batch\n";
    return false;
}

static bool run_batched_gemm_multi_d_add(const ProblemSize& problem_size,
                                         const ExecutionConfig& config)
{
    const auto& [M,
                 N,
                 K,
                 stride_A,
                 stride_B,
                 stride_D,
                 stride_E,
                 batch_stride_A,
                 batch_stride_B,
                 batch_stride_D,
                 batch_stride_E,
                 batch_count] = problem_size;

    Tensor<ADataType> a_g_m_k(
        make_host_tensor_descriptor(batch_count, M, K, stride_A, batch_stride_A, ALayout{}));
    Tensor<BDataType> b_g_k_n(
        make_host_tensor_descriptor(batch_count, K, N, stride_B, batch_stride_B, BLayout{}));
    Tensor<DDataType> d_g_m_n(
        make_host_tensor_descriptor(batch_count, M, N, stride_D, batch_stride_D, DLayout{}));
    Tensor<EDataType> e_g_m_n_device_result(
        make_host_tensor_descriptor(batch_count, M, N, stride_E, batch_stride_E, ELayout{}));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "d_g_m_n: " << d_g_m_n.mDesc << std::endl;
    std::cout << "e_g_m_n: " << e_g_m_n_device_result.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{0, 2});
        d_g_m_n.GenerateTensorValue(GeneratorTensor_2<DDataType>{-2, 2});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_g_m_n.GenerateTensorValue(GeneratorTensor_3<DDataType>{-0.5, 0.5});
        break;
    }

    ck::DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    ck::DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpaceSize());
    ck::DeviceMem d_device_buf(sizeof(DDataType) * d_g_m_n.mDesc.GetElementSpaceSize());
    ck::DeviceMem e_device_buf(sizeof(EDataType) *
                               e_g_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());
    d_device_buf.ToDevice(d_g_m_n.mData.data());

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    auto argument = gemm.MakeArgument(
        a_device_buf.GetDeviceBuffer(),
        b_device_buf.GetDeviceBuffer(),
        std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
        e_device_buf.GetDeviceBuffer(),
        M,
        N,
        K,
        batch_count,
        stride_A,
        stride_B,
        std::array<ck::index_t, 1>{stride_D},
        stride_E,
        batch_stride_A,
        batch_stride_B,
        std::array<ck::index_t, 1>{batch_stride_D},
        batch_stride_E,
        AElementOp{},
        BElementOp{},
        CDEElementOp{});

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;
        return false;
    }

    invoker.Run(argument, StreamConfig{nullptr, false});

    if(config.time_kernel)
    {
        const float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
        const std::size_t flop = std::size_t{2} * batch_count * M * N * K;
        const std::size_t num_btype =
            sizeof(ADataType) * batch_count * M * K + sizeof(BDataType) * batch_count * K * N +
            sizeof(DDataType) * batch_count * M * N + sizeof(EDataType) * batch_count * M * N;
        const float tflops     = static_cast<float>(flop) / 1.E9f / ave_time;
        const float gb_per_sec = static_cast<float>(num_btype) / 1.E6f / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }

    if(!config.do_verification)
    {
        return true;
    }

    Tensor<CShuffleDataType> c_g_m_n_host_result(
        make_host_tensor_descriptor(batch_count, M, N, stride_E, batch_stride_E, ELayout{}));
    Tensor<EDataType> e_g_m_n_host_result(
        make_host_tensor_descriptor(batch_count, M, N, stride_E, batch_stride_E, ELayout{}));

    auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
    auto ref_invoker      = ref_batched_gemm.MakeInvoker();
    auto ref_argument     = ref_batched_gemm.MakeArgument(
        a_g_m_k, b_g_k_n, c_g_m_n_host_result, AElementOp{}, BElementOp{}, PassThrough{});

    ref_invoker.Run(ref_argument);

    for(ck::index_t g = 0; g < batch_count; ++g)
    {
        for(ck::index_t m = 0; m < M; ++m)
        {
            for(ck::index_t n = 0; n < N; ++n)
            {
                const float c = ck::type_convert<float>(c_g_m_n_host_result(g, m, n));
                const float d = ck::type_convert<float>(d_g_m_n(g, m, n));
                e_g_m_n_host_result(g, m, n) = ck::type_convert<EDataType>(c + d);
            }
        }
    }

    e_device_buf.FromDevice(e_g_m_n_device_result.mData.data());

    return ck::utils::check_err(
        e_g_m_n_device_result, e_g_m_n_host_result, "Error: Incorrect results c", 5e-2, 5e-2);
}

int main(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    if(!parse_cmd_args(argc, argv, problem_size, config))
    {
        return 1;
    }

    return run_batched_gemm_multi_d_add(problem_size, config) ? 0 : 1;
}
