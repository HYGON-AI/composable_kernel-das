// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <array>
#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

using ::ck::DeviceMem;
using ::HostTensorDescriptor;
using ::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;

    ck::index_t KBatch = 1;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

using ADataType        = BF16;
using BDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using DsDataType       = ck::Tuple<>;
using EDataType        = BF16;

using ALayout  = Row;
using BLayout  = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
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

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, EDataType, AccDataType, AElementOp, BElementOp, CDEElementOp>;

static bool parse_cmd_args(int argc, char* argv[], ProblemSize& problem_size, ExecutionConfig& config)
{
    if(argc == 1)
    {
        return true;
    }
    if(argc == 4 || argc == 11)
    {
        config.do_verification = std::stoi(argv[1]) != 0;
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]) != 0;

        if(argc == 11)
        {
            problem_size.M       = std::stoi(argv[4]);
            problem_size.N       = std::stoi(argv[5]);
            problem_size.K       = std::stoi(argv[6]);
            problem_size.StrideA = std::stoi(argv[7]);
            problem_size.StrideB = std::stoi(argv[8]);
            problem_size.StrideE = std::stoi(argv[9]);
            problem_size.KBatch  = std::stoi(argv[10]);
        }

        return true;
    }

    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << "arg4 to 10: M, N, K, StrideA, StrideB, StrideE, KBatch\n";
    return false;
}

static bool run_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    const auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(
        problem_size.M, problem_size.K, problem_size.StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(
        problem_size.K, problem_size.N, problem_size.StrideB, BLayout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(
        problem_size.M, problem_size.N, problem_size.StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(
        problem_size.M, problem_size.N, problem_size.StrideE, ELayout{}));

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{0, 2});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto device_op = DeviceGemmInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                           b_device_buf.GetDeviceBuffer(),
                                           {},
                                           e_device_buf.GetDeviceBuffer(),
                                           problem_size.M,
                                           problem_size.N,
                                           problem_size.K,
                                           problem_size.StrideA,
                                           problem_size.StrideB,
                                           {},
                                           problem_size.StrideE,
                                           problem_size.KBatch,
                                           AElementOp{},
                                           BElementOp{},
                                           CDEElementOp{});

    if(!device_op.IsSupportedArgument(argument))
    {
        std::cerr << device_op.GetTypeString() << " does not support this problem" << std::endl;
        return false;
    }

    const float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
    std::cout << "Perf: " << ave_time << " ms, " << device_op.GetTypeString() << std::endl;

    if(!config.do_verification)
    {
        return true;
    }

    auto ref_gemm    = ReferenceGemmInstance{};
    auto ref_invoker = ref_gemm.MakeInvoker();
    auto ref_argument =
        ref_gemm.MakeArgument(a_m_k, b_k_n, e_m_n_host_result, AElementOp{}, BElementOp{}, CDEElementOp{});

    ref_invoker.Run(ref_argument);
    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    return ck::utils::check_err(
        e_m_n_device_result, e_m_n_host_result, "Error: Incorrect results!", 5e-2, 5e-2);
}

int main(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    if(!parse_cmd_args(argc, argv, problem_size, config))
    {
        return 1;
    }

    return run_gemm(problem_size, config) ? 0 : 1;
}
