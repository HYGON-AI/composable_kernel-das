// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multi_abd_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
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
using Add         = ck::tensor_operation::element_wise::Add;

struct ProblemSize final
{
    ck::index_t M = 256;
    ck::index_t N = 256;
    ck::index_t K = 128;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = N;
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
using DDataType        = BF16;
using EDataType        = BF16;

using AsDataType = ck::Tuple<ADataType, ADataType>;
using BsDataType = ck::Tuple<BDataType, BDataType>;
using DsDataType = ck::Tuple<DDataType>;

using ALayout  = Row;
using BLayout  = Col;
using DLayout  = Row;
using ELayout  = Row;
using AsLayout = ck::Tuple<ALayout, ALayout>;
using BsLayout = ck::Tuple<BLayout, BLayout>;
using DsLayout = ck::Tuple<DLayout>;

using AElementOp   = Add;
using BElementOp   = Add;
using CDEElementOp = Add;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmMultiABD_Xdl_CShuffle_V3<
    AsLayout,
    BsLayout,
    DsLayout,
    ELayout,
    AsDataType,
    BsDataType,
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

using ReferenceGemmInstance =
    ck::tensor_operation::host::ReferenceGemm<ADataType,
                                              BDataType,
                                              CShuffleDataType,
                                              AccDataType,
                                              PassThrough,
                                              PassThrough,
                                              PassThrough>;

static bool parse_cmd_args(int argc, char* argv[], ProblemSize& problem_size, ExecutionConfig& cfg)
{
    if(argc == 1)
    {
        return true;
    }

    if(argc == 4 || argc == 12)
    {
        cfg.do_verification = std::stoi(argv[1]) != 0;
        cfg.init_method     = std::stoi(argv[2]);
        cfg.time_kernel     = std::stoi(argv[3]) != 0;

        if(argc == 12)
        {
            problem_size.M       = std::stoi(argv[4]);
            problem_size.N       = std::stoi(argv[5]);
            problem_size.K       = std::stoi(argv[6]);
            problem_size.StrideA = std::stoi(argv[7]);
            problem_size.StrideB = std::stoi(argv[8]);
            problem_size.StrideD = std::stoi(argv[9]);
            problem_size.StrideE = std::stoi(argv[10]);
            problem_size.KBatch  = std::stoi(argv[11]);
        }

        return true;
    }

    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << "arg4 to 11: M, N, K, StrideA, StrideB, StrideD, StrideE, KBatch\n";
    return false;
}

static bool run_gemm(const ProblemSize& problem_size, const ExecutionConfig& cfg)
{
    using namespace ck::literals;

    const auto make_desc = [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
        if constexpr(std::is_same_v<decltype(layout), Row>)
        {
            return HostTensorDescriptor({row, col}, {stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({row, col}, {1_uz, stride});
        }
    };

    Tensor<ADataType> a0_m_k(
        make_desc(problem_size.M, problem_size.K, problem_size.StrideA, ALayout{}));
    Tensor<ADataType> a1_m_k(
        make_desc(problem_size.M, problem_size.K, problem_size.StrideA, ALayout{}));
    Tensor<ADataType> a_m_k(
        make_desc(problem_size.M, problem_size.K, problem_size.StrideA, ALayout{}));
    Tensor<BDataType> b0_k_n(
        make_desc(problem_size.K, problem_size.N, problem_size.StrideB, BLayout{}));
    Tensor<BDataType> b1_k_n(
        make_desc(problem_size.K, problem_size.N, problem_size.StrideB, BLayout{}));
    Tensor<BDataType> b_k_n(
        make_desc(problem_size.K, problem_size.N, problem_size.StrideB, BLayout{}));
    Tensor<DDataType> d_m_n(
        make_desc(problem_size.M, problem_size.N, problem_size.StrideD, DLayout{}));
    Tensor<EDataType> e_m_n_host_result(
        make_desc(problem_size.M, problem_size.N, problem_size.StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(
        make_desc(problem_size.M, problem_size.N, problem_size.StrideE, ELayout{}));

    switch(cfg.init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        a1_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-1, 1});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{0, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-1, 1});
        d_m_n.GenerateTensorValue(GeneratorTensor_2<DDataType>{-2, 2});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        a1_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{0.0, 1.0});
        d_m_n.GenerateTensorValue(GeneratorTensor_3<DDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a0_device_buf(sizeof(ADataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem a1_device_buf(sizeof(ADataType) * a1_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(BDataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(BDataType) * b1_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    a1_device_buf.ToDevice(a1_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n.mData.data());
    b1_device_buf.ToDevice(b1_k_n.mData.data());
    d_device_buf.ToDevice(d_m_n.mData.data());

    auto device_op = DeviceGemmInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument({a0_device_buf.GetDeviceBuffer(),
                                             a1_device_buf.GetDeviceBuffer()},
                                           {b0_device_buf.GetDeviceBuffer(),
                                            b1_device_buf.GetDeviceBuffer()},
                                           {d_device_buf.GetDeviceBuffer()},
                                           e_device_buf.GetDeviceBuffer(),
                                           problem_size.M,
                                           problem_size.N,
                                           problem_size.K,
                                           {problem_size.StrideA, problem_size.StrideA},
                                           {problem_size.StrideB, problem_size.StrideB},
                                           {problem_size.StrideD},
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

    const float ave_time = invoker.Run(argument, StreamConfig{nullptr, cfg.time_kernel});
    std::cout << "Perf: " << ave_time << " ms, " << device_op.GetTypeString() << std::endl;

    if(!cfg.do_verification)
    {
        return true;
    }

    Tensor<CShuffleDataType> c_m_n_host_result(
        make_desc(problem_size.M, problem_size.N, problem_size.StrideE, ELayout{}));

    for(ck::index_t m = 0; m < problem_size.M; ++m)
    {
        for(ck::index_t k = 0; k < problem_size.K; ++k)
        {
            AElementOp{}(a_m_k(m, k), a0_m_k(m, k), a1_m_k(m, k));
        }
    }
    for(ck::index_t k = 0; k < problem_size.K; ++k)
    {
        for(ck::index_t n = 0; n < problem_size.N; ++n)
        {
            BElementOp{}(b_k_n(k, n), b0_k_n(k, n), b1_k_n(k, n));
        }
    }

    auto ref_gemm     = ReferenceGemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(
        a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

    ref_invoker.Run(ref_argument);

    for(ck::index_t m = 0; m < problem_size.M; ++m)
    {
        for(ck::index_t n = 0; n < problem_size.N; ++n)
        {
            const float c = ck::type_convert<float>(c_m_n_host_result(m, n));
            const float d = ck::type_convert<float>(d_m_n(m, n));
            e_m_n_host_result(m, n) = ck::type_convert<EDataType>(c + d);
        }
    }

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    return ck::utils::check_err(
        e_m_n_device_result, e_m_n_host_result, "Error: Incorrect results!", 5e-2, 5e-2);
}

int main(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig cfg;

    if(!parse_cmd_args(argc, argv, problem_size, cfg))
    {
        return 1;
    }

    return run_gemm(problem_size, cfg) ? 0 : 1;
}
