// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

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
    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(
        problem_size.M, problem_size.N, problem_size.StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(
        problem_size.M, problem_size.N, problem_size.StrideC, CLayout{}));

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                      problem_size.M,
                                      problem_size.N,
                                      problem_size.K,
                                      problem_size.StrideA,
                                      problem_size.StrideB,
                                      problem_size.StrideC,
                                      problem_size.KBatch,
                                      AElementOp{},
                                      BElementOp{},
                                      CElementOp{});

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;
        return true;
    }

    const float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    const std::size_t flop = 2_uz * problem_size.M * problem_size.N * problem_size.K;
    const std::size_t num_btype =
        sizeof(ADataType) * problem_size.M * problem_size.K +
        sizeof(BDataType) * problem_size.K * problem_size.N +
        sizeof(CDataType) * problem_size.M * problem_size.N;

    std::cout << "Perf: " << ave_time << " ms, "
              << static_cast<float>(flop) / 1.E9 / ave_time << " TFlops, "
              << num_btype / 1.E6 / ave_time << " GB/s, " << gemm.GetTypeString() << std::endl;

    if(!config.do_verification)
    {
        return true;
    }

    auto ref_gemm    = ReferenceGemmInstance{};
    auto ref_invoker = ref_gemm.MakeInvoker();
    auto ref_argument =
        ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n_host_result, AElementOp{}, BElementOp{}, CElementOp{});

    ref_invoker.Run(ref_argument);
    c_device_buf.FromDevice(c_m_n_device_result.mData.data());

    return ck::utils::check_err(
        c_m_n_device_result, c_m_n_host_result, "Error: Incorrect results!", 5e-2, 5e-2);
}

bool run_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}
