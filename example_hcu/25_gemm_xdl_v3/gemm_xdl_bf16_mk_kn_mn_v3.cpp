// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
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
    ck::index_t M = 256;
    ck::index_t N = 256;
    ck::index_t K = 128;

    ck::index_t StrideA = K;
    ck::index_t StrideB = N;
    ck::index_t StrideC = N;

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
using CDataType        = BF16;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    256,            // BlockSize
    128,            // MPerBlock
    128,            // NPerBlock
    32,             // KPerBlock
    8,              // AK1
    8,              // BK1
    16,             // MPerXDL
    16,             // NPerXDL
    4,              // MXdlPerWave
    4,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_AK1
    0,              // ABlockLdsExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
    S<0, 2, 1>,     // BBlockTransferThreadClusterArrangeOrder
    S<0, 2, 1>,     // BBlockTransferSrcAccessOrder
    1,              // BBlockTransferSrcVectorDim
    2,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_BK1
    0,              // BBlockLdsExtraN
    1,              // CShuffleMXdlPerWavePerShuffle
    1,              // CShuffleNXdlPerWavePerShuffle
    S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    4,              // CShuffleBlockTransferScalarPerVector_NPerBlock
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v3>;

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

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
            problem_size.StrideC = std::stoi(argv[9]);
            problem_size.KBatch  = std::stoi(argv[10]);
        }

        return true;
    }

    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << "arg4 to 10: M, N, K, StrideA, StrideB, StrideC, KBatch\n";
    return false;
}

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
