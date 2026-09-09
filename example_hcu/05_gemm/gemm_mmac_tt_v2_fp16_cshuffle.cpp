// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_mmac_tt_v2_cshuffle.hpp"

struct ProblemSize final
{
    ck::index_t M = 128;
    ck::index_t N = 512;
    ck::index_t K = 256;

    ck::index_t StrideA = K;
    ck::index_t StrideB = N;
    ck::index_t StrideC = N;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 2;
    bool time_kernel     = true;
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

inline bool
parse_cmd_args(int argc, char* argv[], ProblemSize& problem_size, ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC" << std::endl;
        return false;
    }

    return true;
}

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using AccDataType = float;
using CDataType   = ck::half_t;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance0 =
    ck::tensor_operation::device::DeviceGemm_mmac_tt_mk_n0k0k1n1_mn_v2_cshuffle<
        ADataType,
        BDataType,
        AccDataType,
        CDataType,
        ck::InMemoryDataOperationEnum::Set,
        AElementOp,
        BElementOp,
        CElementOp,
        GemmDefault,
        256,
        128,
        4,
        32,
        1,
        16,
        16,
        16,
        4,
        2,
        1,
        1,
        1,
        2,
        S<1, 128, 2>,
        2,
        8,
        S<1, 4, 1, 16, 4>,
        4,
        8,
        1,
        1,
        S<1, 1, 16, 1, 1, 16>,
        4,
        2>;

using DeviceGemmInstance = DeviceGemmInstance0;

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
