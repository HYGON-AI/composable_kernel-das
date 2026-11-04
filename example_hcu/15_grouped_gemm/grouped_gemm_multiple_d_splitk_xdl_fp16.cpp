// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Based on AMD CK example/15_grouped_gemm/grouped_gemm_multiple_d_splitk_xdl_fp16.cpp.
#include "grouped_gemm_splitk_common.hpp"
int main(int argc, char** argv)
{
    using namespace splitk_example;
    using AddAdd = tensor_operation::element_wise::AddAdd;
    using Op =
        TestOp<TwoStage<half_t, float, Tuple<half_t, half_t>, Tuple<Row, Row>, AddAdd>, AddAdd>;
    try
    {
        const int batch = argc > 1 ? std::stoi(argv[1]) : 4;
        return !Run<Op, half_t, float, Row, Col, 2>({64, 127, 255, 129, 260, 190, 77},
                                                    {252, 252, 252, 252, 252, 252, 252},
                                                    {4608, 4608, 4608, 4608, 4608, 4608, 4608},
                                                    batch,
                                                    argc > 2,
                                                    argc > 3);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
