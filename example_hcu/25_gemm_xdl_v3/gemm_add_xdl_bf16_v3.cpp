// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_epilogue_bf16_v3_common.hpp"

using Add = ck::tensor_operation::element_wise::Add;

int main(int argc, char* argv[])
{
    return GemmEpilogueBf16V3Example<Add, false>::main(argc, argv);
}
