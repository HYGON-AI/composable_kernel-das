// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_epilogue_bf16_v3_common.hpp"

using AddRelu = ck::tensor_operation::element_wise::AddRelu;

int main(int argc, char* argv[])
{
    return GemmEpilogueBf16V3Example<AddRelu, true>::main(argc, argv);
}
