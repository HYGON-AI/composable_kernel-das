// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "gemm_multi_d_f16_v3_common.hpp"

// A [M,K]: Col; B [K,N]: Row; E [M,N]: Row (CRR).
using ALayout = hcu_f16_multi_d::Col;
using BLayout = hcu_f16_multi_d::Row;

int main(int argc, char* argv[])
{
    return hcu_f16_multi_d::run_example<ALayout, BLayout>(argc, argv);
}
