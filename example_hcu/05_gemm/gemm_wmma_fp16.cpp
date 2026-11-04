// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#include "gemm_wmma_hcu_example.hpp"

int main(int argc, char* argv[]) { return wmma_example::run<ck::half_t>(argc, argv); }
