// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "gemm_wmma_hcu_example.hpp"

int main(int argc, char* argv[]) { return wmma_example::run<ck::bhalf_t>(argc, argv); }
