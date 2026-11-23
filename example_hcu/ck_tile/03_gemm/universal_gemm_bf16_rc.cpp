// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "universal_gemm_dispatch_impl.hpp"

template float launch<ck_tile::bf16_t, Col, false>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
template float launch<ck_tile::bf16_t, Col, true>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
