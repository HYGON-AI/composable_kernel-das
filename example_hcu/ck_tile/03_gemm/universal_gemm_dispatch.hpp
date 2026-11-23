// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "gemm_utils.hpp"

using Scheduler = ck_tile::GemmPipelineScheduler;

struct UniversalGemmOptions
{
    ck_tile::index_t m, n, k;
    ck_tile::index_t lda, ldb, ldc;
    int init, verify;
    Pipeline pipeline;
    Scheduler scheduler;
    bool shuffle;
    ck_tile::stream_config stream;
};

template <typename T, typename BLayout, bool Shuffle>
float launch(const UniversalGemmOptions& options, const ck_tile::UniversalGemmKernelArgs<>& args);

extern template float launch<ck_tile::half_t, Row, false>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
extern template float launch<ck_tile::half_t, Row, true>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
extern template float launch<ck_tile::half_t, Col, false>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
extern template float launch<ck_tile::half_t, Col, true>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
extern template float launch<ck_tile::bf16_t, Row, false>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
extern template float launch<ck_tile::bf16_t, Row, true>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
extern template float launch<ck_tile::bf16_t, Col, false>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
extern template float launch<ck_tile::bf16_t, Col, true>(
    const UniversalGemmOptions&, const ck_tile::UniversalGemmKernelArgs<>&);
