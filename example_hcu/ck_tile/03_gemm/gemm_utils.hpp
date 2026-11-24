// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/ops/gemm.hpp"

enum class Pipeline { CompV3, CompV4, Mem };

template <Pipeline>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<Pipeline::CompV3>
{
    template <typename Problem>
    using Base = ck_tile::BaseGemmPipelineAgBgCrCompV3<Problem>;
    template <typename Problem>
    using Gemm = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
};

template <>
struct PipelineTypeTraits<Pipeline::CompV4>
{
    template <typename Problem>
    using Base = ck_tile::BaseGemmPipelineAgBgCrCompV4<Problem>;
    template <typename Problem>
    using Gemm = ck_tile::GemmPipelineAgBgCrCompV4<Problem>;
};

template <>
struct PipelineTypeTraits<Pipeline::Mem>
{
    template <typename Problem>
    using Base = ck_tile::BaseGemmPipelineAgBgCrMem<Problem>;
    template <typename Problem>
    using Gemm = ck_tile::GemmPipelineAgBgCrMem<Problem>;
};

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 32>,
                                       ck_tile::sequence<2, 2, 1>,
                                       ck_tile::sequence<32, 32, 16>>;
