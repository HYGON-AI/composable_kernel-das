// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

namespace ck_tile {

struct JengaMmacQKConfig
{
    static constexpr ck_tile::index_t kN         = 64;
    static constexpr ck_tile::index_t kK         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 128>;
};

struct JengaMmacDPK32Config
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 32;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 32>;
};

struct JengaMmacQKChunkConfig
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 64;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<4, 1, 1>;
    using WarpTile   = ck_tile::sequence<16, 32, 64>;
};

struct JengaMmacQKHalfConfig
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 64;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 64>;
};

struct JengaMmacDVConfig
{
    static constexpr ck_tile::index_t kM         = 64;
    static constexpr ck_tile::index_t kN         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<1, 4, 1>;
    using WarpTile   = ck_tile::sequence<16, 64, 32>;
};

struct JengaMmacDPConfig
{
    static constexpr ck_tile::index_t kN         = 64;
    static constexpr ck_tile::index_t kK         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 128>;
};

struct JengaMmacDKConfig
{
    static constexpr ck_tile::index_t kM         = 64;
    static constexpr ck_tile::index_t kN         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 64>;
};

template <typename Problem>
using JengaMmacQKGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::kBlockM,
                                             JengaMmacQKConfig::kN,
                                             JengaMmacQKConfig::kK>,
                           typename JengaMmacQKConfig::BlockWarps,
                           typename JengaMmacQKConfig::WarpTile>;

template <typename Problem>
using JengaMmacQKGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::KDataType,
                              typename Problem::QDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacQKGemmShape<Problem>>;

template <typename Problem>
using JengaMmacQKChunkGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::kBlockM,
                                             JengaMmacQKChunkConfig::kN,
                                             JengaMmacQKChunkConfig::kK>,
                           typename JengaMmacQKChunkConfig::BlockWarps,
                           typename JengaMmacQKChunkConfig::WarpTile>;

template <typename Problem>
using JengaMmacQKChunkGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::KDataType,
                              typename Problem::QDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacQKChunkGemmShape<Problem>>;

template <typename Problem>
using JengaMmacQKHalfGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<32,
                                             JengaMmacQKHalfConfig::kN,
                                             JengaMmacQKHalfConfig::kK>,
                           typename JengaMmacQKHalfConfig::BlockWarps,
                           typename JengaMmacQKHalfConfig::WarpTile>;

template <typename Problem>
using JengaMmacQKHalfGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::KDataType,
                              typename Problem::QDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacQKHalfGemmShape<Problem>>;

template <typename Problem>
using JengaMmacDVGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<JengaMmacDVConfig::kM,
                                             JengaMmacDVConfig::kN,
                                             Problem::kBlockM>,
                           typename JengaMmacDVConfig::BlockWarps,
                           typename JengaMmacDVConfig::WarpTile>;

template <typename Problem>
using JengaMmacDVGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::OGradDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacDVGemmShape<Problem>>;

template <typename Problem>
using JengaMmacDVK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<JengaMmacDVConfig::kM,
                                             JengaMmacDVConfig::kN,
                                             32>,
                           typename JengaMmacDVConfig::BlockWarps,
                           typename JengaMmacDVConfig::WarpTile>;

template <typename Problem>
using JengaMmacDVK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::OGradDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacDVK32GemmShape<Problem>>;

template <typename Problem>
using JengaMmacDPGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::kBlockM,
                                             JengaMmacDPConfig::kN,
                                             JengaMmacDPConfig::kK>,
                           typename JengaMmacDPConfig::BlockWarps,
                           typename JengaMmacDPConfig::WarpTile>;

template <typename Problem>
using JengaMmacDPGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::VDataType,
                              typename Problem::OGradDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacDPGemmShape<Problem>>;

template <typename Problem>
using JengaMmacDPK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::kBlockM,
                                             JengaMmacDPK32Config::kN,
                                             JengaMmacDPK32Config::kK>,
                           typename JengaMmacDPK32Config::BlockWarps,
                           typename JengaMmacDPK32Config::WarpTile>;

template <typename Problem>
using JengaMmacDPK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::VDataType,
                              typename Problem::OGradDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacDPK32GemmShape<Problem>>;

template <typename Problem>
using JengaMmacDKGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<JengaMmacDKConfig::kM,
                                             JengaMmacDKConfig::kN,
                                             Problem::kBlockM>,
                           typename JengaMmacDKConfig::BlockWarps,
                           typename JengaMmacDKConfig::WarpTile>;

template <typename Problem>
using JengaMmacDKGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::QDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacDKGemmShape<Problem>>;

template <typename Problem>
using JengaMmacDQK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<JengaMmacDVConfig::kM,
                                             JengaMmacDVConfig::kN,
                                             32>,
                           typename JengaMmacDPK32Config::BlockWarps,
                           typename JengaMmacDPK32Config::WarpTile>;

template <typename Problem>
using JengaMmacDQK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::KDataType,
                              float,
                              Problem::kBlockSize,
                              JengaMmacDQK32GemmShape<Problem>>;

} // namespace ck_tile
