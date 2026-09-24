// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"

namespace ck_tile {

struct SlaAttnMmacQKConfig
{
    static constexpr ck_tile::index_t kN         = 64;
    static constexpr ck_tile::index_t kK         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 128>;
};

struct SlaAttnMmacDPK32Config
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 32;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 32>;
};

struct SlaAttnMmacQKChunkConfig
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 64;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<4, 1, 1>;
    using WarpTile   = ck_tile::sequence<16, 32, 64>;
};

struct SlaAttnMmacQKHalfConfig
{
    static constexpr ck_tile::index_t kN         = 32;
    static constexpr ck_tile::index_t kK         = 64;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 64>;
};

struct SlaAttnMmacDVConfig
{
    static constexpr ck_tile::index_t kM         = 64;
    static constexpr ck_tile::index_t kN         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<1, 4, 1>;
    using WarpTile   = ck_tile::sequence<16, 64, 32>;
};

struct SlaAttnMmacDPConfig
{
    static constexpr ck_tile::index_t kN         = 64;
    static constexpr ck_tile::index_t kK         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 128>;
};

struct SlaAttnMmacDKConfig
{
    static constexpr ck_tile::index_t kM         = 64;
    static constexpr ck_tile::index_t kN         = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    using BlockWarps = ck_tile::sequence<2, 2, 1>;
    using WarpTile   = ck_tile::sequence<16, 16, 64>;
};

template <typename Problem>
using SlaAttnMmacQKGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::kBlockM,
                                             SlaAttnMmacQKConfig::kN,
                                             SlaAttnMmacQKConfig::kK>,
                           typename SlaAttnMmacQKConfig::BlockWarps,
                           typename SlaAttnMmacQKConfig::WarpTile>;

template <typename Problem>
using SlaAttnMmacQKGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::KDataType,
                              typename Problem::QDataType,
                              float,
                              256,
                              SlaAttnMmacQKGemmShape<Problem>>;

template <typename Problem>
using SlaAttnMmacQKChunkGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::kBlockM,
                                             SlaAttnMmacQKChunkConfig::kN,
                                             SlaAttnMmacQKChunkConfig::kK>,
                           typename SlaAttnMmacQKChunkConfig::BlockWarps,
                           typename SlaAttnMmacQKChunkConfig::WarpTile>;

template <typename Problem>
using SlaAttnMmacQKChunkGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::KDataType,
                              typename Problem::QDataType,
                              float,
                              256,
                              SlaAttnMmacQKChunkGemmShape<Problem>>;

template <typename Problem>
using SlaAttnMmacQKHalfGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<32,
                                             SlaAttnMmacQKHalfConfig::kN,
                                             SlaAttnMmacQKHalfConfig::kK>,
                           typename SlaAttnMmacQKHalfConfig::BlockWarps,
                           typename SlaAttnMmacQKHalfConfig::WarpTile>;

template <typename Problem>
using SlaAttnMmacQKHalfGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::KDataType,
                              typename Problem::QDataType,
                              float,
                              256,
                              SlaAttnMmacQKHalfGemmShape<Problem>>;

template <typename Problem>
using SlaAttnMmacDVGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<SlaAttnMmacDVConfig::kM,
                                             SlaAttnMmacDVConfig::kN,
                                             Problem::kBlockM>,
                           typename SlaAttnMmacDVConfig::BlockWarps,
                           typename SlaAttnMmacDVConfig::WarpTile>;

template <typename Problem>
using SlaAttnMmacDVGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::OGradDataType,
                              float,
                              256,
                              SlaAttnMmacDVGemmShape<Problem>>;

template <typename Problem>
using SlaAttnMmacDVK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<SlaAttnMmacDVConfig::kM,
                                             SlaAttnMmacDVConfig::kN,
                                             32>,
                           typename SlaAttnMmacDVConfig::BlockWarps,
                           typename SlaAttnMmacDVConfig::WarpTile>;

template <typename Problem>
using SlaAttnMmacDVK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::OGradDataType,
                              float,
                              256,
                              SlaAttnMmacDVK32GemmShape<Problem>>;

template <typename Problem>
using SlaAttnMmacDPGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::kBlockM,
                                             SlaAttnMmacDPConfig::kN,
                                             SlaAttnMmacDPConfig::kK>,
                           typename SlaAttnMmacDPConfig::BlockWarps,
                           typename SlaAttnMmacDPConfig::WarpTile>;

template <typename Problem>
using SlaAttnMmacDPGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::VDataType,
                              typename Problem::OGradDataType,
                              float,
                              256,
                              SlaAttnMmacDPGemmShape<Problem>>;

template <typename Problem>
using SlaAttnMmacDPK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<Problem::kBlockM,
                                             SlaAttnMmacDPK32Config::kN,
                                             SlaAttnMmacDPK32Config::kK>,
                           typename SlaAttnMmacDPK32Config::BlockWarps,
                           typename SlaAttnMmacDPK32Config::WarpTile>;

template <typename Problem>
using SlaAttnMmacDPK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::VDataType,
                              typename Problem::OGradDataType,
                              float,
                              256,
                              SlaAttnMmacDPK32GemmShape<Problem>>;

template <typename Problem>
using SlaAttnMmacDKGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<SlaAttnMmacDKConfig::kM,
                                             SlaAttnMmacDKConfig::kN,
                                             Problem::kBlockM>,
                           typename SlaAttnMmacDKConfig::BlockWarps,
                           typename SlaAttnMmacDKConfig::WarpTile>;

template <typename Problem>
using SlaAttnMmacDKGemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::QDataType,
                              float,
                              256,
                              SlaAttnMmacDKGemmShape<Problem>>;

template <typename Problem>
using SlaAttnMmacDQK32GemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<SlaAttnMmacDVConfig::kM,
                                             SlaAttnMmacDVConfig::kN,
                                             32>,
                           typename SlaAttnMmacDPK32Config::BlockWarps,
                           typename SlaAttnMmacDPK32Config::WarpTile>;

template <typename Problem>
using SlaAttnMmacDQK32GemmProblem =
    ck_tile::BlockGemmProblem<typename Problem::QDataType,
                              typename Problem::KDataType,
                              float,
                              256,
                              SlaAttnMmacDQK32GemmShape<Problem>>;

} // namespace ck_tile
