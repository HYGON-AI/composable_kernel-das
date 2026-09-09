// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <ck_tile/core.hpp>

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename I,
          typename W,
          typename O,
          typename ST,
          typename SW,
          typename SQ,
          typename KW,
          typename ZP,
          typename BlockTile0_, // seq<b_token, b_interm, b_hidden, b_down>
          typename WarpPerBlock_,
          typename WarpTile0_, // seq<*,*,*>, used to select mfma
          typename BlockTile1_,
          typename WarpTile1_,
          typename GmemLoadVectorLengths_,
          typename SmemStoreVectorLengths_,
          typename SmemLoadVectorLengths_,
          ck_tile::index_t OGmemStoreVectorLength_,
          ck_tile::index_t Gemm0NInterleave_,
          ck_tile::index_t Gemm1NInterleave_,
          ck_tile::index_t Activation_ = 1, // 0: Gelu 1: Silu
          ck_tile::index_t GateOnly_   = 0,
          ck_tile::index_t FusedQuant_ = 0,
          ck_tile::index_t QT_block_n_ = 1,
          ck_tile::index_t QT_block_k_ = 1,
          bool swizzled_               = true,
          bool preshuffle_             = false,
          ck_tile::index_t IntermediateTilesPerBlock_ = 1,
          bool DebugWriterCount_ = false>
struct fmoe_ // traits, ugly name, only used for internal
{
    using TypeConfig = FusedMoeGemmTypeConfig<I, W, O, ST, SW, SQ, KW, ZP>;

    using ADataType            = ck_tile::remove_cvref_t<typename TypeConfig::ADataType>;
    using GDataType            = ck_tile::remove_cvref_t<typename TypeConfig::GDataType>;
    using DDataType            = ck_tile::remove_cvref_t<typename TypeConfig::DDataType>;
    using AccDataType          = ck_tile::remove_cvref_t<typename TypeConfig::AccDataType>;
    using ODataType            = ck_tile::remove_cvref_t<typename TypeConfig::ODataType>;
    using AScaleDataType       = ck_tile::remove_cvref_t<typename TypeConfig::AScaleDataType>;
    using GScaleDataType       = ck_tile::remove_cvref_t<typename TypeConfig::GScaleDataType>;
    using DScaleDataType       = ck_tile::remove_cvref_t<typename TypeConfig::DScaleDataType>;
    using GZeroPointDataType   = ck_tile::remove_cvref_t<typename TypeConfig::GZeroPointDataType>;
    using DZeroPointDataType   = ck_tile::remove_cvref_t<typename TypeConfig::DZeroPointDataType>;
    using YSmoothScaleDataType = ck_tile::remove_cvref_t<typename TypeConfig::YSmoothScaleDataType>;
    using TopkWeightDataType   = ck_tile::remove_cvref_t<typename TypeConfig::TopkWeightDataType>;
    using IndexDataType        = ck_tile::remove_cvref_t<typename TypeConfig::IndexDataType>;

    // static constexpr ck_tile::index_t BT_ = BlockTIle_::at(ck_tile::number<0>{}); // block token
    // static constexpr ck_tile::index_t BI_ =
    //     BlockTIle_::at(ck_tile::number<1>{}); // block intermediate
    // static constexpr ck_tile::index_t BH_ = BlockTIle_::at(ck_tile::number<2>{}); // block hidden
    // static constexpr ck_tile::index_t BD_ = BlockTIle_::at(ck_tile::number<3>{}); // block down

    using BlockTile_0    = ck_tile::remove_cvref_t<BlockTile0_>;
    using WarpPerBlock_0 = ck_tile::remove_cvref_t<WarpPerBlock_>;
    using WarpTile_0     = ck_tile::remove_cvref_t<WarpTile0_>;

    using BlockTile_1    = ck_tile::remove_cvref_t<BlockTile1_>;
    using WarpPerBlock_1 = ck_tile::remove_cvref_t<WarpPerBlock_>;
    using WarpTile_1     = ck_tile::remove_cvref_t<WarpTile1_>;

    using GmemLoadVectorLengths  = ck_tile::remove_cvref_t<GmemLoadVectorLengths_>;
    using SmemStoreVectorLengths = ck_tile::remove_cvref_t<SmemStoreVectorLengths_>;
    using SmemLoadVectorLengths  = ck_tile::remove_cvref_t<SmemLoadVectorLengths_>;

    static constexpr ck_tile::index_t OGmemStoreVectorLength = OGmemStoreVectorLength_;
    static constexpr ck_tile::index_t Gemm0NInterleave       = Gemm0NInterleave_;
    static constexpr ck_tile::index_t Gemm1NInterleave       = Gemm1NInterleave_;

    static constexpr ck_tile::index_t Activation = Activation_; // 0: Gelu 1: Silu
    static constexpr ck_tile::index_t GateOnly   = GateOnly_;
    static constexpr ck_tile::index_t FusedQuant = FusedQuant_;
    static constexpr ck_tile::index_t QuantBlockSizeN = QT_block_n_;
    static constexpr ck_tile::index_t QuantBlockSizeK = QT_block_k_;
    static constexpr bool swizzled               = swizzled_;
    static constexpr bool preshuffle             = preshuffle_;
    static constexpr ck_tile::index_t IntermediateTilesPerBlock =
        IntermediateTilesPerBlock_;
    static constexpr bool DebugWriterCount = DebugWriterCount_;
};
