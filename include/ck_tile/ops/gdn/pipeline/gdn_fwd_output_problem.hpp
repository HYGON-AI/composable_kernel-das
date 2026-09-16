// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_config.hpp"
#include <ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp>

namespace gdn {

template <typename DataType>
CK_TILE_DEVICE DataType gdn_output_cast_from_float(float value)
{
    return ck_tile::type_convert<DataType>(value);
}

template <typename DataType>
struct GdnOutputFwdKargs
{
    const DataType* q;
    const DataType* k;
    const DataType* v_new;
    const DataType* h;
    const float* g;
    const int64_t* cu_seqlens;
    const int64_t* chunk_indices;
    DataType* o;
    float scale;
    ck_tile::index_t total_tokens;
    ck_tile::index_t num_chunks;
    ck_tile::index_t num_sequences;
    ck_tile::index_t num_qk_heads;
    ck_tile::index_t num_value_heads;
    bool is_varlen;
};

template <typename DataType_,
          ck_tile::index_t GroupSize_,
          ck_tile::index_t ValueSplit_ = 1,
          bool PreshuffledH_ = false,
          bool TwoRows_ = false>
struct GdnOutputFwdProblem
{
    using Config   = GdnOutputFwdConfig;
    using DataType = DataType_;
    using Kargs    = GdnOutputFwdKargs<DataType>;

    static constexpr ck_tile::index_t kGroupSize = GroupSize_;
    static constexpr ck_tile::index_t kValueSplit = ValueSplit_;
    static constexpr bool kPreshuffledH = PreshuffledH_;
    static constexpr bool kTwoRows = TwoRows_;
    static constexpr ck_tile::index_t kChunkSize = Config::kChunkSize;
    static constexpr ck_tile::index_t kHeadDim   = Config::kHeadDim;
    static constexpr ck_tile::index_t kValueDim  = Config::kValueDim;
    static constexpr ck_tile::index_t kRowTile   = Config::kRowTile;
    static constexpr ck_tile::index_t kValueTile =
        kTwoRows ? 32 : (kGroupSize == 1 ? 64 : 32);
    static constexpr ck_tile::index_t kBlockSize = Config::kBlockSize;

    static_assert(kValueSplit == 1 || kValueSplit == 2,
                  "fwd_output supports one or two value-dimension grid splits");
    static_assert(kGroupSize == 4 || kGroupSize == 1 || kValueSplit == 1,
                  "value-dimension splitting is only implemented for GroupSize=4");

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<M, N, K>,
        ck_tile::sequence<(M >= 64 ? 4 : 2), 2, 1>,
        ck_tile::sequence<16, (N >= 64 ? 32 : 16), K>>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    struct GemmProblem
    {
        using ADataType      = DataType;
        using BDataType      = DataType;
        using CDataType      = float;
        using BlockGemmShape = GemmShape<M, N, K>;
        static constexpr ck_tile::index_t kBlockSize =
            GdnOutputFwdProblem::kBlockSize;
    };
};

} // namespace gdn
