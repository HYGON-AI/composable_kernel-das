// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_pipeline.hpp"

namespace gdn {

template <typename DataType,
          ck_tile::index_t GroupSize = 1,
          ck_tile::index_t ValueSplit = 1,
          bool PreshuffledH = false,
          bool TwoRows = false,
          bool PairedEpilogue = false>
struct GdnOutputTiledMmacFwdKernel
{
    using Problem =
        GdnOutputFwdProblem<DataType, GroupSize, ValueSplit, PreshuffledH, TwoRows, PairedEpilogue>;
    using Pipeline = GdnOutputFwdPipeline<Problem>;
    using Kargs    = typename Problem::Kargs;

    static constexpr ck_tile::index_t kRowTile   = Problem::kRowTile;
    static constexpr ck_tile::index_t kValueTile = Problem::kValueTile;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;
    static constexpr ck_tile::index_t kChunkSize = Problem::kChunkSize;
    static constexpr ck_tile::index_t kValueSplit = Problem::kValueSplit;

    CK_TILE_DEVICE void operator()(Kargs args) const { Pipeline{}(args); }
};

} // namespace gdn
