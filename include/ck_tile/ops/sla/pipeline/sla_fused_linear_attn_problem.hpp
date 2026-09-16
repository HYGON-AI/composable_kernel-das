// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include <ck_tile/core.hpp>

namespace sla {

enum class ActType {
    NONE,
    SOFTMAX,
    RELU,
    ELU
};

template <typename DataType_>
struct SlaLinearKargs
{
    using DataType = DataType_;

    const DataType* q;
    const float* num;
    const float* ksum;
    DataType* out;

    ck_tile::index_t L;
};

template <typename DataType_>
struct SlaFusedLinearAttnProblem
{
    using DataType = DataType_;
    using Kargs = SlaLinearKargs<DataType>;

    static constexpr ck_tile::index_t kHeadDim = 128;
    static constexpr ck_tile::index_t kBlockSize = 256;
    static constexpr ck_tile::index_t kWaveSize = 64;
};

} // namespace sla
