// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_l2norm_pipeline.hpp"
#include "ck_tile/host.hpp"

struct GdnL2NormFwdKernelArg
{
    const void* x;
    void* y;
    float* rstd;
    ck_tile::long_index_t rows;
    ck_tile::index_t dim;
    float epsilon;
};

template <typename Policy, bool SmallD>
struct GdnL2NormFwdKernel
{
    using Kargs = GdnL2NormFwdKernelArg;
    using Pipeline = ck_tile::GdnL2NormFwdPipeline<Policy>;
    using XDataType = typename Policy::Problem::XDataType;
    using YDataType = typename Policy::Problem::YDataType;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* x,
                                                  void* y,
                                                  float* rstd,
                                                  ck_tile::long_index_t rows,
                                                  ck_tile::index_t dim,
                                                  float epsilon)
    {
        return Kargs{x, y, rstd, rows, dim, epsilon};
    }

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& args)
    {
        if constexpr(SmallD)
        {
            return dim3(static_cast<unsigned int>(
                (args.rows + Policy::kRowsPerBlock - 1) / Policy::kRowsPerBlock));
        }
        else
        {
            return dim3(static_cast<unsigned int>(args.rows));
        }
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(Policy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        __shared__ float shared[Policy::kNumWaves];
        if constexpr(SmallD)
        {
            Pipeline{}.RunSmall(reinterpret_cast<const XDataType*>(args.x),
                                reinterpret_cast<YDataType*>(args.y),
                                args.rstd,
                                args.rows,
                                args.dim,
                                args.epsilon);
        }
        else
        {
            Pipeline{}.RunLarge(reinterpret_cast<const XDataType*>(args.x),
                                reinterpret_cast<YDataType*>(args.y),
                                args.rstd,
                                args.rows,
                                args.dim,
                                args.epsilon,
                                shared);
        }
    }
};
