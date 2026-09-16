// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_fused_linear_pipeline.hpp"

namespace sla {

template <ActType Type, typename Policy = SlaLinearDefaultPolicy>
struct SlaLinearKFeatureKernel
{
    using DataType = typename Policy::Problem::DataType;
    struct Kargs
    {
        const DataType* k;
        DataType* k_feat;
        float* ksum;
        int L;
        int BH;
    };

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        constexpr int rows_per_block =
            Policy::kKRows * Policy::kFeatureTilesPerBlock;
        return dim3((arg.L + rows_per_block - 1) / rows_per_block, 2, arg.BH);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(Policy::kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        SlaLinearKFeatureComputePipeline<Type, Policy>{}(arg);
    }
};

template <int SplitL = LinearPolicy::kKvSplitL,
          typename Policy = SlaLinearDefaultPolicy>
struct SlaLinearKvSplitKernel
{
    using DataType = typename Policy::Problem::DataType;
    struct Kargs
    {
        const DataType* k_feat;
        const DataType* v;
        float* kv;
        int L;
        int BH;
    };

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(4, SplitL, arg.BH);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(Policy::kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int D = Policy::kHeadDim;
        const int bh = blockIdx.z;
        const int64_t base = static_cast<int64_t>(bh) * arg.L * D;
        auto k_feat_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                arg.k_feat + base,
                ck_tile::make_tuple(D, arg.L),
                ck_tile::make_tuple(1, D),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
        auto v_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                arg.v + base,
                ck_tile::make_tuple(D, arg.L),
                ck_tile::make_tuple(1, D),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
        k_feat_view.init_raw();
        v_view.init_raw();
        SlaLinearKvSplitPipeline<SplitL, Policy>{}(arg, k_feat_view, v_view);
    }
};

template <typename DataType>
struct SlaLinearKvGemmKernel
{
    using GemmKernel = typename SlaLinearKvGemmPolicy<DataType>::Kernel;
    struct Kargs
    {
        const DataType* k_feat;
        const DataType* v;
        DataType* kv;
        int L;
        int BH;
    };

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        const auto gemm_grid = GemmKernel::GridSize(128, 128, 1);
        return dim3(gemm_grid.x, gemm_grid.y, arg.BH);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return GemmKernel::BlockSize(); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        SlaLinearKvGemmPipeline<DataType>{}(arg);
    }
};

template <typename DataType>
struct SlaLinearKvProjectionKernel
{
    using GemmKernel = typename SlaLinearKvProjectionGemmPolicy<DataType>::Kernel;
    struct Kargs
    {
        const DataType* kv;
        const DataType* weight;
        DataType* kv_projected;
        int BH;
    };

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        const auto gemm_grid = GemmKernel::GridSize(128, 128, 1);
        return dim3(gemm_grid.x, gemm_grid.y, arg.BH);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return GemmKernel::BlockSize(); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        SlaLinearKvProjectionPipeline<DataType>{}(arg);
    }
};

template <ActType Type,
          int SplitL = LinearPolicy::kKvSplitL,
          typename Policy = SlaLinearDefaultPolicy>
struct SlaLinearKSplitKernel
{
    using DataType = typename Policy::Problem::DataType;
    struct Kargs
    {
        const DataType* k;
        const DataType* v;
        float* partial_kv;
        float* partial_ksum;
        int L;
        int BH;
    };

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(4, SplitL, arg.BH);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(Policy::kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int D = Policy::kHeadDim;
        const int bh = blockIdx.z;
        const int64_t base = static_cast<int64_t>(bh) * arg.L * D;
        auto v_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                arg.v + base,
                ck_tile::make_tuple(D, arg.L),
                ck_tile::make_tuple(1, D),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
        v_view.init_raw();
        SlaLinearKSplitPipeline<Type, SplitL, Policy>{}(arg, v_view);
    }
};

template <typename DataType, int SplitL = LinearPolicy::kKvSplitL>
struct SlaLinearKSplitReduceKernel
{
    struct Kargs
    {
        const float* partial_kv;
        const float* partial_ksum;
        DataType* kv;
        float* ksum;
        int BH;
    };

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        constexpr int D = LinearPolicy::kHeadDim;
        constexpr int values_per_head = D * D + D;
        return dim3((values_per_head + LinearPolicy::kBlockSize - 1) /
                        LinearPolicy::kBlockSize,
                    arg.BH);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(LinearPolicy::kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        SlaLinearKSplitReducePipeline<DataType, SplitL>{}(arg);
    }
};

template <ActType Type, typename QPolicy = LinearPolicy>
struct SlaLinearQFusedKernel
{
    using DataType = typename QPolicy::Problem::DataType;
    struct Kargs
    {
        const DataType* q;
        const DataType* kv;
        const float* ksum;
        const DataType* bias;
        const DataType* os;
        DataType* output;
        int L;
        int BH;
    };

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3((arg.L + QPolicy::kQRows - 1) / QPolicy::kQRows, arg.BH);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(QPolicy::kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int D = QPolicy::kHeadDim;
        const int bh = blockIdx.y;
        auto kv_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                arg.kv + static_cast<int64_t>(bh) * D * D,
                ck_tile::make_tuple(D, D),
                ck_tile::make_tuple(1, D),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
        kv_view.init_raw();
        SlaLinearQFusedPipeline<Type, QPolicy>{}(arg, kv_view);
    }
};

} // namespace sla
