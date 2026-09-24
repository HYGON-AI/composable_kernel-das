// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_sparse_map_pipeline.hpp"

namespace sla {

template <typename DataType>
struct SlaSparseMapMeanKKernel
{
    using Problem = SlaSparseMapProblem<DataType>;
    using Policy = SlaSparseMapPolicy<Problem>;
    using Pipeline = SlaSparseMapMeanKPipeline<Policy>;
    using Kargs = typename Problem::Kargs;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        const ck_tile::index_t vec_idx = blockIdx.x;
        const ck_tile::index_t bh = blockIdx.y;
        const ck_tile::index_t tid = ck_tile::get_thread_local_1d_id();
        __shared__ float scratch[8 * Policy::kBlockSize];
        Pipeline{}(args.k, args.k_mean, args.Lk, bh, vec_idx, tid, scratch);
    }
};

template <typename DataType>
struct SlaSparseMapPoolQKernel
{
    using Problem = SlaSparseMapProblem<DataType>;
    using Policy = SlaSparseMapPolicy<Problem>;
    using Pipeline = SlaSparseMapPoolQPipeline<Policy>;
    using Kargs = typename Problem::Kargs;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        const ck_tile::index_t blk_idx = blockIdx.x;
        const ck_tile::index_t bh = blockIdx.y;
        const ck_tile::index_t tid = ck_tile::get_thread_local_1d_id();

        const ck_tile::index_t l_start = blk_idx * args.BLKQ;
        const ck_tile::index_t l_end = min(l_start + args.BLKQ, args.Lq);

        const DataType* q_ptr = args.q + bh * args.Lq * Policy::kHeadDim;
        DataType* pooled_q_ptr = args.pooled_q + (bh * args.Lq_blocks + blk_idx) * Policy::kHeadDim;

        auto q_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            q_ptr,
            ck_tile::make_tuple(args.Lq, Policy::kHeadDim),
            ck_tile::make_tuple(Policy::kHeadDim, 1),
            ck_tile::number<1>{},
            ck_tile::number<1>{});

        auto pooled_q_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            pooled_q_ptr,
            ck_tile::make_tuple(Policy::kHeadDim),
            ck_tile::make_tuple(1),
            ck_tile::number<1>{},
            ck_tile::number<1>{});

        Pipeline{}(q_view, pooled_q_view, l_start, l_end, tid);
    }
};

template <typename DataType>
struct SlaSparseMapPoolKKernel
{
    using Problem = SlaSparseMapProblem<DataType>;
    using Policy = SlaSparseMapPolicy<Problem>;
    using Pipeline = SlaSparseMapPoolKPipeline<Policy>;
    using Kargs = typename Problem::Kargs;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        const ck_tile::index_t blk_idx = blockIdx.x;
        const ck_tile::index_t bh = blockIdx.y;
        const ck_tile::index_t tid = ck_tile::get_thread_local_1d_id();

        const ck_tile::index_t l_start = blk_idx * args.BLKK;
        const ck_tile::index_t l_end = min(l_start + args.BLKK, args.Lk);

        const DataType* k_ptr = args.k + bh * args.Lk * Policy::kHeadDim;
        DataType* pooled_k_ptr = args.pooled_k + (bh * args.Lk_blocks + blk_idx) * Policy::kHeadDim;

        auto k_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            k_ptr,
            ck_tile::make_tuple(args.Lk, Policy::kHeadDim),
            ck_tile::make_tuple(Policy::kHeadDim, 1),
            ck_tile::number<1>{},
            ck_tile::number<1>{});

        auto pooled_k_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            pooled_k_ptr,
            ck_tile::make_tuple(Policy::kHeadDim),
            ck_tile::make_tuple(1),
            ck_tile::number<1>{},
            ck_tile::number<1>{});

        Pipeline{}(k_view, pooled_k_view, l_start, l_end, tid);
    }
};

template <typename DataType>
struct SlaSparseMapScoreKernel
{
    using Problem = SlaSparseMapProblem<DataType>;
    using Policy = SlaSparseMapPolicy<Problem>;
    using Pipeline = SlaSparseMapScorePipeline<Policy>;
    using Kargs = typename Problem::Kargs;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args, float* pooled_score) const
    {
        const ck_tile::index_t k_blk = blockIdx.x;
        const ck_tile::index_t q_blk = blockIdx.y;
        const ck_tile::index_t bh = blockIdx.z;
        const ck_tile::index_t tid = ck_tile::get_thread_local_1d_id();

        Pipeline{}(args.pooled_q, args.pooled_k, pooled_score, args.Lq_blocks, args.Lk_blocks, bh, q_blk, k_blk, tid);
    }
};

template <typename DataType>
struct SlaSparseMapTopkKernel
{
    using Problem = SlaSparseMapProblem<DataType>;
    using Policy = SlaSparseMapPolicy<Problem>;
    using Pipeline = SlaSparseMapTopkPipeline<Policy>;
    using Kargs = typename Problem::Kargs;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args, const float* pooled_score, int64_t* lut, int8_t* sparse_map, int topk) const
    {
        const ck_tile::index_t q_blk = blockIdx.x;
        const ck_tile::index_t bh = blockIdx.y;
        const ck_tile::index_t tid = ck_tile::get_thread_local_1d_id();

        Pipeline{}(pooled_score, lut, sparse_map, args.Lq_blocks, args.Lk_blocks, topk, bh, q_blk, tid);
    }
};

template <typename DataType>
struct SlaSparseMapScoreRowKernel
{
    using Problem = SlaSparseMapProblem<DataType>;
    using Policy = SlaSparseMapPolicy<Problem>;
    using Pipeline = SlaSparseMapScoreRowPipeline<Policy>;
    using Kargs = typename Problem::Kargs;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args,
                                   DataType* pooled_score) const
    {
        const ck_tile::index_t q_blk = blockIdx.x;
        const ck_tile::index_t bh = blockIdx.y;
        const ck_tile::index_t tid = ck_tile::get_thread_local_1d_id();
        Pipeline{}(args.pooled_q,
                   args.pooled_k,
                   pooled_score,
                   args.Lq_blocks,
                   args.Lk_blocks,
                   bh,
                   q_blk,
                   tid);
    }
};

template <typename DataType>
struct SlaSparseMapScatterKernel
{
    using Problem = SlaSparseMapProblem<DataType>;
    using Policy = SlaSparseMapPolicy<Problem>;
    using Pipeline = SlaSparseMapScatterPipeline<Policy>;
    using Kargs = typename Problem::Kargs;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args,
                                   const int32_t* lut_i32,
                                   int64_t* lut,
                                   int8_t* sparse_map,
                                   int topk) const
    {
        const ck_tile::index_t q_blk = blockIdx.x;
        const ck_tile::index_t bh = blockIdx.y;
        const ck_tile::index_t tid = ck_tile::get_thread_local_1d_id();
        Pipeline{}(lut_i32,
                   lut,
                   sparse_map,
                   args.Lq_blocks,
                   args.Lk_blocks,
                   topk,
                   bh,
                   q_blk,
                   tid);
    }
};

template <typename DataType, ck_tile::index_t PaddedCandidates>
struct SlaSparseMapScoreRadixKernel
{
    using Problem = SlaSparseMapProblem<DataType>;
    using Policy = SlaSparseMapPolicy<Problem>;
    using Pipeline = SlaSparseMapScoreRadixPipeline<Policy, PaddedCandidates>;
    using Kargs = typename Problem::Kargs;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args,
                                   int64_t* lut,
                                   int8_t* sparse_map,
                                   int topk) const
    {
        const ck_tile::index_t q_blk = blockIdx.x;
        const ck_tile::index_t bh = blockIdx.y;
        const ck_tile::index_t tid = ck_tile::get_thread_local_1d_id();
        __shared__ uint16_t keys[PaddedCandidates];
        __shared__ int32_t histogram[256];
        __shared__ int32_t metadata[8];
        Pipeline{}(args.pooled_q,
                   args.pooled_k,
                   lut,
                   sparse_map,
                   args.Lq_blocks,
                   args.Lk_blocks,
                   topk,
                   bh,
                   q_blk,
                   tid,
                   keys,
                   histogram,
                   metadata);
    }
};

} // namespace sla
