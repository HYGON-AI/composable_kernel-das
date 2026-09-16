// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/jenga/pipeline/jenga_ck_tile_mask_builder_policy.hpp"
#include "ck_tile/ops/reduce.hpp"
#include <cstdint>

template <typename PipelinePolicy, bool IsQuery>
struct JengaMaskPoolPipeline
{
    using Problem = typename PipelinePolicy::Problem;
    using DataType = typename Problem::DataType;

    static constexpr int kRows = IsQuery ? PipelinePolicy::kBlockM : PipelinePolicy::kBlockN;
    static constexpr int kHeadDim = PipelinePolicy::kHeadDim;

    CK_TILE_DEVICE void operator()(const uint16_t* x,
                                   float* pooled,
                                   int H,
                                   int N,
                                   int num_blocks) const
    {
        (void)H;
        const int g = blockIdx.x;
        const int block_id = g % num_blocks;
        const int bh = g / num_blocks;
        const int start = block_id * kRows;
        const int remain = N - start;
        const int valid_len = remain < kRows ? (remain > 0 ? remain : 0) : kRows;

        auto x_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            reinterpret_cast<const DataType*>(x) + static_cast<int64_t>(bh) * N * kHeadDim,
            ck_tile::make_tuple(N, ck_tile::number<kHeadDim>{}),
            ck_tile::make_tuple(ck_tile::number<kHeadDim>{}, ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});
        auto x_pad = ck_tile::pad_tensor_view(
            x_view,
            ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<kHeadDim>{}),
            ck_tile::sequence<true, false>{});
        auto x_win = ck_tile::make_tile_window(
            x_pad,
            ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<kHeadDim>{}),
            ck_tile::multi_index<2>{start, 0},
            IsQuery ? PipelinePolicy::MakeQPoolInputDistribution()
                    : PipelinePolicy::MakeKPoolInputDistribution());

        auto x_tile = ck_tile::cast_tile<float>(ck_tile::load_tile(x_win));
        ck_tile::sweep_tile(x_tile, [&](auto idx) {
            const auto coord = ck_tile::get_x_indices_from_distributed_indices(
                x_tile.get_tile_distribution(), idx);
            const int row = coord.at(ck_tile::number<0>{});
            if (row >= valid_len) {
                x_tile(idx) = 0.0f;
            }
        });

        const auto f_sum = [](auto a, auto b) { return a + b; };
        auto sum = ck_tile::block_tile_reduce<float>(
            x_tile, ck_tile::sequence<0>{}, f_sum, 0.0f);
        ck_tile::block_tile_reduce_sync(sum, f_sum, ck_tile::bool_constant<false>{});
        ck_tile::tile_elementwise_inout(
            [&](auto& v) {
                v = valid_len > 0 ? v / static_cast<float>(valid_len) : 0.0f;
            },
            sum);

        auto out_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            pooled + static_cast<int64_t>(g) * kHeadDim,
            ck_tile::make_tuple(ck_tile::number<kHeadDim>{}),
            ck_tile::make_tuple(ck_tile::number<1>{}),
            ck_tile::number<1>{},
            ck_tile::number<1>{});
        auto out_win = ck_tile::make_tile_window(
            out_view,
            ck_tile::make_tuple(ck_tile::number<kHeadDim>{}),
            ck_tile::multi_index<1>{0},
            sum.get_tile_distribution());
        ck_tile::store_tile(out_win, sum);
    }
};

template <typename PipelinePolicy>
struct JengaMaskScorePipeline
{
    using Problem = typename PipelinePolicy::Problem;
    static constexpr int kHeadDim = PipelinePolicy::kHeadDim;
    static constexpr int kScoreTile = PipelinePolicy::kScoreTile;

    CK_TILE_DEVICE void operator()(const float* q_pool,
                                   const float* k_pool,
                                   float* scores,
                                   int H,
                                   int num_query_blocks,
                                   int num_blocks,
                                   int text_start_block) const
    {
        (void)H;
        const int row = blockIdx.x;
        const int ktile = blockIdx.y;
        const int qb = row % num_query_blocks;
        const int bh = row / num_query_blocks;
        const int k_begin = ktile * kScoreTile;

        auto k_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            k_pool + static_cast<int64_t>(bh) * num_blocks * kHeadDim,
            ck_tile::make_tuple(num_blocks, ck_tile::number<kHeadDim>{}),
            ck_tile::make_tuple(ck_tile::number<kHeadDim>{}, ck_tile::number<1>{}),
            ck_tile::number<1>{},
            ck_tile::number<1>{});
        auto k_pad = ck_tile::pad_tensor_view(
            k_view,
            ck_tile::make_tuple(ck_tile::number<kScoreTile>{}, ck_tile::number<kHeadDim>{}),
            ck_tile::sequence<true, false>{});
        auto k_win = ck_tile::make_tile_window(
            k_pad,
            ck_tile::make_tuple(ck_tile::number<kScoreTile>{}, ck_tile::number<kHeadDim>{}),
            ck_tile::multi_index<2>{k_begin, 0},
            PipelinePolicy::MakeScoreInputDistribution());
        auto k_tile = ck_tile::load_tile(k_win);

        auto prod = ck_tile::make_static_distributed_tensor<float>(k_tile.get_tile_distribution());
        const float* q_row =
            q_pool + (static_cast<int64_t>(bh) * num_query_blocks + qb) * kHeadDim;
        ck_tile::sweep_tile(prod, [&](auto idx) {
            const auto coord = ck_tile::get_x_indices_from_distributed_indices(
                prod.get_tile_distribution(), idx);
            const int kb_local = coord.at(ck_tile::number<0>{});
            const int d = coord.at(ck_tile::number<1>{});
            const int kb = k_begin + kb_local;
            prod(idx) = kb < text_start_block
                            ? ck_tile::type_convert<float>(k_tile[idx]) * q_row[d]
                            : 0.0f;
        });

        const auto f_sum = [](auto a, auto b) { return a + b; };
        auto dot = ck_tile::block_tile_reduce<float>(
            prod, ck_tile::sequence<1>{}, f_sum, 0.0f);
        ck_tile::block_tile_reduce_sync(dot, f_sum, ck_tile::bool_constant<false>{});
        const float scale = rsqrtf(static_cast<float>(kHeadDim));
        ck_tile::tile_elementwise_inout([&](auto& v) { v *= scale; }, dot);

        auto score_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            scores + static_cast<int64_t>(row) * num_blocks,
            ck_tile::make_tuple(num_blocks),
            ck_tile::make_tuple(ck_tile::number<1>{}),
            ck_tile::number<1>{},
            ck_tile::number<1>{});
        auto score_pad = ck_tile::pad_tensor_view(
            score_view,
            ck_tile::make_tuple(ck_tile::number<kScoreTile>{}),
            ck_tile::sequence<true>{});
        auto score_win = ck_tile::make_tile_window(
            score_pad,
            ck_tile::make_tuple(ck_tile::number<kScoreTile>{}),
            ck_tile::multi_index<1>{k_begin},
            dot.get_tile_distribution());
        ck_tile::store_tile(score_win, dot);
    }
};

template <typename PipelinePolicy>
struct JengaMaskSelectPipeline
{
    static constexpr int kScoreTile = PipelinePolicy::kScoreTile;

    CK_TILE_DEVICE void operator()(float* scores,
                                   const bool* neighbor_mask,
                                   bool* out,
                                   int rows,
                                   int num_query_blocks,
                                   int num_blocks,
                                   int text_start_block,
                                   int top_k,
                                   float prob_threshold,
                                   int candidate_k,
                                   int text_blocks,
                                   int first_frame_blocks,
                                   bool has_neighbors) const
    {
        const int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        const int qb = row % num_query_blocks;
        bool* out_row = out + static_cast<int64_t>(row) * num_blocks;
        float* score_row = scores + static_cast<int64_t>(row) * num_blocks;

        auto score_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            score_row,
            ck_tile::make_tuple(ck_tile::number<1>{}, num_blocks),
            ck_tile::make_tuple(num_blocks, ck_tile::number<1>{}),
            ck_tile::number<1>{},
            ck_tile::number<1>{});
        auto score_pad = ck_tile::pad_tensor_view(
            score_view,
            ck_tile::make_tuple(ck_tile::number<4>{}, ck_tile::number<kScoreTile>{}),
            ck_tile::sequence<true, true>{});

        const auto f_max = [](auto a, auto b) { return ck_tile::max(a, b); };
        const auto f_sum = [](auto a, auto b) { return a + b; };

        float row_max = -ck_tile::numeric<float>::infinity();
        for (int k_begin = 0; k_begin < text_start_block; k_begin += kScoreTile) {
            auto win = ck_tile::make_tile_window(
                score_pad,
                ck_tile::make_tuple(ck_tile::number<4>{}, ck_tile::number<kScoreTile>{}),
                ck_tile::multi_index<2>{0, k_begin},
                PipelinePolicy::MakeScoreRowDistribution());
            auto tile = ck_tile::load_tile(win);
            ck_tile::sweep_tile(tile, [&](auto idx) {
                const auto coord = ck_tile::get_x_indices_from_distributed_indices(
                    tile.get_tile_distribution(), idx);
                const int r = coord.at(ck_tile::number<0>{});
                const int kb = k_begin + coord.at(ck_tile::number<1>{});
                if (r != 0 || kb >= text_start_block) {
                    tile(idx) = -ck_tile::numeric<float>::infinity();
                }
            });
            auto local = ck_tile::block_tile_reduce<float>(
                tile, ck_tile::sequence<1>{}, f_max, -ck_tile::numeric<float>::infinity());
            ck_tile::block_tile_reduce_sync(local, f_max, ck_tile::bool_constant<false>{});
            ck_tile::sweep_tile(local, [&](auto idx) {
                row_max = ck_tile::max(row_max, local[idx]);
            });
        }

        float denom = 0.0f;
        for (int k_begin = 0; k_begin < text_start_block; k_begin += kScoreTile) {
            auto win = ck_tile::make_tile_window(
                score_pad,
                ck_tile::make_tuple(ck_tile::number<4>{}, ck_tile::number<kScoreTile>{}),
                ck_tile::multi_index<2>{0, k_begin},
                PipelinePolicy::MakeScoreRowDistribution());
            auto tile = ck_tile::load_tile(win);
            ck_tile::sweep_tile(tile, [&](auto idx) {
                const auto coord = ck_tile::get_x_indices_from_distributed_indices(
                    tile.get_tile_distribution(), idx);
                const int r = coord.at(ck_tile::number<0>{});
                const int kb = k_begin + coord.at(ck_tile::number<1>{});
                tile(idx) = (r == 0 && kb < text_start_block) ? expf(tile[idx] - row_max) : 0.0f;
            });
            auto local = ck_tile::block_tile_reduce<float>(
                tile, ck_tile::sequence<1>{}, f_sum, 0.0f);
            ck_tile::block_tile_reduce_sync(local, f_sum, ck_tile::bool_constant<false>{});
            ck_tile::sweep_tile(local, [&](auto idx) { denom += local[idx]; });
        }

        float cumulative = 0.0f;
        if (candidate_k > text_start_block) {
            candidate_k = text_start_block;
        }
        for (int rank = 0; rank < candidate_k; ++rank) {
            // Keep max(top_k, count(cumsum <= threshold) + 1) entries.
            // Equivalently, keep this rank when top_k still requires it, or when
            // the cumulative probability before this rank has not crossed the
            // threshold.  The old code marked every candidate_k entry and then
            // discarded the computed `need`, producing a much denser mask.
            const bool keep = rank < top_k || cumulative <= prob_threshold;
            if (!keep) {
                break;
            }
            float best = -ck_tile::numeric<float>::infinity();
            for (int k_begin = 0; k_begin < text_start_block; k_begin += kScoreTile) {
                auto win = ck_tile::make_tile_window(
                    score_pad,
                    ck_tile::make_tuple(ck_tile::number<4>{}, ck_tile::number<kScoreTile>{}),
                    ck_tile::multi_index<2>{0, k_begin},
                    PipelinePolicy::MakeScoreRowDistribution());
                auto tile = ck_tile::load_tile(win);
                ck_tile::sweep_tile(tile, [&](auto idx) {
                    const auto coord = ck_tile::get_x_indices_from_distributed_indices(
                        tile.get_tile_distribution(), idx);
                    const int r = coord.at(ck_tile::number<0>{});
                    const int kb = k_begin + coord.at(ck_tile::number<1>{});
                    if (r != 0 || kb >= text_start_block) {
                        tile(idx) = -ck_tile::numeric<float>::infinity();
                    }
                });
                auto local = ck_tile::block_tile_reduce<float>(
                    tile, ck_tile::sequence<1>{}, f_max, -ck_tile::numeric<float>::infinity());
                ck_tile::block_tile_reduce_sync(local, f_max, ck_tile::bool_constant<false>{});
                ck_tile::sweep_tile(local, [&](auto idx) { best = ck_tile::max(best, local[idx]); });
            }
            cumulative += denom > 0.0f ? expf(best - row_max) / denom : 0.0f;
            for (int k_begin = 0; k_begin < text_start_block; k_begin += kScoreTile) {
                auto win = ck_tile::make_tile_window(
                    score_pad,
                    ck_tile::make_tuple(ck_tile::number<4>{}, ck_tile::number<kScoreTile>{}),
                    ck_tile::multi_index<2>{0, k_begin},
                    PipelinePolicy::MakeScoreRowDistribution());
                auto tile = ck_tile::load_tile(win);
                ck_tile::sweep_tile(tile, [&](auto idx) {
                    const auto coord = ck_tile::get_x_indices_from_distributed_indices(
                        tile.get_tile_distribution(), idx);
                    const int r = coord.at(ck_tile::number<0>{});
                    const int kb = k_begin + coord.at(ck_tile::number<1>{});
                    if (r == 0 && kb < text_start_block && tile[idx] == best) {
                        out_row[kb] = true;
                        tile(idx) = -ck_tile::numeric<float>::infinity();
                    }
                });
                ck_tile::store_tile(win, tile);
            }
        }

        if (has_neighbors) {
            const bool* neigh = neighbor_mask + static_cast<int64_t>(qb) * text_start_block;
            for (int kb = 0; kb < text_start_block; ++kb) {
                if (neigh[kb]) {
                    out_row[kb] = true;
                }
            }
        }

        if (first_frame_blocks > 0 && qb < first_frame_blocks) {
            for (int kb = 0; kb < first_frame_blocks && kb < num_blocks; ++kb) {
                out_row[kb] = true;
            }
        }
        if (text_blocks > 0) {
            const int end = ck_tile::min(num_blocks, text_start_block + text_blocks);
            for (int kb = text_start_block; kb < end; ++kb) {
                out_row[kb] = true;
            }
        }
    }
};
