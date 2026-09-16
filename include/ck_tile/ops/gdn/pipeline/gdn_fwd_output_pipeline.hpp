// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_policy.hpp"

namespace gdn {

template <typename Problem_>
struct GdnOutputFwdPipeline
{
    using Problem  = Problem_;
    using Policy   = GdnOutputFwdPolicy<Problem>;
    using DataType = typename Problem::DataType;
    using Kargs    = typename Problem::Kargs;

    static constexpr ck_tile::index_t kGroupSize = Problem::kGroupSize;
    static constexpr bool kTwoRows = Problem::kTwoRows;
    static constexpr ck_tile::index_t kChunkSize = Problem::kChunkSize;
    static constexpr ck_tile::index_t kHeadDim   = Problem::kHeadDim;
    static constexpr ck_tile::index_t kValueDim  = Problem::kValueDim;
    static constexpr ck_tile::index_t kRowTile   = Problem::kRowTile;
    static constexpr ck_tile::index_t kValueTile = Problem::kValueTile;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_DEVICE void operator()(Kargs args) const;
};

#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_pipeline_two_rows.hpp"

template <typename Problem>
CK_TILE_DEVICE void GdnOutputFwdPipeline<Problem>::operator()(Kargs args) const
{
    constexpr ck_tile::index_t GroupSize = kGroupSize;
    if constexpr(GroupSize > 1 || kTwoRows)
    {
        gdn_output_tiled_two_rows<Problem>(args);
    }
    else
    {

    const ck_tile::index_t chunk_row_tile = static_cast<ck_tile::index_t>(blockIdx.x);
    const ck_tile::index_t global_chunk   = chunk_row_tile / (kChunkSize / kRowTile);
    const ck_tile::index_t row_tile       = chunk_row_tile % (kChunkSize / kRowTile);
    const ck_tile::index_t qh             = GroupSize > 1 ? static_cast<ck_tile::index_t>(blockIdx.y) : static_cast<ck_tile::index_t>(blockIdx.y) / (args.num_value_heads / args.num_qk_heads);
    const ck_tile::index_t vh_base        = GroupSize > 1 ? qh * GroupSize : static_cast<ck_tile::index_t>(blockIdx.y);
    const ck_tile::index_t tid            = ck_tile::get_thread_id();

    if(global_chunk >= args.num_chunks || qh >= args.num_qk_heads)
    {
        return;
    }

    ck_tile::index_t seq;
    ck_tile::index_t local_chunk;
    ck_tile::index_t bos;
    ck_tile::index_t eos;
    if(args.is_varlen)
    {
        seq = static_cast<ck_tile::index_t>(args.chunk_indices[global_chunk * 2]);
        local_chunk =
            static_cast<ck_tile::index_t>(args.chunk_indices[global_chunk * 2 + 1]);
        bos = static_cast<ck_tile::index_t>(args.cu_seqlens[seq]);
        eos = static_cast<ck_tile::index_t>(args.cu_seqlens[seq + 1]);
    }
    else
    {
        const ck_tile::index_t chunks_per_sequence =
            ck_tile::integer_divide_ceil(args.total_tokens, kChunkSize);
        seq = global_chunk / chunks_per_sequence;
        local_chunk = global_chunk % chunks_per_sequence;
        bos = seq * args.total_tokens;
        eos = bos + args.total_tokens;
    }
    const ck_tile::index_t token_begin = bos + local_chunk * kChunkSize;
    const ck_tile::index_t valid_tokens = ck_tile::min(kChunkSize, eos - token_begin);
    const ck_tile::index_t row_begin = row_tile * kRowTile;

    // The GroupSize=1 launch uses two row CTAs per chunk.  A short tail may
    // contain no rows for the second CTA; avoid constructing zero-length
    // global windows for that inactive tile.
    if(row_begin >= valid_tokens)
    {
        return;
    }

    using QKProblem = typename Policy::template QKProblem<
        kRowTile, kChunkSize, kHeadDim>;
    using QKPolicy = typename Policy::template QKPolicy<
        kRowTile, kChunkSize, kHeadDim>;
    using QKGemm = typename Policy::template QKBlockGemm<
        kRowTile, kChunkSize, kHeadDim>;
    using QHProblem = typename Policy::template QHProblem<
        kRowTile, kValueTile, kHeadDim>;
    using QHPolicy = typename Policy::template QHPolicy<
        kRowTile, kValueTile, kHeadDim>;
    using QHGemm = typename Policy::template QHBlockGemm<
        kRowTile, kValueTile, kHeadDim>;
    using PVProblem = typename Policy::template PVProblem<
        kRowTile, kValueTile, kChunkSize>;
    using PVPolicy = typename Policy::template PVPolicy<
        kRowTile, kValueTile, kChunkSize>;
    using PVGemm = typename Policy::template PVBlockGemm<
        kRowTile, kValueTile, kChunkSize>;

    constexpr ck_tile::index_t kLdsPadding = Problem::Config::kLdsPadding;

    // Peak LDS Optimization (Aliasing disjoint buffers):
    //   k_lds, h_lds, and v_lds are aliased into a single shared buffer.
    //   During QK phase: holds k_lds (16 KiB).
    //   During value phase: holds h_lds and v_lds sequentially.
    //   We pad the row stride of h_lds and v_lds by kLdsPadding elements to prevent bank conflicts.
    __shared__ char shared_lds_bytes[128 * (kValueTile + kLdsPadding) * sizeof(DataType)];
    DataType* const k_lds = reinterpret_cast<DataType*>(shared_lds_bytes);
    DataType* const h_lds = reinterpret_cast<DataType*>(shared_lds_bytes);
    DataType* const v_lds = reinterpret_cast<DataType*>(shared_lds_bytes);

    // Pad the row stride by one element to break LDS bank-aliasing patterns when the
    // gated score is reloaded with PVGemm's distributed layout.  The logical tile is
    // still [kRowTile, kChunkSize]; the extra column is never read.
    DataType* const score_lds = reinterpret_cast<DataType*>(shared_lds_bytes);
    __shared__ float gate_lds[kChunkSize + kRowTile];

    const bool full_chunk = valid_tokens == kChunkSize;

    // 1. Load K to LDS
    const DataType* k_global_base =
        args.k +
        (static_cast<int64_t>(token_begin) * args.num_qk_heads + qh) * kHeadDim;
    constexpr auto k_copy_distribution =
        Policy::template MakeGlobalCopyDistribution<kChunkSize, kHeadDim>();
    auto k_dram_view =
        ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            k_global_base,
            ck_tile::make_tuple(valid_tokens, ck_tile::number<kHeadDim>{}),
            ck_tile::make_tuple(args.num_qk_heads * kHeadDim,
                                ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});
    auto k_padded_view = ck_tile::pad_tensor_view(
        k_dram_view,
        ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                            ck_tile::number<kHeadDim>{}),
        ck_tile::sequence<true, false>{});
    auto k_dram_window = ck_tile::make_tile_window(
        k_padded_view,
        ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                            ck_tile::number<kHeadDim>{}),
        ck_tile::multi_index<2>{0, 0},
        k_copy_distribution);
    constexpr auto k_lds_store_desc =
        Policy::template MakePaddedRowMajorLdsDescriptor<kChunkSize, kHeadDim, 0>();
    auto k_lds_store_view =
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            k_lds, k_lds_store_desc);
    auto k_lds_store_window = ck_tile::make_tile_window(
        k_lds_store_view,
        ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                            ck_tile::number<kHeadDim>{}),
        ck_tile::multi_index<2>{0, 0},
        k_copy_distribution);
    auto k_copy_tile =
        ck_tile::make_static_distributed_tensor<DataType>(k_copy_distribution);
    if(full_chunk)
    {
        ck_tile::load_tile(k_copy_tile,
                           k_dram_window,
                           ck_tile::bool_constant<false>{});
    }
    else
    {
        ck_tile::load_tile(k_copy_tile, k_dram_window, ck_tile::bool_constant<true>{});
    }
    ck_tile::store_tile(k_lds_store_window, k_copy_tile);
    ck_tile::block_sync_lds(); // Sync K load

    // 3. Compute score = Q @ K^T
    const ck_tile::index_t token_capacity =
        args.is_varlen ? args.total_tokens : args.num_sequences * args.total_tokens;
    auto q_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        args.q + qh * kHeadDim,
        ck_tile::make_tuple(token_capacity, ck_tile::number<kHeadDim>{}),
        ck_tile::make_tuple(args.num_qk_heads * kHeadDim, ck_tile::number<1>{}),
        ck_tile::number<8>{},
        ck_tile::number<1>{});

    auto load_q = [&](auto tile_seed) {
        auto q_window = ck_tile::make_tile_window(
            q_view,
            ck_tile::make_tuple(ck_tile::number<kRowTile>{}, ck_tile::number<kHeadDim>{}),
            ck_tile::multi_index<2>{token_begin + row_begin, 0},
            tile_seed.get_tile_distribution());
        return ck_tile::load_tile(q_window);
    };

    constexpr auto qk_gemm = QKGemm{};
    auto q_for_score = load_q(
        Policy::template MakeARegTile<QKProblem, QKPolicy>());
    auto score = QKGemm::MakeCBlockTile();
    ck_tile::clear_tile(score);

    constexpr auto k_desc = ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(ck_tile::number<kChunkSize>{}, ck_tile::number<kHeadDim>{}),
        ck_tile::make_tuple(ck_tile::number<kHeadDim>{}, ck_tile::number<1>{}));
    auto k_view =
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(k_lds, k_desc);
    auto k_window = ck_tile::make_tile_window(
        k_view,
        ck_tile::make_tuple(ck_tile::number<kChunkSize>{}, ck_tile::number<kHeadDim>{}),
        {0, 0});
    qk_gemm(score, q_for_score, k_window);
    auto score_output =
        Policy::template MakeCOutputLayout<QKProblem, QKPolicy>(score);

    // 4. Loop over GQA Value Heads
    auto q_for_h = load_q(
        Policy::template MakeARegTile<QHProblem, QHPolicy>());

    for(ck_tile::index_t g_v = 0; g_v < GroupSize; ++g_v)
    {
        const ck_tile::index_t vh = vh_base + g_v;
        if(vh >= args.num_value_heads)
        {
            break;
        }

        // Load gates for the current head g_v (384 Bytes)
        for(ck_tile::index_t idx = tid; idx < kChunkSize + kRowTile; idx += kBlockSize)
        {
            if(idx < kChunkSize)
            {
                const ck_tile::index_t t_id = idx;
                float gate = 0.0f;
                if(t_id < valid_tokens)
                {
                    gate = args.g[static_cast<int64_t>(token_begin + t_id) * args.num_value_heads + vh];
                }
                gate_lds[idx] = t_id < valid_tokens ? __builtin_amdgcn_exp2f(-gate) : 0.0f;
            }
            else
            {
                const ck_tile::index_t r_id = idx - kChunkSize;
                const ck_tile::index_t row = row_begin + r_id;
                float gate = 0.0f;
                if(row < valid_tokens)
                {
                    gate = args.g[static_cast<int64_t>(token_begin + row) * args.num_value_heads + vh];
                }
                gate_lds[idx] = row < valid_tokens ? __builtin_amdgcn_exp2f(gate) : 0.0f;
            }
        }
        ck_tile::block_sync_lds();

        // Apply decay to score and save to score_lds (for layout transform)
        auto gated_score =
            ck_tile::make_static_distributed_tensor<float>(
                score_output.get_tile_distribution());
        ck_tile::sweep_tile(score_output, [&](auto idx) {
            const auto x = ck_tile::get_x_indices_from_distributed_indices(
                score_output.get_tile_distribution(), idx);
            const ck_tile::index_t row = x[ck_tile::number<0>{}];
            const ck_tile::index_t col = x[ck_tile::number<1>{}];
            const ck_tile::index_t global_row = row_begin + row;
            float value = 0.0f;
            if(global_row < valid_tokens && col < valid_tokens && col <= global_row)
            {
                value = score_output[idx] * gate_lds[kChunkSize + row] * gate_lds[col];
            }
            gated_score(idx) = value;
        });
        auto score_store_tile = ck_tile::cast_tile<DataType>(gated_score);
        constexpr auto score_desc =
            Policy::template MakePaddedRowMajorLdsDescriptor<
                kRowTile, kChunkSize, Problem::Config::kScorePadding>();
        auto score_store_view =
            ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                score_lds, score_desc);
        auto score_store_window = ck_tile::make_tile_window(
            score_store_view,
            ck_tile::make_tuple(ck_tile::number<kRowTile>{},
                                ck_tile::number<kChunkSize>{}),
            ck_tile::multi_index<2>{0, 0},
            score_store_tile.get_tile_distribution());
        ck_tile::store_tile(score_store_window, score_store_tile);
        ck_tile::block_sync_lds();

        // Load score from score_lds to score_areg registers
        auto score_view =
            ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(score_lds, score_desc);
        auto score_areg =
            Policy::template MakeARegTile<PVProblem, PVPolicy>();
        auto score_window = ck_tile::make_tile_window(
            score_view,
            ck_tile::make_tuple(ck_tile::number<kRowTile>{}, ck_tile::number<kChunkSize>{}),
            {0, 0},
            score_areg.get_tile_distribution());
        ck_tile::load_tile(score_areg, score_window);
        ck_tile::block_sync_lds();

        // Load Q for H computation and apply gating
        auto q_gated =
            Policy::template MakeARegTile<QHProblem, QHPolicy>();
        constexpr auto q_gated_spans = decltype(q_gated)::get_distributed_spans();
        ck_tile::sweep_tile_span(q_gated_spans[ck_tile::number<0>{}], [&](auto idx0) {
            ck_tile::sweep_tile_span(q_gated_spans[ck_tile::number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = ck_tile::make_tuple(idx0, idx1);
                const auto x = ck_tile::get_x_indices_from_distributed_indices(
                    q_gated.get_tile_distribution(), dstr_idx);
                const ck_tile::index_t row = x[ck_tile::number<0>{}];
                q_gated(dstr_idx) = gdn_output_cast_from_float<DataType>(
                    ck_tile::type_convert<float>(q_for_h[dstr_idx]) * gate_lds[kChunkSize + row]);
            });
        });

        constexpr ck_tile::index_t kNumValueTiles = (kValueDim + kValueTile - 1) / kValueTile;
        constexpr auto qh_gemm = QHGemm{};
        constexpr auto pv_gemm = PVGemm{};

        constexpr auto h_desc =
            Policy::template MakePaddedTransposeLdsDescriptor<
                kHeadDim, kValueTile, kLdsPadding>();
        constexpr auto v_desc =
            Policy::template MakePaddedTransposeLdsDescriptor<
                kChunkSize, kValueTile, kLdsPadding>();

        constexpr auto h_copy_distribution =
            Policy::template MakeGlobalCopyDistribution<kHeadDim, kValueTile>();
        auto load_h_copy_tile = [&](ck_tile::index_t value_begin) {
            const DataType* h_base =
                args.h + (static_cast<int64_t>(global_chunk) * args.num_value_heads + vh) *
                             kHeadDim * kValueDim + value_begin;
            auto h_dram_view =
                ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    h_base,
                    ck_tile::make_tuple(ck_tile::number<kHeadDim>{},
                                        ck_tile::number<kValueTile>{}),
                    ck_tile::make_tuple(ck_tile::number<kValueDim>{},
                                        ck_tile::number<1>{}),
                    ck_tile::number<8>{},
                    ck_tile::number<1>{});
            auto h_dram_window = ck_tile::make_tile_window(
                h_dram_view,
                ck_tile::make_tuple(ck_tile::number<kHeadDim>{},
                                    ck_tile::number<kValueTile>{}),
                ck_tile::multi_index<2>{0, 0},
                h_copy_distribution);
            return ck_tile::load_tile(h_dram_window,
                                      ck_tile::bool_constant<false>{});
        };

        constexpr auto v_copy_distribution =
            Policy::template MakeGlobalCopyDistribution<kChunkSize, kValueTile>();
        auto load_v_copy_tile = [&](ck_tile::index_t value_begin) {
            const DataType* v_base =
                args.v_new +
                (static_cast<int64_t>(token_begin) * args.num_value_heads + vh) * kValueDim +
                value_begin;
            auto v_dram_view =
                ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    v_base,
                    ck_tile::make_tuple(valid_tokens,
                                        ck_tile::number<kValueTile>{}),
                    ck_tile::make_tuple(args.num_value_heads * kValueDim,
                                        ck_tile::number<1>{}),
                    ck_tile::number<8>{},
                    ck_tile::number<1>{});
            auto v_padded_view = ck_tile::pad_tensor_view(
                v_dram_view,
                ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                                    ck_tile::number<kValueTile>{}),
                ck_tile::sequence<true, false>{});
            auto v_dram_window = ck_tile::make_tile_window(
                v_padded_view,
                ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                                    ck_tile::number<kValueTile>{}),
                ck_tile::multi_index<2>{0, 0},
                v_copy_distribution);
            auto tile =
                ck_tile::make_static_distributed_tensor<DataType>(v_copy_distribution);
            if(full_chunk)
                ck_tile::load_tile(tile, v_dram_window, ck_tile::bool_constant<false>{});
            else
                ck_tile::load_tile(tile, v_dram_window, ck_tile::bool_constant<true>{});
            return tile;
        };

        auto h_prefetch_tile = load_h_copy_tile(0);
        auto v_prefetch_tile = load_v_copy_tile(0);

        for(ck_tile::index_t value_tile = 0; value_tile < kNumValueTiles; ++value_tile)
        {
            const ck_tile::index_t value_begin = value_tile * kValueTile;

            constexpr auto h_lds_store_desc =
                Policy::template MakePaddedRowMajorLdsDescriptor<
                    kHeadDim, kValueTile, kLdsPadding>();
            auto h_lds_store_view =
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    h_lds, h_lds_store_desc);
            auto h_lds_store_window = ck_tile::make_tile_window(
                h_lds_store_view,
                ck_tile::make_tuple(ck_tile::number<kHeadDim>{},
                                    ck_tile::number<kValueTile>{}),
                ck_tile::multi_index<2>{0, 0},
                h_copy_distribution);
            ck_tile::store_tile(h_lds_store_window, h_prefetch_tile);
            ck_tile::block_sync_lds();

            // 4. Load H from h_lds to registers h_breg
            auto h_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                h_lds, h_desc);
            auto h_breg =
                Policy::template MakeBRegTile<QHProblem, QHPolicy>();
            auto h_window = ck_tile::make_tile_window(
                h_view,
                ck_tile::make_tuple(ck_tile::number<kValueTile>{}, ck_tile::number<kHeadDim>{}),
                {0, 0},
                h_breg.get_tile_distribution());
            ck_tile::load_tile(h_breg, h_window);
            ck_tile::block_sync_lds(); // Sync to make sure all threads have read H before overwriting h_lds!

            constexpr auto v_lds_store_desc =
                Policy::template MakePaddedRowMajorLdsDescriptor<
                    kChunkSize, kValueTile, kLdsPadding>();
            auto v_lds_store_view =
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    v_lds, v_lds_store_desc);
            auto v_lds_store_window = ck_tile::make_tile_window(
                v_lds_store_view,
                ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                                    ck_tile::number<kValueTile>{}),
                ck_tile::multi_index<2>{0, 0},
                v_copy_distribution);
            ck_tile::store_tile(v_lds_store_window, v_prefetch_tile);
            ck_tile::block_sync_lds();

            // 6. Load V from v_lds to registers v_breg
            auto v_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                v_lds, v_desc);
            auto v_breg =
                Policy::template MakeBRegTile<PVProblem, PVPolicy>();
            auto v_window = ck_tile::make_tile_window(
                v_view,
                ck_tile::make_tuple(ck_tile::number<kValueTile>{}, ck_tile::number<kChunkSize>{}),
                {0, 0},
                v_breg.get_tile_distribution());
            ck_tile::load_tile(v_breg, v_window);
            ck_tile::block_sync_lds(); // Sync to make sure all threads have read V before next iteration!

            // 7. Prefetch H and V for the NEXT iteration asynchronously in the background
            const ck_tile::index_t next_vt = value_tile + 1;
            if(next_vt < kNumValueTiles)
            {
                const ck_tile::index_t next_v_begin = next_vt * kValueTile;

                h_prefetch_tile = load_h_copy_tile(next_v_begin);
                v_prefetch_tile = load_v_copy_tile(next_v_begin);
            }

            // 8. Compute QH GEMM
            auto output = QHGemm::MakeCBlockTile();
            ck_tile::clear_tile(output);
            qh_gemm(output, q_gated, h_breg);

            // 9. Compute PV GEMM
            auto local_output = PVGemm::MakeCBlockTile();
            ck_tile::clear_tile(local_output);
            pv_gemm(local_output, score_areg, v_breg);

            // 10. Elementwise addition and scaling in registers
            auto hist_logical =
                Policy::template MakeCOutputLayout<QHProblem, QHPolicy>(output);
            auto local_logical =
                Policy::template MakeCOutputLayout<PVProblem, PVPolicy>(local_output);

            auto output_tile_float = hist_logical;
            ck_tile::tile_elementwise_inout([&](auto& x, const auto& y) {
                x = (x + y) * args.scale;
            }, output_tile_float, local_logical);

            // 11. Epilogue LDS Exchange & 16-byte Vector Coalesced Write-back
            auto output_tile = ck_tile::cast_tile<DataType>(output_tile_float);

            // A. Store output_tile to LDS (using the 4 KiB workspace segment of score_lds)
            constexpr auto epilogue_lds_desc =
                Policy::template MakePaddedRowMajorLdsDescriptor<
                    kRowTile, kValueTile, 0>();

            auto epilogue_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                score_lds, epilogue_lds_desc);

            auto epilogue_lds_window = ck_tile::make_tile_window(
                epilogue_lds_view,
                ck_tile::make_tuple(ck_tile::number<kRowTile>{}, ck_tile::number<kValueTile>{}),
                {0, 0},
                output_tile.get_tile_distribution());

            ck_tile::store_tile(epilogue_lds_window, output_tile);
            ck_tile::block_sync_lds(); // Sync to make sure LDS store is completed by all threads

            // B. Policy-defined LDS->Reg->GMEM vector transfer.
            constexpr auto o_copy_distribution =
                Policy::template MakeOutputCopyDistribution<kRowTile, kValueTile>();
            auto o_lds_window = ck_tile::make_tile_window(
                epilogue_lds_view,
                ck_tile::make_tuple(ck_tile::number<kRowTile>{},
                                    ck_tile::number<kValueTile>{}),
                ck_tile::multi_index<2>{0, 0},
                o_copy_distribution);
            auto o_copy_tile = ck_tile::load_tile(o_lds_window);
            ck_tile::block_sync_lds();

            DataType* o_base =
                args.o +
                (static_cast<int64_t>(token_begin + row_begin) * args.num_value_heads + vh) *
                    kValueDim +
                value_begin;
            const ck_tile::index_t o_valid_rows =
                row_begin < valid_tokens ? valid_tokens - row_begin : 0;
            auto o_dram_view =
                ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    o_base,
                    ck_tile::make_tuple(o_valid_rows,
                                        ck_tile::number<kValueTile>{}),
                    ck_tile::make_tuple(args.num_value_heads * kValueDim,
                                        ck_tile::number<1>{}),
                    ck_tile::number<8>{},
                    ck_tile::number<1>{});
            auto o_padded_view = ck_tile::pad_tensor_view(
                o_dram_view,
                ck_tile::make_tuple(ck_tile::number<kRowTile>{},
                                    ck_tile::number<kValueTile>{}),
                ck_tile::sequence<true, false>{});
            auto o_dram_window = ck_tile::make_tile_window(
                o_padded_view,
                ck_tile::make_tuple(ck_tile::number<kRowTile>{},
                                    ck_tile::number<kValueTile>{}),
                ck_tile::multi_index<2>{0, 0},
                o_copy_distribution);
            ck_tile::store_tile(o_dram_window, o_copy_tile);
        }
    }
    }
}

} // namespace gdn
