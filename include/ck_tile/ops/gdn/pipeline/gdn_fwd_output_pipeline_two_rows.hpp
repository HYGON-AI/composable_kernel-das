// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/gdn/gdn_numeric.hpp"

// Included from gdn_fwd_output_pipeline.hpp inside namespace gdn, after the
// pipeline and vector-load helpers are declared.
// Long-sequence GQA path: one CTA processes both 32-row tiles and reuses K/H/V.

template <typename Problem>
CK_TILE_DEVICE void gdn_output_tiled_two_rows(
    typename Problem::Kargs args)
{
    using DataType = typename Problem::DataType;
    using Kernel = Problem;
    using Policy = GdnOutputFwdPolicy<Problem>;
    constexpr ck_tile::index_t GroupSize = Problem::kGroupSize;
    constexpr ck_tile::index_t kValueSplit = Problem::kValueSplit;
    constexpr ck_tile::index_t kChunkSize = Problem::kChunkSize;
    constexpr ck_tile::index_t kHeadDim = Problem::kHeadDim;
    constexpr ck_tile::index_t kValueDim = Problem::kValueDim;
    constexpr ck_tile::index_t kRowTile = Kernel::kRowTile;
    constexpr ck_tile::index_t kValueTile = Kernel::kValueTile;
    constexpr ck_tile::index_t kScoreStride =
        kChunkSize + Problem::Config::kScorePadding;
    constexpr ck_tile::index_t kLdsPadding = Problem::Config::kLdsPadding;
    static_assert(GroupSize == 2 || GroupSize == 4 ||
                  (GroupSize == 1 && Problem::kTwoRows));
    static_assert(kValueTile == 32);
    static_assert((kValueDim / kValueTile) % kValueSplit == 0);

    const ck_tile::index_t global_chunk = static_cast<ck_tile::index_t>(blockIdx.x);
    const ck_tile::index_t g_ratio = args.num_value_heads / args.num_qk_heads;
    const ck_tile::index_t groups_per_qh = g_ratio / GroupSize;
    const ck_tile::index_t group_block = static_cast<ck_tile::index_t>(blockIdx.y);
    const ck_tile::index_t physical_qh = group_block / groups_per_qh;
    // Adjacent linear workgroups otherwise walk chunks at a fixed q-head;
    // the 2-MiB H chunk stride repeatedly targets the same low cache-set bits.
    // Rotate q-head by chunk on long GroupSize=1/2/4 paths to distribute
    // concurrent H traffic across VL1/TCP cacheline allocators.
    const ck_tile::index_t qh =
        (GroupSize == 4 || GroupSize == 2 ||
         (GroupSize == 1 && Problem::kTwoRows))
            ? (physical_qh + global_chunk) & (args.num_qk_heads - 1)
            : physical_qh;
    const ck_tile::index_t vh_base =
        qh * g_ratio + (group_block % groups_per_qh) * GroupSize;
    const ck_tile::index_t tid = ck_tile::get_thread_id();

    if(global_chunk >= args.num_chunks || qh >= args.num_qk_heads)
        return;

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
    const bool full_chunk = valid_tokens == kChunkSize;

    using QKProblem = typename Policy::template QKProblem<
        kRowTile, kChunkSize, kHeadDim>;
    using QKPolicy = typename Policy::template QKPolicy<
        kRowTile, kChunkSize, kHeadDim>;
    using QKGemm = typename Policy::template QKBlockGemmBReg<
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

    constexpr ck_tile::index_t kHvLdsElements =
        kHeadDim * (kValueTile + kLdsPadding);
    constexpr ck_tile::index_t kIoLdsElements = kHvLdsElements;
    // The gate values are only live after K has been consumed and before the
    // H/V phases start.  Keep them in the unused tail following the logical
    // score exchange tile instead of allocating a second shared object.  This
    // keeps the staged direct-K workspace at 10 KiB.
    constexpr ck_tile::index_t kScoreLdsElements = kRowTile * kScoreStride;
    constexpr ck_tile::index_t kGateLdsElements = kChunkSize + kChunkSize;
    constexpr ck_tile::index_t kGateLdsBytes = kGateLdsElements * sizeof(float);
    constexpr ck_tile::index_t kGateLdsOffsetBytes =
        kScoreLdsElements * sizeof(DataType);
    static_assert(kGateLdsOffsetBytes % alignof(float) == 0);
    static_assert(kGateLdsOffsetBytes + kGateLdsBytes <=
                  kIoLdsElements * sizeof(DataType));
    // Direct H/V need no shared staging during the value loop. Both output
    // row tiles fit in the score workspace below the still-live gate vector.
    constexpr bool kUsePairedEpilogue =
        Problem::kPairedEpilogue;
    constexpr ck_tile::index_t kPairedEpilogueElements =
        kUsePairedEpilogue ? 2 * kRowTile * kValueTile : 0;
    constexpr ck_tile::index_t kIoLdsBytes =
        kIoLdsElements * sizeof(DataType);
    static_assert(kPairedEpilogueElements * sizeof(DataType) <= kGateLdsOffsetBytes);
    constexpr ck_tile::index_t kPairedEpilogueBytes = 0;
    __shared__ char shared_lds_bytes[kIoLdsBytes + kPairedEpilogueBytes];
    DataType* const io_lds = reinterpret_cast<DataType*>(shared_lds_bytes);
    DataType* const score_lds = io_lds;
    // Both score tiles are already in registers before the epilogue begins.
    // Their 4-KiB region is dead and disjoint from the live gates at 4608 B.
    DataType* const epilogue_lds = io_lds;
    float* const gate_lds = reinterpret_cast<float*>(
        shared_lds_bytes + kGateLdsOffsetBytes);

    auto load_q = [&](ck_tile::index_t row_begin, auto tile_seed) {
        const DataType* q_base =
            args.q +
            (static_cast<int64_t>(token_begin + row_begin) * args.num_qk_heads + qh) *
                kHeadDim;
        auto q_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            q_base,
            ck_tile::make_tuple(ck_tile::number<kRowTile>{},
                                ck_tile::number<kHeadDim>{}),
            ck_tile::make_tuple(args.num_qk_heads * kHeadDim,
                                ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});
        auto q_window = ck_tile::make_tile_window(
            q_view,
            ck_tile::make_tuple(ck_tile::number<kRowTile>{}, ck_tile::number<kHeadDim>{}),
            ck_tile::multi_index<2>{0, 0},
            tile_seed.get_tile_distribution());
        return ck_tile::load_tile(q_window);
    };

    // Full K is contiguous and loaded once into the native QK B distribution.
    const DataType* k_global_base =
        args.k +
        (static_cast<int64_t>(token_begin) * args.num_qk_heads + qh) *
            kHeadDim;
    auto k_dram_view =
        ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            k_global_base,
            ck_tile::make_tuple(valid_tokens,
                                ck_tile::number<kHeadDim>{}),
            ck_tile::make_tuple(args.num_qk_heads * kHeadDim,
                                ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});
    auto k_padded_view = ck_tile::pad_tensor_view(
        k_dram_view,
        ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                            ck_tile::number<kHeadDim>{}),
        ck_tile::sequence<true, false>{});
    auto k_breg = Policy::template MakeBRegTile<QKProblem, QKPolicy>();
    auto k_window = ck_tile::make_tile_window(
        k_padded_view,
        ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                            ck_tile::number<kHeadDim>{}),
        ck_tile::multi_index<2>{0, 0},
        k_breg.get_tile_distribution());
    if(full_chunk)
        ck_tile::load_tile(k_breg,
                           k_window,
                           ck_tile::bool_constant<false>{});
    else
        ck_tile::load_tile(k_breg,
                           k_window,
                           ck_tile::bool_constant<true>{});

    constexpr auto qk_gemm = QKGemm{};
    auto compute_raw_score = [&](ck_tile::index_t row_begin) {
        auto q = load_q(
            row_begin, Policy::template MakeARegTile<QKProblem, QKPolicy>());
        auto score = QKGemm::MakeCBlockTile();
        ck_tile::clear_tile(score);
        qk_gemm(score, q, k_breg);
        return Policy::template MakeCOutputLayout<QKProblem, QKPolicy>(score);
    };
    auto score_output0 = compute_raw_score(0);
    auto score_output1 = compute_raw_score(kRowTile);

    auto q_for_h0 = load_q(
        0, Policy::template MakeARegTile<QHProblem, QHPolicy>());
    auto q_for_h1 = load_q(
        kRowTile, Policy::template MakeARegTile<QHProblem, QHPolicy>());

    constexpr auto score_desc =
        Policy::template MakePaddedRowMajorLdsDescriptor<
            kRowTile,
            kChunkSize,
            Problem::Config::kScorePadding>();

    constexpr auto qh_gemm = QHGemm{};
    constexpr auto pv_gemm = PVGemm{};
    constexpr ck_tile::index_t kNumValueTiles =
        (kValueDim + kValueTile - 1) / kValueTile;
    constexpr ck_tile::index_t kValueTilesPerSplit =
        kNumValueTiles / kValueSplit;
    const ck_tile::index_t value_tile_offset =
        static_cast<ck_tile::index_t>(blockIdx.z) * kValueTilesPerSplit;
    for(ck_tile::index_t g_v = 0; g_v < GroupSize; ++g_v)
    {
        const ck_tile::index_t logical_g_v =
            GroupSize == 4 ? (g_v + (global_chunk & 3)) & 3 : g_v;
        const ck_tile::index_t vh = vh_base + logical_g_v;
        if(vh >= args.num_value_heads)
            break;

        // One inverse gate vector and both 32-row gate vectors.
        for(ck_tile::index_t idx = tid; idx < kChunkSize + kChunkSize;
            idx += Kernel::kBlockSize)
        {
            if(idx < kChunkSize)
            {
                float gate = 0.0f;
                if(idx < valid_tokens)
                    gate = args.g[static_cast<int64_t>(token_begin + idx) *
                                      args.num_value_heads +
                                  vh];
                gate_lds[idx] = idx < valid_tokens ? __builtin_amdgcn_exp2f(-gate) : 0.0f;
            }
            else
            {
                const ck_tile::index_t row = idx - kChunkSize;
                float gate = 0.0f;
                if(row < valid_tokens)
                    gate = args.g[static_cast<int64_t>(token_begin + row) *
                                      args.num_value_heads +
                                  vh];
                gate_lds[idx] = row < valid_tokens ? __builtin_amdgcn_exp2f(gate) : 0.0f;
            }
        }
        ck_tile::block_sync_lds();

        auto make_score_areg = [&](const auto& score_output,
                                   ck_tile::index_t row_begin) {
            using ScoreTile = ck_tile::remove_cvref_t<decltype(score_output)>;
            ScoreTile gated_score;
            gated_score.get_thread_buffer() = score_output.get_thread_buffer();
            ck_tile::sweep_tile(gated_score, [&](auto idx) {
                const auto x = ck_tile::get_x_indices_from_distributed_indices(
                    gated_score.get_tile_distribution(), idx);
                const ck_tile::index_t row = x[ck_tile::number<0>{}];
                const ck_tile::index_t col = x[ck_tile::number<1>{}];
                const ck_tile::index_t global_row = row_begin + row;
                float value = 0.0f;
                if(global_row < valid_tokens && col < valid_tokens && col <= global_row)
                    value = score_output[idx] * gate_lds[kChunkSize + global_row] *
                            gate_lds[col];
                gated_score(idx) = value;
            });
            auto score_store_tile = ck_tile::gdn_cast_tile<DataType>(gated_score);
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

            auto score_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                score_lds, score_desc);
            auto score_areg =
                Policy::template MakeARegTile<PVProblem, PVPolicy>();
            auto score_window = ck_tile::make_tile_window(
                score_view,
                ck_tile::make_tuple(ck_tile::number<kRowTile>{},
                                    ck_tile::number<kChunkSize>{}),
                {0, 0},
                score_areg.get_tile_distribution());
            ck_tile::load_tile(score_areg, score_window);
            ck_tile::block_sync_lds();
            return score_areg;
        };

        auto score_areg0 = make_score_areg(score_output0, 0);
        auto score_areg1 = make_score_areg(score_output1, kRowTile);

        auto make_q_gated = [&](const auto& q_for_h, ck_tile::index_t row_begin) {
            auto q_gated = q_for_h;
            // Direct H/V leave the gate LDS region live through the epilogue.
            if constexpr(std::is_same_v<DataType, ck_tile::fp16_t> && Problem::kPairedEpilogue && GroupSize == 4) return q_gated;
            constexpr auto spans = decltype(q_gated)::get_distributed_spans();
            ck_tile::sweep_tile_span(spans[ck_tile::number<0>{}], [&](auto idx0) {
                ck_tile::sweep_tile_span(spans[ck_tile::number<1>{}], [&](auto idx1) {
                    constexpr auto dstr_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x = ck_tile::get_x_indices_from_distributed_indices(
                        q_gated.get_tile_distribution(), dstr_idx);
                    const ck_tile::index_t row = x[ck_tile::number<0>{}];
                    q_gated(dstr_idx) = gdn_output_cast_from_float<DataType>(
                        ck_tile::type_convert<float>(q_for_h[dstr_idx]) *
                        gate_lds[kChunkSize + row_begin + row]);
                });
            });
            return q_gated;
        };

        auto q_gated0 = make_q_gated(q_for_h0, 0);
        auto q_gated1 = make_q_gated(q_for_h1, kRowTile);

        for(ck_tile::index_t local_value_tile = 0;
            local_value_tile < kValueTilesPerSplit;
            ++local_value_tile)
        {
            const ck_tile::index_t value_tile =
                value_tile_offset + local_value_tile;
            const ck_tile::index_t logical_value_tile =
                GroupSize == 4
                    ? (value_tile + (global_chunk & 3)) & 3
                    : value_tile;
            const ck_tile::index_t value_begin = logical_value_tile * kValueTile;

            auto h_breg = [&]() {
                if constexpr(Problem::kPreshuffledH)
                {
                    const DataType* h_head_base =
                        args.h +
                        (static_cast<int64_t>(global_chunk) * args.num_value_heads + vh) *
                            kHeadDim * kValueDim;
                    return Policy::template LoadPreshuffledH<QHProblem, QHPolicy>(
                        h_head_base, value_begin);
                }
                else
                {
                    // H is physically [K, value_dim].  Expose it as logical
                    // [N, K] and load directly into the MMAC B distribution.
                    const DataType* h_base =
                        args.h +
                        (static_cast<int64_t>(global_chunk) * args.num_value_heads + vh) *
                            kHeadDim * kValueDim +
                        value_begin;
                    const auto h_global_desc = ck_tile::make_naive_tensor_descriptor(
                        ck_tile::make_tuple(ck_tile::number<kValueTile>{},
                                            ck_tile::number<kHeadDim>{}),
                        ck_tile::make_tuple(ck_tile::number<1>{},
                                            ck_tile::number<kValueDim>{}));
                    const auto h_global_view =
                        ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
                            h_base, h_global_desc);
                    auto tile =
                        Policy::template MakeBRegTile<QHProblem, QHPolicy>();
                    auto window = ck_tile::make_tile_window(
                        h_global_view,
                        ck_tile::make_tuple(ck_tile::number<kValueTile>{},
                                            ck_tile::number<kHeadDim>{}),
                        ck_tile::multi_index<2>{0, 0},
                        tile.get_tile_distribution());
                    ck_tile::load_tile(tile, window);
                    return tile;
                }
            }();

            // Load V once for both row tiles in the native MMAC B layout.
            const DataType* v_base =
                args.v_new +
                (static_cast<int64_t>(token_begin) * args.num_value_heads + vh) *
                    kValueDim +
                value_begin;
            auto v_breg = [&]() {
                auto tile =
                    Policy::template MakeBRegTile<PVProblem, PVPolicy>();
                const auto v_global_desc = ck_tile::make_naive_tensor_descriptor(
                    ck_tile::make_tuple(ck_tile::number<kValueTile>{},
                                        valid_tokens),
                    ck_tile::make_tuple(ck_tile::number<1>{},
                                        args.num_value_heads * kValueDim));
                const auto v_global_view =
                    ck_tile::make_tensor_view<ck_tile::address_space_enum::global>(
                        v_base, v_global_desc);
                auto v_padded_view = ck_tile::pad_tensor_view(
                    v_global_view,
                    ck_tile::make_tuple(ck_tile::number<kValueTile>{},
                                        ck_tile::number<kChunkSize>{}),
                    ck_tile::sequence<false, true>{});
                auto window = ck_tile::make_tile_window(
                    v_padded_view,
                    ck_tile::make_tuple(ck_tile::number<kValueTile>{},
                                        ck_tile::number<kChunkSize>{}),
                    ck_tile::multi_index<2>{0, 0},
                    tile.get_tile_distribution());
                if(full_chunk)
                    ck_tile::load_tile(tile,
                                       window,
                                       ck_tile::bool_constant<false>{});
                else
                    ck_tile::load_tile(tile,
                                       window,
                                       ck_tile::bool_constant<true>{});

                return tile;
            }();

            auto compute_and_stage = [&](const auto& q_gated,
                                         const auto& score_areg,
                                         ck_tile::index_t epilogue_offset, ck_tile::index_t row_begin) {
                auto hist = QHGemm::MakeCBlockTile();
                ck_tile::clear_tile(hist);
                qh_gemm(hist, q_gated, h_breg);
                auto local = PVGemm::MakeCBlockTile();
                ck_tile::clear_tile(local);
                pv_gemm(local, score_areg, v_breg);

                auto output_float =
                    Policy::template MakeCOutputLayout<QHProblem, QHPolicy>(hist);
                auto local_logical =
                    Policy::template MakeCOutputLayout<PVProblem, PVPolicy>(local);
                if constexpr(std::is_same_v<DataType, ck_tile::fp16_t> && Problem::kPairedEpilogue && GroupSize == 4)
                {
                    ck_tile::sweep_tile(output_float, [&](auto idx) {
                        const auto coord=ck_tile::get_x_indices_from_distributed_indices(
                            output_float.get_tile_distribution(),idx);
                        output_float(idx) *= gate_lds[kChunkSize+row_begin+coord[ck_tile::number<0>{}]];
                    });
                }
                ck_tile::tile_elementwise_inout(
                    [&](auto& x, const auto& y) { x = (x + y) * args.scale; },
                    output_float,
                    local_logical);
                auto output_tile = ck_tile::gdn_cast_tile<DataType>(output_float);

                constexpr auto epilogue_desc =
                    Policy::template MakePaddedRowMajorLdsDescriptor<
                        kRowTile,
                        kValueTile,
                        0>();
                auto epilogue_view =
                    ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                        epilogue_lds + epilogue_offset, epilogue_desc);
                auto epilogue_window = ck_tile::make_tile_window(
                    epilogue_view,
                    ck_tile::make_tuple(ck_tile::number<kRowTile>{},
                                        ck_tile::number<kValueTile>{}),
                    {0, 0},
                    output_tile.get_tile_distribution());
                ck_tile::store_tile(epilogue_window, output_tile);
            };

            if constexpr(kUsePairedEpilogue)
            {
                // Stage both row tiles into disjoint LDS regions, then let all
                // 256 lanes perform one useful vec8 output store.
                constexpr ck_tile::index_t kEpilogueTileElements =
                    kRowTile * kValueTile;
                compute_and_stage(q_gated0, score_areg0, 0, 0);
                compute_and_stage(q_gated1, score_areg1, kEpilogueTileElements, kRowTile);
                ck_tile::block_sync_lds();
                constexpr auto o_copy_distribution =
                    Policy::template MakeOutputCopyDistribution<kChunkSize, kValueTile>();
                constexpr auto o_lds_desc =
                    Policy::template MakePaddedRowMajorLdsDescriptor<
                        kChunkSize,
                        kValueTile,
                        0>();
                auto o_lds_view =
                    ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                        epilogue_lds, o_lds_desc);
                auto o_lds_window = ck_tile::make_tile_window(
                    o_lds_view,
                    ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                                        ck_tile::number<kValueTile>{}),
                    ck_tile::multi_index<2>{0, 0},
                    o_copy_distribution);
                auto o_copy_tile = ck_tile::load_tile(o_lds_window);
                ck_tile::block_sync_lds();

                DataType* o_base =
                    args.o +
                    (static_cast<int64_t>(token_begin) * args.num_value_heads + vh) *
                        kValueDim +
                    value_begin;
                auto o_dram_view =
                    ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                        o_base,
                        ck_tile::make_tuple(valid_tokens,
                                            ck_tile::number<kValueTile>{}),
                        ck_tile::make_tuple(args.num_value_heads * kValueDim,
                                            ck_tile::number<1>{}),
                        ck_tile::number<8>{},
                        ck_tile::number<1>{});
                auto o_padded_view = ck_tile::pad_tensor_view(
                    o_dram_view,
                    ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                                        ck_tile::number<kValueTile>{}),
                    ck_tile::sequence<true, false>{});
                auto o_dram_window = ck_tile::make_tile_window(
                    o_padded_view,
                    ck_tile::make_tuple(ck_tile::number<kChunkSize>{},
                                        ck_tile::number<kValueTile>{}),
                    ck_tile::multi_index<2>{0, 0},
                    o_copy_distribution);
                ck_tile::store_tile(o_dram_window, o_copy_tile);
            }
            else
            {
                auto stage_and_store_row = [&](const auto& q_gated,
                                               const auto& score_areg,
                                               ck_tile::index_t row_begin) {
                    compute_and_stage(q_gated, score_areg, 0, row_begin);
                    ck_tile::block_sync_lds();

                    constexpr auto o_copy_distribution =
                        Policy::template MakeOutputCopyDistribution<
                            kRowTile, kValueTile>();
                    constexpr auto o_lds_desc =
                        Policy::template MakePaddedRowMajorLdsDescriptor<
                            kRowTile, kValueTile, 0>();
                    auto o_lds_view =
                        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                            epilogue_lds, o_lds_desc);
                    auto o_lds_window = ck_tile::make_tile_window(
                        o_lds_view,
                        ck_tile::make_tuple(ck_tile::number<kRowTile>{},
                                            ck_tile::number<kValueTile>{}),
                        ck_tile::multi_index<2>{0, 0},
                        o_copy_distribution);
                    auto o_copy_tile = ck_tile::load_tile(o_lds_window);
                    ck_tile::block_sync_lds();

                    DataType* o_base =
                        args.o +
                        (static_cast<int64_t>(token_begin + row_begin) *
                             args.num_value_heads +
                         vh) *
                            kValueDim +
                        value_begin;
                    const ck_tile::index_t o_valid_rows =
                        row_begin < valid_tokens ? valid_tokens - row_begin : 0;
                    auto o_dram_view =
                        ck_tile::make_naive_tensor_view<
                            ck_tile::address_space_enum::global>(
                            o_base,
                            ck_tile::make_tuple(o_valid_rows,
                                                ck_tile::number<kValueTile>{}),
                            ck_tile::make_tuple(args.num_value_heads * kValueDim,
                                                ck_tile::number<1>{}),
                            ck_tile::number<4>{},
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
                };

                stage_and_store_row(q_gated0, score_areg0, 0);
                // The padded tensor view must not be rooted at an address past
                // the logical sequence.  This branch is CTA-uniform and also
                // avoids the second-row epilogue for a <=32-token tail chunk.
                if(valid_tokens > kRowTile)
                    stage_and_store_row(q_gated1, score_areg1, kRowTile);
            }
        }
    }
}
