// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/unified_attention/pipeline/unified_attention_3d_problem.hpp"

namespace ua {

template <typename Problem_>
struct UnifiedAttentionPipeline
{
    using Problem     = Problem_;
    using DataType    = typename Problem::DataType;
    using FmhaProblem = typename Problem::FmhaProblem;
    using Policy      = typename Problem::Policy;

    static constexpr ck_tile::index_t kM       = Problem::kM;
    static constexpr ck_tile::index_t kN       = Problem::kN;
    static constexpr ck_tile::index_t kK0      = Problem::kK0;
    static constexpr ck_tile::index_t kK1      = Problem::kK1;
    static constexpr ck_tile::index_t kHeadDim = Problem::kHeadDim;
    static constexpr bool kGroupQHeads         = Problem::kGroupQHeads;
    static constexpr ck_tile::index_t kPagedBlockSize = Problem::kPagedBlockSize;
    static constexpr bool kSimpleDecode = Problem::kSimpleDecode;
    static constexpr bool kFastAlibi = Problem::kFastAlibi;
    static constexpr ck_tile::index_t kVSmemStride = Problem::kVSmemStride;

    // CK Tile's block_tile_reduce_sync only reduces inside one wave. Grouped
    // decode splits the sequence tile over two N-waves, so combine those
    // per-wave row reductions through a small LDS scratch tile. The ownership
    // test mirrors block_tile_reduce_sync: only the lane whose reduced
    // replication indices are all zero publishes a row value for its wave.
    template <typename Tile, typename ReduceFunc>
    CK_TILE_DEVICE void sync_grouped_row_reduce(Tile& tile,
                                                const ReduceFunc& reduce_func,
                                                float identity,
                                                void* smem_ptr,
                                                int scratch_slot) const
    {
        ck_tile::block_tile_reduce_sync(
            tile, reduce_func, ck_tile::bool_constant<false>{});

        constexpr int num_warps = Problem::kBlockSize / ck_tile::get_warp_size();
        auto* scratch = reinterpret_cast<float*>(
                            static_cast<char*>(smem_ptr) + Problem::kDataSmemSize) +
                        scratch_slot * kM * num_warps;

        using TileType         = ck_tile::remove_cvref_t<Tile>;
        using Dstr             = typename TileType::StaticTileDistribution;
        using DstrEncode       = typename Dstr::DstrEncode;
        using DstrEncodeDetail = typename DstrEncode::detail;
        constexpr ck_tile::index_t n_dim_p = Dstr::get_num_of_dimension_p();
        constexpr ck_tile::index_t n_dim_r = Dstr::get_num_of_dimension_r();
        constexpr ck_tile::index_t lane_p  = n_dim_p - 1;
        const auto ps_idx =
            ck_tile::detail::get_partition_index(tile.get_tile_distribution());
        const auto rs_idx =
            tile.get_tile_distribution().calculate_rs_index_from_ps_index(ps_idx);
        bool owns_reduced_value = true;
        ck_tile::static_for<0, n_dim_r, 1>{}([&](auto r_dim) {
            if constexpr(DstrEncodeDetail::does_p_own_r_[lane_p][r_dim])
                owns_reduced_value = owns_reduced_value && rs_idx[r_dim] == 0;
        });

        if(owns_reduced_value)
        {
            // Exactly one lane per (row, wave) owns the reduced value. Every
            // logical row exists even when it is masked, in which case its
            // reduction already equals identity. Therefore all scratch slots
            // are overwritten unconditionally and a separate clear phase is
            // unnecessary.
            const int warp = ck_tile::get_warp_id();
            ck_tile::sweep_tile(tile, [&](auto idx) {
                const auto x = ck_tile::get_x_indices_from_distributed_indices(
                    tile.get_tile_distribution(), idx);
                const int row = x[ck_tile::number<0>{}];
                scratch[row * num_warps + warp] = tile[idx];
            });
        }
        ck_tile::block_sync_lds();

        ck_tile::sweep_tile(tile, [&](auto idx) {
            const auto x = ck_tile::get_x_indices_from_distributed_indices(
                tile.get_tile_distribution(), idx);
            const int row = x[ck_tile::number<0>{}];
            float value   = identity;
            ck_tile::static_for<0, num_warps, 1>{}([&](auto warp) {
                value = reduce_func(value, scratch[row * num_warps + warp]);
            });
            tile(idx) = value;
        });
        // Max and sum use separate slots. The second V half ends in a CTA
        // barrier before the next tile, so no wave can overwrite either slot
        // while a slower wave is still reading it here.
    }

    template <typename Args>
    CK_TILE_DEVICE auto operator()(const Args& a,
                                   int seq,
                                   int local_block,
                                   int qhead,
                                   int segment,
                                   int num_segments,
                                   void* smem_ptr,
                                   int tile_override = -1,
                                   bool force_sink_segment = false) const
    {
        const int qlen       = a.cu_q[seq + 1] - a.cu_q[seq];
        const int seqlen     = a.seq_lens[seq];
        const int context    = seqlen - qlen;
        int numq_per_kv;
        if constexpr(kSimpleDecode)
            numq_per_kv = a.numq_per_kv;
        else
            numq_per_kv = a.num_q_heads / a.num_kv_heads;
        const int query_base = kGroupQHeads ? 0 : local_block * kM;
        const int kvhead = kGroupQHeads ? qhead : qhead / numq_per_kv;
        const int qhead_base = kGroupQHeads ? kvhead * numq_per_kv : qhead;

        int64_t q_s0, q_s1;
        int64_t k_s0, k_s1, k_s2, k_s3;
        int64_t v_s0, v_s1, v_s2, v_s3;
        if constexpr(kSimpleDecode)
        {
            q_s1 = a.head_size;
            q_s0 = static_cast<int64_t>(a.num_q_heads) * a.head_size;
            k_s3 = 1;
            k_s2 = a.head_size;
            k_s1 = static_cast<int64_t>(a.num_kv_heads) * a.head_size;
            k_s0 = static_cast<int64_t>(kPagedBlockSize) * k_s1;
            v_s0 = k_s0;
            v_s1 = k_s1;
            v_s2 = k_s2;
            v_s3 = k_s3;
        }
        else
        {
            q_s0 = a.q_s0;
            q_s1 = a.q_s1;
            k_s0 = a.k_s0;
            k_s1 = a.k_s1;
            k_s2 = a.k_s2;
            k_s3 = a.k_s3;
            v_s0 = a.v_s0;
            v_s1 = a.v_s1;
            v_s2 = a.v_s2;
            v_s3 = a.v_s3;
        }

        const auto* q = static_cast<const DataType*>(a.q);
        const auto* k = static_cast<const DataType*>(a.k);
        const auto* v = static_cast<const DataType*>(a.v);

        constexpr auto gemm_qk = Policy::template GetQKBlockGemm<FmhaProblem>();
        constexpr auto gemm_pv = Policy::template GetKVBlockGemm<FmhaProblem>();

        const int q_rows = kGroupQHeads ? numq_per_kv : qlen;
        const int64_t q_row_stride = kGroupQHeads ? q_s1 : q_s0;
        const auto* q_ptr = q + static_cast<int64_t>(a.cu_q[seq]) * q_s0 +
                            static_cast<int64_t>(qhead_base) * q_s1;
        const auto q_view_raw =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                q_ptr,
                ck_tile::make_tuple(q_rows, a.head_size),
                ck_tile::make_tuple(q_row_stride, ck_tile::number<1>{}),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
        const auto q_view = ck_tile::pad_tensor_view(
            q_view_raw,
            ck_tile::make_tuple(ck_tile::number<kM>{}, ck_tile::number<kHeadDim>{}),
            ck_tile::sequence<true, true>{});
        auto* lds = static_cast<DataType*>(smem_ptr);
        // Stage an exact divisor of the real head dimension. For dimensions
        // between 128 and 256, two equal stages avoid padding while keeping
        // each K stage within the LDS budget.
        constexpr int kKStageDim = kSimpleDecode
                                        ? (kHeadDim <= 128 ? kHeadDim
                                                           : (kHeadDim <= 192
                                                                  ? kHeadDim
                                                                  : 128))
                                        : kK0;
        constexpr int kKSmemStride =
            kSimpleDecode && kPagedBlockSize > 0 ? kKStageDim + 4 : kKStageDim;
        static_assert(kHeadDim % kKStageDim == 0);
        static_assert(kKStageDim % kK0 == 0);
        constexpr auto k_lds_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<kN>{},
                                ck_tile::number<kKStageDim>{}),
            ck_tile::make_tuple(ck_tile::number<kKSmemStride>{},
                                ck_tile::number<1>{}));
        const auto k_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            lds, k_lds_desc);
        auto k_lds_window = ck_tile::make_tile_window(
            k_lds_view,
            ck_tile::make_tuple(ck_tile::number<kN>{}, ck_tile::number<kK0>{}),
            {0, 0});
        constexpr auto v_lds_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<kHeadDim>{}, ck_tile::number<kK1>{}),
            ck_tile::make_tuple(ck_tile::number<kVSmemStride>{},
                                ck_tile::number<1>{}));
        const auto v_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            lds, v_lds_desc);
        auto v_lds_window = ck_tile::make_tile_window(
            v_lds_view,
            ck_tile::make_tuple(ck_tile::number<kHeadDim>{}, ck_tile::number<kK1>{}),
            {0, 0});
        constexpr int kPSmemStride =
            kSimpleDecode && kPagedBlockSize > 0 ? kN + 4 : kN;
        constexpr auto p_lds_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<kM>{}, ck_tile::number<kN>{}),
            ck_tile::make_tuple(ck_tile::number<kPSmemStride>{},
                                ck_tile::number<1>{}));
        const auto p_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            lds, p_lds_desc);
        auto p_lds_window = ck_tile::make_tile_window(
            p_lds_view,
            ck_tile::make_tuple(ck_tile::number<kM>{}, ck_tile::number<kN>{}),
            {0, 0});

        using SNativeAccTile = decltype(gemm_qk.MakeCBlockTile());
        using SOutputAccTile = decltype(gemm_qk.MakeCOutputLayout(SNativeAccTile{}));
        using STile = decltype(ck_tile::cast_tile<float>(SOutputAccTile{}));
        const auto f_max = [](auto x, auto y) { return ck_tile::max(x, y); };
        const auto f_sum = [](auto x, auto y) { return x + y; };
        using MLTile = decltype(ck_tile::block_tile_reduce<float>(
            STile{}, ck_tile::sequence<1>{}, f_max, 0.0f));
        using OAccTile = decltype(gemm_pv.MakeCBlockTile());

        auto m     = MLTile{};
        auto l     = MLTile{};
        auto o_acc = OAccTile{};
        ck_tile::set_tile(m, -ck_tile::numeric<float>::infinity());
        ck_tile::clear_tile(l);
        ck_tile::clear_tile(o_acc);

        if(a.use_sinks && (segment == 0 || force_sink_segment))
        {
            if constexpr(kGroupQHeads)
            {
                ck_tile::sweep_tile(m, [&](auto idx) {
                    const auto x = ck_tile::get_x_indices_from_distributed_indices(
                        m.get_tile_distribution(), idx);
                    const int row = x[ck_tile::number<0>{}];
                    m(idx) = row < numq_per_kv
                                 ? a.sinks[qhead_base + row]
                                 : -ck_tile::numeric<float>::infinity();
                    l(idx) = row < numq_per_kv ? 1.0f : 0.0f;
                });
            }
            else
            {
                const float sink = a.sinks[qhead];
                ck_tile::tile_elementwise_inout([=](auto& x) { x = sink; }, m);
                ck_tile::tile_elementwise_inout([](auto& x) { x = 1.0f; }, l);
            }
        }

        int tile_begin = 0;
        int tile_end   = (seqlen + kN - 1) / kN;
        if constexpr(kSimpleDecode)
        {
            const int last_query = kGroupQHeads
                                       ? 0
                                       : ck_tile::min(query_base + kM - 1, qlen - 1);
            const int last_key   = context + last_query;
            tile_end             = ck_tile::min(tile_end, last_key / kN + 1);
            if(a.sliding_window > 0)
            {
                const int first_key = context + query_base - a.sliding_window + 1;
                tile_begin = ck_tile::max(0, first_key / kN);
            }
        }
        else if(!a.use_mm_prefix)
        {
            const int last_query = kGroupQHeads
                                       ? 0
                                       : ck_tile::min(query_base + kM - 1, qlen - 1);
            const int last_key   = context + last_query;
            tile_end             = ck_tile::min(tile_end, last_key / kN + 1);
            if(a.sliding_window > 0)
            {
                const int first_key = context + query_base - a.sliding_window + 1;
                tile_begin = ck_tile::max(0, first_key / kN);
            }
        }
        if(num_segments > 1)
        {
            const int tiles_per_segment =
                (seqlen + num_segments * kN - 1) / (num_segments * kN);
            tile_begin = ck_tile::max(tile_begin, segment * tiles_per_segment);
            tile_end = ck_tile::min(tile_end, (segment + 1) * tiles_per_segment);
        }
        if(tile_override >= 0)
        {
            tile_begin = ck_tile::max(tile_begin, tile_override);
            tile_end   = ck_tile::min(tile_end, tile_override + 1);
        }

        const auto load_q_chunk = [&](auto i_k) {
            auto q_k = gemm_qk.MakeABlockTile();
            if constexpr(kSimpleDecode)
            {
                // MMAC A layout maps lane[3:0] to M and lane[5:4] to
                // four contiguous K elements. Load that FP16x4 directly;
                // the generic tile loader otherwise scalarizes the access.
                constexpr int kLoadVec = 4;
                using QVec = ck_tile::ext_vector_t<DataType, kLoadVec>;
                ck_tile::clear_tile(q_k);
                const int lane = static_cast<int>(threadIdx.x) %
                                 ck_tile::get_warp_size();
                const int row = lane & 15;
                const int dim_base = i_k * kK0 + (lane >> 4) * kLoadVec;
                if(row < q_rows)
                {
                    q_k.get_thread_buffer().template get_as<QVec>()(
                        ck_tile::number<0>{}) =
                        *reinterpret_cast<const QVec*>(
                            q_ptr + static_cast<int64_t>(row) * q_row_stride +
                            dim_base);
                }
            }
            else
            {
                const auto q_k_window = ck_tile::make_tile_window(
                    q_view,
                    ck_tile::make_tuple(ck_tile::number<kM>{},
                                        ck_tile::number<kK0>{}),
                    {query_base, i_k * kK0},
                    q_k.get_tile_distribution());
                ck_tile::load_tile(q_k, q_k_window);
            }
            return q_k;
        };

        for(int tile_index = tile_begin; tile_index < tile_end; ++tile_index)
        {
            const int key_base = tile_index * kN;
            auto s_acc         = SNativeAccTile{};
            ck_tile::clear_tile(s_acc);

            // For the thread-raked K distribution (64x16, vector width 4),
            // each thread owns four dimensions from exactly one sequence row.
            // Resolve that row's physical page once per tile, not once for
            // every head-dimension chunk.
            int64_t k_thread_base = 0;
            bool k_thread_valid   = false;
            if constexpr(kPagedBlockSize > 0)
            {
                constexpr int kLoadVec = 4;
                constexpr int kRowsPerWave =
                    ck_tile::get_warp_size() / (kK0 / kLoadVec);
                const int row = static_cast<int>(threadIdx.x) / (kK0 / kLoadVec);
                const int wave_row_base = ck_tile::get_warp_id() * kRowsPerWave;
                const int kpos = key_base + row;
                k_thread_valid = kpos < seqlen;
                if(k_thread_valid)
                {
                    const int physical =
                        a.block_table[static_cast<int64_t>(seq) * a.bt_s0 +
                                      (key_base + wave_row_base) / kPagedBlockSize];
                    k_thread_base =
                        static_cast<int64_t>(physical) * k_s0 +
                        static_cast<int64_t>(kpos % kPagedBlockSize) * k_s1 +
                        static_cast<int64_t>(kvhead) * k_s2;
                }
            }

            if constexpr(kSimpleDecode && kPagedBlockSize > 0)
            {
                constexpr int kLoadVec = 4;
                constexpr int kChunksPerStage = kKStageDim / kK0;
                constexpr int kNumStages = kHeadDim / kKStageDim;
                using KVec = ck_tile::ext_vector_t<DataType, kLoadVec>;

                const int row = static_cast<int>(threadIdx.x) / (kK0 / kLoadVec);
                const int dim_lane =
                    (static_cast<int>(threadIdx.x) % (kK0 / kLoadVec)) * kLoadVec;

                ck_tile::static_for<0, kNumStages, 1>{}([&](auto i_stage) {
#pragma unroll 12
                    for(int i = 0; i < kChunksPerStage; ++i)
                    {
                        const int local_dim = i * kK0 + dim_lane;
                        const int dim_base = i_stage * kKStageDim + local_dim;
                        ck_tile::thread_buffer<DataType, kLoadVec> value_buf;
                        ck_tile::static_for<0, kLoadVec, 1>{}([&](auto j) {
                            value_buf(j) = ck_tile::type_convert<DataType>(0.0f);
                        });
                        auto& value = value_buf.template get_as<KVec>();
                        if(k_thread_valid)
                        {
                            value(ck_tile::number<0>{}) =
                                *reinterpret_cast<const KVec*>(
                                    k + k_thread_base +
                                    static_cast<int64_t>(dim_base) * k_s3);
                        }
                        *reinterpret_cast<KVec*>(
                            lds + row * kKSmemStride + local_dim) =
                            value(ck_tile::number<0>{});
                    }
                    // Each N-wave writes and consumes its own 16 K rows. The
                    // reduction scratch lives after the data LDS region, so a
                    // faster wave cannot overwrite another wave's K rows.
#pragma unroll 12
                    for(int i = 0; i < kChunksPerStage; ++i)
                    {
                        const int i_k = i_stage * kChunksPerStage + i;
                        const auto q_k = load_q_chunk(i_k);
                        auto staged_k_window = ck_tile::make_tile_window(
                            k_lds_view,
                            ck_tile::make_tuple(ck_tile::number<kN>{},
                                                ck_tile::number<kK0>{}),
                            {0, i * kK0});
                        gemm_qk(s_acc, q_k, staged_k_window);
                    }
                    if constexpr(decltype(i_stage)::value + 1 < kNumStages)
                        ck_tile::lds_wait();
                });
            }
            else
            {
                ck_tile::static_for<0, kHeadDim / kK0, 1>{}([&](auto i_k) {
                    const auto q_k = load_q_chunk(i_k);
                    auto k_reg = ck_tile::make_static_distributed_tensor<DataType>(
                        Problem::MakeKLoadDistribution());
                    ck_tile::sweep_tile(k_reg, [&](auto idx) {
                        const auto x = ck_tile::get_x_indices_from_distributed_indices(
                            k_reg.get_tile_distribution(), idx);
                        const int row  = x[ck_tile::number<0>{}];
                        const int dim  = i_k * kK0 + x[ck_tile::number<1>{}];
                        const int kpos = key_base + row;
                        DataType value = ck_tile::type_convert<DataType>(0.0f);
                        if(kpos < seqlen)
                        {
                            const int physical =
                                a.block_table[static_cast<int64_t>(seq) * a.bt_s0 +
                                              kpos / a.block_size];
                            const int64_t offset =
                                static_cast<int64_t>(physical) * k_s0 +
                                static_cast<int64_t>(kpos % a.block_size) * k_s1 +
                                static_cast<int64_t>(kvhead) * k_s2 +
                                static_cast<int64_t>(dim) * k_s3;
                            value = k[offset];
                        }
                        k_reg(idx) = value;
                    });

                    ck_tile::block_sync_lds();
                    ck_tile::store_tile(k_lds_window, k_reg);
                    ck_tile::block_sync_lds();
                    gemm_qk(s_acc, q_k, k_lds_window);
                });
            }

            const auto s_output = gemm_qk.MakeCOutputLayout(s_acc);
            auto s = ck_tile::cast_tile<float>(s_output);
            ck_tile::sweep_tile(s, [&](auto idx) {
                const auto x = ck_tile::get_x_indices_from_distributed_indices(
                    s.get_tile_distribution(), idx);
                const int row    = x[ck_tile::number<0>{}];
                const int col    = x[ck_tile::number<1>{}];
                const int qlocal = kGroupQHeads ? 0 : query_base + row;
                const int row_qhead = qhead_base + (kGroupQHeads ? row : 0);
                const int qabs   = context + qlocal;
                const int kpos   = key_base + col;

                const bool row_valid = kGroupQHeads ? row < numq_per_kv : qlocal < qlen;
                bool allowed = row_valid && kpos < seqlen;
                const bool causal_allowed =
                    kpos <= qabs &&
                    (a.sliding_window <= 0 || qabs - kpos < a.sliding_window);
                if constexpr(kSimpleDecode)
                    allowed = allowed && causal_allowed;
                else
                    allowed = allowed &&
                              (causal_allowed || mm_allowed(a, seq, qabs, kpos));

                if(!allowed)
                {
                    s(idx) = -ck_tile::numeric<float>::infinity();
                    return;
                }

                float score = s[idx] * a.scale;
                if constexpr(kSimpleDecode)
                {
                    if constexpr(kFastAlibi)
                    {
                        // Match Triton's linear ALiBi specialization without
                        // falling back to the generic paged-address path.
                        score += a.alibi[row_qhead] *
                                 static_cast<float>(kpos - context);
                    }
                }
                else
                {
                    if(a.softcap > 0.0f)
                        score = a.softcap * tanhf(score / a.softcap);
                    if(a.use_alibi)
                    {
                        const int relative = kpos - qabs;
                        const float offset = a.alibi_sqrt
                            ? (relative <= 0 ? -sqrtf(static_cast<float>(-relative)) : 0.0f)
                            : static_cast<float>(kpos - context);
                        score += a.alibi[row_qhead] * offset;
                    }
                    if(a.use_qq_bias)
                    {
                        const int key_relative = kpos - context;
                        if(key_relative >= 0 && key_relative < qlen)
                        {
                            const auto* bias = static_cast<const DataType*>(a.qq_bias);
                            score += ck_tile::type_convert<float>(
                                bias[static_cast<int64_t>(qlocal) * a.bias_s0 + key_relative]);
                        }
                    }
                }
                s(idx) = score;
            });

            auto m_local = ck_tile::block_tile_reduce<float>(
                s,
                ck_tile::sequence<1>{},
                f_max,
                -ck_tile::numeric<float>::infinity());
            if constexpr(kGroupQHeads)
                sync_grouped_row_reduce(
                    m_local,
                    f_max,
                    -ck_tile::numeric<float>::infinity(),
                    smem_ptr,
                    0);
            else
                ck_tile::block_tile_reduce_sync(
                    m_local, f_max, ck_tile::bool_constant<false>{});

            const auto m_old = m;
            ck_tile::tile_elementwise_inout(
                [](auto& out, auto old_m, auto local_m) {
                    out = ck_tile::max(old_m, local_m);
                },
                m,
                m_old,
                m_local);

            auto p = ck_tile::make_static_distributed_tensor<float>(s.get_tile_distribution());
            constexpr auto p_spans = decltype(p)::get_distributed_spans();
            ck_tile::sweep_tile_span(p_spans[ck_tile::number<0>{}], [&](auto idx0) {
                constexpr auto row_idx = ck_tile::make_tuple(idx0);
                const float row_m = m[row_idx] == -ck_tile::numeric<float>::infinity()
                                        ? 0.0f
                                        : m[row_idx];
                ck_tile::sweep_tile_span(p_spans[ck_tile::number<1>{}], [&](auto idx1) {
                    constexpr auto idx = ck_tile::make_tuple(idx0, idx1);
                    p(idx) = expf(s[idx] - row_m);
                });
            });

            auto row_sum = ck_tile::block_tile_reduce<float>(
                p, ck_tile::sequence<1>{}, f_sum, 0.0f);
            if constexpr(kGroupQHeads)
                sync_grouped_row_reduce(row_sum, f_sum, 0.0f, smem_ptr, 1);
            else
                ck_tile::block_tile_reduce_sync(
                    row_sum, f_sum, ck_tile::bool_constant<false>{});

            constexpr auto o_spans = OAccTile::get_distributed_spans();
            ck_tile::sweep_tile_span(o_spans[ck_tile::number<0>{}], [&](auto idx0) {
                constexpr auto row_idx = ck_tile::make_tuple(idx0);
                const float new_m = m[row_idx] == -ck_tile::numeric<float>::infinity()
                                        ? 0.0f
                                        : m[row_idx];
                const float alpha = expf(m_old[row_idx] - new_m);
                l(row_idx) = alpha * l[row_idx] + row_sum[row_idx];
                ck_tile::sweep_tile_span(o_spans[ck_tile::number<1>{}], [&](auto idx1) {
                    constexpr auto idx = ck_tile::make_tuple(idx0, idx1);
                    o_acc(idx) *= alpha;
                });
            });

            const auto p_output = ck_tile::cast_tile<DataType>(p);
            ck_tile::store_tile(p_lds_window, p_output);
            ck_tile::block_sync_lds();

            // PV consumes K1=32 columns per MMAC step. When the sequence tile
            // is 64, preserve both P halves in registers before V overwrites
            // the shared LDS buffer.
            auto p_gemm0 = gemm_pv.MakeABlockTile();
            auto p_gemm0_window = ck_tile::make_tile_window(
                p_lds_view,
                ck_tile::make_tuple(ck_tile::number<kM>{}, ck_tile::number<kK1>{}),
                {0, 0},
                p_gemm0.get_tile_distribution());
            ck_tile::load_tile(p_gemm0, p_gemm0_window);

            auto p_gemm1 = gemm_pv.MakeABlockTile();
            if constexpr(kN > kK1)
            {
                auto p_gemm1_window = ck_tile::make_tile_window(
                    p_lds_view,
                    ck_tile::make_tuple(ck_tile::number<kM>{}, ck_tile::number<kK1>{}),
                    {0, kK1},
                    p_gemm1.get_tile_distribution());
                ck_tile::load_tile(p_gemm1, p_gemm1_window);
            }
            ck_tile::block_sync_lds();

            const auto accumulate_v = [&](auto i_v, const auto& p_gemm) {
                if constexpr(kPagedBlockSize > 0)
                {
                    constexpr int kLoadVec = 4;
                    constexpr int kNumWarps =
                        Problem::kBlockSize / ck_tile::get_warp_size();
                    constexpr int kDChunk = 16;
                    constexpr int kDIter = kHeadDim / (kNumWarps * kDChunk);
                    constexpr int kDimsPerWave = kDIter * kDChunk;
                    constexpr int kXThreads = kDimsPerWave / kLoadVec;
                    constexpr int kVecsPerWave = kK1 * kXThreads;
                    constexpr int kVecsPerLane =
                        kVecsPerWave / ck_tile::get_warp_size();
                    static_assert(kHeadDim % (kNumWarps * kDChunk) == 0);
                    static_assert(kVecsPerWave % ck_tile::get_warp_size() == 0);
                    static_assert(kK1 % kPagedBlockSize == 0);
                    static_assert(kK1 / kPagedBlockSize <= 2);
                    using VVec = ck_tile::ext_vector_t<DataType, kLoadVec>;

                    ck_tile::thread_buffer<DataType, kVecsPerLane * kLoadVec> v_reg;
                    ck_tile::static_for<0, kVecsPerLane * kLoadVec, 1>{}([&](auto i) {
                        v_reg(i) = ck_tile::type_convert<DataType>(0.0f);
                    });
                    auto& dst_vec = v_reg.template get_as<VVec>();

                    const int lane = static_cast<int>(threadIdx.x) %
                                     ck_tile::get_warp_size();
                    const int warp = ck_tile::get_warp_id();
                    const int half_kpos = key_base + i_v * kK1;
                    int64_t v_page_base0 = 0;
                    int64_t v_page_base1 = 0;
                    if(half_kpos < seqlen)
                    {
                        const int physical =
                            a.block_table[static_cast<int64_t>(seq) * a.bt_s0 +
                                          half_kpos / kPagedBlockSize];
                        v_page_base0 =
                            static_cast<int64_t>(physical) * v_s0 +
                            static_cast<int64_t>(kvhead) * v_s2;
                    }
                    if constexpr(kK1 > kPagedBlockSize)
                    {
                        if(half_kpos + kPagedBlockSize < seqlen)
                        {
                            const int physical =
                                a.block_table[static_cast<int64_t>(seq) * a.bt_s0 +
                                              half_kpos / kPagedBlockSize + 1];
                            v_page_base1 =
                                static_cast<int64_t>(physical) * v_s0 +
                                static_cast<int64_t>(kvhead) * v_s2;
                        }
                    }

                    ck_tile::static_for<0, kVecsPerLane, 1>{}([&](auto iy) {
                        const int linear = lane + iy * ck_tile::get_warp_size();
                        const int row = linear / kXThreads;
                        const int wave_dim = (linear % kXThreads) * kLoadVec;
                        const int dim_base = (wave_dim / kDChunk) *
                                                 (kNumWarps * kDChunk) +
                                             warp * kDChunk + wave_dim % kDChunk;
                        const int kpos = half_kpos + row;
                        if(kpos < seqlen)
                        {
                            const int64_t page_base = [&]() {
                                if constexpr(kK1 > kPagedBlockSize)
                                    return row < kPagedBlockSize ? v_page_base0
                                                                 : v_page_base1;
                                else
                                    return v_page_base0;
                            }();
                            dst_vec(iy) = *reinterpret_cast<const VVec*>(
                                v + page_base +
                                static_cast<int64_t>(row % kPagedBlockSize) * v_s1 +
                                static_cast<int64_t>(dim_base) * v_s3);
                        }
                    });

                    ck_tile::static_for<0, kVecsPerLane, 1>{}([&](auto iy) {
                        const int linear = lane + iy * ck_tile::get_warp_size();
                        const int row = linear / kXThreads;
                        const int wave_dim = (linear % kXThreads) * kLoadVec;
                        const int dim_base = (wave_dim / kDChunk) *
                                                 (kNumWarps * kDChunk) +
                                             warp * kDChunk + wave_dim % kDChunk;
                        ck_tile::static_for<0, kLoadVec, 1>{}([&](auto j) {
                            lds[(dim_base + j) * kVSmemStride + row] =
                                v_reg(iy * kLoadVec + j);
                        });
                    });
                }
                else
                {
                    auto v_reg = ck_tile::make_static_distributed_tensor<DataType>(
                        Problem::MakeVLoadDistribution());
                    ck_tile::sweep_tile(v_reg, [&](auto idx) {
                        const auto x = ck_tile::get_x_indices_from_distributed_indices(
                            v_reg.get_tile_distribution(), idx);
                        const int dim  = x[ck_tile::number<0>{}];
                        const int kpos = key_base + i_v * kK1 +
                                         x[ck_tile::number<1>{}];
                        DataType value = ck_tile::type_convert<DataType>(0.0f);
                        if(kpos < seqlen)
                        {
                            const int physical =
                                a.block_table[static_cast<int64_t>(seq) * a.bt_s0 +
                                              kpos / a.block_size];
                            const int64_t offset =
                                static_cast<int64_t>(physical) * v_s0 +
                                static_cast<int64_t>(kpos % a.block_size) * v_s1 +
                                static_cast<int64_t>(kvhead) * v_s2 +
                                static_cast<int64_t>(dim) * v_s3;
                            value = v[offset];
                        }
                        v_reg(idx) = value;
                    });

                    ck_tile::block_sync_lds();
                    ck_tile::store_tile(v_lds_window, v_reg);
                }
                if constexpr(kPagedBlockSize == 0)
                    ck_tile::block_sync_lds();

                gemm_pv(o_acc, p_gemm, v_lds_window);
                if constexpr(kPagedBlockSize == 0)
                    ck_tile::block_sync_lds();
            };
            accumulate_v(ck_tile::number<0>{}, p_gemm0);
            if constexpr(kN > kK1)
                accumulate_v(ck_tile::number<1>{}, p_gemm1);
        }

        const auto o_output = gemm_pv.MakeCOutputLayout(o_acc);
        return ck_tile::make_tuple(o_output, m, l, tile_begin < tile_end);
    }
};

} // namespace ua
