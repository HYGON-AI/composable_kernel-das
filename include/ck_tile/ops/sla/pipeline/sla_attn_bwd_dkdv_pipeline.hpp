// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sla/kernel/sla_attn_bwd_dkdv_kernel.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dkdv_problem.hpp"
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dkdv_policy.hpp"

namespace ck_tile {
namespace example {
namespace sla {

template <typename Problem, typename Policy = SlaAttnBwdDkdvDefaultPolicy<Problem>>
struct SlaAttnBwdDkdvPipeline
{
    using QDataType     = typename Problem::QDataType;
    using KDataType     = typename Problem::KDataType;
    using VDataType     = typename Problem::VDataType;
    using OGradDataType = typename Problem::OGradDataType;
    using KGradDataType = typename Problem::KGradDataType;
    using VGradDataType = typename Problem::VGradDataType;
    using AccDataType   = typename Problem::AccDataType;
    using LSEDataType   = typename Problem::LSEDataType;

    static constexpr index_t BlockM          = Problem::BlockM;
    static constexpr index_t BlockN          = Problem::BlockN;
    static constexpr index_t HeadDim         = Problem::HeadDim;
    static constexpr index_t MaxNnz          = Problem::MaxNnz;
    static constexpr index_t ThreadsPerBlock = Problem::ThreadsPerBlock;

    template <typename QDramView,
              typename KDramView,
              typename VDramView,
              typename DoDramView,
              typename DkDramView,
              typename DvDramView>
    CK_TILE_DEVICE void operator()(const SlaAttnBwdDkdvKernelArgument& args,
                                   const QDramView& q_dram_view,
                                   const KDramView& k_dram_view,
                                   const VDramView& v_dram_view,
                                   const DoDramView& do_dram_view,
                                   DkDramView& dk_dram_view,
                                   DvDramView& dv_dram_view,
                                   ck_tile::index_t kv_block,
                                   ck_tile::index_t off_hz,
                                   ck_tile::index_t batch,
                                   ck_tile::index_t head,
                                   ck_tile::index_t seqlen)
    {
        execute_impl<true, true>(args,
                             q_dram_view,
                             k_dram_view,
                             v_dram_view,
                             do_dram_view,
                             dk_dram_view,
                             dv_dram_view,
                             kv_block,
                             off_hz,
                             batch,
                             head,
                             seqlen);
    }

    template <bool ComputeDV,
              bool ComputeDK,
              typename QDramView,
              typename KDramView,
              typename VDramView,
              typename DoDramView,
              typename DkDramView,
              typename DvDramView>
    CK_TILE_DEVICE static void execute_impl(const SlaAttnBwdDkdvKernelArgument& args,
                                        const QDramView& q_dram_view,
                                        const KDramView& k_dram_view,
                                        const VDramView& v_dram_view,
                                        const DoDramView& do_dram_view,
                                        DkDramView& dk_dram_view,
                                        DvDramView& dv_dram_view,
                                        ck_tile::index_t kv_block,
                                        ck_tile::index_t off_hz,
                                        ck_tile::index_t /*batch*/,
                                        ck_tile::index_t /*head*/,
                                        ck_tile::index_t seqlen)
    {
        constexpr auto qk_chunk_asmem_breg_gemm = typename Policy::QKChunkASmemBRegBlockGemm{};
        constexpr auto dv_breg_gemm = typename Policy::DVBRegBlockGemm{};
        constexpr auto dv_k32_breg_gemm = typename Policy::DVK32BRegBlockGemm{};
        constexpr auto dp_k32_gemm = typename Policy::DPK32BlockGemm{};
        constexpr auto dp_k32_areg_breg_gemm = typename Policy::DPK32ARegBRegBlockGemm{};
        constexpr auto dk_k32_breg_gemm = typename Policy::DKK32BRegBlockGemm{};
        const ck_tile::index_t start_n = kv_block * BlockN;
        constexpr ck_tile::index_t PublicBlockN = 64;
        static_assert(PublicBlockN % BlockN == 0,
                      "internal dK/dV BlockN must divide the public KV block");
        const ck_tile::index_t public_kv_block = start_n / PublicBlockN;

        const AccDataType* delta_ptr =
            static_cast<const AccDataType*>(args.delta_ptr) +
            static_cast<ck_tile::long_index_t>(off_hz) * args.stride_delta_z;

        const int32_t* rlut_ptr =
            args.rlut_ptr + static_cast<ck_tile::long_index_t>(off_hz) * args.stride_rlutz +
            static_cast<ck_tile::long_index_t>(public_kv_block) * args.stride_rlutn;
        const int32_t num_active =
            args.rlut_size_ptr[static_cast<ck_tile::long_index_t>(off_hz) *
                                   args.stride_rlut_size_z +
                               static_cast<ck_tile::long_index_t>(public_kv_block) *
                                   args.stride_rlut_size_n];
        if(num_active <= 0)
        {
            return;
        }

        const bool is_text_block = public_kv_block >= args.text_block_start;
        const LSEDataType* lse_ptr =
            static_cast<const LSEDataType*>(is_text_block ? args.lse_text_ptr : args.lse_ptr) +
            static_cast<ck_tile::long_index_t>(off_hz) * args.stride_lz;
        constexpr float qk_scale = ck_tile::log2e_v<float>;
        const float scale_log2e  = args.sm_scale * qk_scale;
        constexpr int QVecLoadIters = (BlockM * HeadDim) / (8 * ThreadsPerBlock);
        constexpr int KVVecLoadIters = (BlockN * HeadDim) / (8 * ThreadsPerBlock);
        constexpr int QKChunk = SlaMmacQKChunkConfig::kK;
        constexpr int QKSubN = 32;
        constexpr int QKDChunks = HeadDim / QKChunk;
        static_assert(HeadDim == QKDChunks * QKChunk, "dV split-QK path expects even chunks");
        static_assert(QKDChunks == 4, "dV fwd-style QK path currently expects four 32-wide chunks");
        static_assert(BlockN == QKSubN || BlockN == 2 * QKSubN,
                      "dK/dV supports internal BlockN=32 or 64");
        constexpr int TransposedMVecIters = BlockM / 16;

        __shared__ typename Policy::LdsStorage smem;

        const bool block_is_boundary = (start_n + BlockN > seqlen) || (args.D < HeadDim);
        auto dv_acc = decltype(dv_breg_gemm.MakeCBlockTile()){};
        auto dk_acc = decltype(dk_k32_breg_gemm.MakeCBlockTile()){};
        if constexpr(ComputeDV)
        {
            ck_tile::clear_tile(dv_acc);
        }
        if constexpr(ComputeDK)
        {
            ck_tile::clear_tile(dk_acc);
        }

        QDataType* q_async_smem = smem.q_async;

        auto k_breg_n0_d0 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n0_d1 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n0_d2 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n0_d3 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n1_d0 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n1_d1 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n1_d2 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n1_d3 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto v_areg_d0 = dp_k32_areg_breg_gemm.MakeABlockTile();
        auto v_areg_d1 = dp_k32_areg_breg_gemm.MakeABlockTile();
        auto v_areg_d2 = dp_k32_areg_breg_gemm.MakeABlockTile();
        auto v_areg_d3 = dp_k32_areg_breg_gemm.MakeABlockTile();
        if constexpr(Policy::LeanFusedStorage)
        {
            static_assert(BlockN == BlockM,
                          "K/V direct-to-LDS staging reuses the 64-row Q geometry");
            static_assert(std::is_same_v<QDataType, KDataType> &&
                              std::is_same_v<QDataType, VDataType>,
                          "K/V staging reuses the Q-typed LDS buffer");

            auto load_kv_tile_to_lds = [&](const auto& dram_view) {
                auto dram_window = ck_tile::make_tile_window(
                    dram_view,
                    ck_tile::make_tuple(ck_tile::number<BlockN>{},
                                        ck_tile::number<QKChunk>{}),
                    ck_tile::multi_index<2>{start_n, 0},
                    MakeBwdQAsyncDramDistribution<BlockN, QKChunk>());

                auto load_kv_chunk_to_lds = [&](auto buf_num) {
                    constexpr int buf = decltype(buf_num)::value;

                    auto lds_store_view =
                        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                            q_async_smem,
                            MakeBwdQAsyncLdsStoreDescriptor<BlockN, QKChunk, buf>());
                    auto lds_store_window = ck_tile::make_tile_window(
                        lds_store_view,
                        MakeBwdQAsyncLdsStoreDescriptor<BlockN, QKChunk, buf>().get_lengths(),
                        ck_tile::multi_index<3>{0, 0, 0});
                    ck_tile::async_load_tile_by_inline_asm(
                        lds_store_window,
                        dram_window,
                        ck_tile::bool_constant<true>{},
                        ck_tile::bool_constant<false>{});
                    if constexpr(buf + 1 < QKDChunks)
                    {
                        ck_tile::move_tile_window(
                            dram_window, ck_tile::multi_index<2>{0, QKChunk});
                    }
                };

                load_kv_chunk_to_lds(ck_tile::number<0>{});
                load_kv_chunk_to_lds(ck_tile::number<1>{});
                load_kv_chunk_to_lds(ck_tile::number<2>{});
                load_kv_chunk_to_lds(ck_tile::number<3>{});
            };

            auto load_k_chunk_breg_from_lds = [&](ck_tile::index_t n_base, auto buf_num) {
                constexpr int buf = decltype(buf_num)::value;
                auto k_breg       = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
                auto k_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    q_async_smem, MakeBwdQAsyncLdsLoadDescriptor<BlockN, QKChunk>());
                auto k_lds_window = ck_tile::make_tile_window(
                    k_lds_view,
                    ck_tile::make_tuple(ck_tile::number<QKSubN>{},
                                        ck_tile::number<QKChunk>{}),
                    ck_tile::multi_index<2>{buf * BlockN + n_base, 0},
                    k_breg.get_tile_distribution());
                return ck_tile::load_tile(k_lds_window);
            };

            auto load_v_chunk_areg_from_lds = [&](auto buf_num) {
                constexpr int buf = decltype(buf_num)::value;
                auto v_areg       = dp_k32_areg_breg_gemm.MakeABlockTile();
                auto v_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    q_async_smem, MakeBwdQAsyncLdsLoadDescriptor<BlockN, QKChunk>());
                auto v_lds_window = ck_tile::make_tile_window(
                    v_lds_view,
                    ck_tile::make_tuple(ck_tile::number<BlockN>{},
                                        ck_tile::number<QKChunk>{}),
                    ck_tile::multi_index<2>{buf * BlockN, 0},
                    v_areg.get_tile_distribution());
                return ck_tile::load_tile(v_lds_window);
            };

            if(!block_is_boundary)
            {
                load_kv_tile_to_lds(k_dram_view);
                ck_tile::async_load_fence(0);
                SlaBwdBlockSyncLdsLight();

                k_breg_n0_d0 =
                    load_k_chunk_breg_from_lds(0, ck_tile::number<0>{});
                k_breg_n0_d1 =
                    load_k_chunk_breg_from_lds(0, ck_tile::number<1>{});
                k_breg_n0_d2 =
                    load_k_chunk_breg_from_lds(0, ck_tile::number<2>{});
                k_breg_n0_d3 =
                    load_k_chunk_breg_from_lds(0, ck_tile::number<3>{});
                if constexpr(BlockN == 2 * QKSubN)
                {
                    k_breg_n1_d0 =
                        load_k_chunk_breg_from_lds(QKSubN, ck_tile::number<0>{});
                    k_breg_n1_d1 =
                        load_k_chunk_breg_from_lds(QKSubN, ck_tile::number<1>{});
                    k_breg_n1_d2 =
                        load_k_chunk_breg_from_lds(QKSubN, ck_tile::number<2>{});
                    k_breg_n1_d3 =
                        load_k_chunk_breg_from_lds(QKSubN, ck_tile::number<3>{});
                }
                if constexpr(ComputeDK)
                {
                    ck_tile::block_sync_lds();
                    load_kv_tile_to_lds(v_dram_view);
                    ck_tile::async_load_fence(0);
                    SlaBwdBlockSyncLdsLight();

                    v_areg_d0 = load_v_chunk_areg_from_lds(ck_tile::number<0>{});
                    v_areg_d1 = load_v_chunk_areg_from_lds(ck_tile::number<1>{});
                    v_areg_d2 = load_v_chunk_areg_from_lds(ck_tile::number<2>{});
                    v_areg_d3 = load_v_chunk_areg_from_lds(ck_tile::number<3>{});
                }

                // K/V fragments are now persistent in registers; make the shared union
                // available to the per-active-Q pipeline.
                ck_tile::block_sync_lds();
            }
        }

        for(ck_tile::index_t active_i = 0; active_i < num_active; ++active_i)
        {

            const ck_tile::index_t qb =
                rlut_ptr[static_cast<ck_tile::long_index_t>(active_i) * args.stride_rlutk];
            const ck_tile::index_t q_start = qb * BlockM;
            if(q_start >= args.N_Q)
            {
                continue;
            }

            const bool is_boundary = block_is_boundary || (q_start + BlockM > seqlen);
            if constexpr(Policy::LeanFusedStorage)
            {
                if(is_boundary)
                {
                    continue;
                }
            }
            {
                    auto qk_acc_n0 = decltype(qk_chunk_asmem_breg_gemm.MakeCBlockTile()){};
                    auto qk_acc_n1 = decltype(qk_chunk_asmem_breg_gemm.MakeCBlockTile()){};
                    ck_tile::clear_tile(qk_acc_n0);
                    ck_tile::clear_tile(qk_acc_n1);

                    auto q_dram_window = ck_tile::make_tile_window(
                        q_dram_view,
                        ck_tile::make_tuple(ck_tile::number<BlockM>{},
                                            ck_tile::number<QKChunk>{}),
                        ck_tile::multi_index<2>{q_start, 0},
                        MakeBwdQAsyncDramDistribution<BlockM, QKChunk>());
                    auto load_q_chunk_to_lds = [&](auto buf_num) {
                        constexpr int buf = decltype(buf_num)::value;

                        auto q_lds_store_view =
                            ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                                q_async_smem,
                                MakeBwdQAsyncLdsStoreDescriptor<BlockM, QKChunk, buf>());
                        auto q_lds_store_window = ck_tile::make_tile_window(
                            q_lds_store_view,
                            MakeBwdQAsyncLdsStoreDescriptor<BlockM, QKChunk, buf>().get_lengths(),
                            ck_tile::multi_index<3>{0, 0, 0});
                        ck_tile::async_load_tile_by_inline_asm(
                            q_lds_store_window,
                            q_dram_window,
                            ck_tile::bool_constant<true>{},
                            ck_tile::bool_constant<false>{});
                        if constexpr(buf + 1 < QKDChunks)
                        {
                            ck_tile::move_tile_window(
                                q_dram_window, ck_tile::multi_index<2>{0, QKChunk});
                        }
                    };

                    auto make_q_chunk_window = [&](auto buf_num) {
                        constexpr int buf = decltype(buf_num)::value;
                        return ck_tile::make_tile_window(
                            ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                                q_async_smem, MakeBwdQAsyncLdsLoadDescriptor<BlockM, QKChunk>()),
                            ck_tile::make_tuple(ck_tile::number<BlockM>{},
                                                ck_tile::number<QKChunk>{}),
                            ck_tile::multi_index<2>{buf * BlockM, 0});
                    };

                    // All four 32-wide Q head chunks are staged into four LDS
                    // buffers with one fence/barrier, then the QK gemms run
                    // back to back without intermediate syncs.
                    load_q_chunk_to_lds(ck_tile::number<0>{});
                    load_q_chunk_to_lds(ck_tile::number<1>{});
                    load_q_chunk_to_lds(ck_tile::number<2>{});
                    load_q_chunk_to_lds(ck_tile::number<3>{});
                    ck_tile::async_load_fence(0);
                    SlaBwdBlockSyncLdsLight();

                    // Build the dK Q fragments a quarter at a time. Each LDS
                    // issue group is followed by a QK chunk so its latency can
                    // overlap MMAC work instead of accumulating after QK.
                    constexpr auto q_lds_desc =
                        MakeBwdQAsyncLdsLoadDescriptor<BlockM, QKChunk>();
                    constexpr ck_tile::index_t q_half1_lds_offset =
                        q_lds_desc.calculate_offset(
                            ck_tile::multi_index<2>{QKChunk, 0});
                    auto q_t_breg0 = dk_k32_breg_gemm.MakeInterleavedBBlockTile();
                    auto q_t_breg1 = dk_k32_breg_gemm.MakeInterleavedBBlockTile();
                    constexpr ck_tile::index_t q_dwave_lds_stride =
                        SlaBwdAsyncQGeometry<BlockM, QKChunk>::kSmemElements;

                    dk_k32_breg_gemm.template LoadBBlockTileDwordFromLdsKIter<0>(
                        q_t_breg0, q_async_smem, q_lds_desc, q_dwave_lds_stride);

                    auto q_chunk0_lds_window = make_q_chunk_window(ck_tile::number<0>{});
                    qk_chunk_asmem_breg_gemm(qk_acc_n0, q_chunk0_lds_window, k_breg_n0_d0);
                    if constexpr(BlockN == 2 * QKSubN)
                    {
                        qk_chunk_asmem_breg_gemm(
                            qk_acc_n1, q_chunk0_lds_window, k_breg_n1_d0);
                    }
                    dk_k32_breg_gemm.template LoadBBlockTileDwordFromLdsKIter<0>(
                        q_t_breg1,
                        q_async_smem + q_half1_lds_offset,
                        q_lds_desc,
                        q_dwave_lds_stride);
                    dk_k32_breg_gemm.template PrepareBBlockTileDwordKIter<0>(q_t_breg0);

                    auto q_chunk1_lds_window = make_q_chunk_window(ck_tile::number<1>{});
                    qk_chunk_asmem_breg_gemm(qk_acc_n0, q_chunk1_lds_window, k_breg_n0_d1);
                    if constexpr(BlockN == 2 * QKSubN)
                    {
                        qk_chunk_asmem_breg_gemm(
                            qk_acc_n1, q_chunk1_lds_window, k_breg_n1_d1);
                    }
                    dk_k32_breg_gemm.template LoadBBlockTileDwordFromLdsKIter<1>(
                        q_t_breg0, q_async_smem, q_lds_desc, q_dwave_lds_stride);
                    dk_k32_breg_gemm.template PrepareBBlockTileDwordKIter<0>(q_t_breg1);

                    auto q_chunk2_lds_window = make_q_chunk_window(ck_tile::number<2>{});
                    qk_chunk_asmem_breg_gemm(qk_acc_n0, q_chunk2_lds_window, k_breg_n0_d2);
                    if constexpr(BlockN == 2 * QKSubN)
                    {
                        qk_chunk_asmem_breg_gemm(
                            qk_acc_n1, q_chunk2_lds_window, k_breg_n1_d2);
                    }
                    dk_k32_breg_gemm.template LoadBBlockTileDwordFromLdsKIter<1>(
                        q_t_breg1,
                        q_async_smem + q_half1_lds_offset,
                        q_lds_desc,
                        q_dwave_lds_stride);
                    dk_k32_breg_gemm.template PrepareBBlockTileDwordKIter<1>(q_t_breg0);

                    auto q_chunk3_lds_window = make_q_chunk_window(ck_tile::number<3>{});
                    qk_chunk_asmem_breg_gemm(qk_acc_n0, q_chunk3_lds_window, k_breg_n0_d3);
                    if constexpr(BlockN == 2 * QKSubN)
                    {
                        qk_chunk_asmem_breg_gemm(
                            qk_acc_n1, q_chunk3_lds_window, k_breg_n1_d3);
                    }
                    dk_k32_breg_gemm.template PrepareBBlockTileDwordKIter<1>(q_t_breg1);
                    // Drain the interleaved fragment reads and all QK LDS
                    // accesses before the shared-memory union becomes P/dO.
                    ck_tile::block_sync_lds();

                    // Issue the first dO half's async copy before the softmax
                    // reconstruction: its LDS region ([p_t, p_t+dO chunk
                    // bytes)) is disjoint from the p_t writes, so the DRAM
                    // fetch overlaps with exp2/LSE work and the barrier below.
                    OGradDataType* do_qhalf_smem =
                        reinterpret_cast<OGradDataType*>(smem.dv_input.p_t);
                    auto load_do_qhalf_to_lds = [&](ck_tile::index_t q_base) {
                        auto do_dram_window = ck_tile::make_tile_window(
                            do_dram_view,
                            ck_tile::make_tuple(ck_tile::number<32>{},
                                                ck_tile::number<QKChunk>{}),
                            ck_tile::multi_index<2>{q_start + q_base, 0},
                            MakeBwdDoAsyncDramDistribution<32, QKChunk>());
                        auto load_do_chunk_to_lds = [&](auto chunk_num) {
                            constexpr int chunk = decltype(chunk_num)::value;

                            auto do_lds_store_view =
                                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                                    do_qhalf_smem,
                                    MakeBwdDoAsyncLdsStoreDescriptor<
                                        QKChunk,
                                        Policy::DoQHalfLdsBaseOffset,
                                        chunk>());
                            auto do_lds_store_window = ck_tile::make_tile_window(
                                do_lds_store_view,
                                MakeBwdDoAsyncLdsStoreDescriptor<
                                    QKChunk,
                                    Policy::DoQHalfLdsBaseOffset,
                                    chunk>().get_lengths(),
                                ck_tile::multi_index<3>{0, 0, 0});
                            ck_tile::async_load_tile_by_inline_asm(
                                do_lds_store_window,
                                do_dram_window,
                                ck_tile::bool_constant<true>{},
                                ck_tile::bool_constant<false>{});
                            if constexpr(chunk + 1 < QKDChunks)
                            {
                                ck_tile::move_tile_window(
                                    do_dram_window, ck_tile::multi_index<2>{0, QKChunk});
                            }
                        };

                        load_do_chunk_to_lds(ck_tile::number<0>{});
                        load_do_chunk_to_lds(ck_tile::number<1>{});
                        load_do_chunk_to_lds(ck_tile::number<2>{});
                        load_do_chunk_to_lds(ck_tile::number<3>{});
                    };
                    load_do_qhalf_to_lds(0);

                    QDataType* p_t_smem = smem.dv_input.p_t;
                    const auto qk_out_n0 = qk_chunk_asmem_breg_gemm.MakeOuputLayout(qk_acc_n0);
                    const auto qk_out_n1 = qk_chunk_asmem_breg_gemm.MakeOuputLayout(qk_acc_n1);
                    constexpr auto qk_spans = decltype(qk_out_n0)::get_distributed_spans();
                    constexpr auto M_spans = qk_spans[ck_tile::number<0>{}];
                    constexpr auto N_spans = qk_spans[ck_tile::number<1>{}];
                    float lse_regs[decltype(M_spans)::Impl::size()];
                    ck_tile::index_t m_rel_regs[decltype(M_spans)::Impl::size()];

                    ck_tile::sweep_tile_span(M_spans, [&](auto idx0) {
                        constexpr auto dummy_tile_idx = ck_tile::make_tuple(
                            idx0, ck_tile::tile_distributed_index<1, 0, 0, 0>{});
                        const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                            qk_out_n0.get_tile_distribution(), dummy_tile_idx);
                        const ck_tile::index_t m_rel = x_idx.at(ck_tile::number<0>{});
                        const ck_tile::index_t m     = q_start + m_rel;
                        const bool row_valid         = m < args.N_Q && m < seqlen;
                        constexpr int slot = decltype(idx0)::Impl::at(0);
                        m_rel_regs[slot] = m_rel;
                        lse_regs[slot] = row_valid
                                             ? lse_ptr[static_cast<ck_tile::long_index_t>(m) *
                                                       args.stride_lm]
                                             : 0.0f;
                    });
                    ck_tile::sweep_tile_span(M_spans, [&](auto idx0) {
                        constexpr int m_slot = decltype(idx0)::Impl::at(0);
                        const ck_tile::index_t m_rel = m_rel_regs[m_slot];
                        const float row_lse_log2 = lse_regs[m_slot];
                        ck_tile::sweep_tile_span(N_spans, [&](auto idx1) {
                            constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                            const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                                qk_out_n0.get_tile_distribution(), tile_idx);
                            const ck_tile::index_t n_rel = x_idx.at(ck_tile::number<1>{});
                            const float p0 = SlaBwdFastExp2(qk_out_n0[tile_idx] * scale_log2e -
                                                              row_lse_log2);
                            p_t_smem[static_cast<ck_tile::long_index_t>(n_rel) *
                                         Policy::PTransposeLdsStride +
                                     m_rel] = ck_tile::type_convert<QDataType>(p0);
                            if constexpr(BlockN == 2 * QKSubN)
                            {
                                const float p1 =
                                    SlaBwdFastExp2(qk_out_n1[tile_idx] * scale_log2e -
                                                   row_lse_log2);
                                p_t_smem[static_cast<ck_tile::long_index_t>(QKSubN + n_rel) *
                                             Policy::PTransposeLdsStride +
                                         m_rel] = ck_tile::type_convert<QDataType>(p1);
                            }
                        });
                    });

                    ck_tile::async_load_fence(0);
                    SlaBwdBlockSyncLdsLight();

                    auto p_t_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                        p_t_smem,
                        MakePaddedRowMajorLdsDescriptor<BlockN,
                                                        BlockM,
                                                        Policy::PTransposeLdsStride>());
                    auto accumulate_dv_qhalf_from_lds = [&](ck_tile::index_t q_base) {
                        constexpr auto do_lds_desc = MakeBwdDoAsyncLdsLoadDescriptor<
                            QKChunk,
                            Policy::DoQHalfLdsBaseOffset,
                            0>();
                        auto do_t_breg = dv_k32_breg_gemm.LoadBBlockTileDwordFromLds(
                            do_qhalf_smem,
                            do_lds_desc,
                            SlaBwdAsyncDoQHalfGeometry<QKChunk>::kSmemElements);
                        dv_k32_breg_gemm.template PrepareBBlockTileDwordKIter<0>(do_t_breg);
                        dv_k32_breg_gemm.template PrepareBBlockTileDwordKIter<1>(do_t_breg);
                        auto p_t_lds_window = ck_tile::make_tile_window(
                            p_t_lds_view,
                            ck_tile::make_tuple(ck_tile::number<BlockN>{}, ck_tile::number<32>{}),
                            ck_tile::multi_index<2>{0, q_base},
                            dv_k32_breg_gemm.MakeABlockTile().get_tile_distribution());
                        auto p_t_areg = ck_tile::load_tile(p_t_lds_window);
                        dv_k32_breg_gemm.RunPreparedKIter(
                            dv_acc, p_t_areg, do_t_breg, ck_tile::number<0>{});
                        dv_k32_breg_gemm.RunPreparedKIter(
                            dv_acc, p_t_areg, do_t_breg, ck_tile::number<1>{});
                    };

                    if constexpr(ComputeDV && !ComputeDK)
                    {
                        accumulate_dv_qhalf_from_lds(0);

                        ck_tile::block_sync_lds();
                        load_do_qhalf_to_lds(32);
                        ck_tile::async_load_fence(0);
                        SlaBwdBlockSyncLdsLight();
                        accumulate_dv_qhalf_from_lds(32);
                    }

                    if constexpr(ComputeDK)
                    {
                        auto dp_t_acc0 = decltype(dp_k32_areg_breg_gemm.MakeCBlockTile()){};
                        auto dp_t_acc1 = decltype(dp_k32_areg_breg_gemm.MakeCBlockTile()){};
                        ck_tile::clear_tile(dp_t_acc0);
                        ck_tile::clear_tile(dp_t_acc1);

                        auto load_do_dp_k32_breg_from_lds = [&](auto chunk_num) {
                            constexpr int chunk = decltype(chunk_num)::value;
                            auto do_breg = dp_k32_gemm.MakeBBlockTile();
                            auto do_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                                do_qhalf_smem,
                                MakeBwdDoAsyncLdsLoadDescriptor<
                                    QKChunk,
                                    Policy::DoQHalfLdsBaseOffset,
                                    chunk>());
                            auto do_lds_window = ck_tile::make_tile_window(
                                do_lds_view,
                                ck_tile::make_tuple(ck_tile::number<32>{},
                                                    ck_tile::number<QKChunk>{}),
                                ck_tile::multi_index<2>{0, 0},
                                do_breg.get_tile_distribution());
                            return ck_tile::load_tile(do_lds_window);
                        };

                        auto accumulate_dp_t_qhalf_from_lds = [&](auto& dp_t_acc) {
                            auto do_dp_breg0 = load_do_dp_k32_breg_from_lds(ck_tile::number<0>{});
                            auto do_dp_breg1 = load_do_dp_k32_breg_from_lds(ck_tile::number<1>{});
                            auto do_dp_breg2 = load_do_dp_k32_breg_from_lds(ck_tile::number<2>{});
                            auto do_dp_breg3 = load_do_dp_k32_breg_from_lds(ck_tile::number<3>{});
                            dp_k32_areg_breg_gemm(dp_t_acc, v_areg_d0, do_dp_breg0);
                            dp_k32_areg_breg_gemm(dp_t_acc, v_areg_d1, do_dp_breg1);
                            dp_k32_areg_breg_gemm(dp_t_acc, v_areg_d2, do_dp_breg2);
                            dp_k32_areg_breg_gemm(dp_t_acc, v_areg_d3, do_dp_breg3);
                        };

                        if constexpr(ComputeDV)
                        {
                            accumulate_dv_qhalf_from_lds(0);
                        }
                        // Load the first half's dP B tiles before the barrier,
                        // then issue the second half's async copy so its DRAM
                        // fetch overlaps the first half's dP GEMMs instead of
                        // stalling right after the barrier.
                        auto do_dp_breg0 = load_do_dp_k32_breg_from_lds(ck_tile::number<0>{});
                        auto do_dp_breg1 = load_do_dp_k32_breg_from_lds(ck_tile::number<1>{});
                        auto do_dp_breg2 = load_do_dp_k32_breg_from_lds(ck_tile::number<2>{});
                        auto do_dp_breg3 = load_do_dp_k32_breg_from_lds(ck_tile::number<3>{});
                        ck_tile::block_sync_lds();
                        load_do_qhalf_to_lds(32);
                        dp_k32_areg_breg_gemm(dp_t_acc0, v_areg_d0, do_dp_breg0);
                        dp_k32_areg_breg_gemm(dp_t_acc0, v_areg_d1, do_dp_breg1);
                        dp_k32_areg_breg_gemm(dp_t_acc0, v_areg_d2, do_dp_breg2);
                        dp_k32_areg_breg_gemm(dp_t_acc0, v_areg_d3, do_dp_breg3);
                        ck_tile::async_load_fence(0);
                        SlaBwdBlockSyncLdsLight();
                        if constexpr(ComputeDV)
                        {
                            accumulate_dv_qhalf_from_lds(32);
                        }
                        accumulate_dp_t_qhalf_from_lds(dp_t_acc1);
                        ck_tile::block_sync_lds();

                        auto delta_smem =
                            reinterpret_cast<AccDataType*>(p_t_smem + Policy::DoQHalfLdsBaseOffset);
                        if(threadIdx.x < BlockM)
                        {
                            delta_smem[threadIdx.x] =
                                delta_ptr[static_cast<ck_tile::long_index_t>(q_start + threadIdx.x) *
                                          args.stride_delta_m];
                        }
                        ck_tile::block_sync_lds();

                        const auto dp_t_out0 = dp_k32_areg_breg_gemm.MakeOuputLayout(dp_t_acc0);
                        const auto dp_t_out1 = dp_k32_areg_breg_gemm.MakeOuputLayout(dp_t_acc1);
                        constexpr auto dp_t_spans = decltype(dp_t_out0)::get_distributed_spans();
                        ck_tile::sweep_tile_span(dp_t_spans[ck_tile::number<0>{}], [&](auto idx0) {
                            ck_tile::sweep_tile_span(dp_t_spans[ck_tile::number<1>{}], [&](auto idx1) {
                                constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                                const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                                    dp_t_out0.get_tile_distribution(), tile_idx);
                                const ck_tile::index_t n_rel = x_idx.at(ck_tile::number<0>{});
                                const ck_tile::index_t m_rel = x_idx.at(ck_tile::number<1>{});
                                const float delta0 = delta_smem[m_rel];
                                const float delta1 = delta_smem[32 + m_rel];
                                const ck_tile::long_index_t p0_offset =
                                    static_cast<ck_tile::long_index_t>(n_rel) *
                                        Policy::PTransposeLdsStride +
                                    m_rel;
                                const ck_tile::long_index_t p1_offset =
                                    static_cast<ck_tile::long_index_t>(n_rel) *
                                        Policy::PTransposeLdsStride +
                                    32 + m_rel;
                                const float p0 = ck_tile::type_convert<float>(p_t_smem[p0_offset]);
                                const float p1 = ck_tile::type_convert<float>(p_t_smem[p1_offset]);
                                p_t_smem[p0_offset] = ck_tile::type_convert<QDataType>(
                                    p0 * (dp_t_out0[tile_idx] - delta0));
                                p_t_smem[p1_offset] = ck_tile::type_convert<QDataType>(
                                    p1 * (dp_t_out1[tile_idx] - delta1));
                            });
                        });

                        ck_tile::block_sync_lds();

                        auto ds_t_lds_window0 = ck_tile::make_tile_window(
                            p_t_lds_view,
                            ck_tile::make_tuple(ck_tile::number<BlockN>{}, ck_tile::number<32>{}),
                            ck_tile::multi_index<2>{0, 0},
                            dk_k32_breg_gemm.MakeABlockTile().get_tile_distribution());
                        auto ds_t_areg0 = ck_tile::load_tile(ds_t_lds_window0);

                        auto ds_t_lds_window1 = ck_tile::make_tile_window(
                            p_t_lds_view,
                            ck_tile::make_tuple(ck_tile::number<BlockN>{}, ck_tile::number<32>{}),
                            ck_tile::multi_index<2>{0, 32},
                            dk_k32_breg_gemm.MakeABlockTile().get_tile_distribution());
                        auto ds_t_areg1 = ck_tile::load_tile(ds_t_lds_window1);

                        dk_k32_breg_gemm.RunPreparedKIter(
                            dk_acc, ds_t_areg0, q_t_breg0, ck_tile::number<0>{});
                        dk_k32_breg_gemm.RunPreparedKIter(
                            dk_acc, ds_t_areg0, q_t_breg0, ck_tile::number<1>{});
                        dk_k32_breg_gemm.RunPreparedKIter(
                            dk_acc, ds_t_areg1, q_t_breg1, ck_tile::number<0>{});
                        dk_k32_breg_gemm.RunPreparedKIter(
                            dk_acc, ds_t_areg1, q_t_breg1, ck_tile::number<1>{});
                    }

                    ck_tile::block_sync_lds();
                    continue;
                }
            }
        static_assert(std::is_same_v<QDataType, VGradDataType> &&
                          std::is_same_v<QDataType, KGradDataType>,
                      "coalesced epilogue requires matching 16-bit output types");
        constexpr ck_tile::index_t OutputLdsStride = HeadDim + 4;
        constexpr ck_tile::index_t OutputVector = 4;
        constexpr ck_tile::index_t OutputVectorsPerThread =
            (BlockN * HeadDim / OutputVector) / ThreadsPerBlock;
        static_assert(BlockN * OutputLdsStride <=
                          4 * SlaBwdAsyncQGeometry<BlockM, QKChunk>::kSmemElements,
                      "coalesced output tile exceeds reusable LDS storage");
        QDataType* output_smem = q_async_smem;

        auto store_output_from_lds = [&](auto& dst_view,
                                         ck_tile::long_index_t stride_n,
                                         ck_tile::long_index_t stride_d) {
            const auto dst_resource = dst_view.get_buffer_view().cached_buf_res_;
            #pragma unroll
            for(ck_tile::index_t i = 0; i < OutputVectorsPerThread; ++i)
            {
                const ck_tile::index_t vector_idx =
                    i * ThreadsPerBlock + ck_tile::get_thread_id();
                const ck_tile::index_t linear = vector_idx * OutputVector;
                const ck_tile::index_t n_rel  = linear / HeadDim;
                const ck_tile::index_t d      = linear % HeadDim;
                const ck_tile::index_t n     = start_n + n_rel;
                if(n < seqlen && d + OutputVector - 1 < args.D)
                {
                    uint32x2_t packed;
                    __builtin_memcpy(&packed, &output_smem[static_cast<ck_tile::long_index_t>(n_rel) *
                                         OutputLdsStride +
                                     d], sizeof(packed));
                    using StoreBuffer = ck_tile::thread_buffer<QDataType, OutputVector>;
                    const StoreBuffer store_buffer = ck_tile::bit_cast<StoreBuffer>(packed);
                    const ck_tile::index_t byte_offset = static_cast<ck_tile::index_t>(
                        (static_cast<ck_tile::long_index_t>(n) * stride_n +
                         static_cast<ck_tile::long_index_t>(d) * stride_d) *
                        sizeof(QDataType));
                    ck_tile::amd_buffer_store_impl<QDataType, OutputVector>(
                        store_buffer, dst_resource, byte_offset, 0);
                }
            }
        };

        if constexpr(ComputeDV)
        {
            const auto dv_out = dv_breg_gemm.MakeOuputLayout(dv_acc);
            constexpr auto dv_spans = decltype(dv_out)::get_distributed_spans();
            ck_tile::sweep_tile_span(dv_spans[ck_tile::number<0>{}], [&](auto idx0) {
                VGradDataType even[4];
                ck_tile::index_t lane_pos = 0;
                ck_tile::sweep_tile_span(dv_spans[ck_tile::number<1>{}], [&](auto idx1) {
                    constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                        dv_out.get_tile_distribution(), tile_idx);
                    const ck_tile::index_t n_rel = x_idx.at(ck_tile::number<0>{});
                    const ck_tile::index_t d_interleaved = x_idx.at(ck_tile::number<1>{});
                    const ck_tile::index_t n_iter = d_interleaved / 64;
                    const ck_tile::index_t n_wave = (d_interleaved % 64) / 16;
                    const ck_tile::index_t n_lane = d_interleaved % 16;
                    const ck_tile::index_t d = n_wave * 32 + n_lane * 2 + n_iter;
                    const VGradDataType v =
                        ck_tile::type_convert<VGradDataType>(dv_out[tile_idx]);
                    if(n_iter == 0)
                    {
                        even[lane_pos & 3] = v;
                    }
                    else
                    {
                        const uint32_t packed =
                            static_cast<uint32_t>(ck_tile::bit_cast<uint16_t>(
                                even[lane_pos & 3])) |
                            (static_cast<uint32_t>(ck_tile::bit_cast<uint16_t>(v)) << 16);
                        __builtin_memcpy(&output_smem[static_cast<ck_tile::long_index_t>(n_rel) *
                                             OutputLdsStride +
                                         d - 1], &packed, sizeof(packed));
                    }
                    ++lane_pos;
                });
            });
            ck_tile::block_sync_lds();
            store_output_from_lds(dv_dram_view, args.stride_dvn, args.stride_dvd);
        }

        if constexpr(ComputeDK)
        {
            if constexpr(ComputeDV)
            {
                ck_tile::block_sync_lds();
            }
            const auto dk_out = dk_k32_breg_gemm.MakeOuputLayout(dk_acc);
            constexpr auto dk_spans = decltype(dk_out)::get_distributed_spans();
            ck_tile::sweep_tile_span(dk_spans[ck_tile::number<0>{}], [&](auto idx0) {
                KGradDataType even[4];
                ck_tile::index_t lane_pos = 0;
                ck_tile::sweep_tile_span(dk_spans[ck_tile::number<1>{}], [&](auto idx1) {
                    constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                        dk_out.get_tile_distribution(), tile_idx);
                    const ck_tile::index_t n_rel = x_idx.at(ck_tile::number<0>{});
                    const ck_tile::index_t d_interleaved = x_idx.at(ck_tile::number<1>{});
                    const ck_tile::index_t n_iter = d_interleaved / 64;
                    const ck_tile::index_t n_wave = (d_interleaved % 64) / 16;
                    const ck_tile::index_t n_lane = d_interleaved % 16;
                    const ck_tile::index_t d = n_wave * 32 + n_lane * 2 + n_iter;
                    const KGradDataType v = ck_tile::type_convert<KGradDataType>(
                        dk_out[tile_idx] * args.sm_scale);
                    if(n_iter == 0)
                    {
                        even[lane_pos & 3] = v;
                    }
                    else
                    {
                        const uint32_t packed =
                            static_cast<uint32_t>(ck_tile::bit_cast<uint16_t>(
                                even[lane_pos & 3])) |
                            (static_cast<uint32_t>(ck_tile::bit_cast<uint16_t>(v)) << 16);
                        __builtin_memcpy(&output_smem[static_cast<ck_tile::long_index_t>(n_rel) *
                                             OutputLdsStride +
                                         d - 1], &packed, sizeof(packed));
                    }
                    ++lane_pos;
                });
            });
            ck_tile::block_sync_lds();
            store_output_from_lds(dk_dram_view, args.stride_dkn, args.stride_dkd);
        }
    }
};

} // namespace sla
} // namespace example
} // namespace ck_tile
