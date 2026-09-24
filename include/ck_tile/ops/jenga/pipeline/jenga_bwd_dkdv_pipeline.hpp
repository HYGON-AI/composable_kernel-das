// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dkdv_config.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dkdv_policy.hpp"

namespace ck_tile {
namespace example {
namespace jenga {

template <typename Problem, typename Policy = JengaBwdDkdvDefaultPolicy<Problem>>
struct JengaBwdDkdvPipeline
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
    static constexpr index_t ThreadsPerBlock = Problem::ThreadsPerBlock;

    CK_TILE_DEVICE static void run(const jenga_bwd_dkdv_args& args,
                                   ck_tile::index_t kv_block,
                                   ck_tile::index_t off_hz,
                                   ck_tile::index_t batch,
                                   ck_tile::index_t head,
                                   ck_tile::index_t seqlen,
                                   ck_tile::index_t split_id = 0)
    {
        run_impl<true, true>(args, kv_block, off_hz, batch, head, seqlen, split_id);
    }

    template <bool ComputeDV, bool ComputeDK>
    CK_TILE_DEVICE static void run_impl(const jenga_bwd_dkdv_args& args,
                                        ck_tile::index_t kv_block,
                                        ck_tile::index_t off_hz,
                                        ck_tile::index_t batch,
                                        ck_tile::index_t head,
                                        ck_tile::index_t seqlen,
                                        ck_tile::index_t split_id = 0)
    {
        constexpr auto qk_chunk_asmem_breg_gemm = typename Policy::QKChunkASmemBRegBlockGemm{};
        constexpr auto dv_breg_gemm = typename Policy::DVBRegBlockGemm{};
        constexpr auto dv_k32_breg_gemm = typename Policy::DVK32BRegBlockGemm{};
        constexpr auto dp_k32_gemm = typename Policy::DPK32BlockGemm{};
        constexpr auto dp_k32_areg_breg_gemm = typename Policy::DPK32ARegBRegBlockGemm{};
        constexpr auto dk_k32_breg_gemm = typename Policy::DKK32BRegBlockGemm{};
        const ck_tile::index_t start_n = kv_block * BlockN;

        const QDataType* q_ptr =
            static_cast<const QDataType*>(args.q_ptr) +
            static_cast<ck_tile::long_index_t>(batch) * args.stride_qz +
            static_cast<ck_tile::long_index_t>(head) * args.stride_qh;
        const KDataType* k_ptr =
            static_cast<const KDataType*>(args.k_ptr) +
            static_cast<ck_tile::long_index_t>(batch) * args.stride_kz +
            static_cast<ck_tile::long_index_t>(head) * args.stride_kh;
        const OGradDataType* do_ptr =
            static_cast<const OGradDataType*>(args.do_ptr) +
            static_cast<ck_tile::long_index_t>(batch) * args.stride_doz +
            static_cast<ck_tile::long_index_t>(head) * args.stride_doh;
        const VDataType* v_ptr =
            static_cast<const VDataType*>(args.v_ptr) +
            static_cast<ck_tile::long_index_t>(batch) * args.stride_vz +
            static_cast<ck_tile::long_index_t>(head) * args.stride_vh;
        const AccDataType* delta_ptr =
            static_cast<const AccDataType*>(args.delta_ptr) +
            static_cast<ck_tile::long_index_t>(off_hz) * args.stride_delta_z;

        KGradDataType* dk_ptr = static_cast<KGradDataType*>(args.split_kv > 1 ? args.workspace_dk_ptr : args.dk_ptr);
        if(args.split_kv > 1)
        {
            dk_ptr += static_cast<ck_tile::long_index_t>(split_id) * args.stride_work_dk_s;
        }
        dk_ptr += static_cast<ck_tile::long_index_t>(batch) * args.stride_dkz +
                  static_cast<ck_tile::long_index_t>(head) * args.stride_dkh;

        VGradDataType* dv_ptr = static_cast<VGradDataType*>(args.split_kv > 1 ? args.workspace_dv_ptr : args.dv_ptr);
        if(args.split_kv > 1)
        {
            dv_ptr += static_cast<ck_tile::long_index_t>(split_id) * args.stride_work_dv_s;
        }
        dv_ptr += static_cast<ck_tile::long_index_t>(batch) * args.stride_dvz +
                  static_cast<ck_tile::long_index_t>(head) * args.stride_dvh;

        const int32_t* rlut_ptr =
            args.rlut_ptr + static_cast<ck_tile::long_index_t>(off_hz) * args.stride_rlutz +
            static_cast<ck_tile::long_index_t>(kv_block) * args.stride_rlutn;
        const int32_t num_active =
            args.rlut_size_ptr[static_cast<ck_tile::long_index_t>(off_hz) *
                                   args.stride_rlut_size_z +
                               static_cast<ck_tile::long_index_t>(kv_block) *
                                   args.stride_rlut_size_n];
        if(num_active <= 0)
        {
            return;
        }

        const bool is_text_block = kv_block >= args.text_block_start;
        const LSEDataType* lse_ptr =
            static_cast<const LSEDataType*>(is_text_block ? args.lse_text_ptr : args.lse_ptr) +
            static_cast<ck_tile::long_index_t>(off_hz) * args.stride_lz;
        constexpr float qk_scale = ck_tile::log2e_v<float>;
        const float scale_log2e  = args.sm_scale * qk_scale;
        constexpr int QVecLoadIters = (BlockM * HeadDim) / (8 * ThreadsPerBlock);
        constexpr int KVVecLoadIters = (BlockN * HeadDim) / (8 * ThreadsPerBlock);
        constexpr int QKChunk = JengaMmacQKChunkConfig::kK;
        constexpr int QKSubN = 32;
        constexpr int QKDChunks = HeadDim / QKChunk;
        static_assert(HeadDim == QKDChunks * QKChunk, "dV split-QK path expects even chunks");
        static_assert(QKDChunks == 4, "dV fwd-style QK path currently expects four 32-wide chunks");
        static_assert(BlockN == 2 * QKSubN, "dV split-N QK path expects BlockN=64");
        constexpr int TransposedMVecIters = BlockM / 16;

        __shared__ typename Policy::LdsStorage smem;

        const bool block_is_boundary = (start_n + BlockN > seqlen) || (args.D < HeadDim);
        const bool block_can_vectorize_transposed_do = (args.stride_dom % 8 == 0) && (args.stride_dod == 1) && (reinterpret_cast<std::uintptr_t>(do_ptr) % 16 == 0);
        const bool block_can_vectorize_transposed_q  = (args.stride_qm % 8 == 0) && (args.stride_qd == 1) && (reinterpret_cast<std::uintptr_t>(q_ptr) % 16 == 0);

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

        auto load_do_t_breg = [&](ck_tile::index_t q_start) {
            auto do_t_dram_view =
                ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    do_ptr + static_cast<ck_tile::long_index_t>(q_start) * args.stride_dom,
                    ck_tile::make_tuple(ck_tile::number<HeadDim>{}, ck_tile::number<BlockM>{}),
                    ck_tile::make_tuple(args.stride_dod, args.stride_dom),
                    ck_tile::number<1>{},
                    ck_tile::number<1>{});
            auto do_t_breg = dv_breg_gemm.MakeBBlockTile();
            auto do_t_dram_window = ck_tile::make_tile_window(
                do_t_dram_view,
                ck_tile::make_tuple(ck_tile::number<HeadDim>{}, ck_tile::number<BlockM>{}),
                ck_tile::multi_index<2>{0, 0},
                do_t_breg.get_tile_distribution());
            return ck_tile::load_tile(do_t_dram_window);
        };

        auto load_q_t_k32_breg = [&](ck_tile::index_t q_start, ck_tile::index_t k_base) {
            auto q_t_dram_view =
                ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    q_ptr + static_cast<ck_tile::long_index_t>(q_start + k_base) * args.stride_qm,
                    ck_tile::make_tuple(ck_tile::number<HeadDim>{}, ck_tile::number<BlockM>{}),
                    ck_tile::make_tuple(args.stride_qd, args.stride_qm),
                    ck_tile::number<1>{},
                    ck_tile::number<1>{});
            auto q_t_breg = dk_k32_breg_gemm.MakeBBlockTile();
            auto q_t_dram_window = ck_tile::make_tile_window(
                q_t_dram_view,
                ck_tile::make_tuple(ck_tile::number<32>{}, ck_tile::number<BlockM>{}),
                ck_tile::multi_index<2>{0, 0},
                q_t_breg.get_tile_distribution());
            return ck_tile::load_tile(q_t_dram_window);
        };

        auto load_do_t_k32_breg = [&](ck_tile::index_t q_start, ck_tile::index_t k_base) {
            auto do_t_dram_view =
                ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    do_ptr + static_cast<ck_tile::long_index_t>(q_start + k_base) * args.stride_dom,
                    ck_tile::make_tuple(ck_tile::number<HeadDim>{}, ck_tile::number<BlockM>{}),
                    ck_tile::make_tuple(args.stride_dod, args.stride_dom),
                    ck_tile::number<1>{},
                    ck_tile::number<1>{});
            auto do_t_breg = dv_k32_breg_gemm.MakeBBlockTile();
            auto do_t_dram_window = ck_tile::make_tile_window(
                do_t_dram_view,
                ck_tile::make_tuple(ck_tile::number<32>{}, ck_tile::number<BlockM>{}),
                ck_tile::multi_index<2>{0, 0},
                do_t_breg.get_tile_distribution());
            return ck_tile::load_tile(do_t_dram_window);
        };

        QDataType* q_async_smem = smem.q_async;
        auto k_breg_n0_d0 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n0_d1 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n0_d2 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n0_d3 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n1_d0 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n1_d1 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n1_d2 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto k_breg_n1_d3 = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
        auto load_v_dp_k32_areg = [&](ck_tile::index_t d_base) {
            auto v_dram_view =
                ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    v_ptr + static_cast<ck_tile::long_index_t>(start_n) * args.stride_vn +
                        static_cast<ck_tile::long_index_t>(d_base) * args.stride_vd,
                    ck_tile::make_tuple(ck_tile::number<BlockN>{}, ck_tile::number<QKChunk>{}),
                    ck_tile::make_tuple(args.stride_vn, args.stride_vd),
                    ck_tile::number<8>{},
                    ck_tile::number<1>{});
            auto v_areg = dp_k32_areg_breg_gemm.MakeABlockTile();
            auto v_dram_window = ck_tile::make_tile_window(
                v_dram_view,
                ck_tile::make_tuple(ck_tile::number<BlockN>{}, ck_tile::number<QKChunk>{}),
                ck_tile::multi_index<2>{0, 0},
                v_areg.get_tile_distribution());
            return ck_tile::load_tile(v_dram_window);
        };
        auto v_areg_d0 = dp_k32_areg_breg_gemm.MakeABlockTile();
        auto v_areg_d1 = dp_k32_areg_breg_gemm.MakeABlockTile();
        auto v_areg_d2 = dp_k32_areg_breg_gemm.MakeABlockTile();
        auto v_areg_d3 = dp_k32_areg_breg_gemm.MakeABlockTile();
        if constexpr(Policy::LeanFusedStorage)
        {
            auto load_k_chunk_breg = [&](ck_tile::index_t n_base, ck_tile::index_t d_base) {
                auto k_dram_view =
                    ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                        k_ptr + static_cast<ck_tile::long_index_t>(start_n + n_base) * args.stride_kn +
                            static_cast<ck_tile::long_index_t>(d_base) * args.stride_kd,
                        ck_tile::make_tuple(ck_tile::number<QKSubN>{}, ck_tile::number<QKChunk>{}),
                        ck_tile::make_tuple(args.stride_kn, args.stride_kd),
                        ck_tile::number<8>{},
                        ck_tile::number<1>{});
                auto k_breg = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
                auto k_dram_window = ck_tile::make_tile_window(
                    k_dram_view,
                    ck_tile::make_tuple(ck_tile::number<QKSubN>{}, ck_tile::number<QKChunk>{}),
                    ck_tile::multi_index<2>{0, 0},
                    k_breg.get_tile_distribution());
                return ck_tile::load_tile(k_dram_window);
            };

            if(!block_is_boundary)
            {
                k_breg_n0_d0 = load_k_chunk_breg(0, 0);
                k_breg_n0_d1 = load_k_chunk_breg(0, QKChunk);
                k_breg_n0_d2 = load_k_chunk_breg(0, 2 * QKChunk);
                k_breg_n0_d3 = load_k_chunk_breg(0, 3 * QKChunk);
                k_breg_n1_d0 = load_k_chunk_breg(QKSubN, 0);
                k_breg_n1_d1 = load_k_chunk_breg(QKSubN, QKChunk);
                k_breg_n1_d2 = load_k_chunk_breg(QKSubN, 2 * QKChunk);
                k_breg_n1_d3 = load_k_chunk_breg(QKSubN, 3 * QKChunk);
                if constexpr(ComputeDK)
                {
                    v_areg_d0 = load_v_dp_k32_areg(0);
                    v_areg_d1 = load_v_dp_k32_areg(QKChunk);
                    v_areg_d2 = load_v_dp_k32_areg(2 * QKChunk);
                    v_areg_d3 = load_v_dp_k32_areg(3 * QKChunk);
                }
            }
        }

        ck_tile::index_t active_start = 0;
        ck_tile::index_t active_end   = num_active;

        if(args.split_kv > 1)
        {
            ck_tile::index_t chunk_size = (num_active + args.split_kv - 1) / args.split_kv;
            active_start = split_id * chunk_size;
            active_end   = ck_tile::min((split_id + 1) * chunk_size, num_active);
        }

        if(active_start >= active_end)
        {
            if(args.split_kv > 1)
            {
                const ck_tile::index_t thread_id = ck_tile::get_thread_id();
                const ck_tile::index_t total_elements = BlockN * args.D;
                const ck_tile::index_t elements_per_thread = total_elements / 256;

                #pragma unroll
                for(ck_tile::index_t i = 0; i < elements_per_thread; ++i)
                {
                    const ck_tile::index_t linear_idx = thread_id * elements_per_thread + i;
                    const ck_tile::index_t n_rel = linear_idx / args.D;
                    const ck_tile::index_t d = linear_idx % args.D;
                    const ck_tile::index_t n = start_n + n_rel;

                    if(n < seqlen && d < args.D)
                    {
                        if constexpr(ComputeDV)
                        {
                            dv_ptr[static_cast<ck_tile::long_index_t>(n) * args.stride_dvn +
                                   static_cast<ck_tile::long_index_t>(d) * args.stride_dvd] =
                                ck_tile::type_convert<VGradDataType>(0.0f);
                        }
                        if constexpr(ComputeDK)
                        {
                            dk_ptr[static_cast<ck_tile::long_index_t>(n) * args.stride_dkn +
                                   static_cast<ck_tile::long_index_t>(d) * args.stride_dkd] =
                                ck_tile::type_convert<KGradDataType>(0.0f);
                        }
                    }
                }
            }
            return;
        }

        for(ck_tile::index_t active_i = active_start; active_i < active_end; ++active_i)
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

                    auto load_q_chunk_to_lds = [&](ck_tile::index_t d_base, auto buf_num) {
                        constexpr int buf = decltype(buf_num)::value;
                        auto q_dram_view =
                            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                                q_ptr + static_cast<ck_tile::long_index_t>(q_start) *
                                            args.stride_qm +
                                    static_cast<ck_tile::long_index_t>(d_base) * args.stride_qd,
                                ck_tile::make_tuple(ck_tile::number<BlockM>{},
                                                    ck_tile::number<QKChunk>{}),
                                ck_tile::make_tuple(args.stride_qm, args.stride_qd),
                                ck_tile::number<8>{},
                                ck_tile::number<1>{});
                        auto q_dram_window = ck_tile::make_tile_window(
                            q_dram_view,
                            ck_tile::make_tuple(ck_tile::number<BlockM>{},
                                                ck_tile::number<QKChunk>{}),
                            ck_tile::multi_index<2>{0, 0},
                            MakeBwdQAsyncDramDistribution<BlockM, QKChunk>());
                        q_dram_window.init_raw();

                        auto q_lds_store_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                            q_async_smem, MakeBwdQAsyncLdsStoreDescriptor<BlockM, QKChunk, buf>());
                        auto q_lds_store_window = ck_tile::make_tile_window(
                            q_lds_store_view,
                            MakeBwdQAsyncLdsStoreDescriptor<BlockM, QKChunk, buf>().get_lengths(),
                            ck_tile::multi_index<3>{0, 0, 0});
                        ck_tile::async_load_tile_raw(q_lds_store_window,
                                                     q_dram_window,
                                                     ck_tile::bool_constant<true>{},
                                                     ck_tile::bool_constant<false>{});
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

                    load_q_chunk_to_lds(0, ck_tile::number<0>{});
                    ck_tile::async_load_fence(0);
                    JengaBwdBlockSyncLdsLight();

                    load_q_chunk_to_lds(QKChunk, ck_tile::number<1>{});
                    auto q_chunk0_lds_window = make_q_chunk_window(ck_tile::number<0>{});
                    qk_chunk_asmem_breg_gemm(qk_acc_n0, q_chunk0_lds_window, k_breg_n0_d0);
                    qk_chunk_asmem_breg_gemm(qk_acc_n1, q_chunk0_lds_window, k_breg_n1_d0);

                    ck_tile::async_load_fence(0);
                    JengaBwdBlockSyncLdsLight();
                    load_q_chunk_to_lds(2 * QKChunk, ck_tile::number<0>{});
                    auto q_chunk1_lds_window = make_q_chunk_window(ck_tile::number<1>{});
                    qk_chunk_asmem_breg_gemm(qk_acc_n0, q_chunk1_lds_window, k_breg_n0_d1);
                    qk_chunk_asmem_breg_gemm(qk_acc_n1, q_chunk1_lds_window, k_breg_n1_d1);

                    ck_tile::async_load_fence(0);
                    JengaBwdBlockSyncLdsLight();
                    load_q_chunk_to_lds(3 * QKChunk, ck_tile::number<1>{});
                    auto q_chunk2_lds_window = make_q_chunk_window(ck_tile::number<0>{});
                    qk_chunk_asmem_breg_gemm(qk_acc_n0, q_chunk2_lds_window, k_breg_n0_d2);
                    qk_chunk_asmem_breg_gemm(qk_acc_n1, q_chunk2_lds_window, k_breg_n1_d2);

                    ck_tile::async_load_fence(0);
                    JengaBwdBlockSyncLdsLight();
                    auto q_chunk3_lds_window = make_q_chunk_window(ck_tile::number<1>{});
                    qk_chunk_asmem_breg_gemm(qk_acc_n0, q_chunk3_lds_window, k_breg_n0_d3);
                    qk_chunk_asmem_breg_gemm(qk_acc_n1, q_chunk3_lds_window, k_breg_n1_d3);
                    ck_tile::block_sync_lds();

                    QDataType* p_t_smem = smem.dv_input.p_t;
                    const auto qk_out_n0 = qk_chunk_asmem_breg_gemm.MakeOuputLayout(qk_acc_n0);
                    const auto qk_out_n1 = qk_chunk_asmem_breg_gemm.MakeOuputLayout(qk_acc_n1);
                    constexpr auto qk_spans = decltype(qk_out_n0)::get_distributed_spans();
                    constexpr auto M_spans = qk_spans[ck_tile::number<0>{}];
                    constexpr auto N_spans = qk_spans[ck_tile::number<1>{}];
                    float lse_regs[decltype(M_spans)::Impl::size()];
                    float delta_regs[decltype(M_spans)::Impl::size()];
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
                        delta_regs[slot] = row_valid
                                               ? delta_ptr[static_cast<ck_tile::long_index_t>(m) *
                                                           args.stride_delta_m]
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
                            const float p0 = JengaBwdFastExp2(qk_out_n0[tile_idx] * scale_log2e -
                                                              row_lse_log2);
                            const float p1 = JengaBwdFastExp2(qk_out_n1[tile_idx] * scale_log2e -
                                                              row_lse_log2);
                            p_t_smem[static_cast<ck_tile::long_index_t>(n_rel) *
                                         Policy::PTransposeLdsStride +
                                     m_rel] = ck_tile::type_convert<QDataType>(p0);
                            p_t_smem[static_cast<ck_tile::long_index_t>(QKSubN + n_rel) *
                                         Policy::PTransposeLdsStride +
                                     m_rel] = ck_tile::type_convert<QDataType>(p1);
                        });
                    });

                    ck_tile::block_sync_lds();

                    auto p_t_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                        p_t_smem,
                        MakePaddedRowMajorLdsDescriptor<BlockN,
                                                        BlockM,
                                                        Policy::PTransposeLdsStride>());
                    OGradDataType* do_qhalf_smem =
                        reinterpret_cast<OGradDataType*>(smem.dv_input.p_t);
                    auto load_do_qhalf_to_lds = [&](ck_tile::index_t q_base) {
                        auto load_do_chunk_to_lds = [&](ck_tile::index_t d_base, auto chunk_num) {
                            constexpr int chunk = decltype(chunk_num)::value;
                            auto do_dram_view =
                                ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                                    do_ptr + static_cast<ck_tile::long_index_t>(q_start + q_base) *
                                                 args.stride_dom +
                                        static_cast<ck_tile::long_index_t>(d_base) * args.stride_dod,
                                    ck_tile::make_tuple(ck_tile::number<32>{},
                                                        ck_tile::number<QKChunk>{}),
                                    ck_tile::make_tuple(args.stride_dom, args.stride_dod),
                                    ck_tile::number<kBwdAsyncDoVector>{},
                                    ck_tile::number<1>{});
                            auto do_dram_window = ck_tile::make_tile_window(
                                do_dram_view,
                                ck_tile::make_tuple(ck_tile::number<32>{},
                                                    ck_tile::number<QKChunk>{}),
                                ck_tile::multi_index<2>{0, 0},
                                MakeBwdDoAsyncDramDistribution<32, QKChunk>());
                            do_dram_window.init_raw();

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
                            ck_tile::async_load_tile_raw(do_lds_store_window,
                                                         do_dram_window,
                                                         ck_tile::bool_constant<true>{},
                                                         ck_tile::bool_constant<false>{});
                        };

                        load_do_chunk_to_lds(0, ck_tile::number<0>{});
                        load_do_chunk_to_lds(QKChunk, ck_tile::number<1>{});
                        load_do_chunk_to_lds(2 * QKChunk, ck_tile::number<2>{});
                        load_do_chunk_to_lds(3 * QKChunk, ck_tile::number<3>{});
                    };

                    if constexpr(ComputeDV)
                    {
                        if constexpr(ComputeDK)
                        {
                            load_do_qhalf_to_lds(0);
                        }
                        auto do_t_breg0 = load_do_t_k32_breg(q_start, 0);
                        auto do_t_breg1 = load_do_t_k32_breg(q_start, 32);
                        auto p_t_lds_window0 = ck_tile::make_tile_window(
                            p_t_lds_view,
                            ck_tile::make_tuple(ck_tile::number<BlockN>{}, ck_tile::number<32>{}),
                            ck_tile::multi_index<2>{0, 0},
                            dv_k32_breg_gemm.MakeABlockTile().get_tile_distribution());
                        auto p_t_areg0 = ck_tile::load_tile(p_t_lds_window0);
                        dv_k32_breg_gemm(dv_acc, p_t_areg0, do_t_breg0);

                        auto p_t_lds_window1 = ck_tile::make_tile_window(
                            p_t_lds_view,
                            ck_tile::make_tuple(ck_tile::number<BlockN>{}, ck_tile::number<32>{}),
                            ck_tile::multi_index<2>{0, 32},
                            dv_k32_breg_gemm.MakeABlockTile().get_tile_distribution());
                        auto p_t_areg1 = ck_tile::load_tile(p_t_lds_window1);
                        dv_k32_breg_gemm(dv_acc, p_t_areg1, do_t_breg1);
                    }

                    ck_tile::block_sync_lds();

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

                        if constexpr(!ComputeDV)
                        {
                            load_do_qhalf_to_lds(0);
                        }
                        ck_tile::async_load_fence(0);
                        JengaBwdBlockSyncLdsLight();
                        accumulate_dp_t_qhalf_from_lds(dp_t_acc0);

                        ck_tile::block_sync_lds();
                        load_do_qhalf_to_lds(32);
                        ck_tile::async_load_fence(0);
                        JengaBwdBlockSyncLdsLight();
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
                        auto q_t_breg0 = load_q_t_k32_breg(q_start, 0);
                        dk_k32_breg_gemm(dk_acc, ds_t_areg0, q_t_breg0);

                        auto ds_t_lds_window1 = ck_tile::make_tile_window(
                            p_t_lds_view,
                            ck_tile::make_tuple(ck_tile::number<BlockN>{}, ck_tile::number<32>{}),
                            ck_tile::multi_index<2>{0, 32},
                            dk_k32_breg_gemm.MakeABlockTile().get_tile_distribution());
                        auto ds_t_areg1 = ck_tile::load_tile(ds_t_lds_window1);
                        auto q_t_breg1 = load_q_t_k32_breg(q_start, 32);
                        dk_k32_breg_gemm(dk_acc, ds_t_areg1, q_t_breg1);
                    }

                    ck_tile::block_sync_lds();
                    continue;
                }
            }
        if constexpr(ComputeDV)
        {
                const auto dv_out = dv_breg_gemm.MakeOuputLayout(dv_acc);
                constexpr auto dv_spans = decltype(dv_out)::get_distributed_spans();
                ck_tile::sweep_tile_span(dv_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    ck_tile::sweep_tile_span(dv_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                            dv_out.get_tile_distribution(), tile_idx);
                        const ck_tile::index_t n_rel = x_idx.at(ck_tile::number<0>{});
                        const ck_tile::index_t d     = x_idx.at(ck_tile::number<1>{});
                        const ck_tile::index_t n     = start_n + n_rel;
                        if(n < seqlen && d < args.D)
                        {
                            dv_ptr[static_cast<ck_tile::long_index_t>(n) * args.stride_dvn +
                                   static_cast<ck_tile::long_index_t>(d) * args.stride_dvd] =
                                ck_tile::type_convert<VGradDataType>(dv_out[tile_idx]);
                        }
                    });
                });
        }

        if constexpr(ComputeDK)
        {
                const auto dk_out = dk_k32_breg_gemm.MakeOuputLayout(dk_acc);
                constexpr auto dk_spans = decltype(dk_out)::get_distributed_spans();
                ck_tile::sweep_tile_span(dk_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    ck_tile::sweep_tile_span(dk_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                            dk_out.get_tile_distribution(), tile_idx);
                        const ck_tile::index_t n_rel = x_idx.at(ck_tile::number<0>{});
                        const ck_tile::index_t d     = x_idx.at(ck_tile::number<1>{});
                        const ck_tile::index_t n     = start_n + n_rel;
                        if(n < seqlen && d < args.D)
                        {
                            dk_ptr[static_cast<ck_tile::long_index_t>(n) * args.stride_dkn +
                                   static_cast<ck_tile::long_index_t>(d) * args.stride_dkd] =
                                ck_tile::type_convert<KGradDataType>(dk_out[tile_idx] * args.sm_scale);
                        }
                    });
                });
        }
    }
};

} // namespace jenga
} // namespace example
} // namespace ck_tile
