// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dq_config.hpp"
#include "ck_tile/ops/jenga/pipeline/jenga_bwd_dq_policy.hpp"
namespace ck_tile {
template <typename Problem, typename Policy = JengaBwdDqDefaultPolicy<Problem>>
struct JengaBwdDqPipeline
{
    using QDataType     = typename Problem::QDataType;
    using KDataType     = typename Problem::KDataType;
    using VDataType     = typename Problem::VDataType;
    using OGradDataType = typename Problem::OGradDataType;
    using QGradDataType = typename Problem::QGradDataType;
    using AccDataType   = float;
    using LSEDataType   = typename Problem::LSEDataType;
    using DDataType     = typename Problem::DDataType;
    static constexpr index_t BlockM          = Problem::kBlockM;
    static constexpr index_t BlockN          = Problem::kBlockN;
    static constexpr index_t HeadDim         = Problem::kHeadDim;
    static constexpr index_t MaxNnz          = Problem::kMaxNnz;
    static constexpr index_t ThreadsPerBlock = Problem::kBlockSize;
    static constexpr index_t kBlockSize      = Problem::kBlockSize;
    static constexpr index_t kBlockPerCu     = Problem::kBlockPerCu;
    CK_TILE_HOST static std::string GetName()
    {
        return std::string("jenga_bwd_dq_m") + std::to_string(BlockM) + "_n" +
               std::to_string(BlockN) + "_d" + std::to_string(HeadDim) + "_nnz" +
               std::to_string(MaxNnz) + "_bf16_o" + std::to_string(kBlockPerCu);
    }
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }
    template <typename Kargs,
              typename QDram,
              typename DoDram,
              typename KDramQK,
              typename VDramQK,
              typename KDramDQ,
              typename DQDram>
    CK_TILE_DEVICE void operator()(const Kargs& args,
                                   ck_tile::index_t start_m_block,
                                   ck_tile::index_t off_hz,
                                   ck_tile::index_t seqlen_q,
                                   ck_tile::index_t seqlen_k,
                                   ck_tile::index_t start_m,
                                   ck_tile::index_t kv_stage,
                                   QGradDataType* dq,
                                   const QDram& q_dram,
                                   const DoDram& do_dram,
                                   const KDramQK& k_dram_qk,
                                   const VDramQK& v_dram_qk,
                                   const KDramDQ& k_dram_dq,
                                   const DQDram& dq_dram,
                                   const LSEDataType* lse_ptr,
                                   const DDataType* delta_ptr,
                                   const int32_t* lut_ptr_args) const
    {
        constexpr auto qk_chunk_asmem_breg_gemm = typename Policy::QKChunkASmemBRegBlockGemm{};
        constexpr auto dp_k32_areg_breg_gemm = typename Policy::DPK32ARegBRegBlockGemm{};
        constexpr auto dq_k32_breg_gemm = typename Policy::DQK32BRegBlockGemm{};
        (void)off_hz;
        const int32_t* lut = lut_ptr_args;
        const int32_t num_active =
            args.lut_size_ptr[static_cast<ck_tile::long_index_t>(off_hz) * args.num_q_blocks +
                              start_m_block];
        if(num_active <= 0)
        {
            return;
        }
        constexpr float qk_scale = ck_tile::log2e_v<float>;
        const float scale_log2e  = args.sm_scale * qk_scale;
        constexpr int QKHeadChunk = JengaMmacQKChunkConfig::kK;
        constexpr int DPHeadChunk = 32;
        constexpr int QKSubN      = 32;
        __shared__ typename Policy::LdsStorage smem;
        auto dq_acc = decltype(dq_k32_breg_gemm.MakeCBlockTile()){};
        ck_tile::clear_tile(dq_acc);
        auto load_q_chunk_breg = [&](ck_tile::index_t m_base, ck_tile::index_t d_base) {
            auto q_breg = qk_chunk_asmem_breg_gemm.MakeBBlockTile();
            auto q_dram_window = ck_tile::make_tile_window(
                q_dram,
                ck_tile::make_tuple(ck_tile::number<QKSubN>{}, ck_tile::number<QKHeadChunk>{}),
                ck_tile::multi_index<2>{start_m + m_base, d_base},
                q_breg.get_tile_distribution());
            return ck_tile::load_tile(q_dram_window);
        };
        auto load_do_dp_k32_areg = [&](ck_tile::index_t d_base) {
            auto do_areg = dp_k32_areg_breg_gemm.MakeABlockTile();
            auto do_dram_window = ck_tile::make_tile_window(
                do_dram,
                ck_tile::make_tuple(ck_tile::number<BlockM>{}, ck_tile::number<DPHeadChunk>{}),
                ck_tile::multi_index<2>{start_m, d_base},
                do_areg.get_tile_distribution());
            return ck_tile::load_tile(do_dram_window);
        };
        auto load_k_t_k32_breg = [&](ck_tile::index_t n_start, ck_tile::index_t k_base) {
            auto k_t_breg = dq_k32_breg_gemm.MakeBBlockTile();
            auto k_t_dram_window = ck_tile::make_tile_window(
                k_dram_dq,
                ck_tile::make_tuple(ck_tile::number<HeadDim>{}, ck_tile::number<32>{}),
                ck_tile::multi_index<2>{0, n_start + k_base},
                k_t_breg.get_tile_distribution());
            return ck_tile::load_tile(k_t_dram_window);
        };
        QDataType* q_async_smem = smem.q_async;
        __shared__ float lse_s[BlockM];
        __shared__ float delta_s[BlockM];
        for(ck_tile::index_t m_in_block = threadIdx.x; m_in_block < BlockM; m_in_block += blockDim.x)
        {
            const ck_tile::index_t m = start_m + m_in_block;
            lse_s[m_in_block] = (m < seqlen_q)
                                    ? lse_ptr[static_cast<ck_tile::long_index_t>(m) *
                                              args.stride_lm]
                                    : 0.0f;
            delta_s[m_in_block] = (m < seqlen_q)
                                      ? delta_ptr[static_cast<ck_tile::long_index_t>(m) *
                                                  args.stride_dm]
                                      : 0.0f;
        }
        auto q_breg_n0_d0 = load_q_chunk_breg(0, 0);
        auto q_breg_n0_d1 = load_q_chunk_breg(0, QKHeadChunk);
        auto q_breg_n1_d0 = load_q_chunk_breg(QKSubN, 0);
        auto q_breg_n1_d1 = load_q_chunk_breg(QKSubN, QKHeadChunk);
        auto do_areg_d0 = load_do_dp_k32_areg(0);
        auto do_areg_d1 = load_do_dp_k32_areg(DPHeadChunk);
        auto do_areg_d2 = load_do_dp_k32_areg(2 * DPHeadChunk);
        auto do_areg_d3 = load_do_dp_k32_areg(3 * DPHeadChunk);
        const ck_tile::index_t active_total =
            num_active < args.max_nnz ? num_active : args.max_nnz;
        const ck_tile::index_t active_begin =
            active_total * kv_stage / args.kv_stage_count;
        const ck_tile::index_t active_end =
            active_total * (kv_stage + 1) / args.kv_stage_count;
        const ck_tile::index_t stage_active = active_end - active_begin;
        __shared__ int32_t active_lut_s[MaxNnz];
        for(ck_tile::index_t i = threadIdx.x; i < stage_active; i += blockDim.x)
        {
            active_lut_s[i] =
                lut[static_cast<ck_tile::long_index_t>(active_begin + i) * args.stride_lutk];
        }
        ck_tile::block_sync_lds();
        #pragma unroll 2
        for(ck_tile::index_t active_i = 0; active_i < stage_active; ++active_i)
        {
            const ck_tile::index_t block_idx = active_lut_s[active_i];
            const ck_tile::index_t start_n = block_idx * BlockN;
            if(start_n >= seqlen_k)
            {
                continue;
            }
            {
                auto qk_acc_n0 = decltype(qk_chunk_asmem_breg_gemm.MakeCBlockTile()){};
                auto qk_acc_n1 = decltype(qk_chunk_asmem_breg_gemm.MakeCBlockTile()){};
                ck_tile::clear_tile(qk_acc_n0);
                ck_tile::clear_tile(qk_acc_n1);
                auto load_k_chunk_to_lds = [&](ck_tile::index_t d_base, auto buf_num) {
                    constexpr int buf = decltype(buf_num)::value;
                    auto k_dram_window = ck_tile::make_tile_window(
                        k_dram_qk,
                        ck_tile::make_tuple(ck_tile::number<BlockN>{},
                                            ck_tile::number<QKHeadChunk>{}),
                        ck_tile::multi_index<2>{start_n, d_base},
                        MakeBwdQAsyncDramDistribution<BlockN, QKHeadChunk>());
                    k_dram_window.init_raw();
                    auto k_lds_store_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                        q_async_smem,
                        MakeBwdQAsyncLdsStoreDescriptor<BlockN, QKHeadChunk, buf>());
                    auto k_lds_store_window = ck_tile::make_tile_window(
                        k_lds_store_view,
                        MakeBwdQAsyncLdsStoreDescriptor<BlockN, QKHeadChunk, buf>().get_lengths(),
                        ck_tile::multi_index<3>{0, 0, 0});
                    ck_tile::async_load_tile_raw(k_lds_store_window,
                                                 k_dram_window,
                                                 ck_tile::bool_constant<true>{},
                                                 ck_tile::bool_constant<false>{});
                };
                auto make_k_chunk_window = [&](auto buf_num) {
                    constexpr int buf = decltype(buf_num)::value;
                    return ck_tile::make_tile_window(
                        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                            q_async_smem,
                            MakeBwdQAsyncLdsLoadDescriptor<BlockN, QKHeadChunk>()),
                        ck_tile::make_tuple(ck_tile::number<BlockN>{},
                                            ck_tile::number<QKHeadChunk>{}),
                        ck_tile::multi_index<2>{buf * BlockN, 0});
                };
                load_k_chunk_to_lds(0, ck_tile::number<0>{});
                ck_tile::async_load_fence(0);
                JengaBwdBlockSyncLdsLight();
                load_k_chunk_to_lds(QKHeadChunk, ck_tile::number<1>{});
                auto k_chunk0_lds_window = make_k_chunk_window(ck_tile::number<0>{});
                qk_chunk_asmem_breg_gemm(qk_acc_n0, k_chunk0_lds_window, q_breg_n0_d0);
                qk_chunk_asmem_breg_gemm(qk_acc_n1, k_chunk0_lds_window, q_breg_n1_d0);
                ck_tile::async_load_fence(0);
                JengaBwdBlockSyncLdsLight();
                auto k_chunk1_lds_window = make_k_chunk_window(ck_tile::number<1>{});
                qk_chunk_asmem_breg_gemm(qk_acc_n0, k_chunk1_lds_window, q_breg_n0_d1);
                qk_chunk_asmem_breg_gemm(qk_acc_n1, k_chunk1_lds_window, q_breg_n1_d1);
                QDataType* p_t_smem = smem.dv_input.p_t;
                const auto qk_out_n0 = qk_chunk_asmem_breg_gemm.MakeOuputLayout(qk_acc_n0);
                const auto qk_out_n1 = qk_chunk_asmem_breg_gemm.MakeOuputLayout(qk_acc_n1);
                const float qk_bias =
                    (block_idx >= args.text_block_start) ? args.text_amp : 0.0f;
                constexpr auto qk_spans = decltype(qk_out_n0)::get_distributed_spans();
                constexpr auto N_spans = qk_spans[ck_tile::number<0>{}];
                constexpr auto M_spans = qk_spans[ck_tile::number<1>{}];
                ck_tile::sweep_tile_span(M_spans, [&](auto idx1) {
                    ck_tile::sweep_tile_span(N_spans, [&](auto idx0) {
                        constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                            qk_out_n0.get_tile_distribution(), tile_idx);
                        const ck_tile::index_t n_rel = x_idx.at(ck_tile::number<0>{});
                        const ck_tile::index_t m_rel = x_idx.at(ck_tile::number<1>{});
                        const float row_lse0 = lse_s[m_rel];
                        const float row_lse1 = lse_s[32 + m_rel];
                        const float p0 = JengaBwdFastExp2(qk_out_n0[tile_idx] * scale_log2e + qk_bias -
                                                          row_lse0);
                        const float p1 = JengaBwdFastExp2(qk_out_n1[tile_idx] * scale_log2e + qk_bias -
                                                          row_lse1);
                        p_t_smem[static_cast<ck_tile::long_index_t>(m_rel) *
                                     Policy::PTransposeLdsStride +
                                 n_rel] = ck_tile::type_convert<QDataType>(p0);
                        p_t_smem[static_cast<ck_tile::long_index_t>(32 + m_rel) *
                                     Policy::PTransposeLdsStride +
                                 n_rel] = ck_tile::type_convert<QDataType>(p1);
                    });
                });
                auto p_t_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    p_t_smem,
                    MakePaddedRowMajorLdsDescriptor<BlockM,
                                                    BlockN,
                                                    Policy::PTransposeLdsStride>());

                VDataType* v_qhalf_smem =
                    reinterpret_cast<VDataType*>(smem.dv_input.p_t);
                auto load_v_qhalf_to_lds = [&](ck_tile::index_t n_base) {
                    auto load_v_chunk_to_lds = [&](ck_tile::index_t d_base, auto chunk_num) {
                        constexpr int chunk = decltype(chunk_num)::value;
                        auto v_dram_window = ck_tile::make_tile_window(
                            v_dram_qk,
                            ck_tile::make_tuple(ck_tile::number<32>{},
                                                ck_tile::number<DPHeadChunk>{}),
                            ck_tile::multi_index<2>{start_n + n_base, d_base},
                            MakeBwdDoAsyncDramDistribution<32, DPHeadChunk>());
                        v_dram_window.init_raw();
                        auto v_lds_store_view =
                            ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                                v_qhalf_smem,
                                MakeBwdDoAsyncLdsStoreDescriptor<
                                    DPHeadChunk,
                                    Policy::DoQHalfLdsBaseOffset,
                                    chunk>());
                        auto v_lds_store_window = ck_tile::make_tile_window(
                            v_lds_store_view,
                            MakeBwdDoAsyncLdsStoreDescriptor<
                                DPHeadChunk,
                                Policy::DoQHalfLdsBaseOffset,
                                chunk>().get_lengths(),
                            ck_tile::multi_index<3>{0, 0, 0});
                        ck_tile::async_load_tile_raw(v_lds_store_window,
                                                     v_dram_window,
                                                     ck_tile::bool_constant<true>{},
                                                     ck_tile::bool_constant<false>{});
                    };
                    load_v_chunk_to_lds(0, ck_tile::number<0>{});
                    load_v_chunk_to_lds(DPHeadChunk, ck_tile::number<1>{});
                    load_v_chunk_to_lds(2 * DPHeadChunk, ck_tile::number<2>{});
                    load_v_chunk_to_lds(3 * DPHeadChunk, ck_tile::number<3>{});
                };
                auto load_v_dp_k32_breg_from_lds = [&](auto chunk_num) {
                    constexpr int chunk = decltype(chunk_num)::value;
                    auto v_breg = dp_k32_areg_breg_gemm.MakeBBlockTile();
                    auto v_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                        v_qhalf_smem,
                        MakeBwdDoAsyncLdsLoadDescriptor<
                            DPHeadChunk,
                            Policy::DoQHalfLdsBaseOffset,
                            chunk>());
                    auto v_lds_window = ck_tile::make_tile_window(
                        v_lds_view,
                        ck_tile::make_tuple(ck_tile::number<32>{},
                                            ck_tile::number<DPHeadChunk>{}),
                        ck_tile::multi_index<2>{0, 0},
                        v_breg.get_tile_distribution());
                    return ck_tile::load_tile(v_lds_window);
                };
                auto dp_t_acc0 = decltype(dp_k32_areg_breg_gemm.MakeCBlockTile()){};
                auto dp_t_acc1 = decltype(dp_k32_areg_breg_gemm.MakeCBlockTile()){};
                ck_tile::clear_tile(dp_t_acc0);
                ck_tile::clear_tile(dp_t_acc1);
                auto accumulate_dp_t_qhalf_from_lds = [&](auto& dp_t_acc) {
                    auto v_dp_breg0 = load_v_dp_k32_breg_from_lds(ck_tile::number<0>{});
                    auto v_dp_breg1 = load_v_dp_k32_breg_from_lds(ck_tile::number<1>{});
                    auto v_dp_breg2 = load_v_dp_k32_breg_from_lds(ck_tile::number<2>{});
                    auto v_dp_breg3 = load_v_dp_k32_breg_from_lds(ck_tile::number<3>{});
                    dp_k32_areg_breg_gemm(dp_t_acc, do_areg_d0, v_dp_breg0);
                    dp_k32_areg_breg_gemm(dp_t_acc, do_areg_d1, v_dp_breg1);
                    dp_k32_areg_breg_gemm(dp_t_acc, do_areg_d2, v_dp_breg2);
                    dp_k32_areg_breg_gemm(dp_t_acc, do_areg_d3, v_dp_breg3);
                };
                load_v_qhalf_to_lds(0);
                ck_tile::async_load_fence(0);
                JengaBwdBlockSyncLdsLight();
                accumulate_dp_t_qhalf_from_lds(dp_t_acc0);
                load_v_qhalf_to_lds(32);
                ck_tile::async_load_fence(0);
                JengaBwdBlockSyncLdsLight();
                accumulate_dp_t_qhalf_from_lds(dp_t_acc1);
                const auto dp_t_out0 = dp_k32_areg_breg_gemm.MakeOuputLayout(dp_t_acc0);
                const auto dp_t_out1 = dp_k32_areg_breg_gemm.MakeOuputLayout(dp_t_acc1);
                constexpr auto dp_t_spans = decltype(dp_t_out0)::get_distributed_spans();
                auto k_t_breg0 = load_k_t_k32_breg(start_n, 0);
                auto k_t_breg1 = load_k_t_k32_breg(start_n, 32);

                ck_tile::sweep_tile_span(dp_t_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    ck_tile::sweep_tile_span(dp_t_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                            dp_t_out0.get_tile_distribution(), tile_idx);
                        const ck_tile::index_t m_rel = x_idx.at(ck_tile::number<0>{});
                        const ck_tile::index_t n_rel = x_idx.at(ck_tile::number<1>{});

                        const float delta = delta_s[m_rel];
                        const ck_tile::long_index_t p0_offset =
                            static_cast<ck_tile::long_index_t>(m_rel) *
                                Policy::PTransposeLdsStride +
                            n_rel;
                        const ck_tile::long_index_t p1_offset =
                            static_cast<ck_tile::long_index_t>(m_rel) *
                                Policy::PTransposeLdsStride +
                            (32 + n_rel);
                        const float p0 = ck_tile::type_convert<float>(p_t_smem[p0_offset]);
                        const float p1 = ck_tile::type_convert<float>(p_t_smem[p1_offset]);
                        p_t_smem[p0_offset] = ck_tile::type_convert<QDataType>(
                            p0 * (dp_t_out0[tile_idx] - delta));
                        p_t_smem[p1_offset] = ck_tile::type_convert<QDataType>(
                            p1 * (dp_t_out1[tile_idx] - delta));
                    });
                });
                ck_tile::block_sync_lds();
                auto ds_lds_window0 = ck_tile::make_tile_window(
                    p_t_lds_view,
                    ck_tile::make_tuple(ck_tile::number<BlockM>{}, ck_tile::number<32>{}),
                    ck_tile::multi_index<2>{0, 0},
                    dq_k32_breg_gemm.MakeABlockTile().get_tile_distribution());
                auto ds_areg0 = ck_tile::load_tile(ds_lds_window0);
                dq_k32_breg_gemm(dq_acc, ds_areg0, k_t_breg0);
                auto ds_lds_window1 = ck_tile::make_tile_window(
                    p_t_lds_view,
                    ck_tile::make_tuple(ck_tile::number<BlockM>{}, ck_tile::number<32>{}),
                    ck_tile::multi_index<2>{0, 32},
                    dq_k32_breg_gemm.MakeABlockTile().get_tile_distribution());
                auto ds_areg1 = ck_tile::load_tile(ds_lds_window1);
                dq_k32_breg_gemm(dq_acc, ds_areg1, k_t_breg1);
                ck_tile::block_sync_lds();
            }
        }
        const auto dq_out = dq_k32_breg_gemm.MakeOuputLayout(dq_acc);
        (void)dq;
        auto dq_scaled = ck_tile::tile_elementwise_in(
            [&](const auto& x) {
                return ck_tile::type_convert<QGradDataType>(x * args.sm_scale);
            },
            dq_out);
        auto dq_dram_window = ck_tile::make_tile_window(
            dq_dram,
            ck_tile::make_tuple(ck_tile::number<BlockM>{}, ck_tile::number<HeadDim>{}),
            ck_tile::multi_index<2>{start_m, 0},
            dq_scaled.get_tile_distribution());
        ck_tile::store_tile(dq_dram_window, dq_scaled);
    }
};
} // namespace ck_tile
