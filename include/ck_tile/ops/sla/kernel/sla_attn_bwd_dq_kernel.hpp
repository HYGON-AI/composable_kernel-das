// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dq_pipeline.hpp"
#include <string>
namespace ck_tile {
template <typename Problem_, typename Policy_ = SlaAttnBwdDqDefaultPolicy<Problem_>>
struct SlaAttnBwdDqKernel
{
    using Problem  = ck_tile::remove_cvref_t<Problem_>;
    using Policy   = ck_tile::remove_cvref_t<Policy_>;
    using Pipeline = SlaAttnBwdDqPipeline<Problem, Policy>;
    static constexpr ck_tile::index_t kBlockSize = Pipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = Pipeline::kBlockPerCu;
    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* do_ptr;
        const void* delta_ptr;
        const void* lse_ptr;
        void* dq_ptr;
        void* dq_workspace;
        int kv_stage_count;
        const int64_t* lut_ptr;
        float sm_scale;
        ck_tile::index_t max_seqlen_q;
        ck_tile::index_t topk;
        ck_tile::index_t lut_block_m;
        ck_tile::index_t stride_qz;
        ck_tile::index_t stride_qm;
        ck_tile::index_t stride_qk;
        ck_tile::index_t stride_kz;
        ck_tile::index_t stride_kn;
        ck_tile::index_t stride_kk;
        ck_tile::index_t stride_vz;
        ck_tile::index_t stride_vn;
        ck_tile::index_t stride_vk;
        ck_tile::index_t stride_doz;
        ck_tile::index_t stride_dom;
        ck_tile::index_t stride_dok;
        ck_tile::index_t stride_dqz;
        ck_tile::index_t stride_dqm;
        ck_tile::index_t stride_dqk;
        ck_tile::index_t stride_dz;
        ck_tile::index_t stride_dm;
        ck_tile::index_t stride_lz;
        ck_tile::index_t stride_lm;
        ck_tile::index_t stride_lutz;
        ck_tile::index_t stride_lutm;
        ck_tile::index_t stride_lutk;
    };
    CK_TILE_HOST static std::string GetName() { return Pipeline::GetName(); }
    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* q_ptr,
                                                   const void* k_ptr,
                                                   const void* v_ptr,
                                                   const void* do_ptr,
                                                   const void* delta_ptr,
                                                   const void* lse_ptr,
                                                   void* dq_ptr,
                                                   void* dq_workspace,
                                                   int kv_stage_count,
                                                   const void* lut_ptr,
                                                   float sm_scale,
                                                   ck_tile::index_t max_seqlen_q,
                                                   ck_tile::index_t topk,
                                                   ck_tile::index_t lut_block_m,
                                                   ck_tile::index_t stride_qz,
                                                   ck_tile::index_t stride_qm,
                                                   ck_tile::index_t stride_qk,
                                                   ck_tile::index_t stride_kz,
                                                   ck_tile::index_t stride_kn,
                                                   ck_tile::index_t stride_kk,
                                                   ck_tile::index_t stride_vz,
                                                   ck_tile::index_t stride_vn,
                                                   ck_tile::index_t stride_vk,
                                                   ck_tile::index_t stride_doz,
                                                   ck_tile::index_t stride_dom,
                                                   ck_tile::index_t stride_dok,
                                                   ck_tile::index_t stride_dqz,
                                                   ck_tile::index_t stride_dqm,
                                                   ck_tile::index_t stride_dqk,
                                                   ck_tile::index_t stride_dz,
                                                   ck_tile::index_t stride_dm,
                                                   ck_tile::index_t stride_lz,
                                                   ck_tile::index_t stride_lm,
                                                   ck_tile::index_t stride_lutz,
                                                   ck_tile::index_t stride_lutm,
                                                   ck_tile::index_t stride_lutk)
    {
        return Kargs{q_ptr,
                     k_ptr,
                     v_ptr,
                     do_ptr,
                     delta_ptr,
                     lse_ptr,
                     dq_ptr,
                     dq_workspace,
                     kv_stage_count,
                     reinterpret_cast<const int64_t*>(lut_ptr),
                     sm_scale,
                     max_seqlen_q,
                     topk,
                     lut_block_m,
                     stride_qz,
                     stride_qm,
                     stride_qk,
                     stride_kz,
                     stride_kn,
                     stride_kk,
                     stride_vz,
                     stride_vn,
                     stride_vk,
                     stride_doz,
                     stride_dom,
                     stride_dok,
                     stride_dqz,
                     stride_dqm,
                     stride_dqk,
                     stride_dz,
                     stride_dm,
                     stride_lz,
                     stride_lm,
                     stride_lutz,
                     stride_lutm,
                     stride_lutk};
    }
    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size,
             ck_tile::index_t num_heads,
             ck_tile::index_t max_seqlen_q)
    {
        return dim3(ck_tile::integer_divide_ceil(max_seqlen_q, Problem::kBlockM),
                    batch_size * num_heads,
                    1);
    }
    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Pipeline::GetSmemSize();
    }
    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using QDataType     = typename Problem::QDataType;
        using KDataType     = typename Problem::KDataType;
        using VDataType     = typename Problem::VDataType;
        using OGradDataType = typename Problem::OGradDataType;
        using DDataType     = typename Problem::DDataType;
        using LSEDataType   = typename Problem::LSEDataType;
        using QGradDataType = typename Problem::QGradDataType;
        constexpr index_t kBlockM  = Problem::kBlockM;
        constexpr index_t kBlockN  = Problem::kBlockN;
        constexpr index_t kHeadDim = Problem::kHeadDim;
        const index_t start_m_block = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const index_t off_hz        = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const index_t seqlen_q      = kargs.max_seqlen_q;
        const index_t seqlen_k      = seqlen_q;
        const index_t start_m  = start_m_block * kBlockM;
        if(start_m >= seqlen_q)
        {
            return;
        }
        const index_t kv_stage = __builtin_amdgcn_readfirstlane(blockIdx.z);
        const auto q = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                       static_cast<long_index_t>(off_hz) * kargs.stride_qz;
        const auto k = reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                       static_cast<long_index_t>(off_hz) * kargs.stride_kz;
        const auto v = reinterpret_cast<const VDataType*>(kargs.v_ptr) +
                       static_cast<long_index_t>(off_hz) * kargs.stride_vz;
        const auto do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                            static_cast<long_index_t>(off_hz) * kargs.stride_doz;
        auto dq = (kargs.kv_stage_count > 1)
            ? (reinterpret_cast<QGradDataType*>(kargs.dq_workspace) +
               static_cast<long_index_t>(kv_stage) * (static_cast<long_index_t>(gridDim.y) * kargs.stride_dqz) +
               static_cast<long_index_t>(off_hz) * kargs.stride_dqz)
            : (reinterpret_cast<QGradDataType*>(kargs.dq_ptr) +
               static_cast<long_index_t>(off_hz) * kargs.stride_dqz);
        const index_t lut_q_block = start_m / kargs.lut_block_m;
        const auto lut = kargs.lut_ptr + static_cast<long_index_t>(off_hz) * kargs.stride_lutz +
                         static_cast<long_index_t>(lut_q_block) * kargs.stride_lutm;
        const auto lse = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                         static_cast<long_index_t>(off_hz) * kargs.stride_lz;
        const auto delta = reinterpret_cast<const DDataType*>(kargs.delta_ptr) +
                           static_cast<long_index_t>(off_hz) * kargs.stride_dz;
        const auto q_dram = make_naive_tensor_view<address_space_enum::global>(
            q,
            make_tuple(seqlen_q, kHeadDim),
            make_tuple(kargs.stride_qm, kargs.stride_qk),
            number<8>{},
            number<1>{});
        const auto do_dram = make_naive_tensor_view<address_space_enum::global>(
            do_ptr,
            make_tuple(seqlen_q, kHeadDim),
            make_tuple(kargs.stride_dom, kargs.stride_dok),
            number<8>{},
            number<1>{});
        const auto k_dram_qk = make_naive_tensor_view<address_space_enum::global>(
            k,
            make_tuple(seqlen_k, kHeadDim),
            make_tuple(kargs.stride_kn, kargs.stride_kk),
            number<8>{},
            number<1>{});
        const auto v_dram_qk = make_naive_tensor_view<address_space_enum::global>(
            v,
            make_tuple(seqlen_k, kHeadDim),
            make_tuple(kargs.stride_vn, kargs.stride_vk),
            number<8>{},
            number<1>{});
        const auto k_dram_dq = make_naive_tensor_view<address_space_enum::global>(
            k,
            make_tuple(kHeadDim, seqlen_k),
            make_tuple(kargs.stride_kk, kargs.stride_kn),
            number<1>{},
            number<1>{});
        const auto dq_dram = make_naive_tensor_view<address_space_enum::global>(
            dq,
            make_tuple(seqlen_q, kHeadDim),
            make_tuple(kargs.stride_dqm, kargs.stride_dqk),
            number<1>{},
            number<1>{});
        Pipeline{}(kargs,
                   start_m_block,
                   off_hz,
                   seqlen_q,
                   seqlen_k,
                   start_m,
                   kv_stage,
                   dq,
                   q_dram,
                   do_dram,
                   k_dram_qk,
                   v_dram_qk,
                   k_dram_dq,
                   dq_dram,
                   lse,
                   delta,
                   lut);
    }
};
} // namespace ck_tile
