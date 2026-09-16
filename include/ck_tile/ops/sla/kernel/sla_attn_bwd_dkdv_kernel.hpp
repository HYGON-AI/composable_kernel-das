// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile {
namespace example {
namespace sla {

struct SlaAttnBwdDkdvKernelArgument
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* do_ptr;
    const void* delta_ptr;
    const void* lse_ptr;
    const void* lse_text_ptr;
    const int32_t* rlut_ptr;
    const int32_t* rlut_size_ptr;
    const int32_t* kv_perm_ptr;
    const int32_t* seqlens_ptr;
    void* dk_ptr;
    void* dv_ptr;

    index_t H;
    index_t N_Q;
    index_t D;
    index_t text_block_start;
    float sm_scale;

    index_t stride_qz;
    index_t stride_qh;
    index_t stride_qm;
    index_t stride_qd;
    index_t stride_kz;
    index_t stride_kh;
    index_t stride_kn;
    index_t stride_kd;
    index_t stride_vz;
    index_t stride_vh;
    index_t stride_vn;
    index_t stride_vd;
    index_t stride_doz;
    index_t stride_doh;
    index_t stride_dom;
    index_t stride_dod;
    index_t stride_delta_z;
    index_t stride_delta_m;
    index_t stride_dkz;
    index_t stride_dkh;
    index_t stride_dkn;
    index_t stride_dkd;
    index_t stride_dvz;
    index_t stride_dvh;
    index_t stride_dvn;
    index_t stride_dvd;
    index_t stride_lz;
    index_t stride_lm;
    index_t stride_rlutz;
    index_t stride_rlutn;
    index_t stride_rlutk;
    index_t stride_rlut_size_z;
    index_t stride_rlut_size_n;

};

} // namespace sla
} // namespace example
} // namespace ck_tile

#include "ck_tile/ops/sla/pipeline/sla_attn_bwd_dkdv_pipeline.hpp"

namespace ck_tile {
namespace example {
namespace sla {

template <typename Pipeline_>
struct SlaAttnBwdDkdvTilePartitioner
{
    using Pipeline = remove_cvref_t<Pipeline_>;

    CK_TILE_HOST static dim3 GridSize(ck_tile::index_t n_kv_blocks,
                                      ck_tile::index_t batch_heads)
    {
        return dim3(n_kv_blocks, batch_heads);
    }

    CK_TILE_DEVICE auto operator()() const
    {
        return make_tuple(static_cast<index_t>(blockIdx.x),
                          static_cast<index_t>(blockIdx.y));
    }
};

template <typename Pipeline_>
struct SlaAttnBwdDkdvKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using TilePartitioner = SlaAttnBwdDkdvTilePartitioner<Pipeline>;

    static constexpr index_t BlockM          = Pipeline::BlockM;
    static constexpr index_t BlockN          = Pipeline::BlockN;
    static constexpr index_t HeadDim         = Pipeline::HeadDim;
    static constexpr index_t ThreadsPerBlock = Pipeline::ThreadsPerBlock;
    static constexpr index_t KernelBlockSize = ThreadsPerBlock;

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t n_kv_blocks, ck_tile::index_t batch_heads)
    {
        return TilePartitioner::GridSize(n_kv_blocks, batch_heads);
    }

    CK_TILE_DEVICE void operator()(SlaAttnBwdDkdvKernelArgument args) const
    {
        const auto [kv_block_orig, off_hz] = TilePartitioner{}();
        const ck_tile::index_t batch  = off_hz / args.H;
        const ck_tile::index_t head   = off_hz - batch * args.H;
        const ck_tile::index_t seqlen =
            args.seqlens_ptr == nullptr ? args.N_Q : args.seqlens_ptr[batch];

        // Optional host-computed workgroup permutation: improves L2 reuse of
        // Q/dO rows by dispatching KV blocks with overlapping active sets
        // close together. Outputs are per-KV-block and order independent.
        ck_tile::index_t kv_block = kv_block_orig;
        if(args.kv_perm_ptr != nullptr)
        {
            kv_block =
                args.kv_perm_ptr[static_cast<ck_tile::long_index_t>(off_hz) * gridDim.x +
                                 kv_block_orig];
        }

        if(kv_block * BlockN >= seqlen)
        {
            return;
        }

        using QDataType     = typename Pipeline::QDataType;
        using KDataType     = typename Pipeline::KDataType;
        using VDataType     = typename Pipeline::VDataType;
        using OGradDataType = typename Pipeline::OGradDataType;
        using KGradDataType = typename Pipeline::KGradDataType;
        using VGradDataType = typename Pipeline::VGradDataType;

        const auto batch_offset = static_cast<ck_tile::long_index_t>(batch);
        const auto head_offset  = static_cast<ck_tile::long_index_t>(head);
        const auto* q_ptr = static_cast<const QDataType*>(args.q_ptr) +
                            batch_offset * args.stride_qz +
                            head_offset * args.stride_qh;
        const auto* k_ptr = static_cast<const KDataType*>(args.k_ptr) +
                            batch_offset * args.stride_kz +
                            head_offset * args.stride_kh;
        const auto* v_ptr = static_cast<const VDataType*>(args.v_ptr) +
                            batch_offset * args.stride_vz +
                            head_offset * args.stride_vh;
        const auto* do_ptr = static_cast<const OGradDataType*>(args.do_ptr) +
                             batch_offset * args.stride_doz +
                             head_offset * args.stride_doh;
        auto* dk_ptr = static_cast<KGradDataType*>(args.dk_ptr);
        auto* dv_ptr = static_cast<VGradDataType*>(args.dv_ptr);
        dk_ptr += batch_offset * args.stride_dkz + head_offset * args.stride_dkh;
        dv_ptr += batch_offset * args.stride_dvz + head_offset * args.stride_dvh;

        auto q_dram_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                q_ptr,
                ck_tile::make_tuple(args.N_Q, ck_tile::number<HeadDim>{}),
                ck_tile::make_tuple(args.stride_qm, args.stride_qd),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        auto k_dram_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                k_ptr,
                ck_tile::make_tuple(args.N_Q, ck_tile::number<HeadDim>{}),
                ck_tile::make_tuple(args.stride_kn, args.stride_kd),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        auto v_dram_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                v_ptr,
                ck_tile::make_tuple(args.N_Q, ck_tile::number<HeadDim>{}),
                ck_tile::make_tuple(args.stride_vn, args.stride_vd),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        auto do_dram_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                do_ptr,
                ck_tile::make_tuple(args.N_Q, ck_tile::number<HeadDim>{}),
                ck_tile::make_tuple(args.stride_dom, args.stride_dod),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        auto dk_dram_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                dk_ptr,
                ck_tile::make_tuple(args.N_Q, ck_tile::number<HeadDim>{}),
                ck_tile::make_tuple(args.stride_dkn, args.stride_dkd),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        auto dv_dram_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                dv_ptr,
                ck_tile::make_tuple(args.N_Q, ck_tile::number<HeadDim>{}),
                ck_tile::make_tuple(args.stride_dvn, args.stride_dvd),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        q_dram_view.init_raw();
        k_dram_view.init_raw();
        v_dram_view.init_raw();
        do_dram_view.init_raw();
        dk_dram_view.init_raw();
        dv_dram_view.init_raw();

        Pipeline{}(args,
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
};

} // namespace sla
} // namespace example
} // namespace ck_tile
