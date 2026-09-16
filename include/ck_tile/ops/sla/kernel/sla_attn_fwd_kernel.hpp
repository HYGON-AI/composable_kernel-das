// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.


#pragma once

#include "ck_tile/ops/sla/pipeline/sla_attn_fwd_pipeline.hpp"
#include "ck_tile/ops/sla/kernel/sla_attn_fwd_reduce_kernel.hpp"
#include "ck_tile/ops/sla/kernel/sla_sparse_lut_kernel.hpp"
#include "ck_tile/host.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>

struct SlaAttnFwdKernelArgument
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const int64_t* lut;
    const int32_t* stage_offsets;
    uint16_t* o;
    float* lse;
    int B;
    int H;
    int L;
    int D;
    int topk;
    int lut_q_blocks;
    int kv_stage_count;
    int stage_offset_stride;
    int64_t output_stage_stride;
    int64_t lse_stage_stride;
    float qk_scale_log2e;
};

template <int NTile,
          bool HasPadding,
          typename PipelinePolicy,
          int KvStageCount>
struct SlaAttnFwdKernel
{
    using Kargs = SlaAttnFwdKernelArgument;
    using Problem = typename PipelinePolicy::Problem;
    using Pipeline =
        SlaAttnFwdFusedQkSoftmaxPvPipeline<NTile, HasPadding, false, PipelinePolicy>;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        const int stage_count =
            KvStageCount == 0 ? arg.kv_stage_count : KvStageCount;
        return dim3(static_cast<unsigned int>(arg.B * arg.H * arg.lut_q_blocks),
                    static_cast<unsigned int>(stage_count));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(PipelinePolicy::kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int g             = blockIdx.x;
        const int kv_stage      = blockIdx.y;
        const int bh            = g / arg.lut_q_blocks;
        const int q_tile        = g - bh * arg.lut_q_blocks;
        const int lut_qb        = q_tile;
        const int start_m       = q_tile * Problem::kBlockM;
        const int64_t lut_row_index =
            static_cast<int64_t>(bh) * arg.lut_q_blocks + lut_qb;
        int active_begin = 0;
        int active_end   = arg.topk;
        if constexpr(KvStageCount != 1)
        {
            if(arg.kv_stage_count > 1)
            {
                const int32_t* row_offsets =
                    arg.stage_offsets + lut_row_index * arg.stage_offset_stride;
                active_begin = row_offsets[kv_stage];
                active_end   = row_offsets[kv_stage + 1];
            }
        }
        const int64_t* lut_row  =
            arg.lut + lut_row_index * arg.topk + active_begin;
        const uint16_t* q_bh = arg.q + static_cast<int64_t>(bh) * arg.L * arg.D;
        const uint16_t* k_bh = arg.k + static_cast<int64_t>(bh) * arg.L * arg.D;
        const uint16_t* v_bh = arg.v + static_cast<int64_t>(bh) * arg.L * arg.D;
        uint16_t* o_bh       =
            arg.o + static_cast<int64_t>(kv_stage) * arg.output_stage_stride +
            static_cast<int64_t>(bh) * arg.L * arg.D;
        float* lse_stage =
            arg.lse + static_cast<int64_t>(kv_stage) * arg.lse_stage_stride;

        auto q_dram =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                reinterpret_cast<const typename Problem::QDataType*>(q_bh),
                ck_tile::make_tuple(arg.L, ck_tile::number<NTile>{}),
                ck_tile::make_tuple(ck_tile::number<NTile>{}, ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        auto q_view = [&]() {
            if constexpr(HasPadding)
            {
                return ck_tile::pad_tensor_view(
                    q_dram,
                    ck_tile::make_tuple(ck_tile::number<Problem::kBlockM>{},
                                        ck_tile::number<64>{}),
                    ck_tile::sequence<true, false>{});
            }
            else
            {
                return q_dram;
            }
        }();

        auto k_dram =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                reinterpret_cast<const typename Problem::KDataType*>(k_bh),
                ck_tile::make_tuple(arg.L, ck_tile::number<NTile>{}),
                ck_tile::make_tuple(ck_tile::number<NTile>{}, ck_tile::number<1>{}),
                ck_tile::number<8>{},
                ck_tile::number<1>{});
        auto k_view = [&]() {
            if constexpr(HasPadding)
            {
                return ck_tile::pad_tensor_view(
                    k_dram,
                    ck_tile::make_tuple(ck_tile::number<32>{},
                                        ck_tile::number<64>{}),
                    ck_tile::sequence<true, false>{});
            }
            else
            {
                return k_dram;
            }
        }();

        auto v_dram =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                reinterpret_cast<const typename Problem::VDataType*>(v_bh),
                ck_tile::make_tuple(ck_tile::number<NTile>{}, arg.L),
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<NTile>{}),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
        auto v_view = [&]() {
            if constexpr(HasPadding)
            {
                return ck_tile::pad_tensor_view(
                    v_dram,
                    ck_tile::make_tuple(ck_tile::number<NTile>{},
                                        ck_tile::number<32>{}),
                    ck_tile::sequence<false, true>{});
            }
            else
            {
                return v_dram;
            }
        }();

        __shared__ typename Pipeline::SharedStorage scratch;
        Pipeline{}(q_view,
                   k_view,
                   v_view,
                   lut_row,
                   active_end - active_begin,
                   o_bh,
                   lse_stage,
                   bh,
                   q_tile,
                   0,
                   0,
                   start_m,
                   arg.L,
                   arg.L,
                   arg.L,
                   arg.D,
                   arg.D,
                   0,
                   0.0f,
                   arg.qk_scale_log2e,
                   arg.D,
                   1,
                   arg.L,
                   1,
                   scratch);

    }
};

template <typename PipelinePolicy,
          int NTile,
          bool HasPadding,
          int KvStageCount>
inline void launch_sla_attn_fwd(const SlaAttnFwdKernelArgument& arg, hipStream_t stream)
{
    static_assert(KvStageCount == 0 || KvStageCount == 1 || KvStageCount == 4,
                  "stage 0 is the runtime path; stage 1/4 are tuned paths");
    using Kernel =
        SlaAttnFwdKernel<NTile, HasPadding, PipelinePolicy, KvStageCount>;
    const auto grid    = Kernel::GridSize(arg);
    constexpr auto block = Kernel::BlockSize();
    ck_tile::stream_config stream_config{stream};
    ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<block.x, PipelinePolicy::kLaunchMinBlocks>(
            Kernel{}, grid, block, 0, arg));
}
