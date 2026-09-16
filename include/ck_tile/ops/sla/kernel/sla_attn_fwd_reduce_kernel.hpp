// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_attn_fwd_pipeline.hpp"
#include "ck_tile/host.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>

struct SlaAttnFwdKvStageReduceArgument
{
    const uint16_t* partial_o;
    const float* partial_lse;
    uint16_t* o;
    float* lse;
    int B;
    int H;
    int L;
    int D;
    int stage_count;
    int64_t output_stage_stride;
    int64_t lse_stage_stride;
};

template <typename PipelinePolicy, int KvStageCount>
struct SlaAttnFwdKvStageReducePipeline
{
    using Kargs = SlaAttnFwdKvStageReduceArgument;
    using DataType = typename PipelinePolicy::Problem::QDataType;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        const int64_t rows = static_cast<int64_t>(arg.B) * arg.H * arg.L;
        return dim3(static_cast<unsigned int>((rows + 7) / 8));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(PipelinePolicy::kBlockSize);
    }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int kMaxRuntimeStageCount = 16;
        constexpr int kStageCapacity =
            KvStageCount == 0 ? kMaxRuntimeStageCount : KvStageCount;
        const int stage_count =
            KvStageCount == 0 ? arg.stage_count : KvStageCount;
        const int tid = static_cast<int>(threadIdx.x);
        const int row_group = tid >> 5;
        const int d_lane = tid & 31;
        const int64_t row =
            static_cast<int64_t>(blockIdx.x) * 8 + row_group;
        const int64_t total_rows =
            static_cast<int64_t>(arg.B) * arg.H * arg.L;
        const bool row_valid = row_group < 8 && row < total_rows;

        if(!row_valid)
        {
            return;
        }

        float stage_weights[kStageCapacity];

        if(d_lane == 0)
        {
            float stage_lse_regs[kStageCapacity];
            float m = -ck_tile::numeric<float>::infinity();
#pragma unroll
            for(int stage = 0; stage < stage_count; ++stage)
            {
                const float stage_lse =
                    arg.partial_lse[
                        static_cast<int64_t>(stage) * arg.lse_stage_stride +
                        row];
                stage_lse_regs[stage] = stage_lse;
                m = ck_tile::max(m, stage_lse);
            }

            float denom = 0.0f;
#pragma unroll
            for(int stage = 0; stage < stage_count; ++stage)
            {
                const float stage_lse = stage_lse_regs[stage];
                const float weight =
                    stage_lse == -ck_tile::numeric<float>::infinity()
                        ? 0.0f
                        : sla_attn_fwd_fast_exp2(stage_lse - m);
                stage_weights[stage] = weight;
                denom += weight;
            }

            const float inv_denom =
                denom > 1.0e-10f ? 1.0f / denom : 0.0f;
#pragma unroll
            for(int stage = 0; stage < stage_count; ++stage)
            {
                stage_weights[stage] *= inv_denom;
            }
            arg.lse[row] =
                denom > 1.0e-10f
                    ? m + log2f(denom)
                    : -ck_tile::numeric<float>::infinity();
        }

        const int src_lane = (tid & ~31);
#pragma unroll
        for(int stage = 0; stage < stage_count; ++stage)
        {
            stage_weights[stage] = __shfl(stage_weights[stage], src_lane, 64);
        }

        union Bf16x4
        {
            uint2 packed;
            DataType values[4];
        };

        if(arg.D == 128)
        {
            const int64_t vector_offset =
                (row * static_cast<int64_t>(arg.D)) / 4 + d_lane;
            float acc[4] = {};
#pragma unroll
            for(int stage = 0; stage < stage_count; ++stage)
            {
                Bf16x4 stage_o;
                stage_o.packed =
                    reinterpret_cast<const uint2*>(arg.partial_o)[
                        static_cast<int64_t>(stage) *
                            (arg.output_stage_stride / 4) +
                        vector_offset];
#pragma unroll
                for(int i = 0; i < 4; ++i)
                {
                    acc[i] += ck_tile::type_convert<float>(
                                  stage_o.values[i]) *
                              stage_weights[stage];
                }
            }
            Bf16x4 result;
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                result.values[i] = ck_tile::type_convert<DataType>(acc[i]);
            }
            reinterpret_cast<uint2*>(arg.o)[vector_offset] =
                result.packed;
        }
        else
        {
            for(int d = d_lane; d < arg.D; d += 32)
            {
                const int64_t output_offset = row * arg.D + d;
                float acc = 0.0f;
                for(int stage = 0; stage < stage_count; ++stage)
                {
                    const auto stage_o = ck_tile::bit_cast<DataType>(
                        arg.partial_o[
                            static_cast<int64_t>(stage) *
                                arg.output_stage_stride +
                            output_offset]);
                    acc += ck_tile::type_convert<float>(stage_o) *
                           stage_weights[stage];
                }
                arg.o[output_offset] = ck_tile::bit_cast<uint16_t>(
                    ck_tile::type_convert<DataType>(acc));
            }
        }
    }
};

template <typename PipelinePolicy, int KvStageCount>
struct SlaAttnFwdKvStageReduceKernel
{
    using Pipeline = SlaAttnFwdKvStageReducePipeline<PipelinePolicy, KvStageCount>;
    using Kargs    = typename Pipeline::Kargs;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg)
    {
        return Pipeline::GridSize(arg);
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return Pipeline::BlockSize(); }
    CK_TILE_DEVICE void operator()(const Kargs& arg) const { Pipeline{}(arg); }
};

template <typename PipelinePolicy, int KvStageCount>
inline void launch_sla_attn_fwd_kv_stage_reduce(
    const SlaAttnFwdKvStageReduceArgument& arg,
    hipStream_t stream)
{
    static_assert(KvStageCount == 0 || KvStageCount == 4,
                  "stage 0 is the runtime path; stage 4 is the tuned path");
    using Kernel =
        SlaAttnFwdKvStageReduceKernel<PipelinePolicy, KvStageCount>;
    const auto grid = Kernel::GridSize(arg);
    constexpr auto block = Kernel::BlockSize();
    ck_tile::stream_config stream_config{stream};
    ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<block.x, 1>(Kernel{}, grid, block, 0, arg));
}
