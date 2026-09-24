// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/gdn/gdn_numeric.hpp"

#include "ck_tile/ops/gdn/pipeline/cp/chunk_delta_h_wave_reg_pipeline.hpp"
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_wave_reg_pipeline.hpp"

namespace ck_tile {

CK_TILE_DEVICE
void get_swizzled_block_idx(index_t& block_x, index_t& block_y)
{
    const index_t gx = gridDim.x;
    const index_t gy = gridDim.y;
    if (gx <= 1) {
        block_x = blockIdx.x;
        block_y = blockIdx.y;
        return;
    }
    constexpr index_t SWIZZLE_SIZE = 4;
    const index_t flat_bid = blockIdx.x + blockIdx.y * gx;
    const index_t full_y = (gy / SWIZZLE_SIZE) * SWIZZLE_SIZE;
    const index_t full_blocks = full_y * gx;
    if (flat_bid >= full_blocks) {
        const index_t tail_bid = flat_bid - full_blocks;
        block_x = tail_bid % gx;
        block_y = full_y + tail_bid / gx;
        return;
    }
    const index_t group_span = gx * SWIZZLE_SIZE;
    const index_t group_idx = flat_bid / group_span;
    const index_t in_group_bid = flat_bid % group_span;

    const index_t wy = group_idx * SWIZZLE_SIZE + (in_group_bid % SWIZZLE_SIZE);
    const index_t wx = in_group_bid / SWIZZLE_SIZE;
    block_x = wx;
    block_y = wy;
}

template <typename DataType>
struct ChunkDeltaHGroupedScanKargs
{
    ChunkDeltaHScanFwdKargs<DataType> scan;
    index_t num_chunks;
    index_t num_groups;
};

template <typename GroupedKargs>
CK_TILE_DEVICE void get_group_chunk_range(const GroupedKargs& args,
                                          index_t group,
                                          index_t& chunk_base,
                                          index_t& local_chunks)
{
    const index_t base = args.num_chunks / args.num_groups;
    const index_t extra = args.num_chunks - base * args.num_groups;
    local_chunks = base + (group < extra ? 1 : 0);
    chunk_base = group * base + min(group, extra);
}

template <typename Traits, typename Policy>
struct ChunkDeltaHGroupFusedSummaryKernel
{
    using DataType = typename Traits::DataType;
    using GroupedKargs = ChunkDeltaHGroupedScanKargs<DataType>;
    using Pipeline = ChunkDeltaHCpSummaryPipeline<Traits,
                                                Policy,
                                                false,
                                                false,
                                                false,
                                                false,
                                                false,
                                                true,
                                                false,
                                                false,
                                                true>;

    CK_TILE_DEVICE void operator()(GroupedKargs grouped_args) const
    {
        const auto& args = grouped_args.scan;
        index_t block_x, block_y;
        get_swizzled_block_idx(block_x, block_y);
        const index_t v_begin = block_x * 64;
        const index_t group_vh = block_y;
        const index_t group = group_vh / args.num_value_heads;
        const index_t vh = group_vh - group * args.num_value_heads;
        const index_t qh = args.num_qk_heads == args.num_value_heads
            ? vh
            : vh / (args.num_value_heads / args.num_qk_heads);

        index_t chunk_base, local_chunks;
        get_group_chunk_range(
            grouped_args, group, chunk_base, local_chunks);
        const index_t bos = chunk_base * Policy::kChunkSize;
        const index_t eos = ck_tile::min(
            args.total_tokens, bos + local_chunks * Policy::kChunkSize);

        __shared__ typename Pipeline::LdsStorage smem;
        Pipeline{}(args,
                   group,
                   vh,
                   qh,
                   v_begin,
                   bos,
                   eos,
                   chunk_base,
                   local_chunks,
                   smem);
    }
};

// The first logical group has a real initial state and can therefore be
// replayed while the independent affine summaries for the middle groups are
// produced.  This deliberately keeps the original, non-compact group indexing:
// slot 0 replays logical group 0, while slots 1..G-2 summarize the matching
// logical groups.
template <typename DataType>
struct ChunkDeltaHFirstReplayTailSummaryKargs
{
    ChunkDeltaHGroupedScanKargs<DataType> grouped;
    const float* first_initial_state;
    DataType* first_h_start;
    DataType* first_v_new;
    float* first_end_state;
    bool first_has_initial_state;
};

template <typename Traits,
          typename SummaryPolicy,
          typename ReplayPolicy,
          bool PreshuffledH = false>
struct ChunkDeltaHFirstReplayTailSummaryKernel
{
    using DataType = typename Traits::DataType;
    using HybridKargs =
        ChunkDeltaHFirstReplayTailSummaryKargs<DataType>;
    using SummaryPipeline = ChunkDeltaHCpSummaryPipeline<Traits,
                                                         SummaryPolicy,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         true,
                                                         false,
                                                         false,
                                                         true>;
    using ReplayPipeline =
        ChunkDeltaHWaveRegPipeline<Traits, ReplayPolicy, PreshuffledH, true>;

    static_assert(SummaryPolicy::kBlockSize == ReplayPolicy::kBlockSize);
    static_assert(SummaryPolicy::kVTile == 128);
    static_assert(ReplayPolicy::kVTile == 64);

    union LdsStorage
    {
        typename SummaryPipeline::LdsStorage summary;
        typename ReplayPipeline::LdsStorage replay;
    };

    CK_TILE_DEVICE void operator()(HybridKargs hybrid_args) const
    {
        const auto& grouped_args = hybrid_args.grouped;
        auto args = grouped_args.scan;
        index_t block_x, block_y;
        get_swizzled_block_idx(block_x, block_y);

        const index_t slot_vh = block_y;
        const index_t slot = slot_vh / args.num_value_heads;
        const index_t vh = slot_vh - slot * args.num_value_heads;
        const index_t qh = args.num_qk_heads == args.num_value_heads
            ? vh
            : vh / (args.num_value_heads / args.num_qk_heads);

        // In the non-compact bring-up, the launch slot is also the logical
        // group: slot 0 is the actual replay and slot n>0 is summary n.
        const index_t logical_group = slot;
        index_t chunk_base, local_chunks;
        get_group_chunk_range(
            grouped_args, logical_group, chunk_base, local_chunks);
        const index_t bos = chunk_base * SummaryPolicy::kChunkSize;
        const index_t eos = ck_tile::min(
            args.total_tokens,
            bos + local_chunks * SummaryPolicy::kChunkSize);

        __shared__ LdsStorage smem;
        if(slot == 0)
        {
            args.initial_state = hybrid_args.first_initial_state;
            args.has_initial_state = hybrid_args.first_has_initial_state;
            args.h_start = hybrid_args.first_h_start;
            args.v_new = hybrid_args.first_v_new;
            args.final_state = hybrid_args.first_end_state;
            args.store_final_state = true;
            args.save_new_value = true;
            ReplayPipeline{}(args,
                             0,
                             vh,
                             qh,
                             block_x * ReplayPolicy::kVTile,
                             bos,
                             eos,
                             chunk_base,
                             local_chunks,
                             smem.replay);
        }
        else
        {
            SummaryPipeline{}(args,
                              logical_group,
                              vh,
                              qh,
                              block_x * 64,
                              bos,
                              eos,
                              chunk_base,
                              local_chunks,
                              smem.summary);
        }
    }
};

template <typename Traits,
          typename Policy,
          bool PreshuffledH = false,
          index_t GroupOffset = 0>
struct ChunkDeltaHGroupReplayKernel
{
    using DataType = typename Traits::DataType;
    using GroupedKargs = ChunkDeltaHGroupedScanKargs<DataType>;
    using Pipeline =
        ChunkDeltaHWaveRegPipeline<Traits, Policy, PreshuffledH, true>;

    CK_TILE_DEVICE void operator()(GroupedKargs grouped_args) const
    {
        auto args = grouped_args.scan;
        index_t block_x, block_y;
        get_swizzled_block_idx(block_x, block_y);
        const index_t v_begin = block_x * Policy::kVTile;
        const index_t group_vh = block_y;
        const index_t replay_group = group_vh / args.num_value_heads;
        const index_t group = replay_group + GroupOffset;
        const index_t vh =
            group_vh - replay_group * args.num_value_heads;
        const index_t qh = args.num_qk_heads == args.num_value_heads
            ? vh
            : vh / (args.num_value_heads / args.num_qk_heads);

        index_t chunk_base, local_chunks;
        get_group_chunk_range(
            grouped_args, group, chunk_base, local_chunks);
        const index_t bos = chunk_base * Policy::kChunkSize;
        const index_t eos = ck_tile::min(
            args.total_tokens, bos + local_chunks * Policy::kChunkSize);

        args.has_initial_state = true;
        args.store_final_state =
            args.final_state != nullptr && group + 1 == grouped_args.num_groups;

        __shared__ typename Pipeline::LdsStorage smem;
        Pipeline{}(args,
                   group,
                   vh,
                   qh,
                   v_begin,
                   bos,
                   eos,
                   chunk_base,
                   local_chunks,
                   smem);
    }
};

template <typename DataType,
          index_t VTile,
          index_t BlockSize,
          bool StoreInitialStart = true>
__global__ __launch_bounds__(BlockSize) void chunk_delta_h_group_prefix_kernel(
    const DataType* group_a,
    const DataType* group_b,
    const float* initial_state,
    float* group_start,
    float* final_state,
    index_t num_groups,
    index_t num_value_heads,
    index_t value_dim,
    bool has_initial_state,
    bool store_final_state)
{
    constexpr index_t kHeadDim = 128;
    constexpr index_t kPad = 8;
    constexpr index_t kBlockSize = BlockSize;
    // Eight M-direction waves split the 128 output rows evenly.  LDS limits
    // this kernel to one workgroup per CU, so the wider workgroup uses the
    // otherwise idle second wave slot on each SIMD.
    static_assert(kBlockSize == 512);
    constexpr index_t kStatePerThread = kHeadDim * VTile / kBlockSize;
    constexpr auto gemm =
        GdnCpPrefixMmacBlockGemm<DataType, VTile, kBlockSize>{};

    struct AffineAndOutput {
        DataType a[kHeadDim * (kHeadDim + kPad)];
        float output[kHeadDim * (VTile + kPad)];
    };
    __shared__ AffineAndOutput storage;
    __shared__ DataType state_t[VTile * (kHeadDim + kPad)];

    index_t block_x, block_y;
    get_swizzled_block_idx(block_x, block_y);
    const index_t v_begin = block_x * VTile;
    const index_t vh = block_y;
    const index_t vv = threadIdx.x % VTile;
    const index_t row_stride = kBlockSize / VTile;
    const index_t row_base = threadIdx.x / VTile;
    float state[kStatePerThread];

    #pragma unroll
    for (index_t i = 0; i < kStatePerThread; ++i) {
        const index_t row = row_base + i * row_stride;
        float value = 0.0f;
        if (has_initial_state && v_begin + vv < value_dim) {
            value = initial_state[
                (static_cast<long_index_t>(vh) * kHeadDim + row) * value_dim
                + v_begin + vv];
        }
        state[i] = value;
    }

    for (index_t group = 0; group < num_groups; ++group) {
        const long_index_t base =
            (static_cast<long_index_t>(group) * num_value_heads + vh)
            * kHeadDim * value_dim;

        if constexpr(StoreInitialStart) {
            #pragma unroll
            for (index_t i = 0; i < kStatePerThread; ++i) {
                const index_t row = row_base + i * row_stride;
                group_start[base + static_cast<long_index_t>(row) * value_dim
                            + v_begin + vv] = state[i];
            }
        } else if(group != 0) {
            #pragma unroll
            for (index_t i = 0; i < kStatePerThread; ++i) {
                const index_t row = row_base + i * row_stride;
                group_start[base + static_cast<long_index_t>(row) * value_dim
                            + v_begin + vv] = state[i];
            }
        }

        #pragma unroll
        for (index_t i = 0; i < kStatePerThread; ++i) {
            const index_t row = row_base + i * row_stride;
            state_t[vv * (kHeadDim + kPad) + row] = gdn_type_convert<DataType>(state[i]);
        }

        auto a_dram = make_naive_tensor_view<address_space_enum::global>(
            group_a + base,
            make_tuple(number<kHeadDim>{}, number<kHeadDim>{}),
            make_tuple(number<kHeadDim>{}, number<1>{}),
            number<8>{},
            number<1>{});
        auto a_dram_window = make_tile_window(
            a_dram,
            make_tuple(number<kHeadDim>{}, number<kHeadDim>{}),
            multi_index<2>{0, 0},
            MakeGdnCpAsyncDramDistribution<kHeadDim, kHeadDim, 2, 8>());
        a_dram_window.init_raw();
        auto a_async_lds = make_tensor_view<address_space_enum::lds>(
            storage.a,
            MakeGdnAsyncLdsStoreDescriptor<
                kHeadDim, kHeadDim, 2, kPad, 8>());
        auto a_async_lds_window = make_tile_window(
            a_async_lds,
            MakeGdnAsyncLdsStoreDescriptor<
                kHeadDim, kHeadDim, 2, kPad, 8>().get_lengths(),
            multi_index<3>{0, 0, 0});
        async_load_tile_raw(a_async_lds_window,
                            a_dram_window,
                            bool_constant<true>{},
                            bool_constant<false>{});
        // Raw global-to-LDS copies are asynchronous. A workgroup barrier alone
        // does not wait for those VMEM writes before the prefix GEMM reads A.
        async_load_fence(0);
        block_sync_lds();

        auto a_view = make_tensor_view<address_space_enum::lds>(
            storage.a, MakeGdnSimpleLdsDescriptor<kHeadDim, kHeadDim, kPad>());
        auto state_view = make_tensor_view<address_space_enum::lds>(
            state_t, MakeGdnSimpleLdsDescriptor<VTile, kHeadDim, kPad>());
        auto a_window = make_tile_window(
            a_view,
            make_tuple(number<kHeadDim>{}, number<kHeadDim>{}),
            multi_index<2>{0, 0});
        auto state_window = make_tile_window(
            state_view,
            make_tuple(number<VTile>{}, number<kHeadDim>{}),
            multi_index<2>{0, 0});
        auto acc = decltype(gemm.MakeCBlockTile()){};
        clear_tile(acc);
        gemm(acc, a_window, state_window);
        const auto out = gemm.MakeOuputLayout(acc);
        constexpr auto spans = decltype(out)::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto tile_idx = make_tuple(idx0, idx1);
                const auto x_idx = get_x_indices_from_distributed_indices(
                    out.get_tile_distribution(), tile_idx);
                const index_t row = x_idx.at(number<0>{});
                const index_t col = x_idx.at(number<1>{});
                storage.output[row * (VTile + kPad) + col] =
                    out[tile_idx]
                    + type_convert<float>(group_b[
                        base + static_cast<long_index_t>(row) * value_dim
                        + v_begin + col]);
            });
        });
        __syncthreads();

        #pragma unroll
        for (index_t i = 0; i < kStatePerThread; ++i) {
            const index_t row = row_base + i * row_stride;
            state[i] = storage.output[row * (VTile + kPad) + vv];
        }
        __syncthreads();
    }

    const long_index_t last_start_base =
        (static_cast<long_index_t>(num_groups) * num_value_heads + vh)
        * kHeadDim * value_dim;
    #pragma unroll
    for (index_t i = 0; i < kStatePerThread; ++i) {
        const index_t row = row_base + i * row_stride;
        group_start[last_start_base + static_cast<long_index_t>(row) * value_dim
                    + v_begin + vv] = state[i];
    }

    if (store_final_state) {
        #pragma unroll
        for (index_t i = 0; i < kStatePerThread; ++i) {
            const index_t row = row_base + i * row_stride;
            final_state[(static_cast<long_index_t>(vh) * kHeadDim + row) * value_dim
                        + v_begin + vv] = state[i];
        }
    }
}

} // namespace ck_tile
