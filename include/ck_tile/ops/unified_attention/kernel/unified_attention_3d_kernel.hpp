// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/unified_attention/pipeline/unified_attention_3d_pipeline.hpp"

namespace ua {

template <ck_tile::index_t HeadDim>
CK_TILE_HOST_DEVICE constexpr auto make_head_distribution()
{
    return ck_tile::tile_distribution_encoding_pattern_2d<
        64,
        1,
        HeadDim,
        8,
        ck_tile::tile_distribution_pattern::thread_raked,
        1>::make_2d_static_tile_distribution();
}

template <ck_tile::index_t HeadDim, typename T>
CK_TILE_DEVICE auto load_head_tile(const T* ptr, int64_t stride)
{
    auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        ptr,
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<HeadDim>{}),
        ck_tile::make_tuple(stride, ck_tile::number<1>{}),
        ck_tile::number<4>{},
        ck_tile::number<1>{});
    auto window = ck_tile::make_tile_window(
        view,
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<HeadDim>{}),
        ck_tile::multi_index<2>{0, 0},
        make_head_distribution<HeadDim>());
    return ck_tile::load_tile(window);
}

template <ck_tile::index_t HeadDim, typename T, typename Tile>
CK_TILE_DEVICE void store_head_tile(T* ptr, int64_t stride, const Tile& tile)
{
    auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        ptr,
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<HeadDim>{}),
        ck_tile::make_tuple(stride, ck_tile::number<1>{}),
        ck_tile::number<4>{},
        ck_tile::number<1>{});
    auto window = ck_tile::make_tile_window(
        view,
        ck_tile::make_tuple(ck_tile::number<1>{}, ck_tile::number<HeadDim>{}),
        ck_tile::multi_index<2>{0, 0},
        make_head_distribution<HeadDim>());
    ck_tile::store_tile(window, tile);
}

template <typename DataType,
          ck_tile::index_t HeadDim,
          ck_tile::index_t PagedBlockSize = 0,
          bool SimpleDecode = false,
          bool FastAlibi = false>
struct CkTileAttention3DKernel
{
    // Decode packs all query heads sharing one KV head into the M dimension.
    // M=16 exactly covers the supported GQA/MQA ratio; waves split the N
    // dimension and the pipeline combines their row softmax values via LDS.
    using Problem  = UnifiedAttentionProblem<DataType,
                                             HeadDim,
                                             16,
                                             true,
                                             PagedBlockSize,
                                             SimpleDecode,
                                             FastAlibi>;
    using Pipeline = UnifiedAttentionPipeline<Problem>;
    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    template <typename Args>
    CK_TILE_DEVICE void operator()(Args a,
                                   DataType* seg_out,
                                   float* seg_max,
                                   float* seg_sum,
                                   int num_segments,
                                   int output_segments,
                                   int,
                                   int head_padded) const
    {
        const int global_block = static_cast<int>(blockIdx.x);
        const int kvhead       = static_cast<int>(blockIdx.y);
        const int segment_slot = static_cast<int>(blockIdx.z);
        int segment            = segment_slot;
        int tile_override      = -1;
        bool force_sink_segment = false;
        int seq = 0;
        int local_block = 0;
        if constexpr(SimpleDecode)
        {
            // The 3D binding requires one query token per sequence, hence the
            // global query block is the sequence index directly. The launch
            // grid already uses the exact sequence/head/segment counts.
            seq = global_block;

            // Sliding-window launches use compact output slots.  Map each
            // physical slot back to the logical segment used to partition KV.
            // The mapping is sequence-local, so mixed sequence lengths remain
            // correct without launching all logical segments.
            if(output_segments < num_segments)
            {
                const int seqlen = a.seq_lens[seq];
                const int tiles_per_segment =
                    ck_tile::max(1,
                                 (seqlen + num_segments * Problem::kN - 1) /
                                     (num_segments * Problem::kN));
                const int tile_begin =
                    a.sliding_window > 0
                        ? ck_tile::max(0, (seqlen - a.sliding_window) / Problem::kN)
                        : 0;
                const int tile_end = (seqlen + Problem::kN - 1) / Problem::kN;
                tile_override = tile_begin + segment_slot;
                segment = ck_tile::min(num_segments - 1,
                                       tile_override / tiles_per_segment);
            }
            // Keep the sink in the first materialized compact partial. This
            // preserves the same running-softmax state as the ordinary
            // segment-0 path, including when ALiBi makes the sink weight
            // significant relative to the visible KV tail.
            if(output_segments < num_segments)
            {
                force_sink_segment = a.use_sinks && segment_slot == 0;
                a.use_sinks = force_sink_segment;
            }
        }
        else
        {
            if(kvhead >= a.num_kv_heads || segment >= num_segments)
                return;
            if(!map_query_block(a, global_block, Problem::kM, seq, local_block))
                return;
        }

        __shared__ char smem[Problem::kSmemSize];
        auto result =
            Pipeline{}(a,
                       seq,
                       local_block,
                       kvhead,
                       segment,
                       num_segments,
                       smem,
                       tile_override,
                       force_sink_segment);
        auto o_acc       = result.at(ck_tile::number<0>{});
        const auto m     = result.at(ck_tile::number<1>{});
        const auto l     = result.at(ck_tile::number<2>{});
        const bool has_tiles = result.at(ck_tile::number<3>{});

        const int token = a.cu_q[seq];
        int numq_per_kv;
        if constexpr(SimpleDecode)
            numq_per_kv = a.numq_per_kv;
        else
            numq_per_kv = a.num_q_heads / a.num_kv_heads;
        const int qhead_base = kvhead * numq_per_kv;
        // An M16N16 MMAC leaves four FP32 values per lane in a jump layout:
        // lane group g owns columns {g, g+4, g+8, g+12}. Transpose the 4x4
        // lane-group/register matrix with two butterfly shuffles so each lane
        // owns four contiguous columns. This is the CK Tile cshuffle pattern,
        // performed in registers, and avoids an LDS round trip plus barriers.
        constexpr int kStoreVec = 4;
        using StoreVec = ck_tile::ext_vector_t<DataType, kStoreVec>;
        constexpr int kWarpSize = ck_tile::get_warp_size();
        constexpr int kNumWarps = Problem::kBlockSize / kWarpSize;
        constexpr int kOutputVecsPerThread =
            decltype(o_acc)::get_thread_buffer_size() / kStoreVec;
        static_assert(decltype(o_acc)::get_thread_buffer_size() % kStoreVec == 0);

        const int lane  = ck_tile::get_lane_id();
        const int warp  = ck_tile::get_warp_id();

        // Sliding-window pruning leaves most parallel-softmax segments empty
        // for decode. Their accumulator is known to be zero, so write it
        // directly instead of running the MMAC jump-layout cshuffle on every
        // inactive block. Segment 0 still preserves the optional sink state.
        if(!has_tiles)
        {
            const StoreVec zero{};
            const int dim = lane * kStoreVec;
            for(int empty_row = warp; empty_row < numq_per_kv;
                empty_row += kNumWarps)
            {
                const int empty_qhead = qhead_base + empty_row;
                const int64_t empty_base =
                    (static_cast<int64_t>(token) * a.num_q_heads + empty_qhead) *
                        output_segments +
                    segment_slot;
                if(dim < HeadDim)
                {
                    *reinterpret_cast<StoreVec*>(
                        seg_out + empty_base * head_padded + dim) = zero;
                }
            }
            if(warp == 0 && lane < numq_per_kv)
            {
                const int empty_qhead = qhead_base + lane;
                const int64_t empty_base =
                    (static_cast<int64_t>(token) * a.num_q_heads + empty_qhead) *
                        output_segments +
                    segment_slot;
                const bool sink_segment =
                    a.use_sinks && (segment == 0 || force_sink_segment);
                seg_max[empty_base] =
                    sink_segment
                        ? a.sinks[empty_qhead]
                        : -ck_tile::numeric<float>::infinity();
                seg_sum[empty_base] = sink_segment ? 1.0f : 0.0f;
            }
            return;
        }

        const int row   = lane & 15;
        const int group = lane >> 4;
        const int qhead = qhead_base + row;
        const int64_t base =
            (static_cast<int64_t>(token) * a.num_q_heads + qhead) * output_segments +
            segment_slot;

        ck_tile::static_for<0, kOutputVecsPerThread, 1>{}([&](auto i_vec) {
            ck_tile::thread_buffer<float, kStoreVec> x0;
            ck_tile::thread_buffer<float, kStoreVec> x1;
            ck_tile::thread_buffer<float, kStoreVec> x2;
            ck_tile::static_for<0, kStoreVec, 1>{}([&](auto j) {
                x0(j) = o_acc.get_thread_buffer()(i_vec * kStoreVec + j);
            });

            // Exchange bit 0, then bit 1, between the lane-group coordinate
            // and the register coordinate. The result is x2[j] = old[j][g].
            ck_tile::static_for<0, kStoreVec, 1>{}([&](auto j) {
                const float remote =
                    ck_tile::warp_shuffle(x0(j ^ ck_tile::number<1>{}), lane ^ 16);
                x1(j) = ((group ^ j) & 1) != 0 ? remote : x0(j);
            });
            ck_tile::static_for<0, kStoreVec, 1>{}([&](auto j) {
                const float remote =
                    ck_tile::warp_shuffle(x1(j ^ ck_tile::number<2>{}), lane ^ 32);
                x2(j) = ((group ^ j) & 2) != 0 ? remote : x1(j);
            });

            const int dim = i_vec * (kNumWarps * 16) + warp * 16 + group * 4;
            if(row < numq_per_kv)
            {
                ck_tile::thread_buffer<DataType, kStoreVec> converted;
                ck_tile::static_for<0, kStoreVec, 1>{}([&](auto j) {
                    converted(j) = ck_tile::type_convert<DataType>(x2(j));
                });
                *reinterpret_cast<StoreVec*>(seg_out + base * head_padded + dim) =
                    converted.template get_as<StoreVec>()(ck_tile::number<0>{});
            }
        });

        // M/L are replicated across the four N-waves. Wave 0 owns the scalar
        // epilogue; allowing every wave to write the same addresses generated
        // four redundant global stores per query head.
        if(ck_tile::get_warp_id() == 0)
        {
            ck_tile::sweep_tile(m, [&](auto idx) {
                const auto x = ck_tile::get_x_indices_from_distributed_indices(
                    m.get_tile_distribution(), idx);
                const int row = x[ck_tile::number<0>{}];
                if(row < numq_per_kv)
                {
                    const int qhead = qhead_base + row;
                    const int64_t base =
                        (static_cast<int64_t>(token) * a.num_q_heads + qhead) *
                            output_segments +
                        segment_slot;
                    seg_max[base] = m[idx];
                    seg_sum[base] = l[idx];
                }
            });
        }
    }
};

template <typename DataType, ck_tile::index_t HeadDim, ck_tile::index_t StaticSegments>
struct CkTileReduceSegmentsSerialKernel
{
    static_assert(StaticSegments > 0 && StaticSegments <= 4);
    static constexpr ck_tile::index_t kBlockSize = 64;

    CK_TILE_DEVICE void operator()(DataType* out,
                                   const DataType* seg_out,
                                   const float* seg_max,
                                   const float* seg_sum,
                                   int num_tokens,
                                   int num_heads,
                                   int head_padded,
                                   int num_segments,
                                   int64_t o_s0,
                                   int64_t o_s1,
                                   const float* sinks) const
    {
        const int token = static_cast<int>(blockIdx.x);
        const int head  = static_cast<int>(blockIdx.y);
        if(token >= num_tokens || head >= num_heads)
            return;
        const int64_t base =
            (static_cast<int64_t>(token) * num_heads + head) * num_segments;

        auto acc = ck_tile::make_static_distributed_tensor<float>(
            make_head_distribution<HeadDim>());
        ck_tile::clear_tile(acc);
        ck_tile::thread_buffer<float, StaticSegments> sums;
        ck_tile::thread_buffer<float, StaticSegments> maxima;
        ck_tile::static_for<0, StaticSegments, 1>{}([&](auto s) {
            sums(s)   = seg_sum[base + s];
            maxima(s) = seg_max[base + s];
        });

        float global_m = sinks != nullptr
                             ? sinks[head]
                             : -ck_tile::numeric<float>::infinity();
        ck_tile::static_for<0, StaticSegments, 1>{}([&](auto s) {
            if(sums(s) > 0.0f)
                global_m = ck_tile::max(global_m, maxima(s));
        });

        float global_l = sinks != nullptr ? expf(sinks[head] - global_m) : 0.0f;
        ck_tile::static_for<0, StaticSegments, 1>{}([&](auto s) {
            if(sums(s) > 0.0f)
            {
                const float weight = expf(maxima(s) - global_m);
                const auto partial = load_head_tile<HeadDim>(
                    seg_out + (base + s) * head_padded, head_padded);
                ck_tile::tile_elementwise_inout(
                    [=](auto& x, auto y) {
                        x += weight * ck_tile::type_convert<float>(y);
                    },
                    acc,
                    partial);
                global_l += sums(s) * weight;
            }
        });
        ck_tile::tile_elementwise_inout(
            [=](auto& x) { x = global_l == 0.0f ? 0.0f : x / global_l; }, acc);
        const auto result = ck_tile::cast_tile<DataType>(acc);
        store_head_tile<HeadDim>(out + static_cast<int64_t>(token) * o_s0 +
                                    static_cast<int64_t>(head) * o_s1,
                                o_s1,
                                result);
    }
};

template <typename DataType, ck_tile::index_t HeadDim, ck_tile::index_t StaticSegments = 0>
struct CkTileReduceSegmentsKernel
{
    // A 64-segment full-context decode gives four 256-thread waves sixteen
    // serial partials each. Eight waves halve that dependency chain while the
    // extra wave-partial LDS still stays below 8 KiB for D192/D256.
    static constexpr ck_tile::index_t kBlockSize = 512;
    static constexpr int kWarpSize               = ck_tile::get_warp_size();
    static constexpr int kNumWarps               = kBlockSize / kWarpSize;
    static constexpr int kMaxSegments            = 64;
    static constexpr int kHeadLdsStride          = HeadDim + 1;

    CK_TILE_DEVICE void operator()(DataType* out,
                                   const DataType* seg_out,
                                   const float* seg_max,
                                   const float* seg_sum,
                                   int num_tokens,
                                   int num_heads,
                                   int head_padded,
                                   int num_segments,
                                   int64_t o_s0,
                                   int64_t o_s1,
                                   const float* sinks) const
    {
        const int token = static_cast<int>(blockIdx.x);
        const int head = static_cast<int>(blockIdx.y);
        if(token >= num_tokens || head >= num_heads)
            return;
        if(num_segments > kMaxSegments)
            return;

        const int tid  = static_cast<int>(threadIdx.x);
        const int lane = tid & (kWarpSize - 1);
        const int warp = tid / kWarpSize;
        const int64_t base =
            (static_cast<int64_t>(token) * num_heads + head) * num_segments;

        // Wave 0 reduces the metadata vector in registers and publishes one
        // weight per segment. The old reducer made all 64 lanes reload the
        // metadata and repeat every expf; this computes each weight once.
        __shared__ float weights[kMaxSegments];
        __shared__ float inv_l;
        __shared__ float wave_partial[kNumWarps][kHeadLdsStride];

        if(warp == 0)
        {
            const bool active = lane < num_segments;
            const float sum   = active ? seg_sum[base + lane] : 0.0f;
            float local_m = sum > 0.0f
                                ? seg_max[base + lane]
                                : -ck_tile::numeric<float>::infinity();
            if(lane == 0 && sinks != nullptr)
                local_m = ck_tile::max(local_m, sinks[head]);
#pragma unroll
            for(int delta = kWarpSize / 2; delta > 0; delta /= 2)
                local_m = ck_tile::max(
                    local_m, ck_tile::warp_shuffle_down(local_m, delta));
            const float global_m = ck_tile::warp_shuffle(local_m, 0);

            const float weight = sum > 0.0f
                                     ? expf(seg_max[base + lane] - global_m)
                                     : 0.0f;
            if(active)
                weights[lane] = weight;
            float local_l = sum * weight;
            if(lane == 0 && sinks != nullptr)
                local_l += expf(sinks[head] - global_m);
#pragma unroll
            for(int delta = kWarpSize / 2; delta > 0; delta /= 2)
                local_l += ck_tile::warp_shuffle_down(local_l, delta);
            if(lane == 0)
                inv_l = local_l == 0.0f ? 0.0f : 1.0f / local_l;
        }
        __syncthreads();

        // Eight waves split the segment dimension. Lanes own D positions in
        // 64-element stripes, so every load instruction remains coalesced.
        constexpr int kDimsPerLane = (HeadDim + kWarpSize - 1) / kWarpSize;
        float acc[kDimsPerLane]{};
        for(int s = warp; s < num_segments; s += kNumWarps)
        {
            const float weight = weights[s];
            if(weight == 0.0f)
                continue;
            const DataType* partial = seg_out + (base + s) * head_padded;
            ck_tile::static_for<0, kDimsPerLane, 1>{}([&](auto i) {
                const int dim = lane + i * kWarpSize;
                if(dim < HeadDim)
                    acc[i] += weight * ck_tile::type_convert<float>(partial[dim]);
            });
        }
        ck_tile::static_for<0, kDimsPerLane, 1>{}([&](auto i) {
            const int dim = lane + i * kWarpSize;
            if(dim < HeadDim)
                wave_partial[warp][dim] = acc[i];
        });
        __syncthreads();

        // The first HeadDim threads merge the four wave-private partials.
        // A +1 row pad prevents identical columns in adjacent wave rows from
        // repeatedly mapping onto the same LDS bank.
        if(tid < HeadDim)
        {
            float value = 0.0f;
#pragma unroll
            for(int w = 0; w < kNumWarps; ++w)
                value += wave_partial[w][tid];
            out[static_cast<int64_t>(token) * o_s0 +
                static_cast<int64_t>(head) * o_s1 + tid] =
                ck_tile::type_convert<DataType>(value * inv_l);
        }
    }
};

} // namespace ua
