// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

#include <cstddef>
#include <cstdint>

namespace ck_tile {
namespace example {
namespace sla {

struct SlaAttnBwdDqKvPartitionKargs
{
    const int64_t* lut;
    int64_t* partitioned_lut;
    int64_t row_count;
    int topk;
    int kv_block_count;
    int stage_count;
};

template <int MaxKvStageCount>
struct SlaAttnBwdDqKvPartitionFastPipeline
{
    using Kargs = SlaAttnBwdDqKvPartitionKargs;
    static constexpr int kBucketsPerStage = 8;
    static constexpr int kMaxBucketCount =
        MaxKvStageCount * kBucketsPerStage;

    CK_TILE_DEVICE void operator()(const Kargs& arg) const
    {
        const int64_t row = ck_tile::get_block_1d_id();
        if(row >= arg.row_count)
        {
            return;
        }

        const int tid = ck_tile::get_thread_local_1d_id();
        const int64_t* input_row = arg.lut + row * arg.topk;
        int64_t* output_row      = arg.partitioned_lut + row * arg.topk;
        const int bucket_count = arg.stage_count * kBucketsPerStage;
        __shared__ int counts[kMaxBucketCount];
        __shared__ int write_offsets[kMaxBucketCount];

        for(int bucket = tid; bucket < bucket_count; bucket += 64)
        {
            counts[bucket] = 0;
        }
        __syncthreads();

        int elem_bucket[2];
        int64_t elem_kb[2];
        int elem_count = 0;
        for(int i = tid; i < arg.topk; i += 64)
        {
            const int64_t kb = input_row[i];
            int bucket = static_cast<int>(kb * bucket_count / arg.kv_block_count);
            bucket     = bucket < bucket_count ? bucket : bucket_count - 1;
            elem_bucket[elem_count] = bucket;
            elem_kb[elem_count]     = kb;
            ++elem_count;
            atomicAdd(&counts[bucket], 1);
        }
        __syncthreads();

        if(tid == 0)
        {
            int prefix = 0;
            for(int bucket = 0; bucket < bucket_count; ++bucket)
            {
                write_offsets[bucket] = prefix;
                prefix += counts[bucket];
            }
        }
        __syncthreads();

        for(int i = 0; i < elem_count; ++i)
        {
            const int pos = atomicAdd(&write_offsets[elem_bucket[i]], 1);
            output_row[pos] = elem_kb[i];
        }
    }
};

template <int MaxKvStageCount>
struct SlaAttnBwdDqKvPartitionGenericPipeline
{
    using Kargs = SlaAttnBwdDqKvPartitionKargs;
    static constexpr int kBucketsPerStage = 8;
    static constexpr int kMaxBucketCount =
        MaxKvStageCount * kBucketsPerStage;

    CK_TILE_DEVICE void operator()(const Kargs& arg) const
    {
        const int64_t row = ck_tile::get_block_1d_id();
        if(row >= arg.row_count)
        {
            return;
        }

        const int tid = ck_tile::get_thread_local_1d_id();
        const int64_t* input_row = arg.lut + row * arg.topk;
        int64_t* output_row      = arg.partitioned_lut + row * arg.topk;
        const int bucket_count   = arg.stage_count * kBucketsPerStage;
        __shared__ int counts[kMaxBucketCount];
        __shared__ int write_offsets[kMaxBucketCount];

        for(int bucket = tid; bucket < bucket_count; bucket += 64)
        {
            counts[bucket] = 0;
        }
        __syncthreads();

        for(int i = tid; i < arg.topk; i += 64)
        {
            const int64_t kb = input_row[i];
            int bucket = static_cast<int>(kb * bucket_count / arg.kv_block_count);
            bucket     = bucket < bucket_count ? bucket : bucket_count - 1;
            atomicAdd(&counts[bucket], 1);
        }
        __syncthreads();

        if(tid == 0)
        {
            int prefix = 0;
            for(int bucket = 0; bucket < bucket_count; ++bucket)
            {
                write_offsets[bucket] = prefix;
                prefix += counts[bucket];
            }
        }
        __syncthreads();

        for(int i = tid; i < arg.topk; i += 64)
        {
            const int64_t kb = input_row[i];
            int bucket = static_cast<int>(kb * bucket_count / arg.kv_block_count);
            bucket     = bucket < bucket_count ? bucket : bucket_count - 1;
            const int pos = atomicAdd(&write_offsets[bucket], 1);
            output_row[pos] = kb;
        }
    }
};

struct SlaBwdDkdvBucketLutKargs
{
    const int8_t* sparse_map;
    int32_t q_blocks;
    int32_t kv_blocks;
    int32_t block_m;
    int32_t* reverse_lut;
    int32_t* reverse_lut_size;
    int32_t* kv_perm;
};

struct SlaBwdDkdvBucketLutPipeline
{
    using Kargs = SlaBwdDkdvBucketLutKargs;
    static constexpr int kBlockSize = 256;
    static constexpr int kBucketCount = 256;

    CK_TILE_DEVICE void operator()(const Kargs& arg) const
    {
        const int32_t bh  = ck_tile::get_block_1d_id();
        const int32_t tid = ck_tile::get_thread_local_1d_id();
        const int8_t* sm = arg.sparse_map +
            static_cast<size_t>(bh) * arg.q_blocks * arg.kv_blocks;
        int32_t* rlut_head = arg.reverse_lut +
            static_cast<size_t>(bh) * arg.kv_blocks * arg.kv_blocks;
        int32_t* rsize_head =
            arg.reverse_lut_size + static_cast<size_t>(bh) * arg.kv_blocks;
        int32_t* perm_head = arg.kv_perm + static_cast<size_t>(bh) * arg.kv_blocks;

        extern __shared__ int32_t bucket_smem[];
        int32_t* bucket_counts  = bucket_smem;
        int32_t* bucket_cursors = bucket_counts + kBucketCount;

        for(int32_t i = tid; i < 2 * kBucketCount; i += kBlockSize)
        {
            bucket_smem[i] = 0;
        }
        __syncthreads();

        const int32_t q_slices = arg.block_m / 64;
        for(int32_t k = tid; k < arg.kv_blocks; k += kBlockSize)
        {
            int32_t count   = 0;
            uint32_t minhash = 0xffffffffu;
            int32_t* out = rlut_head + static_cast<size_t>(k) * arg.kv_blocks;
            for(int32_t q = 0; q < arg.q_blocks; ++q)
            {
                if(sm[static_cast<size_t>(q) * arg.kv_blocks + k] != 0)
                {
                    const int32_t q0 = q * q_slices;
                    out[count++]     = q0;
                    if(q_slices == 2)
                    {
                        out[count++] = q0 + 1;
                    }
                    uint32_t h = static_cast<uint32_t>(q) + 0x9e3779b9u;
                    h ^= h >> 16;
                    h *= 0x7feb352du;
                    h ^= h >> 15;
                    h *= 0x846ca68bu;
                    h ^= h >> 16;
                    if(h < minhash)
                    {
                        minhash = h;
                    }
                }
            }
            const uint32_t bucket = minhash >> 24;
            const uint32_t meta =
                (bucket << 16) | static_cast<uint32_t>(count);
            rsize_head[k] = static_cast<int32_t>(meta);
            atomicAdd(bucket_counts + bucket, 1);
        }
        __syncthreads();

        if(tid == 0)
        {
            int32_t offset = 0;
            for(int32_t bucket = 0; bucket < kBucketCount; ++bucket)
            {
                bucket_cursors[bucket] = offset;
                offset += bucket_counts[bucket];
            }
        }
        __syncthreads();

        for(int32_t k = tid; k < arg.kv_blocks; k += kBlockSize)
        {
            const uint32_t meta = static_cast<uint32_t>(rsize_head[k]);
            const int32_t count = static_cast<int32_t>(meta & 0xffffu);
            const int32_t bucket = static_cast<int32_t>(meta >> 16);
            const int32_t position = atomicAdd(bucket_cursors + bucket, 1);
            perm_head[position] = k;
            rsize_head[k]       = count;
        }
    }
};

struct SlaBwdDkdvClusterGreedyKargs
{
    const int32_t* reverse_lut;
    const int32_t* reverse_lut_size;
    int32_t q_blocks;
    int32_t kv_blocks;
    int32_t block_m;
    int32_t* kv_perm;
};

struct SlaBwdDkdvClusterGreedyPipeline
{
    using Kargs = SlaBwdDkdvClusterGreedyKargs;
    static constexpr int kBlockSize = 64;
    static constexpr int kSegmentSize = 64;

    CK_TILE_DEVICE void operator()(const Kargs& arg) const
    {
        const int32_t segments =
            (arg.kv_blocks + kSegmentSize - 1) / kSegmentSize;
        const int32_t block_id = ck_tile::get_block_1d_id();
        const int32_t bh = block_id / segments;
        const int32_t segment = block_id - bh * segments;
        const int32_t segment_begin = segment * kSegmentSize;
        const int32_t segment_count =
            segment_begin + kSegmentSize <= arg.kv_blocks
                ? kSegmentSize
                : arg.kv_blocks - segment_begin;
        const int32_t lane = ck_tile::get_thread_local_1d_id();
        const bool lane_has_candidate = lane < segment_count;

        const int32_t* rlut_head = arg.reverse_lut +
            static_cast<size_t>(bh) * arg.kv_blocks * arg.kv_blocks;
        const int32_t* rsize_head =
            arg.reverse_lut_size + static_cast<size_t>(bh) * arg.kv_blocks;
        int32_t* perm_head = arg.kv_perm + static_cast<size_t>(bh) * arg.kv_blocks;
        const int32_t candidate =
            lane_has_candidate ? perm_head[segment_begin + lane] : -1;

        extern __shared__ int32_t greedy_smem_i32[];
        int32_t* refs = greedy_smem_i32;
        volatile int32_t* order = refs + arg.q_blocks;
        const int32_t q_words = (arg.q_blocks + 63) / 64;
        const size_t scalar_count =
            static_cast<size_t>(arg.q_blocks + kSegmentSize);
        const size_t aligned_scalar_count = (scalar_count + 1) & ~size_t{1};
        uint64_t* masks =
            reinterpret_cast<uint64_t*>(greedy_smem_i32 + aligned_scalar_count);

        for(int32_t word = 0; word < q_words; ++word)
        {
            if(lane_has_candidate)
            {
                masks[static_cast<size_t>(word) * kSegmentSize + lane] = 0;
            }
        }
        if(lane_has_candidate)
        {
            const int32_t q_slices = arg.block_m / 64;
            const int32_t active_count = rsize_head[candidate];
            const int32_t* active =
                rlut_head + static_cast<size_t>(candidate) * arg.kv_blocks;
            for(int32_t i = 0; i < active_count; ++i)
            {
                const int32_t public_q = active[i] / q_slices;
                masks[static_cast<size_t>(public_q >> 6) * kSegmentSize + lane] |=
                    uint64_t{1} << (public_q & 63);
            }
        }
        for(int32_t q = lane; q < arg.q_blocks; q += kBlockSize)
        {
            refs[q] = 0;
        }
        __syncthreads();

        int32_t score = 0;
        bool visited = !lane_has_candidate;
        const int32_t window = segment_count / 2 > 16 ? segment_count / 2 : 16;

        for(int32_t step = 0; step < segment_count; ++step)
        {
            uint64_t key =
                visited ? 0
                        : (static_cast<uint64_t>(score) << 32) |
                              static_cast<uint32_t>(arg.kv_blocks - candidate);
#pragma unroll
            for(int32_t offset = 32; offset > 0; offset >>= 1)
            {
                const uint32_t other_lo =
                    __shfl_xor(static_cast<uint32_t>(key), offset);
                const uint32_t other_hi =
                    __shfl_xor(static_cast<uint32_t>(key >> 32), offset);
                const uint64_t other =
                    (static_cast<uint64_t>(other_hi) << 32) | other_lo;
                if(other > key)
                {
                    key = other;
                }
            }

            const int32_t selected =
                arg.kv_blocks - static_cast<int32_t>(key & 0xffffffffu);
            const uint64_t selected_lanes =
                __ballot(lane_has_candidate && candidate == selected);
            const int32_t best_lane = __builtin_ctzll(selected_lanes);
            if(lane == 0)
            {
                order[step] = best_lane;
                perm_head[segment_begin + step] = selected;
            }

            const int32_t old_lane =
                step + 1 > window ? order[step - window] : -1;
            for(int32_t word = 0; word < q_words; ++word)
            {
                const int32_t q = word * 64 + lane;
                const uint64_t best_mask =
                    masks[static_cast<size_t>(word) * kSegmentSize + best_lane];
                const uint64_t old_mask =
                    old_lane >= 0
                        ? masks[static_cast<size_t>(word) * kSegmentSize + old_lane]
                        : 0;
                const uint64_t own_mask =
                    lane_has_candidate
                        ? masks[static_cast<size_t>(word) * kSegmentSize + lane]
                        : 0;

                bool entered = false;
                bool exited  = false;
                if(q < arg.q_blocks)
                {
                    const int32_t previous = refs[q];
                    const int32_t next =
                        previous + static_cast<int32_t>((best_mask >> lane) & 1) -
                        static_cast<int32_t>((old_mask >> lane) & 1);
                    refs[q] = next;
                    entered = previous == 0 && next != 0;
                    exited  = previous != 0 && next == 0;
                }
                score += __builtin_popcountll(__ballot(entered) & own_mask);
                score -= __builtin_popcountll(__ballot(exited) & own_mask);
            }
            visited = visited || lane == best_lane;
        }
    }
};

} // namespace sla
} // namespace example
} // namespace ck_tile

struct SlaAttnFwdKvStagePartitionArgument
{
    const int64_t* lut;
    int64_t* partitioned_lut;
    int32_t* stage_offsets;
    int64_t row_count;
    int topk;
    int kv_block_count;
    int kv_stage_count;
    int stage_offset_stride;
};

template <int MaxKvStageCount>
struct SlaAttnFwdKvStagePartitionFastPipeline
{
    using Kargs = SlaAttnFwdKvStagePartitionArgument;
    static constexpr int kMaxBucketsPerStage = 8;
    static constexpr int kMaxBucketCount =
        MaxKvStageCount * kMaxBucketsPerStage;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.row_count));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(64); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int64_t row = static_cast<int64_t>(blockIdx.x);
        if(row >= arg.row_count)
        {
            return;
        }

        const int tid = static_cast<int>(threadIdx.x);
        const int topk = arg.topk;
        const int64_t* input_row = arg.lut + row * topk;
        int64_t* output_row = arg.partitioned_lut + row * topk;
        int32_t* offsets = arg.stage_offsets + row * arg.stage_offset_stride;

        const int bucket_count =
            arg.kv_stage_count * kMaxBucketsPerStage;
        __shared__ int s_counts[kMaxBucketCount];
        __shared__ int s_offsets[kMaxBucketCount];

        for(int bucket = tid; bucket < bucket_count; bucket += 64)
        {
            s_counts[bucket] = 0;
        }
        __syncthreads();

        int elem_bucket[2];
        int64_t elem_kb[2];
        int elem_count = 0;

        for(int i = tid; i < topk; i += 64)
        {
            const int64_t kb = input_row[i];
            int bucket = static_cast<int>(
                kb * bucket_count / arg.kv_block_count);
            bucket = bucket < bucket_count ? bucket : bucket_count - 1;
            elem_kb[elem_count] = kb;
            elem_bucket[elem_count] = bucket;
            elem_count++;
            atomicAdd(&s_counts[bucket], 1);
        }
        __syncthreads();

        if(tid == 0)
        {
            int prefix = 0;
            offsets[0] = 0;
            for(int bucket = 0; bucket < bucket_count; ++bucket)
            {
                s_offsets[bucket] = prefix;
                prefix += s_counts[bucket];
                if((bucket + 1) % kMaxBucketsPerStage == 0)
                {
                    offsets[(bucket + 1) / kMaxBucketsPerStage] = prefix;
                }
            }
        }
        __syncthreads();

        for(int idx = 0; idx < elem_count; ++idx)
        {
            int bucket = elem_bucket[idx];
            int pos = atomicAdd(&s_offsets[bucket], 1);
            output_row[pos] = elem_kb[idx];
        }
    }
};

template <int MaxKvStageCount>
struct SlaAttnFwdKvStagePartitionGenericPipeline
{
    using Kargs = SlaAttnFwdKvStagePartitionArgument;
    static constexpr int kMaxBucketsPerStage = 8;
    static constexpr int kMaxBucketCount =
        MaxKvStageCount * kMaxBucketsPerStage;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.row_count));
    }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(64); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int64_t row = static_cast<int64_t>(blockIdx.x);
        if(row >= arg.row_count)
        {
            return;
        }

        const int tid = static_cast<int>(threadIdx.x);
        const int topk = arg.topk;
        const int64_t* input_row = arg.lut + row * topk;
        int64_t* output_row = arg.partitioned_lut + row * topk;
        int32_t* offsets = arg.stage_offsets + row * arg.stage_offset_stride;
        const int bucket_count = arg.kv_stage_count * kMaxBucketsPerStage;
        __shared__ int s_counts[kMaxBucketCount];
        __shared__ int s_offsets[kMaxBucketCount];

        for(int bucket = tid; bucket < bucket_count; bucket += 64)
        {
            s_counts[bucket] = 0;
        }
        __syncthreads();

        for(int i = tid; i < topk; i += 64)
        {
            const int64_t kb = input_row[i];
            int bucket = static_cast<int>(kb * bucket_count / arg.kv_block_count);
            bucket = bucket < bucket_count ? bucket : bucket_count - 1;
            atomicAdd(&s_counts[bucket], 1);
        }
        __syncthreads();

        if(tid == 0)
        {
            int prefix = 0;
            offsets[0] = 0;
            for(int bucket = 0; bucket < bucket_count; ++bucket)
            {
                s_offsets[bucket] = prefix;
                prefix += s_counts[bucket];
                if((bucket + 1) % kMaxBucketsPerStage == 0)
                {
                    offsets[(bucket + 1) / kMaxBucketsPerStage] = prefix;
                }
            }
        }
        __syncthreads();

        for(int i = tid; i < topk; i += 64)
        {
            const int64_t kb = input_row[i];
            int bucket = static_cast<int>(kb * bucket_count / arg.kv_block_count);
            bucket = bucket < bucket_count ? bucket : bucket_count - 1;
            const int pos = atomicAdd(&s_offsets[bucket], 1);
            output_row[pos] = kb;
        }
    }
};
