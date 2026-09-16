// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/unified_attention/kernel/unified_attention_d256_mmac_kernel.hpp"
#include "ck_tile/host.hpp"

// This is the same fused active-metadata/Q/K/V prepare kernel used by the
// extension entry. Keep its task partition and packed layouts in sync there.
struct UnifiedAttentionD256PrepareSeqArgument
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    uint16_t* q_pack;
    uint16_t* k_pack;
    uint16_t* v_pack;
    const int32_t* block_table;
    const int32_t* mm_ranges;
    int32_t* active_counts;
    int32_t* active_indices;
    int task_begin;
    int active_blocks;
    int q_blocks;
    int kv_blocks;
    int n_q_pad;
    int n_kv_pad;
    int pack_start;
    int h_q;
    int h_kv;
    int d;
    int d_pad;
    int page_size;
    int nb;
    int max_mm_ranges;
    int sliding_window;
    int64_t q_stride_t;
    int64_t q_stride_h;
    int64_t k_s0;
    int64_t k_s1;
    int64_t k_s2;
    int64_t v_s0;
    int64_t v_s1;
    int64_t v_s2;
};

struct UnifiedAttentionD256PrepareBatchArgument
{
    UnifiedAttentionD256PrepareSeqArgument seqs[kMaxBatchSeqs];
    const int32_t* q_offsets;
    const int32_t* n_kvs;
    int num_seqs;
    int total_blocks;
};

struct UnifiedAttentionD256PrepareBatchKernel
{
    using Kargs = UnifiedAttentionD256PrepareBatchArgument;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& arg)
    {
        return dim3(static_cast<unsigned int>(arg.total_blocks));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(256); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
    const int task = static_cast<int>(blockIdx.x);
    int seq = 0;
    while(seq + 1 < arg.num_seqs && task >= arg.seqs[seq + 1].task_begin)
        ++seq;

    const auto& prep = arg.seqs[seq];
    int local_block = task - prep.task_begin;
    const int tid = static_cast<int>(threadIdx.x);
    const int q_begin = arg.q_offsets[seq];
    const int n_q = arg.q_offsets[seq + 1] - q_begin;
    const int n_kv = arg.n_kvs[seq];
    const int nqb = (n_q + kBlockM - 1) / kBlockM;
    const int nb = (n_kv + kBlockN - 1) / kBlockN;

    if(local_block < prep.active_blocks)
    {
        extern __shared__ uint32_t shared_seen[];
        constexpr int kPrepareBlockM = 64;
        const int row = local_block;
        const int rows = prep.h_q * nqb;
        if(row >= rows)
            return;
        const int n_seen_words = (nb + 31) / 32;
        for(int i = tid; i < n_seen_words; i += blockDim.x)
            shared_seen[i] = 0;
        __syncthreads();

        const int qb = row % nqb;
        int32_t* row_indices =
            prep.active_indices + static_cast<int64_t>(row) * prep.nb;
        const int context = n_kv - n_q;
        const int first_q = qb * kPrepareBlockM;
        const int last_q = min(first_q + kPrepareBlockM - 1, n_q - 1);

        auto set_block = [&](int kb) {
            if(kb >= 0 && kb < nb)
                atomicOr(&shared_seen[kb >> 5], 1u << (kb & 31));
        };

        int first_block = 0;
        if(prep.sliding_window > 0)
        {
            const int64_t abs_first =
                static_cast<int64_t>(context) + first_q - prep.sliding_window + 1;
            first_block =
                abs_first > 0 ? static_cast<int>(abs_first / kPrepareBlockM) : 0;
        }
        const int last_block =
            min(n_kv - 1, (context + last_q) / kPrepareBlockM);
        for(int kb = first_block + tid; kb <= last_block; kb += blockDim.x)
            set_block(kb);

        const int q_abs_first = context + first_q;
        const int q_abs_last = context + last_q;
        if(prep.mm_ranges != nullptr)
        {
            for(int i = 0; i < prep.max_mm_ranges; ++i)
            {
                const int begin = prep.mm_ranges[i * 2];
                const int end = prep.mm_ranges[i * 2 + 1];
                if(begin < end && q_abs_first <= end && q_abs_last >= begin)
                {
                    const int prefix_first = begin > 0 ? begin / kPrepareBlockM : 0;
                    const int prefix_last = min(n_kv - 1, end / kPrepareBlockM);
                    for(int kb = prefix_first + tid; kb <= prefix_last; kb += blockDim.x)
                        set_block(kb);
                }
            }
        }
        __syncthreads();
        if(tid == 0)
        {
            int count = 0;
            for(int kb = 0; kb < nb; ++kb)
            {
                if((shared_seen[kb >> 5] & (1u << (kb & 31))) != 0)
                    row_indices[count++] = kb;
            }
            prep.active_counts[row] = count;
        }
        return;
    }

    local_block -= prep.active_blocks;
    if(local_block < prep.q_blocks)
    {
        constexpr int kVector = kAsyncKVector;
        const int vectors_per_row = prep.d / kVector;
        constexpr int kLanesPerRow = 32;
        constexpr int kRowsPerBlock = 256 / kLanesPerRow;
        const int row_group = tid / kLanesPerRow;
        const int lane_in_row = tid % kLanesPerRow;
        const int shuffle_leader = (tid % kAsyncWarpSize) & ~(kLanesPerRow - 1);
        const int64_t total_rows = static_cast<int64_t>(n_q) * prep.h_q;
        for(int64_t row = static_cast<int64_t>(local_block) * kRowsPerBlock + row_group;
            row < total_rows;
            row += static_cast<int64_t>(prep.q_blocks) * kRowsPerBlock)
        {
            int t = 0;
            int h = 0;
            if(lane_in_row == 0)
            {
                t = row % n_q;
                h = row / n_q;
            }
            t = ck_tile::warp_shuffle(t, shuffle_leader);
            h = ck_tile::warp_shuffle(h, shuffle_leader);
            if(lane_in_row < vectors_per_row)
            {
                const int x = lane_in_row * kVector;
                const int64_t src_offset =
                    static_cast<int64_t>(q_begin + t) * prep.q_stride_t +
                    static_cast<int64_t>(h) * prep.q_stride_h + x;
                const int64_t dst_offset =
                    (static_cast<int64_t>(h) * prep.n_q_pad + t) * prep.d_pad + x;
                *reinterpret_cast<ck_tile::uint16x8_t*>(prep.q_pack + dst_offset) =
                    *reinterpret_cast<const ck_tile::uint16x8_t*>(prep.q + src_offset);
            }
        }
        return;
    }

    local_block -= prep.q_blocks;
    constexpr int kVector = kAsyncKVector;
    const int vectors_per_row = prep.d / kVector;
    const int pack_span = n_kv - prep.pack_start;
    if(pack_span <= 0)
        return;
    constexpr int kLanesPerRow = 32;
    constexpr int kRowsPerBlock = 256 / kLanesPerRow;
    const int row_group = tid / kLanesPerRow;
    const int lane_in_row = tid % kLanesPerRow;
    const int shuffle_leader = (tid % kAsyncWarpSize) & ~(kLanesPerRow - 1);
    const int64_t total_rows = static_cast<int64_t>(pack_span) * prep.h_kv;
    for(int64_t packed_row = static_cast<int64_t>(local_block) * kRowsPerBlock + row_group;
        packed_row < total_rows;
        packed_row += static_cast<int64_t>(prep.kv_blocks) * kRowsPerBlock)
    {
        int pos = 0;
        int h = 0;
        int physical_page = 0;
        int page_offset = 0;
        if(lane_in_row == 0)
        {
            pos = prep.pack_start + packed_row % pack_span;
            h = packed_row / pack_span;
            physical_page = prep.block_table[pos / prep.page_size];
            page_offset = pos % prep.page_size;
        }
        pos = ck_tile::warp_shuffle(pos, shuffle_leader);
        h = ck_tile::warp_shuffle(h, shuffle_leader);
        physical_page = ck_tile::warp_shuffle(physical_page, shuffle_leader);
        page_offset = ck_tile::warp_shuffle(page_offset, shuffle_leader);
        if(lane_in_row < vectors_per_row)
        {
            const int x = lane_in_row * kVector;
            const int64_t dst_offset =
                (static_cast<int64_t>(h) * prep.n_kv_pad + pos) * prep.d_pad + x;
            const int64_t k_offset = static_cast<int64_t>(physical_page) * prep.k_s0 +
                                     static_cast<int64_t>(page_offset) * prep.k_s1 +
                                     static_cast<int64_t>(h) * prep.k_s2 + x;
            const int64_t v_offset = static_cast<int64_t>(physical_page) * prep.v_s0 +
                                     static_cast<int64_t>(page_offset) * prep.v_s1 +
                                     static_cast<int64_t>(h) * prep.v_s2 + x;
            *reinterpret_cast<ck_tile::uint16x8_t*>(prep.k_pack + dst_offset) =
                *reinterpret_cast<const ck_tile::uint16x8_t*>(prep.k + k_offset);
            *reinterpret_cast<ck_tile::uint16x8_t*>(prep.v_pack + dst_offset) =
                *reinterpret_cast<const ck_tile::uint16x8_t*>(prep.v + v_offset);
        }
    }
    }
};
