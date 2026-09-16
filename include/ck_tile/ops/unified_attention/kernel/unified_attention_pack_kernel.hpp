// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

// Lightweight CK-Tile kernels used to transform strided/paged Q/K/V into the
// layouts consumed by the MMAC pipelines.  The copy width and work mapping are
// compile-time properties; tensor geometry remains runtime data.

enum class UnifiedAttentionSinglePackSchedule
{
    Concatenated,
    ParallelQKv,
};

struct UnifiedAttentionSinglePackArgs
{
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    uint16_t* q_dst;
    uint16_t* k_dst;
    uint16_t* v_dst;
    const int32_t* block_table;
    int n_q;
    int n_q_pad;
    int h_q;
    int n_kv;
    int n_kv_pad;
    int pack_start;
    int h_kv;
    int d;
    int d_pad;
    int page_size;
    int64_t q_s0;
    int64_t q_s1;
    int64_t k_s0;
    int64_t k_s1;
    int64_t k_s2;
    int64_t v_s0;
    int64_t v_s1;
    int64_t v_s2;
    bool grouped_m16_q;
    unsigned int grid_size;
};

template <int VectorSize, UnifiedAttentionSinglePackSchedule Schedule>
struct UnifiedAttentionSinglePackKernel
{
    static constexpr int kVectorSize = 8;
    static_assert(VectorSize == 1 || VectorSize == kVectorSize);
    static constexpr int kBlockSize = 256;
    static constexpr int kGroupedQueryHeads = 8;
    using Kargs = UnifiedAttentionSinglePackArgs;

    CK_TILE_HOST static dim3 GridSize(const Kargs& arg) { return dim3(arg.grid_size); }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int units_per_row = arg.d_pad / VectorSize;
        const int64_t q_total =
            static_cast<int64_t>(arg.n_q_pad) * arg.h_q * units_per_row;
        const int pack_span = arg.n_kv_pad - arg.pack_start;
        const int64_t kv_total =
            static_cast<int64_t>(pack_span) * arg.h_kv * units_per_row;
        const int64_t global_tid =
            static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t grid_stride =
            static_cast<int64_t>(blockDim.x) * gridDim.x;

        auto copy_q = [&](int64_t linear) {
            const int unit = linear % units_per_row;
            const int x = unit * VectorSize;
            const int64_t row = linear / units_per_row;
            int t;
            int h;
            bool valid_row = true;
            if(arg.grouped_m16_q)
            {
                const int q_blocks = (arg.n_q + 1) / 2;
                const int grouped_rows = arg.h_kv * q_blocks * 16;
                valid_row = row < grouped_rows;
                const int grouped_row = static_cast<int>(row);
                const int lane_row = grouped_row % 16;
                const int qb = (grouped_row / 16) % q_blocks;
                const int kv_h = grouped_row / (q_blocks * 16);
                t = qb * 2 + lane_row / kGroupedQueryHeads;
                h = kv_h * kGroupedQueryHeads + lane_row % kGroupedQueryHeads;
            }
            else
            {
                t = static_cast<int>(row % arg.n_q_pad);
                h = static_cast<int>(row / arg.n_q_pad);
            }
            const bool valid = valid_row && t < arg.n_q && x < arg.d;
            if constexpr(VectorSize == kVectorSize)
            {
                auto* dst = reinterpret_cast<uint4*>(arg.q_dst) + linear;
                if(valid)
                {
                    const int64_t src_offset = static_cast<int64_t>(t) * arg.q_s0 +
                                               static_cast<int64_t>(h) * arg.q_s1 + x;
                    *dst = *reinterpret_cast<const uint4*>(arg.q + src_offset);
                }
                else
                {
                    *dst = uint4{0, 0, 0, 0};
                }
            }
            else
            {
                arg.q_dst[linear] =
                    valid ? arg.q[static_cast<int64_t>(t) * arg.q_s0 +
                                  static_cast<int64_t>(h) * arg.q_s1 + x]
                          : uint16_t{0};
            }
        };

        auto copy_kv = [&](int64_t linear) {
            const int unit = linear % units_per_row;
            const int x = unit * VectorSize;
            const int64_t row = linear / units_per_row;
            const int pos = arg.pack_start + static_cast<int>(row % pack_span);
            const int h = static_cast<int>(row / pack_span);
            const int64_t dst_offset =
                (static_cast<int64_t>(h) * arg.n_kv_pad + pos) * arg.d_pad + x;
            const bool valid = pos < arg.n_kv && x < arg.d;
            if constexpr(VectorSize == kVectorSize)
            {
                auto* k_dst = reinterpret_cast<uint4*>(arg.k_dst + dst_offset);
                auto* v_dst = reinterpret_cast<uint4*>(arg.v_dst + dst_offset);
                if(valid)
                {
                    const int physical_page = arg.block_table[pos / arg.page_size];
                    const int page_offset = pos % arg.page_size;
                    const int64_t k_offset = static_cast<int64_t>(physical_page) * arg.k_s0 +
                                             static_cast<int64_t>(page_offset) * arg.k_s1 +
                                             static_cast<int64_t>(h) * arg.k_s2 + x;
                    const int64_t v_offset = static_cast<int64_t>(physical_page) * arg.v_s0 +
                                             static_cast<int64_t>(page_offset) * arg.v_s1 +
                                             static_cast<int64_t>(h) * arg.v_s2 + x;
                    *k_dst = *reinterpret_cast<const uint4*>(arg.k + k_offset);
                    *v_dst = *reinterpret_cast<const uint4*>(arg.v + v_offset);
                }
                else
                {
                    *k_dst = uint4{0, 0, 0, 0};
                    *v_dst = uint4{0, 0, 0, 0};
                }
            }
            else
            {
                if(valid)
                {
                    const int physical_page = arg.block_table[pos / arg.page_size];
                    const int page_offset = pos % arg.page_size;
                    arg.k_dst[dst_offset] =
                        arg.k[static_cast<int64_t>(physical_page) * arg.k_s0 +
                              static_cast<int64_t>(page_offset) * arg.k_s1 +
                              static_cast<int64_t>(h) * arg.k_s2 + x];
                    arg.v_dst[dst_offset] =
                        arg.v[static_cast<int64_t>(physical_page) * arg.v_s0 +
                              static_cast<int64_t>(page_offset) * arg.v_s1 +
                              static_cast<int64_t>(h) * arg.v_s2 + x];
                }
                else
                {
                    arg.k_dst[dst_offset] = uint16_t{0};
                    arg.v_dst[dst_offset] = uint16_t{0};
                }
            }
        };

        if constexpr(Schedule == UnifiedAttentionSinglePackSchedule::ParallelQKv)
        {
            if(global_tid < q_total) copy_q(global_tid);
            for(int64_t linear = global_tid; linear < kv_total; linear += grid_stride)
                copy_kv(linear);
        }
        else
        {
            for(int64_t linear = global_tid; linear < q_total + kv_total;
                linear += grid_stride)
            {
                if(linear < q_total)
                    copy_q(linear);
                else
                    copy_kv(linear - q_total);
            }
        }
    }
};

struct UnifiedAttentionBatchedQPackArgs
{
    const uint16_t* src;
    uint16_t* dst;
    const int32_t* cu_q;
    int batch;
    int heads;
    int total_nq_pad;
    int d;
    int d_pad;
    int64_t src_s0;
    int64_t src_s1;
    unsigned int grid_size;
};

struct UnifiedAttentionBatchedQPackKernel
{
    static constexpr int kBlockSize = 256;
    using Kargs = UnifiedAttentionBatchedQPackArgs;
    CK_TILE_HOST static dim3 GridSize(const Kargs& arg) { return dim3(arg.grid_size); }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int64_t total =
            static_cast<int64_t>(arg.heads) * arg.total_nq_pad * arg.d_pad;
        for(int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            linear < total;
            linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int x = linear % arg.d_pad;
            const int64_t row = linear / arg.d_pad;
            const int padded_q_pos = row % arg.total_nq_pad;
            const int h = row / arg.total_nq_pad;
            int b = 0;
            int padded_q_begin = 0;
            for(; b < arg.batch; ++b)
            {
                const int length = arg.cu_q[b + 1] - arg.cu_q[b];
                const int padded_length =
                    ((length + kBlockM - 1) / kBlockM) * kBlockM;
                if(padded_q_pos < padded_q_begin + padded_length) break;
                padded_q_begin += padded_length;
            }
            const int q_pos = padded_q_pos - padded_q_begin;
            const int q_len = arg.cu_q[b + 1] - arg.cu_q[b];
            arg.dst[linear] = q_pos < q_len && x < arg.d
                ? arg.src[static_cast<int64_t>(arg.cu_q[b] + q_pos) * arg.src_s0 +
                          static_cast<int64_t>(h) * arg.src_s1 + x]
                : uint16_t{0};
        }
    }
};

struct UnifiedAttentionBatchedKvPackArgs
{
    const uint16_t* k;
    const uint16_t* v;
    uint16_t* k_dst;
    uint16_t* v_dst;
    const int32_t* cu_q;
    const int32_t* seqlens_k;
    const int32_t* block_table;
    int batch;
    int n_kv_pad;
    int pack_start;
    int h_kv;
    int sliding_window;
    bool include_pack_start;
    int d;
    int d_pad;
    int page_size;
    int64_t block_table_s0;
    int64_t k_s0;
    int64_t k_s1;
    int64_t k_s2;
    int64_t v_s0;
    int64_t v_s1;
    int64_t v_s2;
    unsigned int grid_size;
};

struct UnifiedAttentionBatchedKvPackKernel
{
    static constexpr int kBlockSize = 256;
    static constexpr int kElemsPerVec = 8;
    static constexpr int kMetadataBlockSize = 64;
    using Kargs = UnifiedAttentionBatchedKvPackArgs;
    CK_TILE_HOST static dim3 GridSize(const Kargs& arg) { return dim3(arg.grid_size); }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        const int vecs_per_row = arg.d_pad / kElemsPerVec;
        const int64_t total = static_cast<int64_t>(gridDim.x) * blockDim.x;
        const uint4 zero{0, 0, 0, 0};
        for(int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            linear < total;
            linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int x = static_cast<int>(linear % vecs_per_row) * kElemsPerVec;
            int64_t compact_row = linear / vecs_per_row;
            int b = 0;
            int h = 0;
            int pos = 0;
            for(; b < arg.batch; ++b)
            {
                const int q_len = arg.cu_q[b + 1] - arg.cu_q[b];
                const int raw_start =
                    arg.seqlens_k[b] - q_len - arg.sliding_window + 1;
                int batch_start = arg.sliding_window > 0
                    ? ((raw_start > 0 ? raw_start : 0) / kMetadataBlockSize) *
                          kMetadataBlockSize
                    : arg.pack_start;
                if(arg.include_pack_start && arg.pack_start < batch_start)
                    batch_start = arg.pack_start;
                const int span = arg.n_kv_pad - batch_start;
                const int64_t batch_rows = static_cast<int64_t>(arg.h_kv) * span;
                if(compact_row < batch_rows)
                {
                    pos = batch_start + compact_row % span;
                    h = compact_row / span;
                    break;
                }
                compact_row -= batch_rows;
            }
            if(b == arg.batch) continue;
            const int64_t dst_offset =
                ((static_cast<int64_t>(b) * arg.h_kv + h) * arg.n_kv_pad + pos) *
                    arg.d_pad + x;
            auto* k_dst = reinterpret_cast<uint4*>(arg.k_dst + dst_offset);
            auto* v_dst = reinterpret_cast<uint4*>(arg.v_dst + dst_offset);
            if(pos < arg.seqlens_k[b] && x < arg.d)
            {
                const int physical_page =
                    arg.block_table[static_cast<int64_t>(b) * arg.block_table_s0 +
                                    pos / arg.page_size];
                const int page_offset = pos % arg.page_size;
                const int64_t k_offset = static_cast<int64_t>(physical_page) * arg.k_s0 +
                                         static_cast<int64_t>(page_offset) * arg.k_s1 +
                                         static_cast<int64_t>(h) * arg.k_s2 + x;
                const int64_t v_offset = static_cast<int64_t>(physical_page) * arg.v_s0 +
                                         static_cast<int64_t>(page_offset) * arg.v_s1 +
                                         static_cast<int64_t>(h) * arg.v_s2 + x;
                *k_dst = *reinterpret_cast<const uint4*>(arg.k + k_offset);
                *v_dst = *reinterpret_cast<const uint4*>(arg.v + v_offset);
            }
            else
            {
                *k_dst = zero;
                *v_dst = zero;
            }
        }
    }
};

template <typename Kernel>
CK_TILE_HOST void launch_unified_attention_pack(
    const typename Kernel::Kargs& args, hipStream_t stream)
{
    constexpr auto block = Kernel::BlockSize();
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<block.x, 1>(
            Kernel{}, Kernel::GridSize(args), block, 0, args));
}
