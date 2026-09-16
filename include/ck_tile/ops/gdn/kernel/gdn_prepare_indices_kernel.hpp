// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ck_tile {

struct PrepareChunkIndicesKernel
{
    struct Kargs
    {
        const int64_t* cu_seqlens;
        int64_t* chunk_offsets;
        int64_t* chunk_indices;
        ck_tile::index_t num_sequences;
        ck_tile::index_t chunk_size;
    };

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        const ck_tile::index_t tid =
            static_cast<ck_tile::index_t>(threadIdx.x);
        if(tid == 0)
        {
            args.chunk_offsets[0] = 0;
            for(ck_tile::index_t sequence = 0;
                sequence < args.num_sequences;
                ++sequence)
            {
                const int64_t length =
                    args.cu_seqlens[sequence + 1] -
                    args.cu_seqlens[sequence];
                args.chunk_offsets[sequence + 1] =
                    args.chunk_offsets[sequence] +
                    (length + args.chunk_size - 1) / args.chunk_size;
            }
        }
        __syncthreads();

        for(ck_tile::index_t sequence = tid;
            sequence < args.num_sequences;
            sequence += static_cast<ck_tile::index_t>(blockDim.x))
        {
            const int64_t begin = args.chunk_offsets[sequence];
            const int64_t end   = args.chunk_offsets[sequence + 1];
            for(int64_t global_chunk = begin;
                global_chunk < end;
                ++global_chunk)
            {
                args.chunk_indices[global_chunk * 2] =
                    static_cast<int64_t>(sequence);
                args.chunk_indices[global_chunk * 2 + 1] =
                    global_chunk - begin;
            }
        }
    }
};

} // namespace ck_tile
