// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

template <typename DataType_>
struct ChunkDeltaHScanFwdTraits {
    using DataType = DataType_;
};

template <typename DataType>
struct ChunkDeltaHScanFwdKargs {
    const DataType*  k;
    const DataType*  w;
    const DataType*  u;
    const float*     g;
    const float*     gk;
    const float*     initial_state;
    const int64_t*   cu_seqlens;
    const int64_t*   chunk_offsets;
    DataType*        h;
    DataType*        v_new;
    float*           final_state;
    index_t          total_tokens;
    index_t          num_sequences;
    index_t          num_chunks;
    index_t          num_qk_heads;
    index_t          num_value_heads;
    bool             has_initial_state;
    bool             store_final_state;
    bool             save_new_value;
    bool             use_g;
    bool             use_gk;
    bool             use_exp2;
    bool             transpose_state;
    bool             is_varlen;

    // Scan specific pointers.
    DataType*        h_local;       // [NT, B, HV, K, V]
    float*           decay_local;   // [NT, B, HV, K]
    DataType*        h_start;       // [NT, B, HV, K, V]
};

} // namespace ck_tile
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_scan_prefix_pipeline.hpp"
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_wave_reg_pipeline.hpp"

namespace ck_tile {

template <typename Traits,
          typename Policy,
          bool UseMainPipeline,
          bool PreshuffledH = false>
struct ChunkDeltaHScanKernel {
    using DataType = typename Traits::DataType;
    using Kargs    = ChunkDeltaHScanFwdKargs<DataType>;
    using Pipeline = std::conditional_t<
        UseMainPipeline,
        ChunkDeltaHWaveRegPipeline<Traits, Policy, PreshuffledH>,
        ChunkDeltaHScanPrefixPipeline<Traits, Policy>>;

    CK_TILE_DEVICE void operator()(Kargs args) const {
        // grid: (V_tiles, B * HV, 1)
        const index_t v_begin = blockIdx.x * Policy::kVTile;
        const index_t seq_vh  = blockIdx.y;

        const index_t seq = seq_vh / args.num_value_heads;
        const index_t vh  = seq_vh % args.num_value_heads;

        index_t bos, eos, chunk_base, local_chunks;
        if (args.is_varlen) {
            bos = args.cu_seqlens[seq];
            eos = args.cu_seqlens[seq + 1];
            chunk_base = args.chunk_offsets[seq];
            local_chunks = integer_divide_ceil(eos - bos, static_cast<index_t>(64));
        } else {
            bos = seq * args.total_tokens;
            eos = bos + args.total_tokens;
            chunk_base = seq * args.num_chunks;
            local_chunks = args.num_chunks;
        }

        __shared__ typename Pipeline::LdsStorage smem;

        const index_t qh = (args.num_qk_heads == args.num_value_heads) ? vh : ((args.num_value_heads > args.num_qk_heads) ? vh / (args.num_value_heads / args.num_qk_heads) : vh * (args.num_qk_heads / args.num_value_heads));

        if constexpr(UseMainPipeline)
        {
            Pipeline{}(args,
                       seq,
                       vh,
                       qh,
                       v_begin,
                       bos,
                       eos,
                       chunk_base,
                       local_chunks,
                       smem);
        }
        else
        {
            // DRAM base pointers for token-major tensors: [B, T, H, D].
            // Varlen input is packed along the token axis in batch 0; bos/eos already
            // provide the global token offset, so only fixed-length batches need seq offset.
            const long_index_t seq_k_base =
                args.is_varlen
                    ? 0
                    : static_cast<long_index_t>(seq) * args.total_tokens * args.num_qk_heads *
                          Policy::kHeadDim;
            const long_index_t seq_v_base =
                args.is_varlen
                    ? 0
                    : static_cast<long_index_t>(seq) * args.total_tokens *
                          args.num_value_heads * Policy::kHeadDim;
            const long_index_t seq_u_base =
                args.is_varlen
                    ? 0
                    : static_cast<long_index_t>(seq) * args.total_tokens *
                          args.num_value_heads * Policy::kValueDim;
            const auto k_ptr = args.k + seq_k_base + qh * Policy::kHeadDim;
            const auto w_ptr = args.w + seq_v_base + vh * Policy::kHeadDim;
            const auto u_ptr = args.u + seq_u_base + vh * Policy::kValueDim;

            const auto k_dram = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(args.total_tokens, number<Policy::kHeadDim>{}),
                make_tuple(args.num_qk_heads * Policy::kHeadDim, number<1>{}),
                number<8>{},
                number<1>{});

            const auto w_dram = make_naive_tensor_view<address_space_enum::global>(
                w_ptr,
                make_tuple(args.total_tokens, number<Policy::kHeadDim>{}),
                make_tuple(args.num_value_heads * Policy::kHeadDim, number<1>{}),
                number<8>{},
                number<1>{});

            const auto u_dram = make_naive_tensor_view<address_space_enum::global>(
                u_ptr,
                make_tuple(args.total_tokens, number<Policy::kValueDim>{}),
                make_tuple(args.num_value_heads * Policy::kValueDim, number<1>{}),
                number<8>{},
                number<1>{});

            Pipeline{}(args,
                       seq,
                       vh,
                       qh,
                       v_begin,
                       bos,
                       eos,
                       chunk_base,
                       local_chunks,
                       k_dram,
                       w_dram,
                       u_dram,
                       smem);
        }
    }
};

} // namespace ck_tile
