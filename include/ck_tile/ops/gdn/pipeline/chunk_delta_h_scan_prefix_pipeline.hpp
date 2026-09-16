// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once
#include <ck_tile/core.hpp>
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_scan_policy.hpp"

namespace ck_tile {

template <typename DataType, index_t BT, index_t HD, index_t VT, index_t Pad, bool Compact>
struct ChunkDeltaHScanPrefixLdsStorage;

template <typename DataType, index_t BT, index_t HD, index_t VT, index_t Pad>
struct ChunkDeltaHScanPrefixLdsStorage<DataType, BT, HD, VT, Pad, false> {
    float g_lds[BT];
    float gk_decay_lds[HD];
    DataType w_lds[BT * (HD + Pad)];
    union {
        DataType u_lds[BT * (VT + Pad)];
        DataType k_t_lds[HD * (BT + Pad)];
    };
    union {
        DataType state_t_lds[VT * (HD + Pad)];
        DataType residual_t_lds[VT * (BT + Pad)];
    };
    float update_lds[HD * (VT + Pad)];
};

template <typename DataType, index_t BT, index_t HD, index_t VT, index_t Pad>
struct ChunkDeltaHScanPrefixLdsStorage<DataType, BT, HD, VT, Pad, true> {
    float g_lds[BT];
    float gk_decay_lds[HD];
    union {
        DataType w_lds[BT * (HD + Pad)];
        float update_lds[HD * (VT + Pad)];
    };
    union {
        DataType u_lds[BT * (VT + Pad)];
        DataType residual_t_lds[VT * (BT + Pad)];
    };
    union {
        DataType state_t_lds[VT * (HD + Pad)];
        DataType k_t_lds[HD * (BT + Pad)];
    };
};

template <typename Traits, typename Policy>
struct ChunkDeltaHScanPrefixPipeline {
    using DataType = typename Traits::DataType;

    static constexpr index_t kBT  = Policy::kChunkSize;
    static constexpr index_t kHD  = Policy::kHeadDim;
    static constexpr index_t kVT  = Policy::kVTile;
    static constexpr index_t kVD  = Policy::kValueDim;
    static constexpr index_t kBS  = Policy::kBlockSize;
    static constexpr index_t kSPT = Policy::kSPT;
    static constexpr index_t kRPT = kBT * kVT / kBS;

    static_assert(kHD * kVT % kBS == 0,
                  "prefix pipeline requires complete per-thread state coverage");
    static_assert(kBT * kVT % kBS == 0,
                  "prefix pipeline requires complete per-thread residual coverage");

    static constexpr index_t kLdsPad      = 4;
    static constexpr index_t kHD_padded   = kHD + kLdsPad;
    static constexpr index_t kVT_padded   = kVT + kLdsPad;
    static constexpr index_t kBT_padded   = kBT + kLdsPad;
    CK_TILE_DEVICE static float fast_exp2(float x)
    {
        return __builtin_amdgcn_exp2f(x);
    }

    using LdsStorage = ChunkDeltaHScanPrefixLdsStorage<
        DataType, kBT, kHD, kVT, kLdsPad, kVT == 64>;

    template <typename KDramView, typename WDramView, typename UDramView>
    CK_TILE_DEVICE void operator()(
        const ChunkDeltaHScanFwdKargs<DataType>& args,
        index_t seq,
        index_t vh,
        index_t qh,
        index_t v_begin,
        index_t bos,
        index_t eos,
        index_t chunk_base,
        index_t local_chunks,
        const KDramView& k_dram,
        const WDramView& w_dram,
        const UDramView& u_dram,
        LdsStorage& smem
    ) const {
        const index_t tid = ck_tile::get_thread_id();
        constexpr auto proj_gemm = typename Policy::template ProjectionBlockGemm<DataType>{};
        constexpr auto update_gemm = typename Policy::template UpdateBlockGemm<DataType>{};
        // Register array for local state
        float s[kSPT];

        const index_t vv = tid % kVT;
        const index_t kd_stride = kBS / kVT;
        const index_t kd_base = tid / kVT;

        // Initialize state (zero or from initial_state)
        #pragma unroll
        for (index_t ii = 0; ii < kSPT; ++ii) {
            const index_t kd = kd_base + ii * kd_stride;
            s[ii] = 0.0f;
            if (args.has_initial_state && v_begin + vv < kVD) {
                const int64_t off =
                    ((static_cast<int64_t>(seq) * args.num_value_heads + vh) * kHD + kd)
                    * kVD + v_begin + vv;
                s[ii] = args.initial_state[off];
            }
        }

        // 1. Initial prefetch load for lc = 0
        constexpr index_t u_vector = kVT >= 32 ? 8 : 4;
        const index_t tb_init = args.is_varlen ? bos : 0;

        auto make_w_dram_window = [&](index_t token) {
            return make_tile_window(
                w_dram,
                make_tuple(number<kBT>{}, number<kHD>{}),
                multi_index<2>{token, 0},
                MakeGdnAsyncDramDistribution<kBT, kHD>());
        };
        auto w_dram_window_prefetch = make_w_dram_window(tb_init);
        auto u_dram_window_prefetch = make_tile_window(
            u_dram,
            make_tuple(number<kBT>{}, number<kVT>{}),
            multi_index<2>{tb_init, v_begin},
            MakeGdnAsyncDramDistribution<kBT, kVT, u_vector>());
        auto k_dram_window_prefetch = make_tile_window(
            k_dram,
            make_tuple(number<kBT>{}, number<kHD>{}),
            multi_index<2>{tb_init, 0},
            MakeGdnAsyncDramDistribution<kBT, kHD>());

        auto w_reg_prefetch = load_tile(w_dram_window_prefetch);
        auto u_reg_prefetch = load_tile(u_dram_window_prefetch);
        auto k_reg_prefetch = load_tile(k_dram_window_prefetch);

        // Main loop over chunks
        for (index_t lc = 0; lc < local_chunks; ++lc) {
            const index_t gc  = chunk_base + lc;
            const index_t tb      = args.is_varlen ? (bos + lc * kBT) : (lc * kBT);
            const index_t tb_abs  = bos + lc * kBT;
            const index_t vt      = ck_tile::min(kBT, eos - tb_abs);

            // Write the chunk-start state and stage its transpose for projection.
            const int64_t gc_vh_khd =
                (static_cast<int64_t>(gc) * args.num_value_heads + vh) * kHD;
            #pragma unroll
            for (index_t ii = 0; ii < kSPT; ++ii) {
                const index_t kd = kd_base + ii * kd_stride;
                smem.state_t_lds[vv * kHD_padded + kd] = type_convert<DataType>(s[ii]);
                if (v_begin + vv < kVD) {
                    const int64_t off = (gc_vh_khd + kd) * kVD + v_begin + vv;
                    args.h_start[off] = type_convert<DataType>(s[ii]);
                }
            }

            // Construct LDS Tile Windows
            auto w_lds_view = make_tensor_view<address_space_enum::lds>(
                smem.w_lds,
                MakeGdnSimpleLdsDescriptor<kBT, kHD, kLdsPad>());
            auto w_lds_window = make_tile_window(
                w_lds_view,
                make_tuple(number<kBT>{}, number<kHD>{}),
                multi_index<2>{0, 0});

            auto u_lds_view = make_tensor_view<address_space_enum::lds>(
                smem.u_lds,
                MakeGdnSimpleLdsDescriptor<kBT, kVT, kLdsPad>());
            auto u_lds_window = make_tile_window(
                u_lds_view,
                make_tuple(number<kBT>{}, number<kVT>{}),
                multi_index<2>{0, 0});

            // Store prefetched w and u into LDS
            store_tile(w_lds_window, w_reg_prefetch);
            store_tile(u_lds_window, u_reg_prefetch);

            if (lc + 1 < local_chunks) {
                const index_t tb_next = args.is_varlen ? (bos + (lc + 1) * kBT) : ((lc + 1) * kBT);
                w_dram_window_prefetch = make_w_dram_window(tb_next);
                w_reg_prefetch = load_tile(w_dram_window_prefetch);
                u_dram_window_prefetch = make_tile_window(
                    u_dram,
                    make_tuple(number<kBT>{}, number<kVT>{}),
                    multi_index<2>{tb_next, v_begin},
                    MakeGdnAsyncDramDistribution<kBT, kVT, u_vector>());
                u_reg_prefetch = load_tile(u_dram_window_prefetch);
            }

            float state_decay = 1.0f;

            // The fallback handles optional gate combinations at runtime.
            if (args.use_g) {
                const index_t last_tok = tb_abs + vt - 1;
                const float g_last = args.g[static_cast<int64_t>(last_tok) * args.num_value_heads + vh];
                state_decay = args.use_exp2
                                  ? fast_exp2(g_last)
                                  : fast_exp2(g_last * ck_tile::log2e_v<float>);
                if (tid < kBT) {
                    const index_t gt = tb_abs + tid;
                    if (gt < eos) {
                        float g_tok = args.g[static_cast<int64_t>(gt) * args.num_value_heads + vh];
                        float diff = g_last - g_tok;
                        smem.g_lds[tid] =
                            args.use_exp2 ? fast_exp2(diff)
                                          : fast_exp2(diff * ck_tile::log2e_v<float>);
                    } else {
                        smem.g_lds[tid] = 0.0f;
                    }
                }
            }
            if (args.use_gk && tid < kHD) {
                const index_t last_tok = tb_abs + vt - 1;
                const int64_t gk_off =
                    (static_cast<int64_t>(last_tok) * args.num_value_heads + vh)
                    * kHD + tid;
                float gk_val = args.gk[gk_off];
                smem.gk_decay_lds[tid] = args.use_exp2
                    ? fast_exp2(gk_val)
                    : fast_exp2(gk_val * ck_tile::log2e_v<float>);
            }

            __syncthreads();

            auto w_mmac_view = make_tensor_view<address_space_enum::lds>(
                smem.w_lds, MakeGdnSimpleLdsDescriptor<kBT, kHD, kLdsPad>());
            auto state_t_mmac_view = make_tensor_view<address_space_enum::lds>(
                smem.state_t_lds, MakeGdnSimpleLdsDescriptor<kVT, kHD, kLdsPad>());

            auto write_projection = [&](const auto& proj_out) {
                constexpr auto proj_spans = remove_cvref_t<decltype(proj_out)>::get_distributed_spans();
                DataType residual_values[kRPT];
                index_t residual_index = 0;
                const int64_t base_vh = static_cast<int64_t>(tb_abs) * args.num_value_heads + vh;
                ck_tile::sweep_tile_span(proj_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    ck_tile::sweep_tile_span(proj_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                            proj_out.get_tile_distribution(), tile_idx);
                        const index_t tok = x_idx.at(ck_tile::number<0>{});
                        const index_t vv_idx = x_idx.at(ck_tile::number<1>{});
                        DataType residual_t = type_convert<DataType>(0.0f);
                        if (tok < vt && v_begin + vv_idx < kVD) {
                            const int64_t u_off =
                                (base_vh + static_cast<int64_t>(tok) * args.num_value_heads)
                                * kVD + v_begin + vv_idx;
                            const index_t u_lin = tok * kVT_padded + vv_idx;
                            const float u_value = type_convert<float>(smem.u_lds[u_lin]);
                            const float res = u_value - proj_out[tile_idx];
                            if (args.save_new_value) {
                                args.v_new[u_off] = type_convert<DataType>(res);
                            }
                            const float gate = args.use_g ? smem.g_lds[tok] : 1.0f;
                            residual_t = type_convert<DataType>(res * gate);
                        }
                        if constexpr (kVT == 64) {
                            residual_values[residual_index++] = residual_t;
                        } else {
                            smem.residual_t_lds[vv_idx * kBT_padded + tok] = residual_t;
                        }
                    });
                });
                if constexpr (kVT == 64) {
                    __syncthreads();
                    residual_index = 0;
                    ck_tile::sweep_tile_span(proj_spans[ck_tile::number<0>{}], [&](auto idx0) {
                        ck_tile::sweep_tile_span(proj_spans[ck_tile::number<1>{}], [&](auto idx1) {
                            constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                            const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                                proj_out.get_tile_distribution(), tile_idx);
                            const index_t tok = x_idx.at(ck_tile::number<0>{});
                            const index_t vv_idx = x_idx.at(ck_tile::number<1>{});
                            smem.residual_t_lds[vv_idx * kBT_padded + tok] =
                                residual_values[residual_index++];
                        });
                    });
                }
            };

            auto w_mmac_window = make_tile_window(
                w_mmac_view,
                make_tuple(number<kBT>{}, number<kHD>{}),
                multi_index<2>{0, 0});
            auto state_t_mmac_window = make_tile_window(
                state_t_mmac_view,
                make_tuple(number<kVT>{}, number<kHD>{}),
                multi_index<2>{0, 0});

            auto proj_acc = decltype(proj_gemm.MakeCBlockTile()){};
            clear_tile(proj_acc);
            proj_gemm(proj_acc, w_mmac_window, state_t_mmac_window);
            const auto proj_out = proj_gemm.MakeOuputLayout(proj_acc);
            write_projection(proj_out);
            __syncthreads();

            // Apply decay to registers
            #pragma unroll
            for (index_t ii = 0; ii < kSPT; ++ii) {
                s[ii] *= state_decay;
                if (args.use_gk) {
                    const index_t kd = kd_base + ii * kd_stride;
                    s[ii] *= smem.gk_decay_lds[kd];
                }
            }

            auto write_k_transpose = [&](const auto& k_tile) {
                constexpr auto k_spans = remove_cvref_t<decltype(k_tile)>::get_distributed_spans();
                ck_tile::sweep_tile_span(k_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    ck_tile::sweep_tile_span(k_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                            k_tile.get_tile_distribution(), tile_idx);
                        const index_t tok = x_idx.at(ck_tile::number<0>{});
                        const index_t kd = x_idx.at(ck_tile::number<1>{});
                        smem.k_t_lds[kd * kBT_padded + tok] =
                            tok < vt ? k_tile[tile_idx] : type_convert<DataType>(0.0f);
                    });
                });
            };

            write_k_transpose(k_reg_prefetch);
            __syncthreads();

            if (lc + 1 < local_chunks) {
                const index_t tb_next = args.is_varlen ? (bos + (lc + 1) * kBT) : ((lc + 1) * kBT);
                k_dram_window_prefetch = make_tile_window(
                    k_dram,
                    make_tuple(number<kBT>{}, number<kHD>{}),
                    multi_index<2>{tb_next, 0},
                    MakeGdnAsyncDramDistribution<kBT, kHD>());
                k_reg_prefetch = load_tile(k_dram_window_prefetch);
            }

            auto k_t_mmac_view = make_tensor_view<address_space_enum::lds>(
                smem.k_t_lds, MakeGdnSimpleLdsDescriptor<kHD, kBT, kLdsPad>());
            auto residual_t_mmac_view = make_tensor_view<address_space_enum::lds>(
                smem.residual_t_lds, MakeGdnSimpleLdsDescriptor<kVT, kBT, kLdsPad>());
            auto residual_t_mmac_window = make_tile_window(
                residual_t_mmac_view,
                make_tuple(number<kVT>{}, number<kBT>{}),
                multi_index<2>{0, 0});

            auto k_t_mmac_window = make_tile_window(
                k_t_mmac_view,
                make_tuple(number<kHD>{}, number<kBT>{}),
                multi_index<2>{0, 0});

            auto update_acc = decltype(update_gemm.MakeCBlockTile()){};
            clear_tile(update_acc);
            update_gemm(update_acc, k_t_mmac_window, residual_t_mmac_window);
            const auto update_out = update_gemm.MakeOuputLayout(update_acc);
            constexpr auto update_spans = decltype(update_out)::get_distributed_spans();

            ck_tile::sweep_tile_span(update_spans[ck_tile::number<0>{}], [&](auto idx0) {
                ck_tile::sweep_tile_span(update_spans[ck_tile::number<1>{}], [&](auto idx1) {
                    constexpr auto tile_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x_idx = ck_tile::get_x_indices_from_distributed_indices(
                        update_out.get_tile_distribution(), tile_idx);
                    const index_t kd = x_idx.at(ck_tile::number<0>{});
                    const index_t vv_idx = x_idx.at(ck_tile::number<1>{});
                    smem.update_lds[kd * kVT_padded + vv_idx] = update_out[tile_idx];
                });
            });
            __syncthreads();

            // Add update to s
            #pragma unroll
            for (index_t ii = 0; ii < kSPT; ++ii) {
                const index_t kd = kd_base + ii * kd_stride;
                s[ii] += smem.update_lds[kd * kVT_padded + vv];
            }
        }

        if (args.store_final_state) {
            const int64_t seq_vh_khd =
                (static_cast<int64_t>(seq) * args.num_value_heads + vh) * kHD;
            #pragma unroll
            for (index_t ii = 0; ii < kSPT; ++ii) {
                const index_t kd = kd_base + ii * kd_stride;
                if (v_begin + vv < kVD) {
                    const int64_t off = (seq_vh_khd + kd) * kVD + v_begin + vv;
                    args.final_state[off] = s[ii];
                }
            }
        }
    }
};

} // namespace ck_tile
