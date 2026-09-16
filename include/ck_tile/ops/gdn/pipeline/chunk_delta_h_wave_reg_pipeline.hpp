// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <ck_tile/core.hpp>

#include "ck_tile/ops/gdn/block/chunk_delta_h_k_block_gemm.hpp"
#include "ck_tile/ops/gdn/block/chunk_delta_h_projection_block_gemm.hpp"
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_scan_policy.hpp"

namespace ck_tile {

template <typename DataType>
struct ChunkDeltaHWaveMmacImpl;

template <>
struct ChunkDeltaHWaveMmacImpl<bf16_t>
{
    using Type = WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <>
struct ChunkDeltaHWaveMmacImpl<half_t>
{
    using Type = WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <typename Traits,
          typename Policy,
          bool PreshuffledH = false,
          bool FinalStateSeqZero = false>
struct ChunkDeltaHWaveRegPipeline
{
    using DataType = typename Traits::DataType;
    using Kargs = ChunkDeltaHScanFwdKargs<DataType>;
    using Mmac = typename ChunkDeltaHWaveMmacImpl<DataType>::Type;
    using AVec = typename Mmac::AVecType;
    using BVec = typename Mmac::BVecType;
    using CVec = typename Mmac::CVecType;
    using KBlockGemm = ChunkDeltaHKUpdateBlockGemm<DataType, Mmac>;
    using ProjectionBlockGemm = ChunkDeltaHProjectionBlockGemm<DataType, Mmac>;

    static constexpr index_t kBT = Policy::kChunkSize;
    static constexpr index_t kHD = Policy::kHeadDim;
    static constexpr index_t kVD = Policy::kValueDim;
    static constexpr index_t kVT = Policy::kVTile;
    static constexpr index_t kWaveV = 16;
    static constexpr index_t kComputeWaves = kVT / kWaveV;
    static constexpr index_t kFirstLoaderWave = kComputeWaves;
    static constexpr index_t kLoaderWaves = 4;
    static constexpr index_t kBlockWaves = kComputeWaves + kLoaderWaves;
    static constexpr index_t kLoaderRows = kBT / kLoaderWaves;
    static constexpr index_t kKTiles = kHD / 16;
    static constexpr index_t kTTiles = kBT / 16;
    static constexpr index_t kWKTile = 64;
    static constexpr index_t kWStages = kHD / kWKTile;
    static constexpr index_t kCopyVector = 8;
    static constexpr index_t kWLdsStride = kWKTile + 12;
    static constexpr index_t kKLdsStride = kWKTile + 12;
    static constexpr index_t kWSmemElements = kBT * kKLdsStride;
    static constexpr bool kUseFourLdsTiles = kVT == 128;
    static constexpr index_t kLdsTiles = kUseFourLdsTiles ? 4 : 2;

    static_assert(kBT == 64 && kHD == 128 &&
                  (kVT == 16 || kVT == 32 || kVT == 64 || kVT == 128));
    static_assert(Policy::kBlockSize == kBlockWaves * 64);
    static_assert(Policy::kSpecializeCommonFlags);
    static_assert(kHD % kWKTile == 0);
    static_assert(kWStages == 2);

    struct LdsStorage
    {
        // Replay128 keeps W0/W1/K0/K1 in four disjoint regions.  Smaller VTile
        // kernels, including Hybrid64, retain the original two rotating tiles.
        alignas(16) DataType tile[kLdsTiles][kWSmemElements];
        alignas(16) float g[kBT];
    };
    static_assert(kUseFourLdsTiles ? sizeof(LdsStorage) == 39168
                                  : sizeof(LdsStorage) == 19712);

    using WarpAttribute = WarpGemmAttributeMmacIterateK<Mmac, 1, 1, 1, 1, 1>;
    using BDistribution = remove_cvref_t<
        decltype(make_static_tile_distribution(typename WarpAttribute::BWarpDstrEncoding{}))>;
    using ADistribution = remove_cvref_t<
        decltype(make_static_tile_distribution(typename WarpAttribute::AWarpDstrEncoding{}))>;
    using GCopyDistribution = remove_cvref_t<decltype(
        tile_distribution_encoding_pattern_2d<64,
                                              1,
                                              64,
                                              1,
                                              tile_distribution_pattern::thread_raked,
                                              1>::make_2d_static_tile_distribution())>;
    using LoaderCopyDistribution = remove_cvref_t<decltype(
        make_static_tile_distribution(tile_distribution_encoding<
                                      sequence<>,
                                      tuple<sequence<8, 2>, sequence<8, 8>>,
                                      tuple<sequence<1, 2>>,
                                      tuple<sequence<0, 0>>,
                                      sequence<1, 2>,
                                      sequence<1, 1>>{}))>;
    using GTile = static_distributed_tensor<float, ADistribution>;
    using GValues = static_distributed_tensor<float, GCopyDistribution>;
    template <typename I>
    CK_TILE_DEVICE static float c_at(const CVec& c, I i)
    {
        union
        {
            CVec v;
            float e[4];
        } u{c};
        return u.e[i];
    }

    template <typename F>
    CK_TILE_DEVICE static void transform_c(CVec& c, F&& f)
    {
        union
        {
            CVec v;
            float e[4];
        } u{c};
        static_for<0, 4, 1>{}([&](auto i) { u.e[i] = f(u.e[i], i); });
        c = u.v;
    }

    CK_TILE_DEVICE static CVec zero_c()
    {
        return CVec{0.0f, 0.0f, 0.0f, 0.0f};
    }

    template <typename I0, typename I1>
    CK_TILE_DEVICE static uint32_t pack_c_pair(const CVec& c, I0 i0, I1 i1)
    {
        const uint32_t lo = bit_cast<uint16_t>(type_convert<DataType>(c_at(c, i0)));
        const uint32_t hi = bit_cast<uint16_t>(type_convert<DataType>(c_at(c, i1)));
        return lo | (hi << 16);
    }

    CK_TILE_DEVICE static BVec state_to_b(const CVec& c)
    {
        union
        {
            CVec v;
            float e[4];
        } u{c};
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}(
            [&](auto e) { buf(e) = type_convert<DataType>(u.e[e]); });
        return buf.template get_as<BVec>()[number<0>{}];
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_loader_copy_distribution()
    {
        // A warp-level distribution is required here: each producer is a
        // physical loader wave, but its local partition must still be lanes
        // 0..63. A nominal one-wave block distribution retains a block P
        // dimension and would use the physical warp id, addressing beyond both
        // the global and LDS tiles.
        return LoaderCopyDistribution{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_w_lds_descriptor()
    {
        // Store the inverse of the projection token permutation in LDS. This
        // keeps Vec8 writes aligned while distributing the eight rows touched
        // by each half-wave over all eight 4-bank phases of the padded stride.
        constexpr auto desc = make_naive_tensor_descriptor(
            make_tuple(number<4>{}, number<4>{}, number<4>{}, number<kWKTile>{}),
            make_tuple(number<16 * kWLdsStride>{},
                       number<4 * kWLdsStride>{},
                       number<kWLdsStride>{},
                       number<1>{}),
            number<kCopyVector>{},
            number<1>{});
        return transform_tensor_descriptor(
            desc,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<4>{}, number<4>{}, number<4>{})),
                make_pass_through_transform(number<kWKTile>{})),
            make_tuple(sequence<0, 2, 1>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_w_projection_lds_descriptor()
    {
        return make_naive_tensor_descriptor(
            make_tuple(number<kBT>{}, number<kWKTile>{}),
            make_tuple(number<kWLdsStride>{}, number<1>{}),
            number<1>{},
            number<1>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_k_b_lds_descriptor()
    {
        // K is transposed by the loader and stored as [N=K, K=T].  A B-operand
        // lane then reads its four reduction elements from consecutive LDS
        // addresses instead of four strided ds_read_u16 operations.  Swapping
        // the two low 2-bit K coordinates preserves the C(V,K)->S(K,V)
        // physical-layout identity used by the transposed update GEMM.
        constexpr auto desc = make_naive_tensor_descriptor(
            make_tuple(number<kBT>{}, number<4>{}, number<4>{}, number<4>{}),
            make_tuple(number<1>{},
                       number<16 * kKLdsStride>{},
                       number<kKLdsStride>{},
                       number<4 * kKLdsStride>{}),
            number<1>{},
            number<1>{});
        return transform_tensor_descriptor(
            desc,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<4>{}, number<4>{}, number<4>{})),
                make_pass_through_transform(number<kBT>{})),
            make_tuple(sequence<1, 3, 2>{}, sequence<0>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_k_transposed_lds_descriptor()
    {
        constexpr auto desc = make_naive_tensor_descriptor(
            make_tuple(number<kBT>{}, number<4>{}, number<4>{}, number<4>{}),
            make_tuple(number<1>{},
                       number<16 * kKLdsStride>{},
                       number<kKLdsStride>{},
                       number<4 * kKLdsStride>{}),
            number<1>{},
            number<1>{});
        return transform_tensor_descriptor(
            desc,
            make_tuple(
                make_pass_through_transform(number<kBT>{}),
                make_merge_transform(
                    make_tuple(number<4>{}, number<4>{}, number<4>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    CK_TILE_DEVICE static auto load_loader_tail(const DataType* ptr,
                                                index_t row_stride,
                                                index_t row_begin,
                                                index_t valid_rows)
    {
        auto tile = make_static_distributed_tensor<DataType>(
            make_loader_copy_distribution());
        constexpr auto spans = remove_cvref_t<decltype(tile)>::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto tile_idx = make_tuple(idx0, idx1);
                const auto x_idx = get_x_indices_from_distributed_indices(
                    tile.get_tile_distribution(), tile_idx);
                const index_t row = x_idx.at(number<0>{});
                tile(tile_idx) = row < valid_rows
                                     ? ptr[static_cast<long_index_t>(row_begin + row) *
                                               row_stride +
                                           x_idx.at(number<1>{})]
                                     : type_convert<DataType>(0.0f);
            });
        });
        return tile;
    }

    CK_TILE_DEVICE static void block_sync_keep_vmem()
    {
        // Publish the current LDS tile without draining the loader wave's
        // independent prefetch for the following producer stage.
        asm volatile("s_waitcnt lgkmcnt(0)\n"
                     "s_barrier\n"
                     ::: "memory");
    }

    CK_TILE_DEVICE static void block_sync_vmcnt7()
    {
        asm volatile("s_waitcnt vmcnt(7)\n"
                     "s_waitcnt lgkmcnt(0)\n"
                     "s_barrier\n"
                     ::: "memory");
    }

    CK_TILE_DEVICE static void wait_vmcnt7()
    {
        asm volatile("s_waitcnt vmcnt(7)\n" ::: "memory");
    }

    template <typename Stage>
    CK_TILE_DEVICE static auto load_w_dram_loader(const Kargs& args,
                                                  index_t token_base,
                                                  index_t vh,
                                                  index_t valid_tokens,
                                                  index_t row_begin,
                                                  Stage)
    {
        constexpr index_t stage = Stage::value;
        const auto* w_base =
            args.w +
            ((static_cast<long_index_t>(token_base) *
                  args.num_value_heads +
              vh) *
             kHD) +
            stage * kWKTile;
        if constexpr(Policy::kGuardTail)
        {
            const index_t valid_rows =
                max(index_t{0}, min(kLoaderRows, valid_tokens - row_begin));
            if(valid_rows != kLoaderRows)
            {
                return load_loader_tail(
                    w_base, args.num_value_heads * kHD, row_begin, valid_rows);
            }
        }
        const auto w_dram = make_naive_tensor_view<address_space_enum::global>(
            w_base + static_cast<long_index_t>(row_begin) *
                         args.num_value_heads * kHD,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            make_tuple(args.num_value_heads * kHD, number<1>{}),
            number<8>{},
            number<1>{});
        auto w_dram_window = make_tile_window(
            w_dram,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            multi_index<2>{0, 0},
            make_loader_copy_distribution());
        return load_tile(w_dram_window);
    }

    template <typename Tile>
    CK_TILE_DEVICE static void
    store_loader_lds(DataType* lds_ptr, const Tile& tile, index_t row_begin)
    {
        auto w_lds =
            make_tensor_view<address_space_enum::lds>(lds_ptr, make_w_lds_descriptor());
        auto w_lds_window = make_tile_window(
            w_lds,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            multi_index<2>{row_begin, 0},
            make_loader_copy_distribution());
        store_tile(w_lds_window, tile);
    }

    template <typename Tile>
    CK_TILE_DEVICE static void
    store_k_loader_lds(DataType* lds_ptr, const Tile& tile, index_t row_begin)
    {
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            lds_ptr, make_k_transposed_lds_descriptor());
        auto k_lds_window = make_tile_window(
            k_lds,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            multi_index<2>{row_begin, 0},
            make_loader_copy_distribution());
        store_tile(k_lds_window, tile);
    }

    template <typename Stage>
    CK_TILE_DEVICE static auto load_k_dram_loader(const Kargs& args,
                                                  index_t token_base,
                                                  index_t qh,
                                                  index_t valid_tokens,
                                                  index_t row_begin,
                                                  Stage)
    {
        constexpr index_t stage = Stage::value;
        const auto* k_base =
            args.k +
            ((static_cast<long_index_t>(token_base) *
                  args.num_qk_heads +
              qh) *
             kHD) +
            stage * kWKTile;
        if constexpr(Policy::kGuardTail)
        {
            const index_t valid_rows =
                max(index_t{0}, min(kLoaderRows, valid_tokens - row_begin));
            if(valid_rows != kLoaderRows)
            {
                return load_loader_tail(
                    k_base, args.num_qk_heads * kHD, row_begin, valid_rows);
            }
        }
        const auto k_dram = make_naive_tensor_view<address_space_enum::global>(
            k_base + static_cast<long_index_t>(row_begin) *
                         args.num_qk_heads * kHD,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            make_tuple(args.num_qk_heads * kHD, number<1>{}),
            number<8>{},
            number<1>{});
        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            multi_index<2>{0, 0},
            make_loader_copy_distribution());
        return load_tile(k_dram_window);
    }

    CK_TILE_DEVICE static GValues load_g_values_full(const Kargs& args,
                                                     index_t token_base,
                                                     index_t vh)
    {
        const auto g_dram = make_naive_tensor_view<address_space_enum::global>(
            args.g + static_cast<long_index_t>(token_base) * args.num_value_heads + vh,
            make_tuple(number<1>{}, number<kBT>{}),
            make_tuple(number<0>{}, args.num_value_heads),
            number<1>{},
            number<1>{});
        const auto g_dram_window = make_tile_window(
            g_dram,
            make_tuple(number<1>{}, number<kBT>{}),
            multi_index<2>{0, 0},
            GCopyDistribution{});
        return load_tile(g_dram_window);
    }

    CK_TILE_DEVICE static void store_g_values_lds(LdsStorage& smem,
                                                  const GValues& g_values)
    {
        auto g_lds = make_naive_tensor_view<address_space_enum::lds>(
            smem.g,
            make_tuple(number<1>{}, number<kBT>{}),
            make_tuple(number<kBT>{}, number<1>{}),
            number<1>{},
            number<1>{});
        auto g_lds_window = make_tile_window(
            g_lds,
            make_tuple(number<1>{}, number<kBT>{}),
            multi_index<2>{0, 0},
            GCopyDistribution{});
        store_tile(g_lds_window, g_values);
    }

    template <bool PadTail, typename GLength>
    CK_TILE_DEVICE static void load_g_lds_impl(LdsStorage& smem,
                                               const Kargs& args,
                                               index_t token_base,
                                               index_t vh,
                                               GLength g_length)
    {
        const auto g_dram = make_naive_tensor_view<address_space_enum::global>(
            args.g + static_cast<long_index_t>(token_base) * args.num_value_heads + vh,
            make_tuple(number<1>{}, number<kBT>{}),
            make_tuple(number<0>{}, args.num_value_heads),
            number<1>{},
            number<1>{});
        const auto g_dram_window = make_tile_window(
            g_dram,
            make_tuple(number<1>{}, number<kBT>{}),
            multi_index<2>{0, 0},
            GCopyDistribution{});
        auto g_values = [&]() {
            if constexpr(!PadTail)
            {
                return load_g_values_full(args, token_base, vh);
            }
            else
            {
                auto tile = make_static_distributed_tensor<float>(GCopyDistribution{});
                constexpr auto spans =
                    remove_cvref_t<decltype(tile)>::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = make_tuple(idx0, idx1);
                        const auto x_idx = get_x_indices_from_distributed_indices(
                            tile.get_tile_distribution(), tile_idx);
                        const index_t token = x_idx.at(number<1>{});
                        tile(tile_idx) = token < g_length
                                             ? args.g[static_cast<long_index_t>(token_base + token) *
                                                          args.num_value_heads +
                                                      vh]
                                             : 0.0f;
                    });
                });
                return tile;
            }
        }();
        if constexpr(PadTail)
        {
            const float g_last = warp_shuffle(
                g_values.get_thread_buffer()[number<0>{}], g_length - 1);
            if(get_lane_id() == kBT - 1)
            {
                g_values.get_thread_buffer()[number<0>{}] = g_last;
            }
        }

        store_g_values_lds(smem, g_values);
    }

    CK_TILE_DEVICE static void load_g_lds(LdsStorage& smem,
                                          const Kargs& args,
                                          index_t token_base,
                                          index_t vh,
                                          index_t valid_tokens)
    {
        if constexpr(!Policy::kGuardTail)
        {
            load_g_lds_impl<false>(smem, args, token_base, vh, number<kBT>{});
        }
        else
        {
            if(valid_tokens == kBT)
            {
                load_g_lds_impl<false>(smem, args, token_base, vh, number<kBT>{});
            }
            else
            {
                load_g_lds_impl<true>(smem, args, token_base, vh, valid_tokens);
            }
        }
    }

    CK_TILE_DEVICE static GTile load_g_tile(const LdsStorage& smem, index_t t_tile)
    {
        const auto g_lds = make_naive_tensor_view<address_space_enum::lds>(
            const_cast<float*>(smem.g) + t_tile * 16,
            make_tuple(number<16>{}, number<16>{}),
            make_tuple(number<0>{}, number<1>{}),
            number<1>{},
            number<1>{});
        const auto g_window = make_tile_window(
            g_lds,
            make_tuple(number<16>{}, number<16>{}),
            multi_index<2>{0, 0},
            ADistribution{});
        return load_tile(g_window);
    }

    CK_TILE_DEVICE static auto
    load_u_direct(const Kargs& args,
                  index_t token_base,
                  index_t vh,
                  index_t v_begin,
                  index_t valid_tokens,
                  index_t t_tile)
    {
        const auto* u_base =
            args.u +
            ((static_cast<long_index_t>(token_base) * args.num_value_heads + vh) * kVD) +
            v_begin + get_warp_id() * kWaveV;
        if constexpr(Policy::kGuardTail)
        {
            const index_t valid_tile_tokens =
                max(index_t{0}, min(index_t{16}, valid_tokens - t_tile * 16));
            if(valid_tile_tokens != 16)
            {
                auto tile = make_static_distributed_tensor<DataType>(ADistribution{});
                constexpr auto spans =
                    remove_cvref_t<decltype(tile)>::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = make_tuple(idx0, idx1);
                        const auto x_idx = get_x_indices_from_distributed_indices(
                            tile.get_tile_distribution(), tile_idx);
                        const index_t token = x_idx.at(number<1>{});
                        tile(tile_idx) = token < valid_tile_tokens
                                             ? u_base[static_cast<long_index_t>(t_tile * 16 + token) *
                                                          args.num_value_heads * kVD +
                                                      x_idx.at(number<0>{})]
                                             : type_convert<DataType>(0.0f);
                    });
                });
                return tile;
            }
        }
        const auto u_full_dram = make_naive_tensor_view<address_space_enum::global>(
            u_base + static_cast<long_index_t>(t_tile * 16) *
                         args.num_value_heads * kVD,
            make_tuple(number<16>{}, number<16>{}),
            make_tuple(number<1>{}, args.num_value_heads * kVD),
            number<1>{},
            number<1>{});
        const auto u_window = make_tile_window(
            u_full_dram,
            make_tuple(number<16>{}, number<16>{}),
            multi_index<2>{0, 0},
            ADistribution{});
        return load_tile(u_window);
    }

    template <typename VTile>
    CK_TILE_DEVICE static void store_v_new_direct(const Kargs& args,
                                                  index_t token_base,
                                                  index_t vh,
                                                  index_t v_begin,
                                                  index_t valid_tokens,
                                                  index_t t_tile,
                                                  const VTile& value)
    {
        auto* v_base =
            args.v_new +
            ((static_cast<long_index_t>(token_base) * args.num_value_heads + vh) * kVD) +
            v_begin + get_warp_id() * kWaveV;
        if constexpr(Policy::kGuardTail)
        {
            const index_t valid_tile_tokens =
                max(index_t{0}, min(index_t{16}, valid_tokens - t_tile * 16));
            if(valid_tile_tokens != 16)
            {
                constexpr auto spans =
                    remove_cvref_t<VTile>::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = make_tuple(idx0, idx1);
                        const auto x_idx = get_x_indices_from_distributed_indices(
                            value.get_tile_distribution(), tile_idx);
                        const index_t token = x_idx.at(number<1>{});
                        if(token < valid_tile_tokens)
                        {
                            v_base[static_cast<long_index_t>(t_tile * 16 + token) *
                                       args.num_value_heads * kVD +
                                   x_idx.at(number<0>{})] = value[tile_idx];
                        }
                    });
                });
                return;
            }
        }
        auto v_full_dram = make_naive_tensor_view<address_space_enum::global>(
            v_base + static_cast<long_index_t>(t_tile * 16) *
                         args.num_value_heads * kVD,
            make_tuple(number<16>{}, number<16>{}),
            make_tuple(number<1>{}, args.num_value_heads * kVD),
            number<1>{},
            number<1>{});
        auto v_window = make_tile_window(
            v_full_dram,
            make_tuple(number<16>{}, number<16>{}),
            multi_index<2>{0, 0},
            ADistribution{});
        store_tile(v_window, value);
    }

    CK_TILE_DEVICE static CVec load_state(const float* ptr,
                                          index_t kd_base,
                                          index_t vv_base)
    {
        const auto state_view = make_naive_tensor_view<address_space_enum::global>(
            ptr + static_cast<long_index_t>(kd_base) * kVD + vv_base,
            make_tuple(number<16>{}, number<16>{}),
            make_tuple(number<1>{}, number<kVD>{}),
            number<1>{},
            number<1>{});
        const auto state_window = make_tile_window(
            state_view,
            make_tuple(number<16>{}, number<16>{}),
            multi_index<2>{0, 0},
            BDistribution{});
        return load_tile(state_window)
            .get_thread_buffer()
            .template get_as<CVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static void store_state_direct(DataType* ptr,
                                                  const BVec* state,
                                                  index_t v_begin)
    {
        const index_t vv_base = v_begin + get_warp_id() * kWaveV;
        static_for<0, kKTiles, 1>{}([&](auto kt) {
            auto state_view = make_naive_tensor_view<address_space_enum::global>(
                ptr + static_cast<long_index_t>(kt * 16) * kVD + vv_base,
                make_tuple(number<16>{}, number<16>{}),
                make_tuple(number<1>{}, number<kVD>{}),
                number<1>{},
                number<1>{});
            auto state_window = make_tile_window(
                state_view,
                make_tuple(number<16>{}, number<16>{}),
                multi_index<2>{0, 0},
                BDistribution{});
            auto state_tile = make_static_distributed_tensor<DataType>(BDistribution{});
            state_tile.get_thread_buffer().template set_as<BVec>(
                number<0>{}, state[kt]);
            store_tile(state_window, state_tile);
        });
    }

    CK_TILE_DEVICE static void store_state_preshuffled(DataType* ptr,
                                                       const CVec* state,
                                                       index_t v_begin)
    {
        // Physical order consumed by fwd_output's native BReg loader:
        // [V/16, K/8, V_lane=16, K_vec=8].
        const index_t lane = get_lane_id();
        const index_t v_lane = lane & 15;
        const index_t k_lane = lane >> 4;
        const index_t wave_v = get_warp_id() * kWaveV;
        const index_t v = v_begin + wave_v + v_lane;
        const int32x4_t resource = make_wave_buffer_resource(ptr);

        static_for<0, kKTiles, 1>{}([&](auto kt) {
            thread_buffer<uint32_t, 2> output;
            output(number<0>{}) =
                pack_c_pair(state[kt], number<0>{}, number<1>{});
            output(number<1>{}) =
                pack_c_pair(state[kt], number<2>{}, number<3>{});
            const index_t k_base = kt * 16 + k_lane * 4;
            const index_t offset =
                (((v / 16) * 16 + k_base / 8) * 16 + v % 16) * 8 + k_base % 8;
            buffer_store<8>{}(output, resource, offset * sizeof(DataType), 0, 0);
        });
    }

    CK_TILE_DEVICE static void store_final_state(float* ptr,
                                                 const CVec& value,
                                                 index_t kd_base,
                                                 index_t vv_base)
    {
        auto state_view = make_naive_tensor_view<address_space_enum::global>(
            ptr + static_cast<long_index_t>(kd_base) * kVD + vv_base,
            make_tuple(number<16>{}, number<16>{}),
            make_tuple(number<1>{}, number<kVD>{}),
            number<1>{},
            number<1>{});
        auto state_window = make_tile_window(
            state_view,
            make_tuple(number<16>{}, number<16>{}),
            multi_index<2>{0, 0},
            BDistribution{});
        auto state_tile = make_static_distributed_tensor<float>(BDistribution{});
        state_tile.get_thread_buffer().template set_as<CVec>(number<0>{}, value);
        store_tile(state_window, state_tile);
    }

    CK_TILE_DEVICE void operator()(const Kargs& args,
                                   index_t seq,
                                   index_t vh,
                                   index_t qh,
                                   index_t v_begin,
                                   index_t bos,
                                   index_t eos,
                                   index_t chunk_base,
                                   index_t local_chunks,
                                   LdsStorage& smem) const
    {
        if(get_warp_id() >= kFirstLoaderWave)
        {
            const index_t row_begin =
                (get_warp_id() - kFirstLoaderWave) * kLoaderRows;
            if(local_chunks > 0)
            {
                const index_t first_valid_tokens = min(kBT, eos - bos);
                if constexpr(kUseFourLdsTiles)
                {
                    // Four-tile V128 schedule. Publish both W stages together,
                    // then both K stages, then both W stages for the next
                    // chunk. Disjoint W/K LDS regions make this a two-barrier
                    // steady-state loop without changing the smaller VTiles.
                    auto loader_queue0 = load_w_dram_loader(
                        args, bos, vh, first_valid_tokens, row_begin, number<0>{});
                    auto loader_queue1 = load_w_dram_loader(
                        args, bos, vh, first_valid_tokens, row_begin, number<1>{});
                    store_loader_lds(
                        smem.tile[0], loader_queue0, row_begin);
                    store_loader_lds(
                        smem.tile[1], loader_queue1, row_begin);
                    block_sync_keep_vmem();

                    for(index_t lc = 0; lc < local_chunks; ++lc)
                    {
                        const index_t token_base = bos + lc * kBT;
                        const index_t valid_tokens = min(kBT, eos - token_base);
                        const bool has_next_chunk = lc + 1 < local_chunks;
                        const index_t next_token_base =
                            has_next_chunk ? token_base + kBT : token_base;
                        const index_t next_valid_tokens =
                            has_next_chunk ? min(kBT, eos - next_token_base) : 0;

                        loader_queue0 = load_k_dram_loader(args,
                                                          token_base,
                                                          qh,
                                                          valid_tokens,
                                                          row_begin,
                                                          number<0>{});
                        loader_queue1 = load_k_dram_loader(args,
                                                          token_base,
                                                          qh,
                                                          valid_tokens,
                                                          row_begin,
                                                          number<1>{});
                        store_k_loader_lds(
                            smem.tile[2], loader_queue0, row_begin);
                        store_k_loader_lds(
                            smem.tile[3], loader_queue1, row_begin);
                        block_sync_keep_vmem();

                        if(has_next_chunk)
                        {
                            loader_queue0 = load_w_dram_loader(args,
                                                              next_token_base,
                                                              vh,
                                                              next_valid_tokens,
                                                              row_begin,
                                                              number<0>{});
                            loader_queue1 = load_w_dram_loader(args,
                                                              next_token_base,
                                                              vh,
                                                              next_valid_tokens,
                                                              row_begin,
                                                              number<1>{});
                            store_loader_lds(
                                smem.tile[0], loader_queue0, row_begin);
                            store_loader_lds(
                                smem.tile[1], loader_queue1, row_begin);
                        }
                        block_sync_keep_vmem();
                    }
                }
                else
                {
                    // Original two-tile producer schedule for VTile < 128.
                    auto loader_queue0 = load_w_dram_loader(
                        args, bos, vh, first_valid_tokens, row_begin, number<0>{});
                    auto loader_queue1 = load_w_dram_loader(
                        args, bos, vh, first_valid_tokens, row_begin, number<1>{});
                    store_loader_lds(
                        smem.tile[0], loader_queue0, row_begin);
                    block_sync_keep_vmem();

                    for(index_t lc = 0; lc < local_chunks; ++lc)
                    {
                        const index_t token_base = bos + lc * kBT;
                        const index_t valid_tokens = min(kBT, eos - token_base);
                        const bool has_next_chunk = lc + 1 < local_chunks;
                        const index_t next_token_base =
                            has_next_chunk ? token_base + kBT : token_base;
                        const index_t next_valid_tokens =
                            has_next_chunk ? min(kBT, eos - next_token_base) : 0;

                        loader_queue0 = load_k_dram_loader(
                            args,
                            token_base,
                            qh,
                            valid_tokens,
                            row_begin,
                            number<0>{});
                        store_loader_lds(
                            smem.tile[1], loader_queue1, row_begin);
                        block_sync_keep_vmem();

                        loader_queue1 = load_k_dram_loader(
                            args,
                            token_base,
                            qh,
                            valid_tokens,
                            row_begin,
                            number<1>{});
                        store_k_loader_lds(
                            smem.tile[0], loader_queue0, row_begin);
                        block_sync_keep_vmem();

                        if(has_next_chunk)
                        {
                            loader_queue0 = load_w_dram_loader(args,
                                                              next_token_base,
                                                              vh,
                                                              next_valid_tokens,
                                                              row_begin,
                                                              number<0>{});
                        }
                        store_k_loader_lds(
                            smem.tile[1], loader_queue1, row_begin);
                        block_sync_keep_vmem();

                        if(has_next_chunk)
                        {
                            loader_queue1 = load_w_dram_loader(args,
                                                              next_token_base,
                                                              vh,
                                                              next_valid_tokens,
                                                              row_begin,
                                                              number<1>{});
                            store_loader_lds(
                                smem.tile[0], loader_queue0, row_begin);
                        }
                        block_sync_keep_vmem();
                    }
                }
            }
            return;
        }

        const index_t vv_base = v_begin + get_warp_id() * kWaveV;
        CVec state[kKTiles];
        BVec state_b[kKTiles];
        static_for<0, kKTiles, 1>{}([&](auto kt) {
            state[kt] = zero_c();
            if(args.has_initial_state)
            {
                const auto* initial =
                    args.initial_state +
                    (static_cast<long_index_t>(seq) * args.num_value_heads + vh) * kHD * kVD;
                state[kt] = load_state(initial, kt * 16, vv_base);
            }
            state_b[kt] = state_to_b(state[kt]);
        });

        if(local_chunks > 0)
        {
            auto* first_chunk_state =
                args.h_start +
                (static_cast<long_index_t>(chunk_base) * args.num_value_heads + vh) *
                    kHD * kVD;
            if constexpr(PreshuffledH)
                store_state_preshuffled(first_chunk_state, state, v_begin);
            else
                store_state_direct(first_chunk_state, state_b, v_begin);
            if(get_warp_id() == 0)
            {
                load_g_lds(smem, args, bos, vh, min(kBT, eos - bos));
            }
            block_sync_lds_direct_load();
        }

        for(index_t lc = 0; lc < local_chunks; ++lc)
        {
            const index_t gc = chunk_base + lc;
            const index_t token_base = bos + lc * kBT;
            const index_t valid_tokens = min(kBT, eos - token_base);
            const bool has_next_chunk = lc + 1 < local_chunks;
            GValues next_g_values;
            CVec residual[kTTiles];
            AVec residual_t[kTTiles];
            static_for<0, kTTiles, 1>{}([&](auto tt) { residual[tt] = zero_c(); });
            auto compute_w_stage = [&](auto stage) {
                constexpr index_t buffer = decltype(stage)::value;
                auto w_lds = make_tensor_view<address_space_enum::lds>(
                    smem.tile[buffer], make_w_projection_lds_descriptor());
                constexpr index_t state_offset = stage * (kWKTile / 16);
                ProjectionBlockGemm{}(
                    residual, state_b + state_offset, w_lds);
            };

            const float g_last = smem.g[kBT - 1];
            const float state_decay = __builtin_amdgcn_exp2f(g_last);

            // U is wave-private and remains a direct HBM-to-VGPR transfer.
            // The loader waves prepare both K stages while W0/W1 are consumed.
            using UTile = remove_cvref_t<decltype(load_u_direct(
                args, token_base, vh, v_begin, valid_tokens, index_t{0}))>;
            UTile u_prefetch[kTTiles];

            compute_w_stage(number<0>{});
            if constexpr(!kUseFourLdsTiles)
            {
                // The original two-tile path publishes W1 after W0 releases
                // its LDS region. Four-tile Replay128 published both W stages
                // in the preceding barrier.
                block_sync_keep_vmem();
            }
            static_for<0, kTTiles, 1>{}([&](auto tt) {
                u_prefetch[tt] = load_u_direct(
                    args, token_base, vh, v_begin, valid_tokens, tt);
            });
            if constexpr(!Policy::kGuardTail)
            {
                if(has_next_chunk && get_warp_id() == 0)
                {
                    next_g_values =
                        load_g_values_full(args, token_base + kBT, vh);
                }
            }

            compute_w_stage(number<1>{});
            if constexpr(!kUseFourLdsTiles)
            {
                // Original two-tile B2: release W0 before it is reused for K0.
                block_sync_keep_vmem();
            }

            static_for<0, kTTiles, 1>{}([&](auto tt) {
                auto g_tile = load_g_tile(smem, tt);
                auto& u = u_prefetch[tt];
                auto v_new = make_static_distributed_tensor<DataType>(ADistribution{});
                thread_buffer<DataType, 4> update;
                static_for<0, 4, 1>{}([&](auto e) {
                    const float g_token = g_tile.get_thread_buffer()(e);
                    const float decay = __builtin_amdgcn_exp2f(g_last - g_token);
                    const float value = type_convert<float>(
                                            u.get_thread_buffer()(e)) -
                                        c_at(residual[tt], e);
                    v_new.get_thread_buffer()(e) = type_convert<DataType>(value);
                    update(e) = type_convert<DataType>(value * decay);
                });
                residual_t[tt] =
                    update.template get_as<AVec>()[number<0>{}];
                store_v_new_direct(
                    args, token_base, vh, v_begin, valid_tokens, tt, v_new);
            });

            static_for<0, kKTiles, 1>{}(
                [&](auto kt) {
                    transform_c(state[kt], [&](float x, auto) {
                        return x * state_decay;
                    });
                });

            if constexpr(kUseFourLdsTiles)
            {
                // V128 B1: both K stages are published. The producer may now
                // refill W0 while the compute waves consume the K tiles.
                block_sync_keep_vmem();
            }

            auto compute_k_stage = [&](auto stage) {
                constexpr index_t buffer = decltype(stage)::value;
                constexpr index_t lds_buffer =
                    kUseFourLdsTiles ? buffer + 2 : buffer;
                auto k_lds = make_tensor_view<address_space_enum::lds>(
                    const_cast<DataType*>(smem.tile[lds_buffer]),
                    make_k_b_lds_descriptor());
                constexpr index_t state_offset = stage * (kWKTile / 16);
                KBlockGemm{}(state + state_offset, residual_t, k_lds);
            };

            compute_k_stage(number<0>{});
            if constexpr(kUseFourLdsTiles)
            {
                // K1 was already published at B1; retain only the VMEM
                // throttle between the two K MMAC stages.
                wait_vmcnt7();
            }
            else
            {
                // Original two-tile B3 publishes K1 and releases K0.
                block_sync_vmcnt7();
            }
            compute_k_stage(number<1>{});
            if(has_next_chunk)
            {
                static_for<0, kKTiles, 1>{}(
                    [&](auto kt) { state_b[kt] = state_to_b(state[kt]); });
                auto* next_chunk_state =
                    args.h_start +
                    (static_cast<long_index_t>(gc + 1) * args.num_value_heads + vh) *
                        kHD * kVD;
                if constexpr(PreshuffledH)
                    store_state_preshuffled(next_chunk_state, state, v_begin);
                else
                    store_state_direct(next_chunk_state, state_b, v_begin);
            }
            if(has_next_chunk && get_warp_id() == 0)
            {
                if constexpr(Policy::kGuardTail)
                {
                    load_g_lds(smem,
                               args,
                               token_base + kBT,
                               vh,
                               min(kBT, eos - (token_base + kBT)));
                }
                else
                {
                    store_g_values_lds(smem, next_g_values);
                }
            }
            block_sync_lds_direct_load();

        }

        if(args.store_final_state)
        {
            const index_t final_seq = FinalStateSeqZero ? 0 : seq;
            auto* final =
                args.final_state +
                (static_cast<long_index_t>(final_seq) * args.num_value_heads + vh) *
                    kHD * kVD;
            static_for<0, kKTiles, 1>{}(
                [&](auto kt) { store_final_state(final, state[kt], kt * 16, vv_base); });
        }
    }
};

} // namespace ck_tile
