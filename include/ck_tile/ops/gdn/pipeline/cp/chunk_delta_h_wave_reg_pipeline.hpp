// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/gdn/gdn_numeric.hpp"

#include <ck_tile/core.hpp>

#include "ck_tile/ops/gdn/block/chunk_delta_h_k_block_gemm.hpp"
#include "ck_tile/ops/gdn/block/chunk_delta_h_projection_block_gemm.hpp"
#include "ck_tile/ops/gdn/pipeline/chunk_delta_h_scan_policy.hpp"
#include "ck_tile/ops/gdn/pipeline/cp/chunk_delta_h_scan_policy.hpp"
#include "ck_tile/ops/gdn/kernel/chunk_delta_h_scan_kernel.hpp"

namespace ck_tile {

template <typename DataType>
struct ChunkDeltaHCpSummaryMmacImpl;

template <>
struct ChunkDeltaHCpSummaryMmacImpl<bf16_t>
{
    using Type = WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <>
struct ChunkDeltaHCpSummaryMmacImpl<half_t>
{
    using Type = WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <typename Traits,
          typename Policy,
          bool EnableTail = false,
          bool InitialFromHLocal = false,
          bool StoreFinalState = true,
          bool ZeroU = false,
          bool IdentityInit = false,
          bool StoreSummary = false,
          bool WriteChunkStates = true,
          bool WriteVNew = true,
          bool FusedSummaryAB = false,
          bool FinalStateSeqZero = false>
struct ChunkDeltaHCpSummaryPipeline
{
    using DataType = typename Traits::DataType;
    using Kargs = ChunkDeltaHScanFwdKargs<DataType>;
    using Mmac = typename ChunkDeltaHCpSummaryMmacImpl<DataType>::Type;
    using AVec = typename Mmac::AVecType;
    using BVec = typename Mmac::BVecType;
    using CVec = typename Mmac::CVecType;
    using KBlockGemm = ChunkDeltaHKUpdateBlockGemm<DataType, Mmac>;
    using ProjectionBlockGemm =
        ChunkDeltaHProjectionBlockGemm<DataType, Mmac>;

    static constexpr index_t kBT = Policy::kChunkSize;
    static constexpr index_t kHD = Policy::kHeadDim;
    static constexpr index_t kVD = Policy::kValueDim;
    static constexpr index_t kVT = Policy::kVTile;
    static constexpr index_t kWaveV = 16;
    // Four compute waves cover one 64-value grid-x slice and keep both affine
    // components, while four loader waves overlap the next LDS tile.
    static constexpr index_t kComputeWaves =
        FusedSummaryAB ? kVT / (2 * kWaveV) : kVT / kWaveV;
    static constexpr index_t kLoaderWaves = FusedSummaryAB ? 4 : 0;
    static constexpr index_t kFirstLoaderWave = kComputeWaves;
    static constexpr index_t kBlockWaves = kComputeWaves + kLoaderWaves;
    static constexpr index_t kLoaderRows =
        kLoaderWaves == 0 ? 0 : kBT / kLoaderWaves;
    static constexpr index_t kKTiles = kHD / 16;
    static constexpr index_t kTTiles = kBT / 16;
    static constexpr index_t kWKTile = 64;
    static constexpr index_t kWStages = kHD / kWKTile;
    static constexpr index_t kCopyVector = 8;
    // Twelve BF16 padding elements rotate consecutive rows by six 32-bit LDS
    // banks. This gives 16 distinct row phases while retaining 8-byte
    // alignment; it is the best measured layout for the fused summary.
    static constexpr index_t kLdsStride = kWKTile + 12;
    static constexpr index_t kWSmemElements = kBT * kLdsStride;

    static_assert(kBT == 64 && kHD == 128 &&
                  (kVT == 16 || kVT == 32 || kVT == 64 || kVT == 128));
    static_assert(Policy::kBlockSize == kBlockWaves * 64);
    static_assert(Policy::kSpecializeCommonFlags);
    static_assert(kHD % kWKTile == 0);
    static_assert(kWStages == 2);

    struct LdsStorage
    {
        alignas(16) DataType tile[kWSmemElements];
        alignas(16) DataType w0_tile[kWSmemElements];
        alignas(16) float g[kBT];
    };

    using WarpAttribute = WarpGemmAttributeMmacIterateK<Mmac, 1, 1, 1, 1, 1>;
    using CDistribution = remove_cvref_t<
        decltype(make_static_tile_distribution(typename WarpAttribute::CWarpDstrEncoding{}))>;
    using COutputDistribution = remove_cvref_t<
        decltype(make_static_tile_distribution(typename WarpAttribute::CWarpOutputDstrEncoding{}))>;
    using BDistribution = remove_cvref_t<
        decltype(make_static_tile_distribution(typename WarpAttribute::BWarpDstrEncoding{}))>;
    using ADistribution = remove_cvref_t<
        decltype(make_static_tile_distribution(typename WarpAttribute::AWarpDstrEncoding{}))>;
    using RowDistribution = remove_cvref_t<decltype(
        make_static_tile_distribution(tile_distribution_encoding<
                                      sequence<>,
                                      tuple<sequence<16>, sequence<4, 4>>,
                                      tuple<sequence<1, 2>>,
                                      tuple<sequence<0, 0>>,
                                      sequence<2>,
                                      sequence<1>>{}))>;
    using CTile = static_distributed_tensor<float, CDistribution>;
    using COutputTile = static_distributed_tensor<float, COutputDistribution>;
    using RowTile = static_distributed_tensor<float, RowDistribution>;
    using GDistribution = remove_cvref_t<decltype(
        tile_distribution_encoding_pattern_2d<64,
                                              4,
                                              16,
                                              1,
                                              tile_distribution_pattern::thread_raked,
                                              1>::make_2d_static_tile_distribution())>;
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

    CK_TILE_DEVICE static index_t value_wave_id()
    {
        if constexpr(FusedSummaryAB)
            return get_warp_id() & 3;
        else
            return get_warp_id();
    }

    CK_TILE_DEVICE static CVec make_identity_state(index_t kd_base,
                                                   index_t vv_base)
    {
        auto tile = make_static_distributed_tensor<float>(BDistribution{});
        constexpr auto spans = decltype(tile)::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto i0) {
            sweep_tile_span(spans[number<1>{}], [&](auto i1) {
                constexpr auto idx = make_tuple(i0, i1);
                const auto x_idx = get_x_indices_from_distributed_indices(
                    tile.get_tile_distribution(), idx);
                tile(idx) = kd_base + x_idx.at(number<0>{}) ==
                                   vv_base + x_idx.at(number<1>{})
                    ? 1.0f
                    : 0.0f;
            });
        });
        return tile.get_thread_buffer()
            .template get_as<CVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static CTile make_c_tile(const CVec& value)
    {
        auto tile = make_static_distributed_tensor<float>(CDistribution{});
        tile.get_thread_buffer().template set_as<CVec>(number<0>{}, value);
        return tile;
    }

    CK_TILE_DEVICE static CVec get_c_vector(const CTile& tile)
    {
        return tile.get_thread_buffer().template get_as<CVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static COutputTile make_c_output_tile(const CVec& value)
    {
        return WarpAttribute{}.MakeCOutputLayout(make_c_tile(value));
    }

    CK_TILE_DEVICE static CVec make_c_vector(const COutputTile& output)
    {
        return output.get_thread_buffer().template get_as<CVec>()[number<0>{}];
    }

    template <typename I0, typename I1>
    CK_TILE_DEVICE static uint32_t pack_c_pair(const CVec& c, I0 i0, I1 i1)
    {
        const uint32_t lo = bit_cast<uint16_t>(gdn_type_convert<DataType>(c_at(c, i0)));
        const uint32_t hi = bit_cast<uint16_t>(gdn_type_convert<DataType>(c_at(c, i1)));
        return lo | (hi << 16);
    }

    CK_TILE_DEVICE static DataType shuffle_packed_c(uint32_t c01,
                                                   uint32_t c23,
                                                   index_t src_lane,
                                                   index_t src_e)
    {
        const uint32_t remote01 = warp_shuffle(c01, src_lane);
        const uint32_t remote23 = warp_shuffle(c23, src_lane);
        const uint32_t packed = src_e < 2 ? remote01 : remote23;
        const uint16_t raw = static_cast<uint16_t>(packed >> ((src_e & 1) * 16));
        return bit_cast<DataType>(raw);
    }

    CK_TILE_DEVICE static BVec state_to_b(const CVec& c)
    {
        // Persistent state uses the projection B-operand lane order.  The update
        // GEMM produces the same physical values as a transposed, K-permuted C tile.
        union
        {
            CVec v;
            float e[4];
        } u{c};
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}(
            [&](auto e) { buf(e) = gdn_type_convert<DataType>(u.e[e]); });
        return buf.template get_as<BVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static AVec residual_to_a_transpose(const CVec& c)
    {
        const index_t lane = get_lane_id();
        const index_t v = lane & 15;
        const index_t t_group = (lane >> 4) & 3;
        const uint32_t c01 = pack_c_pair(c, number<0>{}, number<1>{});
        const uint32_t c23 = pack_c_pair(c, number<2>{}, number<3>{});
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t t = t_group * 4 + e;
            const index_t src_lane = t + 16 * (v & 3);
            buf(e) = shuffle_packed_c(c01, c23, src_lane, v >> 2);
        });
        return buf.template get_as<AVec>()[number<0>{}];
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_w_copy_distribution()
    {
        return tile_distribution_encoding_pattern_2d<Policy::kBlockSize,
                                                     kBT,
                                                     kWKTile,
                                                     kCopyVector,
                                                     tile_distribution_pattern::thread_raked,
                                                     1>::make_2d_static_tile_distribution();
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_loader_copy_distribution()
    {
        return LoaderCopyDistribution{};
    }

    CK_TILE_DEVICE static void block_sync_keep_vmem()
    {
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

    CK_TILE_HOST_DEVICE static constexpr auto make_u_copy_distribution()
    {
        return tile_distribution_encoding_pattern_2d<Policy::kBlockSize,
                                                     16,
                                                     kVT,
                                                     4,
                                                     tile_distribution_pattern::thread_raked,
                                                     1>::make_2d_static_tile_distribution();
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_w_lds_descriptor()
    {
        constexpr auto desc = make_naive_tensor_descriptor(
            make_tuple(number<4>{}, number<4>{}, number<4>{}, number<kWKTile>{}),
            make_tuple(number<16 * kLdsStride>{},
                       number<4 * kLdsStride>{},
                       number<kLdsStride>{},
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
            make_tuple(number<kLdsStride>{}, number<1>{}),
            number<1>{},
            number<1>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_k_b_lds_descriptor()
    {
        // K is transposed while entering LDS. The B operand consequently reads
        // four reduction elements from adjacent addresses, and the low K
        // coordinate swap preserves the C(V,K) to S(K,V) physical identity.
        constexpr auto desc = make_naive_tensor_descriptor(
            make_tuple(number<kBT>{}, number<4>{}, number<4>{}, number<4>{}),
            make_tuple(number<1>{},
                       number<16 * kLdsStride>{},
                       number<kLdsStride>{},
                       number<4 * kLdsStride>{}),
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
                       number<16 * kLdsStride>{},
                       number<kLdsStride>{},
                       number<4 * kLdsStride>{}),
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

    template <typename Stage>
    CK_TILE_DEVICE static auto load_w_dram_loader(const Kargs& args,
                                                  index_t token_base,
                                                  index_t vh,
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
            stage * kWKTile +
            static_cast<long_index_t>(row_begin) *
                args.num_value_heads * kHD;
        const auto w_dram =
            make_naive_tensor_view<address_space_enum::global>(
                w_base,
                make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
                make_tuple(args.num_value_heads * kHD, number<1>{}),
                number<8>{},
                number<1>{});
        const auto w_window = make_tile_window(
            w_dram,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            multi_index<2>{0, 0},
            make_loader_copy_distribution());
        return load_tile(w_window);
    }

    template <typename Stage>
    CK_TILE_DEVICE static auto load_k_dram_loader(const Kargs& args,
                                                  index_t token_base,
                                                  index_t qh,
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
            stage * kWKTile +
            static_cast<long_index_t>(row_begin) *
                args.num_qk_heads * kHD;
        const auto k_dram =
            make_naive_tensor_view<address_space_enum::global>(
                k_base,
                make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
                make_tuple(args.num_qk_heads * kHD, number<1>{}),
                number<8>{},
                number<1>{});
        const auto k_window = make_tile_window(
            k_dram,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            multi_index<2>{0, 0},
            make_loader_copy_distribution());
        return load_tile(k_window);
    }

    template <typename Tile>
    CK_TILE_DEVICE static void
    store_w_loader_lds(DataType* lds_ptr,
                       const Tile& tile,
                       index_t row_begin)
    {
        auto w_lds = make_tensor_view<address_space_enum::lds>(
            lds_ptr, make_w_lds_descriptor());
        auto w_window = make_tile_window(
            w_lds,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            multi_index<2>{row_begin, 0},
            make_loader_copy_distribution());
        store_tile(w_window, tile);
    }

    template <typename Tile>
    CK_TILE_DEVICE static void
    store_k_loader_lds(DataType* lds_ptr,
                       const Tile& tile,
                       index_t row_begin)
    {
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            lds_ptr, make_k_transposed_lds_descriptor());
        auto k_window = make_tile_window(
            k_lds,
            make_tuple(number<kLoaderRows>{}, number<kWKTile>{}),
            multi_index<2>{row_begin, 0},
            make_loader_copy_distribution());
        store_tile(k_window, tile);
    }

    template <typename Stage>
    CK_TILE_DEVICE static void stage_w_lds(DataType* lds_ptr,
                                           const Kargs& args,
                                           index_t token_base,
                                           index_t vh,
                                           index_t valid_tokens,
                                           Stage)
    {
        constexpr index_t stage = Stage::value;
        const auto* w_ptr =
            args.w +
            ((static_cast<long_index_t>(token_base) * args.num_value_heads + vh) * kHD) +
            stage * kWKTile;
        const auto w_dram = make_naive_tensor_view<address_space_enum::global>(
            w_ptr,
            make_tuple(valid_tokens, number<kWKTile>{}),
            make_tuple(args.num_value_heads * kHD, number<1>{}),
            number<8>{},
            number<1>{});
        auto w_dram_window = make_tile_window(
            w_dram,
            make_tuple(number<kBT>{}, number<kWKTile>{}),
            multi_index<2>{0, 0},
            make_w_copy_distribution());
        const auto w = load_tile(w_dram_window);

        auto w_lds =
            make_tensor_view<address_space_enum::lds>(lds_ptr, make_w_lds_descriptor());
        auto w_lds_window = make_tile_window(
            w_lds,
            make_tuple(number<kBT>{}, number<kWKTile>{}),
            multi_index<2>{0, 0},
            make_w_copy_distribution());
        store_tile(w_lds_window, w);
    }

    template <typename Stage>
    CK_TILE_DEVICE static void stage_k_lds(DataType* lds_ptr,
                                           const Kargs& args,
                                           index_t token_base,
                                           index_t qh,
                                           index_t valid_tokens,
                                           Stage)
    {
        constexpr index_t stage = Stage::value;
        const auto* k_ptr =
            args.k +
            ((static_cast<long_index_t>(token_base) * args.num_qk_heads + qh) * kHD) +
            stage * kWKTile;
        const auto k_dram = make_naive_tensor_view<address_space_enum::global>(
            k_ptr,
            make_tuple(valid_tokens, number<kWKTile>{}),
            make_tuple(args.num_qk_heads * kHD, number<1>{}),
            number<8>{},
            number<1>{});
        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<kBT>{}, number<kWKTile>{}),
            multi_index<2>{0, 0},
            make_w_copy_distribution());
        const auto k = load_tile(k_dram_window);

        auto k_lds = make_tensor_view<address_space_enum::lds>(
            lds_ptr, make_k_transposed_lds_descriptor());
        auto k_lds_window = make_tile_window(
            k_lds,
            make_tuple(number<kBT>{}, number<kWKTile>{}),
            multi_index<2>{0, 0},
            make_w_copy_distribution());
        store_tile(k_lds_window, k);
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
            make_tuple(number<1>{}, g_length),
            make_tuple(number<0>{}, args.num_value_heads),
            number<1>{},
            number<1>{});
        const auto g_dram_window = make_tile_window(
            g_dram,
            make_tuple(number<1>{}, number<kBT>{}),
            multi_index<2>{0, 0},
            GCopyDistribution{});
        auto g_values = load_tile(g_dram_window);
        if constexpr(PadTail)
        {
            const float g_last = warp_shuffle(
                g_values.get_thread_buffer()[number<0>{}], g_length - 1);
            if(get_lane_id() == kBT - 1)
                g_values.get_thread_buffer()[number<0>{}] = g_last;
        }

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

    CK_TILE_DEVICE static void load_g_lds(LdsStorage& smem,
                                          const Kargs& args,
                                          index_t token_base,
                                          index_t vh,
                                          index_t valid_tokens)
    {
        if(get_warp_id() != 0)
            return;

        if constexpr(EnableTail)
        {
            if(valid_tokens == kBT)
                load_g_lds_impl<false>(smem, args, token_base, vh, number<kBT>{});
            else
                load_g_lds_impl<true>(smem, args, token_base, vh, valid_tokens);
        }
        else
        {
            load_g_lds_impl<false>(smem, args, token_base, vh, number<kBT>{});
        }
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

    CK_TILE_DEVICE static GTile load_g_tile(const LdsStorage& smem, index_t t_tile)
    {
        const auto g_lds = make_naive_tensor_view<address_space_enum::lds>(
            const_cast<float*>(smem.g) + t_tile * 16,
            make_tuple(number<4>{}, number<16>{}),
            make_tuple(number<0>{}, number<1>{}),
            number<1>{},
            number<1>{});
        const auto g_window = make_tile_window(
            g_lds,
            make_tuple(number<4>{}, number<16>{}),
            multi_index<2>{0, 0},
            ADistribution{});
        return load_tile(g_window);
    }

    CK_TILE_DEVICE static RowTile projection_to_row(const CVec& projection)
    {
        const index_t lane = get_lane_id();
        const index_t t = lane >> 2;
        const index_t dst_group = lane & 3;
        const uint32_t c01 = pack_c_pair(projection, number<0>{}, number<1>{});
        const uint32_t c23 = pack_c_pair(projection, number<2>{}, number<3>{});
        auto row = make_static_distributed_tensor<float>(RowDistribution{});

        static_for<0, 4, 1>{}([&](auto n_lane) {
            row.get_thread_buffer()(n_lane) = type_convert<float>(
                shuffle_packed_c(c01, c23, n_lane * 16 + t, dst_group));
        });
        return row;
    }

    CK_TILE_DEVICE static auto load_u_direct(const Kargs& args,
                                             index_t token_base,
                                             index_t vh,
                                             index_t v_begin,
                                             index_t valid_tokens,
                                             index_t t_tile)
    {
        const index_t valid_tile_tokens =
            max(index_t{0}, min(index_t{16}, valid_tokens - t_tile * 16));
        const auto* u_base =
            args.u +
            ((static_cast<long_index_t>(token_base) * args.num_value_heads + vh) *
             kVD) +
            v_begin + value_wave_id() * kWaveV;
        if constexpr(EnableTail)
        {
            if(valid_tile_tokens != 16)
            {
                auto tile =
                    make_static_distributed_tensor<DataType>(ADistribution{});
                constexpr auto spans =
                    remove_cvref_t<decltype(tile)>::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto tile_idx = make_tuple(idx0, idx1);
                        const auto x_idx =
                            get_x_indices_from_distributed_indices(
                                tile.get_tile_distribution(), tile_idx);
                        const index_t token = x_idx.at(number<1>{});
                        tile(tile_idx) =
                            token < valid_tile_tokens
                                ? u_base[static_cast<long_index_t>(
                                             t_tile * 16 + token) *
                                             args.num_value_heads * kVD +
                                         x_idx.at(number<0>{})]
                                : gdn_type_convert<DataType>(0.0f);
                    });
                });
                return tile;
            }
        }

        const auto u_dram = make_naive_tensor_view<address_space_enum::global>(
            u_base + static_cast<long_index_t>(t_tile * 16) *
                         args.num_value_heads * kVD,
            make_tuple(number<16>{}, number<16>{}),
            make_tuple(number<1>{}, args.num_value_heads * kVD),
            number<1>{},
            number<1>{});
        const auto u_window = make_tile_window(
            u_dram,
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
        const index_t valid_tile_tokens =
            max(index_t{0}, min(index_t{16}, valid_tokens - t_tile * 16));
        auto v_dram = make_naive_tensor_view<address_space_enum::global>(
            args.v_new +
                ((static_cast<long_index_t>(token_base + t_tile * 16) *
                      args.num_value_heads +
                  vh) *
                     kVD) +
                v_begin + value_wave_id() * kWaveV,
            make_tuple(valid_tile_tokens, number<16>{}),
            make_tuple(args.num_value_heads * kVD, number<1>{}),
            number<4>{},
            number<1>{});
        auto v_window = make_tile_window(
            v_dram,
            make_tuple(number<16>{}, number<16>{}),
            multi_index<2>{0, 0},
            RowDistribution{});
        store_tile(v_window, value);
    }

    CK_TILE_DEVICE static AVec row_residual_to_a(const RowTile& residual)
    {
        const index_t lane = get_lane_id();
        const index_t v = lane & 15;
        const index_t t_group = lane >> 4;
        const uint32_t r01 =
            static_cast<uint32_t>(bit_cast<uint16_t>(gdn_type_convert<DataType>(
                residual.get_thread_buffer()[number<0>{}]))) |
            (static_cast<uint32_t>(bit_cast<uint16_t>(gdn_type_convert<DataType>(
                 residual.get_thread_buffer()[number<1>{}])))
             << 16);
        const uint32_t r23 =
            static_cast<uint32_t>(bit_cast<uint16_t>(gdn_type_convert<DataType>(
                residual.get_thread_buffer()[number<2>{}]))) |
            (static_cast<uint32_t>(bit_cast<uint16_t>(gdn_type_convert<DataType>(
                 residual.get_thread_buffer()[number<3>{}])))
             << 16);
        thread_buffer<DataType, 4> a;

        static_for<0, 4, 1>{}([&](auto e) {
            const index_t t = t_group * 4 + e;
            a(e) = shuffle_packed_c(r01, r23, t * 4 + (v >> 2), v & 3);
        });
        return a.template get_as<AVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static void stage_u_lds_tile(LdsStorage& smem,
                                                const Kargs& args,
                                                index_t token_base,
                                                index_t vh,
                                                index_t v_begin,
                                                index_t valid_tokens,
                                                index_t t_tile)
    {
        const index_t valid_tile_tokens =
            max(index_t{0}, min(index_t{16}, valid_tokens - t_tile * 16));
        const auto u_dram = make_naive_tensor_view<address_space_enum::global>(
            args.u +
                ((static_cast<long_index_t>(token_base + t_tile * 16) *
                      args.num_value_heads +
                  vh) *
                     kVD) +
                v_begin,
            make_tuple(valid_tile_tokens, number<kVT>{}),
            make_tuple(args.num_value_heads * kVD, number<1>{}),
            number<4>{},
            number<1>{});
        const auto u_dram_window = make_tile_window(
            u_dram,
            make_tuple(number<16>{}, number<kVT>{}),
            multi_index<2>{0, 0},
            make_u_copy_distribution());
        const auto u = load_tile(u_dram_window);
        auto u_lds =
            make_tensor_view<address_space_enum::lds>(smem.tile, make_w_lds_descriptor());
        auto u_lds_window = make_tile_window(
            u_lds,
            make_tuple(number<16>{}, number<kVT>{}),
            multi_index<2>{t_tile * 16, 0},
            make_u_copy_distribution());
        store_tile(u_lds_window, u);
    }

    CK_TILE_DEVICE static auto load_u_lds_tile(const LdsStorage& smem, index_t t_tile)
    {
        auto u_lds = make_tensor_view<address_space_enum::lds>(
            const_cast<DataType*>(smem.tile), make_w_lds_descriptor());
        auto u_lds_window = make_tile_window(
            u_lds,
            make_tuple(number<16>{}, number<16>{}),
            multi_index<2>{t_tile * 16, value_wave_id() * kWaveV},
            COutputDistribution{});
        return load_tile(u_lds_window);
    }

    template <typename VTile>
    CK_TILE_DEVICE static void store_v_new_lds_tile(LdsStorage& smem,
                                                    const VTile& value,
                                                    index_t t_tile)
    {
        auto v_lds =
            make_tensor_view<address_space_enum::lds>(smem.tile, make_w_lds_descriptor());
        auto v_lds_window = make_tile_window(
            v_lds,
            make_tuple(number<16>{}, number<16>{}),
            multi_index<2>{t_tile * 16, value_wave_id() * kWaveV},
            COutputDistribution{});
        store_tile(v_lds_window, gdn_cast_tile<DataType>(value));
    }

    CK_TILE_DEVICE static void store_v_new_block(LdsStorage& smem,
                                                 const Kargs& args,
                                                 index_t token_base,
                                                 index_t vh,
                                                 index_t v_begin,
                                                 index_t valid_tokens)
    {
        block_sync_lds();

        auto v_lds =
            make_tensor_view<address_space_enum::lds>(smem.tile, make_w_lds_descriptor());
        auto v_lds_window = make_tile_window(
            v_lds,
            make_tuple(number<kBT>{}, number<kVT>{}),
            multi_index<2>{0, 0},
            make_w_copy_distribution());
        const auto v_block = load_tile(v_lds_window);
        block_sync_lds();

        auto v_dram = make_naive_tensor_view<address_space_enum::global>(
            args.v_new +
                ((static_cast<long_index_t>(token_base) * args.num_value_heads + vh) * kVD) +
                v_begin,
            make_tuple(valid_tokens, number<kVT>{}),
            make_tuple(args.num_value_heads * kVD, number<1>{}),
            number<kCopyVector>{},
            number<1>{});
        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<kBT>{}, number<kVT>{}),
            multi_index<2>{0, 0},
            make_w_copy_distribution());
        store_tile(v_dram_window, v_block);
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

    CK_TILE_DEVICE static CVec load_local_state(const DataType* ptr,
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
        const auto state_tile = cast_tile<float>(load_tile(state_window));
        return state_tile.get_thread_buffer()
            .template get_as<CVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static void store_state_direct(DataType* ptr,
                                                  const BVec* state,
                                                  index_t v_begin)
    {
        const index_t vv_base = v_begin + value_wave_id() * kWaveV;
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
            auto state_tile =
                make_static_distributed_tensor<DataType>(BDistribution{});
            state_tile.get_thread_buffer().template set_as<BVec>(
                number<0>{}, state[kt]);
            store_tile(state_window, state_tile);
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
        if constexpr(FusedSummaryAB)
        {
            if(get_warp_id() >= kFirstLoaderWave)
            {
                const index_t row_begin =
                    (get_warp_id() - kFirstLoaderWave) * kLoaderRows;
                if(local_chunks > 0)
                {
                    auto loader_queue0 = load_w_dram_loader(
                        args, bos, vh, row_begin, number<0>{});
                    auto loader_queue1 = load_w_dram_loader(
                        args, bos, vh, row_begin, number<1>{});
                    store_w_loader_lds(
                        smem.w0_tile, loader_queue0, row_begin);
                    block_sync_keep_vmem();

                    for(index_t lc = 0; lc < local_chunks; ++lc)
                    {
                        const index_t token_base = bos + lc * kBT;
                        const bool has_next_chunk = lc + 1 < local_chunks;
                        const index_t next_token_base =
                            has_next_chunk ? token_base + kBT : token_base;

                        loader_queue0 = load_k_dram_loader(
                            args, token_base, qh, row_begin, number<0>{});
                        store_w_loader_lds(
                            smem.tile, loader_queue1, row_begin);
                        block_sync_keep_vmem();

                        loader_queue1 = load_k_dram_loader(
                            args, token_base, qh, row_begin, number<1>{});
                        store_k_loader_lds(
                            smem.w0_tile, loader_queue0, row_begin);
                        block_sync_keep_vmem();

                        if(has_next_chunk)
                        {
                            loader_queue0 = load_w_dram_loader(
                                args,
                                next_token_base,
                                vh,
                                row_begin,
                                number<0>{});
                        }
                        store_k_loader_lds(
                            smem.tile, loader_queue1, row_begin);
                        block_sync_keep_vmem();

                        if(has_next_chunk)
                        {
                            loader_queue1 = load_w_dram_loader(
                                args,
                                next_token_base,
                                vh,
                                row_begin,
                                number<1>{});
                            store_w_loader_lds(
                                smem.w0_tile, loader_queue0, row_begin);
                        }
                        block_sync_keep_vmem();
                    }
                }
                return;
            }
        }

        const index_t vv_base = v_begin + value_wave_id() * kWaveV;
        CVec state[kKTiles];
        BVec state_b[kKTiles];
        CVec affine_state[kKTiles];
        BVec affine_state_b[kKTiles];
        static_for<0, kKTiles, 1>{}([&](auto kt) {
            state[kt] = zero_c();
            if constexpr(FusedSummaryAB)
            {
                state[kt] = make_identity_state(kt * 16, vv_base);
                affine_state[kt] = zero_c();
                affine_state_b[kt] = state_to_b(affine_state[kt]);
            }
            else if constexpr(IdentityInit)
            {
                state[kt] = make_identity_state(kt * 16, vv_base);
            }
            else if constexpr(InitialFromHLocal)
            {
                const auto* initial =
                    args.h_local +
                    (static_cast<long_index_t>(seq) * args.num_value_heads + vh) *
                        kHD * kVD;
                state[kt] = load_local_state(initial, kt * 16, vv_base);
            }
            else if(args.has_initial_state)
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
            if constexpr(WriteChunkStates)
            {
                auto* first_chunk_state =
                    args.h_start +
                    (static_cast<long_index_t>(chunk_base) * args.num_value_heads + vh) *
                        kHD * kVD;
                store_state_direct(first_chunk_state, state_b, v_begin);
            }
            const index_t first_valid_tokens = min(kBT, eos - bos);
            if constexpr(FusedSummaryAB)
            {
                if(get_warp_id() == 0)
                    load_g_lds(
                        smem, args, bos, vh, first_valid_tokens);
                block_sync_keep_vmem();
            }
            else
            {
                stage_w_lds(
                    smem.w0_tile,
                    args,
                    bos,
                    vh,
                    first_valid_tokens,
                    number<0>{});
            }
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
            CVec affine_residual[kTTiles];
            AVec affine_residual_t[kTTiles];
            static_for<0, kTTiles, 1>{}([&](auto tt) {
                residual[tt] = zero_c();
                if constexpr(FusedSummaryAB)
                    affine_residual[tt] = zero_c();
            });
            auto compute_w_stage = [&](auto stage) {
                const DataType* w_lds_ptr =
                    stage == number<0>{} ? smem.w0_tile : smem.tile;
                auto w_lds = make_tensor_view<address_space_enum::lds>(
                    const_cast<DataType*>(w_lds_ptr),
                    make_w_projection_lds_descriptor());
                constexpr index_t state_offset =
                    decltype(stage)::value * (kWKTile / 16);
                ProjectionBlockGemm{}(
                    residual, state_b + state_offset, w_lds);
                if constexpr(FusedSummaryAB)
                {
                    if(lc != 0)
                        ProjectionBlockGemm{}(
                            affine_residual,
                            affine_state_b + state_offset,
                            w_lds);
                }
            };

            if constexpr(!FusedSummaryAB)
            {
                load_g_lds(smem, args, token_base, vh, valid_tokens);
                block_sync_lds_direct_load();
            }
            const float g_last = smem.g[kBT - 1];
            const float state_decay = __builtin_amdgcn_exp2f(g_last);

            using UTile = remove_cvref_t<decltype(load_u_direct(
                args, token_base, vh, v_begin, valid_tokens, index_t{0}))>;
            UTile u_prefetch[kTTiles];
            if constexpr(!ZeroU && !FusedSummaryAB)
            {
                static_for<0, kTTiles, 1>{}([&](auto tt) {
                    u_prefetch[tt] = load_u_direct(
                        args, token_base, vh, v_begin, valid_tokens, tt);
                });
            }

            compute_w_stage(number<0>{});
            if constexpr(FusedSummaryAB)
            {
                block_sync_keep_vmem();
                static_for<0, kTTiles, 1>{}([&](auto tt) {
                    u_prefetch[tt] = load_u_direct(
                        args,
                        token_base,
                        vh,
                        v_begin,
                        valid_tokens,
                        tt);
                });
                if(has_next_chunk && get_warp_id() == 0)
                    next_g_values =
                        load_g_values_full(args, token_base + kBT, vh);
            }
            else
            {
                stage_w_lds(
                    smem.tile,
                    args,
                    token_base,
                    vh,
                    valid_tokens,
                    number<1>{});
                block_sync_lds_direct_load();
            }
            compute_w_stage(number<1>{});
            if constexpr(FusedSummaryAB)
                block_sync_keep_vmem();
            else
                block_sync_lds();

            static_for<0, kTTiles, 1>{}([&](auto tt) {
                auto g_tile = load_g_tile(smem, tt);
                if constexpr(FusedSummaryAB)
                {
                    thread_buffer<DataType, 4> linear_update;
                    thread_buffer<DataType, 4> affine_update;
                    static_for<0, 4, 1>{}([&](auto e) {
                        const float decay = __builtin_amdgcn_exp2f(
                            g_last - g_tile.get_thread_buffer()(e));
                        linear_update(e) =
                            gdn_type_convert<DataType>(-c_at(residual[tt], e) * decay);
                        const float affine_value =
                            type_convert<float>(
                                u_prefetch[tt].get_thread_buffer()(e)) -
                            c_at(affine_residual[tt], e);
                        affine_update(e) =
                            gdn_type_convert<DataType>(affine_value * decay);
                    });
                    residual_t[tt] =
                        linear_update.template get_as<AVec>()[number<0>{}];
                    affine_residual_t[tt] =
                        affine_update.template get_as<AVec>()[number<0>{}];
                }
                else
                {
                    thread_buffer<DataType, 4> update;
                    static_for<0, 4, 1>{}([&](auto e) {
                        const float decay = __builtin_amdgcn_exp2f(
                            g_last - g_tile.get_thread_buffer()(e));
                        float value = -c_at(residual[tt], e);
                        if constexpr(!ZeroU)
                            value += type_convert<float>(
                                u_prefetch[tt].get_thread_buffer()(e));
                        update(e) = gdn_type_convert<DataType>(value * decay);
                    });
                    residual_t[tt] =
                        update.template get_as<AVec>()[number<0>{}];
                }
            });

            // K0 and K1 use the two LDS tiles left by the W projection.  Staging
            // both before the update avoids the mid-K overwrite/barrier cycle.
            if constexpr(!FusedSummaryAB)
            {
                stage_k_lds(
                    smem.w0_tile,
                    args,
                    token_base,
                    qh,
                    valid_tokens,
                    number<0>{});
                stage_k_lds(
                    smem.tile,
                    args,
                    token_base,
                    qh,
                    valid_tokens,
                    number<1>{});
            }

            static_for<0, kKTiles, 1>{}(
                [&](auto kt) {
                    transform_c(state[kt], [&](float x, auto) {
                        return x * state_decay;
                    });
                    if constexpr(FusedSummaryAB)
                    {
                        transform_c(affine_state[kt], [&](float x, auto) {
                            return x * state_decay;
                        });
                    }
                });

            auto compute_k_stage = [&](auto stage) {
                const DataType* k_lds_ptr =
                    stage == number<0>{} ? smem.w0_tile : smem.tile;
                auto k_lds = make_tensor_view<address_space_enum::lds>(
                    const_cast<DataType*>(k_lds_ptr), make_k_b_lds_descriptor());
                constexpr index_t state_offset = stage * (kWKTile / 16);
                KBlockGemm{}(state + state_offset, residual_t, k_lds);
                if constexpr(FusedSummaryAB)
                {
                    KBlockGemm{}(
                        affine_state + state_offset,
                        affine_residual_t,
                        k_lds);
                }
            };

            if constexpr(FusedSummaryAB)
            {
                compute_k_stage(number<0>{});
                block_sync_vmcnt7();
                compute_k_stage(number<1>{});
            }
            else
            {
                block_sync_lds_direct_load();
                compute_k_stage(number<0>{});
                compute_k_stage(number<1>{});
            }
            static_for<0, kKTiles, 1>{}([&](auto kt) {
                state_b[kt] = state_to_b(state[kt]);
                if constexpr(FusedSummaryAB)
                    affine_state_b[kt] = state_to_b(affine_state[kt]);
            });

            if(lc + 1 < local_chunks)
            {
                if constexpr(!FusedSummaryAB)
                {
                    // Prevent a leading wave from replacing K0 with the next
                    // chunk's W0 while another wave is still consuming K0.
                    block_sync_lds();
                    const index_t next_token_base = token_base + kBT;
                    const index_t next_valid_tokens =
                        min(kBT, eos - next_token_base);
                    stage_w_lds(smem.w0_tile,
                                args,
                                next_token_base,
                                vh,
                                next_valid_tokens,
                                number<0>{});
                }

                if constexpr(WriteChunkStates)
                {
                    auto* next_chunk_state =
                        args.h_start +
                        (static_cast<long_index_t>(gc + 1) * args.num_value_heads + vh) *
                            kHD * kVD;
                    store_state_direct(next_chunk_state, state_b, v_begin);
                }
            }
            if constexpr(FusedSummaryAB)
            {
                if(has_next_chunk && get_warp_id() == 0)
                    store_g_values_lds(smem, next_g_values);
                block_sync_keep_vmem();
            }
        }

        if constexpr(StoreSummary)
        {
            if constexpr(FusedSummaryAB)
            {
                auto* linear_summary =
                    args.h_start +
                    (static_cast<long_index_t>(seq) * args.num_value_heads + vh) *
                        kHD * kVD;
                auto* affine_summary =
                    args.h_local +
                    (static_cast<long_index_t>(seq) * args.num_value_heads + vh) *
                        kHD * kVD;
                store_state_direct(linear_summary, state_b, v_begin);
                store_state_direct(
                    affine_summary, affine_state_b, v_begin);
            }
            else
            {
                auto* summary =
                    args.h_start +
                    (static_cast<long_index_t>(seq) * args.num_value_heads + vh) *
                        kHD * kVD;
                store_state_direct(summary, state_b, v_begin);
            }
        }
        else if constexpr(StoreFinalState)
        {
            if(args.final_state != nullptr)
            {
                const index_t final_seq = FinalStateSeqZero ? 0 : seq;
                auto* final =
                    args.final_state +
                    (static_cast<long_index_t>(final_seq) * args.num_value_heads + vh) *
                        kHD * kVD;
                static_for<0, kKTiles, 1>{}([&](auto kt) {
                    store_final_state(final, state[kt], kt * 16, vv_base);
                });
            }
        }
    }
};

} // namespace ck_tile
