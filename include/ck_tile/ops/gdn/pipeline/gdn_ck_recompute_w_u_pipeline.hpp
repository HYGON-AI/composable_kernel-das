// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/gdn/gdn_numeric.hpp"
//
// Split GDN recompute_w_u CK pipeline.
//
// This file intentionally contains only CK tile building blocks. Do not add
// hand-written thread/warp scheduling here; express work with CK distributed
// tensors, tile windows, static_for/sweep helpers, and BlockGemm.

#include "ck_tile/ops/gdn/pipeline/gdn_ck_recompute_w_u_policy.hpp"
#include "ck_tile/ops/gdn/block/gdn_ck_recompute_w_u_block_gemm.hpp"

namespace ck_tile {

template <typename Policy>
struct GdnRecomputeWUPipeline
{
    static constexpr int kChunkSize = Policy::kChunkSize;
    static constexpr int kHeadDim   = Policy::kHeadDim;

    using DataType = typename Policy::DataType;
    using AccType  = float;
    static constexpr int kMWarps = 4;
    static constexpr int kWarpM  = 16;
    static constexpr int kWarpK  = std::is_same_v<DataType, bf16_t> ? 16 : 32;

    struct RecomputeGemmProblem
    {
        using ADataType      = DataType;
        using BDataType      = DataType;
        using CDataType      = AccType;
        using BlockGemmShape = TileGemmShape<
            sequence<64, 64, 64>,
            sequence<kMWarps, 1, 1>,
            sequence<kWarpM, 64, kWarpK>>;
        static constexpr index_t kBlockSize = Policy::kBlockSize;
    };
    using RecomputeBlockGemmPolicy =
        GdnRecomputeWUARegBSmemPolicy<sequence<kMWarps, 1, 1>, typename Policy::WarpGemm>;
    using RecomputeBlockGemm =
        GdnRecomputeWUBlockGemmARegBReg<RecomputeGemmProblem, RecomputeBlockGemmPolicy>;

    struct SharedStorage
    {
        DataType rhs_lds[kChunkSize * 64];
        float scale_lds[kChunkSize];
    };

    static constexpr int kRhsVec = 8;
    static constexpr int kRhsLoadsPerThread =
        (kChunkSize * 64 / kRhsVec) / Policy::kBlockSize;
    using RhsBuffer = thread_buffer<DataType, kRhsVec * kRhsLoadsPerThread>;

    CK_TILE_DEVICE static void block_sync_lds_relaxed()
    {
        block_sync_lds();
    }

    CK_TILE_DEVICE void stage_scale_to_lds(const float* __restrict__ beta_global,
                                           const float* __restrict__ g_global,
                                           int tc,
                                           int ivh,
                                           int stride_g_t,
                                           bool with_gate,
                                           bool use_exp2,
                                           SharedStorage& smem) const
    {
        const int tid = get_thread_id();
        if(tid < kChunkSize)
        {
            float scale = beta_global[(tc + tid) * stride_g_t + ivh];
            if(with_gate)
            {
                const float g = g_global[(tc + tid) * stride_g_t + ivh];
                scale *= use_exp2 ? __builtin_amdgcn_exp2f(g) : expf(g);
            }
            smem.scale_lds[tid] = scale;
        }
        block_sync_lds_relaxed();
    }

    CK_TILE_DEVICE void stage_scale_to_lds_masked(const float* __restrict__ beta_global,
                                                  const float* __restrict__ g_global,
                                                  int tc,
                                                  int ivh,
                                                  int stride_g_t,
                                                  int valid_rows,
                                                  bool with_gate,
                                                  bool use_exp2,
                                                  SharedStorage& smem) const
    {
        const int tid = get_thread_id();
        if(tid < kChunkSize)
        {
            float scale = 0.0f;
            if(tid < valid_rows)
            {
                scale = beta_global[(tc + tid) * stride_g_t + ivh];
                if(with_gate)
                {
                    const float g = g_global[(tc + tid) * stride_g_t + ivh];
                    scale *= use_exp2 ? __builtin_amdgcn_exp2f(g) : expf(g);
                }
            }
            smem.scale_lds[tid] = scale;
        }
        block_sync_lds_relaxed();
    }

    CK_TILE_DEVICE void stage_gate_to_lds(const float* __restrict__ g_global,
                                          int tc,
                                          int ivh,
                                          int stride_g_t,
                                          bool use_exp2,
                                          SharedStorage& smem) const
    {
        const int tid = get_thread_id();
        if(tid < kChunkSize)
        {
            const float g = g_global[(tc + tid) * stride_g_t + ivh];
            smem.scale_lds[tid] = use_exp2 ? __builtin_amdgcn_exp2f(g) : expf(g);
        }
        block_sync_lds_relaxed();
    }

    CK_TILE_DEVICE void stage_gate_to_lds_masked(const float* __restrict__ g_global,
                                                 int tc,
                                                 int ivh,
                                                 int stride_g_t,
                                                 int valid_rows,
                                                 bool use_exp2,
                                                 SharedStorage& smem) const
    {
        const int tid = get_thread_id();
        if(tid < kChunkSize)
        {
            float scale = 0.0f;
            if(tid < valid_rows)
            {
                const float g = g_global[(tc + tid) * stride_g_t + ivh];
                scale = use_exp2 ? __builtin_amdgcn_exp2f(g) : expf(g);
            }
            smem.scale_lds[tid] = scale;
        }
        block_sync_lds_relaxed();
    }

    CK_TILE_DEVICE auto load_a_tile(const DataType* __restrict__ A_global,
                                    int tc,
                                    int ivh,
                                    int stride_A_t) const
    {
        auto A_view = make_naive_tensor_view<address_space_enum::global>(
            A_global + tc * stride_A_t + ivh * kChunkSize,
            make_tuple(number<kChunkSize>{}, number<kChunkSize>{}),
            make_tuple(stride_A_t, number<1>{}),
            number<8>{},
            number<1>{});
        auto a_tile = RecomputeBlockGemm::MakeABlockTile();
        auto a_win = make_tile_window(
            A_view,
            make_tuple(number<kChunkSize>{}, number<kChunkSize>{}),
            multi_index<2>{0, 0},
            a_tile.get_tile_distribution());
        load_tile(a_tile, a_win);
        return a_tile;
    }

    CK_TILE_DEVICE auto load_a_tile_lower(const DataType* __restrict__ A_global,
                                          int tc,
                                          int ivh,
                                          int stride_A_t) const
    {
        auto A_view = make_naive_tensor_view<address_space_enum::global>(
            A_global + tc * stride_A_t + ivh * kChunkSize,
            make_tuple(number<kChunkSize>{}, number<kChunkSize>{}),
            make_tuple(stride_A_t, number<1>{}),
            number<8>{},
            number<1>{});
        auto a_tile = RecomputeBlockGemm::MakeABlockTile();
        clear_tile(a_tile);
        auto a_win = make_tile_window_linear(
            A_view,
            make_tuple(number<kChunkSize>{}, number<kChunkSize>{}),
            multi_index<2>{0, 0},
            a_tile.get_tile_distribution(),
            sequence<0, 1>{});
        static_for<0, remove_cvref_t<decltype(a_win)>::NumAccess, 1>{}(
            [&](auto i_access) {
                if(i_access <= get_warp_id())
                {
                    a_win.load(a_tile, i_access);
                }
            });
        return a_tile;
    }

    CK_TILE_DEVICE auto load_a_tile_masked(const DataType* __restrict__ A_global,
                                           int tc,
                                           int ivh,
                                           int stride_A_t,
                                           int valid_rows) const
    {
        auto A_view = make_naive_tensor_view<address_space_enum::global>(
            A_global + tc * stride_A_t + ivh * kChunkSize,
            make_tuple(valid_rows, number<kChunkSize>{}),
            make_tuple(stride_A_t, number<1>{}),
            number<8>{},
            number<1>{});
        auto a_tile = RecomputeBlockGemm::MakeABlockTile();
        auto a_win = make_tile_window(
            A_view,
            make_tuple(number<kChunkSize>{}, number<kChunkSize>{}),
            multi_index<2>{0, 0},
            a_tile.get_tile_distribution());
        load_tile(a_tile, a_win);
        if constexpr(std::is_same_v<DataType, half_t>)
        {
            constexpr auto spans = remove_cvref_t<decltype(a_tile)>::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto dstr_idx = make_tuple(idx0, idx1);
                    const auto tile_idx =
                        get_x_indices_from_distributed_indices(a_tile.get_tile_distribution(), dstr_idx);
                    if(tile_idx.at(number<0>{}) >= valid_rows ||
                       tile_idx.at(number<1>{}) >= valid_rows)
                        a_tile(dstr_idx) = DataType{0};
                });
            });
        }
        return a_tile;
    }

    template <typename ABlockTile>
    CK_TILE_DEVICE void scale_a_columns(ABlockTile& a_tile,
                                        SharedStorage& smem) const
    {
        const int k_end = (get_warp_id() + 1) * kWarpK;
        constexpr auto spans = remove_cvref_t<ABlockTile>::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(a_tile.get_tile_distribution(), dstr_idx);
                const int k = tile_idx.at(number<1>{});
                if(k < k_end)
                {
                    const float a = type_convert<float>(a_tile(dstr_idx));
                    a_tile(dstr_idx) = gdn_type_convert<DataType>(a * smem.scale_lds[k]);
                }
            });
        });
    }

    template <typename RhsView>
    CK_TILE_DEVICE auto load_rhs_as_b(const RhsView& rhs_view) const
    {
        auto rhs_t_view = make_naive_tensor_view<address_space_enum::lds>(
            rhs_view.get_buffer_view().p_data_,
            make_tuple(number<64>{}, number<kChunkSize>{}),
            make_tuple(number<1>{}, number<64>{}),
            number<1>{},
            number<1>{});
        auto b_win = make_tile_window(
            rhs_t_view,
            make_tuple(number<64>{}, number<kChunkSize>{}),
            multi_index<2>{0, 0},
            RecomputeBlockGemm::MakeBBlockTile().get_tile_distribution());
        return load_tile(b_win);
    }

    template <typename RhsView>
    CK_TILE_DEVICE auto load_rhs_as_b_dsreadm(const RhsView& rhs_view) const
    {
        auto rhs_t_view = make_naive_tensor_view<address_space_enum::lds>(
            rhs_view.get_buffer_view().p_data_,
            make_tuple(number<64>{}, number<kChunkSize>{}),
            make_tuple(number<1>{}, number<64>{}),
            number<1>{},
            number<1>{});
        return RecomputeBlockGemm{}.LoadBByDsreadm(rhs_t_view);
    }

    // FP16 and BF16 use different K iteration counts, but the same logical
    // C output distribution. Reuse the wave transpose only while that contract
    // holds. The caller keeps the barrier protecting the next RHS LDS write.
    template <typename OutTile>
    CK_TILE_DEVICE void store_output_tile(const OutTile& out,
                                          DataType* __restrict__ out_global,
                                          int tc,
                                          int ivh,
                                          int col,
                                          int stride_out_t,
                                          SharedStorage&,
                                          int valid_rows = kChunkSize) const
    {
        using Bf16Warp = typename GdnRecomputeWUWarpGemmSelector<bf16_t>::Type;
        static_assert(std::is_same_v<typename Policy::WarpGemm::CWarpOutputDstrEncoding,
                                     typename Bf16Warp::CWarpOutputDstrEncoding>);
        static_assert(OutTile::get_thread_buffer_size() == 16);
        store_output_tile_direct_masked(
            out, out_global, tc, ivh, col, stride_out_t, valid_rows);
    }

    template <int LaneXor>
    CK_TILE_DEVICE static int32_t wave_xor(int32_t x)
    {
        if constexpr(LaneXor == 16)
        {
            return __builtin_amdgcn_ds_swizzle(x, 0x401f);
        }
        else
        {
            return __builtin_amdgcn_ds_bpermute((get_lane_id() ^ LaneXor) << 2, x);
        }
    }

    CK_TILE_DEVICE static int32_t bit_select(int32_t on_false, int32_t on_true, int32_t mask)
    {
        return (on_false & ~mask) | (on_true & mask);
    }

    CK_TILE_DEVICE static void transpose_wave4_dwords(int32_t& x0,
                                                      int32_t& x1,
                                                      int32_t& x2,
                                                      int32_t& x3)
    {
        const int lane = get_lane_id();
        const int32_t lane_bit0_mask = -static_cast<int32_t>((lane >> 4) & 1);
        const int32_t lane_bit1_mask = -static_cast<int32_t>((lane >> 5) & 1);

        // Exchange register-index bit 0 with wave-lane bit 4.
        const int32_t t0 = bit_select(x0, wave_xor<16>(x1), lane_bit0_mask);
        const int32_t t1 = bit_select(wave_xor<16>(x0), x1, lane_bit0_mask);
        const int32_t t2 = bit_select(x2, wave_xor<16>(x3), lane_bit0_mask);
        const int32_t t3 = bit_select(wave_xor<16>(x2), x3, lane_bit0_mask);

        // Exchange register-index bit 1 with wave-lane bit 5.
        x0 = bit_select(t0, wave_xor<32>(t2), lane_bit1_mask);
        x1 = bit_select(t1, wave_xor<32>(t3), lane_bit1_mask);
        x2 = bit_select(wave_xor<32>(t0), t2, lane_bit1_mask);
        x3 = bit_select(wave_xor<32>(t1), t3, lane_bit1_mask);
    }

    template <typename OutTile>
    CK_TILE_DEVICE void store_output_tile_direct(const OutTile& out,
                                                      DataType* __restrict__ out_global,
                                                      int tc,
                                                      int ivh,
                                                      int col,
                                                      int stride_out_t) const
    {
        auto out_bf16 = gdn_cast_tile<DataType>(out);
        using Bf16x16 = ext_vector_t<DataType, 16>;
        using Dwordx8 = ext_vector_t<int32_t, 8>;
        using Dwordx4 = ext_vector_t<int32_t, 4>;
        using Vec     = ext_vector_t<DataType, 8>;

        const Bf16x16 values =
            out_bf16.get_thread_buffer().template get_as<Bf16x16>()[number<0>{}];
        const Dwordx8 packed = bit_cast<Dwordx8>(values);

        // For each half of a lane's strided output, transpose the four MMAC
        // column lanes so that every lane owns one contiguous 16-column segment.
        int32_t e0 = packed[0];
        int32_t e1 = packed[2];
        int32_t e2 = packed[4];
        int32_t e3 = packed[6];
        int32_t o0 = packed[1];
        int32_t o1 = packed[3];
        int32_t o2 = packed[5];
        int32_t o3 = packed[7];
        transpose_wave4_dwords(e0, e1, e2, e3);
        transpose_wave4_dwords(o0, o1, o2, o3);

        constexpr int32_t low_half  = 0x05040100;
        constexpr int32_t high_half = 0x07060302;
        const Dwordx4 out_lo{
            static_cast<int32_t>(__builtin_amdgcn_perm(e1, e0, low_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(e3, e2, low_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(e1, e0, high_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(e3, e2, high_half))};
        const Dwordx4 out_hi{
            static_cast<int32_t>(__builtin_amdgcn_perm(o1, o0, low_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(o3, o2, low_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(o1, o0, high_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(o3, o2, high_half))};

        const int tid       = get_thread_id();
        const int lane      = tid % get_warp_size();
        const int warp      = tid / get_warp_size();
        const int row       = warp * kWarpM + lane % 16;
        const int col_begin = col + (lane / 16) * 16;
        DataType* dst = out_global + (tc + row) * stride_out_t + ivh * kHeadDim + col_begin;
        *reinterpret_cast<Vec*>(dst)     = bit_cast<Vec>(out_lo);
        *reinterpret_cast<Vec*>(dst + 8) = bit_cast<Vec>(out_hi);
    }

    template <typename OutTile>
    CK_TILE_DEVICE void store_output_tile_direct_masked(
        const OutTile& out,
        DataType* __restrict__ out_global,
        int tc,
        int ivh,
        int col,
        int stride_out_t,
        int valid_rows) const
    {
        auto out_bf16 = gdn_cast_tile<DataType>(out);
        using Bf16x16 = ext_vector_t<DataType, 16>;
        using Dwordx8 = ext_vector_t<int32_t, 8>;
        using Dwordx4 = ext_vector_t<int32_t, 4>;
        using Vec     = ext_vector_t<DataType, 8>;

        const Bf16x16 values =
            out_bf16.get_thread_buffer().template get_as<Bf16x16>()[number<0>{}];
        const Dwordx8 packed = bit_cast<Dwordx8>(values);
        int32_t e0 = packed[0];
        int32_t e1 = packed[2];
        int32_t e2 = packed[4];
        int32_t e3 = packed[6];
        int32_t o0 = packed[1];
        int32_t o1 = packed[3];
        int32_t o2 = packed[5];
        int32_t o3 = packed[7];
        transpose_wave4_dwords(e0, e1, e2, e3);
        transpose_wave4_dwords(o0, o1, o2, o3);

        constexpr int32_t low_half  = 0x05040100;
        constexpr int32_t high_half = 0x07060302;
        const Dwordx4 out_lo{
            static_cast<int32_t>(__builtin_amdgcn_perm(e1, e0, low_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(e3, e2, low_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(e1, e0, high_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(e3, e2, high_half))};
        const Dwordx4 out_hi{
            static_cast<int32_t>(__builtin_amdgcn_perm(o1, o0, low_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(o3, o2, low_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(o1, o0, high_half)),
            static_cast<int32_t>(__builtin_amdgcn_perm(o3, o2, high_half))};

        const int tid       = get_thread_id();
        const int lane      = tid % get_warp_size();
        const int warp      = tid / get_warp_size();
        const int row       = warp * kWarpM + lane % 16;
        const int col_begin = col + (lane / 16) * 16;
        if(row < valid_rows)
        {
            DataType* dst =
                out_global + (tc + row) * stride_out_t + ivh * kHeadDim + col_begin;
            *reinterpret_cast<Vec*>(dst)     = bit_cast<Vec>(out_lo);
            *reinterpret_cast<Vec*>(dst + 8) = bit_cast<Vec>(out_hi);
        }
    }

    CK_TILE_DEVICE RhsBuffer load_rhs_global(const DataType* __restrict__ rhs_global,
                                             int stride_rhs_t,
                                             int col) const
    {
        using Vec = ext_vector_t<DataType, kRhsVec>;
        const int tid = get_thread_id();
        RhsBuffer rhs_buf;

        static_for<0, kRhsLoadsPerThread, 1>{}([&](auto i_load) {
            const int i = tid + i_load * Policy::kBlockSize;
            const int r = i / (64 / kRhsVec);
            const int c = (i - r * (64 / kRhsVec)) * kRhsVec;
            rhs_buf.template get_as<Vec>()(i_load) =
                *reinterpret_cast<const Vec*>(rhs_global + r * stride_rhs_t + col + c);
        });
        return rhs_buf;
    }

    CK_TILE_DEVICE RhsBuffer load_rhs_global_masked(
        const DataType* __restrict__ rhs_global,
        int stride_rhs_t,
        int col,
        int valid_rows) const
    {
        using Vec = ext_vector_t<DataType, kRhsVec>;
        const int tid = get_thread_id();
        RhsBuffer rhs_buf;

        static_for<0, kRhsLoadsPerThread, 1>{}([&](auto i_load) {
            const int i = tid + i_load * Policy::kBlockSize;
            const int r = i / (64 / kRhsVec);
            const int c = (i - r * (64 / kRhsVec)) * kRhsVec;
            rhs_buf.template get_as<Vec>()(i_load) =
                r < valid_rows
                    ? *reinterpret_cast<const Vec*>(rhs_global + r * stride_rhs_t + col + c)
                    : Vec{};
        });
        return rhs_buf;
    }

    CK_TILE_DEVICE auto stage_scale_rhs_lds_view(const RhsBuffer& rhs_buf,
                                                 SharedStorage& smem) const
    {
        auto rhs_view = make_naive_tensor_view<address_space_enum::lds>(
            smem.rhs_lds,
            make_tuple(number<kChunkSize>{}, number<64>{}),
            make_tuple(number<64>{}, number<1>{}),
            number<1>{},
            number<1>{});

        using Vec = ext_vector_t<DataType, kRhsVec>;
        const int tid = get_thread_id();

        static_for<0, kRhsLoadsPerThread, 1>{}([&](auto i_load) {
            const int i = tid + i_load * Policy::kBlockSize;
            const int r = i / (64 / kRhsVec);
            const int c = (i - r * (64 / kRhsVec)) * kRhsVec;
            thread_buffer<DataType, kRhsVec> out_buf;
            static_for<0, kRhsVec, 1>{}([&](auto j) {
                const float rhs = type_convert<float>(rhs_buf[i_load * kRhsVec + j]);
                out_buf(j) = gdn_type_convert<DataType>(rhs * smem.scale_lds[r]);
            });
            *reinterpret_cast<Vec*>(smem.rhs_lds + r * 64 + c) =
                out_buf.template get_as<Vec>()[number<0>{}];
        });
        block_sync_lds_relaxed();
        return rhs_view;
    }

    CK_TILE_DEVICE auto stage_scale_rhs_to_lds(const RhsBuffer& rhs_buf,
                                               SharedStorage& smem) const
    {
        auto rhs_view = stage_scale_rhs_lds_view(rhs_buf, smem);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            return load_rhs_as_b_dsreadm(rhs_view);
        }
        else
        {
            return load_rhs_as_b(rhs_view);
        }
    }

    CK_TILE_DEVICE auto stage_rhs_lds_view(const RhsBuffer& rhs_buf,
                                           SharedStorage& smem) const
    {
        auto rhs_view = make_naive_tensor_view<address_space_enum::lds>(
            smem.rhs_lds,
            make_tuple(number<kChunkSize>{}, number<64>{}),
            make_tuple(number<64>{}, number<1>{}),
            number<1>{},
            number<1>{});

        using Vec = ext_vector_t<DataType, kRhsVec>;
        const int tid = get_thread_id();
        static_for<0, kRhsLoadsPerThread, 1>{}([&](auto i_load) {
            const int i = tid + i_load * Policy::kBlockSize;
            const int r = i / (64 / kRhsVec);
            const int c = (i - r * (64 / kRhsVec)) * kRhsVec;
            *reinterpret_cast<Vec*>(smem.rhs_lds + r * 64 + c) =
                rhs_buf.template get_as<Vec>()(i_load);
        });
        block_sync_lds_relaxed();
        return rhs_view;
    }

    CK_TILE_DEVICE auto stage_rhs_to_lds(const RhsBuffer& rhs_buf,
                                         SharedStorage& smem) const
    {
        auto rhs_view = stage_rhs_lds_view(rhs_buf, smem);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            return load_rhs_as_b_dsreadm(rhs_view);
        }
        else
        {
            return load_rhs_as_b(rhs_view);
        }
    }

    template <typename ABlockTile>
    CK_TILE_DEVICE void compute_rhs_buffer(const ABlockTile& a_tile,
                                           const RhsBuffer& rhs_buf,
                                           DataType* __restrict__ out_global,
                                           int tc,
                                           int ivh,
                                           int stride_out_t,
                                           int col,
                                           SharedStorage& smem) const
    {
        constexpr auto bg = RecomputeBlockGemm{};
        auto c = RecomputeBlockGemm::MakeCBlockTile();
        clear_tile(c);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            auto rhs_view = stage_scale_rhs_lds_view(rhs_buf, smem);
            bg.RunWithStreamingDsreadmB(c, a_tile, rhs_view);
        }
        else
        {
            auto b_tile = stage_scale_rhs_to_lds(rhs_buf, smem);
            bg(c, a_tile, b_tile);
        }
        auto out = bg.MakeOuputLayout(c);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            store_output_tile_direct(out, out_global, tc, ivh, col, stride_out_t);
        }
        else
        {
            store_output_tile(out, out_global, tc, ivh, col, stride_out_t, smem);
            wg_sync_lds(bool_constant<true>{});
        }
    }

    template <typename ABlockTile>
    CK_TILE_DEVICE void compute_rhs_buffer_masked(const ABlockTile& a_tile,
                                                  const RhsBuffer& rhs_buf,
                                                  DataType* __restrict__ out_global,
                                                  int tc,
                                                  int ivh,
                                                  int stride_out_t,
                                                  int col,
                                                  int valid_rows,
                                                  SharedStorage& smem) const
    {
        constexpr auto bg = RecomputeBlockGemm{};
        auto c = RecomputeBlockGemm::MakeCBlockTile();
        clear_tile(c);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            auto rhs_view = stage_scale_rhs_lds_view(rhs_buf, smem);
            bg.RunWithStreamingDsreadmB(c, a_tile, rhs_view);
        }
        else
        {
            auto b_tile = stage_scale_rhs_to_lds(rhs_buf, smem);
            bg(c, a_tile, b_tile);
        }
        auto out = bg.MakeOuputLayout(c);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            store_output_tile_direct_masked(
                out, out_global, tc, ivh, col, stride_out_t, valid_rows);
        }
        else
        {
            store_output_tile(out, out_global, tc, ivh, col, stride_out_t,
                              smem, valid_rows);
            wg_sync_lds(bool_constant<true>{});
        }
    }

    template <bool SyncLdsAfter = false, typename ABlockTile>
    CK_TILE_DEVICE void compute_rhs_buffer_unscaled(const ABlockTile& a_tile,
                                                    const RhsBuffer& rhs_buf,
                                                    DataType* __restrict__ out_global,
                                                    int tc,
                                                    int ivh,
                                                    int stride_out_t,
                                                    int col,
                                                    SharedStorage& smem) const
    {
        constexpr auto bg = RecomputeBlockGemm{};
        auto c = RecomputeBlockGemm::MakeCBlockTile();
        clear_tile(c);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            auto rhs_view = stage_rhs_lds_view(rhs_buf, smem);
            bg.RunLowerWithPrefetch2DsreadmB(c, a_tile, rhs_view);
        }
        else
        {
            auto b_tile = stage_rhs_to_lds(rhs_buf, smem);
            bg(c, a_tile, b_tile);
        }
        if constexpr(SyncLdsAfter)
        {
            block_sync_lds_relaxed();
        }
        auto out = bg.MakeOuputLayout(c);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            store_output_tile_direct(out, out_global, tc, ivh, col, stride_out_t);
        }
        else
        {
            store_output_tile(out, out_global, tc, ivh, col, stride_out_t, smem);
            wg_sync_lds(bool_constant<true>{});
        }
    }

    template <bool SyncLdsAfter = false, typename ABlockTile>
    CK_TILE_DEVICE void compute_rhs_buffer_unscaled_masked(
        const ABlockTile& a_tile,
        const RhsBuffer& rhs_buf,
        DataType* __restrict__ out_global,
        int tc,
        int ivh,
        int stride_out_t,
        int col,
        int valid_rows,
        SharedStorage& smem) const
    {
        constexpr auto bg = RecomputeBlockGemm{};
        auto c = RecomputeBlockGemm::MakeCBlockTile();
        clear_tile(c);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            auto rhs_view = stage_rhs_lds_view(rhs_buf, smem);
            bg.RunLowerWithPrefetch2DsreadmB(c, a_tile, rhs_view);
        }
        else
        {
            auto b_tile = stage_rhs_to_lds(rhs_buf, smem);
            bg(c, a_tile, b_tile);
        }
        if constexpr(SyncLdsAfter)
        {
            block_sync_lds_relaxed();
        }
        auto out = bg.MakeOuputLayout(c);
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            store_output_tile_direct_masked(
                out, out_global, tc, ivh, col, stride_out_t, valid_rows);
        }
        else
        {
            store_output_tile(out, out_global, tc, ivh, col, stride_out_t,
                              smem, valid_rows);
            wg_sync_lds(bool_constant<true>{});
        }
    }

    template <typename ABlockTile>
    CK_TILE_DEVICE void compute_rhs_pair_unscaled(const ABlockTile& a_tile,
                                                  const RhsBuffer& rhs_col0,
                                                  const RhsBuffer& rhs_col1,
                                                  DataType* __restrict__ out_global,
                                                  int tc,
                                                  int ivh,
                                                  int stride_out_t,
                                                  SharedStorage& smem) const
    {
        static_for<0, 2, 1>{}([&](auto i_col) {
            constexpr int col = i_col * 64;
            if constexpr(i_col == 0)
            {
                compute_rhs_buffer_unscaled<true>(
                    a_tile, rhs_col0, out_global, tc, ivh, stride_out_t, col, smem);
            }
            else
            {
                compute_rhs_buffer_unscaled(
                    a_tile, rhs_col1, out_global, tc, ivh, stride_out_t, col, smem);
            }
        });
    }

    template <typename ABlockTile>
    CK_TILE_DEVICE void compute_rhs_pair_unscaled_masked(
        const ABlockTile& a_tile,
        const RhsBuffer& rhs_col0,
        const RhsBuffer& rhs_col1,
        DataType* __restrict__ out_global,
        int tc,
        int ivh,
        int stride_out_t,
        int valid_rows,
        SharedStorage& smem) const
    {
        compute_rhs_buffer_unscaled_masked<true>(
            a_tile,
            rhs_col0,
            out_global,
            tc,
            ivh,
            stride_out_t,
            0,
            valid_rows,
            smem);
        compute_rhs_buffer_unscaled_masked(
            a_tile,
            rhs_col1,
            out_global,
            tc,
            ivh,
            stride_out_t,
            64,
            valid_rows,
            smem);
    }

    template <typename ABlockTile>
    CK_TILE_DEVICE void compute_rhs_column(const ABlockTile& a_tile,
                                           const DataType* __restrict__ rhs_global,
                                           const float* __restrict__ beta_global,
                                           const float* __restrict__ g_global,
                                           DataType* __restrict__ out_global,
                                           int tc,
                                           int ivh,
                                           int stride_rhs_t,
                                           int stride_g_t,
                                           int stride_out_t,
                                           int col,
                                           bool with_gate,
                                           bool use_exp2,
                                           SharedStorage& smem) const
    {
        stage_scale_to_lds(
            beta_global, g_global, tc, ivh, stride_g_t, with_gate, use_exp2, smem);
        auto rhs_buf = load_rhs_global(rhs_global, stride_rhs_t, col);
        compute_rhs_buffer(a_tile, rhs_buf, out_global, tc, ivh, stride_out_t, col, smem);
    }

    template <typename ABlockTile>
    CK_TILE_DEVICE void compute_rhs_column_masked(
        const ABlockTile& a_tile,
        const DataType* __restrict__ rhs_global,
        const float* __restrict__ beta_global,
        const float* __restrict__ g_global,
        DataType* __restrict__ out_global,
        int tc,
        int ivh,
        int stride_rhs_t,
        int stride_g_t,
        int stride_out_t,
        int col,
        int valid_rows,
        bool with_gate,
        bool use_exp2,
        SharedStorage& smem) const
    {
        stage_scale_to_lds_masked(beta_global,
                                  g_global,
                                  tc,
                                  ivh,
                                  stride_g_t,
                                  valid_rows,
                                  with_gate,
                                  use_exp2,
                                  smem);
        auto rhs_buf =
            load_rhs_global_masked(rhs_global, stride_rhs_t, col, valid_rows);
        compute_rhs_buffer_masked(
            a_tile, rhs_buf, out_global, tc, ivh, stride_out_t, col, valid_rows, smem);
    }

    CK_TILE_DEVICE void operator()(
        const DataType* __restrict__ k_global,
        const DataType* __restrict__ v_global,
        const float*    __restrict__ beta_global,
        const DataType* __restrict__ A_global,
        const float*    __restrict__ g_global,
        DataType*       __restrict__ w_global,
        DataType*       __restrict__ u_global,
        int tc, int ih, int ivh,
        int, int, int,
        int stride_k_t, int stride_v_t, int stride_g_t, int stride_A_t,
        int stride_w_t, int stride_u_t,
        bool use_exp2,
        SharedStorage& smem) const
    {
        auto a_tile = load_a_tile_lower(A_global, tc, ivh, stride_A_t);

        stage_scale_to_lds(
            beta_global, g_global, tc, ivh, stride_g_t, false, use_exp2, smem);
        auto v_col0 = load_rhs_global(
            v_global + tc * stride_v_t + ivh * kHeadDim, stride_v_t, 0);
        auto v_col1 = load_rhs_global(
            v_global + tc * stride_v_t + ivh * kHeadDim, stride_v_t, 64);
        scale_a_columns(a_tile, smem);
        compute_rhs_pair_unscaled(
            a_tile, v_col0, v_col1, u_global, tc, ivh, stride_u_t, smem);

        stage_gate_to_lds(g_global, tc, ivh, stride_g_t, use_exp2, smem);
        auto k_col0 = load_rhs_global(
            k_global + tc * stride_k_t + ih * kHeadDim, stride_k_t, 0);
        auto k_col1 = load_rhs_global(
            k_global + tc * stride_k_t + ih * kHeadDim, stride_k_t, 64);
        scale_a_columns(a_tile, smem);
        compute_rhs_pair_unscaled(
            a_tile, k_col0, k_col1, w_global, tc, ivh, stride_w_t, smem);
    }

    CK_TILE_DEVICE void run_masked(
        const DataType* __restrict__ k_global,
        const DataType* __restrict__ v_global,
        const float* __restrict__ beta_global,
        const DataType* __restrict__ A_global,
        const float* __restrict__ g_global,
        DataType* __restrict__ w_global,
        DataType* __restrict__ u_global,
        int tc,
        int ih,
        int ivh,
        int stride_k_t,
        int stride_v_t,
        int stride_g_t,
        int stride_A_t,
        int stride_w_t,
        int stride_u_t,
        int valid_rows,
        bool use_exp2,
        SharedStorage& smem) const
    {
        auto a_tile = load_a_tile_masked(A_global, tc, ivh, stride_A_t, valid_rows);

        stage_scale_to_lds_masked(
            beta_global, g_global, tc, ivh, stride_g_t, valid_rows, false, use_exp2, smem);
        auto v_col0 = load_rhs_global_masked(
            v_global + tc * stride_v_t + ivh * kHeadDim, stride_v_t, 0, valid_rows);
        auto v_col1 = load_rhs_global_masked(
            v_global + tc * stride_v_t + ivh * kHeadDim, stride_v_t, 64, valid_rows);
        scale_a_columns(a_tile, smem);
        compute_rhs_pair_unscaled_masked(
            a_tile, v_col0, v_col1, u_global, tc, ivh, stride_u_t, valid_rows, smem);

        stage_gate_to_lds_masked(
            g_global, tc, ivh, stride_g_t, valid_rows, use_exp2, smem);
        auto k_col0 = load_rhs_global_masked(
            k_global + tc * stride_k_t + ih * kHeadDim, stride_k_t, 0, valid_rows);
        auto k_col1 = load_rhs_global_masked(
            k_global + tc * stride_k_t + ih * kHeadDim, stride_k_t, 64, valid_rows);
        scale_a_columns(a_tile, smem);
        compute_rhs_pair_unscaled_masked(
            a_tile, k_col0, k_col1, w_global, tc, ivh, stride_w_t, valid_rows, smem);
    }

    CK_TILE_DEVICE void run_split4_task(
        const DataType* __restrict__ k_global,
        const DataType* __restrict__ v_global,
        const float*    __restrict__ beta_global,
        const DataType* __restrict__ A_global,
        const float*    __restrict__ g_global,
        DataType*       w_global,
        DataType*       u_global,
        int tc, int ih, int ivh,
        int stride_k_t, int stride_v_t, int stride_g_t, int stride_A_t,
        int stride_w_t, int stride_u_t,
        bool use_exp2,
        int task,
        SharedStorage& smem) const
    {
        auto a_tile = load_a_tile(A_global, tc, ivh, stride_A_t);
        if(task < 2)
        {
            compute_rhs_column(a_tile,
                               v_global + tc * stride_v_t + ivh * kHeadDim,
                               beta_global, g_global, u_global,
                               tc, ivh, stride_v_t, stride_g_t, stride_u_t,
                               task * 64, false, use_exp2, smem);
        }
        else
        {
            compute_rhs_column(a_tile,
                               k_global + tc * stride_k_t + ih * kHeadDim,
                               beta_global, g_global, w_global,
                               tc, ivh, stride_k_t, stride_g_t, stride_w_t,
                               (task - 2) * 64, true, use_exp2, smem);
        }
    }

    CK_TILE_DEVICE void run_split4_task_masked(
        const DataType* __restrict__ k_global,
        const DataType* __restrict__ v_global,
        const float* __restrict__ beta_global,
        const DataType* __restrict__ A_global,
        const float* __restrict__ g_global,
        DataType* w_global,
        DataType* u_global,
        int tc,
        int ih,
        int ivh,
        int stride_k_t,
        int stride_v_t,
        int stride_g_t,
        int stride_A_t,
        int stride_w_t,
        int stride_u_t,
        int valid_rows,
        bool use_exp2,
        int task,
        SharedStorage& smem) const
    {
        auto a_tile = load_a_tile_masked(A_global, tc, ivh, stride_A_t, valid_rows);
        if(task < 2)
        {
            compute_rhs_column_masked(a_tile,
                                      v_global + tc * stride_v_t + ivh * kHeadDim,
                                      beta_global,
                                      g_global,
                                      u_global,
                                      tc,
                                      ivh,
                                      stride_v_t,
                                      stride_g_t,
                                      stride_u_t,
                                      task * 64,
                                      valid_rows,
                                      false,
                                      use_exp2,
                                      smem);
        }
        else
        {
            compute_rhs_column_masked(a_tile,
                                      k_global + tc * stride_k_t + ih * kHeadDim,
                                      beta_global,
                                      g_global,
                                      w_global,
                                      tc,
                                      ivh,
                                      stride_k_t,
                                      stride_g_t,
                                      stride_w_t,
                                      (task - 2) * 64,
                                      valid_rows,
                                      true,
                                      use_exp2,
                                      smem);
        }
    }

};

} // namespace ck_tile
