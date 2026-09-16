// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_kkt_solve_policy.hpp"
#include "ck_tile/ops/gdn/block/gdn_ck_kkt_solve_block_gemm.hpp"

namespace ck_tile {

template <typename Policy, bool MaskKRows_ = false, int NumWarps = 4>
struct GdnKktSolvePipeline
{
    static_assert(NumWarps == 1 || NumWarps == 2 || NumWarps == 4);

    static constexpr bool MaskKRows = MaskKRows_;
    static constexpr int kChunkSize = Policy::kChunkSize;
    static constexpr int kSubChunk  = Policy::kSubChunk;
    static constexpr int kHeadDim   = Policy::kHeadDim;

    using DataType = typename Policy::DataType;
    using AccType  = float;
    static constexpr int kBlockElems  = kSubChunk * kSubChunk;
    static constexpr int kLowerBlocks = 10;
    static constexpr int S00 = 0;
    static constexpr int S10 = 1;
    static constexpr int S11 = 2;
    static constexpr int S20 = 3;
    static constexpr int S21 = 4;
    static constexpr int S22 = 5;
    static constexpr int S30 = 6;
    static constexpr int S31 = 7;
    static constexpr int S32 = 8;
    static constexpr int S33 = 9;

    struct KktGemmProblem
    {
        using ADataType      = DataType;
        using BDataType      = DataType;
        using CDataType      = AccType;
        using BlockGemmShape = TileGemmShape<
            sequence<64, 64, 128>,
            sequence<4, 1, 1>,
            sequence<16, 64, 32>>;
        static constexpr index_t kBlockSize = Policy::kBlockSize;
    };
    using KktBlockGemmPolicy =
        GdnKktARegBSmemPolicy<sequence<4, 1, 1>, typename Policy::KktWarpGemm>;

    struct Block16GemmProblem
    {
        using ADataType      = DataType;
        using BDataType      = DataType;
        using CDataType      = AccType;
        using BlockGemmShape = TileGemmShape<
            sequence<16, 16, 16>,
            sequence<1, 1, 1>,
            sequence<16, 16, 16>>;
        static constexpr index_t kBlockSize = Policy::kBlockSize;
    };
    using Block16GemmPolicy =
        GdnKktARegBSmemPolicy<sequence<1, 1, 1>, typename Policy::SmallWarpGemm>;
    using Block16Gemm = GdnKktBlockGemmARegBReg<Block16GemmProblem, Block16GemmPolicy>;

    struct SharedStorage
    {
        float diag_work[NumWarps][2][kBlockElems];
    };

    using RegGemm16 = GdnKktWarpRegGemm16<DataType>;
    using RegCVec = typename RegGemm16::CVec;

    struct RegLowerBlocks
    {
        RegCVec s00, s10, s11, s20, s21, s22, s30, s31, s32, s33;
    };

    template <typename Tile>
    CK_TILE_DEVICE void mask_scale_kkt(Tile& tile,
                                       const float* __restrict__ g_global,
                                       const float* __restrict__ b_global,
                                       int tc,
                                       int ivh,
                                       int T,
                                       int stride_g_t,
                                       bool use_exp2) const
    {
        constexpr auto spans = Tile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(tile.get_tile_distribution(), dstr_idx);
                const int r = tile_idx.at(number<0>{});
                const int c = tile_idx.at(number<1>{});
                if(r > c && tc + r < T)
                {
                    const float gd = g_global[(tc + r) * stride_g_t + ivh] -
                                     g_global[(tc + c) * stride_g_t + ivh];
                    const float gate = use_exp2 ? __builtin_amdgcn_exp2f(gd) : expf(gd);
                    tile(dstr_idx) *= gate * b_global[(tc + r) * stride_g_t + ivh];
                }
                else
                {
                    tile(dstr_idx) = 0.0f;
                }
            });
        });
    }

    template <typename Tile>
    CK_TILE_DEVICE void add_identity(Tile& tile) const
    {
        constexpr auto spans = Tile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(tile.get_tile_distribution(), dstr_idx);
                if(tile_idx.at(number<0>{}) == tile_idx.at(number<1>{}))
                    tile(dstr_idx) += 1.0f;
            });
        });
    }

    template <typename Tile>
    CK_TILE_DEVICE void neg_tile(Tile& tile) const
    {
        tile_elementwise_inout([](auto& x) { x = -x; }, tile);
    }

    template <typename Tile>
    CK_TILE_DEVICE void accumulate(Tile& dst, const Tile& src) const
    {
        tile_elementwise_inout([](auto& x, const auto& y) { x += y; }, dst, src);
    }

    template <typename ATile, typename BTile>
    CK_TILE_DEVICE auto gemm16(const ATile& a, const BTile& b) const
    {
        constexpr auto bg = Block16Gemm{};
        auto c = Block16Gemm::MakeCBlockTile();
        clear_tile(c);
        bg(c, a, b);
        return bg.MakeOuputLayout(c);
    }

    CK_TILE_DEVICE static constexpr auto MakeCOutBlockTile()
    {
        constexpr auto bg = Block16Gemm{};
        auto c = Block16Gemm::MakeCBlockTile();
        return bg.MakeOuputLayout(c);
    }

    template <typename LdsView, typename ProtoTile>
    CK_TILE_DEVICE auto load16(const LdsView& view, int r, int c, const ProtoTile& proto) const
    {
        auto win = make_tile_window(
            view,
            make_tuple(number<kSubChunk>{}, number<kSubChunk>{}),
            multi_index<2>{r, c},
            proto.get_tile_distribution());
        return load_tile(win);
    }

    template <typename LdsView>
    CK_TILE_DEVICE auto load16_a(const LdsView& view, int r, int c) const
    {
        return cast_tile<DataType>(load16(view, r, c, Block16Gemm::MakeABlockTile()));
    }

    template <typename LdsView>
    CK_TILE_DEVICE auto load16_b(const LdsView& view, int r, int c) const
    {
        auto b_t_view = make_naive_tensor_view<address_space_enum::lds>(
            view.get_buffer_view().p_data_ + r * kChunkSize + c,
            make_tuple(number<kSubChunk>{}, number<kSubChunk>{}),
            make_tuple(number<1>{}, number<kChunkSize>{}),
            number<1>{},
            number<1>{});
        return cast_tile<DataType>(load16(b_t_view, 0, 0, Block16Gemm::MakeBBlockTile()));
    }

    template <typename TmpView>
    CK_TILE_DEVICE auto load16_b_tmp(const TmpView& tmp_view) const
    {
        auto b_t_view = make_naive_tensor_view<address_space_enum::lds>(
            tmp_view.get_buffer_view().p_data_,
            make_tuple(number<kSubChunk>{}, number<kSubChunk>{}),
            make_tuple(number<1>{}, number<kSubChunk>{}),
            number<1>{},
            number<1>{});
        return cast_tile<DataType>(load16(b_t_view, 0, 0, Block16Gemm::MakeBBlockTile()));
    }

    template <typename LdsView>
    CK_TILE_DEVICE auto load16_c_out(const LdsView& view, int r, int c) const
    {
        return load16(view, r, c, MakeCOutBlockTile());
    }

    template <typename LdsView, typename Tile>
    CK_TILE_DEVICE void store16(const LdsView& view, int r, int c, const Tile& tile) const
    {
        auto win = make_tile_window(
            view,
            make_tuple(number<kSubChunk>{}, number<kSubChunk>{}),
            multi_index<2>{r, c},
            tile.get_tile_distribution());
        store_tile(win, tile);
    }

    template <typename TmpView, typename Tile>
    CK_TILE_DEVICE auto c_to_a_via_lds(const TmpView& tmp_view, const Tile& tile) const
    {
        store16(tmp_view, 0, 0, tile);
        block_sync_lds();
        return load16_a(tmp_view, 0, 0);
    }

    template <typename TmpView, typename Tile>
    CK_TILE_DEVICE auto c_to_b_via_lds(const TmpView& tmp_view, const Tile& tile) const
    {
        store16(tmp_view, 0, 0, tile);
        block_sync_lds();
        return load16_b_tmp(tmp_view);
    }

    template <typename TmpView, typename Tile>
    CK_TILE_DEVICE auto c_to_c_via_lds(const TmpView& tmp_view, const Tile& tile) const
    {
        store16(tmp_view, 0, 0, tile);
        block_sync_lds();
        return load16_c_out(tmp_view, 0, 0);
    }

    template <typename LdsView>
    CK_TILE_DEVICE auto mm16_lds(const LdsView& view, int ar, int ac, int br, int bc) const
    {
        auto a = load16_a(view, ar, ac);
        auto b = load16_b(view, br, bc);
        return gemm16(a, b);
    }

    template <typename LdsView, typename TmpView>
    CK_TILE_DEVICE void invert_diag16(const LdsView& view, const TmpView& tmp_view, int base) const
    {
        auto l = load16(view, base, base, Block16Gemm::MakeCBlockTile());
        auto neg_l_b = load16_b(view, base, base);
        neg_tile(neg_l_b);

        auto ai = l;
        neg_tile(ai);
        add_identity(ai);

        auto power = load16_a(view, base, base);
        neg_tile(power);
        static_for<2, 16, 1>{}([&](auto) {
            auto prod = gemm16(power, neg_l_b);
            auto prod_c = c_to_c_via_lds(tmp_view, prod);
            accumulate(ai, prod_c);
            power = c_to_a_via_lds(tmp_view, prod_c);
        });

        store16(view, base, base, ai);
    }

    template <int OutSlot, int LeftDiagSlot, int DiagSlot, typename LdsView>
    CK_TILE_DEVICE void merge_level1_adjacent(LdsView& view) const
    {
        using WG16 = GdnKktWarpMmac16<DataType>;
        auto mid = WG16::template mm_slot_slot<OutSlot, LeftDiagSlot>(view);
        auto out = WG16::template mm_slot_breg<DiagSlot>(view, WG16::c_to_b(mid));
        WG16::template store_c_slot<OutSlot>(view, WG16::neg_c(out));
    }

    template <int OutSlot,
              int Src0Slot,
              int Inv0Slot,
              int Src1Slot,
              int Inv1Slot,
              int DiagSlot,
              typename LdsView>
    CK_TILE_DEVICE void merge_level2_two_term(LdsView& view) const
    {
        using WG16 = GdnKktWarpMmac16<DataType>;
        auto acc = WG16::template mm_slot_slot<Src0Slot, Inv0Slot>(view);
        auto rhs = WG16::template mm_slot_slot<Src1Slot, Inv1Slot>(view);
        acc += rhs;
        auto out = WG16::template mm_slot_breg<DiagSlot>(view, WG16::c_to_b(acc));
        WG16::template store_c_slot<OutSlot>(view, WG16::neg_c(out));
    }

    template <int OutSlot,
              int Src0Slot,
              int Inv0Slot,
              int Src1Slot,
              int Inv1Slot,
              int Src2Slot,
              int Inv2Slot,
              int DiagSlot,
              typename LdsView>
    CK_TILE_DEVICE void merge_level3_three_term(LdsView& view) const
    {
        using WG16 = GdnKktWarpMmac16<DataType>;
        auto acc = WG16::template mm_slot_slot<Src0Slot, Inv0Slot>(view);
        auto rhs0 = WG16::template mm_slot_slot<Src1Slot, Inv1Slot>(view);
        auto rhs1 = WG16::template mm_slot_slot<Src2Slot, Inv2Slot>(view);
        acc += rhs0;
        acc += rhs1;
        auto out = WG16::template mm_slot_breg<DiagSlot>(view, WG16::c_to_b(acc));
        WG16::template store_c_slot<OutSlot>(view, WG16::neg_c(out));
    }

    template <typename LdsView>
    CK_TILE_DEVICE void solve_merge(LdsView& view) const
    {
        const index_t warp = get_warp_id();

        auto init_diag = [&](auto bi, auto slot_c) {
            constexpr int slot = slot_c;
            if(warp == bi)
            {
                const index_t lane = get_lane_id();
                static_for<0, 4, 1>{}([&](auto e) {
                    const index_t idx = lane + e * 64;
                    const index_t r = idx / kSubChunk;
                    const index_t c = idx - r * kSubChunk;
                    const index_t off = slot * kBlockElems + r * kSubChunk + c;
                    const float v = view.get_buffer_view()[off];
                    view.get_buffer_view()(off) = (r > c) ? -v : 0.0f;
                });
            }
        };

        auto finish_diag = [&](auto bi, auto slot_c) {
            constexpr int slot = slot_c;
            if(warp == bi)
            {
                const index_t lane = get_lane_id();
                static_for<0, 4, 1>{}([&](auto e) {
                    const index_t idx = lane + e * 64;
                    const index_t r = idx / kSubChunk;
                    const index_t c = idx - r * kSubChunk;
                    if(r == c)
                    {
                        view.get_buffer_view()(slot * kBlockElems + r * kSubChunk + c) += 1.0f;
                    }
                });
            }
        };

        auto update_diag_row = [&](auto bi, auto slot_c, auto ii) {
            constexpr int slot = slot_c;
            if(warp == bi)
            {
                const index_t j = get_lane_id() & 15;
                float row_j = 0.0f;
                if(j < ii)
                {
                    row_j = view.get_buffer_view()[slot * kBlockElems + ii * kSubChunk + j];
                    float corr = 0.0f;
                    static_for<0, kSubChunk, 1>{}([&](auto kk) {
                        if(kk < ii)
                        {
                            const float aik = warp_shuffle(row_j, kk);
                            const float akj = view.get_buffer_view()[
                                slot * kBlockElems + kk * kSubChunk + j];
                            corr += aik * akj;
                        }
                    });
                    row_j += corr;
                }
                view.get_buffer_view()(slot * kBlockElems + ii * kSubChunk + j) = row_j;
            }
        };

        init_diag(number<0>{}, number<S00>{});
        init_diag(number<1>{}, number<S11>{});
        init_diag(number<2>{}, number<S22>{});
        init_diag(number<3>{}, number<S33>{});
        block_sync_lds();

        static_for<2, kSubChunk, 1>{}([&](auto ii) {
            update_diag_row(number<0>{}, number<S00>{}, ii);
            update_diag_row(number<1>{}, number<S11>{}, ii);
            update_diag_row(number<2>{}, number<S22>{}, ii);
            update_diag_row(number<3>{}, number<S33>{}, ii);
        });
        block_sync_lds();

        finish_diag(number<0>{}, number<S00>{});
        finish_diag(number<1>{}, number<S11>{});
        finish_diag(number<2>{}, number<S22>{});
        finish_diag(number<3>{}, number<S33>{});
        block_sync_lds();

        if(warp == 0)
        {
            merge_level1_adjacent<S10, S00, S11>(view);
        }
        if(warp == 1)
        {
            merge_level1_adjacent<S21, S11, S22>(view);
        }
        if(warp == 2)
        {
            merge_level1_adjacent<S32, S22, S33>(view);
        }
        block_sync_lds();

        if(warp == 0)
        {
            merge_level2_two_term<S20, S20, S00, S21, S10, S22>(view);
        }
        if(warp == 1)
        {
            merge_level2_two_term<S31, S31, S11, S32, S21, S33>(view);
        }
        block_sync_lds();

        if(warp == 0)
        {
            merge_level3_three_term<S30, S30, S00, S31, S10, S32, S20, S33>(view);
        }
    }

    template <typename LdsView>
    CK_TILE_DEVICE void compute_store_kkt16_direct(const DataType* __restrict__ k_base,
                                                   LdsView& lds_view,
                                                   int slot,
                                                   int row_base,
                                                   int col_base,
                                                   int tc,
                                                   int T,
                                                   int stride_k_t,
                                                   const float* __restrict__ row_factor,
                                                   const float* __restrict__ col_factor) const
    {
        using WG16 = GdnKktWarpKkt16<DataType>;
        const auto c = WG16::template compute<MaskKRows>(
            k_base, stride_k_t, row_base, col_base, tc, T);
        WG16::store_scaled_slot(lds_view, slot, row_base, col_base, c,
                                row_factor, col_factor, tc, T);
    }

    CK_TILE_DEVICE void precompute_gate_factors(const float* __restrict__ g_global,
                                                const float* __restrict__ b_global,
                                                float* __restrict__ row_factor,
                                                float* __restrict__ col_factor,
                                                int tc,
                                                int ivh,
                                                int T,
                                                int stride_g_t,
                                                bool use_exp2) const
    {
        const index_t tid = get_thread_local_1d_id();
        if(tid < kChunkSize)
        {
            float rf = 0.0f;
            float cf = 0.0f;
            if(tc + tid < T)
            {
                const float g = g_global[(tc + tid) * stride_g_t + ivh];
                const float eg = use_exp2 ? __builtin_amdgcn_exp2f(g) : expf(g);
                const float emg = use_exp2 ? __builtin_amdgcn_exp2f(-g) : expf(-g);
                rf = eg * b_global[(tc + tid) * stride_g_t + ivh];
                cf = emg;
            }
            row_factor[tid] = rf;
            col_factor[tid] = cf;
        }
    }

    template <typename LdsView>
    CK_TILE_DEVICE void compute_store_kkt16_pair_same_row(const DataType* __restrict__ k_base,
                                                          LdsView& lds_view,
                                                          int slot0,
                                                          int col_base0,
                                                          int slot1,
                                                          int col_base1,
                                                          int row_base,
                                                          int tc,
                                                          int T,
                                                          int stride_k_t,
                                                          const float* __restrict__ row_factor,
                                                          const float* __restrict__ col_factor) const
    {
        using WG16 = GdnKktWarpKkt16<DataType>;
        typename WG16::CVec c0{0.f, 0.f, 0.f, 0.f};
        typename WG16::CVec c1{0.f, 0.f, 0.f, 0.f};
        static_for<0, 8, 1>{}([&](auto k_iter) {
            constexpr int k_base_col = k_iter * 16;
            const auto a = WG16::template load_a<MaskKRows>(
                k_base, stride_k_t, row_base, k_base_col, tc, T);
            const auto b0 = WG16::template load_b<MaskKRows>(
                k_base, stride_k_t, col_base0, k_base_col, tc, T);
            const auto b1 = WG16::template load_b<MaskKRows>(
                k_base, stride_k_t, col_base1, k_base_col, tc, T);
            typename WG16::Impl impl;
            impl(c0, a, b0);
            impl(c1, a, b1);
        });
        WG16::store_scaled_slot(lds_view, slot0, row_base, col_base0, c0,
                                row_factor, col_factor, tc, T);
        WG16::store_scaled_slot(lds_view, slot1, row_base, col_base1, c1,
                                row_factor, col_factor, tc, T);
    }

    template <bool ZeroUpper, typename LdsView>
    CK_TILE_DEVICE void store_full_block(const LdsView& view,
                                         DataType* __restrict__ A_base,
                                         int slot,
                                         int row_base,
                                         int col_base,
                                         int tc,
                                         int T,
                                         int stride_A_t) const
    {
        const index_t lane = get_lane_id();
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t idx = lane + e * 64;
            const index_t r = idx / kSubChunk;
            const index_t c = idx - r * kSubChunk;
            if(tc + row_base + r < T)
            {
                float v = 0.0f;
                if constexpr(ZeroUpper)
                {
                    if(r >= c)
                    {
                        v = view.get_buffer_view()[slot * kBlockElems + r * kSubChunk + c];
                    }
                }
                else
                {
                    v = view.get_buffer_view()[slot * kBlockElems + r * kSubChunk + c];
                }
                thread_buffer<DataType, 1> out;
                out(number<0>{}) = type_convert<DataType>(v);
                const index_t out_offset = (row_base + r) * stride_A_t + col_base + c;
                A_base[out_offset] = out(number<0>{});
            }
        });
    }

    CK_TILE_DEVICE void store_zero_full_block(DataType* __restrict__ A_base,
                                              int row_base,
                                              int col_base,
                                              int tc,
                                              int T,
                                              int stride_A_t) const
    {
        const index_t lane = get_lane_id();
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t idx = lane + e * 64;
            const index_t r = idx / kSubChunk;
            const index_t c = idx - r * kSubChunk;
            if(tc + row_base + r < T)
            {
                thread_buffer<DataType, 1> out;
                out(number<0>{}) = type_convert<DataType>(0.0f);
                const index_t out_offset = (row_base + r) * stride_A_t + col_base + c;
                A_base[out_offset] = out(number<0>{});
            }
        });
    }

    template <typename LdsView>
    CK_TILE_DEVICE void store_output_blocks(const LdsView& view,
                                            DataType* __restrict__ A_base,
                                            int tc,
                                            int T,
                                            int stride_A_t) const
    {
        const index_t warp = get_warp_id();
        if(warp == 0)
        {
            store_full_block<true>(view, A_base, S00, 0, 0, tc, T, stride_A_t);
            store_full_block<false>(view, A_base, S20, 32, 0, tc, T, stride_A_t);
            store_full_block<false>(view, A_base, S31, 48, 16, tc, T, stride_A_t);
            store_zero_full_block(A_base, 0, 16, tc, T, stride_A_t);
        }
        if(warp == 1)
        {
            store_full_block<true>(view, A_base, S11, 16, 16, tc, T, stride_A_t);
            store_full_block<false>(view, A_base, S21, 32, 16, tc, T, stride_A_t);
            store_full_block<false>(view, A_base, S32, 48, 32, tc, T, stride_A_t);
            store_zero_full_block(A_base, 0, 32, tc, T, stride_A_t);
        }
        if(warp == 2)
        {
            store_full_block<true>(view, A_base, S22, 32, 32, tc, T, stride_A_t);
            store_full_block<false>(view, A_base, S10, 16, 0, tc, T, stride_A_t);
            store_zero_full_block(A_base, 0, 48, tc, T, stride_A_t);
            store_zero_full_block(A_base, 16, 32, tc, T, stride_A_t);
        }
        if(warp == 3)
        {
            store_full_block<true>(view, A_base, S33, 48, 48, tc, T, stride_A_t);
            store_full_block<false>(view, A_base, S30, 48, 0, tc, T, stride_A_t);
            store_zero_full_block(A_base, 16, 48, tc, T, stride_A_t);
            store_zero_full_block(A_base, 32, 48, tc, T, stride_A_t);
        }
    }

    CK_TILE_DEVICE RegCVec scale_kkt_block(RegCVec c,
                                           int row_base,
                                           int col_base,
                                           float row_factor,
                                           float col_factor,
                                           int tc,
                                           int T) const
    {
        union {
            RegCVec v;
            float e[4];
        } u{c};
        const index_t lane = get_lane_id();
        const index_t col = col_base + (lane & 15);
        const index_t m_group = (lane >> 4) & 3;
        const float cf = warp_shuffle(col_factor, col);
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t r = row_base + m_group + e * 4;
            const float rf = warp_shuffle(row_factor, r);
            if(tc + r < T && tc + col < T &&
               (row_base != col_base || r > col))
            {
                u.e[e] *= rf * cf;
            }
            else
            {
                u.e[e] = 0.0f;
            }
        });
        return u.v;
    }

    template <int ActiveBlocks>
    CK_TILE_DEVICE void compute_kkt_active_reg(const DataType* __restrict__ k_base,
                                               int stride_k_t,
                                               float row_factor,
                                               float col_factor,
                                               int tc,
                                               int T,
                                               RegLowerBlocks& b) const
    {
        static_assert(ActiveBlocks >= 1 && ActiveBlocks <= 4);
        using Kkt16 = GdnKktWarpKkt16<DataType>;
        using AVec = typename Kkt16::AVec;
        using BVec = typename Kkt16::BVec;
        static_assert(std::is_same_v<AVec, BVec>);

        b.s00 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s10 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s11 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s20 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s21 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s22 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s30 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s31 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s32 = RegCVec{0.f, 0.f, 0.f, 0.f};
        b.s33 = RegCVec{0.f, 0.f, 0.f, 0.f};

        static_for<0, 8, 1>{}([&](auto k_iter) {
            constexpr int k_base_col = k_iter * 16;
            const AVec k0 = Kkt16::template load_a<MaskKRows>(
                k_base, stride_k_t, 0, k_base_col, tc, T);
            typename Kkt16::Impl impl;
            impl(b.s00, k0, k0);
            if constexpr(ActiveBlocks >= 2)
            {
                const AVec k1 = Kkt16::template load_a<MaskKRows>(
                    k_base, stride_k_t, 16, k_base_col, tc, T);
                impl(b.s10, k1, k0);
                impl(b.s11, k1, k1);
                if constexpr(ActiveBlocks >= 3)
                {
                    const AVec k2 = Kkt16::template load_a<MaskKRows>(
                        k_base, stride_k_t, 32, k_base_col, tc, T);
                    impl(b.s20, k2, k0);
                    impl(b.s21, k2, k1);
                    impl(b.s22, k2, k2);
                    if constexpr(ActiveBlocks >= 4)
                    {
                        const AVec k3 = Kkt16::template load_a<MaskKRows>(
                            k_base, stride_k_t, 48, k_base_col, tc, T);
                        impl(b.s30, k3, k0);
                        impl(b.s31, k3, k1);
                        impl(b.s32, k3, k2);
                        impl(b.s33, k3, k3);
                    }
                }
            }
        });

        b.s00 = scale_kkt_block(b.s00, 0, 0, row_factor, col_factor, tc, T);
        if constexpr(ActiveBlocks >= 2)
        {
            b.s10 = scale_kkt_block(b.s10, 16, 0, row_factor, col_factor, tc, T);
            b.s11 = scale_kkt_block(b.s11, 16, 16, row_factor, col_factor, tc, T);
        }
        if constexpr(ActiveBlocks >= 3)
        {
            b.s20 = scale_kkt_block(b.s20, 32, 0, row_factor, col_factor, tc, T);
            b.s21 = scale_kkt_block(b.s21, 32, 16, row_factor, col_factor, tc, T);
            b.s22 = scale_kkt_block(b.s22, 32, 32, row_factor, col_factor, tc, T);
        }
        if constexpr(ActiveBlocks >= 4)
        {
            b.s30 = scale_kkt_block(b.s30, 48, 0, row_factor, col_factor, tc, T);
            b.s31 = scale_kkt_block(b.s31, 48, 16, row_factor, col_factor, tc, T);
            b.s32 = scale_kkt_block(b.s32, 48, 32, row_factor, col_factor, tc, T);
            b.s33 = scale_kkt_block(b.s33, 48, 48, row_factor, col_factor, tc, T);
        }
    }

    CK_TILE_DEVICE RegCVec compute_kkt_single_reg(const DataType* __restrict__ k_base,
                                                  int row_base,
                                                  int col_base,
                                                  int stride_k_t,
                                                  float row_factor,
                                                  float col_factor,
                                                  int tc,
                                                  int T) const
    {
        using Kkt16 = GdnKktWarpKkt16<DataType>;
        return scale_kkt_block(Kkt16::template compute<false>(
                                   k_base, stride_k_t, row_base, col_base, tc, T),
                               row_base, col_base, row_factor, col_factor, tc, T);
    }

    CK_TILE_DEVICE void compute_kkt_pair_reg(const DataType* __restrict__ k_base,
                                             int row_base,
                                             int col_base0,
                                             int col_base1,
                                             int stride_k_t,
                                             float row_factor,
                                             float col_factor,
                                             int tc,
                                             int T,
                                             RegCVec& c0,
                                             RegCVec& c1) const
    {
        using Kkt16 = GdnKktWarpKkt16<DataType>;
        c0 = RegCVec{0.f, 0.f, 0.f, 0.f};
        c1 = RegCVec{0.f, 0.f, 0.f, 0.f};
        static_for<0, 8, 1>{}([&](auto k_iter) {
            constexpr int k_base_col = k_iter * 16;
            const auto a = Kkt16::template load_a<false>(
                k_base, stride_k_t, row_base, k_base_col, tc, T);
            const auto b0 = Kkt16::template load_b<false>(
                k_base, stride_k_t, col_base0, k_base_col, tc, T);
            const auto b1 = Kkt16::template load_b<false>(
                k_base, stride_k_t, col_base1, k_base_col, tc, T);
            typename Kkt16::Impl impl;
            impl(c0, a, b0);
            impl(c1, a, b1);
        });
        c0 = scale_kkt_block(c0, row_base, col_base0,
                             row_factor, col_factor, tc, T);
        c1 = scale_kkt_block(c1, row_base, col_base1,
                             row_factor, col_factor, tc, T);
    }

    CK_TILE_DEVICE void store_diag_work(float* __restrict__ work, const RegCVec& c) const
    {
        const index_t lane = get_lane_id();
        const index_t col = lane & 15;
        const index_t m_group = (lane >> 4) & 3;
        static_for<0, 4, 1>{}([&](auto e) {
            work[(m_group + e * 4) * 16 + col] = RegGemm16::c_at(c, e);
        });
    }

    CK_TILE_DEVICE RegCVec load_diag_work(const float* __restrict__ work) const
    {
        RegCVec c{0.f, 0.f, 0.f, 0.f};
        union {
            RegCVec v;
            float e[4];
        } u{c};
        const index_t lane = get_lane_id();
        const index_t col = lane & 15;
        const index_t m_group = (lane >> 4) & 3;
        static_for<0, 4, 1>{}([&](auto e) {
            u.e[e] = work[(m_group + e * 4) * 16 + col];
        });
        return u.v;
    }

    CK_TILE_DEVICE void invert_diag_pair_work(float* __restrict__ work0,
                                              float* __restrict__ work1) const
    {
        const index_t lane = get_lane_id();
        const index_t pair = lane >> 5;
        const index_t local_lane = lane & 31;
        float* work = pair == 0 ? work0 : work1;

        static_for<0, 8, 1>{}([&](auto e) {
            const index_t idx = local_lane + e * 32;
            const index_t r = idx / 16;
            const index_t c = idx - r * 16;
            const float v = work[idx];
            work[idx] = r > c ? -v : 0.0f;
        });

        static_for<2, 16, 1>{}([&](auto ii) {
            const index_t j = local_lane & 15;
            const index_t helper = local_lane >> 4;
            float row_j = work[ii * 16 + j];
            float corr = 0.0f;
            static_for<0, 8, 1>{}([&](auto pair_k) {
                const index_t kk = pair_k * 2 + helper;
                if(kk < ii)
                {
                    const float aik = warp_shuffle(row_j, pair * 32 + kk);
                    corr += aik * work[kk * 16 + j];
                }
            });
            const float helper_corr =
                warp_shuffle(corr, pair * 32 + 16 + j);
            if(helper == 0 && j < ii)
            {
                row_j += corr + helper_corr;
                work[ii * 16 + j] = row_j;
            }
        });

        if(local_lane < 16)
        {
            work[local_lane * 16 + local_lane] = 1.0f;
        }
    }

    CK_TILE_DEVICE void invert_diag_pair(RegCVec& c0,
                                         RegCVec& c1,
                                         float* __restrict__ work0,
                                         float* __restrict__ work1) const
    {
        store_diag_work(work0, c0);
        store_diag_work(work1, c1);
        lds_wait();
        invert_diag_pair_work(work0, work1);
        lds_wait();
        c0 = load_diag_work(work0);
        c1 = load_diag_work(work1);
        lds_wait();
    }

    template <int ActiveBlocks>
    CK_TILE_DEVICE void invert_active_diags(RegLowerBlocks& b,
                                            float* __restrict__ work0,
                                            float* __restrict__ work1) const
    {
        invert_diag_pair(b.s00, b.s11, work0, work1);
        if constexpr(ActiveBlocks >= 3)
        {
            invert_diag_pair(b.s22, b.s33, work0, work1);
        }
    }

    template <int ActiveBlocks>
    CK_TILE_DEVICE void solve_merge_reg(RegLowerBlocks& b) const
    {
        // For L * X = I, X_ij = -inv(L_ii) * sum_k(L_ik * X_kj).
        // Keep the original off-diagonal blocks until their last use.
        if constexpr(ActiveBlocks >= 2)
        {
            const auto rhs = RegGemm16::multiply(b.s10, b.s00);
            b.s10 = RegGemm16::negate(RegGemm16::multiply(b.s11, rhs));
        }

        if constexpr(ActiveBlocks >= 3)
        {
            const auto l21 = b.s21;
            auto rhs = RegGemm16::multiply(l21, b.s11);
            b.s21 = RegGemm16::negate(RegGemm16::multiply(b.s22, rhs));

            rhs = RegGemm16::multiply(b.s20, b.s00);
            rhs += RegGemm16::multiply(l21, b.s10);
            b.s20 = RegGemm16::negate(RegGemm16::multiply(b.s22, rhs));
        }

        if constexpr(ActiveBlocks >= 4)
        {
            const auto l31 = b.s31;
            const auto l32 = b.s32;
            auto rhs = RegGemm16::multiply(l32, b.s22);
            b.s32 = RegGemm16::negate(RegGemm16::multiply(b.s33, rhs));

            rhs = RegGemm16::multiply(l31, b.s11);
            rhs += RegGemm16::multiply(l32, b.s21);
            b.s31 = RegGemm16::negate(RegGemm16::multiply(b.s33, rhs));

            rhs = RegGemm16::multiply(b.s30, b.s00);
            rhs += RegGemm16::multiply(l31, b.s10);
            rhs += RegGemm16::multiply(l32, b.s20);
            b.s30 = RegGemm16::negate(RegGemm16::multiply(b.s33, rhs));
        }
    }

    template <bool ZeroUpper>
    CK_TILE_DEVICE void store_reg_block(const RegCVec& c,
                                        DataType* __restrict__ A_base,
                                        const int32x4_t resource,
                                        int row_base,
                                        int col_base,
                                        int tc,
                                        int T,
                                        int stride_A_t) const
    {
        const index_t lane = get_lane_id();
        const index_t col = lane & 15;
        const index_t m_group = (lane >> 4) & 3;
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t row = m_group + e * 4;
            if(tc + row_base + row < T)
            {
                float v = RegGemm16::c_at(c, e);
                if constexpr(ZeroUpper)
                {
                    if(row < col)
                        v = 0.0f;
                }
                const index_t offset =
                    (row_base + row) * stride_A_t + col_base + col;
                buffer_store<2>{}(type_convert<DataType>(v),
                                  resource,
                                  offset * sizeof(DataType),
                                  0,
                                  0);
            }
        });
    }

    CK_TILE_DEVICE void store_reg_zero_block(DataType* __restrict__ A_base,
                                             const int32x4_t resource,
                                             int row_base,
                                             int col_base,
                                             int tc,
                                             int T,
                                             int stride_A_t) const
    {
        const index_t lane = get_lane_id();
        const index_t row = lane >> 2;
        const index_t col = (lane & 3) * 4;
        thread_buffer<DataType, 4> zero4;
        static_for<0, 4, 1>{}([&](auto e) {
            zero4(e) = type_convert<DataType>(0.0f);
        });
        if(tc + row_base + row < T)
        {
            const index_t offset =
                (row_base + row) * stride_A_t + col_base + col;
            buffer_store<8>{}(zero4,
                              resource,
                              offset * sizeof(DataType),
                              0,
                              0);
        }
    }

    template <int ActiveBlocks>
    CK_TILE_DEVICE void store_reg_output(const RegLowerBlocks& b,
                                         DataType* __restrict__ A_base,
                                         int tc,
                                         int T,
                                         int stride_A_t) const
    {
        const int32x4_t resource = make_wave_buffer_resource(A_base);
        store_reg_block<true>(b.s00, A_base, resource, 0, 0, tc, T, stride_A_t);
        store_reg_zero_block(A_base, resource, 0, 16, tc, T, stride_A_t);
        store_reg_zero_block(A_base, resource, 0, 32, tc, T, stride_A_t);
        store_reg_zero_block(A_base, resource, 0, 48, tc, T, stride_A_t);

        if constexpr(ActiveBlocks >= 2)
        {
            store_reg_block<false>(b.s10, A_base, resource, 16, 0, tc, T, stride_A_t);
            store_reg_block<true>(b.s11, A_base, resource, 16, 16, tc, T, stride_A_t);
            store_reg_zero_block(A_base, resource, 16, 32, tc, T, stride_A_t);
            store_reg_zero_block(A_base, resource, 16, 48, tc, T, stride_A_t);
        }

        if constexpr(ActiveBlocks >= 3)
        {
            store_reg_block<false>(b.s20, A_base, resource, 32, 0, tc, T, stride_A_t);
            store_reg_block<false>(b.s21, A_base, resource, 32, 16, tc, T, stride_A_t);
            store_reg_block<true>(b.s22, A_base, resource, 32, 32, tc, T, stride_A_t);
            store_reg_zero_block(A_base, resource, 32, 48, tc, T, stride_A_t);
        }

        if constexpr(ActiveBlocks >= 4)
        {
            store_reg_block<false>(b.s30, A_base, resource, 48, 0, tc, T, stride_A_t);
            store_reg_block<false>(b.s31, A_base, resource, 48, 16, tc, T, stride_A_t);
            store_reg_block<false>(b.s32, A_base, resource, 48, 32, tc, T, stride_A_t);
            store_reg_block<true>(b.s33, A_base, resource, 48, 48, tc, T, stride_A_t);
        }
    }

    template <int ActiveBlocks>
    CK_TILE_DEVICE void run_active_blocks(const DataType* __restrict__ k_base,
                                          DataType* __restrict__ A_base,
                                          int stride_k_t,
                                          int stride_A_t,
                                          float row_factor,
                                          float col_factor,
                                          int tc,
                                          int T,
                                          float* __restrict__ diag_work0,
                                          float* __restrict__ diag_work1) const
    {
        RegLowerBlocks b;
        compute_kkt_active_reg<ActiveBlocks>(
            k_base, stride_k_t, row_factor, col_factor, tc, T, b);
        invert_active_diags<ActiveBlocks>(b, diag_work0, diag_work1);
        solve_merge_reg<ActiveBlocks>(b);
        store_reg_output<ActiveBlocks>(b, A_base, tc, T, stride_A_t);
    }

    CK_TILE_DEVICE void operator()(
        const DataType* __restrict__ k_global,
        const float*    __restrict__ g_global,
        const float*    __restrict__ b_global,
        DataType*       __restrict__ A_global,
        int tc, int ih, int,
        int T, int H, int HV,
        int stride_k_t, int stride_g_t, int stride_A_t,
        bool use_exp2,
        SharedStorage& smem) const
    {
        const index_t warp = get_warp_id();
        const int ratio_hv = HV / H;
        const int ivh = ih * ratio_hv + warp;
        const DataType* k_base = k_global + tc * stride_k_t + ih * kHeadDim;

        const index_t lane = get_lane_id();
        float row_factor = 0.0f;
        float col_factor = 0.0f;
        if(tc + lane < T)
        {
            const float g = g_global[(tc + lane) * stride_g_t + ivh];
            row_factor = (use_exp2 ? __builtin_amdgcn_exp2f(g) : expf(g)) *
                         b_global[(tc + lane) * stride_g_t + ivh];
            col_factor = use_exp2 ? __builtin_amdgcn_exp2f(-g) : expf(-g);
        }

        DataType* A_base = A_global + tc * stride_A_t + ivh * kChunkSize;
        float* diag_work0 = smem.diag_work[warp][0];
        float* diag_work1 = smem.diag_work[warp][1];

        if constexpr(MaskKRows)
        {
            const int remaining = T - tc;
            if(remaining < kChunkSize)
            {
                if(remaining <= 16)
                {
                    run_active_blocks<1>(k_base,
                                         A_base,
                                         stride_k_t,
                                         stride_A_t,
                                         row_factor,
                                         col_factor,
                                         tc,
                                         T,
                                         diag_work0,
                                         diag_work1);
                }
                else if(remaining <= 32)
                {
                    run_active_blocks<2>(k_base,
                                         A_base,
                                         stride_k_t,
                                         stride_A_t,
                                         row_factor,
                                         col_factor,
                                         tc,
                                         T,
                                         diag_work0,
                                         diag_work1);
                }
                else if(remaining <= 48)
                {
                    run_active_blocks<3>(k_base,
                                         A_base,
                                         stride_k_t,
                                         stride_A_t,
                                         row_factor,
                                         col_factor,
                                         tc,
                                         T,
                                         diag_work0,
                                         diag_work1);
                }
                else
                {
                    run_active_blocks<4>(k_base,
                                         A_base,
                                         stride_k_t,
                                         stride_A_t,
                                         row_factor,
                                         col_factor,
                                         tc,
                                         T,
                                         diag_work0,
                                         diag_work1);
                }
                return;
            }
        }

        RegLowerBlocks b;
        b.s00 = compute_kkt_single_reg(k_base, 0, 0, stride_k_t,
                                       row_factor, col_factor, tc, T);
        compute_kkt_pair_reg(k_base, 16, 0, 16, stride_k_t,
                             row_factor, col_factor, tc, T, b.s10, b.s11);
        compute_kkt_pair_reg(k_base, 32, 0, 32, stride_k_t,
                             row_factor, col_factor, tc, T, b.s20, b.s22);
        b.s21 = compute_kkt_single_reg(k_base, 32, 16, stride_k_t,
                                       row_factor, col_factor, tc, T);
        compute_kkt_pair_reg(k_base, 48, 0, 16, stride_k_t,
                             row_factor, col_factor, tc, T, b.s30, b.s31);
        compute_kkt_pair_reg(k_base, 48, 32, 48, stride_k_t,
                             row_factor, col_factor, tc, T, b.s32, b.s33);
        invert_active_diags<4>(b, diag_work0, diag_work1);
        solve_merge_reg<4>(b);
        store_reg_output<4>(b, A_base, tc, T, stride_A_t);
    }
};

} // namespace ck_tile
