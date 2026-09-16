// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_pre_process_policy.hpp"
#include "ck_tile/ops/gdn/block/gdn_ck_pre_process_block_gemm.hpp"

namespace ck_tile {

template <typename Policy>
struct GdnPreProcessPipeline
{
    static constexpr int kChunkSize = Policy::kChunkSize;
    static constexpr int kHeadDim   = Policy::kHeadDim;

    using DataType = typename Policy::DataType;
    using AccType  = float;

    struct GemmProblem
    {
        using ADataType      = DataType;
        using BDataType      = DataType;
        using CDataType      = AccType;
        using BlockGemmShape = TileGemmShape<
            sequence<64, 64, 64>,
            sequence<4, 1, 1>,
            sequence<16, 64, 32>>;
        static constexpr index_t kBlockSize = Policy::kBlockSize;
    };
    using BlockGemmPolicy =
        GdnPreProcessARegBSmemPolicy<sequence<4, 1, 1>, typename Policy::WarpGemm>;
    using BlockGemm = GdnPreProcessBlockGemmARegBReg<GemmProblem, BlockGemmPolicy>;

    struct SharedStorage
    {
        DataType tmp_lds[kChunkSize * kChunkSize];
        float gate_scale[kChunkSize];
    };

    CK_TILE_DEVICE auto make_tmp_view(SharedStorage& smem) const
    {
        return make_naive_tensor_view<address_space_enum::lds>(
            smem.tmp_lds,
            make_tuple(number<kChunkSize>{}, number<kChunkSize>{}),
            make_tuple(number<kChunkSize>{}, number<1>{}),
            number<1>{},
            number<1>{});
    }

    template <typename Tile>
    CK_TILE_DEVICE void clear(Tile& tile) const
    {
        clear_tile(tile);
    }

    CK_TILE_DEVICE auto make_acc_tile() const
    {
        constexpr auto bg = BlockGemm{};
        auto c = BlockGemm::MakeCBlockTile();
        clear_tile(c);
        return bg.MakeOuputLayout(c);
    }

    template <typename Tile>
    CK_TILE_DEVICE void add_inplace(Tile& dst, const Tile& src) const
    {
        tile_elementwise_inout([](auto& x, const auto& y) { x += y; }, dst, src);
    }

    template <typename Tile>
    CK_TILE_DEVICE void sub_inplace(Tile& dst, const Tile& src) const
    {
        tile_elementwise_inout([](auto& x, const auto& y) { x -= y; }, dst, src);
    }

    template <typename Tile>
    CK_TILE_DEVICE void scale_rows(Tile& tile, SharedStorage& smem) const
    {
        constexpr auto spans = Tile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(tile.get_tile_distribution(), dstr_idx);
                const int r = tile_idx.at(number<0>{});
                const float s = smem.gate_scale[r];
                tile(dstr_idx) = type_convert<DataType>(type_convert<float>(tile(dstr_idx)) * s);
            });
        });
    }

    template <typename Tile>
    CK_TILE_DEVICE void scale_cols(Tile& tile, SharedStorage& smem) const
    {
        constexpr auto spans = Tile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(tile.get_tile_distribution(), dstr_idx);
                const int c = tile_idx.at(number<1>{});
                const float s = smem.gate_scale[c];
                tile(dstr_idx) = type_convert<DataType>(type_convert<float>(tile(dstr_idx)) * s);
            });
        });
    }

    template <typename Tile>
    CK_TILE_DEVICE void scale_rows_acc(Tile& tile, SharedStorage& smem) const
    {
        constexpr auto spans = Tile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(tile.get_tile_distribution(), dstr_idx);
                const int r = tile_idx.at(number<0>{});
                const float s = smem.gate_scale[r];
                tile(dstr_idx) *= s;
            });
        });
    }

    CK_TILE_DEVICE void load_gate_scale(const float* __restrict__ g_global,
                                        int tc,
                                        int ivh,
                                        int stride_g_t,
                                        float g_last,
                                        bool use_exp2,
                                        SharedStorage& smem) const
    {
        const index_t tid = get_thread_id();
        if(tid < kChunkSize)
        {
            const float gd = g_last - g_global[(tc + tid) * stride_g_t + ivh];
            smem.gate_scale[tid] = use_exp2 ? __builtin_amdgcn_exp2f(gd) : expf(gd);
        }
        block_sync_lds();
    }

    template <typename Tile>
    CK_TILE_DEVICE void scale_tile(Tile& tile, float s) const
    {
        tile_elementwise_inout([&](auto& x) { x *= s; }, tile);
    }

    template <typename Tile>
    CK_TILE_DEVICE void make_identity(Tile& tile, int col_base) const
    {
        constexpr auto spans = Tile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(tile.get_tile_distribution(), dstr_idx);
                const int r = tile_idx.at(number<0>{});
                const int c = col_base + tile_idx.at(number<1>{});
                tile(dstr_idx) = (r == c) ? 1.0f : 0.0f;
            });
        });
    }

    template <typename View>
    CK_TILE_DEVICE auto load_a(const View& view, int r, int c) const
    {
        auto tile = BlockGemm::MakeABlockTile();
        auto win = make_tile_window(
            view,
            make_tuple(number<64>{}, number<64>{}),
            multi_index<2>{r, c},
            tile.get_tile_distribution());
        load_tile(tile, win);
        return tile;
    }

    template <typename View>
    CK_TILE_DEVICE auto load_b_from_transposed_view(const View& view) const
    {
        auto b_view = make_naive_tensor_view<address_space_enum::lds>(
            view.get_buffer_view().p_data_,
            make_tuple(number<64>{}, number<64>{}),
            make_tuple(number<1>{}, number<64>{}),
            number<1>{},
            number<1>{});
        auto tile = BlockGemm::MakeBBlockTile();
        auto win = make_tile_window(
            b_view,
            make_tuple(number<64>{}, number<64>{}),
            multi_index<2>{0, 0},
            tile.get_tile_distribution());
        load_tile(tile, win);
        return tile;
    }

    template <typename Tile>
    CK_TILE_DEVICE auto c_to_a(const Tile& tile, SharedStorage& smem) const
    {
        auto view = make_tmp_view(smem);
        auto tmp = cast_tile<DataType>(tile);
        auto win = make_tile_window(
            view,
            make_tuple(number<64>{}, number<64>{}),
            multi_index<2>{0, 0},
            tmp.get_tile_distribution());
        store_tile(win, tmp);
        block_sync_lds();
        auto a = load_a(view, 0, 0);
        block_sync_lds();
        return a;
    }

    template <typename Tile>
    CK_TILE_DEVICE auto c_to_b(const Tile& tile, SharedStorage& smem) const
    {
        auto view = make_tmp_view(smem);
        auto tmp = cast_tile<DataType>(tile);
        auto win = make_tile_window(
            view,
            make_tuple(number<64>{}, number<64>{}),
            multi_index<2>{0, 0},
            tmp.get_tile_distribution());
        store_tile(win, tmp);
        block_sync_lds();
        auto b = load_b_from_transposed_view(view);
        block_sync_lds();
        return b;
    }

    template <typename GlobalView>
    CK_TILE_DEVICE auto load_scale_global_a(const GlobalView& global_view,
                                            SharedStorage& smem) const
    {
        auto tile = BlockGemm::MakeABlockTile();
        auto win = make_tile_window(
            global_view,
            make_tuple(number<64>{}, number<64>{}),
            multi_index<2>{0, 0},
            tile.get_tile_distribution());
        load_tile(tile, win);
        scale_cols(tile, smem);
        return tile;
    }

    template <typename ATile, typename BTile>
    CK_TILE_DEVICE auto gemm(const ATile& a, const BTile& b) const
    {
        constexpr auto bg = BlockGemm{};
        auto c = BlockGemm::MakeCBlockTile();
        clear_tile(c);
        bg(c, a, b);
        return bg.MakeOuputLayout(c);
    }

    template <typename Tile>
    CK_TILE_DEVICE void store_hm(Tile& tile,
                                 float* __restrict__ hm_global,
                                 int ivh,
                                 int row_base,
                                 int col_base) const
    {
        constexpr auto spans = Tile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(tile.get_tile_distribution(), dstr_idx);
                const int r = row_base + tile_idx.at(number<0>{});
                const int c = col_base + tile_idx.at(number<1>{});
                if(r < kHeadDim && c < kHeadDim + kHeadDim)
                {
                    hm_global[ivh * kHeadDim * (kHeadDim + kHeadDim) +
                              r * (kHeadDim + kHeadDim) + c] = tile(dstr_idx);
                }
            });
        });
    }

    CK_TILE_DEVICE void compute_h(
        const DataType* __restrict__ k_global,
        const DataType* __restrict__ w_global,
        const DataType* __restrict__ u_global,
        const float* __restrict__ g_global,
        float* __restrict__ hm_global,
        int i_v_block,
        int ih,
        int ivh,
        int T,
        int stride_k_t,
        int stride_w_t,
        int stride_u_t,
        int stride_g_t,
        bool use_exp2,
        SharedStorage& smem) const
    {
        auto h0 = make_acc_tile();
        auto h1 = make_acc_tile();
        clear(h0);
        clear(h1);

        constexpr auto bg = BlockGemm{};
        for(int tc = 0; tc < T; tc += kChunkSize)
        {
            const int last = tc + kChunkSize - 1;
            const float g_last_raw = g_global[last * stride_g_t + ivh];
            const float g_last = use_exp2 ? __builtin_amdgcn_exp2f(g_last_raw) : expf(g_last_raw);
            load_gate_scale(g_global, tc, ivh, stride_g_t, g_last_raw, use_exp2, smem);

            auto w_view = make_naive_tensor_view<address_space_enum::global>(
                w_global + tc * stride_w_t + ivh * kHeadDim,
                make_tuple(number<64>{}, number<kHeadDim>{}),
                make_tuple(stride_w_t, number<1>{}),
                number<8>{},
                number<1>{});
            auto u_view = make_naive_tensor_view<address_space_enum::global>(
                u_global + tc * stride_u_t + ivh * kHeadDim + i_v_block * 64,
                make_tuple(number<64>{}, number<64>{}),
                make_tuple(stride_u_t, number<1>{}),
                number<8>{},
                number<1>{});

            auto v_decay = gemm(load_a(w_view, 0, 0), c_to_b(h0, smem));
            auto v_decay1 = gemm(load_a(w_view, 0, 64), c_to_b(h1, smem));
            add_inplace(v_decay, v_decay1);

            auto u_tile_raw = load_tile(make_tile_window(
                u_view,
                make_tuple(number<64>{}, number<64>{}),
                multi_index<2>{0, 0},
                v_decay.get_tile_distribution()));
            auto u_tile = cast_tile<AccType>(u_tile_raw);
            sub_inplace(u_tile, v_decay);
            scale_rows_acc(u_tile, smem);
            auto v_new_b = c_to_b(u_tile, smem);
            scale_tile(h0, g_last);
            scale_tile(h1, g_last);

            auto k0_view = make_naive_tensor_view<address_space_enum::global>(
                k_global + tc * stride_k_t + ih * kHeadDim,
                make_tuple(number<64>{}, number<64>{}),
                make_tuple(number<1>{}, stride_k_t),
                number<1>{},
                number<8>{});
            auto k1_view = make_naive_tensor_view<address_space_enum::global>(
                k_global + tc * stride_k_t + ih * kHeadDim + 64,
                make_tuple(number<64>{}, number<64>{}),
                make_tuple(number<1>{}, stride_k_t),
                number<1>{},
                number<8>{});

            add_inplace(h0, gemm(load_a(k0_view, 0, 0), v_new_b));
            add_inplace(h1, gemm(load_a(k1_view, 0, 0), v_new_b));
        }

        store_hm(h0, hm_global, ivh, 0, i_v_block * 64);
        store_hm(h1, hm_global, ivh, 64, i_v_block * 64);
    }

    CK_TILE_DEVICE void compute_m(
        const DataType* __restrict__ k_global,
        const DataType* __restrict__ w_global,
        const float* __restrict__ g_global,
        float* __restrict__ hm_global,
        int i_k_col,
        int ih,
        int ivh,
        int T,
        int stride_k_t,
        int stride_w_t,
        int stride_g_t,
        bool use_exp2,
        SharedStorage& smem) const
    {
        auto m0 = make_acc_tile();
        auto m1 = make_acc_tile();
        clear(m0);
        clear(m1);
        make_identity(m0, 0);
        if(i_k_col == 1)
        {
            m1 = m0;
            clear(m0);
        }
        else
        {
            clear(m1);
        }

        for(int tc = 0; tc < T; tc += kChunkSize)
        {
            const int last = tc + kChunkSize - 1;
            const float g_last_raw = g_global[last * stride_g_t + ivh];
            const float g_last = use_exp2 ? __builtin_amdgcn_exp2f(g_last_raw) : expf(g_last_raw);
            load_gate_scale(g_global, tc, ivh, stride_g_t, g_last_raw, use_exp2, smem);

            auto w0_view = make_naive_tensor_view<address_space_enum::global>(
                w_global + tc * stride_w_t + ivh * kHeadDim,
                make_tuple(number<64>{}, number<64>{}),
                make_tuple(stride_w_t, number<1>{}),
                number<8>{},
                number<1>{});
            auto w1_view = make_naive_tensor_view<address_space_enum::global>(
                w_global + tc * stride_w_t + ivh * kHeadDim + 64,
                make_tuple(number<64>{}, number<64>{}),
                make_tuple(stride_w_t, number<1>{}),
                number<8>{},
                number<1>{});
            auto w0_a = load_a(w0_view, 0, 0);
            auto w1_a = load_a(w1_view, 0, 0);

            auto k0_view = make_naive_tensor_view<address_space_enum::global>(
                k_global + tc * stride_k_t + ih * kHeadDim,
                make_tuple(number<64>{}, number<64>{}),
                make_tuple(number<1>{}, stride_k_t),
                number<1>{},
                number<8>{});
            auto k1_view = make_naive_tensor_view<address_space_enum::global>(
                k_global + tc * stride_k_t + ih * kHeadDim + 64,
                make_tuple(number<64>{}, number<64>{}),
                make_tuple(number<1>{}, stride_k_t),
                number<1>{},
                number<8>{});
            auto k0_a = load_scale_global_a(k0_view, smem);
            auto k1_a = load_scale_global_a(k1_view, smem);
            auto wm = gemm(w0_a, c_to_b(m0, smem));
            auto w1m1 = gemm(w1_a, c_to_b(m1, smem));
            add_inplace(wm, w1m1);

            auto wm_b = c_to_b(wm, smem);
            auto km0 = gemm(k0_a, wm_b);
            auto km1 = gemm(k1_a, wm_b);

            scale_tile(m0, g_last);
            scale_tile(m1, g_last);
            sub_inplace(m0, km0);
            sub_inplace(m1, km1);
        }

        store_hm(m0, hm_global, ivh, 0, kHeadDim + i_k_col * 64);
        store_hm(m1, hm_global, ivh, 64, kHeadDim + i_k_col * 64);
    }

    CK_TILE_DEVICE void operator()(
        const DataType* __restrict__ k_global,
        const DataType* __restrict__ w_global,
        const DataType* __restrict__ u_global,
        const float* __restrict__ g_global,
        float* __restrict__ hm_global,
        int i_col,
        int ih,
        int ivh,
        int T,
        int stride_k_t,
        int stride_w_t,
        int stride_u_t,
        int stride_g_t,
        bool use_exp2,
        SharedStorage& smem) const
    {
        if(i_col < 2)
        {
            compute_h(k_global, w_global, u_global, g_global, hm_global,
                      i_col, ih, ivh, T,
                      stride_k_t, stride_w_t, stride_u_t, stride_g_t,
                      use_exp2, smem);
        }
        else
        {
            compute_m(k_global, w_global, g_global, hm_global,
                      i_col - 2, ih, ivh, T,
                      stride_k_t, stride_w_t, stride_g_t,
                      use_exp2, smem);
        }
    }
};

} // namespace ck_tile
