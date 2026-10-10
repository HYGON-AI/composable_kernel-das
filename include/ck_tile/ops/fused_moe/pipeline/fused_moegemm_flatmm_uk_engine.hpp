// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_policy.hpp"

namespace ck_tile {

/*
This pipeline deal with a gemm(actually 2 gemm) with one very small(token), one very big(weight)
we need to design the pipeline such that all waves along gemm-N dim (gemm-m only 1 wave)

    <----- gemm-N ------>
    +----+----+----+----+
    | w0 | w1 | w2 | w3 | gemm-m
    +----+----+----+----+
*/
template <typename Problem_, typename Policy_ = FusedMoeGemmPipelineFlatmmPolicy>
struct FusedMoeGemmPipeline_FlatmmUkEngine
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockShape = typename Problem::BlockShape; // this is FusedMoeGemmShape

    using ADataType            = typename Problem::ADataType;
    using GDataType            = typename Problem::GDataType;
    using DDataType            = typename Problem::DDataType;
    using AccDataType          = typename Problem::AccDataType;
    using ODataType            = typename Problem::ODataType;
    using AScaleDataType       = typename Problem::AScaleDataType;
    using GScaleDataType       = typename Problem::GScaleDataType;
    using DScaleDataType       = typename Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Problem::TopkWeightDataType;
    using IndexDataType        = typename Problem::IndexDataType;
    using YDataType            = typename Problem::YDataType;

    using Traits = typename Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();
    static constexpr index_t NumPrefetch = Problem::NumPrefetch;

    static constexpr index_t SLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::SLD_A);
    static constexpr index_t GLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_A);
    static constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
    static constexpr index_t GST_O = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GST_O);

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "fused_moe_flatmm";
    static constexpr bool kSupportsDeterministicFp64Output = true;

    // TODO: there are multiple buffers
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize_A()
    {
        return Policy::template GetSmemSize_A<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetACoord()
    {
        constexpr auto a_dist = Policy::template MakeGlobalTileDistribution_A<Problem>();
        const auto a_coord = a_dist.calculate_index(); // tile distribution index
        return a_coord;
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetOCoord()
    {
        constexpr auto o_dist = Policy::template MakeGlobalTileDistribution_O<Problem>();
        const auto o_coord    = o_dist.calculate_index();
        return o_coord;
    }

    template <typename AWindow,
              typename GWindow,
              typename DWindow,
              typename Epilogue>
    CK_TILE_DEVICE auto
    operator()(const AWindow& a_window_,
               const GWindow& g_window_,
               const DWindow& d_window_,
               const Epilogue& epilogue_,
               CK_TILE_LDS_ADDR void* smem,
               index_t hidden_size,
               index_t intermediate_size,
               index_t intermediate_col_base)
    {
        _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wc++20-extensions\"");
        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};
        constexpr auto I2 = number<2>{};
        constexpr auto I3 = number<3>{};

        CK_TILE_LDS_ADDR ADataType* smem_0 = reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem);

        // real weight rows/columns of this intermediate tile (gemm0 N / gemm1 K
        // extent minus the tile's column offset). Partial tiles (intermediate
        // size not a multiple of Block_N0) must zero-fill beyond this count:
        // gate rows would otherwise read the up half / the next expert, and the
        // manual D load would read the next row's leading columns (or past the
        // W2 buffer for the last expert) - silently wrong results.
        const index_t interm_valid = intermediate_size - intermediate_col_base;

        auto g_view = g_window_.get_bottom_tensor_view();

        //
        auto u_view = [&]() {
            if constexpr(IsGateOnly)
            {
                // FIXME:
                static_assert(false, "GateOnly is not supported yet!");
                return g_view;
            }
            else
            {
                const GDataType* g_ptr =
                    g_window_.get_bottom_tensor_view().get_buffer_view().p_data_;

                const GDataType* u_ptr = g_ptr + intermediate_size * hidden_size;

                // declare only the real up-half rows; the pad transform
                // zero-fills the tail of a partial tile (see interm_valid note)
                const auto u_view_ = make_naive_tensor_view<address_space_enum::global>(
                    u_ptr,
                    make_tuple(interm_valid, hidden_size),
                    make_tuple(hidden_size, 1),
                    number<kAlignmentG>{},
                    number<1>{});

                const auto u_view_1_ = pad_tensor_view(
                    u_view_,
                    make_tuple(number<BlockShape::Block_N0>{},
                               number<2 * BlockShape::Block_K0>{}),
                    sequence<true, PadHiddenSize>{});
                return u_view_1_;
            }
        }();

        // input global load to lds components
        auto a_copy_dram_window = make_tile_window(
            a_window_.get_bottom_tensor_view(),
            make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
            a_window_.get_window_origin(),
            Policy::template MakeGlobalTileDistribution_A<Problem>());
        constexpr auto a_copy_lds_block_desc = Policy::template MakeLdsStoreDesc_A<Problem>();
        constexpr auto a_lds_space_size      = ck_tile::integer_least_multiple(
            Policy::template GetSmemSize_A<Problem>(), BlockShape::Block_K0);
        // constexpr auto a_lds_bufs_elem_offset = 0;
        auto a_copy_lds_window0 = make_tile_window(
            make_tensor_view<address_space_enum::lds>(smem_0, a_copy_lds_block_desc),
            make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
            {0, 0});
        auto a_copy_lds_window1 = make_tile_window(
            make_tensor_view<address_space_enum::lds>(smem_0 + a_lds_space_size,
                                                      a_copy_lds_block_desc),
            make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
            {0, 0});
        auto a_copy_lds_windows = make_tuple(a_copy_lds_window0, a_copy_lds_window1);

        // gate load from global to vgpr then to lds components
        auto g_copy_dram_window = make_tile_window(
            g_window_.get_bottom_tensor_view(), // [hidden_states, moe_intermediate_size]
            make_tuple(number<BlockShape::Block_N0>{},
                       number<BlockShape::Block_K0>{}), //[BLOCK_N0, BLOKC_K0]
            g_window_.get_window_origin(),
            Policy::template MakeGlobalTileDistribution_G<Problem>());

        /*
        //if need write to lds, use this part
        constexpr auto g_copy_lds_block_desc            = Policy::template
        MakeLdsStoreDesc_G<Problem>(); constexpr auto g_lds_space_size                 =
        Policy::template GetSmemSize_G<Problem>(); constexpr auto g_lds_bufs_elem_offset           =
        a_lds_space_size * NumPrefetch; auto g_copy_lds_block                 =
        make_tensor_view<address_space_enum::lds>(smem_0 + a_lds_bufs_elem_offset,
                                                                                          g_copy_lds_block_desc);
        auto g_copy_lds_window = make_tile_window(
            g_copy_lds_block, make_tuple(number<BlockShape::Block_N0>{},
        number<BlockShape::Block_K0>{}), {0, 0});
        */

        // up proj reuse gate LDS and same as gate proj, all vgpr and lds layout
        auto u_copy_dram_window = make_tile_window(
            u_view,
            make_tuple(number<BlockShape::Block_N0>{}, number<BlockShape::Block_K0>{}),
            {0, 0},
            Policy::template MakeGlobalTileDistribution_G<Problem>());
        /*
        // For LDS flow
        constexpr auto u_copy_lds_block_desc            = Policy::template
        MakeLdsStoreDesc_G<Problem>(); constexpr auto u_lds_space_size                 =
        Policy::template GetSmemSize_G<Problem>(); constexpr auto g_lds_bufs_elem_offset           =
        a_lds_space_size * NumPrefetch; auto g_copy_lds_block                 =
        make_tensor_view<address_space_enum::lds>(smem_0 + a_lds_bufs_elem_offset,
                                                                                          g_copy_lds_block_desc);
        auto g_copy_lds_window = make_tile_window(
            g_copy_lds_block, make_tuple(number<BlockShape::Block_N0>{},
        number<BlockShape::Block_K0>{}), {0, 0});
        */

        // gemm0 from LDS to VGPR
        // TODO: support get_sub_warp_id for WASP
        using WarpGemm0  = decltype(Policy::template GetWarpGemm0<Problem>());
        auto warp_gemm_0 = WarpGemm0{};
        // input
        // tile_windows here are for warps
        constexpr auto a_gemm_sld_block_desc = Policy::template MakeLdsLoadDesc_A<Problem>();
        auto a_sld_win0                      = [&]() {
            using WG = WarpGemm0;
            auto a_gemm_sld_block =
                make_tensor_view<address_space_enum::lds>(smem_0, a_gemm_sld_block_desc);
            return make_tile_window(
                a_gemm_sld_block,
                a_gemm_sld_block_desc.get_lengths(),
                {0, 0},
                make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));
        }();
        auto a_sld_win1 = [&]() { // Prefetch window
            using WG              = WarpGemm0;
            auto a_gemm_sld_block = make_tensor_view<address_space_enum::lds>(
                smem_0 + a_lds_space_size, a_gemm_sld_block_desc);
            return make_tile_window(
                a_gemm_sld_block,
                a_gemm_sld_block_desc.get_lengths(),
                {0, 0},
                make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));
        }();

        // weight
        /*
        //TODO: Enable for LDS solution
        constexpr auto gu_gemm_sld_block_desc = Policy::template MakeLdsLoadDesc_G<Problem>();
        auto gu_gemm_sld_block           = make_tensor_view<address_space_enum::lds>(smem_0 +
        g_lds_bufs_elem_offset, g_gemm_sld_block_desc); auto gu_gemm_sld_block_window    =
        make_tile_window(g_gemm_sld_block, g_gemm_sld_block_desc.get_lengths(), {0, 0}); warp window
        auto gu_sld_win = [&](){
            using WG                        = WarpGemm0;
            return make_tile_window(
                gu_gemm_sld_block_window.get_bottom_tensor_view(),
                make_tuple(number<WG::kN>{}, number<WG::kK>{}),
                gu_gemm_sld_block_window.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
                make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));
        }();
        */

        // gemm1 weight global to vgpr
        // TODO: replace buffer vgpr to buffer lds
        auto d_copy_dram_window = make_tile_window(
            d_window_.get_bottom_tensor_view(), // [hidden_states, moe_intermediate_size]
            make_tuple(number<BlockShape::Block_N1>{},
                       number<BlockShape::Block_K1>{}), //[BLOCK_N1, BLOCK_K1]
            d_window_.get_window_origin(),
            Policy::template MakeGlobalTileDistribution_D<Problem>());

        // D tile moves with explicit pointer math (calculate_index() supplies
        // the per-thread base): x0 = 8*y0 + (2*warp + lane/32), x1 = 8*(lane%32)
        // + y1 for the global distribution; the LDS tile is packed [N1, K1]
        // row-major; the warp-B read back is row = 16*warp + lane%16, col =
        // 16*kiter + 4*(lane/16) + kper. (This bypasses tile-window
        // load/store on this path; both layouts were verified on hw with
        // position-encoded weights.)
        const DDataType* d_gptr = d_window_.get_bottom_tensor_view().get_buffer_view().p_data_;
        const auto d_idx_base   = d_copy_dram_window.get_tile_distribution().calculate_index();

        // gemm 1 vgpr to lds: the LDS tile is a packed [Block_N1, Block_K1]
        // row-major block written/read with explicit pointer math below
        CK_TILE_LDS_ADDR DDataType* smem_1 = reinterpret_cast<CK_TILE_LDS_ADDR DDataType*>(smem);

        // gemm1 lds to vgpr
        using WarpGemm1  = decltype(Policy::template GetWarpGemm1<Problem>());
        auto warp_gemm_1 = WarpGemm1{};

        auto d_gemm_sld_block_desc = Policy::template MakeLdsLoadDesc_D<Problem>();
        auto d_sld_win             = [&]() {
            using WG                        = WarpGemm1;
            constexpr auto d_outer_dstr_enc = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<BlockShape::Repeat_N1, BlockShape::WarpPerBlock_N1>,
                      sequence<BlockShape::Repeat_K1>>,
                tuple<sequence<1>>,
                tuple<sequence<1>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto d_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                d_outer_dstr_enc, typename WG::BWarpDstrEncoding{});
            auto d_gemm_sld_block =
                make_tensor_view<address_space_enum::lds>(smem_1, d_gemm_sld_block_desc);
            auto d_gemm_sld_win =
                make_tile_window(d_gemm_sld_block,
                                 d_gemm_sld_block_desc.get_lengths(),
                                 {0, 0},
                                 make_static_tile_distribution(d_block_dstr_encode));
            return d_gemm_sld_win;
        }();

        // output column offset of the current gemm1 N-iteration, advanced by
        // atomic_add_o() after every [Block_M1, Block_N1] C tile
        index_t n_col = 0;

        auto bridge_sst_win = [&]() {
            return make_tile_window(
                make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<YDataType*>(smem),
                    Policy::template MakeBridgeLdsStoreDesc<Problem>()),
                Policy::template MakeBridgeLdsStoreDesc<Problem>().get_lengths(),
                {0, 0});
        }();

        auto bridge_sld_win = [&]() {
            return make_tile_window(make_tensor_view<address_space_enum::lds>(
                                        reinterpret_cast<YDataType*>(smem),
                                        Policy::template MakeBridgeLdsLoadDesc<Problem>()),
                                    Policy::template MakeBridgeLdsLoadDesc<Problem>().get_lengths(),
                                    {0, 0},
                                    Policy::template MakeYTileDistribution<Problem>());
        }();

        // also OK with C array, 2 register buffer, prefetch u or prefetch g
        // statically_indexed_array<g_thread_type, 2> gs;

        // constexpr auto issues_a = number<a_copy_dram_window.get_num_of_access()>{};
        constexpr auto issues_g = number<g_copy_dram_window.get_num_of_access()>{};
        // constexpr auto issues_d = number<d_win.get_num_of_access()>{};
        // constexpr auto issues_o = number<o_win.get_num_of_access()>{};

        // Both manually double-buffered tails below consume two tiles.  Pad-enabled
        // shapes must therefore round the loop count to an even number of tiles;
        // merely rounding to one tile drops the last K0 tile for odd counts (for
        // example 1232 / 32 -> 39, while the old loop executed only 38).
        const index_t num_blocks_k0 = PadHiddenSize
                                          ? ((hidden_size + 2 * BlockShape::Block_K0 - 1) /
                                             (2 * BlockShape::Block_K0)) *
                                                2
                                          : (hidden_size + BlockShape::Block_K0 - 1) /
                                                BlockShape::Block_K0;
        const index_t num_blocks_n1 = PadHiddenSize
                                          ? ((hidden_size + 2 * BlockShape::Block_N1 - 1) /
                                             (2 * BlockShape::Block_N1)) *
                                                2
                                          : (hidden_size + BlockShape::Block_N1 - 1) /
                                                BlockShape::Block_N1;

        using a_thread_type = decltype(load_tile(a_sld_win0));
        using g_thread_type = decltype(load_tile(g_copy_dram_window));
        using d_thread_type = decltype(load_tile(d_copy_dram_window));
        // using sld_d_thread_type = decltype(load_tile(d_sld_win));
        statically_indexed_array<a_thread_type, 1> a;
        statically_indexed_array<g_thread_type, 4> gs;
        statically_indexed_array<d_thread_type, 2> gd;
        // statically_indexed_array<sld_d_thread_type,1> sd;

        auto gld_a_init = [&]() { a_copy_dram_window.init_raw(); };

        auto gld_a = [&](auto& a_store_) {
            async_load_tile_by_inline_asm(a_store_, a_copy_dram_window);
        };
        auto move_a = [&]() {
            move_tile_window(a_copy_dram_window, {number<0>{}, number<BlockShape::Block_K0>{}});
        };
        auto sld_a = [&](auto& a_, auto& win_) { load_tile(a_, win_); };

        auto gld_g_init = [&]() { g_copy_dram_window.init_raw(); };

        auto gld_g = [&](auto& g_) {
            if constexpr(IsGateOnly)
            {
                // TODO
            }
            load_tile(g_, g_copy_dram_window);
        };
        auto move_g = [&]() {
            move_tile_window(g_copy_dram_window, {number<0>{}, number<BlockShape::Block_K0>{}});
        };

        auto gld_u_init = [&]() { u_copy_dram_window.init_raw(); };

        auto gld_u = [&](auto& u_) {
            if constexpr(IsGateOnly)
            {
                load_tile(u_, g_copy_dram_window);
            }
            else
            {
                load_tile(u_, u_copy_dram_window);
            }
        };
        auto move_u = [&]() {
            if constexpr(IsGateOnly)
            {
                move_tile_window(g_copy_dram_window, {number<0>{}, number<BlockShape::Block_K0>{}});
            }
            else
            {
                move_tile_window(u_copy_dram_window, {number<0>{}, number<BlockShape::Block_K0>{}});
            }
        };
        // statically_indexed_array<d_thread_type, 2> ds;
        auto gld_d_init = [&]() { d_copy_dram_window.init_raw(); };
        auto gld_d      = [&](auto& d_) {
#ifdef FMOE_UK_DEBUG_G
            if(get_warp_id() == 0 && (get_lane_id() == 0 || get_lane_id() == 32))
                printf("VEXP w=%d lane=%d org=(%d,%d) base=(%d,%d) is=%d hs=%d d0=%f\n",
                       (int)get_warp_id(),
                       (int)get_lane_id(),
                       (int)d_copy_dram_window.get_window_origin()[number<0>{}],
                       (int)d_copy_dram_window.get_window_origin()[number<1>{}],
                       (int)d_idx_base[number<0>{}],
                       (int)d_idx_base[number<1>{}],
                       (int)intermediate_size,
                       (int)hidden_size,
                       (float)d_gptr[0]);
#endif
            const auto org     = d_copy_dram_window.get_window_origin();
            const index_t row0 = org[number<0>{}] + d_idx_base[number<0>{}];
            const index_t col0 = org[number<1>{}] + d_idx_base[number<1>{}];
            auto& buf          = d_.get_thread_buffer();
            static_for<0, 64, 1>{}([&](auto j) {
                constexpr index_t y0 = j / 8;
                constexpr index_t y1 = j % 8;
                const index_t r      = row0 + 8 * y0;
                const index_t c      = col0 + y1;
                // c is a tile-local column; guard it against the real interm
                // extent (window base already carries intermediate_col_base)
                buf[j] = (r < hidden_size && c < interm_valid)
                             ? d_gptr[static_cast<size_t>(r) * intermediate_size + c]
                             : DDataType{0};
            });
#ifdef FMOE_UK_DEBUG_G
            if(get_warp_id() == 0 && (get_lane_id() == 0 || get_lane_id() == 32))
                printf("VEXP w=%d lane=%d org=(%d,%d) base=(%d,%d) is=%d hs=%d "
                       "d1=%f d256=%f d257=%f b0=%f\n",
                       (int)get_warp_id(),
                       (int)get_lane_id(),
                       (int)d_copy_dram_window.get_window_origin()[number<0>{}],
                       (int)d_copy_dram_window.get_window_origin()[number<1>{}],
                       (int)d_idx_base[number<0>{}],
                       (int)d_idx_base[number<1>{}],
                       (int)intermediate_size,
                       (int)hidden_size,
                       (float)d_gptr[1],
                       (float)d_gptr[256],
                       (float)d_gptr[257],
                       (float)buf[number<0>{}]);
#endif
        };
        // VGPR -> LDS, same (row, col) each element was gathered from
        auto d_vgpr_to_lds = [&](auto& d_) {
            auto& buf = d_.get_thread_buffer();
            static_for<0, 64, 1>{}([&](auto j) {
                constexpr index_t y0 = j / 8;
                constexpr index_t y1 = j % 8;
                const index_t r      = d_idx_base[number<0>{}] + 8 * y0;
                const index_t c      = d_idx_base[number<1>{}] + y1;
                *(smem_1 + r * BlockShape::Block_K1 + c) = buf[j];
            });
        };
        // LDS -> VGPR in the embedded warp-B layout expected by gemm_1()
        auto d_lds_to_vgpr = [&]() {
            using DSl = decltype(load_tile(d_sld_win)); // type only, never executed
            DSl d_tmp;
            const index_t rr = 16 * get_warp_id() + get_lane_id() % 16;
            const index_t c0 = 4 * (get_lane_id() / 16);
            auto& buf        = d_tmp.get_thread_buffer();
            static_for<0, 64, 1>{}([&](auto j) {
                constexpr index_t kiter = j / 4;
                constexpr index_t kper  = j % 4;
                buf[j] = *(smem_1 + rr * BlockShape::Block_K1 + 16 * kiter + c0 + kper);
            });
            return d_tmp;
        };
        auto move_d     = [&]() {
            // d move along gemm-n
            move_tile_window(d_copy_dram_window, {number<BlockShape::Block_N1>{}, number<0>{}});
        };

        // Per-row topk-weighted fp32 atomic scatter of one [Block_M1, Block_N1] C tile.
        //
        // The fork's original epilogue drove this through [4, Block_N1] one-shot
        // indexing-transform scatter windows plus a per-issue scalar weight, whose
        // slot arithmetic (o_coord[0] + i*4) was derived for an MR1 C layout and
        // the policy's output distribution hardcodes MRepeat=1. Under the HCU
        // transposed-C warp layout (MR2: row = 16*MRepeat + 4*(lane/16) + CM1)
        // that arithmetic misassigns rows to tokens. This port instead scatters
        // explicitly: each thread owns 8 C elements with thread-buffer index
        // d = MRepeat*4 + CM1PerLane-index, and the row's own sorted slot
        // supplies both the destination token and its topk weight.
        auto atomic_add_o = [&](auto& o_) {
            using WarpGemm = typename remove_cvref_t<decltype(warp_gemm_1)>::WarpGemmAttribute;
            static_assert(WarpGemm::MRepeat == 2 && WarpGemm::NRepeat == 1 &&
                              WarpGemm::MInterleave == 1 && WarpGemm::NInterleave == 1,
                          "epilogue row/col math locked to MR2/NR1/MI1/NI1");
            static_assert(BlockShape::Repeat_M1 == 1 && BlockShape::WarpPerBlock_M1 == 1 &&
                              BlockShape::Repeat_N1 == 1 && BlockShape::WarpPerBlock_N1 == 4,
                          "epilogue row/col math locked to WarpPerBlock_1 = <1,4,1>");

            const index_t cml   = get_lane_id() / 16;
            const index_t col   = 16 * get_warp_id() + get_lane_id() % 16 + n_col;
            const index_t row_b = epilogue_.row_base + cml;
            auto& buf           = o_.get_thread_buffer();
            static_for<0, 8, 1>{}([&](auto jj) {
                constexpr index_t mrep = jj / 4;
                constexpr index_t cm1  = jj % 4;
                // HCU transposed-C layout (probed on hw): buffer index jj covers
                // row 16*mrep + 4*cm1 + (lane/16); CMLane has stride 1 and the
                // per-lane M1 repeat steps by 4 rows.
                const index_t row = row_b + 16 * mrep + 4 * cm1;
                index_t token     = epilogue_.sorted_ids[row];
#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
                // repo sorting packs token | (topk_idx << 24) when mock ids are
                // enabled (CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID, default on in
                // core/config.hpp); strip the tag exactly like
                // FusedMoeGemmKernel does, otherwise every id looks out of
                // range, the range guard drops ALL writes and o stays zero.
                token &= 0xffffff;
#endif
                const float weight = epilogue_.sorted_weights[row];
                if(token >= 0 && token < epilogue_.num_tokens && col < epilogue_.hidden_size)
                {
                    atomicAdd(epilogue_.o_ptr +
                                  static_cast<size_t>(token) * epilogue_.stride_token + col,
                              static_cast<float>(buf[jj]) * weight);
                }
            });
            n_col += BlockShape::Block_N1;
        };

        // tuple(gate, up)
        auto acc_0 = make_tuple(Policy::template MakeCBlockTile_Gemm0<Problem>(),
                                Policy::template MakeCBlockTile_Gemm0<Problem>());
        // auto acc_1 = Policy::template MakeCBlockTile_Gemm1<Problem>();
        auto acc_1s = generate_tuple(
            [&](auto) { return Policy::template MakeCBlockTile_Gemm1<Problem>(); }, number<2>{});

        // clang-format off
        //t_c refers to all gate/up_proj results in 1 thread.(VGPR), t_a, t_b have similar meaning.
        auto gemm_0 = [&](auto& t_c, auto& t_a, auto& t_b) {
            using WarpGemm = remove_cvref_t<decltype(warp_gemm_0)>;

            using AWarpTensor = typename WarpGemm::AWarpTensor;
            using BWarpTensor = typename WarpGemm::BWarpTensor;
            using CWarpTensor = typename WarpGemm::CWarpTensor;
            using CWarpDstr = typename WarpGemm::CWarpDstr;

            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

            //w_a is same as t_a
            AWarpTensor w_a;
            w_a.get_thread_buffer() = t_a.get_thread_buffer();

            //w_b is same as t_b due to t_b has N x K = 4 x 8 
            BWarpTensor w_b;
            w_b.get_thread_buffer() = t_b.get_thread_buffer();
            CWarpTensor w_c;

            //t_c is blockwise tensor, has repeat and warps. 0,0 here refers to m,n repeat for now only 1 is supported
            // <0,0,0,2> -> offset
            w_c.get_thread_buffer() = t_c.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, 0>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
            warp_gemm_0(w_c, w_a, w_b);
            
            //direct copy, we may use set_thread_buffer?
            //Note: Layout here is MRepeat=2, NRpeat=4(4interleave)
            t_c.set_y_sliced_thread_data(
                        merge_sequences(sequence<0, 0>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        w_c.get_thread_buffer());
        };
        // clang-format on

        // clang-format off
        auto gemm_1 = [&](auto& t_c, auto& t_a, auto& t_b) {
            using WarpGemm = remove_cvref_t<decltype(warp_gemm_1)>;

            using AWarpTensor = typename WarpGemm::AWarpTensor;
            using BWarpTensor = typename WarpGemm::BWarpTensor;
            using CWarpTensor = typename WarpGemm::CWarpTensor;
            using CWarpDstr = typename WarpGemm::CWarpDstr;

            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

            //w_a is same as t_a
            AWarpTensor w_a;
            w_a.get_thread_buffer() = t_a.get_thread_buffer();

            //w_b is same as t_b due to t_b has N x K = 4 x 8 
            BWarpTensor w_b;
            w_b.get_thread_buffer() = t_b.get_thread_buffer();
            CWarpTensor w_c;
            //t_c is blockwise tensor, has repeat and warps. 0,0 here refers to m,n repeat for now only 1 is supported
            w_c.get_thread_buffer() = t_c.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, 0>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
            warp_gemm_1(w_c, w_a, w_b);
            
            //direct copy
            t_c.set_y_sliced_thread_data(
                        merge_sequences(sequence<0, 0>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        w_c.get_thread_buffer());
        };
        // clang-format on
        _Pragma("clang diagnostic pop");

        auto pipeline_gemm0 = [&]() {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(0);
            block_sync_load_raw(2 * issues_g);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            sld_a(a[I0], a_sld_win0);
            gld_a(a_copy_lds_windows(I1));
            __builtin_amdgcn_sched_barrier(0);

            move_a();
            gld_g(gs[I2]);
            move_g();
            gld_u(gs[I3]);
            move_u();

            // pre-load a and next u/g
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(1);

            // first mmac group
            gemm_0(acc_0[I0], a[I0], gs[I0]);
            gemm_0(acc_0[I1], a[I0], gs[I1]);

            // second mmac group
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(0);
            block_sync_load_raw(2 * issues_g);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            sld_a(a[I0], a_sld_win1);
            gld_a(a_copy_lds_windows(I0));
            __builtin_amdgcn_sched_barrier(0);

            move_a();
            gld_g(gs[I0]);
            move_g();
            gld_u(gs[I1]);
            move_u();
            __builtin_amdgcn_s_setprio(1);

            gemm_0(acc_0[I0], a[I0], gs[I2]);
            gemm_0(acc_0[I1], a[I0], gs[I3]);
        };

        auto pipeline_gemm0_tail = [&]() {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(0);
            block_sync_load_raw(2 * issues_g);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            sld_a(a[I0], a_sld_win0);
            gld_a(a_copy_lds_windows(I1));
            __builtin_amdgcn_sched_barrier(0);
            gld_g(gs[I2]);
            gld_u(gs[I3]);

            // pre-load a and next u/g
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(1);

            // first mmac group
            gemm_0(acc_0[I0], a[I0], gs[I0]);
            gemm_0(acc_0[I1], a[I0], gs[I1]);

            // second mmac group
            __builtin_amdgcn_s_setprio(0);
            block_sync_load_raw(2 * issues_g);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            sld_a(a[I0], a_sld_win1);

            // block_sync_load_raw(issues_a + 2 * issues_g);
            __builtin_amdgcn_s_setprio(1);
            gemm_0(acc_0[I0], a[I0], gs[I2]);
            gemm_0(acc_0[I1], a[I0], gs[I3]);
            __builtin_amdgcn_sched_barrier(0);
        };

        auto y = Policy::template MakeYBlockTile<Problem>();

        auto pipeline_bridge = [&]() {
            using WarpGemm  = remove_cvref_t<decltype(warp_gemm_0)>;
            using CDataType = typename WarpGemm::CDataType;
            if constexpr(IsGateOnly)
            {
                static_assert(false, "not supported yet");
            }
            else
            {
                block_sync_lds(); // ensure last input load have been read from lds
                sweep_tile(
                    acc_0.at(number<0>{}),
                    [&](auto idx0, auto idx1) {
                        const YDataType x0 =
                            type_convert<YDataType>(acc_0.at(number<0>{})(idx0));
                        const YDataType x1 =
                            type_convert<YDataType>(acc_0.at(number<0>{})(idx1));
                        AccDataType y0_float;
                        AccDataType y1_float;
                        typename Problem::GateActivation{}(
                            y0_float, type_convert<AccDataType>(x0));
                        typename Problem::GateActivation{}(
                            y1_float, type_convert<AccDataType>(x1));
                        const YDataType y0 = type_convert<YDataType>(y0_float);
                        const YDataType y1 = type_convert<YDataType>(y1_float);
                        acc_0.at(number<0>{})(idx0) = type_convert<AccDataType>(y0);
                        acc_0.at(number<0>{})(idx1) = type_convert<AccDataType>(y1);
                    },
                    sequence<1, 2>{}); // m=1, n=4 continuos element in thread_buffer(VGPR)
                // Match the unfused FP16 contract: GEMM0 output, activation output, and the
                // gate/up product each cross an FP16 rounding boundary before GEMM1.
                auto reduce_acc_0 = tile_elementwise_in(
                    [&](const auto& a_, const auto& b_) {
                        const YDataType activated = type_convert<YDataType>(a_);
                        const YDataType up        = type_convert<YDataType>(b_);
                        const YDataType product   = type_convert<YDataType>(activated * up);
                        return type_convert<AccDataType>(product);
                    },
                    acc_0.at(number<0>{}),
                    acc_0.at(number<1>{}));

                constexpr auto bridge_tile_enc =
                    Policy::template MakeBridgeTileDistribution<Problem>();

                auto bridge_tile_tensor =
                    make_static_distributed_tensor<CDataType>(bridge_tile_enc);
                static_assert(bridge_tile_tensor.get_thread_buffer_size() ==
                                  reduce_acc_0.get_thread_buffer_size(),
                              "bridge tensor is not same as reduce tensor!");

                constexpr index_t thread_buf_size = bridge_tile_tensor.get_thread_buffer_size();
                // Note: here we assume that every mmac output 4 element, and NInterleave mmac will
                // merge together. So here 1 group refers to 4 x NInterleave.
                static_for<0, thread_buf_size, 1>{}([&](auto i) {
                    index_t reduce_index =
                        (i / (Problem::Gemm0NInterleave * 4)) * (Problem::Gemm0NInterleave * 4) +
                        (i % (Problem::Gemm0NInterleave * 4)) / Problem::Gemm0NInterleave +
                        i % Problem::Gemm0NInterleave * 4;
                    bridge_tile_tensor.get_thread_buffer()[i] =
                        reduce_acc_0.get_thread_buffer()[reduce_index];
                });

                // TODO: support i8 quant here
                // auto y_pre = cast_tile<YDataType>(bridge_tile_tensor);
                store_tile(bridge_sst_win, cast_tile<YDataType,false>(bridge_tile_tensor));
                clear_tile(acc_1s(I0));
                block_sync_lds();
                // wave_barrier();
                load_tile(y, bridge_sld_win);
                clear_tile(acc_1s(I1));
                block_sync_lds();
                __builtin_amdgcn_sched_barrier(0);
            }
        };

        // note, gemm-1 start from idx-1 to N-2 (0, 1, 2....N-1)
        auto pipeline_gemm1 = [&]() {
            d_vgpr_to_lds(gd[I0]);
            gld_d(gd[I1]);
            move_d();
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);
            auto d_tmp = d_lds_to_vgpr();
#ifdef FMOE_UK_DEBUG_D
            if(n_col == 0 && get_lane_id() < 16)
                printf("DPRB w=%d bn=%d buf: %f %f %f %f %f %f %f %f\n",
                       (int)get_warp_id(),
                       (int)get_lane_id(),
                       (float)d_tmp.get_thread_buffer()[number<0>{}],
                       (float)d_tmp.get_thread_buffer()[number<1>{}],
                       (float)d_tmp.get_thread_buffer()[number<2>{}],
                       (float)d_tmp.get_thread_buffer()[number<3>{}],
                       (float)d_tmp.get_thread_buffer()[number<4>{}],
                       (float)d_tmp.get_thread_buffer()[number<5>{}],
                       (float)d_tmp.get_thread_buffer()[number<6>{}],
                       (float)d_tmp.get_thread_buffer()[number<7>{}]);
#endif
            gemm_1(acc_1s[I0], y, d_tmp);
            atomic_add_o(acc_1s[I0]);
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);
            d_vgpr_to_lds(gd[I1]);

            // Note: if support pk_atomic_add, here will use 2interleave and f16 atomic
            // move_o();
            gld_d(gd[I0]);
            move_d();
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);

            d_tmp = d_lds_to_vgpr();
            gemm_1(acc_1s[I1], y, d_tmp);
            atomic_add_o(acc_1s[I1]);
            clear_tile(acc_1s(I0));
            clear_tile(acc_1s(I1));
            block_sync_lds();
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
        };

        auto pipeline_gemm1_head = [&]() {
            gld_d_init();
            gld_d(gd[I0]);
            move_d();
        };
        auto pipeline_gemm1_tail = [&]() {
#ifdef FMOE_UK_DEBUG_G
            if(get_lane_id() % 32 == 0)
            {
                auto ginx = d_copy_dram_window.get_tile_distribution().calculate_index();
                printf("GINX w=%d lane=%d x=(%d,%d) buf8to15: %f %f %f %f %f %f %f %f\n",
                       (int)get_warp_id(),
                       (int)get_lane_id(),
                       (int)ginx[number<0>{}],
                       (int)ginx[number<1>{}],
                       (float)gd[I0].get_thread_buffer()[number<8>{}],
                       (float)gd[I0].get_thread_buffer()[number<9>{}],
                       (float)gd[I0].get_thread_buffer()[number<10>{}],
                       (float)gd[I0].get_thread_buffer()[number<11>{}],
                       (float)gd[I0].get_thread_buffer()[number<12>{}],
                       (float)gd[I0].get_thread_buffer()[number<13>{}],
                       (float)gd[I0].get_thread_buffer()[number<14>{}],
                       (float)gd[I0].get_thread_buffer()[number<15>{}]);
            }
#endif
            d_vgpr_to_lds(gd[I0]);
#ifdef FMOE_UK_DEBUG_G
            if(get_lane_id() % 32 == 0)
                printf("GPRB w=%d lane=%d buf: %f %f %f %f %f %f %f %f\n",
                       (int)get_warp_id(),
                       (int)get_lane_id(),
                       (float)gd[I0].get_thread_buffer()[number<0>{}],
                       (float)gd[I0].get_thread_buffer()[number<1>{}],
                       (float)gd[I0].get_thread_buffer()[number<2>{}],
                       (float)gd[I0].get_thread_buffer()[number<3>{}],
                       (float)gd[I0].get_thread_buffer()[number<4>{}],
                       (float)gd[I0].get_thread_buffer()[number<5>{}],
                       (float)gd[I0].get_thread_buffer()[number<6>{}],
                       (float)gd[I0].get_thread_buffer()[number<7>{}]);
#endif
            gld_d(gd[I1]);
            move_d();
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);
            auto d_tmp = d_lds_to_vgpr();
#ifdef FMOE_UK_DEBUG_D
            if(n_col == 0 && get_lane_id() < 16)
                printf("DPRB w=%d bn=%d buf: %f %f %f %f %f %f %f %f\n",
                       (int)get_warp_id(),
                       (int)get_lane_id(),
                       (float)d_tmp.get_thread_buffer()[number<0>{}],
                       (float)d_tmp.get_thread_buffer()[number<1>{}],
                       (float)d_tmp.get_thread_buffer()[number<2>{}],
                       (float)d_tmp.get_thread_buffer()[number<3>{}],
                       (float)d_tmp.get_thread_buffer()[number<4>{}],
                       (float)d_tmp.get_thread_buffer()[number<5>{}],
                       (float)d_tmp.get_thread_buffer()[number<6>{}],
                       (float)d_tmp.get_thread_buffer()[number<7>{}]);
#endif
            gemm_1(acc_1s[I0], y, d_tmp);
#ifdef FMOE_UK_DEBUG_C
            if(n_col == 0 && (get_lane_id() % 16) < 2)
                printf("CPROBE w=%d lane=%d c: %f %f %f %f %f %f %f %f\n",
                       (int)get_warp_id(),
                       (int)get_lane_id(),
                       (float)acc_1s[I0].get_thread_buffer()[number<0>{}],
                       (float)acc_1s[I0].get_thread_buffer()[number<1>{}],
                       (float)acc_1s[I0].get_thread_buffer()[number<2>{}],
                       (float)acc_1s[I0].get_thread_buffer()[number<3>{}],
                       (float)acc_1s[I0].get_thread_buffer()[number<4>{}],
                       (float)acc_1s[I0].get_thread_buffer()[number<5>{}],
                       (float)acc_1s[I0].get_thread_buffer()[number<6>{}],
                       (float)acc_1s[I0].get_thread_buffer()[number<7>{}]);
#endif
            atomic_add_o(acc_1s[I0]);
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);
            d_vgpr_to_lds(gd[I1]);
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);

            d_tmp = d_lds_to_vgpr();
            gemm_1(acc_1s[I1], y, d_tmp);
            atomic_add_o(acc_1s[I1]);
            __builtin_amdgcn_sched_barrier(0);
        };

        // start of pipeline
        // clang-format off
        gld_a_init();
        gld_a(a_copy_lds_windows(I0));
        move_a();
        //quant
        gld_g_init();
        gld_u_init();
        gld_g(gs[I0]);
        move_g();
        gld_u(gs[I1]);
        move_u();
        //b scale load
        clear_tile(acc_0.at(I0));
        clear_tile(acc_0.at(I1));


        // make sure a,g loaded
        // lds_load_fence();
        // block_sync_load_raw(2 * issues_g);

        // we manually unroll double buffer inside hot loop
        const index_t iters_0 = (num_blocks_k0 - 2) / 2;
        index_t i_0 = 0; // (void)i_0; (void)iters_0; (void)pipeline_gemm0;
        while(i_0++ < iters_0)
        {
            pipeline_gemm0();
        }
        pipeline_gemm0_tail();

        pipeline_bridge();

        const index_t iters_1 = (num_blocks_n1 - 2) / 2;
        index_t i_1 = 0; // (void) i_1; (void)iters_1; (void)pipeline_gemm1;
        pipeline_gemm1_head();
        while(i_1++ < iters_1)
        {
            pipeline_gemm1();
        }
        pipeline_gemm1_tail();
        // clang-format on
    }
};

} // namespace ck_tile
