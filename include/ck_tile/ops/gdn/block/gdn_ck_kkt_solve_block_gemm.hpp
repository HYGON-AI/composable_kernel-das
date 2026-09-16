// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
//
// GDN kkt_solve custom block GEMM policies.
// Copies the ARegBSmem pattern from jenga_ck, adapted for kkt_solve shapes.
//
// KKᵀ GEMM:  [M=64, K=128] × [K=128, N=64] → [M=64, N=64]
//   A = k (rows), B = k^T (cols transposed)
//   All 4 warps collaborate: MWarp=4, NWarp=1, using
//   WarpGemmMmacDispatcher<bf16,bf16,f32, 16,64,32, false,1,4,1,1>

#include "ck_tile/ops/gdn/pipeline/gdn_ck_kkt_solve_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"


namespace ck_tile {

template <typename DataType>
struct GdnKktWarpMmac16
{
    using Impl = typename GdnMmacImpl<DataType>::Type;
    using AVec = typename Impl::AVecType;
    using BVec = typename Impl::BVecType;
    using CVec = typename Impl::CVecType;

    template <typename I>
    CK_TILE_DEVICE static float c_at(const CVec& c, I i)
    {
        union {
            CVec v;
            float e[4];
        } u{c};
        return u.e[i];
    }

    CK_TILE_DEVICE static CVec neg_c(CVec c)
    {
        union {
            CVec v;
            float e[4];
        } u{c};
        static_for<0, 4, 1>{}([&](auto i) {
            u.e[i] = -u.e[i];
        });
        return u.v;
    }

    template <typename LdsView>
    CK_TILE_DEVICE static AVec load_a(const LdsView& view, int row_base, int col_base)
    {
        const index_t lane = get_lane_id();
        const index_t row = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            const float v = view.get_buffer_view()[
                (row_base + row) * 64 + col_base + k_group * 4 + e];
            buf(e) = type_convert<DataType>(v);
        });
        return buf.template get_as<AVec>()[number<0>{}];
    }

    template <typename LdsView>
    CK_TILE_DEVICE static AVec load_a_slot(const LdsView& view, int slot)
    {
        const index_t lane = get_lane_id();
        const index_t row = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            const float v = view.get_buffer_view()[slot * 256 + row * 16 + k_group * 4 + e];
            buf(e) = type_convert<DataType>(v);
        });
        return buf.template get_as<AVec>()[number<0>{}];
    }

    template <int Slot, typename LdsView>
    CK_TILE_DEVICE static AVec load_a_slot(const LdsView& view)
    {
        const index_t lane = get_lane_id();
        const index_t row = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            constexpr index_t slot_base = Slot * 256;
            const float v = view.get_buffer_view()[slot_base + row * 16 + k_group * 4 + e];
            buf(e) = type_convert<DataType>(v);
        });
        return buf.template get_as<AVec>()[number<0>{}];
    }

    template <typename LdsView>
    CK_TILE_DEVICE static BVec load_b(const LdsView& view, int row_base, int col_base)
    {
        const index_t lane = get_lane_id();
        const index_t n = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            const float v = view.get_buffer_view()[
                (row_base + k_group * 4 + e) * 64 + col_base + n];
            buf(e) = type_convert<DataType>(v);
        });
        return buf.template get_as<BVec>()[number<0>{}];
    }

    template <typename LdsView>
    CK_TILE_DEVICE static BVec load_b_slot(const LdsView& view, int slot)
    {
        const index_t lane = get_lane_id();
        const index_t n = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            const float v = view.get_buffer_view()[slot * 256 + (k_group * 4 + e) * 16 + n];
            buf(e) = type_convert<DataType>(v);
        });
        return buf.template get_as<BVec>()[number<0>{}];
    }

    template <int Slot, typename LdsView>
    CK_TILE_DEVICE static BVec load_b_slot(const LdsView& view)
    {
        const index_t lane = get_lane_id();
        const index_t n = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            constexpr index_t slot_base = Slot * 256;
            const float v = view.get_buffer_view()[slot_base + (k_group * 4 + e) * 16 + n];
            buf(e) = type_convert<DataType>(v);
        });
        return buf.template get_as<BVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static BVec c_to_b(const CVec& c)
    {
        const index_t lane = get_lane_id();
        const index_t n = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        const index_t src_group = n & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t k = k_group * 4 + e;
            const index_t src_lane = k + 16 * src_group;
            static_for<0, 4, 1>{}([&](auto src_off) {
                const float v = warp_shuffle(c_at(c, src_off), src_lane);
                if(n >> 2 == src_off)
                {
                    buf(e) = type_convert<DataType>(v);
                }
            });
        });
        return buf.template get_as<BVec>()[number<0>{}];
    }

    template <typename LdsView>
    CK_TILE_DEVICE static CVec mm_lds_lds(const LdsView& view,
                                          int ar,
                                          int ac,
                                          int br,
                                          int bc)
    {
        CVec c{0.f, 0.f, 0.f, 0.f};
        auto a = load_a(view, ar, ac);
        auto b = load_b(view, br, bc);
        Impl{}(c, a, b);
        return c;
    }

    template <typename LdsView>
    CK_TILE_DEVICE static CVec mm_slot_slot(const LdsView& view, int a_slot, int b_slot)
    {
        CVec c{0.f, 0.f, 0.f, 0.f};
        auto a = load_a_slot(view, a_slot);
        auto b = load_b_slot(view, b_slot);
        Impl{}(c, a, b);
        return c;
    }

    template <int ASlot, int BSlot, typename LdsView>
    CK_TILE_DEVICE static CVec mm_slot_slot(const LdsView& view)
    {
        CVec c{0.f, 0.f, 0.f, 0.f};
        auto a = load_a_slot<ASlot>(view);
        auto b = load_b_slot<BSlot>(view);
        Impl{}(c, a, b);
        return c;
    }

    template <typename LdsView>
    CK_TILE_DEVICE static CVec mm_lds_breg(const LdsView& view,
                                           int ar,
                                           int ac,
                                           const BVec& b)
    {
        CVec c{0.f, 0.f, 0.f, 0.f};
        auto a = load_a(view, ar, ac);
        Impl{}(c, a, b);
        return c;
    }

    template <typename LdsView>
    CK_TILE_DEVICE static CVec mm_slot_breg(const LdsView& view, int a_slot, const BVec& b)
    {
        CVec c{0.f, 0.f, 0.f, 0.f};
        auto a = load_a_slot(view, a_slot);
        Impl{}(c, a, b);
        return c;
    }

    template <int ASlot, typename LdsView>
    CK_TILE_DEVICE static CVec mm_slot_breg(const LdsView& view, const BVec& b)
    {
        CVec c{0.f, 0.f, 0.f, 0.f};
        auto a = load_a_slot<ASlot>(view);
        Impl{}(c, a, b);
        return c;
    }

    template <typename LdsView>
    CK_TILE_DEVICE static void store_c(LdsView& view, int row_base, int col_base, const CVec& c)
    {
        const index_t lane = get_lane_id();
        const index_t row = lane & 15;
        const index_t n_group = (lane >> 4) & 3;
        static_for<0, 4, 1>{}([&](auto e) {
            view.get_buffer_view()((row_base + row) * 64 + col_base + n_group + e * 4) =
                c_at(c, e);
        });
    }

    template <typename LdsView>
    CK_TILE_DEVICE static void store_c_slot(LdsView& view, int slot, const CVec& c)
    {
        const index_t lane = get_lane_id();
        const index_t row = lane & 15;
        const index_t n_group = (lane >> 4) & 3;
        static_for<0, 4, 1>{}([&](auto e) {
            view.get_buffer_view()(slot * 256 + row * 16 + n_group + e * 4) = c_at(c, e);
        });
    }

    template <int Slot, typename LdsView>
    CK_TILE_DEVICE static void store_c_slot(LdsView& view, const CVec& c)
    {
        const index_t lane = get_lane_id();
        const index_t row = lane & 15;
        const index_t n_group = (lane >> 4) & 3;
        static_for<0, 4, 1>{}([&](auto e) {
            constexpr index_t slot_base = Slot * 256;
            view.get_buffer_view()(slot_base + row * 16 + n_group + e * 4) = c_at(c, e);
        });
    }

};

template <typename DataType>
struct GdnKktWarpKkt16
{
    using Impl = typename GdnMmacTransCImpl<DataType>::Type;
    using AVec = typename Impl::AVecType;
    using BVec = typename Impl::BVecType;
    using CVec = typename Impl::CVecType;

    template <typename I>
    CK_TILE_DEVICE static float c_at(const CVec& c, I i)
    {
        union {
            CVec v;
            float e[4];
        } u{c};
        return u.e[i];
    }

    template <bool MaskRows>
    CK_TILE_DEVICE static AVec load_a(const DataType* __restrict__ k_base,
                                      int stride_k_t,
                                      int row_base,
                                      int k_base_col,
                                      int tc,
                                      int T)
    {
        const index_t lane = get_lane_id();
        const index_t row = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            if constexpr(MaskRows)
                buf(e) = tc + row_base + row < T
                             ? k_base[(row_base + row) * stride_k_t +
                                      k_base_col + k_group * 4 + e]
                             : type_convert<DataType>(0.0f);
            else
                buf(e) = k_base[(row_base + row) * stride_k_t +
                                k_base_col + k_group * 4 + e];
        });
        return buf.template get_as<AVec>()[number<0>{}];
    }

    template <bool MaskRows>
    CK_TILE_DEVICE static BVec load_b(const DataType* __restrict__ k_base,
                                      int stride_k_t,
                                      int col_base,
                                      int k_base_col,
                                      int tc,
                                      int T)
    {
        const index_t lane = get_lane_id();
        const index_t n = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            if constexpr(MaskRows)
                buf(e) = tc + col_base + n < T
                             ? k_base[(col_base + n) * stride_k_t +
                                      k_base_col + k_group * 4 + e]
                             : type_convert<DataType>(0.0f);
            else
                buf(e) = k_base[(col_base + n) * stride_k_t +
                                k_base_col + k_group * 4 + e];
        });
        return buf.template get_as<BVec>()[number<0>{}];
    }

    template <bool MaskRows>
    CK_TILE_DEVICE static CVec compute(const DataType* __restrict__ k_base,
                                       int stride_k_t,
                                       int row_base,
                                       int col_base,
                                       int tc,
                                       int T)
    {
        CVec c{0.f, 0.f, 0.f, 0.f};
        static_for<0, 8, 1>{}([&](auto k_iter) {
            constexpr int k_base_col = k_iter * 16;
            const auto a = load_a<MaskRows>(k_base, stride_k_t, row_base, k_base_col, tc, T);
            const auto b = load_b<MaskRows>(k_base, stride_k_t, col_base, k_base_col, tc, T);
            Impl{}(c, a, b);
        });
        return c;
    }

    template <typename LdsView>
    CK_TILE_DEVICE static void store_scaled(LdsView& view,
                                            int row_base,
                                            int col_base,
                                            const CVec& c,
                                            const float* __restrict__ g_global,
                                            const float* __restrict__ b_global,
                                            int tc,
                                            int ivh,
                                            int T,
                                            int stride_g_t,
                                            bool use_exp2)
    {
        const index_t lane = get_lane_id();
        const index_t col = lane & 15;
        const index_t m_group = (lane >> 4) & 3;
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t local_row = m_group + e * 4;
            const index_t r = row_base + local_row;
            const index_t global_col = col_base + col;
            float v = 0.0f;
            if((tc + r < T) && (tc + global_col < T) &&
               ((row_base != col_base) || (r > global_col)))
            {
                const float gd = g_global[(tc + r) * stride_g_t + ivh] -
                                 g_global[(tc + global_col) * stride_g_t + ivh];
                const float gate = use_exp2 ? __builtin_amdgcn_exp2f(gd) : expf(gd);
                v = c_at(c, e) * gate * b_global[(tc + r) * stride_g_t + ivh];
            }
            view.get_buffer_view()((row_base + local_row) * 64 + col_base + col) = v;
        });
    }

    template <typename LdsView>
    CK_TILE_DEVICE static void store_scaled_slot(LdsView& view,
                                                 int slot,
                                                 int row_base,
                                                 int col_base,
                                                 const CVec& c,
                                                 const float* __restrict__ row_factor,
                                                 const float* __restrict__ col_factor,
                                                 int tc,
                                                 int T)
    {
        const index_t lane = get_lane_id();
        const index_t col = lane & 15;
        const index_t m_group = (lane >> 4) & 3;
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t local_row = m_group + e * 4;
            const index_t r = row_base + local_row;
            const index_t global_col = col_base + col;
            float v = 0.0f;
            if((tc + r < T) && (tc + global_col < T) &&
               ((row_base != col_base) || (r > global_col)))
            {
                v = c_at(c, e) * row_factor[r] * col_factor[global_col];
            }
            view.get_buffer_view()(slot * 256 + local_row * 16 + col) = v;
        });
    }
};

template <typename DataType>
struct GdnKktWarpRegGemm16
{
    using Impl = typename GdnMmacTransCImpl<DataType>::Type;
    using AVec = typename Impl::AVecType;
    using BVec = typename Impl::BVecType;
    using CVec = typename Impl::CVecType;

    template <typename I>
    CK_TILE_DEVICE static float c_at(const CVec& c, I i)
    {
        union {
            CVec v;
            float e[4];
        } u{c};
        return u.e[i];
    }

    template <typename I0, typename I1>
    CK_TILE_DEVICE static uint32_t pack_c_pair(const CVec& c, I0 i0, I1 i1)
    {
        const uint32_t lo = bit_cast<uint16_t>(type_convert<DataType>(c_at(c, i0)));
        const uint32_t hi = bit_cast<uint16_t>(type_convert<DataType>(c_at(c, i1)));
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

    CK_TILE_DEVICE static AVec c_to_a(const CVec& c)
    {
        const index_t lane = get_lane_id();
        const index_t row = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        const uint32_t c01 = pack_c_pair(c, number<0>{}, number<1>{});
        const uint32_t c23 = pack_c_pair(c, number<2>{}, number<3>{});
        const index_t src_e = row >> 2;
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t k = k_group * 4 + e;
            const index_t src_lane = k + 16 * (row & 3);
            buf(e) = shuffle_packed_c(c01, c23, src_lane, src_e);
        });
        return buf.template get_as<AVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static BVec c_to_b(const CVec& c)
    {
        const index_t lane = get_lane_id();
        const index_t n = lane & 15;
        const index_t k_group = (lane >> 4) & 3;
        const uint32_t c01 = pack_c_pair(c, number<0>{}, number<1>{});
        const uint32_t c23 = pack_c_pair(c, number<2>{}, number<3>{});
        thread_buffer<DataType, 4> buf;
        static_for<0, 4, 1>{}([&](auto e) {
            const index_t k = k_group * 4 + e;
            const index_t src_lane = n + 16 * (k & 3);
            buf(e) = shuffle_packed_c(c01, c23, src_lane, k >> 2);
        });
        return buf.template get_as<BVec>()[number<0>{}];
    }

    CK_TILE_DEVICE static CVec multiply(const CVec& a, const CVec& b)
    {
        CVec c{0.f, 0.f, 0.f, 0.f};
        Impl{}(c, c_to_a(a), c_to_b(b));
        return c;
    }

    CK_TILE_DEVICE static CVec negate(CVec c)
    {
        union {
            CVec v;
            float e[4];
        } u{c};
        static_for<0, 4, 1>{}([&](auto e) { u.e[e] = -u.e[e]; });
        return u.v;
    }
};

// ── Policy adapter for block GEMM (jenga_ck LocalARegBSmemPolicy) ─────────

template <typename BlockWarps_, typename WarpGemm_>
struct GdnKktARegBSmemPolicy
{
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpGemm   = remove_cvref_t<WarpGemm_>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemm{},
                          BlockWarps::at(number<0>{}),
                          BlockWarps::at(number<1>{}));
    }
};


// ── ARegBSmem block GEMM (A in registers, B in LDS) ──────────────────────

template <typename Problem_, typename Policy_>
struct GdnKktBlockGemmARegBSmem
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    CK_TILE_DEVICE auto MakeABlockTile() const
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;
        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>,
                                             sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});
        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        return make_static_distributed_tensor<ADataType>(a_block_dstr);
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return make_static_distributed_tensor<CDataType>(c_block_dstr);
    }

    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(
        CBlockTensor& c_block_tensor,
        const ABlockTensorTmp& a_block_tensor_tmp,
        const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
            std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
            std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "type mismatch in GdnKktBlockGemmARegBSmem");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM &&
                      NPerBlock == BlockGemmShape::kN &&
                      KPerBlock == BlockGemmShape::kK, "shape mismatch");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iNWarp = get_warp_id() % NWarp;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>,
                                             sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});
        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        auto a_block_tensor = make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);
        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        // Build B warp tile windows (B is in LDS, viewed through tile windows)
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_block_window_tmp.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp> b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        using AWarpDstr   = typename WG::AWarpDstr;
        using CWarpDstr   = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // ── K → N → M triple loop over warp GEMM ──────────────────────
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    // ── Convert C to row-major output layout ───────────────────────────
    template <typename CBlockTensor>
    CK_TILE_DEVICE auto MakeOuputLayout(const CBlockTensor& c_block_tensor) const
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        using CWarpDstr         = typename WG::CWarpDstr;
        using CWarpOutputDstr   = typename WG::CWarpOutputDstr;
        using CWarpTensor       = typename WG::CWarpTensor;
        using CWarpOutputTensor = typename WG::CWarpOutputTensor;

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        constexpr auto c_block_out_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpOutputDstrEncoding{});
        constexpr auto c_output_block_dstr = make_static_tile_distribution(c_block_out_dstr_encode);
        auto c_block_output_tensor = make_static_distributed_tensor<CDataType>(c_output_block_dstr);

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_output_y_lengths =
            to_sequence(CWarpOutputDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_output_y_index_zeros =
            uniform_sequence_gen_t<CWarpOutputDstr::NDimY, 0>{};

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                CWarpOutputTensor c_warp_output_tensor = WG{}.MakeCOutputLayout(c_warp_tensor);
                c_block_output_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_output_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_output_y_lengths),
                    c_warp_output_tensor.get_thread_buffer());
            });
        });

        return c_block_output_tensor;
    }
};


// ── ARegBReg block GEMM (both A and B in registers) ──────────────────────
// For KKᵀ: k is both A and B.  Load once from global→regs, no LDS needed.

template <typename Problem_, typename Policy_>
struct GdnKktBlockGemmARegBReg
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    CK_TILE_DEVICE static constexpr auto MakeABlockTile()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;
        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});
        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        return make_static_distributed_tensor<ADataType>(a_block_dstr);
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockTile()
    {
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;
        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});
        constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);
        return make_static_distributed_tensor<BDataType>(b_block_dstr);
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return make_static_distributed_tensor<CDataType>(c_block_dstr);
    }

    template <typename CBlockTensor, typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        using AWarpDstr = typename WG::AWarpDstr;
        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                AWarpTensor a_warp_tensor;
                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    template <typename CBlockTensor>
    CK_TILE_DEVICE auto MakeOuputLayout(const CBlockTensor& c_block_tensor) const
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        using CWarpDstr         = typename WG::CWarpDstr;
        using CWarpOutputDstr   = typename WG::CWarpOutputDstr;
        using CWarpTensor       = typename WG::CWarpTensor;
        using CWarpOutputTensor = typename WG::CWarpOutputTensor;

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        constexpr auto c_block_out_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpOutputDstrEncoding{});
        constexpr auto c_output_block_dstr = make_static_tile_distribution(c_block_out_dstr_encode);
        auto c_block_output_tensor = make_static_distributed_tensor<CDataType>(c_output_block_dstr);

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_output_y_lengths =
            to_sequence(CWarpOutputDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_output_y_index_zeros =
            uniform_sequence_gen_t<CWarpOutputDstr::NDimY, 0>{};

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                CWarpOutputTensor c_warp_output_tensor = WG{}.MakeCOutputLayout(c_warp_tensor);
                c_block_output_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_output_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_output_y_lengths),
                    c_warp_output_tensor.get_thread_buffer());
            });
        });
        return c_block_output_tensor;
    }
};

} // namespace ck_tile
