// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/gdn/gdn_numeric.hpp"
//
// Register MMAC helpers used by the GDN KKT solve pipeline.
//
// KKᵀ GEMM:  [M=64, K=128] × [K=128, N=64] → [M=64, N=64]
//   A = k (rows), B = k^T (cols transposed)
//   All 4 warps collaborate: MWarp=4, NWarp=1, using
//   WarpGemmMmacDispatcher<bf16,bf16,f32, 16,64,32, false,1,4,1,1>

#include "ck_tile/ops/gdn/pipeline/gdn_ck_kkt_solve_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"


namespace ck_tile {

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
        const auto off=(row_base+row)*stride_k_t+k_base_col+k_group*4;
        // Head dim 128 and four consecutive 16-bit elements guarantee 8-byte
        // alignment. A masked row becomes one zero vector, not four branches.
        uint64_t bits=0;
        if(!MaskRows || tc+row_base+row<T)
            bits=*reinterpret_cast<const uint64_t*>(k_base+off);
        buf.template set_as<uint64_t>(number<0>{},bits);
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
        return load_a<MaskRows>(k_base,stride_k_t,col_base,k_base_col,tc,T);
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

} // namespace ck_tile
