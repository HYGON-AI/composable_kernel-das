// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dot_do_o.hpp"

#include <string>

namespace ck_tile {

// Jenga sparse-attention backward preprocess:
//   Delta[bh, m] = sum_d Out[bh, m, d] * dOut[bh, m, d]
//
// This is the CK-tile equivalent of _sla_bwd_preprocess_kernel in
// jenga_backward_sla.py.  Unlike the generic FMHA preprocess kernel, the input
// layout is already flattened to [B*H, L, D] and uses seqlens[B] to skip padded
// query blocks.
template <typename FmhaBwdOGradDotO_>
struct JengaBwdPreprocessKernel
{
    using FmhaBwdOGradDotO = ck_tile::remove_cvref_t<FmhaBwdOGradDotO_>;

    static constexpr ck_tile::index_t kBlockSize  = FmhaBwdOGradDotO::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaBwdOGradDotO::kBlockPerCu;
    static constexpr ck_tile::index_t kM0         = kBlockSize;
    static constexpr ck_tile::index_t kVHeaddim   = FmhaBwdOGradDotO::kVHeaddim;

    using DDataType     = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::DDataType>;
    using ODataType     = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::ODataType>;
    using OGradDataType = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::OGradDataType>;

    static constexpr bool kPadSeqLenQ  = FmhaBwdOGradDotO::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV = FmhaBwdOGradDotO::kPadHeadDimV;

    template <typename T>
    struct t2s;
    template <>
    struct t2s<ck_tile::fp16_t>
    {
        static constexpr const char* name = "fp16";
    };
    template <>
    struct t2s<ck_tile::bf16_t>
    {
        static constexpr const char* name = "bf16";
    };

    CK_TILE_HOST static std::string GetName()
    {
        std::string pn;
        if constexpr(kPadSeqLenQ)
            pn += "s";
        if constexpr(kPadHeadDimV)
            pn += "dv";

        return std::string("jenga_bwd_preprocess_d") + std::to_string(kVHeaddim) + "_" +
               t2s<ODataType>::name + "_o" + std::to_string(kBlockPerCu) +
               (pn.empty() ? std::string{} : "_" + pn);
    }

    struct Kargs
    {
        const void* o_ptr;
        const void* do_ptr;
        void* delta_ptr;
        const int32_t* seqlens_ptr;

        ck_tile::index_t num_heads;
        ck_tile::index_t max_seqlen_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t stride_oz;
        ck_tile::index_t stride_om;
        ck_tile::index_t stride_ok;

        ck_tile::index_t stride_doz;
        ck_tile::index_t stride_dom;
        ck_tile::index_t stride_dok;

        ck_tile::index_t stride_dz;
        ck_tile::index_t stride_dm;

        float p_undrop;
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* o_ptr,
                                                  const void* do_ptr,
                                                  void* delta_ptr,
                                                  const void* seqlens_ptr,
                                                  ck_tile::index_t num_heads,
                                                  ck_tile::index_t max_seqlen_q,
                                                  ck_tile::index_t hdim_v,
                                                  ck_tile::index_t stride_oz,
                                                  ck_tile::index_t stride_om,
                                                  ck_tile::index_t stride_ok,
                                                  ck_tile::index_t stride_doz,
                                                  ck_tile::index_t stride_dom,
                                                  ck_tile::index_t stride_dok,
                                                  ck_tile::index_t stride_dz,
                                                  ck_tile::index_t stride_dm,
                                                  float p_undrop = 1.0f)
    {
        return Kargs{o_ptr,
                     do_ptr,
                     delta_ptr,
                     reinterpret_cast<const int32_t*>(seqlens_ptr),
                     num_heads,
                     max_seqlen_q,
                     hdim_v,
                     stride_oz,
                     stride_om,
                     stride_ok,
                     stride_doz,
                     stride_dom,
                     stride_dok,
                     stride_dz,
                     stride_dm,
                     p_undrop};
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size,
             ck_tile::index_t num_heads,
             ck_tile::index_t max_seqlen_q)
    {
        return dim3(ck_tile::integer_divide_ceil(max_seqlen_q, kM0), batch_size * num_heads, 1);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t i_tile_m = blockIdx.x;
        const index_t off_hz   = blockIdx.y;
        const index_t i_batch  = off_hz / kargs.num_heads;

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kM0);
        const index_t seqlen_q =
            kargs.seqlens_ptr == nullptr ? kargs.max_seqlen_q : kargs.seqlens_ptr[i_batch];

        if(i_m0 >= seqlen_q)
        {
            return;
        }

        const auto o_ptr = reinterpret_cast<const ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(off_hz) * kargs.stride_oz;
        const auto do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                            static_cast<long_index_t>(off_hz) * kargs.stride_doz;
        auto delta_ptr = reinterpret_cast<DDataType*>(kargs.delta_ptr) +
                         static_cast<long_index_t>(off_hz) * kargs.stride_dz;

        const auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_om, kargs.stride_ok),
                number<FmhaBwdOGradDotO::kAlignmentO>{},
                number<1>{});
            return pad_tensor_view(o_dram_naive,
                                   make_tuple(number<kM0>{}, number<kVHeaddim>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        const auto do_dram = [&]() {
            const auto do_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                do_ptr,
                make_tuple(seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_dom, kargs.stride_dok),
                number<FmhaBwdOGradDotO::kAlignmentOGrad>{},
                number<1>{});
            return pad_tensor_view(do_dram_naive,
                                   make_tuple(number<kM0>{}, number<kVHeaddim>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto delta_dram = [&]() {
            const auto delta_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                delta_ptr,
                make_tuple(seqlen_q),
                make_tuple(kargs.stride_dm),
                number<1>{},
                number<1>{});
            return pad_tensor_view(
                delta_dram_naive, make_tuple(number<kM0>{}), sequence<kPadSeqLenQ>{});
        }();

        auto o_window =
            make_tile_window(o_dram, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {i_m0, 0});
        auto do_window =
            make_tile_window(do_dram, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {i_m0, 0});
        auto delta_window = make_tile_window(delta_dram, make_tuple(number<kM0>{}), {i_m0});

        FmhaBwdOGradDotO{}(o_window, do_window, delta_window, kargs.p_undrop);
    }
};

} // namespace ck_tile
