// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_rotary_embedding.hpp"

namespace ck_tile {

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kStoreLSE_,
          bool kHasDropout_,
          bool kDoFp8StaticQuant_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK       = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ      = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr auto BiasEnum          = BiasEnum_;
    static constexpr bool kHasBiasGrad      = kHasBiasGrad_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kHasDropout       = kHasDropout_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;
    static constexpr index_t kBlockPerCu    = kBlockPerCu_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kStoreLSE_,
          bool kDoFp8StaticQuant_,
          bool kIsPagedKV_,
          bool kHasUnevenSplits_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaFwdSplitKVTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK       = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ      = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr auto BiasEnum          = BiasEnum_;
    static constexpr bool kHasBiasGrad      = kHasBiasGrad_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;
    static constexpr bool kIsPagedKV        = kIsPagedKV_;
    // determine if some split (length) is not divisible by tile size
    static constexpr bool kHasUnevenSplits = kHasUnevenSplits_;
    static constexpr index_t kBlockPerCu   = kBlockPerCu_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          bool kStoreLSE_,
          bool kDoFp8StaticQuant_,
          index_t kLogMaxSplits_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaFwdSplitKVCombineTraits
{
    static constexpr bool kPadSeqLenQ       = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimV      = kPadHeadDimV_;
    static constexpr bool kStoreLSE         = kStoreLSE_;
    static constexpr bool kDoFp8StaticQuant = kDoFp8StaticQuant_;

    static constexpr index_t kMaxSplits = (1 << kLogMaxSplits_);
    static_assert(kMaxSplits <= get_warp_size() || kMaxSplits % get_warp_size() == 0);
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaFwdAppendKVTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK    = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQ   = kPadHeadDimQ_;
    static constexpr bool kPadHeadDimV   = kPadHeadDimV_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

// Traits consumed by the FMHA BWD dq_dk_dv codegen. The codegen passes
// <kPadHeadDimQ, kPadHeadDimV, BiasEnum, kHasBiasGrad, kBlockPerCu> where the head-dim
// pads arrive as the alignment ints (0 / 8 / 1) used elsewhere, so they are normalized
// to bool here. BWD always tiles seqlen with boundary handling, so seqlen padding is
// always enabled (correct for any seqlen; only forgoes an exact-multiple optimization).
template <index_t kPadHeadDimQ_,
          index_t kPadHeadDimV_,
          BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileFmhaBwdTraits
{
    static constexpr bool kPadSeqLenQ       = true;
    static constexpr bool kPadSeqLenK       = true;
    static constexpr bool kPadHeadDimQ      = (kPadHeadDimQ_ > 0);
    static constexpr bool kPadHeadDimV      = (kPadHeadDimV_ > 0);
    static constexpr auto BiasEnum          = BiasEnum_;
    static constexpr bool kHasBiasGrad      = kHasBiasGrad_;
    static constexpr bool kStoreLSE         = true;
    static constexpr bool kHasDropout       = false;
    static constexpr bool kDoFp8StaticQuant = false;
    static constexpr index_t kBlockPerCu    = kBlockPerCu_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          index_t kBlockPerCu_ = 2 /* hint to occupancy */>
struct TileFmhaBwdOGradDotOTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimV   = kPadHeadDimV_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          index_t kBlockPerCu_ = 2 /* hint to occupancy */>
struct TileFmhaBwdConvertQGradTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadHeadDimQ   = kPadHeadDimQ_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

} // namespace ck_tile
