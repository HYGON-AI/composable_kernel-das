// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/fill.hpp"
#include "ck_tile/core/numeric/math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ck_tile {

inline constexpr int kUnifiedAttention2dQueryBlockSize = 128;
// Q/K active metadata and packed KV ranges use 64-token logical blocks.
inline constexpr int kUnifiedAttentionMetadataBlockSize = 64;
// MM-prefix stores four [begin, end] pairs for every sequence.
inline constexpr int kUnifiedAttentionMmRangePairs       = 4;
inline constexpr int kUnifiedAttentionMmRangeStride      =
    2 * kUnifiedAttentionMmRangePairs;
inline constexpr int kUnifiedAttentionPackVectorSize     = 8;
inline constexpr int kUnifiedAttentionMaxDecodeSegments  = 64;
inline constexpr int kUnifiedAttentionMinBlocksPerSegment = 4;
inline constexpr float kUnifiedAttentionAlibiExponentScale = 8.0f;

inline constexpr int unified_attention_ceil_div(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

inline constexpr int unified_attention_round_up(int value, int multiple)
{
    return unified_attention_ceil_div(value, multiple) * multiple;
}

inline float kQkScaleLog2e(int d)
{
    return ck_tile::log2e_v<float> / std::sqrt(static_cast<float>(d));
}

inline std::vector<int> parse_query_lens_helper(const std::string& text)
{
    std::vector<int> lengths;
    if(text.empty()) return lengths;
    std::stringstream stream(text);
    std::string token;
    while(std::getline(stream, token, ','))
    {
        if(!token.empty())
            lengths.push_back(std::stoi(token));
    }
    return lengths;
}

struct UnifiedAttentionProblem
{
    int batch = 1;
    int h = 16;
    int hkv = 2;
    int d = 192;
    int total_nq = 0;
    int total_nq_pad = 0;
    int max_nq_pad = 0;
    int total_nqb = 0;
    int max_nqb = 0;
    int max_nkv = 0;
    int nkv_pad = 0;
    int nb = 0;
    int page_size = kUnifiedAttentionMetadataBlockSize;
    int pages_needed_per_seq = 0;
    int block_table_width = 0;
    int required_pages = 0;
    int kv_cache_blocks = 0;
    int metadata_rows = 0;
    int kv_stages = 4;
    int kv_pack_start = 0;
    int packed_kv_rows = 0;
    int prefix_begin = 0;
    int prefix_end = -1;
    int compact_segments = 0;
    int sliding_window = 0;
    bool use_sinks = false;
    bool use_alibi = false;
    bool alibi_sqrt = false;
    bool use_qq_bias = false;
    bool qq_bias_fp32 = false;
    bool use_mm_prefix = false;
    float scale = 0.0f;
    float softcap = 0.0f;

    std::vector<int32_t> query_lens;
    std::vector<int32_t> seqlens;
    std::vector<int32_t> cu_q;

    static UnifiedAttentionProblem create(
        const ck_tile::ArgParser& parser,
        const std::vector<int>& explicit_query_lens = {},
        bool is_3d_decode = false)
    {
        UnifiedAttentionProblem p;
        p.h = parser.get_int("h");
        p.hkv = parser.get_int("hkv");
        p.d = parser.get_int("d");
        p.page_size = parser.get_int("block_size");
        const std::string mask = parser.get_str("mask");
        const bool use_sliding = mask == "sliding" || mask == "sliding_prefix";
        p.sliding_window = parser.get_int("sliding_window");
        p.use_sinks = parser.get_bool("sinks");
        p.use_alibi = parser.get_bool("alibi");
        p.alibi_sqrt = parser.get_bool("alibi_sqrt");
        p.use_qq_bias = parser.get_bool("qq_bias");
        p.qq_bias_fp32 = parser.get_str("qq_bias_dtype") == "fp32";
        p.softcap = parser.get_float("softcap");
        p.use_mm_prefix = parser.get_bool("mm_prefix") ||
                          mask == "prefix" || mask == "sliding_prefix";
        p.prefix_begin = parser.get_int("prefix_begin");
        p.prefix_end = parser.get_int("prefix_end");
        p.scale = 1.0f / std::sqrt(static_cast<float>(p.d));

        if(is_3d_decode)
        {
            p.batch = parser.get_int("b");
            const int nkv = parser.get_int("nkv");
            p.query_lens.assign(p.batch, 1);
            p.seqlens.assign(p.batch, nkv);
            const int segments = parser.get_int("segments");
            int effective_segments = segments;
            if(p.sliding_window == 0 && p.d > 32 &&
               segments < kUnifiedAttentionMaxDecodeSegments &&
               nkv >= segments * kUnifiedAttentionMetadataBlockSize *
                          kUnifiedAttentionMinBlocksPerSegment)
                effective_segments =
                    std::min(kUnifiedAttentionMaxDecodeSegments,
                             segments * kUnifiedAttentionMinBlocksPerSegment);
            p.compact_segments = effective_segments;
            if(p.sliding_window > 0)
                p.compact_segments =
                    std::min(effective_segments,
                             unified_attention_ceil_div(
                                 p.sliding_window - 1,
                                 kUnifiedAttentionMetadataBlockSize) +
                                 1);
        }
        else
        {
            p.query_lens = explicit_query_lens;
            if(p.query_lens.empty())
            {
                p.query_lens = parse_query_lens_helper(parser.get_str("query_lens"));
                if(p.query_lens.empty())
                    p.query_lens = {parser.get_int("nq")};
            }
            p.batch = static_cast<int>(p.query_lens.size());
            const auto parsed_kv_lens = parse_query_lens_helper(parser.get_str("kv_lens"));
            const int nkv = parser.get_int("nkv");
            p.seqlens.assign(p.batch, nkv);
            for(int i = 0; i < p.batch; ++i)
            {
                if(!parsed_kv_lens.empty())
                    p.seqlens[i] = parsed_kv_lens[i];
            }
        }

        p.cu_q.assign(p.batch + 1, 0);
        p.total_nq = 0;
        p.total_nq_pad = 0;
        p.max_nq_pad = 0;
        p.total_nqb = 0;
        p.max_nkv = 0;

        for(int i = 0; i < p.batch; ++i)
        {
            p.max_nkv = std::max(p.max_nkv, static_cast<int>(p.seqlens[i]));
            p.total_nq += p.query_lens[i];
            p.cu_q[i + 1] = p.total_nq;
            const int padded = unified_attention_round_up(
                p.query_lens[i], kUnifiedAttentionMetadataBlockSize);
            p.total_nq_pad += padded;
            p.total_nqb += padded / kUnifiedAttentionMetadataBlockSize;
            p.max_nq_pad = std::max(p.max_nq_pad, padded);
        }

        p.max_nqb = p.max_nq_pad / kUnifiedAttentionMetadataBlockSize;
        if(!is_3d_decode && p.d == 192)
            p.kv_stages = p.batch == 1
                              ? (p.sliding_window <= 0 &&
                                         p.total_nq <=
                                             kUnifiedAttentionMetadataBlockSize
                                     ? 10
                                     : 5)
                              : 4;
        p.nkv_pad = unified_attention_round_up(
            p.max_nkv, kUnifiedAttentionMetadataBlockSize);
        p.nb = p.nkv_pad / kUnifiedAttentionMetadataBlockSize;
        p.pages_needed_per_seq = (p.max_nkv + p.page_size - 1) / p.page_size;
        const int requested_table_width = parser.get_int("block_table_width");
        p.block_table_width = requested_table_width > 0 ? requested_table_width
                                                        : p.pages_needed_per_seq;
        p.required_pages = 0;
        for(const int length : p.seqlens)
            p.required_pages += (length + p.page_size - 1) / p.page_size;
        const int requested_cache_blocks = parser.get_int("kv_cache_blocks");
        p.kv_cache_blocks = requested_cache_blocks > 0 ? requested_cache_blocks
                                                       : p.required_pages;
        p.metadata_rows = p.batch * p.h * p.max_nqb;
        int min_active_block = p.nb;
        p.packed_kv_rows = 0;
        for(int b = 0; b < p.batch; ++b)
        {
            const int nqb = unified_attention_ceil_div(
                p.query_lens[b], kUnifiedAttentionMetadataBlockSize);
            const int context = p.seqlens[b] - p.query_lens[b];
            for(int qb = 0; qb < nqb; ++qb)
            {
                const int first_q = qb * kUnifiedAttentionMetadataBlockSize;
                const int first_block =
                    p.sliding_window > 0
                        ? std::max(0,
                                   context + first_q - p.sliding_window + 1) /
                              kUnifiedAttentionMetadataBlockSize
                        : 0;
                min_active_block = std::min(min_active_block, first_block);
                if(p.use_mm_prefix)
                    min_active_block = 0;
            }
        }
        p.kv_pack_start = min_active_block == p.nb
                              ? 0
                              : min_active_block * kUnifiedAttentionMetadataBlockSize;
        for(int b = 0; b < p.batch; ++b)
        {
            const int raw_start = p.seqlens[b] - p.query_lens[b] -
                                  p.sliding_window + 1;
            const int batch_start =
                p.sliding_window > 0
                    ? (std::max(0, raw_start) /
                       kUnifiedAttentionMetadataBlockSize) *
                          kUnifiedAttentionMetadataBlockSize
                    : p.kv_pack_start;
            p.packed_kv_rows += p.hkv * (p.nkv_pad - batch_start);
        }
        return p;
    }
};

template <typename DataType>
struct UnifiedAttentionData
{
    UnifiedAttentionProblem prob;

    ck_tile::DeviceMem q_buf;
    ck_tile::DeviceMem q_pack_buf;
    ck_tile::DeviceMem k_buf;
    ck_tile::DeviceMem v_buf;
    ck_tile::DeviceMem k_pack_buf;
    ck_tile::DeviceMem v_pack_buf;
    ck_tile::DeviceMem out_buf;
    ck_tile::DeviceMem partial_out_buf;
    ck_tile::DeviceMem lse_buf;
    ck_tile::DeviceMem swa_lse_buf;
    ck_tile::DeviceMem cu_q_buf;
    ck_tile::DeviceMem seqlens_buf;
    ck_tile::DeviceMem block_table_buf;
    ck_tile::DeviceMem counts_buf;
    ck_tile::DeviceMem indices_buf;
    ck_tile::DeviceMem sinks_buf;
    ck_tile::DeviceMem alibi_buf;
    ck_tile::DeviceMem qq_bias_buf;
    ck_tile::DeviceMem mm_ranges_buf;

    ck_tile::DeviceMem seg_out_buf;
    ck_tile::DeviceMem seg_max_buf;
    ck_tile::DeviceMem seg_sum_buf;

    ck_tile::HostTensor<DataType> q_host;
    ck_tile::HostTensor<DataType> k_host;
    ck_tile::HostTensor<DataType> v_host;

    const float* sinks_ptr = nullptr;
    const float* alibi_ptr = nullptr;
    const float* qq_bias_ptr = nullptr;
    const int32_t* mm_ranges_ptr = nullptr;

    explicit UnifiedAttentionData(const UnifiedAttentionProblem& p,
                                  uint32_t seed = 11001,
                                  bool is_3d_decode = false)
        : prob(p),
          q_buf(static_cast<std::size_t>(prob.total_nq_pad) * prob.h * prob.d * sizeof(DataType)),
          q_pack_buf(static_cast<std::size_t>(prob.h) * prob.total_nq_pad * prob.d * sizeof(DataType)),
          k_buf(static_cast<std::size_t>(prob.kv_cache_blocks) * prob.page_size * prob.hkv * prob.d * sizeof(DataType)),
          v_buf(static_cast<std::size_t>(prob.kv_cache_blocks) * prob.page_size * prob.hkv * prob.d * sizeof(DataType)),
          k_pack_buf(static_cast<std::size_t>(prob.batch) * prob.hkv * prob.nkv_pad * prob.d * sizeof(DataType)),
          v_pack_buf(static_cast<std::size_t>(prob.batch) * prob.hkv * prob.nkv_pad * prob.d * sizeof(DataType)),
          out_buf(static_cast<std::size_t>(prob.total_nq_pad) * prob.h * prob.d * sizeof(DataType)),
          partial_out_buf(static_cast<std::size_t>(prob.kv_stages) * prob.h * prob.total_nq_pad * prob.d * sizeof(DataType)),
          lse_buf(static_cast<std::size_t>(
                      prob.batch == 1 && prob.d == 192 && prob.sliding_window <= 0
                          ? 2 * prob.kv_stages + 1
                          : prob.kv_stages + 1) *
                  prob.h * prob.total_nq_pad * sizeof(float)),
          swa_lse_buf(static_cast<std::size_t>(prob.batch) * prob.h * prob.max_nq_pad * sizeof(float)),
          cu_q_buf(static_cast<std::size_t>(prob.batch + 1) * sizeof(int32_t)),
          seqlens_buf(static_cast<std::size_t>(prob.batch) * sizeof(int32_t)),
          block_table_buf(static_cast<std::size_t>(prob.batch) * prob.block_table_width * sizeof(int32_t)),
          counts_buf(static_cast<std::size_t>(prob.metadata_rows) * sizeof(int32_t)),
          indices_buf(static_cast<std::size_t>(prob.metadata_rows) * prob.nb * sizeof(int32_t)),
          sinks_buf(static_cast<std::size_t>(prob.h) * sizeof(float)),
          alibi_buf(static_cast<std::size_t>(prob.h) * sizeof(float)),
          qq_bias_buf((prob.use_qq_bias ? static_cast<std::size_t>(prob.max_nq_pad) * prob.max_nq_pad : 1) *
                      (prob.qq_bias_fp32 ? sizeof(float) : sizeof(DataType))),
          mm_ranges_buf(static_cast<std::size_t>(prob.batch) *
                        kUnifiedAttentionMmRangeStride * sizeof(int32_t)),
          seg_out_buf(is_3d_decode ? static_cast<std::size_t>(prob.batch) * prob.h * prob.compact_segments * prob.d * sizeof(DataType) : 0),
          seg_max_buf(is_3d_decode ? static_cast<std::size_t>(prob.batch) * prob.h * prob.compact_segments * sizeof(float) : 0),
          seg_sum_buf(is_3d_decode ? static_cast<std::size_t>(prob.batch) * prob.h * prob.compact_segments * sizeof(float) : 0),
          q_host({static_cast<std::size_t>(prob.total_nq_pad), static_cast<std::size_t>(prob.h), static_cast<std::size_t>(prob.d)}),
          k_host({static_cast<std::size_t>(prob.kv_cache_blocks), static_cast<std::size_t>(prob.page_size), static_cast<std::size_t>(prob.hkv), static_cast<std::size_t>(prob.d)}),
          v_host({static_cast<std::size_t>(prob.kv_cache_blocks), static_cast<std::size_t>(prob.page_size), static_cast<std::size_t>(prob.hkv), static_cast<std::size_t>(prob.d)})
    {

        ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed}(q_host);
        ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed + 1}(k_host);
        ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, seed + 2}(v_host);

        q_buf.ToDevice(q_host.data());
        k_buf.ToDevice(k_host.data());
        v_buf.ToDevice(v_host.data());

        out_buf.SetZero();
        q_pack_buf.SetZero();
        k_pack_buf.SetZero();
        v_pack_buf.SetZero();
        partial_out_buf.SetZero();
        lse_buf.SetZero();
        swa_lse_buf.SetZero();
        if(is_3d_decode)
        {
            seg_out_buf.SetZero();
            seg_max_buf.SetZero();
            seg_sum_buf.SetZero();
        }

        cu_q_buf.ToDevice(prob.cu_q.data());
        seqlens_buf.ToDevice(prob.seqlens.data());

        std::vector<int32_t> block_table(static_cast<std::size_t>(prob.batch) * prob.block_table_width, 0);
        int block_alloc = 0;
        for(int b = 0; b < prob.batch; ++b)
        {
            const int p_count = (prob.seqlens[b] + prob.page_size - 1) / prob.page_size;
            for(int page = 0; page < p_count; ++page)
            {
                block_table[static_cast<std::size_t>(b) * prob.block_table_width + page] = block_alloc++;
            }
        }
        block_table_buf.ToDevice(block_table.data());

        std::vector<int32_t> counts(prob.metadata_rows, 0);
        std::vector<int32_t> indices(
            static_cast<std::size_t>(prob.metadata_rows) * prob.nb, 0);
        for(int b = 0; b < prob.batch; ++b)
        {
            const int nqb = unified_attention_ceil_div(
                prob.query_lens[b], kUnifiedAttentionMetadataBlockSize);
            const int context = prob.seqlens[b] - prob.query_lens[b];
            for(int h_idx = 0; h_idx < prob.h; ++h_idx)
            {
                for(int qb = 0; qb < nqb; ++qb)
                {
                    const int row = (b * prob.h + h_idx) * prob.max_nqb + qb;
                    const int q_start = qb * kUnifiedAttentionMetadataBlockSize;
                    const int q_end =
                        std::min(prob.query_lens[b],
                                 q_start + kUnifiedAttentionMetadataBlockSize) - 1;
                    const int first_block =
                        prob.sliding_window > 0
                            ? std::max(0,
                                       context + q_start - prob.sliding_window + 1) /
                                  kUnifiedAttentionMetadataBlockSize
                            : 0;
                    const int last_block =
                        std::min(prob.seqlens[b] - 1, context + q_end) /
                        kUnifiedAttentionMetadataBlockSize;
                    std::vector<bool> active(prob.nb, false);
                    for(int kb = first_block; kb <= last_block; ++kb)
                        active[kb] = true;
                    if(prob.use_mm_prefix)
                    {
                        const int begin = std::max(0, prob.prefix_begin);
                        const int end = prob.prefix_end < 0
                                            ? prob.seqlens[b] - 1
                                            : std::min(prob.seqlens[b] - 1,
                                                       prob.prefix_end);
                        const int q_abs_first = context + q_start;
                        const int q_abs_last = context + q_end;
                        if(begin < end && q_abs_first <= end && q_abs_last >= begin)
                            for(int kb = std::max(0, begin) /
                                             kUnifiedAttentionMetadataBlockSize;
                                kb <= std::min(prob.seqlens[b] - 1, end) /
                                          kUnifiedAttentionMetadataBlockSize;
                                ++kb)
                                active[kb] = true;
                    }
                    int count = 0;
                    for(int kb = 0; kb < prob.nb; ++kb)
                        if(active[kb])
                            indices[static_cast<std::size_t>(row) * prob.nb + count++] = kb;
                    counts[row] = count;
                }
            }
        }
        counts_buf.ToDevice(counts.data());
        indices_buf.ToDevice(indices.data());

        if(prob.use_sinks)
        {
            std::vector<float> sinks_data(prob.h, 0.0f);
            sinks_buf.ToDevice(sinks_data.data());
            sinks_ptr = static_cast<const float*>(sinks_buf.GetDeviceBuffer());
        }
        if(prob.use_alibi)
        {
            std::vector<float> alibi_data(prob.h);
            for(int i = 0; i < prob.h; ++i)
                alibi_data[i] = -0.1f * static_cast<float>(i + 1);
            alibi_buf.ToDevice(alibi_data.data());
            alibi_ptr = static_cast<const float*>(alibi_buf.GetDeviceBuffer());
        }
        if(prob.use_qq_bias)
        {
            if(prob.qq_bias_fp32)
            {
                ck_tile::HostTensor<float> qq_bias_host({static_cast<std::size_t>(prob.max_nq_pad), static_cast<std::size_t>(prob.max_nq_pad)});
                ck_tile::FillUniformDistribution<float>{-0.05f, 0.05f, seed + 3, true}(qq_bias_host);
                qq_bias_buf.ToDevice(qq_bias_host.data());
            }
            else
            {
                ck_tile::HostTensor<DataType> qq_bias_host({static_cast<std::size_t>(prob.max_nq_pad), static_cast<std::size_t>(prob.max_nq_pad)});
                ck_tile::FillUniformDistribution<DataType>{-0.05f, 0.05f, seed + 3, true}(qq_bias_host);
                qq_bias_buf.ToDevice(qq_bias_host.data());
            }
            qq_bias_ptr = static_cast<const float*>(qq_bias_buf.GetDeviceBuffer());
        }
        if(prob.use_mm_prefix)
        {
            std::vector<int32_t> mm_ranges_data(
                static_cast<std::size_t>(prob.batch) *
                    kUnifiedAttentionMmRangeStride,
                0);
            for(int b = 0; b < prob.batch; ++b)
            {
                const int prefix_end = prob.prefix_end < 0
                                           ? prob.seqlens[b] - 1
                                           : std::min(prob.seqlens[b] - 1,
                                                      prob.prefix_end);
                mm_ranges_data[b * kUnifiedAttentionMmRangeStride] =
                    std::max(0, prob.prefix_begin);
                mm_ranges_data[b * kUnifiedAttentionMmRangeStride + 1] = prefix_end;
            }
            mm_ranges_buf.ToDevice(mm_ranges_data.data());
            mm_ranges_ptr = static_cast<const int32_t*>(mm_ranges_buf.GetDeviceBuffer());
        }
    }

    reference_unified_attention_arg make_reference_arg(uint16_t* ref_out_ptr) const
    {
        return reference_unified_attention_arg{
            static_cast<const uint16_t*>(q_buf.GetDeviceBuffer()),
            static_cast<const uint16_t*>(k_buf.GetDeviceBuffer()),
            static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(block_table_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(cu_q_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(seqlens_buf.GetDeviceBuffer()),
            sinks_ptr,
            alibi_ptr,
            qq_bias_ptr,
            mm_ranges_ptr,
            ref_out_ptr,
            prob.batch,
            prob.total_nq,
            prob.h,
            prob.hkv,
            prob.d,
            prob.page_size,
            prob.block_table_width,
            prob.max_nq_pad,
            prob.use_mm_prefix ? 4 : 0,
            prob.sliding_window,
            prob.h * prob.d,
            prob.d,
            static_cast<int64_t>(prob.page_size) * prob.hkv * prob.d,
            static_cast<int64_t>(prob.hkv) * prob.d,
            prob.d,
            static_cast<int64_t>(prob.page_size) * prob.hkv * prob.d,
            static_cast<int64_t>(prob.hkv) * prob.d,
            prob.d,
            prob.scale,
            prob.softcap,
            prob.alibi_sqrt,
            prob.qq_bias_fp32};
    }
};

} // namespace ck_tile
