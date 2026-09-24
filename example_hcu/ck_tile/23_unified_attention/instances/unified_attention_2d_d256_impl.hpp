// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once


#include "ck_tile/host.hpp"
#define CK_TILE_UNIFIED_ATTENTION_D256
#include "ck_tile/ops/unified_attention.hpp"
#undef CK_TILE_UNIFIED_ATTENTION_D256
#include "../unified_attention_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

inline int select_d256_kv_stages(int nb, int nqb, int h_q, int n_q, int max_stages)
{
    const int rows = h_q * nqb;
    int stages = 1;
    if(h_q <= 2)
        stages = nb <= 16 ? 4 : (nb <= 64 ? 5 : 4);
    else if(nb > nqb)
    {
        if(nqb <= 8) stages = 5;
        else if(nqb <= 16)
            stages = rows <= 64 ? 5 : (rows <= 128 ? 4 : (rows <= 160 ? 2 : 1));
        else if(nqb <= 32) stages = rows <= 128 ? 5 : 1;
    }
    else if(n_q % kBlockM != 0 && nb >= 10 && nb <= 15)
        stages = 2;
    else
        stages = nb <= 384 ? 1 : (nb <= 512 ? 2 : (nb <= 896 ? 3 : 4));
    return std::max(1, std::min(std::min(max_stages, nb), stages));
}

template <typename DataType>
int run_unified_attention_2d_d256_batch(const ck_tile::ArgParser& parser,
                                        const std::vector<int>& query_lens)
{
    const int d = parser.get_int("d");
    const int batch = static_cast<int>(query_lens.size());
    const int h = parser.get_int("h");
    const int hkv = parser.get_int("hkv");
    const std::string mask = parser.get_str("mask");
    const int sliding_window = parser.get_int("sliding_window");
    const bool use_sliding = mask == "sliding" || mask == "sliding_prefix";
    const bool use_mm_prefix = parser.get_bool("mm_prefix") ||
                               mask == "prefix" || mask == "sliding_prefix";
    const bool direct_causal = !use_mm_prefix && sliding_window <= 0;
    const bool use_sinks = parser.get_bool("sinks");
    const bool use_alibi = parser.get_bool("alibi");
    const bool alibi_sqrt = parser.get_bool("alibi_sqrt");
    const bool use_qq_bias = parser.get_bool("qq_bias");
    const float softcap = parser.get_float("softcap");
    const int requested_page_size = parser.get_int("block_size");
    if(batch < 1 || batch > kMaxBatchSeqs || h <= 0 || hkv <= 0 || h % hkv != 0 ||
       !parser.get_bool("causal") ||
       (mask != "causal" && mask != "sliding" && mask != "prefix" &&
        mask != "sliding_prefix") || (use_sliding && sliding_window <= 0) ||
       (!use_sliding && sliding_window != 0) || softcap < 0 ||
       (alibi_sqrt && !use_alibi))
    {
        std::cerr << "invalid D256 multibatch configuration\n";
        return 2;
    }

    int total_nq = 0;
    int total_nqb = 0;
    int max_nq = 0;
    int max_nkv = 0;
    std::vector<int32_t> cu_q(batch + 1, 0);
    const auto parsed_kv_lens =
        ck_tile::parse_query_lens_helper(parser.get_str("kv_lens"), "kv_lens");
    if(!parsed_kv_lens.empty() &&
       static_cast<int>(parsed_kv_lens.size()) != batch)
    {
        std::cerr << "kv_lens must contain one length per query sequence\n";
        return 2;
    }
    std::vector<int32_t> n_kvs(batch, parser.get_int("nkv"));
    for(int i = 0; i < batch; ++i)
    {
        if(!parsed_kv_lens.empty()) n_kvs[i] = parsed_kv_lens[i];
        if(n_kvs[i] < query_lens[i])
        {
            std::cerr << "every KV length must be >= its query length\n";
            return 2;
        }
        total_nq += query_lens[i];
        total_nqb += ck_tile::unified_attention_ceil_div(
            query_lens[i], ck_tile::kUnifiedAttentionMetadataBlockSize);
        max_nq = std::max(max_nq, query_lens[i]);
        max_nkv = std::max(max_nkv, static_cast<int>(n_kvs[i]));
        cu_q[i + 1] = total_nq;
    }

    if(batch > 1)
        total_nqb = total_nq / kBlockM + batch;
    const int max_nq_pad = ck_tile::unified_attention_round_up(
        max_nq, ck_tile::kUnifiedAttentionMetadataBlockSize);
    const int max_nkv_pad = ck_tile::unified_attention_round_up(
        max_nkv, ck_tile::kUnifiedAttentionMetadataBlockSize);
    const int max_nqb = ck_tile::unified_attention_ceil_div(
        max_nq, ck_tile::kUnifiedAttentionMetadataBlockSize);
    const int max_nb = ck_tile::unified_attention_ceil_div(
        max_nkv, ck_tile::kUnifiedAttentionMetadataBlockSize);
    constexpr int max_kv_stages = 5;
    const int page_size = requested_page_size > 0
                              ? requested_page_size
                              : ck_tile::kUnifiedAttentionMetadataBlockSize;
    const int pages_needed_per_seq = (max_nkv + page_size - 1) / page_size;
    const int requested_table_width = parser.get_int("block_table_width");
    const int pages_per_seq =
        requested_table_width > 0 ? requested_table_width : pages_needed_per_seq;
    if(pages_per_seq < pages_needed_per_seq)
    {
        std::cerr << "block_table_width is smaller than ceil(max(kv_lens)/block_size)\n";
        return 2;
    }
    int required_pages = 0;
    for(const int length : n_kvs)
        required_pages += (length + page_size - 1) / page_size;
    const int requested_cache_blocks = parser.get_int("kv_cache_blocks");
    const int physical_pages =
        requested_cache_blocks > 0
            ? requested_cache_blocks
            : required_pages;
    if(physical_pages < required_pages)
    {
        std::cerr << "kv_cache_blocks is smaller than the pages required by kv_lens\n";
        return 2;
    }
    // Optional sink/bias/softcap features use the packed generic kernel.
    const bool direct_page = direct_causal && max_nkv <= page_size &&
                             !use_sinks && !use_qq_bias && softcap <= 0.0f;
    const int selected_kv_stages =
        batch > 1 ? 1
                  : select_d256_kv_stages(max_nb, total_nqb, h, max_nq, max_kv_stages);
    const int kv_stage_count =
        direct_page && max_nb <= 15 && total_nqb * h >= 128 ? 1 : selected_kv_stages;
    const int max_rows = h * max_nqb;
    const std::size_t elem_bytes = sizeof(uint16_t);
    ck_tile::DeviceMem q_buf(
        static_cast<std::size_t>(total_nq) * h * d * elem_bytes);
    ck_tile::DeviceMem k_buf(
        static_cast<std::size_t>(physical_pages) * page_size * hkv * d * elem_bytes);
    ck_tile::DeviceMem v_buf(
        static_cast<std::size_t>(physical_pages) * page_size * hkv * d * elem_bytes);
    ck_tile::DeviceMem q_pack_buf(
        static_cast<std::size_t>(batch) * h * max_nq_pad * d * elem_bytes);
    ck_tile::DeviceMem k_pack_buf(
        static_cast<std::size_t>(batch) * hkv * max_nkv_pad * d * elem_bytes);
    ck_tile::DeviceMem v_pack_buf(
        static_cast<std::size_t>(batch) * hkv * max_nkv_pad * d * elem_bytes);
    ck_tile::DeviceMem out_buf(static_cast<std::size_t>(total_nq) * h * d * elem_bytes);
    ck_tile::DeviceMem partial_out_buf(
        static_cast<std::size_t>(max_kv_stages) * h * total_nq * d * elem_bytes);
    ck_tile::DeviceMem lse_buf(
        static_cast<std::size_t>(max_kv_stages + 1) * h * total_nq * sizeof(float));
    ck_tile::DeviceMem cu_q_buf(static_cast<std::size_t>(batch + 1) * sizeof(int32_t));
    ck_tile::DeviceMem n_kvs_buf(static_cast<std::size_t>(batch) * sizeof(int32_t));
    ck_tile::DeviceMem block_table_buf(
        static_cast<std::size_t>(batch) * pages_per_seq * sizeof(int32_t));
    ck_tile::DeviceMem counts_buf(
        static_cast<std::size_t>(batch) * max_rows * sizeof(int32_t));
    ck_tile::DeviceMem indices_buf(
        static_cast<std::size_t>(batch) * max_rows * max_nb * sizeof(int32_t));
    ck_tile::DeviceMem alibi_buf(static_cast<std::size_t>(h) * sizeof(float));
    ck_tile::DeviceMem sinks_buf(static_cast<std::size_t>(h) * sizeof(float));
    ck_tile::DeviceMem qq_bias_buf(
        (use_qq_bias ? static_cast<std::size_t>(max_nq) * max_nq : 1) * sizeof(float));
    ck_tile::DeviceMem mm_ranges_buf(
        static_cast<std::size_t>(batch) * ck_tile::kUnifiedAttentionMmRangeStride *
        sizeof(int32_t));
    ck_tile::HostTensor<DataType> q_host({static_cast<std::size_t>(total_nq), static_cast<std::size_t>(h), static_cast<std::size_t>(d)});
    ck_tile::HostTensor<DataType> k_host({static_cast<std::size_t>(physical_pages), static_cast<std::size_t>(page_size), static_cast<std::size_t>(hkv), static_cast<std::size_t>(d)});
    ck_tile::HostTensor<DataType> v_host({static_cast<std::size_t>(physical_pages), static_cast<std::size_t>(page_size), static_cast<std::size_t>(hkv), static_cast<std::size_t>(d)});
    ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, 11939, true}(q_host);
    ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, 11940, true}(k_host);
    ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, 11941, true}(v_host);
    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    q_pack_buf.SetZero(); k_pack_buf.SetZero(); v_pack_buf.SetZero();
    out_buf.SetZero();
    partial_out_buf.SetZero(); lse_buf.SetZero();
    cu_q_buf.ToDevice(cu_q.data());
    n_kvs_buf.ToDevice(n_kvs.data());
    std::vector<int32_t> block_table(
        static_cast<std::size_t>(batch) * pages_per_seq, 0);
    int next_physical_page = 0;
    for(int i = 0; i < batch; ++i)
        for(int page = 0; page < (n_kvs[i] + page_size - 1) / page_size; ++page)
            block_table[static_cast<std::size_t>(i) * pages_per_seq + page] =
                next_physical_page++;
    block_table_buf.ToDevice(block_table.data());
    counts_buf.SetZero();
    indices_buf.SetZero();

    const float* alibi = nullptr;
    const float* sinks = nullptr;
    const float* qq_bias = nullptr;
    const int32_t* mm_ranges = nullptr;
    if(use_alibi)
    {
        std::vector<float> values(h);
        for(int i = 0; i < h; ++i)
            values[i] = std::exp2(-ck_tile::kUnifiedAttentionAlibiExponentScale *
                                  static_cast<float>(i + 1) / h);
        alibi_buf.ToDevice(values.data());
        alibi = static_cast<const float*>(alibi_buf.GetDeviceBuffer());
    }
    if(use_sinks)
    {
        std::vector<float> values(h, parser.get_float("sink_value"));
        sinks_buf.ToDevice(values.data());
        sinks = static_cast<const float*>(sinks_buf.GetDeviceBuffer());
    }
    if(use_qq_bias)
    {
        std::vector<float> values(static_cast<std::size_t>(max_nq) * max_nq,
                                  parser.get_float("qq_bias_value"));
        qq_bias_buf.ToDevice(values.data());
        qq_bias = static_cast<const float*>(qq_bias_buf.GetDeviceBuffer());
    }
    if(use_mm_prefix)
    {
        std::vector<int32_t> ranges(
            static_cast<std::size_t>(batch) *
                ck_tile::kUnifiedAttentionMmRangeStride,
            -1);
        for(int i = 0; i < batch; ++i)
        {
            ranges[static_cast<std::size_t>(i) *
                   ck_tile::kUnifiedAttentionMmRangeStride] =
                parser.get_int("prefix_begin");
            ranges[static_cast<std::size_t>(i) *
                       ck_tile::kUnifiedAttentionMmRangeStride +
                   1] =
                parser.get_int("prefix_end") < 0 ? query_lens[i] - 1
                                                  : parser.get_int("prefix_end");
        }
        mm_ranges_buf.ToDevice(ranges.data());
        mm_ranges = static_cast<const int32_t*>(mm_ranges_buf.GetDeviceBuffer());
    }

    UnifiedAttentionD256PrepareBatchArgument prep_arg{};
    prep_arg.q_offsets = static_cast<const int32_t*>(cu_q_buf.GetDeviceBuffer());
    prep_arg.n_kvs = static_cast<const int32_t*>(n_kvs_buf.GetDeviceBuffer());
    prep_arg.num_seqs = batch;
    constexpr int prepare_threads = 256;
    constexpr int rows_per_prepare_block = prepare_threads / 32;
    auto row_blocks_for = [](int64_t rows) {
        return static_cast<int>(
            std::min<int64_t>((rows + rows_per_prepare_block - 1) /
                                  rows_per_prepare_block,
                              4096));
    };
    for(int i = 0; i < batch; ++i)
    {
        auto& prep = prep_arg.seqs[i];
        prep.q = static_cast<const uint16_t*>(q_buf.GetDeviceBuffer());
        prep.k = static_cast<const uint16_t*>(k_buf.GetDeviceBuffer());
        prep.v = static_cast<const uint16_t*>(v_buf.GetDeviceBuffer());
        prep.q_pack = static_cast<uint16_t*>(q_pack_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * h * max_nq_pad * d;
        prep.k_pack = static_cast<uint16_t*>(k_pack_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * hkv * max_nkv_pad * d;
        prep.v_pack = static_cast<uint16_t*>(v_pack_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * hkv * max_nkv_pad * d;
        prep.block_table =
            static_cast<const int32_t*>(block_table_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * pages_per_seq;
        prep.mm_ranges = mm_ranges == nullptr
                             ? nullptr
                             : mm_ranges +
                                   i * ck_tile::kUnifiedAttentionMmRangeStride;
        prep.active_counts =
            static_cast<int32_t*>(counts_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * max_rows;
        prep.active_indices =
            static_cast<int32_t*>(indices_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * max_rows * max_nb;
        prep.task_begin = prep_arg.total_blocks;
        prep.active_blocks = direct_causal ? 0 : max_rows;
        prep.q_blocks = direct_page
                            ? 0
                            : row_blocks_for(static_cast<int64_t>(max_nq) * h);
        prep.kv_blocks =
            direct_page
                ? 0
                : row_blocks_for(static_cast<int64_t>(max_nkv) * hkv);
        prep.n_q_pad = max_nq_pad;
        prep.n_kv_pad = max_nkv_pad;
        prep.pack_start = 0;
        prep.h_q = h;
        prep.h_kv = hkv;
        prep.d = d;
        prep.d_pad = d;
        prep.page_size = page_size;
        prep.nb = max_nb;
        prep.max_mm_ranges = use_mm_prefix ? 4 : 0;
        prep.sliding_window = sliding_window;
        prep.q_stride_t = static_cast<int64_t>(h) * d;
        prep.q_stride_h = d;
        prep.k_s0 = static_cast<int64_t>(page_size) * hkv * d;
        prep.k_s1 = static_cast<int64_t>(hkv) * d;
        prep.k_s2 = d;
        prep.v_s0 = static_cast<int64_t>(page_size) * hkv * d;
        prep.v_s1 = static_cast<int64_t>(hkv) * d;
        prep.v_s2 = d;
        prep_arg.total_blocks +=
            prep.active_blocks + prep.q_blocks + prep.kv_blocks;
    }
    using PrepareKernel = UnifiedAttentionD256PrepareBatchKernel;
    using PreparePolicy =
        UnifiedAttentionD256MmacDefaultPolicy<UnifiedAttentionD256AttentionProblem>;
    constexpr auto prepare_block = PrepareKernel::BlockSize();
    const int max_seen_words = (max_nb + 31) / 32;
    auto prepare_fn = ck_tile::make_kernel<prepare_block.x, PreparePolicy::kLaunchMinBlocks>(
        PrepareKernel{},
        PrepareKernel::GridSize(prep_arg),
        prepare_block,
        static_cast<size_t>(max_seen_words * sizeof(uint32_t)),
        prep_arg);

    UnifiedAttentionD256FusedBatchKernelArgument arg{};
    arg.num_seqs = batch;
    arg.total_nqb = total_nqb;
    arg.total_q_tasks = total_nqb;
    arg.split_qb_begin = -1;
    arg.q_offsets = static_cast<const int32_t*>(cu_q_buf.GetDeviceBuffer());
    arg.n_kvs_device = static_cast<const int32_t*>(n_kvs_buf.GetDeviceBuffer());
    arg.H = h; arg.H_KV = hkv; arg.head_dim = d; arg.padded_dim = d;
    arg.active_capacity = max_nb; arg.kv_stage_count = kv_stage_count;
    arg.max_mm_ranges = use_mm_prefix ? 4 : 0;
    arg.sliding_window = sliding_window; arg.causal = true; arg.alibi_sqrt = alibi_sqrt;
    arg.qk_scale = ck_tile::kQkScaleLog2e(d);
    arg.softcap = softcap; arg.sinks = sinks; arg.alibi = alibi; arg.qq_bias = qq_bias;
    arg.qq_bias_stride = max_nq;
    for(int i = 0; i < batch; ++i)
    {
        arg.q_bases[i] =
            direct_page
                ? static_cast<const uint16_t*>(q_buf.GetDeviceBuffer())
                : static_cast<const uint16_t*>(q_pack_buf.GetDeviceBuffer()) +
                      static_cast<int64_t>(i) * h * max_nq_pad * d;
        arg.k_bases[i] =
            direct_page
                ? static_cast<const uint16_t*>(k_buf.GetDeviceBuffer())
                : static_cast<const uint16_t*>(k_pack_buf.GetDeviceBuffer()) +
                      static_cast<int64_t>(i) * hkv * max_nkv_pad * d;
        arg.v_bases[i] =
            direct_page
                ? static_cast<const uint16_t*>(v_buf.GetDeviceBuffer())
                : static_cast<const uint16_t*>(v_pack_buf.GetDeviceBuffer()) +
                      static_cast<int64_t>(i) * hkv * max_nkv_pad * d;
        arg.o_bases[i] = kv_stage_count == 1
                             ? static_cast<uint16_t*>(out_buf.GetDeviceBuffer())
                             : static_cast<uint16_t*>(partial_out_buf.GetDeviceBuffer());
        arg.lse_bases[i] = kv_stage_count == 1
                               ? static_cast<float*>(lse_buf.GetDeviceBuffer()) +
                                     (static_cast<int64_t>(i) * 2 + 1) * h * max_nq_pad
                               : static_cast<float*>(lse_buf.GetDeviceBuffer());
        arg.active_indices_bases[i] =
            static_cast<const int32_t*>(indices_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * max_rows * max_nb;
        arg.active_counts_bases[i] =
            static_cast<const int32_t*>(counts_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * max_rows;
        arg.mm_ranges_bases[i] =
            mm_ranges == nullptr
                ? nullptr
                : mm_ranges + i * ck_tile::kUnifiedAttentionMmRangeStride;
        arg.block_table_bases[i] =
            static_cast<const int32_t*>(block_table_buf.GetDeviceBuffer()) +
            static_cast<int64_t>(i) * pages_per_seq;
        arg.stride_qh[i] = static_cast<int64_t>(max_nq_pad) * d;
        arg.stride_kh[i] = static_cast<int64_t>(max_nkv_pad) * d;
        arg.stride_vh[i] = static_cast<int64_t>(max_nkv_pad) * d;
        arg.stride_os[i] = kv_stage_count == 1
                               ? 0
                               : static_cast<int64_t>(h) * total_nq * d;
        arg.stride_oz[i] = 0;
        arg.stride_oh[i] = kv_stage_count == 1 ? d : static_cast<int64_t>(total_nq) * d;
        arg.stride_om[i] = kv_stage_count == 1 ? static_cast<int64_t>(h) * d : d;
        arg.stride_ok[i] = 1;
        arg.stride_ls[i] = kv_stage_count == 1 ? 0 : static_cast<int64_t>(h) * total_nq;
        arg.stride_lz[i] = kv_stage_count == 1 ? max_nq_pad : total_nq;
        arg.stride_lm[i] = 1;
    }
    arg.k_page_stride = static_cast<int64_t>(page_size) * hkv * d;
    arg.k_token_stride = static_cast<int64_t>(hkv) * d;
    arg.k_head_stride = d;
    arg.v_page_stride = static_cast<int64_t>(page_size) * hkv * d;
    arg.v_token_stride = static_cast<int64_t>(hkv) * d;
    arg.v_head_stride = d;
    arg.q_token_stride = static_cast<int64_t>(h) * d;
    arg.q_head_stride = d;

    using Problem = UnifiedAttentionD256AttentionProblemT<DataType>;
    using Policy = UnifiedAttentionD256MmacDefaultPolicy<Problem>;
    const ck_tile::stream_config stream_config{
        nullptr, true, 0, parser.get_int("warmup"), parser.get_int("repeat")};
    float ms = 0.0f;
    // The tiny kernel stores one score per wave lane (at most 64 keys).
    const bool use_tiny = max_nq <= 16 && max_nkv <= 64;
    const bool use_compact_direct_page =
        !use_tiny && kv_stage_count == 1 && direct_causal &&
        page_size > 0 && max_nq <= page_size &&
        !use_sinks && !use_qq_bias && softcap <= 0.0f;
    if(use_tiny)
    {
        UnifiedAttentionD256TinyBatchArgument tiny{};
        tiny.q = static_cast<const uint16_t*>(q_buf.GetDeviceBuffer());
        tiny.k = static_cast<const uint16_t*>(k_buf.GetDeviceBuffer());
        tiny.v = static_cast<const uint16_t*>(v_buf.GetDeviceBuffer());
        tiny.out = static_cast<uint16_t*>(out_buf.GetDeviceBuffer());
        tiny.block_table = static_cast<const int32_t*>(block_table_buf.GetDeviceBuffer());
        tiny.sinks = sinks; tiny.alibi = alibi; tiny.qq_bias = qq_bias;
        tiny.mm_ranges = mm_ranges;
        if(batch == 1)
        {
            tiny.q_offsets[0] = 0; tiny.q_offsets[1] = total_nq;
            tiny.n_kvs[0] = n_kvs[0];
        }
        else
        {
            tiny.q_offsets_device = static_cast<const int32_t*>(cu_q_buf.GetDeviceBuffer());
            tiny.n_kvs_device = static_cast<const int32_t*>(n_kvs_buf.GetDeviceBuffer());
        }
        tiny.num_seqs = batch; tiny.h_q = h; tiny.h_kv = hkv; tiny.head_dim = d;
        tiny.page_size = page_size; tiny.max_mm_ranges = use_mm_prefix ? 4 : 0;
        tiny.qq_bias_stride = max_nq; tiny.sliding_window = sliding_window;
        tiny.total_q_rows = total_nq * h;
        tiny.q_stride_t = static_cast<int64_t>(h) * d; tiny.q_stride_h = d;
        tiny.k_stride_page = static_cast<int64_t>(page_size) * hkv * d;
        tiny.k_stride_token = static_cast<int64_t>(hkv) * d; tiny.k_stride_head = d;
        tiny.v_stride_page = static_cast<int64_t>(page_size) * hkv * d;
        tiny.v_stride_token = static_cast<int64_t>(hkv) * d; tiny.v_stride_head = d;
        tiny.out_stride_t = static_cast<int64_t>(h) * d; tiny.out_stride_h = d;
        tiny.block_table_stride = pages_per_seq;
        tiny.qk_scale_log2 = ck_tile::kQkScaleLog2e(d);
        tiny.softcap = softcap; tiny.alibi_sqrt = alibi_sqrt;
#define UA_LAUNCH_D256_TINY(HAS_SOFTCAP, HAS_QQ_BIAS)                              \
        do                                                                           \
        {                                                                            \
            using Kernel = UnifiedAttentionD256TinyBatchKernel<                      \
                DataType, HAS_SOFTCAP, HAS_QQ_BIAS>;                                 \
            constexpr auto block = Kernel::BlockSize();                              \
            ms = ck_tile::launch_kernel(                                             \
                stream_config,                                                       \
                ck_tile::make_kernel<block.x, 1>(                                    \
                    Kernel{}, Kernel::GridSize(tiny), block, 0, tiny));               \
        } while(false)
        if(softcap > 0)
        {
            if(use_qq_bias) UA_LAUNCH_D256_TINY(true, true);
            else UA_LAUNCH_D256_TINY(true, false);
        }
        else if(use_qq_bias) UA_LAUNCH_D256_TINY(false, true);
        else UA_LAUNCH_D256_TINY(false, false);
#undef UA_LAUNCH_D256_TINY
    }
    else if(use_compact_direct_page)
    {
        const UnifiedAttentionD256DirectPageBatchKernelArgument direct_arg{
            static_cast<const uint16_t*>(q_buf.GetDeviceBuffer()),
            static_cast<const uint16_t*>(k_buf.GetDeviceBuffer()),
            static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()),
            static_cast<uint16_t*>(out_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(cu_q_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(n_kvs_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(block_table_buf.GetDeviceBuffer()),
            batch, total_nqb, h, hkv, d, d,
            static_cast<int64_t>(h) * d, d,
            static_cast<int64_t>(page_size) * hkv * d,
            static_cast<int64_t>(hkv) * d, d,
            static_cast<int64_t>(page_size) * hkv * d,
            static_cast<int64_t>(hkv) * d, d,
            static_cast<int64_t>(h) * d, d, 1, pages_per_seq,
            ck_tile::kQkScaleLog2e(d), alibi};
#define UA_LAUNCH_D256_DIRECT(ALIBI_MODE)                                           \
        do                                                                           \
        {                                                                            \
            using Kernel = UnifiedAttentionD256FusedBatchKernel<                     \
                ck_tile::kUnifiedAttention2dQueryBlockSize, true, false, false,      \
                false, true, true, false,                                             \
                ALIBI_MODE, true, Policy>;                                            \
            constexpr auto block = Kernel::BlockSize();                              \
            auto direct_fn = ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(\
                Kernel{}, Kernel::GridSize(direct_arg), block, 0, direct_arg);       \
            if(prep_arg.total_blocks > 0)                                            \
                ms = ck_tile::launch_kernel(stream_config, prepare_fn, direct_fn);   \
            else                                                                     \
                ms = ck_tile::launch_kernel(stream_config, direct_fn);               \
        } while(false)
        if(use_alibi)
        {
            if(alibi_sqrt) UA_LAUNCH_D256_DIRECT(UnifiedAttentionD256AlibiMode::Sqrt);
            else UA_LAUNCH_D256_DIRECT(UnifiedAttentionD256AlibiMode::Linear);
        }
        else UA_LAUNCH_D256_DIRECT(UnifiedAttentionD256AlibiMode::None);
#undef UA_LAUNCH_D256_DIRECT
    }
    else
    {
#define UA_LAUNCH_D256_BATCH(HAS_SOFTCAP, HAS_QQ_BIAS, DIRECT_CAUSAL)                 \
    do                                                                                \
    {                                                                                 \
        using Kernel = UnifiedAttentionD256FusedBatchKernel<                          \
            ck_tile::kUnifiedAttention2dQueryBlockSize, true, false, HAS_SOFTCAP,     \
            HAS_QQ_BIAS, DIRECT_CAUSAL, false, true,                                 \
            UnifiedAttentionD256AlibiMode::Runtime, false, Policy>;                  \
        constexpr auto block = Kernel::BlockSize();                                  \
        auto attn_fn = ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(       \
            Kernel{}, Kernel::GridSize(arg), block, 0, arg);                         \
        if(kv_stage_count > 1)                                                       \
        {                                                                             \
            const auto reduce_arg =                                                  \
                UnifiedAttentionD256KvStageReduceKernel<Policy>::MakeKargs(           \
                    static_cast<const uint16_t*>(partial_out_buf.GetDeviceBuffer()),  \
                    static_cast<const float*>(lse_buf.GetDeviceBuffer()),             \
                    static_cast<const int32_t*>(cu_q_buf.GetDeviceBuffer()) + batch,   \
                    static_cast<uint16_t*>(out_buf.GetDeviceBuffer()),                 \
                    static_cast<float*>(lse_buf.GetDeviceBuffer()) +                  \
                        static_cast<int64_t>(kv_stage_count) * h * total_nq,          \
                    1, h, total_nq, d, kv_stage_count,                                \
                    static_cast<int64_t>(h) * total_nq * d,                           \
                    static_cast<int64_t>(h) * total_nq,                               \
                    0, d, static_cast<int64_t>(h) * d, 1, total_nq, 1);               \
            auto launch_with_reduce = [&](auto stage) {                               \
                constexpr int StageCount = decltype(stage)::value;                     \
                using ReduceKernel =                                                  \
                    UnifiedAttentionD256KvStageReduceKernel<Policy, StageCount>;      \
                constexpr auto reduce_block = ReduceKernel::BlockSize();              \
                auto reduce_fn =                                                      \
                    ck_tile::make_kernel<reduce_block.x, Policy::kLaunchMinBlocks>(   \
                        ReduceKernel{}, ReduceKernel::GridSize(reduce_arg),           \
                        reduce_block, 0, reduce_arg);                                 \
                if(prep_arg.total_blocks > 0)                                         \
                    return ck_tile::launch_kernel(                                    \
                        stream_config, prepare_fn, attn_fn, reduce_fn);               \
                else                                                                  \
                    return ck_tile::launch_kernel(stream_config, attn_fn, reduce_fn); \
            };                                                                        \
            switch(kv_stage_count)                                                    \
            {                                                                         \
            case 2: ms = launch_with_reduce(ck_tile::number<2>{}); break;             \
            case 3: ms = launch_with_reduce(ck_tile::number<3>{}); break;             \
            case 4: ms = launch_with_reduce(ck_tile::number<4>{}); break;             \
            case 5: ms = launch_with_reduce(ck_tile::number<5>{}); break;             \
            default: return 2;                                                        \
            }                                                                         \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            if(prep_arg.total_blocks > 0)                                             \
                ms = ck_tile::launch_kernel(stream_config, prepare_fn, attn_fn);      \
            else                                                                      \
                ms = ck_tile::launch_kernel(stream_config, attn_fn);                  \
        }                                                                             \
    } while(false)
    if(direct_causal)
    {
        if(softcap > 0) { if(use_qq_bias) UA_LAUNCH_D256_BATCH(true, true, true);
                          else UA_LAUNCH_D256_BATCH(true, false, true); }
        else if(use_qq_bias) UA_LAUNCH_D256_BATCH(false, true, true);
        else UA_LAUNCH_D256_BATCH(false, false, true);
    }
    else
    {
        if(softcap > 0) { if(use_qq_bias) UA_LAUNCH_D256_BATCH(true, true, false);
                          else UA_LAUNCH_D256_BATCH(true, false, false); }
        else if(use_qq_bias) UA_LAUNCH_D256_BATCH(false, true, false);
        else UA_LAUNCH_D256_BATCH(false, false, false);
    }
#undef UA_LAUNCH_D256_BATCH
    }
    HIP_CHECK_ERROR(hipGetLastError());
    bool pass = true;
    if(parser.get_bool("v"))
    {
        const std::size_t count = static_cast<std::size_t>(total_nq) * h * d;
        ck_tile::DeviceMem ref_out_buf(count * sizeof(uint16_t));
        const UnifiedAttentionReferenceArgument reference_arg{
            static_cast<const uint16_t*>(q_buf.GetDeviceBuffer()),
            static_cast<const uint16_t*>(k_buf.GetDeviceBuffer()),
            static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(block_table_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(cu_q_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(n_kvs_buf.GetDeviceBuffer()),
            sinks,
            alibi,
            qq_bias,
            mm_ranges,
            static_cast<uint16_t*>(ref_out_buf.GetDeviceBuffer()),
            batch,
            total_nq,
            h,
            hkv,
            d,
            page_size,
            pages_per_seq,
            max_nq,
            use_mm_prefix ? 4 : 0,
            sliding_window,
            h * d,
            d,
            static_cast<int64_t>(page_size) * hkv * d,
            static_cast<int64_t>(hkv) * d,
            d,
            static_cast<int64_t>(page_size) * hkv * d,
            static_cast<int64_t>(hkv) * d,
            d,
            1.0f / std::sqrt(static_cast<float>(d)),
            softcap,
            alibi_sqrt,
            true};
        ck_tile::reference_unified_attention<DataType>(reference_arg);
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        std::vector<DataType> actual(count);
        std::vector<DataType> reference(count);
        out_buf.FromDevice(actual.data());
        ref_out_buf.FromDevice(reference.data());
        pass = ck_tile::check_err(
            actual, reference, "Unified attention reference mismatch", 2.0e-2f, 2.0e-2f);
    }
    const char* path = use_tiny ? "2d-d256-tiny"
                       : use_compact_direct_page ? "2d-d256-direct-page"
                                                 : "2d-d256-batched";
    const char* type_name = std::is_same_v<DataType, ck_tile::bf16_t> ? "bf16" : "fp16";
    std::cout << "Unified Attention 2D Config: path=" << path
              << ", prec=" << type_name
              << ", B=" << batch << ", NQ=" << total_nq << ", NKV_MAX=" << max_nkv
              << ", H=" << h << ", HKV=" << hkv << ", D=" << d
              << ", page_size=" << page_size
              << ", kv_cache_blocks=" << physical_pages
              << ", block_table_width=" << pages_per_seq
              << ", direct_causal=" << direct_causal
              << ", kv_stages=" << kv_stage_count << '\n';
    std::cout << "unified_attention_2d Avg Latency: " << ms << " ms\n";
    std::cout << "Verification (unified_attention_2d): "
              << (parser.get_bool("v") ? (pass ? "PASSED" : "FAILED") : "SKIPPED")
              << '\n';
    return pass ? 0 : 1;
}

template <typename DataType>
int run_unified_attention_2d_d256(const ck_tile::ArgParser& parser)
{
    auto batch_query_lens = ck_tile::parse_query_lens_helper(parser.get_str("query_lens"));
    if(batch_query_lens.empty()) batch_query_lens.push_back(parser.get_int("nq"));
    return run_unified_attention_2d_d256_batch<DataType>(parser, batch_query_lens);
}
