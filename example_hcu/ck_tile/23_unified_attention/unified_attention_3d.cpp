// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#define CK_TILE_UNIFIED_ATTENTION_3D
#include "ck_tile/ops/unified_attention.hpp"
#undef CK_TILE_UNIFIED_ATTENTION_3D
#include "unified_attention_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser parser;
    parser.insert("b", "4", "number of decode sequences")
        .insert("nkv", "256", "KV sequence length")
        .insert("h", "16", "number of query heads")
        .insert("hkv", "2", "number of KV heads")
        .insert("d", "192", "head dimension: 192 or 256")
        .insert("block_size", "16", "paged KV block size: 16 or 32")
        .insert("kv_cache_blocks", "0", "physical KV-cache pages; 0 uses the required page count")
        .insert("block_table_width", "0", "logical pages per sequence; 0 uses ceil(nkv/block_size)")
        .insert("segments", "4", "parallel softmax segments")
        .insert("prec", "fp16", "data type: fp16 or bf16")
        .insert("causal", "1", "causal attention; only 1 is supported")
        .insert("mask", "causal", "causal, sliding, prefix, or sliding_prefix")
        .insert("sliding_window", "0", "left sliding-window size; 0 disables it")
        .insert("sinks", "0", "enable attention sinks")
        .insert("sink_value", "0", "constant sink value")
        .insert("alibi", "0", "enable ALiBi slopes")
        .insert("alibi_sqrt", "0", "use the ALiBi sqrt distance variant")
        .insert("qq_bias", "0", "enable query-query bias")
        .insert("qq_bias_value", "0.01", "constant query-query bias value")
        .insert("softcap", "0", "positive logit softcap; 0 disables it")
        .insert("mm_prefix", "0", "enable multimodal PrefixLM ranges")
        .insert("prefix_begin", "0", "inclusive PrefixLM range begin")
        .insert("prefix_end", "-1", "inclusive PrefixLM range end")
        .insert("warmup", "10", "warmup launches")
        .insert("repeat", "100", "timed launches")
        .insert("kernel_timing", "0", "time attention and segment-reduce kernels separately")
        .insert("print_output", "0", "print flattened CK output values")
        .insert("print_elements", "16", "number of CK output values to print")
        .insert("v", "1", "run full independent GPU reference validation");
    return std::make_tuple(parser.parse(argc, argv), parser);
}

template <typename DataType, int HeadDim, int PageSize>
int run(const ck_tile::ArgParser& parser)
{
    const int b = parser.get_int("b");
    const int nkv = parser.get_int("nkv");
    const int h = parser.get_int("h");
    const int hkv = parser.get_int("hkv");
    const int segments = parser.get_int("segments");
    const bool causal = parser.get_bool("causal");
    const std::string mask = parser.get_str("mask");
    const bool use_sliding = mask == "sliding" || mask == "sliding_prefix";
    const int sliding_window = parser.get_int("sliding_window");
    const bool use_sinks = parser.get_bool("sinks");
    const bool use_alibi = parser.get_bool("alibi");
    const bool alibi_sqrt = parser.get_bool("alibi_sqrt");
    const bool use_qq_bias = parser.get_bool("qq_bias");
    const float softcap = parser.get_float("softcap");
    const bool use_mm_prefix = parser.get_bool("mm_prefix") ||
                               mask == "prefix" || mask == "sliding_prefix";
    constexpr int d = HeadDim;
    if(b <= 0 || nkv <= 0 || h <= 0 || hkv <= 0 || h % hkv != 0 ||
       h / hkv > 16 || segments <= 0 ||
       segments > ck_tile::kUnifiedAttentionMaxDecodeSegments || !causal ||
       (mask != "causal" && mask != "sliding" && mask != "prefix" &&
        mask != "sliding_prefix") || (use_sliding && sliding_window <= 0) ||
       (!use_sliding && sliding_window != 0) || softcap < 0 ||
       (alibi_sqrt && !use_alibi))
    {
        std::cerr << "invalid b/nkv/h/hkv/segments configuration\n";
        return 2;
    }
    int effective_segments = segments;
    if(sliding_window == 0 && d > 32 &&
       segments < ck_tile::kUnifiedAttentionMaxDecodeSegments &&
       nkv >= segments * ck_tile::kUnifiedAttentionMetadataBlockSize *
                  ck_tile::kUnifiedAttentionMinBlocksPerSegment)
        effective_segments =
            std::min(ck_tile::kUnifiedAttentionMaxDecodeSegments,
                     segments * ck_tile::kUnifiedAttentionMinBlocksPerSegment);
    const bool fast_decode = !use_qq_bias && !use_mm_prefix && softcap <= 0 &&
                             (!use_alibi || !alibi_sqrt);
    int compact_segments = effective_segments;
    // Only SimpleDecode maps compact slots to the visible KV tail. The general
    // feature pipeline uses logical segment indices and must launch them all.
    if(sliding_window > 0 && fast_decode)
        compact_segments =
            std::min(effective_segments,
                     ck_tile::unified_attention_ceil_div(
                         sliding_window - 1,
                         ck_tile::kUnifiedAttentionMetadataBlockSize) +
                         1);
    const int pages_needed_per_seq = (nkv + PageSize - 1) / PageSize;
    const int requested_table_width = parser.get_int("block_table_width");
    const int pages_per_seq =
        requested_table_width > 0 ? requested_table_width : pages_needed_per_seq;
    const int required_pages = b * pages_needed_per_seq;
    const int requested_cache_blocks = parser.get_int("kv_cache_blocks");
    const int pages =
        requested_cache_blocks > 0 ? requested_cache_blocks : required_pages;
    if(pages_per_seq < pages_needed_per_seq || pages < required_pages)
    {
        std::cerr << "invalid KV-cache capacity or block-table width\n";
        return 2;
    }
    const std::size_t elem_bytes = sizeof(uint16_t);
    ck_tile::DeviceMem q_buf(static_cast<std::size_t>(b) * h * d * elem_bytes);
    ck_tile::DeviceMem k_buf(static_cast<std::size_t>(pages) * PageSize * hkv * d * elem_bytes);
    ck_tile::DeviceMem v_buf(static_cast<std::size_t>(pages) * PageSize * hkv * d * elem_bytes);
    ck_tile::DeviceMem out_buf(static_cast<std::size_t>(b) * h * d * elem_bytes);
    ck_tile::DeviceMem seg_out_buf(
        static_cast<std::size_t>(b) * h * compact_segments * d * elem_bytes);
    ck_tile::DeviceMem seg_max_buf(
        static_cast<std::size_t>(b) * h * compact_segments * sizeof(float));
    ck_tile::DeviceMem seg_sum_buf(
        static_cast<std::size_t>(b) * h * compact_segments * sizeof(float));
    ck_tile::DeviceMem cu_buf(static_cast<std::size_t>(b + 1) * sizeof(int32_t));
    ck_tile::DeviceMem lens_buf(static_cast<std::size_t>(b) * sizeof(int32_t));
    ck_tile::DeviceMem table_buf(static_cast<std::size_t>(b) * pages_per_seq * sizeof(int32_t));
    ck_tile::DeviceMem sinks_buf(static_cast<std::size_t>(h) * sizeof(float));
    ck_tile::DeviceMem alibi_buf(static_cast<std::size_t>(h) * sizeof(float));
    ck_tile::DeviceMem qq_bias_buf(static_cast<std::size_t>(b) * sizeof(DataType));
    ck_tile::DeviceMem mm_ranges_buf(
        static_cast<std::size_t>(b) * ck_tile::kUnifiedAttentionMmRangeStride *
        sizeof(int32_t));
    ck_tile::HostTensor<DataType> q_host({static_cast<std::size_t>(b), static_cast<std::size_t>(h), static_cast<std::size_t>(d)});
    ck_tile::HostTensor<DataType> k_host({static_cast<std::size_t>(pages), static_cast<std::size_t>(PageSize), static_cast<std::size_t>(hkv), static_cast<std::size_t>(d)});
    ck_tile::HostTensor<DataType> v_host({static_cast<std::size_t>(pages), static_cast<std::size_t>(PageSize), static_cast<std::size_t>(hkv), static_cast<std::size_t>(d)});
    ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, 13001, true}(q_host);
    ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, 13002, true}(k_host);
    ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, 13003, true}(v_host);
    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    out_buf.SetZero();

    std::vector<int32_t> cu(b + 1), lens(b, nkv), table(static_cast<std::size_t>(b) * pages_per_seq);
    for(int i = 0; i <= b; ++i) cu[i] = i;
    int block_alloc = 0;
    for(int seq = 0; seq < b; ++seq)
        for(int page = 0; page < pages_needed_per_seq; ++page)
            table[static_cast<std::size_t>(seq) * pages_per_seq + page] =
                block_alloc++;
    cu_buf.ToDevice(cu.data());
    lens_buf.ToDevice(lens.data());
    table_buf.ToDevice(table.data());

    const float* sinks = nullptr;
    const float* alibi = nullptr;
    const void* qq_bias = nullptr;
    const int32_t* mm_ranges = nullptr;
    if(use_sinks)
    {
        std::vector<float> values(h, parser.get_float("sink_value"));
        sinks_buf.ToDevice(values.data());
        sinks = static_cast<const float*>(sinks_buf.GetDeviceBuffer());
    }
    if(use_alibi)
    {
        std::vector<float> values(h);
        for(int i = 0; i < h; ++i)
            values[i] = std::exp2(-ck_tile::kUnifiedAttentionAlibiExponentScale *
                                  static_cast<float>(i + 1) / h);
        alibi_buf.ToDevice(values.data());
        alibi = static_cast<const float*>(alibi_buf.GetDeviceBuffer());
    }
    if(use_qq_bias)
    {
        std::vector<DataType> values(b,
            ck_tile::type_convert<DataType>(parser.get_float("qq_bias_value")));
        qq_bias_buf.ToDevice(values.data());
        qq_bias = qq_bias_buf.GetDeviceBuffer();
    }
    if(use_mm_prefix)
    {
        std::vector<int32_t> ranges(
            static_cast<std::size_t>(b) * ck_tile::kUnifiedAttentionMmRangeStride,
            -1);
        const int begin = parser.get_int("prefix_begin");
        const int end = parser.get_int("prefix_end") < 0 ? nkv - 1
                                                          : parser.get_int("prefix_end");
        for(int i = 0; i < b; ++i)
        {
            ranges[static_cast<std::size_t>(i) *
                   ck_tile::kUnifiedAttentionMmRangeStride] = begin;
            ranges[static_cast<std::size_t>(i) *
                       ck_tile::kUnifiedAttentionMmRangeStride +
                   1] = end;
        }
        mm_ranges_buf.ToDevice(ranges.data());
        mm_ranges = static_cast<const int32_t*>(mm_ranges_buf.GetDeviceBuffer());
    }

    using ReduceKernel = ua::CkTileReduceSegmentsKernel<DataType, d, 0>;
    constexpr int reduce_block = ReduceKernel::kBlockSize;
    const dim3 attn_grid(b, hkv, compact_segments);
    const dim3 reduce_grid(b, h, 1);
    const ck_tile::stream_config stream_config{
        nullptr, true, 0, parser.get_int("warmup"), parser.get_int("repeat")};
    float ms = 0.0f;
    float attention_ms = 0.0f;
    float reduce_ms = 0.0f;
    auto reduce_fn = ck_tile::make_kernel<reduce_block, 1>(
        ReduceKernel{}, reduce_grid, dim3(reduce_block), 0,
        static_cast<DataType*>(out_buf.GetDeviceBuffer()),
        static_cast<const DataType*>(seg_out_buf.GetDeviceBuffer()),
        static_cast<const float*>(seg_max_buf.GetDeviceBuffer()),
        static_cast<const float*>(seg_sum_buf.GetDeviceBuffer()),
        b, h, d, compact_segments,
        static_cast<int64_t>(h) * d, d,
        nullptr);

#define UA_LAUNCH_3D(KERNEL, ARGS)                                                    \
    do                                                                                \
    {                                                                                 \
        constexpr int attn_block = KERNEL::kBlockSize;                               \
        auto attn_fn = ck_tile::make_kernel<attn_block, 1>(                           \
            KERNEL{}, attn_grid, dim3(attn_block), 0, ARGS,                           \
            static_cast<DataType*>(seg_out_buf.GetDeviceBuffer()),                    \
            static_cast<float*>(seg_max_buf.GetDeviceBuffer()),                       \
            static_cast<float*>(seg_sum_buf.GetDeviceBuffer()),                       \
            effective_segments, compact_segments, 16, d);                            \
        ms = ck_tile::launch_kernel(stream_config, attn_fn, reduce_fn);               \
        if(parser.get_bool("kernel_timing"))                                         \
        {                                                                             \
            attention_ms = ck_tile::launch_kernel(stream_config, attn_fn);            \
            reduce_ms = ck_tile::launch_kernel(stream_config, reduce_fn);             \
        }                                                                             \
    } while(false)
    if(fast_decode)
    {
        const ua::SimpleDecodeArgs args{
            q_buf.GetDeviceBuffer(), k_buf.GetDeviceBuffer(), v_buf.GetDeviceBuffer(),
            static_cast<const int32_t*>(cu_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(lens_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(table_buf.GetDeviceBuffer()),
            sinks, alibi, pages_per_seq, h, hkv, h / hkv, d,
            1.0f / std::sqrt(static_cast<float>(d)), sliding_window, use_sinks};
        if(use_alibi)
        {
            using Kernel = ua::CkTileAttention3DKernel<DataType, d, PageSize, true, true>;
            UA_LAUNCH_3D(Kernel, args);
        }
        else
        {
            using Kernel = ua::CkTileAttention3DKernel<DataType, d, PageSize, true, false>;
            UA_LAUNCH_3D(Kernel, args);
        }
    }
    else
    {
        const ua::CommonArgs args{
            q_buf.GetDeviceBuffer(), k_buf.GetDeviceBuffer(), v_buf.GetDeviceBuffer(),
            out_buf.GetDeviceBuffer(),
            static_cast<const int32_t*>(cu_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(lens_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(table_buf.GetDeviceBuffer()),
            sinks, alibi, qq_bias, mm_ranges,
            static_cast<int64_t>(h) * d, d,
            static_cast<int64_t>(PageSize) * hkv * d,
            static_cast<int64_t>(hkv) * d, d, 1,
            static_cast<int64_t>(PageSize) * hkv * d,
            static_cast<int64_t>(hkv) * d, d, 1,
            static_cast<int64_t>(h) * d, d,
            pages_per_seq, 1,
            b, b, h, hkv, d, PageSize, 4,
            1.0f / std::sqrt(static_cast<float>(d)), softcap, sliding_window,
            use_sinks, use_alibi, use_qq_bias, use_mm_prefix, alibi_sqrt};
        using Kernel = ua::CkTileAttention3DKernel<DataType, d, 0, false, false>;
        UA_LAUNCH_3D(Kernel, args);
    }
#undef UA_LAUNCH_3D
    HIP_CHECK_ERROR(hipGetLastError());

    bool pass = true;
    if(parser.get_bool("v"))
    {
        const std::size_t count = static_cast<std::size_t>(b) * h * d;
        ck_tile::DeviceMem ref_out_buf(count * sizeof(uint16_t));
        const UnifiedAttentionReferenceArgument reference_arg{
            static_cast<const uint16_t*>(q_buf.GetDeviceBuffer()),
            static_cast<const uint16_t*>(k_buf.GetDeviceBuffer()),
            static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(table_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(cu_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(lens_buf.GetDeviceBuffer()),
            sinks,
            alibi,
            reinterpret_cast<const float*>(qq_bias),
            mm_ranges,
            static_cast<uint16_t*>(ref_out_buf.GetDeviceBuffer()),
            b,
            b,
            h,
            hkv,
            d,
            PageSize,
            pages_per_seq,
            1,
            use_mm_prefix ? 4 : 0,
            sliding_window,
            h * d,
            d,
            static_cast<int64_t>(PageSize) * hkv * d,
            static_cast<int64_t>(hkv) * d,
            d,
            static_cast<int64_t>(PageSize) * hkv * d,
            static_cast<int64_t>(hkv) * d,
            d,
            1.0f / std::sqrt(static_cast<float>(d)),
            softcap,
            alibi_sqrt,
            false};
        ck_tile::reference_unified_attention<DataType>(reference_arg);
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        std::vector<DataType> actual(count);
        std::vector<DataType> reference(count);
        out_buf.FromDevice(actual.data());
        ref_out_buf.FromDevice(reference.data());
        pass = ck_tile::check_err(
            actual, reference, "Unified attention reference mismatch", 2.0e-2f, 2.0e-2f);
    }
    const char* type_name = std::is_same_v<DataType, ck_tile::bf16_t> ? "bf16" : "fp16";
    std::cout << "Unified Attention 3D Config: path=3d, prec=" << type_name
              << ", B=" << b << ", NKV=" << nkv << ", H=" << h
              << ", HKV=" << hkv << ", D=" << d << ", page_size=" << PageSize
              << ", segments=" << segments
              << ", effective_segments=" << effective_segments
              << ", compact_segments=" << compact_segments
              << ", causal=" << causal
              << ", mask=" << mask << ", sinks=" << use_sinks
              << ", alibi=" << use_alibi << ", alibi_sqrt=" << alibi_sqrt
              << ", qq_bias=" << use_qq_bias << ", softcap=" << softcap
              << ", mm_prefix=" << use_mm_prefix
              << ", kv_cache_blocks=" << pages
              << ", block_table_width=" << pages_per_seq << '\n';
    std::cout << "unified_attention_3d Avg Latency: " << ms << " ms\n";
    if(parser.get_bool("kernel_timing"))
    {
        std::cout << "ua_3d_attention_ms:" << attention_ms << '\n';
        std::cout << "ua_3d_reduce_ms:" << reduce_ms << '\n';
    }
    if(parser.get_bool("print_output"))
    {
        const std::size_t count = static_cast<std::size_t>(b) * h * d;
        const int requested = parser.get_int("print_elements");
        const std::size_t print_count =
            requested > 0 ? std::min(count, static_cast<std::size_t>(requested)) : 0;
        std::vector<DataType> output(count);
        out_buf.FromDevice(output.data());
        std::cout << "ua_3d_output[0:" << print_count << "]:";
        for(std::size_t i = 0; i < print_count; ++i)
            std::cout << (i == 0 ? " " : ", ")
                      << ck_tile::type_convert<float>(output[i]);
        std::cout << '\n';
    }
    std::cout << "Verification (unified_attention_3d): "
              << (parser.get_bool("v") ? (pass ? "PASSED" : "FAILED") : "SKIPPED")
              << '\n';
    return pass ? 0 : 1;
}

template <typename DataType, int HeadDim>
int dispatch_page(const ck_tile::ArgParser& parser)
{
    return parser.get_int("block_size") == 16 ? run<DataType, HeadDim, 16>(parser)
                                               : run<DataType, HeadDim, 32>(parser);
}

int main(int argc, char* argv[])
{
    auto [ok, parser] = create_args(argc, argv);
    if(!ok) return 0;
    const int d = parser.get_int("d");
    const int page = parser.get_int("block_size");
    const std::string prec = parser.get_str("prec");
    if((d != 192 && d != 256) || (page != 16 && page != 32))
    {
        std::cerr << "d must be 192/256 and block_size must be 16/32\n";
        return 2;
    }
    if(prec == "fp16") return d == 192 ? dispatch_page<ck_tile::fp16_t, 192>(parser)
                                        : dispatch_page<ck_tile::fp16_t, 256>(parser);
    if(prec == "bf16") return d == 192 ? dispatch_page<ck_tile::bf16_t, 192>(parser)
                                        : dispatch_page<ck_tile::bf16_t, 256>(parser);
    std::cerr << "prec must be fp16 or bf16\n";
    return 2;
}
