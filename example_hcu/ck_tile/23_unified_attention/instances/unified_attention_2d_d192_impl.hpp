// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once


#include "ck_tile/host.hpp"
#include "ck_tile/ops/unified_attention.hpp"
#include "../unified_attention_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

template <typename DataType>
int run_unified_attention_2d_d192_single(
    const ck_tile::ArgParser& parser,
    const ck_tile::UnifiedAttentionProblem& prob)
{
    const int d = parser.get_int("d");
    ck_tile::UnifiedAttentionData<DataType> data(prob, 11001, false);
    const int n_q = prob.total_nq;
    const int n_kv = prob.max_nkv;
    const int nqb = ck_tile::unified_attention_ceil_div(
        n_q, ck_tile::kUnifiedAttentionMetadataBlockSize);
    const int nb = ck_tile::unified_attention_ceil_div(
        n_kv, ck_tile::kUnifiedAttentionMetadataBlockSize);
    const int rows = prob.h * nqb;
    const int window_tiles = prob.sliding_window > 0
                                 ? ck_tile::unified_attention_ceil_div(
                                       prob.sliding_window + n_q - 1,
                                       ck_tile::kUnifiedAttentionMetadataBlockSize)
                                 : nb;
    const int kv_stages = prob.sliding_window > 0
                              ? 1
                              : (nb <= 16
                                     ? 1
                                     : std::max(1,
                                                std::min(prob.kv_stages,
                                                         window_tiles)));
    const bool single_stage = kv_stages == 1;
    const bool fuse_d192_tail =
        prob.sliding_window <= 0 && rows * kv_stages >= 160;
    auto blocks_for = [](int64_t work_items) {
        constexpr int block = 256;
        return static_cast<unsigned int>(std::min<int64_t>(
            4096, (work_items + block - 1) / block));
    };
    const int64_t q_work =
        static_cast<int64_t>(prob.max_nq_pad) * prob.h * d /
        ck_tile::kUnifiedAttentionPackVectorSize;
    const int64_t kv_work =
        static_cast<int64_t>(prob.nkv_pad - prob.kv_pack_start) * prob.hkv * d /
        ck_tile::kUnifiedAttentionPackVectorSize;
    const UnifiedAttentionSinglePackArgs pack_args{
        reinterpret_cast<const uint16_t*>(data.q_buf.GetDeviceBuffer()),
        reinterpret_cast<const uint16_t*>(data.k_buf.GetDeviceBuffer()),
        reinterpret_cast<const uint16_t*>(data.v_buf.GetDeviceBuffer()),
        reinterpret_cast<uint16_t*>(data.q_pack_buf.GetDeviceBuffer()),
        reinterpret_cast<uint16_t*>(data.k_pack_buf.GetDeviceBuffer()),
        reinterpret_cast<uint16_t*>(data.v_pack_buf.GetDeviceBuffer()),
        static_cast<const int32_t*>(data.block_table_buf.GetDeviceBuffer()),
        n_q,
        prob.max_nq_pad,
        prob.h,
        n_kv,
        prob.nkv_pad,
        prob.kv_pack_start,
        prob.hkv,
        d,
        d,
        prob.page_size,
        static_cast<int64_t>(prob.h) * d,
        d,
        static_cast<int64_t>(prob.page_size) * prob.hkv * d,
        static_cast<int64_t>(prob.hkv) * d,
        d,
        static_cast<int64_t>(prob.page_size) * prob.hkv * d,
        static_cast<int64_t>(prob.hkv) * d,
        d,
        false,
        blocks_for(prob.sliding_window > 0 ? q_work + kv_work
                                           : std::max(q_work, kv_work))};
    auto pack_fn = [&](const ck_tile::stream_config& s) {
        if(prob.sliding_window > 0)
        {
            using PackKernel = UnifiedAttentionSinglePackKernel<
                ck_tile::kUnifiedAttentionPackVectorSize,
                UnifiedAttentionSinglePackSchedule::Concatenated>;
            launch_unified_attention_pack<PackKernel>(pack_args, s.stream_id_);
        }
        else
        {
            using PackKernel = UnifiedAttentionSinglePackKernel<
                ck_tile::kUnifiedAttentionPackVectorSize,
                UnifiedAttentionSinglePackSchedule::ParallelQKv>;
            launch_unified_attention_pack<PackKernel>(pack_args, s.stream_id_);
        }
    };

    using Problem = UnifiedAttentionMmacProblemT<DataType>;
    using Policy = unified_attention::UnifiedAttentionMmacDefaultPolicy<Problem, 3>;
    auto* fused_out = single_stage
                          ? reinterpret_cast<uint16_t*>(data.out_buf.GetDeviceBuffer())
                          : reinterpret_cast<uint16_t*>(
                                data.partial_out_buf.GetDeviceBuffer());
    auto* fused_lse = static_cast<float*>(data.lse_buf.GetDeviceBuffer()) +
                      (single_stage ? static_cast<int64_t>(prob.h) * n_q : 0);
    const int64_t stride_os =
        single_stage ? 0 : static_cast<int64_t>(prob.h) * n_q * d;
    const int64_t stride_ls =
        single_stage ? 0 : static_cast<int64_t>(prob.h) * n_q;
    const int64_t stride_oh = single_stage ? d : static_cast<int64_t>(n_q) * d;
    const int64_t stride_om = single_stage ? static_cast<int64_t>(prob.h) * d : d;
    const ck_tile::stream_config stream_config{
        nullptr, true, 1, parser.get_int("warmup"), parser.get_int("repeat")};
    float ms = 0.0f;

    auto reduce_fn = [&](const ck_tile::stream_config& s) {
        if(single_stage) return;
        using ReduceKernel = UnifiedAttentionKvStageReduceKernel<Policy>;
        const auto reduce_args = ReduceKernel::MakeKargs(
            reinterpret_cast<const uint16_t*>(data.partial_out_buf.GetDeviceBuffer()),
            static_cast<const float*>(data.lse_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.seqlens_buf.GetDeviceBuffer()),
            reinterpret_cast<uint16_t*>(data.out_buf.GetDeviceBuffer()),
            static_cast<float*>(data.lse_buf.GetDeviceBuffer()) +
                static_cast<int64_t>(kv_stages) * prob.h * n_q,
            1,
            prob.h,
            n_q,
            d,
            kv_stages,
            static_cast<int64_t>(prob.h) * n_q * d,
            static_cast<int64_t>(prob.h) * n_q,
            0,
            d,
            static_cast<int64_t>(prob.h) * d,
            1,
            n_q,
            1);
        UnifiedAttentionKvStageReduceKernelInvoker<Policy>::Run(
            reduce_args, s.stream_id_);
    };

#define UA_RUN_SINGLE(HAS_SOFTCAP, HAS_QQ_BIAS, HAS_MM_PREFIX)                       \
    do                                                                                \
    {                                                                                 \
        using Kernel = UnifiedAttentionFusedKernel<                                   \
            ck_tile::kUnifiedAttention2dQueryBlockSize, true, false, HAS_SOFTCAP,     \
            HAS_QQ_BIAS, Policy, false,                                               \
            HAS_MM_PREFIX>;                                                           \
        using FusedTailKernel = UnifiedAttentionFusedKernel<                          \
            ck_tile::kUnifiedAttention2dQueryBlockSize, true, false, HAS_SOFTCAP,     \
            HAS_QQ_BIAS, Policy, true,                                                \
            HAS_MM_PREFIX>;                                                           \
        const auto kargs = Kernel::MakeKargs(                                         \
            reinterpret_cast<const uint16_t*>(data.q_pack_buf.GetDeviceBuffer()),    \
            reinterpret_cast<const uint16_t*>(data.k_pack_buf.GetDeviceBuffer()),    \
            reinterpret_cast<const uint16_t*>(data.v_pack_buf.GetDeviceBuffer()),    \
            static_cast<const int32_t*>(data.seqlens_buf.GetDeviceBuffer()),          \
            data.sinks_ptr, data.alibi_ptr, data.qq_bias_ptr, data.mm_ranges_ptr,     \
            nullptr,                                                                  \
            static_cast<const int32_t*>(data.indices_buf.GetDeviceBuffer()),          \
            static_cast<const int32_t*>(data.counts_buf.GetDeviceBuffer()),           \
            fused_out, fused_lse,                                                     \
            prob.h, prob.hkv, n_q, n_kv, d, d, rows, nb, nqb, nb, kv_stages, 0, 0,  \
            stride_os, stride_ls, 0.0f,                                               \
            ck_tile::kQkScaleLog2e(d),                                   \
            prob.softcap, prob.max_nq_pad, nb, true, prob.use_mm_prefix ? 4 : 0,     \
            prob.sliding_window, prob.alibi_sqrt,                                    \
            static_cast<int64_t>(prob.max_nq_pad) * d,                               \
            static_cast<int64_t>(prob.nkv_pad) * d,                                  \
            static_cast<int64_t>(prob.nkv_pad) * d,                                  \
            0, stride_oh, stride_om, 1, n_q, 1);                                     \
        if(!Kernel::IsSupportedArgument(kargs)) return 2;                             \
        if(fuse_d192_tail)                                                            \
        {                                                                             \
            constexpr auto block = FusedTailKernel::BlockSize();                     \
            auto attn_fn = ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(   \
                FusedTailKernel{}, FusedTailKernel::GridSize(kargs), block, 0, kargs);\
            ms = ck_tile::launch_kernel(stream_config, pack_fn, attn_fn, reduce_fn); \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            constexpr auto block = Kernel::BlockSize();                              \
            auto attn_fn = ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(   \
                Kernel{}, Kernel::GridSize(kargs), block, 0, kargs);                  \
            ms = ck_tile::launch_kernel(stream_config, pack_fn, attn_fn, reduce_fn); \
        }                                                                             \
    } while(false)

    if(prob.use_mm_prefix)
    {
        if(prob.softcap > 0)
        {
            if(prob.use_qq_bias) UA_RUN_SINGLE(true, true, true);
            else UA_RUN_SINGLE(true, false, true);
        }
        else if(prob.use_qq_bias) UA_RUN_SINGLE(false, true, true);
        else UA_RUN_SINGLE(false, false, true);
    }
    else if(prob.softcap > 0)
    {
        if(prob.use_qq_bias) UA_RUN_SINGLE(true, true, false);
        else UA_RUN_SINGLE(true, false, false);
    }
    else if(prob.use_qq_bias) UA_RUN_SINGLE(false, true, false);
    else UA_RUN_SINGLE(false, false, false);
#undef UA_RUN_SINGLE

    HIP_CHECK_ERROR(hipGetLastError());
    bool pass = true;
    if(parser.get_bool("v"))
    {
        const std::size_t count = static_cast<std::size_t>(n_q) * prob.h * d;
        ck_tile::DeviceMem ref_out_buf(count * sizeof(uint16_t));
        auto reference_arg = data.make_reference_arg(
            static_cast<uint16_t*>(ref_out_buf.GetDeviceBuffer()));
        ck_tile::reference_unified_attention<DataType>(reference_arg);
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        std::vector<DataType> actual(
            static_cast<std::size_t>(prob.total_nq_pad) * prob.h * d);
        std::vector<DataType> reference(count);
        data.out_buf.FromDevice(actual.data());
        ref_out_buf.FromDevice(reference.data());
        actual.resize(count);
        pass = ck_tile::check_err(
            actual, reference, "Unified attention reference mismatch", 2.0e-2f, 2.0e-2f);
    }
    const char* type_name = std::is_same_v<DataType, ck_tile::bf16_t> ? "bf16" : "fp16";
    std::cout << "Unified Attention 2D Config: path=2d, prec=" << type_name
              << ", NQ=" << n_q << ", NKV=" << n_kv << ", H=" << prob.h
              << ", HKV=" << prob.hkv << ", D=" << d
              << ", kv_stages=" << kv_stages << '\n';
    std::cout << "unified_attention_2d Avg Latency: " << ms << " ms\n";
    std::cout << "Verification (unified_attention_2d): "
              << (parser.get_bool("v") ? (pass ? "PASSED" : "FAILED") : "SKIPPED")
              << '\n';
    return pass ? 0 : 1;
}

template <typename DataType>
int run_unified_attention_2d_d192(const ck_tile::ArgParser& parser,
                                  const std::vector<int>& query_lens = {})
{
    const int d = parser.get_int("d");
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

    const auto prob = ck_tile::UnifiedAttentionProblem::create(parser, query_lens, false);
    const bool is_batched = prob.batch > 1;
    const bool use_batched_swa_direct =
        is_batched && use_sliding && softcap <= 0.0f && !use_alibi &&
        !use_qq_bias && !use_mm_prefix;
    const bool has_runtime_features =
        is_batched && (use_sinks || use_alibi || use_qq_bias || use_mm_prefix ||
                       softcap > 0.0f || use_sliding);

    if(prob.batch <= 0 || prob.max_nkv <= 0 || prob.h <= 0 || prob.hkv <= 0 || prob.h % prob.hkv != 0 ||
       !parser.get_bool("causal") ||
       (mask != "causal" && mask != "sliding" && mask != "prefix" &&
        mask != "sliding_prefix") ||
       (use_sliding && sliding_window <= 0) || (!use_sliding && sliding_window != 0) ||
       softcap < 0.0f || (alibi_sqrt && !use_alibi))
    {
        std::cerr << "invalid D192 configuration\n";
        return 2;
    }

    if(prob.page_size <= 0 || prob.block_table_width < prob.pages_needed_per_seq ||
       prob.kv_cache_blocks < prob.required_pages)
    {
        std::cerr << "invalid KV-cache capacity or block-table width\n";
        return 2;
    }

    if(prob.batch == 1)
        return run_unified_attention_2d_d192_single<DataType>(parser, prob);

    ck_tile::UnifiedAttentionData<DataType> data(prob, 11001, false);

    const int warmup = parser.get_int("warmup");
    const int repeat = parser.get_int("repeat");
    ck_tile::stream_config stream_config{nullptr, true, 1, warmup, repeat};
    float ms = 0.0f;

    auto blocks_for = [](int64_t work_items) {
        constexpr int block = 256;
        return static_cast<unsigned int>(std::min<int64_t>(
            4096, (work_items + block - 1) / block));
    };
    auto pack_q_fn = [&](const ck_tile::stream_config& s) {
        const int64_t q_elements =
            static_cast<int64_t>(prob.h) * prob.total_nq_pad * d;
        const UnifiedAttentionBatchedQPackArgs args{
            reinterpret_cast<const uint16_t*>(data.q_buf.GetDeviceBuffer()),
            reinterpret_cast<uint16_t*>(data.q_pack_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.cu_q_buf.GetDeviceBuffer()),
            prob.batch,
            prob.h,
            prob.total_nq_pad,
            d,
            d,
            static_cast<int64_t>(prob.h) * d,
            d,
            blocks_for(q_elements)};
        launch_unified_attention_pack<UnifiedAttentionBatchedQPackKernel>(
            args, s.stream_id_);
    };
    auto pack_kv_fn = [&](const ck_tile::stream_config& s) {
        const int64_t kv_rows = use_batched_swa_direct
                                    ? prob.packed_kv_rows
                                    : static_cast<int64_t>(prob.batch) * prob.hkv *
                                          (prob.nkv_pad - prob.kv_pack_start);
        const UnifiedAttentionBatchedKvPackArgs args{
            reinterpret_cast<const uint16_t*>(data.k_buf.GetDeviceBuffer()),
            reinterpret_cast<const uint16_t*>(data.v_buf.GetDeviceBuffer()),
            reinterpret_cast<uint16_t*>(data.k_pack_buf.GetDeviceBuffer()),
            reinterpret_cast<uint16_t*>(data.v_pack_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.cu_q_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.seqlens_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.block_table_buf.GetDeviceBuffer()),
            prob.batch,
            prob.nkv_pad,
            prob.kv_pack_start,
            prob.hkv,
            prob.sliding_window,
            prob.use_mm_prefix,
            d,
            d,
            prob.page_size,
            static_cast<int64_t>(prob.block_table_width),
            static_cast<int64_t>(prob.page_size) * prob.hkv * d,
            static_cast<int64_t>(prob.hkv) * d,
            d,
            static_cast<int64_t>(prob.page_size) * prob.hkv * d,
            static_cast<int64_t>(prob.hkv) * d,
            d,
            blocks_for(kv_rows * d / ck_tile::kUnifiedAttentionPackVectorSize)};
        launch_unified_attention_pack<UnifiedAttentionBatchedKvPackKernel>(
            args, s.stream_id_);
    };

    if(use_batched_swa_direct)
    {
        using Problem = UnifiedAttentionMmacProblemT<DataType>;
        using Policy = unified_attention::UnifiedAttentionMmacDefaultPolicy<Problem, 3>;
        using Kernel = UnifiedAttentionSwaBatchedKernel<Policy, true>;
        UnifiedAttentionSwaBatchedKernelArgument kargs{
            reinterpret_cast<const uint16_t*>(data.q_buf.GetDeviceBuffer()),
            reinterpret_cast<const uint16_t*>(data.k_pack_buf.GetDeviceBuffer()),
            reinterpret_cast<const uint16_t*>(data.v_pack_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.cu_q_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.seqlens_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.indices_buf.GetDeviceBuffer()),
            static_cast<const int32_t*>(data.counts_buf.GetDeviceBuffer()),
            data.sinks_ptr,
            reinterpret_cast<uint16_t*>(data.out_buf.GetDeviceBuffer()),
            static_cast<float*>(data.swa_lse_buf.GetDeviceBuffer()),
            prob.batch,
            prob.h,
            prob.hkv,
            prob.total_nq_pad,
            prob.total_nqb,
            prob.max_nq_pad,
            prob.max_nqb,
            prob.nkv_pad,
            d,
            d,
            prob.nb,
            prob.sliding_window,
            ck_tile::kQkScaleLog2e(d),
            static_cast<int64_t>(prob.h) * d,
            d,
            static_cast<int64_t>(prob.h) * d,
            d,
            1};
        constexpr auto block = Kernel::BlockSize();
        auto attn_fn = ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
            Kernel{}, Kernel::GridSize(kargs), block, 0, kargs);
        ms = ck_tile::launch_kernel(stream_config, pack_kv_fn, attn_fn);
    }
    else
    {
        using Problem = UnifiedAttentionMmacProblemT<DataType>;
        using Policy = unified_attention::UnifiedAttentionMmacDefaultPolicy<Problem, 3>;

        auto reduce_fn = [&](const ck_tile::stream_config& s) {
            hipStream_t stream = s.stream_id_;
            using ReduceKernel = UnifiedAttentionKvStageReduceKernel<Policy, 5>;
            const auto reduce_args = ReduceKernel::MakeKargs(
                reinterpret_cast<const uint16_t*>(data.partial_out_buf.GetDeviceBuffer()),
                static_cast<const float*>(data.lse_buf.GetDeviceBuffer()),
                // Reduction treats packed queries as one sequence. Its valid
                // row count is total_nq, not the first sequence's KV length.
                static_cast<const int32_t*>(data.cu_q_buf.GetDeviceBuffer()) + prob.batch,
                reinterpret_cast<uint16_t*>(data.out_buf.GetDeviceBuffer()),
                static_cast<float*>(data.lse_buf.GetDeviceBuffer()) +
                    static_cast<int64_t>(prob.kv_stages) * prob.h * prob.total_nq,
                1, prob.h, prob.total_nq, d, prob.kv_stages,
                static_cast<int64_t>(prob.h) * prob.total_nq * d,
                static_cast<int64_t>(prob.h) * prob.total_nq, 0,
                d, static_cast<int64_t>(prob.h) * d,
                1, prob.total_nq, 1);
            UnifiedAttentionKvStageReduceKernelInvoker<Policy>::Run(reduce_args, stream);
        };

        if(has_runtime_features)
        {
            using Problem = UnifiedAttentionMmacProblemT<DataType>;
            using Policy = unified_attention::UnifiedAttentionMmacDefaultPolicy<Problem, 3>;
            using Kernel = UnifiedAttentionDenseBatchedKernel<Policy, true>;
            UnifiedAttentionDenseBatchedFeatureKernelArgument kargs{
                reinterpret_cast<const uint16_t*>(data.q_pack_buf.GetDeviceBuffer()),
                reinterpret_cast<const uint16_t*>(data.k_pack_buf.GetDeviceBuffer()),
                reinterpret_cast<const uint16_t*>(data.v_pack_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(data.cu_q_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(data.seqlens_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(data.indices_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(data.counts_buf.GetDeviceBuffer()),
                data.sinks_ptr,
                data.alibi_ptr,
                data.qq_bias_ptr,
                data.mm_ranges_ptr,
                reinterpret_cast<uint16_t*>(data.partial_out_buf.GetDeviceBuffer()),
                static_cast<float*>(data.lse_buf.GetDeviceBuffer()),
                prob.batch,
                prob.h,
                prob.hkv,
                prob.total_nq,
                prob.total_nq_pad,
                prob.total_nqb,
                prob.max_nq_pad,
                prob.max_nqb,
                prob.nkv_pad,
                d,
                d,
                prob.nb,
                prob.kv_stages,
                prob.sliding_window,
                prob.use_mm_prefix ? 4 : 0,
                prob.max_nq_pad,
                ck_tile::kQkScaleLog2e(d),
                prob.softcap,
                prob.alibi_sqrt,
                static_cast<int64_t>(prob.h) * d,
                d};
            constexpr auto block = Kernel::BlockSize();
            auto attn_fn = ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                Kernel{}, Kernel::GridSize(kargs), block, 0, kargs);
            ms = ck_tile::launch_kernel(
                stream_config, pack_q_fn, pack_kv_fn, attn_fn, reduce_fn);
        }
        else
        {
            using Problem = UnifiedAttentionMmacProblemT<DataType>;
            using Policy = unified_attention::UnifiedAttentionMmacDefaultPolicy<Problem, 3>;
            using Kernel = UnifiedAttentionDenseBatchedKernel<Policy, false>;
            UnifiedAttentionDenseBatchedKernelArgument kargs{
                reinterpret_cast<const uint16_t*>(data.q_buf.GetDeviceBuffer()),
                reinterpret_cast<const uint16_t*>(data.k_pack_buf.GetDeviceBuffer()),
                reinterpret_cast<const uint16_t*>(data.v_pack_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(data.cu_q_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(data.seqlens_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(data.indices_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(data.counts_buf.GetDeviceBuffer()),
                reinterpret_cast<uint16_t*>(data.partial_out_buf.GetDeviceBuffer()),
                static_cast<float*>(data.lse_buf.GetDeviceBuffer()),
                prob.batch,
                prob.h,
                prob.hkv,
                prob.total_nq,
                prob.total_nq_pad,
                prob.total_nqb,
                prob.max_nq_pad,
                prob.max_nqb,
                prob.nkv_pad,
                d,
                d,
                prob.nb,
                prob.kv_stages,
                ck_tile::kQkScaleLog2e(d),
                static_cast<int64_t>(prob.h) * d,
                d};
            constexpr auto block = Kernel::BlockSize();
            auto attn_fn = ck_tile::make_kernel<block.x, Policy::kLaunchMinBlocks>(
                Kernel{}, Kernel::GridSize(kargs), block, 0, kargs);
            ms = ck_tile::launch_kernel(stream_config, pack_kv_fn, attn_fn, reduce_fn);
        }
    }
    HIP_CHECK_ERROR(hipGetLastError());
    bool pass = true;
    if(parser.get_bool("v"))
    {
        const std::size_t count = static_cast<std::size_t>(prob.total_nq) * prob.h * d;
        ck_tile::DeviceMem ref_out_buf(count * sizeof(uint16_t));
        auto reference_arg = data.make_reference_arg(static_cast<uint16_t*>(ref_out_buf.GetDeviceBuffer()));
        ck_tile::reference_unified_attention<DataType>(reference_arg);
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        std::vector<DataType> actual(
            static_cast<std::size_t>(prob.total_nq_pad) * prob.h * d);
        std::vector<DataType> reference(count);
        data.out_buf.FromDevice(actual.data());
        ref_out_buf.FromDevice(reference.data());
        actual.resize(count);
        pass = ck_tile::check_err(
            actual, reference, "Unified attention reference mismatch", 2.0e-2f, 2.0e-2f);
    }
    const char* type_name = std::is_same_v<DataType, ck_tile::bf16_t> ? "bf16" : "fp16";
    std::cout << "Unified Attention 2D Config: path=2d-batched, prec=" << type_name
              << ", B=" << prob.batch << ", NQ=" << prob.total_nq << ", NKV_MAX=" << prob.max_nkv
              << ", H=" << prob.h << ", HKV=" << prob.hkv << ", D=" << d
              << ", mask=" << mask << ", page_size=" << prob.page_size
              << ", kv_cache_blocks=" << prob.kv_cache_blocks
              << ", block_table_width=" << prob.block_table_width << '\n';
    std::cout << "unified_attention_2d Avg Latency: " << ms << " ms\n";
    std::cout << "Verification (unified_attention_2d): "
              << (parser.get_bool("v") ? (pass ? "PASSED" : "FAILED") : "SKIPPED")
              << '\n';
    return pass ? 0 : 1;
}
