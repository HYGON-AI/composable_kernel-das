// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host.hpp"
#include "ck_tile/ops/jenga.hpp"
#include "ck_tile/host/reference/reference_jenga.hpp"
#include "jenga_config.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>



template <typename PipelinePolicy>
float jenga_ck_tile_fwd_onehot(const uint16_t* q,
                              const uint16_t* k,
                              const uint16_t* v,
                              const int32_t* seqlens,
                              const int32_t* active_indices,
                              const int32_t* active_counts,
                              uint16_t* out,
                              float* lse,
                              int B,
                              int H,
                              int N,
                              int D,
                              int nblocks,
                              int active_capacity,
                              int kv_stage_count,
                              int64_t stride_os,
                              int64_t stride_ls,
                              float qk_scale,
                              const ck_tile::stream_config& stream_config)
{
    using Kernel = Jenga64FusedKernel<
        ck_tile::example::jenga::kHeadDim, false, false, PipelinePolicy>;
    const auto args = Kernel::MakeKargs(q, k, v, seqlens, nullptr,
        active_indices, active_counts, out, lse,
        H, N, N, D, D, B * H * nblocks, nblocks, nblocks, active_capacity, kv_stage_count,
        stride_os, stride_ls, 0.0f, qk_scale, nblocks,
        static_cast<int64_t>(N) * D, static_cast<int64_t>(N) * D,
        static_cast<int64_t>(N) * D, static_cast<int64_t>(H) * N * D,
        static_cast<int64_t>(N) * D, D, 1, N, 1);
    return ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<Kernel::BlockSize().x, PipelinePolicy::kLaunchMinBlocks>(
            Kernel{}, Kernel::GridSize(args), Kernel::BlockSize(), 0, args));
}

template <typename PipelinePolicy, int KvStageCount>
float jenga_ck_tile_fwd_reduce(const uint16_t* partial_o,
                               const float* partial_lse,
                               const int32_t* seqlens,
                               uint16_t* out,
                               float* lse,
                               int B,
                               int H,
                               int N,
                               int D,
                               int64_t partial_o_stage_stride,
                               int64_t partial_lse_stage_stride,
                               const ck_tile::stream_config& stream_config)
{
    using Kernel = Jenga64KvStageReduceKernel<PipelinePolicy, KvStageCount>;
    const auto args = Kernel::MakeKargs(partial_o, partial_lse, seqlens, out, lse,
        B, H, N, D, KvStageCount, partial_o_stage_stride, partial_lse_stage_stride,
        static_cast<int64_t>(H) * N * D, static_cast<int64_t>(N) * D,
        D, 1, N, 1);
    constexpr auto block = Kernel::BlockSize();
    return ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<block.x, PipelinePolicy::kLaunchMinBlocks>(
            Kernel{}, Kernel::GridSize(args), block, 0, args));
}

template <typename PipelinePolicy>
float jenga_ck_tile_fwd_reduce_dispatch(const uint16_t* partial_o,
                                        const float* partial_lse,
                                        const int32_t* seqlens,
                                        uint16_t* out,
                                        float* lse,
                                        int B,
                                        int H,
                                        int N,
                                        int D,
                                        int kv_stage_count,
                                        int64_t partial_o_stage_stride,
                                        int64_t partial_lse_stage_stride,
                                        const ck_tile::stream_config& stream_config)
{
    switch(kv_stage_count)
    {
    case 2: return jenga_ck_tile_fwd_reduce<PipelinePolicy, 2>(partial_o, partial_lse,
        seqlens, out, lse, B, H, N, D, partial_o_stage_stride,
        partial_lse_stage_stride, stream_config);
    case 3: return jenga_ck_tile_fwd_reduce<PipelinePolicy, 3>(partial_o, partial_lse,
        seqlens, out, lse, B, H, N, D, partial_o_stage_stride,
        partial_lse_stage_stride, stream_config);
    case 4: return jenga_ck_tile_fwd_reduce<PipelinePolicy, 4>(partial_o, partial_lse,
        seqlens, out, lse, B, H, N, D, partial_o_stage_stride,
        partial_lse_stage_stride, stream_config);
    default: return jenga_ck_tile_fwd_reduce<PipelinePolicy, PipelinePolicy::kKvStageCount>(
        partial_o, partial_lse, seqlens, out, lse, B, H, N, D,
        partial_o_stage_stride, partial_lse_stage_stride, stream_config);
    }
}

template <typename DataType, typename PipelinePolicy>
int run_fwd(const ck_tile::ArgParser& arg_parser)
{
    const int B = arg_parser.get_int("b");
    const int H = arg_parser.get_int("h");
    const int N = arg_parser.get_int("s");
    const int D = arg_parser.get_int("d");
    const int top_k = arg_parser.get_int("topk");
    if(B <= 0 || H <= 0 || N <= 0 ||
       N % ck_tile::example::jenga::kBlockM != 0 ||
       D != ck_tile::example::jenga::kHeadDim)
    {
        std::cerr << "jenga_fwd requires b,h > 0, s aligned to BlockM, "
                     "and the supported head dimension\n";
        return 2;
    }

    const int BH = B * H;
    const int nblocks = N / ck_tile::example::jenga::kBlockM;
    const int text_start_block_arg = arg_parser.get_int("text_start_block");
    const int text_start_block = text_start_block_arg < 0 ? nblocks : text_start_block_arg;
    const int text_blocks = arg_parser.get_int("text_blocks");
    const int first_frame_blocks = arg_parser.get_int("first_frame_blocks");
    const float prob_threshold = arg_parser.get_float("prob_threshold");
    const bool store_lse = arg_parser.get_bool("store_lse");
    const int requested_kv_stages = arg_parser.get_int("kv_stages");
    if(top_k <= 0 || top_k > text_start_block || prob_threshold < 0.0f ||
       prob_threshold > 1.0f || text_start_block < 0 ||
       text_start_block > nblocks || text_blocks < 0 ||
       text_start_block + text_blocks > nblocks || first_frame_blocks < 0 ||
       requested_kv_stages < 0 || requested_kv_stages > PipelinePolicy::kKvStageCount)
    {
        std::cerr << "invalid topk/text block configuration\n";
        return 2;
    }
    using MaskProblem = JengaMaskBuilderProblem<
        DataType,
        ck_tile::example::jenga::kBlockM,
        ck_tile::example::jenga::kBlockN,
        ck_tile::example::jenga::kHeadDim>;
    using MaskPolicy = JengaMaskBuilderPolicy<MaskProblem>;
    using MaskInvoker = JengaMaskBuilderInvoker<MaskPolicy>;
    const int candidate_k = MaskPolicy::GetCandidateK(top_k, text_start_block, prob_threshold);
    const int threshold_capacity = std::min(
        candidate_k,
        std::max(top_k,
                 static_cast<int>(std::floor(prob_threshold * text_start_block)) +
                     ck_tile::example::jenga::kSelectionCapacityMargin));
    const int active_capacity = std::min(
        nblocks, threshold_capacity + text_blocks + std::min(first_frame_blocks, nblocks));
    const int rows = BH * N;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    constexpr float kLog2e = ck_tile::log2e_v<float>;
    const uint32_t seed = arg_parser.get_uint32("seed");
    ck_tile::HostTensor<DataType> q_host({BH, N, D});
    ck_tile::HostTensor<DataType> k_host({BH, N, D});
    ck_tile::HostTensor<DataType> v_host({BH, N, D});
    ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed, true}(q_host);
    ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed + 1, true}(k_host);
    ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, seed + 2, true}(v_host);
    ck_tile::HostTensor<DataType> q_scaled_host = q_host;
    for(auto& x : q_scaled_host)
        x = ck_tile::type_convert<DataType>(ck_tile::type_convert<float>(x) * scale * kLog2e);

    ck_tile::HostTensor<int32_t> seqlens_host({B});
    std::fill(seqlens_host.begin(), seqlens_host.end(), N);

    ck_tile::DeviceMem q_raw_buf(q_host), q_buf(q_scaled_host), k_buf(k_host), v_buf(v_host);
    ck_tile::DeviceMem seqlens_buf(seqlens_host);
    ck_tile::DeviceMem onehot_buf(static_cast<size_t>(BH) * nblocks * nblocks * sizeof(bool));
    ck_tile::DeviceMem q_pool_buf(static_cast<size_t>(BH) * nblocks * D * sizeof(float));
    ck_tile::DeviceMem k_pool_buf(static_cast<size_t>(BH) * nblocks * D * sizeof(float));
    ck_tile::DeviceMem scores_buf(static_cast<size_t>(BH) * nblocks * nblocks * sizeof(float));
    ck_tile::DeviceMem counts_buf(static_cast<size_t>(BH) * nblocks * sizeof(int32_t));
    ck_tile::DeviceMem indices_buf(
        static_cast<size_t>(BH) * nblocks * active_capacity * sizeof(int32_t));
    ck_tile::DeviceMem out_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_buf(static_cast<size_t>(rows) * sizeof(float));
    auto* q = static_cast<const uint16_t*>(q_buf.GetDeviceBuffer());
    auto* k = static_cast<const uint16_t*>(k_buf.GetDeviceBuffer());
    auto* v = static_cast<const uint16_t*>(v_buf.GetDeviceBuffer());
    auto* seqlens = static_cast<const int32_t*>(seqlens_buf.GetDeviceBuffer());
    auto* counts = static_cast<const int32_t*>(counts_buf.GetDeviceBuffer());
    auto* indices = static_cast<const int32_t*>(indices_buf.GetDeviceBuffer());
    auto* out = static_cast<uint16_t*>(out_buf.GetDeviceBuffer());
    auto* lse = static_cast<float*>(lse_buf.GetDeviceBuffer());

    onehot_buf.SetZero();
    JengaMaskBuilderArgument mask_args{};
    mask_args.query = static_cast<const uint16_t*>(q_raw_buf.GetDeviceBuffer());
    mask_args.key = k;
    mask_args.neighbor_mask = nullptr;
    mask_args.out = static_cast<bool*>(onehot_buf.GetDeviceBuffer());
    mask_args.B = B; mask_args.H = H; mask_args.N_Q = N; mask_args.N_K = N; mask_args.D = D;
    mask_args.num_query_blocks = nblocks; mask_args.num_blocks = nblocks;
    mask_args.text_start_block = text_start_block; mask_args.top_k = top_k;
    mask_args.prob_threshold = prob_threshold; mask_args.text_blocks = text_blocks;
    mask_args.first_frame_blocks = first_frame_blocks;
    auto* q_pool = static_cast<float*>(q_pool_buf.GetDeviceBuffer());
    auto* k_pool = static_cast<float*>(k_pool_buf.GetDeviceBuffer());
    auto* scores = static_cast<float*>(scores_buf.GetDeviceBuffer());
    MaskInvoker::RunPoolQ(mask_args, q_pool, k_pool, nullptr);
    MaskInvoker::RunPoolK(mask_args, q_pool, k_pool, nullptr);
    MaskInvoker::RunScore(mask_args, q_pool, k_pool, scores, nullptr);
    MaskInvoker::RunSelect(mask_args, scores, candidate_k, nullptr);
    jenga_onehot_to_lut(mask_args.out,
                        const_cast<int32_t*>(indices),
                        const_cast<int32_t*>(counts),
                        BH * nblocks,
                        nblocks,
                        active_capacity);
    HIP_CHECK_ERROR(hipGetLastError());
    // Mask construction and onehot-to-LUT preprocessing are intentionally excluded from fwd timing.
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<int32_t> selected_counts(static_cast<size_t>(BH) * nblocks);
    counts_buf.FromDevice(selected_counts.data());
    int64_t selected_sum = 0;
    int32_t selected_max = 0;
    for(const int32_t count : selected_counts)
    {
        selected_sum += count;
        selected_max = std::max(selected_max, count);
    }
    const double selected_avg =
        static_cast<double>(selected_sum) / static_cast<double>(selected_counts.size());

    constexpr int target_active_blocks_per_stage =
        ck_tile::example::jenga::kTargetActiveBlocksPerStage;
    const int auto_kv_stage_count =
        N >= ck_tile::example::jenga::kLargeSequenceThreshold
        ? std::min(PipelinePolicy::kKvStageCount,
                   std::max(
                       ck_tile::example::jenga::kMinSplitStageCount,
                       (selected_max + target_active_blocks_per_stage - 1) /
                           target_active_blocks_per_stage))
        : ck_tile::example::jenga::kDefaultStageCount;
    const int runtime_kv_stage_count =
        requested_kv_stages == 0 ? auto_kv_stage_count : requested_kv_stages;
    const int64_t partial_o_stage_stride = static_cast<int64_t>(BH) * N * D;
    const int64_t partial_lse_stage_stride = static_cast<int64_t>(BH) * N;
    ck_tile::DeviceMem partial_out_buf(
        runtime_kv_stage_count > 1
            ? static_cast<size_t>(runtime_kv_stage_count) * q_host.get_element_space_size_in_bytes()
            : 1);
    ck_tile::DeviceMem partial_lse_buf(
        runtime_kv_stage_count > 1
            ? static_cast<size_t>(runtime_kv_stage_count) * rows * sizeof(float)
            : 1);
    auto* fused_out = runtime_kv_stage_count > 1
        ? static_cast<uint16_t*>(partial_out_buf.GetDeviceBuffer()) : out;
    auto* fused_lse = runtime_kv_stage_count > 1
        ? static_cast<float*>(partial_lse_buf.GetDeviceBuffer()) : lse;
    const char* type_name = std::is_same_v<DataType, ck_tile::fp16_t> ? "fp16" : "bf16";
    std::cout << "Jenga FWD Config: B=" << B << ", H=" << H
              << ", S=" << N << ", D=" << D
              << ", prec=" << type_name << ", topk=" << top_k
              << ", selected_k_blocks_avg=" << selected_avg
              << ", kv_stages=" << runtime_kv_stage_count << "\n";
    std::cout << "Benchmarking fwd pipeline (jenga_fwd)..." << std::endl;

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         0,
                                         arg_parser.get_int("warmup"),
                                         arg_parser.get_int("repeat")};
    const float fused_ms = jenga_ck_tile_fwd_onehot<PipelinePolicy>(
        q, k, v, seqlens, indices, counts, fused_out, fused_lse,
        B, H, N, D, nblocks, active_capacity, runtime_kv_stage_count,
        runtime_kv_stage_count > 1 ? partial_o_stage_stride : 0,
        runtime_kv_stage_count > 1 ? partial_lse_stage_stride : 0,
        1.0f,
        stream_config);
    float reduce_ms = 0.0f;
    if(runtime_kv_stage_count > 1)
    {
        reduce_ms = jenga_ck_tile_fwd_reduce_dispatch<PipelinePolicy>(
            fused_out, fused_lse, seqlens, out, lse, B, H, N, D,
            runtime_kv_stage_count, partial_o_stage_stride, partial_lse_stage_stride,
            stream_config);
    }
    const float fwd_ms = fused_ms + reduce_ms;
    HIP_CHECK_ERROR(hipGetLastError());
    std::cout << "jenga_fwd Avg Latency: " << fwd_ms << " ms\n";

    bool valid = true;
    if(arg_parser.get_bool("v"))
    {
        ck_tile::DeviceMem ref_out_buf(q_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem ref_lse_buf(static_cast<size_t>(rows) * sizeof(float));
        auto* ref_out = static_cast<uint16_t*>(ref_out_buf.GetDeviceBuffer());
        auto* ref_lse = static_cast<float*>(ref_lse_buf.GetDeviceBuffer());
        hipLaunchKernelGGL((jenga_reference::fwd<DataType>),
            dim3(rows),
            dim3(jenga_reference::kReferenceBlockSize),
            0,
            0,
            q,
            k,
            v,
            indices,
            counts,
            ref_out,
            ref_lse,
            rows,
            N,
            D,
            ck_tile::example::jenga::kBlockN,
            active_capacity,
            1.0f);
        HIP_CHECK_ERROR(hipGetLastError());
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        ck_tile::HostTensor<DataType> got_o({BH, N, D}), ref_o({BH, N, D});
        ck_tile::HostTensor<float> got_lse({BH, N}), ref_lse_h({BH, N});
        out_buf.FromDevice(got_o.data());
        ref_out_buf.FromDevice(ref_o.data());
        valid &= ck_tile::check_err(
            got_o, ref_o, std::string("OUT Error: Incorrect results!"), 0.03, 0.03);
        if(store_lse)
        {
            lse_buf.FromDevice(got_lse.data());
            ref_lse_buf.FromDevice(ref_lse_h.data());
            valid &= ck_tile::check_err(
                got_lse, ref_lse_h, std::string("LSE Error: Incorrect results!"), 0.01, 0.03);
        }
        std::cout << "Verification (jenga_fwd): " << (valid ? "PASSED" : "FAILED") << "\n";
    }
    else
    {
        std::cout << "Verification (jenga_fwd): SKIPPED\n";
    }
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    return valid ? 0 : 4;
}
