// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host.hpp"
#include "ck_tile/ops/sla.hpp"
#include "sla_sparse_map.hpp"
#include "sla_config.hpp"


namespace sla {

template <ActType Type, typename DataType>
static void launch_fused_linear(
    const DataType* q_ptr,
    const DataType* k_ptr,
    const DataType* v_ptr,
    const DataType* w_ptr,
    const DataType* b_ptr,
    const DataType* os_ptr,
    DataType* out_ptr,
    uint8_t* workspace_ptr,
    int B, int H, int L, int D,
    hipStream_t stream = nullptr)
{
    const int BH = B * H;
    using LinearPolicy = SlaLinearFusedPolicy<SlaFusedLinearAttnProblem<DataType>>;
    const auto align_up = [](std::size_t bytes) {
        constexpr auto alignment = ck_tile::example::sla::kWorkspaceAlignment;
        return (bytes + alignment - 1) & ~(alignment - 1);
    };
    constexpr bool kUseSplitL = Type == ActType::RELU;
    constexpr int kSplitL = LinearPolicy::kKvSplitL;
    const std::size_t k_feat_bytes =
        static_cast<std::size_t>(BH) * L * D * sizeof(DataType);
    const std::size_t matrix_bytes =
        static_cast<std::size_t>(BH) * D * D * sizeof(DataType);
    const std::size_t partial_kv_bytes =
        static_cast<std::size_t>(BH) * kSplitL * D * D * sizeof(float);
    const std::size_t partial_ksum_bytes =
        static_cast<std::size_t>(BH) * kSplitL * D * sizeof(float);
    const std::size_t ksum_bytes =
        static_cast<std::size_t>(BH) * D * sizeof(float);

    const std::size_t stage0_bytes = kUseSplitL ? partial_kv_bytes : k_feat_bytes;
    const std::size_t stage1_bytes = kUseSplitL ? partial_ksum_bytes : 0;
    const std::size_t stage0_offset = 0;
    const std::size_t stage1_offset = align_up(stage0_offset + stage0_bytes);
    const std::size_t kv_offset = align_up(stage1_offset + stage1_bytes);
    const std::size_t projected_offset = align_up(kv_offset + matrix_bytes);
    const std::size_t ksum_offset = align_up(projected_offset + matrix_bytes);

    auto* stage0_ptr = workspace_ptr + stage0_offset;
    auto* stage1_ptr = workspace_ptr + stage1_offset;
    auto* kv_ptr = reinterpret_cast<DataType*>(workspace_ptr + kv_offset);
    auto* projected_ptr = reinterpret_cast<DataType*>(workspace_ptr + projected_offset);
    auto* ksum_ptr = reinterpret_cast<float*>(workspace_ptr + ksum_offset);

    if constexpr(kUseSplitL)
    {
        auto* partial_kv_ptr = reinterpret_cast<float*>(stage0_ptr);
        auto* partial_ksum_ptr = reinterpret_cast<float*>(stage1_ptr);
        using SplitKernel = SlaLinearKSplitKernel<Type, kSplitL, LinearPolicy>;
        typename SplitKernel::Kargs split_args{
            k_ptr, v_ptr, partial_kv_ptr, partial_ksum_ptr, L, BH};
        auto split_callable = ck_tile::make_kernel<LinearPolicy::kBlockSize, 1>(
            SplitKernel{}, SplitKernel::GridSize(split_args), SplitKernel::BlockSize(), 0, split_args);
        split_callable(ck_tile::stream_config{stream, false});

        using ReduceKernel = SlaLinearKSplitReduceKernel<DataType, kSplitL>;
        typename ReduceKernel::Kargs reduce_args{
            partial_kv_ptr, partial_ksum_ptr, kv_ptr, ksum_ptr, BH};
        auto reduce_callable = ck_tile::make_kernel<LinearPolicy::kBlockSize, 1>(
            ReduceKernel{}, ReduceKernel::GridSize(reduce_args), ReduceKernel::BlockSize(), 0, reduce_args);
        reduce_callable(ck_tile::stream_config{stream, false});
    }
    else
    {
        auto* k_feat_ptr = reinterpret_cast<DataType*>(stage0_ptr);
        (void)hipMemsetAsync(ksum_ptr, 0, ksum_bytes, stream);

        using FeatureKernel = SlaLinearKFeatureKernel<Type, LinearPolicy>;
        typename FeatureKernel::Kargs feature_args{k_ptr, k_feat_ptr, ksum_ptr, L, BH};
        auto feature_callable = ck_tile::make_kernel<LinearPolicy::kBlockSize, 1>(
            FeatureKernel{}, FeatureKernel::GridSize(feature_args), FeatureKernel::BlockSize(), 0, feature_args);
        feature_callable(ck_tile::stream_config{stream, false});

        using KvKernel = SlaLinearKvGemmKernel<DataType>;
        typename KvKernel::Kargs kv_args{k_feat_ptr, v_ptr, kv_ptr, L, BH};
        auto kv_callable = ck_tile::make_kernel<LinearPolicy::kBlockSize, 1>(
            KvKernel{}, KvKernel::GridSize(kv_args), KvKernel::BlockSize(), 0, kv_args);
        kv_callable(ck_tile::stream_config{stream, false});
    }

    using ProjectionKernel = SlaLinearKvProjectionKernel<DataType>;
    typename ProjectionKernel::Kargs projection_args{kv_ptr, w_ptr, projected_ptr, BH};
    auto projection_callable = ck_tile::make_kernel<LinearPolicy::kBlockSize, 1>(
        ProjectionKernel{}, ProjectionKernel::GridSize(projection_args), ProjectionKernel::BlockSize(), 0, projection_args);
    projection_callable(ck_tile::stream_config{stream, false});

    using QPolicy = std::conditional_t<
        Type == ActType::SOFTMAX,
        LinearPolicy,
        SlaLinearFusedPolicy<SlaFusedLinearAttnProblem<DataType>, 32>>;
    using QKernel = SlaLinearQFusedKernel<Type, QPolicy>;
    typename QKernel::Kargs qargs{q_ptr, projected_ptr, ksum_ptr, b_ptr, os_ptr, out_ptr, L, BH};
    auto q_callable = ck_tile::make_kernel<LinearPolicy::kBlockSize, 1>(
        QKernel{}, QKernel::GridSize(qargs), QKernel::BlockSize(), 0, qargs);
    q_callable(ck_tile::stream_config{stream, false});
}

template <typename DataType>
inline std::size_t get_fused_linear_workspace_bytes(int B, int H, int L, int D, ActType type)
{
    const std::size_t BH = static_cast<std::size_t>(B) * H;
    using LinearPolicy = SlaLinearFusedPolicy<SlaFusedLinearAttnProblem<DataType>>;
    const auto align_up = [](std::size_t bytes) {
        constexpr auto alignment = ck_tile::example::sla::kWorkspaceAlignment;
        return (bytes + alignment - 1) & ~(alignment - 1);
    };
    const bool kUseSplitL = type == ActType::RELU;
    constexpr int kSplitL = LinearPolicy::kKvSplitL;
    const std::size_t k_feat_bytes = static_cast<std::size_t>(BH) * L * D * sizeof(DataType);
    const std::size_t matrix_bytes = static_cast<std::size_t>(BH) * D * D * sizeof(DataType);
    const std::size_t partial_kv_bytes = static_cast<std::size_t>(BH) * kSplitL * D * D * sizeof(float);
    const std::size_t partial_ksum_bytes = static_cast<std::size_t>(BH) * kSplitL * D * sizeof(float);
    const std::size_t ksum_bytes = static_cast<std::size_t>(BH) * D * sizeof(float);

    const std::size_t stage0_bytes = kUseSplitL ? partial_kv_bytes : k_feat_bytes;
    const std::size_t stage1_bytes = kUseSplitL ? partial_ksum_bytes : 0;
    const std::size_t stage0_offset = 0;
    const std::size_t stage1_offset = align_up(stage0_offset + stage0_bytes);
    const std::size_t kv_offset = align_up(stage1_offset + stage1_bytes);
    const std::size_t projected_offset = align_up(kv_offset + matrix_bytes);
    const std::size_t ksum_offset = align_up(projected_offset + matrix_bytes);
    return align_up(ksum_offset + ksum_bytes);
}

template <typename DataType>
inline void launch_fused_linear(
    const DataType* q_ptr,
    const DataType* k_ptr,
    const DataType* v_ptr,
    const DataType* gate_ptr,
    DataType* out_ptr,
    int B, int H, int L, int D,
    hipStream_t stream = nullptr)
{
    const std::size_t ws_bytes =
        get_fused_linear_workspace_bytes<DataType>(B, H, L, D, ActType::RELU);
    ck_tile::DeviceMem ws_buf(ws_bytes);
    launch_fused_linear<ActType::RELU, DataType>(
        q_ptr, k_ptr, v_ptr, gate_ptr, nullptr, nullptr, out_ptr,
        static_cast<uint8_t*>(ws_buf.GetDeviceBuffer()),
        B, H, L, D, stream);
}

} // namespace sla

namespace ck_tile {
namespace example {
namespace sla {

using ::sla::launch_fused_linear;
using ::sla::get_fused_linear_workspace_bytes;

} // namespace sla
} // namespace example
} // namespace ck_tile
#include "ck_tile/host/reference/reference_sla.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

// -------------------------------------------------------------
// SLA Forward Instance & Dispatcher Pattern
// -------------------------------------------------------------

template <typename DataType, int BlockM>
struct SlaFwdInstance
{
    static bool IsSupported(int s)
    {
        if constexpr(BlockM == ck_tile::example::sla::kLargeBlockM)
            return s == ck_tile::example::sla::kLargeBlockMSequenceLength;
        else
            return s != ck_tile::example::sla::kLargeBlockMSequenceLength;
    }

    static int Run(const ck_tile::ArgParser& arg_parser)
    {
        const int b = arg_parser.get_int("b");
        const int h = arg_parser.get_int("h");
        const int s = arg_parser.get_int("s");
        const int d = arg_parser.get_int("d");
        const float topk_ratio = arg_parser.get_float("topk_ratio");
        const int req_kv_stages = arg_parser.get_int("kv_stages");
        const int run_linear = arg_parser.get_int("linear_attn");
        const int warmup = arg_parser.get_int("warmup");
        const int repeat = arg_parser.get_int("repeat");
        const int verify = arg_parser.get_int("v");
        const uint32_t seed = arg_parser.get_uint32("seed");

        const int block_m = BlockM;
        const int auto_kv_stage_count =
            BlockM == ck_tile::example::sla::kDefaultBlockM
                ? ck_tile::example::sla::kDefaultFwdKvStageCount
                : ck_tile::example::sla::kDefaultStageCount;
        const int kv_stage_count = s == ck_tile::example::sla::kTunedSequenceLength
                                       ? ck_tile::example::sla::kTunedFwdKvStageCount
                                       : (req_kv_stages == 0 ? auto_kv_stage_count
                                                             : req_kv_stages);

        if(b <= 0 || h <= 0 || s <= 0 ||
           s % ck_tile::example::sla::kBlockN != 0 ||
           d != ck_tile::example::sla::kHeadDim ||
           topk_ratio <= 0.0f || topk_ratio > 1.0f ||
           s / ck_tile::example::sla::kBlockN > ck_tile::example::sla::kMaxKvBlocks ||
           req_kv_stages < 0 ||
           req_kv_stages > ck_tile::example::sla::kMaxKvStageCount ||
           kv_stage_count < 1 ||
           kv_stage_count > ck_tile::example::sla::kMaxKvStageCount ||
           (run_linear != 0 && run_linear != 1) || warmup <= 0 || repeat <= 0)
        {
            std::cerr << "unsupported SLA FWD arguments: require b,h>0, s>0 and divisible "
                         "by 64, d=128, KV blocks/topk <=2048, kv_stages in [0,16], "
                         "linear_attn in {0,1}, and warmup/repeat>0\n";
            return 1;
        }
        if(!ck_tile::example::sla::is_sla_shape_indexable(b, h, s, d))
        {
            std::cerr << "unsupported SLA shape: b*h*max(s,d)*d must fit in INT_MAX "
                         "for 32-bit kernel indexing\n";
            return 1;
        }
        if(s == ck_tile::example::sla::kTunedSequenceLength && req_kv_stages != 0 &&
           req_kv_stages != ck_tile::example::sla::kTunedFwdKvStageCount)
        {
            std::cout << "Requested kv_stages=" << req_kv_stages
                      << " is overridden; tuned optimal kv_stages for S=75648 is "
                      << ck_tile::example::sla::kTunedFwdKvStageCount << ".\n";
        }
        std::cout << "SLA FWD Config (Instance " << BlockM << "): B=" << b << ", H=" << h
                  << ", S=" << s << ", D=" << d << ", BlockM=" << block_m
                  << ", prec=" << arg_parser.get_str("prec") << ", topk_ratio=" << topk_ratio
                  << ", kv_stages=" << kv_stage_count << ", linear_attn=" << run_linear << "\n";

        const float sm_scale = 1.0f / std::sqrt(static_cast<float>(d));

        // Allocate host tensors
        ck_tile::HostTensor<DataType> q_host({b * h, s, d});
        ck_tile::HostTensor<DataType> k_host({b * h, s, d});
        ck_tile::HostTensor<DataType> v_host({b * h, s, d});
        ck_tile::HostTensor<DataType> weight_host({d, d});
        ck_tile::HostTensor<DataType> bias_host({d});

        ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed, true}(q_host);
        ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed + 1, true}(k_host);
        ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, seed + 2, true}(v_host);
        ck_tile::FillUniformDistribution<DataType>{-0.1f, 0.1f, seed + 3, true}(weight_host);
        ck_tile::FillUniformDistribution<DataType>{-0.1f, 0.1f, seed + 4, true}(bias_host);

        // Allocate device tensors
        ck_tile::DeviceMem q_buf(q_host);
        ck_tile::DeviceMem k_buf(k_host);
        ck_tile::DeviceMem v_buf(v_host);
        ck_tile::DeviceMem weight_buf(weight_host);
        ck_tile::DeviceMem bias_buf(bias_host);
        ck_tile::DeviceMem out_buf(static_cast<size_t>(b) * h * s * d * sizeof(DataType));
        ck_tile::DeviceMem lse_buf(static_cast<size_t>(b) * h * s * sizeof(float));

        // Allocate sparse_map metadata
        constexpr int block_n = ck_tile::example::sla::kBlockN;
        const int q_blocks = s / block_m;
        const int kv_blocks = s / block_n;
        const int max_nnz = std::max(1, static_cast<int>(topk_ratio * kv_blocks));

        ck_tile::DeviceMem sparse_map_buf(static_cast<size_t>(b) * h * q_blocks * kv_blocks * sizeof(int8_t));
        ck_tile::DeviceMem lut_buf(static_cast<size_t>(b) * h * q_blocks * max_nnz * sizeof(int64_t));
        ck_tile::DeviceMem lut_size_buf(static_cast<size_t>(b) * h * q_blocks * sizeof(int32_t));

        sparse_map_buf.SetZero();
        lut_buf.SetZero();
        lut_size_buf.SetZero();

        hipStream_t stream = nullptr;

        // Build sparse metadata outside the FWD timing region.
        ck_tile::example::sla::launch_sparse_map(
            static_cast<const DataType*>(q_buf.GetDeviceBuffer()),
            static_cast<const DataType*>(k_buf.GetDeviceBuffer()),
            static_cast<int8_t*>(sparse_map_buf.GetDeviceBuffer()),
            static_cast<int64_t*>(lut_buf.GetDeviceBuffer()),
            static_cast<int32_t*>(lut_size_buf.GetDeviceBuffer()),
            b, h, s, s, d,
            block_m, block_n, max_nnz,
            topk_ratio, sm_scale, stream);
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        // 2. Setup multi-stage KV structures if kv_stage_count > 1
        ck_tile::DeviceMem partitioned_lut_buf(kv_stage_count > 1 ? static_cast<size_t>(b) * h * q_blocks * max_nnz * sizeof(int64_t) : 1);
        ck_tile::DeviceMem stage_offsets_buf(kv_stage_count > 1 ? static_cast<size_t>(b) * h * q_blocks * (kv_stage_count + 1) * sizeof(int32_t) : 1);
        ck_tile::DeviceMem partial_out_buf(kv_stage_count > 1 ? static_cast<size_t>(kv_stage_count) * b * h * s * d * sizeof(DataType) : 1);
        ck_tile::DeviceMem partial_lse_buf(kv_stage_count > 1 ? static_cast<size_t>(kv_stage_count) * b * h * s * sizeof(float) : 1);

        const int64_t* kernel_lut = static_cast<const int64_t*>(lut_buf.GetDeviceBuffer());
        const int32_t* kernel_stage_offsets = nullptr;
        uint16_t* kernel_output = reinterpret_cast<uint16_t*>(out_buf.GetDeviceBuffer());
        float* kernel_lse = static_cast<float*>(lse_buf.GetDeviceBuffer());
        int64_t output_stage_stride = 0;
        int64_t lse_stage_stride = 0;

        if(kv_stage_count > 1)
        {
            kernel_lut = static_cast<const int64_t*>(partitioned_lut_buf.GetDeviceBuffer());
            kernel_stage_offsets = static_cast<const int32_t*>(stage_offsets_buf.GetDeviceBuffer());
            kernel_output = reinterpret_cast<uint16_t*>(partial_out_buf.GetDeviceBuffer());
            kernel_lse = static_cast<float*>(partial_lse_buf.GetDeviceBuffer());
            output_stage_stride = static_cast<int64_t>(b) * h * s * d;
            lse_stage_stride = static_cast<int64_t>(b) * h * s;
        }

        using Problem = SlaAttnFwdAttentionProblemM<BlockM, DataType>;
        using Policy = SlaAttnFwdDefaultPipelinePolicyT<Problem>;

        constexpr float kLog2e = ck_tile::log2e_v<float>;
        SlaAttnFwdKernelArgument fwd_arg{
            reinterpret_cast<const uint16_t*>(q_buf.GetDeviceBuffer()),
            reinterpret_cast<const uint16_t*>(k_buf.GetDeviceBuffer()),
            reinterpret_cast<const uint16_t*>(v_buf.GetDeviceBuffer()),
            kernel_lut,
            kernel_stage_offsets,
            kernel_output,
            kernel_lse,
            b, h, s, d, max_nnz, q_blocks,
            kv_stage_count,
            kv_stage_count + 1, // stage_offset_stride
            output_stage_stride,
            lse_stage_stride,
            sm_scale * kLog2e};

        auto run_fwd = [&]() {
            if(kv_stage_count == 4)
            {
                const SlaAttnFwdKvStagePartitionArgument partition_arg{
                    static_cast<const int64_t*>(lut_buf.GetDeviceBuffer()),
                    static_cast<int64_t*>(partitioned_lut_buf.GetDeviceBuffer()),
                    static_cast<int32_t*>(stage_offsets_buf.GetDeviceBuffer()),
                    b * h * q_blocks, max_nnz, kv_blocks, 4, 5};
                launch_sla_attn_fwd_kv_stage_partition<4>(partition_arg, stream);
                launch_sla_attn_fwd<Policy, ck_tile::example::sla::kHeadDim, false, 4>(fwd_arg, stream);
                const SlaAttnFwdKvStageReduceArgument reduce_arg{
                    reinterpret_cast<const uint16_t*>(partial_out_buf.GetDeviceBuffer()),
                    static_cast<const float*>(partial_lse_buf.GetDeviceBuffer()),
                    reinterpret_cast<uint16_t*>(out_buf.GetDeviceBuffer()),
                    static_cast<float*>(lse_buf.GetDeviceBuffer()),
                    b, h, s, d, 4, output_stage_stride, lse_stage_stride};
                launch_sla_attn_fwd_kv_stage_reduce<Policy, 4>(reduce_arg, stream);
            }
            else if(kv_stage_count == 1)
            {
                launch_sla_attn_fwd<Policy, ck_tile::example::sla::kHeadDim, false, 1>(fwd_arg, stream);
            }
            else
            {
                const SlaAttnFwdKvStagePartitionArgument partition_arg{
                    static_cast<const int64_t*>(lut_buf.GetDeviceBuffer()),
                    static_cast<int64_t*>(partitioned_lut_buf.GetDeviceBuffer()),
                    static_cast<int32_t*>(stage_offsets_buf.GetDeviceBuffer()),
                    b * h * q_blocks,
                    max_nnz,
                    kv_blocks,
                    kv_stage_count,
                    kv_stage_count + 1};
                launch_sla_attn_fwd_kv_stage_partition<
                    ck_tile::example::sla::kMaxKvStageCount>(partition_arg, stream);
                launch_sla_attn_fwd<Policy, ck_tile::example::sla::kHeadDim, false, 0>(fwd_arg, stream);
                const SlaAttnFwdKvStageReduceArgument reduce_arg{
                    reinterpret_cast<const uint16_t*>(partial_out_buf.GetDeviceBuffer()),
                    static_cast<const float*>(partial_lse_buf.GetDeviceBuffer()),
                    reinterpret_cast<uint16_t*>(out_buf.GetDeviceBuffer()),
                    static_cast<float*>(lse_buf.GetDeviceBuffer()),
                    b,
                    h,
                    s,
                    d,
                    kv_stage_count,
                    output_stage_stride,
                    lse_stage_stride};
                launch_sla_attn_fwd_kv_stage_reduce<Policy, 0>(reduce_arg, stream);
            }
        };

        std::cout << "Benchmarking fwd pipeline (sparse_attn)..." << std::endl;
        const ck_tile::stream_config benchmark_config{
            stream, true, 0, warmup, repeat, true};
        const float avg_time_ms = ck_tile::launch_kernel(
            benchmark_config, [&](const ck_tile::stream_config&) { run_fwd(); });
        std::cout << "sparse_attn Avg Latency: " << avg_time_ms << " ms\n";

        // core.py computes sparse attention first, then the feature-map / linear
        // branch, projection, and final o_s + o_l merge. Keep this outside the
        // sparse-attention timing, matching sparse-map generation above.
        ck_tile::DeviceMem final_out_buf(
            run_linear ? static_cast<size_t>(b) * h * s * d * sizeof(DataType) : 1);
        const auto linear_workspace_bytes = run_linear
            ? ck_tile::example::sla::get_fused_linear_workspace_bytes<DataType>(
                  b, h, s, d, sla::ActType::SOFTMAX)
            : 1;
        ck_tile::DeviceMem linear_workspace_buf(linear_workspace_bytes);
        if(run_linear)
        {
            ck_tile::example::sla::launch_fused_linear<sla::ActType::SOFTMAX, DataType>(
                static_cast<const DataType*>(q_buf.GetDeviceBuffer()),
                static_cast<const DataType*>(k_buf.GetDeviceBuffer()),
                static_cast<const DataType*>(v_buf.GetDeviceBuffer()),
                static_cast<const DataType*>(weight_buf.GetDeviceBuffer()),
                static_cast<const DataType*>(bias_buf.GetDeviceBuffer()),
                static_cast<const DataType*>(out_buf.GetDeviceBuffer()),
                static_cast<DataType*>(final_out_buf.GetDeviceBuffer()),
                static_cast<uint8_t*>(linear_workspace_buf.GetDeviceBuffer()),
                b,
                h,
                s,
                d,
                stream);
            HIP_CHECK_ERROR(hipDeviceSynchronize());
        }

        if(verify)
        {
            ck_tile::DeviceMem ref_out_buf(static_cast<size_t>(b) * h * s * d * sizeof(DataType));
            ck_tile::DeviceMem ref_lse_buf(static_cast<size_t>(b) * h * s * sizeof(float));

            constexpr int num_threads = sla_reference::kReferenceBlockSize;
            hipLaunchKernelGGL(
                (sla_reference::fwd_ref<DataType>),
                dim3(b * h * s),
                dim3(num_threads),
                0, stream,
                static_cast<const uint16_t*>(q_buf.GetDeviceBuffer()),
                static_cast<const uint16_t*>(k_buf.GetDeviceBuffer()),
                static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()),
                static_cast<const int64_t*>(lut_buf.GetDeviceBuffer()),
                static_cast<const int32_t*>(lut_size_buf.GetDeviceBuffer()),
                static_cast<uint16_t*>(ref_out_buf.GetDeviceBuffer()),
                static_cast<float*>(ref_lse_buf.GetDeviceBuffer()),
                b * h * s, s, d, block_m, max_nnz, sm_scale);
            HIP_CHECK_ERROR(hipDeviceSynchronize());

            if(run_linear)
            {
                ck_tile::DeviceMem ref_k_feature_buf(
                    static_cast<size_t>(b) * h * s * d * sizeof(DataType));
                ck_tile::DeviceMem ref_ksum_buf(
                    static_cast<size_t>(b) * h * d * sizeof(float));
                ck_tile::DeviceMem ref_kv_buf(
                    static_cast<size_t>(b) * h * d * d * sizeof(DataType));
                ck_tile::DeviceMem ref_projected_buf(
                    static_cast<size_t>(b) * h * d * d * sizeof(DataType));
                ck_tile::DeviceMem ref_final_out_buf(
                    static_cast<size_t>(b) * h * s * d * sizeof(DataType));

                hipLaunchKernelGGL(
                    (sla_reference::linear_k_feature_ref<DataType>),
                    dim3(b * h * s), dim3(num_threads), 0, stream,
                    static_cast<const DataType*>(k_buf.GetDeviceBuffer()),
                    static_cast<DataType*>(ref_k_feature_buf.GetDeviceBuffer()),
                    b * h * s, d);
                hipLaunchKernelGGL(
                    (sla_reference::linear_ksum_ref<DataType>),
                    dim3(b * h * d), dim3(num_threads), 0, stream,
                    static_cast<const DataType*>(ref_k_feature_buf.GetDeviceBuffer()),
                    static_cast<float*>(ref_ksum_buf.GetDeviceBuffer()),
                    b * h, s, d);
                hipLaunchKernelGGL(
                    (sla_reference::linear_kv_ref<DataType>),
                    dim3(b * h * d * d), dim3(num_threads), 0, stream,
                    static_cast<const DataType*>(ref_k_feature_buf.GetDeviceBuffer()),
                    static_cast<const DataType*>(v_buf.GetDeviceBuffer()),
                    static_cast<DataType*>(ref_kv_buf.GetDeviceBuffer()),
                    b * h, s, d);
                hipLaunchKernelGGL(
                    (sla_reference::linear_projection_ref<DataType>),
                    dim3(b * h * d * d), dim3(num_threads), 0, stream,
                    static_cast<const DataType*>(ref_kv_buf.GetDeviceBuffer()),
                    static_cast<const DataType*>(weight_buf.GetDeviceBuffer()),
                    static_cast<DataType*>(ref_projected_buf.GetDeviceBuffer()),
                    b * h, d);
                hipLaunchKernelGGL(
                    (sla_reference::linear_final_ref<DataType>),
                    dim3(b * h * s), dim3(num_threads), 0, stream,
                    static_cast<const DataType*>(q_buf.GetDeviceBuffer()),
                    static_cast<const DataType*>(ref_projected_buf.GetDeviceBuffer()),
                    static_cast<const float*>(ref_ksum_buf.GetDeviceBuffer()),
                    static_cast<const DataType*>(bias_buf.GetDeviceBuffer()),
                    static_cast<const DataType*>(ref_out_buf.GetDeviceBuffer()),
                    static_cast<DataType*>(ref_final_out_buf.GetDeviceBuffer()),
                    b * h * s, s, d);
                HIP_CHECK_ERROR(hipDeviceSynchronize());

                ck_tile::HostTensor<DataType> final_host({b * h, s, d});
                ck_tile::HostTensor<DataType> ref_final_host({b * h, s, d});
                final_out_buf.FromDevice(final_host.data());
                ref_final_out_buf.FromDevice(ref_final_host.data());
                const double rtol = std::is_same_v<DataType, ck_tile::fp16_t> ? 2e-2 : 3e-2;
                const double atol = std::is_same_v<DataType, ck_tile::fp16_t> ? 2e-2 : 3e-2;
                const bool pass = ck_tile::check_err(
                    final_host, ref_final_host, "sparse + linear output error", rtol, atol);
                std::cout << "Verification (sparse + linear): "
                          << (pass ? "PASSED" : "FAILED") << "\n";
                return pass ? 0 : 1;
            }

            ck_tile::HostTensor<DataType> out_host({b * h, s, d});
            ck_tile::HostTensor<DataType> ref_out_host({b * h, s, d});
            out_buf.FromDevice(out_host.data());
            ref_out_buf.FromDevice(ref_out_host.data());

            const double rtol = std::is_same_v<DataType, ck_tile::fp16_t> ? 2e-2 : 1e-2;
            const double atol = std::is_same_v<DataType, ck_tile::fp16_t> ? 2e-2 : 1e-2;
            const bool pass = ck_tile::check_err(
                out_host, ref_out_host, "sparse output error", rtol, atol);
            std::cout << "Verification (sparse): " << (pass ? "PASSED" : "FAILED") << "\n";
            return pass ? 0 : 1;
        }

        return 0;
    }
};

template <typename DataType, typename... Instances>
struct SlaDispatcher;

template <typename DataType, typename FirstInstance, typename... RestInstances>
struct SlaDispatcher<DataType, FirstInstance, RestInstances...>
{
    static int Run(const ck_tile::ArgParser& arg_parser, int s)
    {
        if(FirstInstance::IsSupported(s))
        {
            return FirstInstance::Run(arg_parser);
        }
        return SlaDispatcher<DataType, RestInstances...>::Run(arg_parser, s);
    }
};

template <typename DataType>
struct SlaDispatcher<DataType>
{
    static int Run(const ck_tile::ArgParser&, int s)
    {
        std::cerr << "No matching SLA instance found for s=" << s << std::endl;
        return 1;
    }
};

template <typename DataType>
int run_sla_fwd(const ck_tile::ArgParser& arg_parser)
{
    const int s = arg_parser.get_int("s");
    using Dispatcher = SlaDispatcher<
        DataType,
        SlaFwdInstance<DataType, ck_tile::example::sla::kLargeBlockM>,
        SlaFwdInstance<DataType, ck_tile::example::sla::kDefaultBlockM>>;
    return Dispatcher::Run(arg_parser, s);
}
