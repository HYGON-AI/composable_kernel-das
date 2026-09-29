// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host.hpp"
#include "ck_tile/ops/sla.hpp"
#include "sla_sparse_map.hpp"
#include "sla_bwd_launch.hpp"
#include "sla_config.hpp"


#include "ck_tile/host/reference/reference_sla.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// -------------------------------------------------------------
// SLA Backward Instance & Dispatcher Pattern
// -------------------------------------------------------------

template <typename DataType, int BlockM>
struct SlaBwdInstance
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
        const int warmup = arg_parser.get_int("warmup");
        const int repeat = arg_parser.get_int("repeat");
        const int verify = arg_parser.get_int("v");
        const uint32_t seed = arg_parser.get_uint32("seed");

        const int block_m = BlockM;
        constexpr int block_n = ck_tile::example::sla::kBlockN;
        const int q_blocks = s / block_m;
        const int kv_blocks = s / block_n;
        const int max_nnz = std::max(1, static_cast<int>(topk_ratio * kv_blocks));
        const int auto_dq_stage_count =
            s > ck_tile::example::sla::kLargeSequenceThreshold
                ? ck_tile::example::sla::kLargeSequenceDqStageCount
                : ck_tile::example::sla::kDefaultStageCount;
        const int requested_dq_stage_count =
            s == ck_tile::example::sla::kTunedSequenceLength
                ? ck_tile::example::sla::kTunedBwdDqStageCount
                       : (req_kv_stages == 0 ? auto_dq_stage_count : req_kv_stages);
        // The dQ pipeline caches one partitioned stage in kDqLutStageCapacity LDS entries.
        // Increase the number of stages automatically instead of limiting total top-k.
        const int required_dq_stage_count =
            (max_nnz + ck_tile::example::sla::kDqLutStageCapacity - 1) /
            ck_tile::example::sla::kDqLutStageCapacity;
        const int dq_stage_count = std::max(requested_dq_stage_count,
                                            required_dq_stage_count);

        if(b <= 0 || h <= 0 || s <= 0 ||
           s % ck_tile::example::sla::kBlockN != 0 ||
           d != ck_tile::example::sla::kHeadDim ||
           topk_ratio <= 0.0f || topk_ratio > 1.0f ||
           s / ck_tile::example::sla::kBlockN > ck_tile::example::sla::kMaxKvBlocks ||
           req_kv_stages < 0 ||
           req_kv_stages > ck_tile::example::sla::kMaxKvStageCount ||
           dq_stage_count < 1 ||
           dq_stage_count > ck_tile::example::sla::kMaxKvStageCount || warmup <= 0 ||
           repeat <= 0)
        {
            std::cerr << "unsupported SLA BWD arguments: require b,h>0, s>0 and divisible "
                         "by 64, d=128, KV blocks/topk <=2048, kv_stages in [0,16], "
                         "and warmup/repeat>0\n";
            return 1;
        }
        if(!ck_tile::example::sla::is_sla_shape_indexable(b, h, s, d))
        {
            std::cerr << "unsupported SLA shape: b*h*max(s,d)*d must fit in INT_MAX "
                         "for 32-bit kernel indexing\n";
            return 1;
        }
        if(s == ck_tile::example::sla::kTunedSequenceLength && req_kv_stages != 0 &&
           req_kv_stages != ck_tile::example::sla::kTunedBwdDqStageCount)
        {
            std::cout << "Requested kv_stages=" << req_kv_stages
                      << " is overridden; tuned optimal kv_stages for S=75648 is "
                      << ck_tile::example::sla::kTunedBwdDqStageCount << ".\n";
        }
        if(req_kv_stages != 0 && req_kv_stages < required_dq_stage_count)
        {
            std::cout << "Requested kv_stages=" << req_kv_stages
                      << " is increased to " << required_dq_stage_count
                      << " so every dQ LUT stage contains at most "
                      << ck_tile::example::sla::kDqLutStageCapacity << " entries.\n";
        }

        std::cout << "SLA BWD Config (Instance " << BlockM << "): B=" << b << ", H=" << h
                  << ", S=" << s << ", D=" << d << ", BlockM=" << block_m
                  << ", prec=" << arg_parser.get_str("prec") << ", topk_ratio=" << topk_ratio
                  << ", dq_stages=" << dq_stage_count << "\n";

        const float sm_scale = 1.0f / std::sqrt(static_cast<float>(d));

        // Allocate host tensors
        ck_tile::HostTensor<DataType> q_host({b * h, s, d});
        ck_tile::HostTensor<DataType> k_host({b * h, s, d});
        ck_tile::HostTensor<DataType> v_host({b * h, s, d});
        // Random stand-in for O produced by the forward sparse-attention pass.
        ck_tile::HostTensor<DataType> out_host({b * h, s, d});
        // Random stand-in for forward LSE consumed by dQ and dK/dV.
        ck_tile::HostTensor<float> lse_host({b * h, s});
        // dO is the upstream gradient supplied to the backward pass.
        ck_tile::HostTensor<DataType> do_host({b * h, s, d});

        ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed, true}(q_host);
        ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed + 1, true}(k_host);
        ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, seed + 2, true}(v_host);
        ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, seed + 3, true}(out_host);
        // LSE is generated directly rather than by FWD, but must stay in the
        // valid log2-sum-exp range. Using [-2, 2] makes unnormalized attention
        // probabilities hundreds of times too large and magnifies BF16 noise.
        const int random_lse_topk =
            std::max(1,
                     static_cast<int>(topk_ratio *
                                      (s / ck_tile::example::sla::kBlockN)));
        const float random_lse_center =
            std::log2(static_cast<float>(random_lse_topk * block_n));
        ck_tile::FillUniformDistribution<float>{random_lse_center - 0.5f,
                                                 random_lse_center + 0.5f,
                                                 seed + 4,
                                                 true}(lse_host);
        ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, seed + 5, true}(do_host);

        // Allocate device buffers
        ck_tile::DeviceMem q_buf(q_host);
        ck_tile::DeviceMem k_buf(k_host);
        ck_tile::DeviceMem v_buf(v_host);
        ck_tile::DeviceMem out_buf(out_host);
        ck_tile::DeviceMem lse_buf(lse_host);
        ck_tile::DeviceMem do_buf(do_host);

        ck_tile::DeviceMem dq_buf(static_cast<size_t>(b) * h * s * d * sizeof(DataType));
        ck_tile::DeviceMem dk_buf(static_cast<size_t>(b) * h * s * d * sizeof(DataType));
        ck_tile::DeviceMem dv_buf(static_cast<size_t>(b) * h * s * d * sizeof(DataType));
        ck_tile::DeviceMem delta_buf(static_cast<size_t>(b) * h * s * sizeof(float));

        dq_buf.SetZero();
        dk_buf.SetZero();
        dv_buf.SetZero();
        delta_buf.SetZero();

        // Allocate sparse_map and reverse_lut metadata
        const int rlut_stride_n =
            q_blocks * (block_m / ck_tile::example::sla::kQSliceBlockM);

        ck_tile::DeviceMem sparse_map_buf(static_cast<size_t>(b) * h * q_blocks * kv_blocks * sizeof(int8_t));
        ck_tile::DeviceMem lut_buf(static_cast<size_t>(b) * h * q_blocks * max_nnz * sizeof(int64_t));
        ck_tile::DeviceMem lut_size_buf(static_cast<size_t>(b) * h * q_blocks * sizeof(int32_t));

        ck_tile::DeviceMem rlut_buf(static_cast<size_t>(b) * h * kv_blocks * rlut_stride_n * sizeof(int32_t));
        ck_tile::DeviceMem rsize_buf(static_cast<size_t>(b) * h * kv_blocks * sizeof(int32_t));
        ck_tile::DeviceMem kv_perm_buf(static_cast<size_t>(b) * h * kv_blocks * sizeof(int32_t));
        const bool use_dq_partition =
            max_nnz > ck_tile::example::sla::kDqLutStageCapacity ||
            dq_stage_count > 1 || BlockM == ck_tile::example::sla::kLargeBlockM;
        ck_tile::DeviceMem dq_partitioned_lut_buf(
            use_dq_partition
                ? static_cast<size_t>(b) * h * q_blocks * max_nnz * sizeof(int64_t)
                : 1);
        ck_tile::DeviceMem dq_workspace_buf(
            dq_stage_count > 1
                ? static_cast<size_t>(dq_stage_count) * b * h * s * d * sizeof(DataType)
                : 1);

        sparse_map_buf.SetZero();
        lut_buf.SetZero();
        lut_size_buf.SetZero();
        rlut_buf.SetZero();
        rsize_buf.SetZero();
        kv_perm_buf.SetZero();

        const int64_t* dq_kernel_lut = use_dq_partition
                                           ? static_cast<const int64_t*>(
                                                 dq_partitioned_lut_buf.GetDeviceBuffer())
                                           : static_cast<const int64_t*>(lut_buf.GetDeviceBuffer());
        hipStream_t stream = nullptr;

        // Build sparse metadata outside the BWD timing region.
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

        // Setup DQ Kernel Arguments
        using DqProblem = ck_tile::SlaAttnBwdDqAttentionProblem<DataType>;
        using DqPolicy = ck_tile::SlaAttnBwdDqDefaultPolicy<DqProblem>;
        using DqKernel = ck_tile::SlaAttnBwdDqKernel<DqProblem, DqPolicy>;

        const auto dq_kargs = DqKernel::MakeKargs(
            q_buf.GetDeviceBuffer(),
            k_buf.GetDeviceBuffer(),
            v_buf.GetDeviceBuffer(),
            do_buf.GetDeviceBuffer(),
            delta_buf.GetDeviceBuffer(),
            lse_buf.GetDeviceBuffer(),
            dq_buf.GetDeviceBuffer(),
            dq_stage_count > 1 ? dq_workspace_buf.GetDeviceBuffer() : nullptr,
            dq_stage_count,
            dq_kernel_lut,
            sm_scale,
            s,
            max_nnz,
            BlockM,
            s * d, d, 1,
            s * d, d, 1,
            s * d, d, 1,
            s * d, d, 1,
            s * d, d, 1,
            s, 1,
            s, 1,
            q_blocks * max_nnz, max_nnz, 1);

        auto dq_grid = DqKernel::GridSize(b, h, s);
        dq_grid.z = dq_stage_count;
        constexpr auto dq_block = DqKernel::BlockSize();

        auto run_bwd_pipeline = [&]() {
            // Preprocess (delta)
            ck_tile::example::sla::launch_sla_attn_bwd_preprocess(
                static_cast<const DataType*>(out_buf.GetDeviceBuffer()),
                static_cast<const DataType*>(do_buf.GetDeviceBuffer()),
                static_cast<float*>(delta_buf.GetDeviceBuffer()),
                b * h * s, d, stream);

            // dQ LUT partitioning follows the extension path. M128 keeps one
            // stage but still groups nearby KV blocks; long M64 uses two stages.
            if(use_dq_partition)
            {
                const ck_tile::example::sla::SlaAttnBwdDqKvPartitionKargs partition_arg{
                    static_cast<const int64_t*>(lut_buf.GetDeviceBuffer()),
                    static_cast<int64_t*>(dq_partitioned_lut_buf.GetDeviceBuffer()),
                    static_cast<int64_t>(b) * h * q_blocks,
                    max_nnz,
                    kv_blocks,
                    dq_stage_count};
                if(dq_stage_count == 1)
                    ck_tile::example::sla::launch_sla_attn_bwd_dq_kv_partition<1>(
                        partition_arg, stream);
                else if(dq_stage_count == 2)
                    ck_tile::example::sla::launch_sla_attn_bwd_dq_kv_partition<2>(
                        partition_arg, stream);
                else
                    ck_tile::example::sla::launch_sla_attn_bwd_dq_kv_partition<
                        ck_tile::example::sla::kMaxKvStageCount>(
                        partition_arg, stream);
            }

            // DQ
            ck_tile::launch_kernel(
                ck_tile::stream_config{stream},
                ck_tile::make_kernel<DqKernel::kBlockSize, DqKernel::kBlockPerCu>(
                    DqKernel{}, dq_grid, dq_block, 0, dq_kargs));

            if(dq_stage_count > 1)
            {
                ck_tile::example::sla::launch_sla_attn_bwd_dq_reduce(
                    static_cast<const DataType*>(dq_workspace_buf.GetDeviceBuffer()),
                    static_cast<DataType*>(dq_buf.GetDeviceBuffer()),
                    dq_stage_count,
                    static_cast<int64_t>(b) * h * s * d,
                    static_cast<int64_t>(b) * h * s * d,
                    stream);
            }

            // DK & DV
            ck_tile::example::sla::launch_sla_attn_bwd_dkdv<DataType>(
                q_buf.GetDeviceBuffer(),
                k_buf.GetDeviceBuffer(),
                v_buf.GetDeviceBuffer(),
                do_buf.GetDeviceBuffer(),
                static_cast<const float*>(lse_buf.GetDeviceBuffer()),
                static_cast<const float*>(delta_buf.GetDeviceBuffer()),
                static_cast<const int8_t*>(sparse_map_buf.GetDeviceBuffer()),
                dk_buf.GetDeviceBuffer(),
                dv_buf.GetDeviceBuffer(),
                static_cast<int32_t*>(rlut_buf.GetDeviceBuffer()),
                static_cast<int32_t*>(rsize_buf.GetDeviceBuffer()),
                static_cast<int32_t*>(kv_perm_buf.GetDeviceBuffer()),
                b, h, s, d, block_m, block_n, sm_scale, stream);
        };

        std::cout << "Benchmarking bwd pipeline (preprocess + dq + dkdv)..." << std::endl;
        const ck_tile::stream_config benchmark_config{
            stream, true, 0, warmup, repeat, true};
        const float avg_time_ms = ck_tile::launch_kernel(
            benchmark_config,
            [&](const ck_tile::stream_config&) { run_bwd_pipeline(); });
        std::cout << "sparse_attn BWD Avg Latency: " << avg_time_ms << " ms\n";

        if(verify)
        {
            ck_tile::DeviceMem ref_delta_buf(static_cast<size_t>(b) * h * s * sizeof(float));
            ck_tile::DeviceMem ref_dq_buf(static_cast<size_t>(b) * h * s * d * sizeof(float));
            ck_tile::DeviceMem ref_dk_buf(static_cast<size_t>(b) * h * s * d * sizeof(float));
            ck_tile::DeviceMem ref_dv_buf(static_cast<size_t>(b) * h * s * d * sizeof(float));
            constexpr int num_threads = sla_reference::kReferenceBlockSize;
            hipLaunchKernelGGL(
                (sla_reference::preprocess_ref<DataType>),
                dim3(b * h * s),
                dim3(num_threads),
                0, stream,
                static_cast<const uint16_t*>(out_buf.GetDeviceBuffer()),
                static_cast<const uint16_t*>(do_buf.GetDeviceBuffer()),
                static_cast<float*>(ref_delta_buf.GetDeviceBuffer()),
                b * h * s, d);
            for(int bh = 0; bh < b * h; ++bh)
            {
                const size_t tensor_offset = static_cast<size_t>(bh) * s * d;
                hipLaunchKernelGGL(
                    (sla_reference::bwd_dq_ref<DataType>),
                    dim3(s), dim3(num_threads), 0, stream,
                    static_cast<const uint16_t*>(q_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<const uint16_t*>(k_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<const uint16_t*>(do_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<const float*>(lse_buf.GetDeviceBuffer()) + bh * s,
                    static_cast<const float*>(ref_delta_buf.GetDeviceBuffer()) + bh * s,
                    dq_kernel_lut +
                        static_cast<size_t>(bh) * q_blocks * max_nnz,
                    static_cast<float*>(ref_dq_buf.GetDeviceBuffer()) + tensor_offset,
                    s,
                    d,
                    block_m,
                    block_n,
                    max_nnz,
                    dq_stage_count,
                    sm_scale * ck_tile::log2e_v<float>);
                hipLaunchKernelGGL(
                    (sla_reference::bwd_dkdv_ref<DataType>),
                    dim3(s), dim3(num_threads), 0, stream,
                    static_cast<const uint16_t*>(q_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<const uint16_t*>(k_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<const uint16_t*>(do_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<const float*>(lse_buf.GetDeviceBuffer()) + bh * s,
                    static_cast<const float*>(ref_delta_buf.GetDeviceBuffer()) + bh * s,
                    static_cast<const int32_t*>(rlut_buf.GetDeviceBuffer()) +
                        static_cast<size_t>(bh) * kv_blocks * rlut_stride_n,
                    static_cast<const int32_t*>(rsize_buf.GetDeviceBuffer()) + bh * kv_blocks,
                    static_cast<float*>(ref_dk_buf.GetDeviceBuffer()) + tensor_offset,
                    static_cast<float*>(ref_dv_buf.GetDeviceBuffer()) + tensor_offset,
                    s, d, block_n, rlut_stride_n, sm_scale * ck_tile::log2e_v<float>);
            }
            HIP_CHECK_ERROR(hipDeviceSynchronize());

            ck_tile::HostTensor<float> delta_host({b * h, s});
            ck_tile::HostTensor<float> ref_delta_host({b * h, s});
            ck_tile::HostTensor<DataType> dq_host({b * h, s, d});
            ck_tile::HostTensor<DataType> dk_host({b * h, s, d});
            ck_tile::HostTensor<DataType> dv_host({b * h, s, d});
            ck_tile::HostTensor<float> ref_dq_host({b * h, s, d});
            ck_tile::HostTensor<float> ref_dk_host({b * h, s, d});
            ck_tile::HostTensor<float> ref_dv_host({b * h, s, d});
            delta_buf.FromDevice(delta_host.data());
            ref_delta_buf.FromDevice(ref_delta_host.data());
            dq_buf.FromDevice(dq_host.data());
            dk_buf.FromDevice(dk_host.data());
            dv_buf.FromDevice(dv_host.data());
            ref_dq_buf.FromDevice(ref_dq_host.data());
            ref_dk_buf.FromDevice(ref_dk_host.data());
            ref_dv_buf.FromDevice(ref_dv_host.data());

            bool pass = ck_tile::check_err(
                delta_host, ref_delta_host, "delta error check", 1e-2, 1e-2);
            const auto dq_float = dq_host.template CopyAsType<float>();
            const auto dk_float = dk_host.template CopyAsType<float>();
            const auto dv_float = dv_host.template CopyAsType<float>();
            const double grad_rtol =
                std::is_same_v<DataType, ck_tile::fp16_t> ? 1e-2 : 2e-2;
            const double grad_atol =
                std::is_same_v<DataType, ck_tile::fp16_t> ? 1e-3 : 1e-2;
            pass &= ck_tile::check_err(
                dq_float, ref_dq_host, "dQ error check", grad_rtol, grad_atol);
            pass &= ck_tile::check_err(
                dk_float, ref_dk_host, "dK error check", grad_rtol, grad_atol);
            pass &= ck_tile::check_err(
                dv_float, ref_dv_host, "dV error check", grad_rtol, grad_atol);
            std::cout << "BWD Verification (preprocess/dQ/dK/dV): "
                      << (pass ? "PASSED" : "FAILED") << "\n";
            if(!pass)
                return 1;
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
int run_sla_bwd(const ck_tile::ArgParser& arg_parser)
{
    const int s = arg_parser.get_int("s");
    using Dispatcher = SlaDispatcher<
        DataType,
        SlaBwdInstance<DataType, ck_tile::example::sla::kLargeBlockM>,
        SlaBwdInstance<DataType, ck_tile::example::sla::kDefaultBlockM>>;
    return Dispatcher::Run(arg_parser, s);
}
