// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_problem.hpp"
#include "ck_tile/ops/jenga.hpp"
#include "jenga_bwd.hpp"
#include "jenga_config.hpp"

#include "ck_tile/host/reference/reference_jenga.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct PreprocessTraits
{
    static constexpr bool kPadSeqLenQ = true;
    static constexpr bool kPadHeadDimV = false;
    static constexpr ck_tile::index_t kBlockPerCu = 1;
};
template <typename DataType>
using PreprocessProblem = ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<
    DataType,
    DataType,
    float,
    ck_tile::example::jenga::kBlockM,
    ck_tile::example::jenga::kHeadDim,
    false,
    PreprocessTraits>;
template <typename DataType>
using PreprocessDot = ck_tile::BlockFmhaBwdOGradDotO<PreprocessProblem<DataType>>;
template <typename DataType>
using PreprocessKernel = ck_tile::JengaBwdPreprocessKernel<PreprocessDot<DataType>>;

} // namespace

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "1", "batch size")
        .insert("h", "1", "number of heads")
        .insert("s", "1024", "sequence length (multiple of 64)")
        .insert("d", "128", "head dimension (currently 128)")
        .insert("prec", "bf16", "data type: bf16 or fp16")
        .insert("topk", "8", "minimum visual KV blocks selected per Q block")
        .insert("prob_threshold", "0.3", "cumulative probability threshold")
        .insert("text_start_block", "-1", "first text block; -1 means all visual blocks")
        .insert("text_blocks", "0", "number of text blocks always selected")
        .insert("first_frame_blocks", "0", "first-frame blocks always selected")
        .insert("dq_kv_stages", "0", "dQ KV stages: 0=75648 uses 5, otherwise 1")
        .insert("dkdv_split_kv", "0", "dK/dV split: 0=75648 uses 4, otherwise 1")
        .insert("v", "1", "run GPU backward reference")
        .insert("warmup", "10", "warmup launches")
        .insert("repeat", "100", "timed launches");
    return std::make_tuple(arg_parser.parse(argc, argv), arg_parser);
}

template <typename DataType>
int run(const ck_tile::ArgParser& arg_parser)
{
    const int B = arg_parser.get_int("b"), H = arg_parser.get_int("h");
    const int N = arg_parser.get_int("s"), D = arg_parser.get_int("d");
    const int top_k = arg_parser.get_int("topk");
    if(B <= 0 || H <= 0 || N <= 0 ||
       N % ck_tile::example::jenga::kBlockM != 0 ||
       D != ck_tile::example::jenga::kHeadDim)
    {
        std::cerr << "jenga_bwd requires b,h > 0, s aligned to BlockM, "
                     "and the supported head dimension\n";
        return 2;
    }
    const int BH = B * H, blocks = N / ck_tile::example::jenga::kBlockM;
    const int text_start_block_arg = arg_parser.get_int("text_start_block");
    const int text_start_block = text_start_block_arg < 0 ? blocks : text_start_block_arg;
    const int text_blocks = arg_parser.get_int("text_blocks");
    const int first_frame_blocks = arg_parser.get_int("first_frame_blocks");
    const int requested_dq_stages = arg_parser.get_int("dq_kv_stages");
    const int requested_dkdv_split = arg_parser.get_int("dkdv_split_kv");
    const float prob_threshold = arg_parser.get_float("prob_threshold");
    if(top_k <= 0 || top_k > text_start_block || prob_threshold < 0.0f ||
       prob_threshold > 1.0f || text_start_block < 0 ||
       text_start_block > blocks || text_blocks < 0 ||
       text_start_block + text_blocks > blocks || first_frame_blocks < 0 ||
       requested_dq_stages < 0 ||
       requested_dq_stages > ck_tile::example::jenga::kMaxDqStageCount ||
       requested_dkdv_split < 0 ||
       requested_dkdv_split > ck_tile::example::jenga::kMaxDkdvSplitCount)
    {
        std::cerr << "invalid topk/text block configuration\n";
        return 2;
    }
    using MaskProblem = JengaMaskBuilderProblem<DataType,
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
    int active_capacity = std::min(
        blocks, threshold_capacity + text_blocks + std::min(first_frame_blocks, blocks));
    constexpr int dq_max_nnz_capacity = ck_tile::example::jenga::kDqMaxNnzCap;
    if(active_capacity > dq_max_nnz_capacity)
    {
        std::cerr << "generated forward LUT capacity exceeds the configured dQ limit\n";
        return 2;
    }
    const int reverse_capacity = blocks;
    const int dq_kv_stages = requested_dq_stages == 0
                                 ? (N == ck_tile::example::jenga::kTunedSequenceLength
                                        ? ck_tile::example::jenga::kTunedDqStageCount
                                        : ck_tile::example::jenga::kDefaultStageCount)
                                 : requested_dq_stages;
    const int dkdv_split_kv = requested_dkdv_split == 0
                                  ? (N == ck_tile::example::jenga::kTunedSequenceLength
                                         ? ck_tile::example::jenga::kTunedDkdvSplitCount
                                         : ck_tile::example::jenga::kDefaultStageCount)
                                  : requested_dkdv_split;
    const size_t elems = static_cast<size_t>(BH) * N * D;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    constexpr float kLog2e = ck_tile::log2e_v<float>;
    ck_tile::HostTensor<DataType> q_host({BH, N, D});
    ck_tile::HostTensor<DataType> k_host({BH, N, D});
    ck_tile::HostTensor<DataType> v_host({BH, N, D});
    ck_tile::HostTensor<DataType> dout_host({BH, N, D});
    ck_tile::HostTensor<int32_t> seqlens_host({B});
    ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, 11939, true}(q_host);
    ck_tile::FillUniformDistribution<DataType>{-0.25f, 0.25f, 11940, true}(k_host);
    ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, 11941, true}(v_host);
    ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, 11942, true}(dout_host);
    std::fill(seqlens_host.begin(), seqlens_host.end(), N);

    ck_tile::DeviceMem q_buf(q_host), k_buf(k_host), v_buf(v_host), dout_buf(dout_host);
    ck_tile::DeviceMem out_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dq_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dk_buf(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dv_buf(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dq_workspace_buf(
        dq_kv_stages > 1 ? static_cast<size_t>(dq_kv_stages) * elems * sizeof(DataType) : 1);
    ck_tile::DeviceMem dk_workspace_buf(
        dkdv_split_kv > 1 ? static_cast<size_t>(dkdv_split_kv) * elems * sizeof(DataType) : 1);
    ck_tile::DeviceMem dv_workspace_buf(
        dkdv_split_kv > 1 ? static_cast<size_t>(dkdv_split_kv) * elems * sizeof(DataType) : 1);
    ck_tile::DeviceMem delta_buf(static_cast<size_t>(BH) * N * sizeof(float));
    ck_tile::DeviceMem lse_buf(static_cast<size_t>(BH) * N * sizeof(float));
    ck_tile::DeviceMem seqlens_buf(seqlens_host);
    ck_tile::DeviceMem onehot_buf(static_cast<size_t>(BH) * blocks * blocks * sizeof(bool));
    ck_tile::DeviceMem q_pool_buf(static_cast<size_t>(BH) * blocks * D * sizeof(float));
    ck_tile::DeviceMem k_pool_buf(static_cast<size_t>(BH) * blocks * D * sizeof(float));
    ck_tile::DeviceMem scores_buf(static_cast<size_t>(BH) * blocks * blocks * sizeof(float));
    ck_tile::DeviceMem lut_buf(
        static_cast<size_t>(BH) * blocks * active_capacity * sizeof(int32_t));
    ck_tile::DeviceMem lut_size_buf(static_cast<size_t>(BH) * blocks * sizeof(int32_t));
    ck_tile::DeviceMem rlut_buf(
        static_cast<size_t>(BH) * blocks * reverse_capacity * sizeof(int32_t));
    ck_tile::DeviceMem rlut_size_buf(static_cast<size_t>(BH) * blocks * sizeof(int32_t));
    auto* q = static_cast<uint16_t*>(q_buf.GetDeviceBuffer());
    auto* k = static_cast<uint16_t*>(k_buf.GetDeviceBuffer());
    auto* v = static_cast<uint16_t*>(v_buf.GetDeviceBuffer());
    auto* out = static_cast<uint16_t*>(out_buf.GetDeviceBuffer());
    auto* dout = static_cast<uint16_t*>(dout_buf.GetDeviceBuffer());
    auto* dq = static_cast<uint16_t*>(dq_buf.GetDeviceBuffer());
    auto* dk = static_cast<uint16_t*>(dk_buf.GetDeviceBuffer());
    auto* dv = static_cast<uint16_t*>(dv_buf.GetDeviceBuffer());
    auto* delta = static_cast<float*>(delta_buf.GetDeviceBuffer());
    auto* lse = static_cast<float*>(lse_buf.GetDeviceBuffer());
    auto* seqlens = static_cast<int32_t*>(seqlens_buf.GetDeviceBuffer());
    auto* lut = static_cast<int32_t*>(lut_buf.GetDeviceBuffer());
    auto* lut_size = static_cast<int32_t*>(lut_size_buf.GetDeviceBuffer());
    auto* rlut = static_cast<int32_t*>(rlut_buf.GetDeviceBuffer());
    auto* rlut_size = static_cast<int32_t*>(rlut_size_buf.GetDeviceBuffer());

    onehot_buf.SetZero();
    JengaMaskBuilderArgument mask_args{};
    mask_args.query = q; mask_args.key = k; mask_args.neighbor_mask = nullptr;
    mask_args.out = static_cast<bool*>(onehot_buf.GetDeviceBuffer());
    mask_args.B = B; mask_args.H = H; mask_args.N_Q = N; mask_args.N_K = N; mask_args.D = D;
    mask_args.num_query_blocks = blocks; mask_args.num_blocks = blocks;
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
    jenga_onehot_to_lut(mask_args.out, lut, lut_size, BH * blocks, blocks, active_capacity);
    jenga_onehot_to_reverse_lut(
        mask_args.out, rlut, rlut_size, BH, blocks, blocks, reverse_capacity);
    HIP_CHECK_ERROR(hipGetLastError());
    // Mask builder and both LUT conversions are preprocessing and are not timed as backward.
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<int32_t> selected_counts(static_cast<size_t>(BH) * blocks);
    lut_size_buf.FromDevice(selected_counts.data());
    int64_t selected_sum = 0;
    for(const int32_t count : selected_counts)
        selected_sum += count;
    const double selected_avg =
        static_cast<double>(selected_sum) / static_cast<double>(selected_counts.size());

    // O/LSE are independent random inputs for this backward-only example.
    ck_tile::HostTensor<DataType> out_host({BH, N, D});
    ck_tile::HostTensor<float> lse_host({BH, N});
    ck_tile::FillUniformDistribution<DataType>{-0.5f, 0.5f, 11943, true}(out_host);
    const float lse_center = std::log2(std::max(
        1.0, selected_avg * static_cast<double>(ck_tile::example::jenga::kBlockN)));
    ck_tile::FillUniformDistribution<float>{
        lse_center - 0.25f, lse_center + 0.25f, 11944, true}(lse_host);
    out_buf.ToDevice(out_host.data());
    lse_buf.ToDevice(lse_host.data());

    using Preprocess = PreprocessKernel<DataType>;
    auto pkargs = Preprocess::MakeKargs(out, dout, delta, seqlens, H, N, D,
        static_cast<ck_tile::index_t>(N) * D, D, 1,
        static_cast<ck_tile::index_t>(N) * D, D, 1, N, 1);
    const std::string precision = std::is_same_v<DataType, ck_tile::fp16_t> ? "fp16" : "bf16";
    std::cout << "Jenga BWD Config: B=" << B << ", H=" << H
              << ", S=" << N << ", D=" << D
              << ", prec=" << precision << ", topk=" << top_k
              << ", selected_k_blocks_avg=" << selected_avg
              << ", dq_kv_stages=" << dq_kv_stages
              << ", dkdv_split_kv=" << dkdv_split_kv << "\n";
    std::cout << "Benchmarking bwd pipeline (preprocess + dq + dkdv)..." << std::endl;
    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         0,
                                         arg_parser.get_int("warmup"),
                                         arg_parser.get_int("repeat")};
    const float preprocess_ms = ck_tile::launch_kernel(stream_config,
        ck_tile::make_kernel<Preprocess::kBlockSize, Preprocess::kBlockPerCu>(
            Preprocess{}, Preprocess::GridSize(B, H, N), Preprocess::BlockSize(),
            Preprocess::GetSmemSize(), pkargs));

    ck_tile::jenga_dq::jenga_bwd_dq_args da{};
    da.q=q; da.k=k; da.v=v; da.dout=dout; da.deltas=delta; da.lse=lse; da.dq=dq;
    da.lut=lut; da.lut_size=lut_size; da.seqlens=seqlens; da.sm_scale=scale;
    da.text_block_start=text_start_block; da.stride_qz=N*D; da.stride_qm=D; da.stride_qk=1;
    da.stride_kz=N*D; da.stride_kn=D; da.stride_kk=1; da.stride_vz=N*D;
    da.stride_vn=D; da.stride_vk=1; da.stride_doz=N*D; da.stride_dom=D;
    da.stride_dok=1; da.stride_dqz=N*D; da.stride_dqm=D; da.stride_dqk=1;
    da.stride_dz=N; da.stride_dm=1; da.stride_lz=N; da.stride_lm=1;
    da.stride_lutz=blocks*active_capacity; da.stride_lutm=active_capacity; da.stride_lutk=1;
    da.block_m=ck_tile::example::jenga::kBlockM;
    da.block_n=ck_tile::example::jenga::kBlockN;
    da.max_nnz=active_capacity;
    da.batch=B;
    da.nhead=H;
    da.seqlen_q=N; da.kv_stage_count=dq_kv_stages;
    da.dq_workspace = dq_kv_stages > 1 ? dq_workspace_buf.GetDeviceBuffer() : nullptr;
    ck_tile::jenga_dq::jenga_bwd_dq_traits dt{
        precision,
        ck_tile::example::jenga::kBlockM,
        ck_tile::example::jenga::kBlockN,
        ck_tile::example::jenga::kHeadDim,
        active_capacity,
        dq_kv_stages};
    float dq_reduce_ms = 0.0f;
    const float dq_ms = jenga_bwd_dq(dt, da, stream_config, &dq_reduce_ms);
    if(dq_ms < 0)
        return 3;

    ck_tile::example::jenga::jenga_bwd_dkdv_args ka{};
    ka.q_ptr=q; ka.k_ptr=k; ka.v_ptr=v; ka.o_ptr=out; ka.do_ptr=dout;
    ka.delta_ptr=delta; ka.lse_ptr=lse; ka.lse_text_ptr=lse; ka.rlut_ptr=rlut;
    ka.rlut_size_ptr=rlut_size; ka.seqlens_ptr=seqlens; ka.dk_ptr=dk; ka.dv_ptr=dv;
    ka.B=B; ka.H=H; ka.N_Q=N; ka.N_KV=N; ka.D=D; ka.N_Q_BLOCKS=blocks;
    ka.N_KV_BLOCKS=blocks;
    ka.M0=ck_tile::example::jenga::kBlockM;
    ka.N0=ck_tile::example::jenga::kBlockN;
    ka.max_nnz_r=reverse_capacity;
    ka.text_block_start=text_start_block; ka.sm_scale=scale;
    ka.stride_qz=H*N*D; ka.stride_qh=N*D; ka.stride_qm=D; ka.stride_qd=1;
    ka.stride_kz=H*N*D; ka.stride_kh=N*D; ka.stride_kn=D; ka.stride_kd=1;
    ka.stride_vz=H*N*D; ka.stride_vh=N*D; ka.stride_vn=D; ka.stride_vd=1;
    ka.stride_oz=H*N*D; ka.stride_oh=N*D; ka.stride_om=D; ka.stride_od=1;
    ka.stride_doz=H*N*D; ka.stride_doh=N*D; ka.stride_dom=D; ka.stride_dod=1;
    ka.stride_delta_z=N; ka.stride_delta_m=1; ka.stride_dkz=H*N*D;
    ka.stride_dkh=N*D; ka.stride_dkn=D; ka.stride_dkd=1; ka.stride_dvz=H*N*D;
    ka.stride_dvh=N*D; ka.stride_dvn=D; ka.stride_dvd=1; ka.stride_lz=N;
    ka.stride_lm=1; ka.stride_rlutz=blocks*reverse_capacity; ka.stride_rlutn=reverse_capacity;
    ka.stride_rlutk=1; ka.stride_rlut_size_z=blocks; ka.stride_rlut_size_n=1;
    ka.workspace_dk_ptr =
        dkdv_split_kv > 1 ? dk_workspace_buf.GetDeviceBuffer() : nullptr;
    ka.workspace_dv_ptr =
        dkdv_split_kv > 1 ? dv_workspace_buf.GetDeviceBuffer() : nullptr;
    ka.split_kv = dkdv_split_kv;
    ka.stride_work_dk_s = elems;
    ka.stride_work_dv_s = elems;
    ck_tile::example::jenga::jenga_bwd_dkdv_traits dkdv_traits;
    dkdv_traits.data_type = precision;
    dkdv_traits.block_m = ck_tile::example::jenga::kBlockM;
    dkdv_traits.block_n = ck_tile::example::jenga::kBlockN;
    dkdv_traits.head_dim = ck_tile::example::jenga::kHeadDim;
    dkdv_traits.max_nnz = reverse_capacity;
    const float dkdv_fused_ms = jenga_bwd_dkdv_pipeline_calc(dkdv_traits, ka, stream_config);
    float dkdv_reduce_ms = 0.0f;
    if(dkdv_split_kv > 1)
    {
        dkdv_reduce_ms += jenga_bwd_reduce_workspace(
            static_cast<const DataType*>(dk_workspace_buf.GetDeviceBuffer()),
            reinterpret_cast<DataType*>(dk),
            dkdv_split_kv,
            elems,
            elems,
            stream_config);
        dkdv_reduce_ms += jenga_bwd_reduce_workspace(
            static_cast<const DataType*>(dv_workspace_buf.GetDeviceBuffer()),
            reinterpret_cast<DataType*>(dv),
            dkdv_split_kv,
            elems,
            elems,
            stream_config);
    }
    const float dkdv_ms = dkdv_fused_ms + dkdv_reduce_ms;
    const float bwd_total_ms = preprocess_ms + dq_ms + dkdv_ms;
    std::cout << "jenga_bwd Avg Latency: " << bwd_total_ms << " ms (dQ: "
              << dq_ms << " ms, dK/dV: " << dkdv_ms << " ms)\n";

    bool valid = true;
    if(arg_parser.get_bool("v"))
    {
        ck_tile::DeviceMem ref_delta_buf(static_cast<size_t>(BH) * N * sizeof(float));
        ck_tile::DeviceMem ref_dq_buf(elems * sizeof(float));
        ck_tile::DeviceMem ref_dk_buf(elems * sizeof(float));
        ck_tile::DeviceMem ref_dv_buf(elems * sizeof(float));
        auto* ref_delta = static_cast<float*>(ref_delta_buf.GetDeviceBuffer());
        auto* ref_dq = static_cast<float*>(ref_dq_buf.GetDeviceBuffer());
        auto* ref_dk = static_cast<float*>(ref_dk_buf.GetDeviceBuffer());
        auto* ref_dv = static_cast<float*>(ref_dv_buf.GetDeviceBuffer());
        hipLaunchKernelGGL((jenga_reference::preprocess<DataType>),
            dim3(BH * N),
            dim3(jenga_reference::kReferenceBlockSize),
            0,
            0,
            out,
            dout,
            ref_delta,
            BH * N,
            D);
        // Reference is launched independently for each flattened batch/head.
        for(int bh = 0; bh < BH; ++bh)
        {
            hipLaunchKernelGGL((jenga_reference::bwd_dq<DataType>),
                dim3(N),
                dim3(jenga_reference::kReferenceBlockSize),
                0,
                0,
                q + static_cast<size_t>(bh) * N * D,
                k + static_cast<size_t>(bh) * N * D,
                v + static_cast<size_t>(bh) * N * D,
                dout + static_cast<size_t>(bh) * N * D,
                lse + bh * N,
                ref_delta + bh * N,
                lut + bh * blocks * active_capacity,
                lut_size + bh * blocks,
                ref_dq + static_cast<size_t>(bh) * N * D,
                N,
                D,
                ck_tile::example::jenga::kBlockM,
                active_capacity,
                scale * kLog2e);
            hipLaunchKernelGGL((jenga_reference::bwd_dkdv<DataType>),
                dim3(N),
                dim3(jenga_reference::kReferenceBlockSize),
                0,
                0,
                q + static_cast<size_t>(bh) * N * D,
                k + static_cast<size_t>(bh) * N * D,
                v + static_cast<size_t>(bh) * N * D,
                dout + static_cast<size_t>(bh) * N * D,
                lse + bh * N,
                ref_delta + bh * N,
                rlut + bh * blocks * reverse_capacity,
                rlut_size + bh * blocks,
                ref_dk + static_cast<size_t>(bh) * N * D,
                ref_dv + static_cast<size_t>(bh) * N * D,
                N,
                D,
                ck_tile::example::jenga::kBlockN,
                reverse_capacity,
                scale * kLog2e);
        }
        HIP_CHECK_ERROR(hipGetLastError());
        HIP_CHECK_ERROR(hipDeviceSynchronize());

        ck_tile::HostTensor<float> got_delta({BH, N}), ref_delta_h({BH, N});
        ck_tile::HostTensor<DataType> got_dq({BH, N, D}), got_dk({BH, N, D}), got_dv({BH, N, D});
        ck_tile::HostTensor<float> ref_dq_h({BH, N, D}), ref_dk_h({BH, N, D}), ref_dv_h({BH, N, D});
        delta_buf.FromDevice(got_delta.data()); ref_delta_buf.FromDevice(ref_delta_h.data());
        dq_buf.FromDevice(got_dq.data()); dk_buf.FromDevice(got_dk.data()); dv_buf.FromDevice(got_dv.data());
        ref_dq_buf.FromDevice(ref_dq_h.data()); ref_dk_buf.FromDevice(ref_dk_h.data());
        ref_dv_buf.FromDevice(ref_dv_h.data());

        const auto got_dq_float = got_dq.template CopyAsType<float>();
        const auto got_dk_float = got_dk.template CopyAsType<float>();
        const auto got_dv_float = got_dv.template CopyAsType<float>();
        const double delta_rtol = 1e-2;
        const double delta_atol = 1e-2;
        const double grad_rtol =
            std::is_same_v<DataType, ck_tile::fp16_t> ? 1e-2 : 2e-2;
        const double grad_atol =
            std::is_same_v<DataType, ck_tile::fp16_t> ? 1e-3 : 1e-2;
        valid &= ck_tile::check_err(got_delta,
                                    ref_delta_h,
                                    std::string("Delta Error: Incorrect results!"),
                                    delta_rtol,
                                    delta_atol);
        valid &= ck_tile::check_err(got_dq_float,
                                    ref_dq_h,
                                    std::string("dQ Error: Incorrect results!"),
                                    grad_rtol,
                                    grad_atol);
        valid &= ck_tile::check_err(got_dk_float,
                                    ref_dk_h,
                                    std::string("dK Error: Incorrect results!"),
                                    grad_rtol,
                                    grad_atol);
        valid &= ck_tile::check_err(got_dv_float,
                                    ref_dv_h,
                                    std::string("dV Error: Incorrect results!"),
                                    grad_rtol,
                                    grad_atol);
        std::cout << "BWD Verification (preprocess/dQ/dK/dV): "
                  << (valid ? "PASSED" : "FAILED") << "\n";
    }
    else
    {
        std::cout << "BWD Verification (preprocess/dQ/dK/dV): SKIPPED\n";
    }
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    return valid ? 0 : 4;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return 1;
    const auto precision = arg_parser.get_str("prec");
    if(precision == "bf16")
        return run<ck_tile::bf16_t>(arg_parser);
    if(precision == "fp16")
        return run<ck_tile::fp16_t>(arg_parser);
    std::cerr << "unsupported precision: " << precision << std::endl;
    return 1;
}
