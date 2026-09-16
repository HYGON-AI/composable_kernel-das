// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_gdn.hpp"
#include "ck_tile/ops/gdn.hpp"
#include "prefill/gdn_prefill_launch.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("t", "1024", "total sequence length")
        .insert("h", "16", "number of Q/K heads")
        .insert("hv", "64", "number of value heads")
        .insert("head_dim", "128", "head/value dimension; only 128 is supported")
        .insert("prec", "bf16", "input precision: bf16 or fp16")
        .insert("has_initial_state", "1", "use an FP32 initial state in fwd_state")
        .insert("store_final_state", "1", "store and validate fwd_state final state")
        .insert("save_new_value", "1", "store fwd_state v_new")
        .insert("use_g", "1", "use scalar gate in fwd_state")
        .insert("use_gk", "0", "use per-key gate in fwd_state")
        .insert("use_exp2", "1", "input log gates use base 2; 0 uses natural logs")
        .insert("transpose_state", "0", "use transposed fwd_state layout")
        .insert("qk_l2norm", "0", "L2-normalize Q and K before execution")
        .insert("gates", "1", "use random negative log gates; 0 uses zero gates")
        .insert("gate_in_kernel", "0", "compute gate in cumsum from g and A_log")
        .insert("has_dt_bias", "1", "add dt_bias in the gate-in-kernel path")
        .insert("beta_sigmoid_in_kernel", "0", "sigmoid raw beta on the GPU")
        .insert("is_varlen", "0", "enable packed variable-length metadata")
        .insert("seq_endpoints", "", "comma-separated cumulative sequence endpoints")
        .insert("cp_context", "0", "request the reserved CP-context path")
        .insert("cp_world_size", "2", "number of ranks in the simulated CP group")
        .insert("cp_rank", "1", "current rank in the simulated CP group")
        .insert("input_scale", "0.1", "random Q/K/V and state scale")
        .insert("scale", "0", "Q scale; 0 selects 1/sqrt(d)")
        .insert("warmup", "10", "warmup iterations")
        .insert("repeat", "100", "timed iterations")
        .insert("v", "1", "run the independent GPU reference")
        .insert("seed", "42", "random seed");
    return std::make_tuple(arg_parser.parse(argc, argv), arg_parser);
}

template <typename DataType>
void normalize_qk(ck_tile::HostTensor<DataType>& tensor,
                  int t,
                  int heads,
                  int head_dim)
{
    for(int token = 0; token < t; ++token)
    {
        for(int head = 0; head < heads; ++head)
        {
            float norm2 = 0.0f;
            const size_t base =
                (static_cast<size_t>(token) * heads + head) * head_dim;
            for(int d = 0; d < head_dim; ++d)
            {
                const float value =
                    ck_tile::type_convert<float>(tensor.data()[base + d]);
                norm2 += value * value;
            }
            const float inv_norm = 1.0f / std::sqrt(std::max(norm2, 1.0e-12f));
            for(int d = 0; d < head_dim; ++d)
            {
                tensor.data()[base + d] = ck_tile::type_convert<DataType>(
                    ck_tile::type_convert<float>(tensor.data()[base + d]) * inv_norm);
            }
        }
    }
}

std::vector<int64_t> parse_endpoints(const std::string& spec)
{
    std::vector<int64_t> endpoints;
    std::stringstream stream(spec);
    std::string token;
    while(std::getline(stream, token, ','))
    {
        if(!token.empty())
            endpoints.push_back(std::stoll(token));
    }
    return endpoints;
}

template <typename DataType>
int run(const ck_tile::ArgParser& parser)
{
    int t = parser.get_int("t");
    int h_qk = parser.get_int("h");
    int h_v  = parser.get_int("hv");
    const int head_dim = parser.get_int("head_dim");
    auto endpoints = parse_endpoints(parser.get_str("seq_endpoints"));
    const bool is_varlen = parser.get_bool("is_varlen") || !endpoints.empty();
    if(endpoints.empty())
        endpoints.push_back(t);
    else
        t = static_cast<int>(endpoints.back());
    const int num_sequences = is_varlen ? static_cast<int>(endpoints.size()) : 1;
    int64_t previous_endpoint = 0;
    bool endpoints_valid = true;
    for(const int64_t endpoint : endpoints)
    {
        const int64_t length = endpoint - previous_endpoint;
        endpoints_valid &= length > 0;
        previous_endpoint = endpoint;
    }
    if(t <= 0 || !endpoints_valid ||
       head_dim != gdn_example::kSupportedHeadDim ||
       h_qk <= 0 || h_v <= 0 ||
       h_v % h_qk != 0 ||
       (h_v / h_qk != 1 && h_v / h_qk != 2 && h_v / h_qk != 4))
    {
        std::cerr
            << "GDN requires t,h,hv > 0, head_dim=128, hv/h in {1,2,4}; "
               "varlen sequence lengths must be positive\n";
        return 2;
    }
    const bool cp_context = parser.get_bool("cp_context");
    const int cp_world_size = parser.get_int("cp_world_size");
    const int cp_rank = parser.get_int("cp_rank");
    const bool use_qk_l2norm = parser.get_bool("qk_l2norm");
    if(cp_world_size <= 0 || cp_world_size > 8 ||
       cp_rank < 0 || cp_rank >= cp_world_size)
    {
        std::cerr << "cp_world_size must be in [1,8], and cp_rank must be "
                     "in [0,cp_world_size)\n";
        return 3;
    }
    if(cp_context && (is_varlen || t % 64 != 0))
    {
        std::cerr << "cp_context currently requires a fixed sequence with "
                     "T divisible by 64\n";
        return 3;
    }

    const bool use_initial_state = parser.get_bool("has_initial_state");
    const bool output_final_state = parser.get_bool("store_final_state");
    const bool save_new_value = parser.get_bool("save_new_value");
    const bool state_use_g = parser.get_bool("use_g");
    const bool state_use_gk = parser.get_bool("use_gk");
    if(cp_context && state_use_gk)
    {
        std::cerr << "cp_context does not support per-key gates\n";
        return 3;
    }
    const bool state_use_exp2 = parser.get_bool("use_exp2");
    const bool state_transpose = parser.get_bool("transpose_state");
    const bool gate_in_kernel = parser.get_bool("gate_in_kernel");
    const bool has_dt_bias =
        gate_in_kernel && parser.get_bool("has_dt_bias");
    const bool beta_sigmoid_in_kernel =
        parser.get_bool("beta_sigmoid_in_kernel");
    const bool use_random_gates = parser.get_bool("gates");
    const float input_scale = parser.get_float("input_scale");
    const float requested_scale = parser.get_float("scale");
    const float scale = requested_scale == 0.0f ? 1.0f / std::sqrt(static_cast<float>(head_dim))
                                                 : requested_scale;
    const uint32_t seed = parser.get_uint32("seed");
    ck_tile::HostTensor<int64_t> cu_seqlens_host({num_sequences + 1});
    ck_tile::HostTensor<int64_t> chunk_offsets_host({num_sequences + 1});
    cu_seqlens_host.data()[0] = 0;
    chunk_offsets_host.data()[0] = 0;
    for(int sequence = 0; sequence < num_sequences; ++sequence)
    {
        const int64_t endpoint = is_varlen ? endpoints[sequence] : t;
        const int64_t begin = sequence == 0 ? 0 : endpoints[sequence - 1];
        cu_seqlens_host.data()[sequence + 1] = endpoint;
        chunk_offsets_host.data()[sequence + 1] =
            chunk_offsets_host.data()[sequence] +
            ck_tile::integer_divide_ceil(endpoint - begin, int64_t{64});
    }
    const int num_chunks =
        static_cast<int>(chunk_offsets_host.data()[num_sequences]);
    const bool state_cp_eligible =
        std::is_same_v<DataType, ck_tile::bf16_t> && !cp_context &&
        !is_varlen && num_sequences == 1 && t >= 1024 && t % 64 == 0 &&
        h_v <= 16 && state_use_g && !state_use_gk && state_use_exp2 &&
        !state_transpose && save_new_value;
    const int state_cp_groups =
        state_cp_eligible ? gdn_example::select_state_cp_groups(t, h_v) : 0;

    ck_tile::HostTensor<DataType> q_host({t, h_qk, head_dim});
    ck_tile::HostTensor<DataType> k_host({t, h_qk, head_dim});
    ck_tile::HostTensor<DataType> v_host({t, h_v, head_dim});
    ck_tile::HostTensor<float> g_host({t, h_v});
    ck_tile::HostTensor<float> gk_host(
        {state_use_gk ? t : 1, state_use_gk ? h_v : 1, state_use_gk ? head_dim : 1});
    ck_tile::HostTensor<float> beta_host({t, h_v});
    ck_tile::HostTensor<float> initial_state_host({num_sequences, h_v, head_dim, head_dim});
    ck_tile::HostTensor<float> a_log_host({h_v});
    ck_tile::HostTensor<float> dt_bias_host({h_v});

    ck_tile::FillUniformDistribution<DataType>{
        -input_scale, input_scale, seed, true}(q_host);
    ck_tile::FillUniformDistribution<DataType>{
        -input_scale, input_scale, seed + 1, true}(k_host);
    ck_tile::FillUniformDistribution<DataType>{
        -input_scale, input_scale, seed + 2, true}(v_host);
    if(gate_in_kernel)
    {
        ck_tile::FillUniformDistribution<float>{
            -1.0f, 1.0f, seed + 3, true}(g_host);
    }
    else if(use_random_gates)
    {
        ck_tile::FillUniformDistribution<float>{0.0f, 1.0f, seed + 3, true}(g_host);
        for(auto& value : g_host)
            value = -std::log1p(std::exp(-value));
    }
    else
    {
        std::fill(g_host.begin(), g_host.end(), 0.0f);
    }
    if(!use_random_gates)
        std::fill(beta_host.begin(), beta_host.end(), 1.0f);
    else if(beta_sigmoid_in_kernel)
        ck_tile::FillUniformDistribution<float>{
            -1.0f, 1.0f, seed + 4, true}(beta_host);
    else
    {
        ck_tile::FillUniformDistribution<float>{
            0.0f, 1.0f, seed + 4, true}(beta_host);
        for(auto& value : beta_host)
            value = 1.0f / (1.0f + std::exp(-value));
    }
    ck_tile::FillUniformDistribution<float>{
        -input_scale, input_scale, seed + 5, true}(initial_state_host);
    ck_tile::FillUniformDistribution<float>{
        -0.02f, 0.0f, seed + 6, true}(gk_host);
    ck_tile::FillUniformDistribution<float>{
        -2.0f, -1.0f, seed + 7, true}(a_log_host);
    ck_tile::FillUniformDistribution<float>{
        -0.1f, 0.1f, seed + 8, true}(dt_bias_host);

    ck_tile::DeviceMem q_buf(q_host);
    ck_tile::DeviceMem k_buf(k_host);
    ck_tile::DeviceMem v_buf(v_host);
    ck_tile::DeviceMem g_buf(g_host);
    ck_tile::DeviceMem gk_buf(gk_host);
    ck_tile::DeviceMem a_log_buf(a_log_host);
    ck_tile::DeviceMem dt_bias_buf(dt_bias_host);
    ck_tile::DeviceMem beta_buf(beta_host);
    ck_tile::DeviceMem beta_processed_buf(
        beta_sigmoid_in_kernel
            ? static_cast<size_t>(t) * h_v * sizeof(float)
            : 1);
    ck_tile::DeviceMem initial_state_buf(initial_state_host);
    ck_tile::DeviceMem cu_seqlens_buf(cu_seqlens_host);
    ck_tile::DeviceMem chunk_offsets_buf(
        static_cast<size_t>(num_sequences + 1) * sizeof(int64_t));
    ck_tile::DeviceMem chunk_indices_buf(
        static_cast<size_t>(num_chunks) * 2 * sizeof(int64_t));
    ck_tile::DeviceMem g_cum_buf(static_cast<size_t>(t) * h_v * sizeof(float));
    ck_tile::DeviceMem a_buf(
        static_cast<size_t>(t) * h_v * 64 * sizeof(DataType));
    ck_tile::DeviceMem w_buf(
        static_cast<size_t>(t) * h_v * head_dim * sizeof(DataType));
    ck_tile::DeviceMem u_buf(
        static_cast<size_t>(t) * h_v * head_dim * sizeof(DataType));
    ck_tile::DeviceMem h_buf(
        static_cast<size_t>(num_chunks) * h_v * head_dim * head_dim * sizeof(DataType));
    ck_tile::DeviceMem v_new_buf(
        static_cast<size_t>(t) * h_v * head_dim * sizeof(DataType));
    ck_tile::DeviceMem final_state_buf(
        static_cast<size_t>(num_sequences) * h_v * head_dim * head_dim * sizeof(float));
    const size_t state_bytes =
        static_cast<size_t>(num_sequences) * h_v * head_dim * head_dim * sizeof(float);
    ck_tile::DeviceMem state_input_workspace_buf(
        state_transpose ? state_bytes : 1);
    ck_tile::DeviceMem state_final_workspace_buf(
        state_transpose && output_final_state ? state_bytes : 1);
    const size_t state_cp_elements =
        static_cast<size_t>(state_cp_groups) * h_v * head_dim * head_dim;
    ck_tile::DeviceMem state_cp_group_a_buf(
        state_cp_groups > 1 ? state_cp_elements * sizeof(DataType) : 1);
    ck_tile::DeviceMem state_cp_group_b_buf(
        state_cp_groups > 1 ? state_cp_elements * sizeof(DataType) : 1);
    ck_tile::DeviceMem state_cp_group_start_buf(
        state_cp_groups > 1 ? state_cp_elements * sizeof(float) : 1);
    ck_tile::DeviceMem output_buf(
        static_cast<size_t>(t) * h_v * head_dim * sizeof(DataType));
    const size_t qk_bytes =
        static_cast<size_t>(t) * h_qk * head_dim * sizeof(DataType);
    const size_t rstd_bytes =
        static_cast<size_t>(t) * h_qk * sizeof(float);
    ck_tile::DeviceMem q_norm_buf(use_qk_l2norm ? qk_bytes : 1);
    ck_tile::DeviceMem k_norm_buf(use_qk_l2norm ? qk_bytes : 1);
    ck_tile::DeviceMem q_rstd_buf(use_qk_l2norm ? rstd_bytes : 1);
    ck_tile::DeviceMem k_rstd_buf(use_qk_l2norm ? rstd_bytes : 1);

    const size_t cp_summary_bytes =
        static_cast<size_t>(h_v) * head_dim * 256 * sizeof(float);
    const size_t cp_state_bytes =
        static_cast<size_t>(h_v) * head_dim * head_dim * sizeof(float);
    ck_tile::DeviceMem cp_hm_buf(cp_context ? cp_summary_bytes : 1);
    ck_tile::DeviceMem cp_ag_hm_buf(
        cp_context ? cp_summary_bytes * cp_world_size : 1);
    ck_tile::DeviceMem cp_state_buf(cp_context ? cp_state_bytes : 1);

    const void* selected_q = use_qk_l2norm
        ? q_norm_buf.GetDeviceBuffer()
        : q_buf.GetDeviceBuffer();
    const void* selected_k = use_qk_l2norm
        ? k_norm_buf.GetDeviceBuffer()
        : k_buf.GetDeviceBuffer();
    const float* selected_initial_state = cp_context
        ? static_cast<const float*>(cp_state_buf.GetDeviceBuffer())
        : (use_initial_state
               ? static_cast<const float*>(
                     initial_state_buf.GetDeviceBuffer())
               : nullptr);
    const float* selected_beta = beta_sigmoid_in_kernel
        ? static_cast<const float*>(beta_processed_buf.GetDeviceBuffer())
        : static_cast<const float*>(beta_buf.GetDeviceBuffer());

    gdn_example::PrefillArguments args{
        selected_q,
        selected_k,
        v_buf.GetDeviceBuffer(),
        q_buf.GetDeviceBuffer(),
        k_buf.GetDeviceBuffer(),
        q_norm_buf.GetDeviceBuffer(),
        k_norm_buf.GetDeviceBuffer(),
        static_cast<float*>(q_rstd_buf.GetDeviceBuffer()),
        static_cast<float*>(k_rstd_buf.GetDeviceBuffer()),
        static_cast<const float*>(g_buf.GetDeviceBuffer()),
        static_cast<const float*>(gk_buf.GetDeviceBuffer()),
        static_cast<const float*>(a_log_buf.GetDeviceBuffer()),
        has_dt_bias
            ? static_cast<const float*>(dt_bias_buf.GetDeviceBuffer())
            : nullptr,
        selected_beta,
        static_cast<const float*>(beta_buf.GetDeviceBuffer()),
        static_cast<float*>(beta_processed_buf.GetDeviceBuffer()),
        selected_initial_state,
        static_cast<const int64_t*>(cu_seqlens_buf.GetDeviceBuffer()),
        static_cast<int64_t*>(chunk_offsets_buf.GetDeviceBuffer()),
        static_cast<int64_t*>(chunk_indices_buf.GetDeviceBuffer()),
        static_cast<float*>(g_cum_buf.GetDeviceBuffer()),
        static_cast<uint16_t*>(a_buf.GetDeviceBuffer()),
        static_cast<uint16_t*>(w_buf.GetDeviceBuffer()),
        static_cast<uint16_t*>(u_buf.GetDeviceBuffer()),
        static_cast<uint16_t*>(h_buf.GetDeviceBuffer()),
        static_cast<uint16_t*>(v_new_buf.GetDeviceBuffer()),
        static_cast<float*>(final_state_buf.GetDeviceBuffer()),
        static_cast<float*>(state_input_workspace_buf.GetDeviceBuffer()),
        static_cast<float*>(state_final_workspace_buf.GetDeviceBuffer()),
        static_cast<uint16_t*>(state_cp_group_a_buf.GetDeviceBuffer()),
        static_cast<uint16_t*>(state_cp_group_b_buf.GetDeviceBuffer()),
        static_cast<float*>(state_cp_group_start_buf.GetDeviceBuffer()),
        state_cp_groups,
        static_cast<uint16_t*>(output_buf.GetDeviceBuffer()),
        static_cast<float*>(cp_hm_buf.GetDeviceBuffer()),
        static_cast<float*>(cp_ag_hm_buf.GetDeviceBuffer()),
        static_cast<float*>(cp_state_buf.GetDeviceBuffer()),
        cp_world_size,
        cp_rank,
        t,
        h_qk,
        h_v,
        head_dim,
        num_sequences,
        num_chunks,
        scale,
        output_final_state,
        is_varlen && num_sequences > 1,
        use_qk_l2norm,
        cp_context,
        std::is_same_v<DataType, ck_tile::bf16_t>,
        cp_context || use_initial_state,
        output_final_state,
        save_new_value,
        state_use_g,
        state_use_gk,
        state_use_exp2,
        state_transpose,
        is_varlen,
        gate_in_kernel,
        has_dt_bias,
        beta_sigmoid_in_kernel};

    const auto prefill = [&](const ck_tile::stream_config& stream) {
        if constexpr(std::is_same_v<DataType, ck_tile::bf16_t>)
            gdn_example::launch_prefill_bf16(args, stream.stream_id_);
        else
            gdn_example::launch_prefill_fp16(args, stream.stream_id_);
    };
    const ck_tile::stream_config stream{
        nullptr, true, 0, parser.get_int("warmup"), parser.get_int("repeat")};
    const float prefill_us =
        ck_tile::launch_kernel(stream, prefill) * 1000.0f;

    const std::string precision =
        std::is_same_v<DataType, ck_tile::bf16_t> ? "bf16" : "fp16";
    std::cout << "GDN Prefill Config: T=" << t << ", H=" << h_qk
              << ", HV=" << h_v << ", D=" << head_dim << ", prec=" << precision
              << ", sequences=" << num_sequences
              << ", chunks=" << num_chunks << '\n';
    std::cout << "has_initial_state=" << (use_initial_state ? 1 : 0)
              << ", store_final_state=" << (output_final_state ? 1 : 0)
              << ", save_new_value=" << (save_new_value ? 1 : 0)
              << ", use_g=" << (state_use_g ? 1 : 0)
              << ", use_gk=" << (state_use_gk ? 1 : 0)
              << ", use_exp2=" << (state_use_exp2 ? 1 : 0)
              << ", transpose_state=" << (state_transpose ? 1 : 0)
              << ", gate_in_kernel=" << (gate_in_kernel ? 1 : 0)
              << ", has_dt_bias=" << (has_dt_bias ? 1 : 0)
              << ", beta_sigmoid_in_kernel="
              << (beta_sigmoid_in_kernel ? 1 : 0)
              << ", qk_l2norm=" << (use_qk_l2norm ? 1 : 0)
              << ", is_varlen=" << (is_varlen ? 1 : 0)
              << ", state_cp_groups=" << state_cp_groups
              << ", prepare_chunk_indices_timed=" << (is_varlen ? 1 : 0)
              << ", cp_context=" << (cp_context ? 1 : 0)
              << ", cp_world_size=" << cp_world_size
              << ", cp_rank=" << cp_rank << '\n';
    std::cout << "gdn_prefill Avg Latency: " << (prefill_us / 1000.0f) << " ms ("
              << prefill_us << " us)\n";
    if(cp_context)
    {
        std::cout
            << "note: cp_context is a functional implementation without "
               "performance optimization; performance is expected to be poor\n";
    }

    if(!parser.get_bool("v"))
    {
        std::cout << "Verification (gdn_prefill): SKIPPED\n";
        return 0;
    }

    ck_tile::DeviceMem ref_output_buf(
        save_new_value
            ? static_cast<size_t>(t) * h_v * head_dim * sizeof(DataType)
            : 1);
    ck_tile::DeviceMem ref_final_state_buf(
        static_cast<size_t>(num_sequences) * h_v * head_dim * head_dim * sizeof(float));
    gdn_reference::launch<DataType>(
        static_cast<const uint16_t*>(args.q),
        static_cast<const uint16_t*>(args.k),
        static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()),
        static_cast<const float*>(g_buf.GetDeviceBuffer()),
        state_use_gk
            ? static_cast<const float*>(gk_buf.GetDeviceBuffer())
            : nullptr,
        nullptr,
        gate_in_kernel
            ? static_cast<const float*>(a_log_buf.GetDeviceBuffer())
            : nullptr,
        has_dt_bias
            ? static_cast<const float*>(dt_bias_buf.GetDeviceBuffer())
            : nullptr,
        args.beta,
        args.initial_state,
        static_cast<const int64_t*>(cu_seqlens_buf.GetDeviceBuffer()),
        save_new_value
            ? static_cast<uint16_t*>(ref_output_buf.GetDeviceBuffer())
            : nullptr,
        output_final_state
            ? static_cast<float*>(ref_final_state_buf.GetDeviceBuffer())
            : nullptr,
        t,
        h_qk,
        h_v,
        num_sequences,
        is_varlen,
        scale,
        use_qk_l2norm,
        false,
        gate_in_kernel,
        has_dt_bias,
        state_use_g,
        state_use_gk,
        false,
        true,
        state_use_exp2,
        state_transpose);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    const double rtol = std::is_same_v<DataType, ck_tile::bf16_t> ? 0.03 : 0.02;
    const double atol = std::is_same_v<DataType, ck_tile::bf16_t> ? 0.03 : 0.02;
    bool valid = true;
    if(save_new_value)
    {
        ck_tile::HostTensor<DataType> output_host({t, h_v, head_dim});
        ck_tile::HostTensor<DataType> ref_output_host({t, h_v, head_dim});
        output_buf.FromDevice(output_host.data());
        ref_output_buf.FromDevice(ref_output_host.data());
        valid = ck_tile::check_err(output_host,
                                   ref_output_host,
                                   std::string("GDN output mismatch"),
                                   rtol,
                                   atol);
    }

    if(output_final_state)
    {
        ck_tile::HostTensor<float> final_state_host(
            {num_sequences, h_v, head_dim, head_dim});
        ck_tile::HostTensor<float> ref_final_state_host(
            {num_sequences, h_v, head_dim, head_dim});
        final_state_buf.FromDevice(final_state_host.data());
        ref_final_state_buf.FromDevice(ref_final_state_host.data());
        valid &= ck_tile::check_err(final_state_host,
                                    ref_final_state_host,
                                    std::string("GDN final state mismatch"),
                                    0.02,
                                    0.02);
    }
    std::cout << "Verification (gdn_prefill): " << (valid ? "PASSED" : "FAILED") << '\n';
    return valid ? 0 : 4;
}

int main(int argc, char* argv[])
{
    auto [parsed, parser] = create_args(argc, argv);
    if(!parsed)
        return 1;
    const std::string precision = parser.get_str("prec");
    if(precision == "bf16")
        return run<ck_tile::bf16_t>(parser);
    if(precision == "fp16")
        return run<ck_tile::fp16_t>(parser);
    std::cerr << "unsupported precision: " << precision << '\n';
    return 1;
}
