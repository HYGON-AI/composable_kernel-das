// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gdn.hpp"
#include "decode/gdn_decode_launch.hpp"
#include "ck_tile/host/reference/reference_gdn.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

auto create_decode_args(int argc, char* argv[])
{
    ck_tile::ArgParser parser;
    parser.insert("b", "1", "physical batch size; varlen requires b=1")
        .insert("t", "1", "tokens per batch, or total packed tokens for varlen")
        .insert("h", "16", "Q/K heads")
        .insert("hv", "64", "value heads")
        .insert("head_dim", "128", "head/value dimension; only 128 is supported")
        .insert("prec", "bf16", "bf16 or fp16")
        .insert("use_g", "1", "use scalar recurrent gate")
        .insert("use_gk", "0", "use per-key recurrent gate")
        .insert("use_gv", "0", "use per-value recurrent gate")
        .insert("beta_headwise", "1", "use one beta value per value head")
        .insert("use_initial_state", "1", "load the initial recurrent state")
        .insert("store_final_state", "1", "store the final recurrent state")
        .insert("use_qk_l2norm", "0", "normalize Q/K inside the decode kernel")
        .insert("use_exp2", "0", "interpret gate inputs in log2 space")
        .insert("transpose_state", "0", "use transposed state storage")
        .insert("variable_length", "0", "use cu_seqlens metadata")
        .insert("seq_endpoints", "", "comma-separated packed sequence endpoints")
        .insert("gate_in_kernel", "0", "compute scalar decay from dt and A-log")
        .insert("has_dt_bias", "0", "add dt_bias when gate_in_kernel is enabled")
        .insert("warmup", "10", "warmup iterations")
        .insert("repeat", "100", "timed iterations")
        .insert("v", "1", "run the independent GPU reference")
        .insert("seed", "42", "random seed");
    return std::make_tuple(parser.parse(argc, argv), parser);
}

std::vector<int64_t> parse_decode_endpoints(const std::string& spec)
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
int run_decode(const ck_tile::ArgParser& parser)
{
    const int b  = parser.get_int("b");
    int t        = parser.get_int("t");
    const int h  = parser.get_int("h");
    const int hv = parser.get_int("hv");
    const int head_dim = parser.get_int("head_dim");
    auto endpoints = parse_decode_endpoints(parser.get_str("seq_endpoints"));
    const bool variable_length =
        parser.get_bool("variable_length") || !endpoints.empty();
    if(variable_length && b != 1)
    {
        std::cerr
            << "decode varlen uses packed input and requires physical b=1; "
               "logical sequences are defined by seq_endpoints\n";
        return 2;
    }
    if(variable_length && endpoints.empty())
        endpoints.push_back(t);
    if(!variable_length)
    {
        endpoints.clear();
        for(int batch = 1; batch <= b; ++batch)
            endpoints.push_back(static_cast<int64_t>(batch) * t);
    }
    int64_t previous_endpoint = 0;
    bool endpoints_valid = !endpoints.empty();
    for(const int64_t endpoint : endpoints)
    {
        endpoints_valid &= endpoint > previous_endpoint;
        previous_endpoint = endpoint;
    }
    if(variable_length && endpoints_valid)
        t = static_cast<int>(endpoints.back());
    const int sequences = variable_length
        ? static_cast<int>(endpoints.size())
        : b;
    const int total_tokens = variable_length ? t : b * t;
    if(b <= 0 || t <= 0 || h <= 0 || hv <= 0 || hv % h != 0 ||
       !endpoints_valid ||
       head_dim != gdn_decode_example::kSupportedHeadDim)
    {
        std::cerr
            << "decode requires b,t,h,hv > 0, positive sequence lengths, "
               "hv divisible by h, head_dim=128\n";
        return 2;
    }

    const uint32_t seed = parser.get_uint32("seed");
    const bool use_g = parser.get_bool("use_g");
    const bool use_gk = parser.get_bool("use_gk");
    const bool use_gv = parser.get_bool("use_gv");
    const bool beta_headwise = parser.get_bool("beta_headwise");
    const bool use_initial_state = parser.get_bool("use_initial_state");
    const bool store_final_state = parser.get_bool("store_final_state");
    const bool use_qk_l2norm = parser.get_bool("use_qk_l2norm");
    const bool use_exp2 = parser.get_bool("use_exp2");
    const bool transpose_state = parser.get_bool("transpose_state");
    const bool gate_in_kernel = parser.get_bool("gate_in_kernel");
    const bool has_dt_bias = parser.get_bool("has_dt_bias");
    ck_tile::HostTensor<DataType> q_host({total_tokens, h, head_dim});
    ck_tile::HostTensor<DataType> k_host({total_tokens, h, head_dim});
    ck_tile::HostTensor<DataType> v_host({total_tokens, hv, head_dim});
    ck_tile::HostTensor<DataType> beta_host({total_tokens, hv, head_dim});
    ck_tile::HostTensor<float> g_host({total_tokens, hv});
    ck_tile::HostTensor<float> gk_host({total_tokens, hv, head_dim});
    ck_tile::HostTensor<float> gv_host({total_tokens, hv, head_dim});
    ck_tile::HostTensor<float> a_log_host({hv});
    ck_tile::HostTensor<float> dt_bias_host({hv});
    ck_tile::HostTensor<int64_t> cu_seqlens_host({sequences + 1});
    ck_tile::HostTensor<float> initial_host(
        {sequences, hv, head_dim, head_dim});

    ck_tile::FillUniformDistribution<DataType>{
        -0.02f, 0.02f, seed, true}(q_host);
    ck_tile::FillUniformDistribution<DataType>{
        -0.02f, 0.02f, seed + 1, true}(k_host);
    ck_tile::FillUniformDistribution<DataType>{
        -0.02f, 0.02f, seed + 2, true}(v_host);
    ck_tile::FillUniformDistribution<DataType>{
        0.0f, 1.0f, seed + 3, true}(beta_host);
    ck_tile::FillUniformDistribution<float>{
        0.0f, 1.0f, seed + 4, true}(g_host);
    for(auto& value : g_host)
        value = -std::log1p(std::exp(-value));
    ck_tile::FillUniformDistribution<float>{
        -0.01f, 0.01f, seed + 5, true}(initial_host);
    ck_tile::FillUniformDistribution<float>{
        -0.02f, 0.0f, seed + 6, true}(gk_host);
    ck_tile::FillUniformDistribution<float>{
        -0.02f, 0.0f, seed + 7, true}(gv_host);
    ck_tile::FillUniformDistribution<float>{
        -2.0f, -1.0f, seed + 8, true}(a_log_host);
    ck_tile::FillUniformDistribution<float>{
        -0.1f, 0.1f, seed + 9, true}(dt_bias_host);
    cu_seqlens_host.data()[0] = 0;
    for(int sequence = 0; sequence < sequences; ++sequence)
        cu_seqlens_host.data()[sequence + 1] = endpoints[sequence];

    ck_tile::DeviceMem q_buf(q_host);
    ck_tile::DeviceMem k_buf(k_host);
    ck_tile::DeviceMem v_buf(v_host);
    ck_tile::DeviceMem beta_buf(beta_host);
    ck_tile::DeviceMem g_buf(g_host);
    ck_tile::DeviceMem gk_buf(gk_host);
    ck_tile::DeviceMem gv_buf(gv_host);
    ck_tile::DeviceMem a_log_buf(a_log_host);
    ck_tile::DeviceMem dt_bias_buf(dt_bias_host);
    ck_tile::DeviceMem cu_seqlens_buf(cu_seqlens_host);
    ck_tile::DeviceMem initial_buf(initial_host);
    ck_tile::DeviceMem output_buf(
        static_cast<size_t>(total_tokens) * hv * head_dim * sizeof(DataType));
    ck_tile::DeviceMem final_buf(
        static_cast<size_t>(sequences) * hv * head_dim * head_dim * sizeof(float));

    constexpr bool is_bf16 =
        std::is_same_v<DataType, ck_tile::bf16_t>;
    constexpr int data_type_code =
        gdn_tensor_dtype_code(is_bf16 ? GdnTensorDtype::BFloat16
                                      : GdnTensorDtype::Float16);
    constexpr int float32_type_code =
        gdn_tensor_dtype_code(GdnTensorDtype::Float32);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    GdnFusedRecurrentKargs args{
        q_buf.GetDeviceBuffer(),
        k_buf.GetDeviceBuffer(),
        v_buf.GetDeviceBuffer(),
        use_g ? g_buf.GetDeviceBuffer() : nullptr,
        use_gk ? gk_buf.GetDeviceBuffer() : nullptr,
        use_gv ? gv_buf.GetDeviceBuffer() : nullptr,
        beta_buf.GetDeviceBuffer(),
        gate_in_kernel ? a_log_buf.GetDeviceBuffer() : nullptr,
        has_dt_bias ? dt_bias_buf.GetDeviceBuffer() : nullptr,
        use_initial_state
            ? static_cast<const float*>(initial_buf.GetDeviceBuffer())
            : nullptr,
        variable_length
            ? static_cast<const int64_t*>(cu_seqlens_buf.GetDeviceBuffer())
            : nullptr,
        output_buf.GetDeviceBuffer(),
        store_final_state
            ? static_cast<float*>(final_buf.GetDeviceBuffer())
            : nullptr,
        b,
        t,
        h,
        hv,
        head_dim,
        head_dim,
        sequences,
        scale,
        float32_type_code,
        float32_type_code,
        float32_type_code,
        data_type_code,
        float32_type_code,
        float32_type_code,
        use_g,
        use_gk,
        use_gv,
        beta_headwise,
        use_initial_state,
        store_final_state,
        use_qk_l2norm,
        use_exp2,
        transpose_state,
        variable_length,
        gate_in_kernel,
        has_dt_bias};

    const auto kernel = [&](const ck_tile::stream_config& stream) {
        if constexpr(is_bf16)
            gdn_decode_example::launch_bf16(args, stream.stream_id_);
        else
            gdn_decode_example::launch_fp16(args, stream.stream_id_);
    };
    const ck_tile::stream_config stream{
        nullptr, true, 0, parser.get_int("warmup"), parser.get_int("repeat")};
    const float elapsed_us =
        ck_tile::launch_kernel(stream, kernel) * 1000.0f;

    std::cout << "GDN Decode Config: B=" << b << ", T=" << t
              << ", H=" << h << ", HV=" << hv
              << ", D=" << head_dim << ", prec=" << (is_bf16 ? "bf16" : "fp16")
              << ", sequences=" << sequences << '\n';
    std::cout << "use_g=" << use_g << ", use_gk=" << use_gk
              << ", use_gv=" << use_gv
              << ", beta_headwise=" << beta_headwise
              << ", use_initial_state=" << use_initial_state
              << ", store_final_state=" << store_final_state
              << ", qk_l2norm=" << use_qk_l2norm
              << ", use_exp2=" << use_exp2
              << ", transpose_state=" << transpose_state
              << ", variable_length=" << variable_length
              << ", gate_in_kernel=" << gate_in_kernel
              << ", has_dt_bias=" << has_dt_bias << '\n';
    std::cout << "gdn_decode Avg Latency: " << (elapsed_us / 1000.0f) << " ms ("
              << elapsed_us << " us)\n";

    if(!parser.get_bool("v"))
    {
        std::cout << "Verification (gdn_decode): SKIPPED\n";
        return 0;
    }

    ck_tile::DeviceMem reference_buf(
        static_cast<size_t>(total_tokens) * hv * head_dim * sizeof(DataType));
    ck_tile::DeviceMem reference_final_buf(
        static_cast<size_t>(sequences) * hv * head_dim * head_dim * sizeof(float));
    gdn_reference::launch<DataType>(
        static_cast<const uint16_t*>(q_buf.GetDeviceBuffer()),
        static_cast<const uint16_t*>(k_buf.GetDeviceBuffer()),
        static_cast<const uint16_t*>(v_buf.GetDeviceBuffer()),
        static_cast<const float*>(g_buf.GetDeviceBuffer()),
        use_gk ? static_cast<const float*>(gk_buf.GetDeviceBuffer()) : nullptr,
        use_gv ? static_cast<const float*>(gv_buf.GetDeviceBuffer()) : nullptr,
        gate_in_kernel
            ? static_cast<const float*>(a_log_buf.GetDeviceBuffer())
            : nullptr,
        has_dt_bias
            ? static_cast<const float*>(dt_bias_buf.GetDeviceBuffer())
            : nullptr,
        beta_buf.GetDeviceBuffer(),
        use_initial_state
            ? static_cast<const float*>(initial_buf.GetDeviceBuffer())
            : nullptr,
        variable_length
            ? static_cast<const int64_t*>(cu_seqlens_buf.GetDeviceBuffer())
            : nullptr,
        static_cast<uint16_t*>(reference_buf.GetDeviceBuffer()),
        store_final_state
            ? static_cast<float*>(reference_final_buf.GetDeviceBuffer())
            : nullptr,
        t,
        h,
        hv,
        sequences,
        variable_length,
        scale,
        use_qk_l2norm,
        true,
        gate_in_kernel,
        has_dt_bias,
        use_g,
        use_gk,
        use_gv,
        beta_headwise,
        use_exp2,
        transpose_state);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    ck_tile::HostTensor<DataType> output_host({total_tokens, hv, head_dim});
    ck_tile::HostTensor<DataType> reference({total_tokens, hv, head_dim});
    ck_tile::HostTensor<float> final_host(
        {sequences, hv, head_dim, head_dim});
    ck_tile::HostTensor<float> reference_final(
        {sequences, hv, head_dim, head_dim});
    output_buf.FromDevice(output_host.data());
    reference_buf.FromDevice(reference.data());
    if(store_final_state)
    {
        final_buf.FromDevice(final_host.data());
        reference_final_buf.FromDevice(reference_final.data());
    }
    bool valid = ck_tile::check_err(
        output_host,
        reference,
        std::string("GDN decode output mismatch"),
        is_bf16 ? 0.05 : 0.03,
        is_bf16 ? 0.05 : 0.03);
    if(store_final_state)
        valid &= ck_tile::check_err(final_host,
                                    reference_final,
                                    std::string("GDN decode state mismatch"),
                                    0.02,
                                    0.02);
    std::cout << "Verification (gdn_decode): " << (valid ? "PASSED" : "FAILED") << '\n';
    return valid ? 0 : 4;
}

int main(int argc, char* argv[])
{
    auto [result, parser] = create_decode_args(argc, argv);
    if(!result)
        return 1;
    const auto precision = parser.get_str("prec");
    if(precision == "bf16")
        return run_decode<ck_tile::bf16_t>(parser);
    if(precision == "fp16")
        return run_decode<ck_tile::half_t>(parser);
    std::cerr << "prec must be bf16 or fp16\n";
    return 2;
}
