// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "gemm_multi_d_f16_v3_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace hcu_f16_multi_d {

struct ProblemSize
{
    int m, n, k;
    int stride_a, stride_b, stride_e;
    int k_batch;
    int stride_d0, stride_d1;
};

struct ExecutionConfig
{
    bool verify = true;
    int init_method = 2;
    bool time_kernel = false;
    bool fused = false;
    int warmup = 5;
    int repeat = 50;
};

template <typename ALayout, typename BLayout>
ProblemSize make_problem(int m, int n, int k, int gap = 0, int k_batch = 1)
{
    return {m, n, k,
            (std::is_same_v<ALayout, Row> ? k : m) + gap,
            (std::is_same_v<BLayout, Row> ? n : k) + gap,
            n + gap, k_batch, n + gap, n + gap + 4};
}

template <typename Layout>
HostTensorDescriptor descriptor(int rows, int cols, int stride)
{
    const std::size_t r = static_cast<std::size_t>(rows);
    const std::size_t c = static_cast<std::size_t>(cols);
    const std::size_t s = static_cast<std::size_t>(stride);
    if constexpr(std::is_same_v<Layout, Row>)
        return HostTensorDescriptor({r, c}, {s, std::size_t{1}});
    else
        return HostTensorDescriptor({r, c}, {std::size_t{1}, s});
}

// Run single invocations so warmup/repeat are explicit and split-K starts from
// fresh output every time. Events exclude host setup and the full-buffer reset;
// the invoker's own split-K reset remains part of the measured operation.
template <typename Invoke, typename Reset>
float execute(const ExecutionConfig& config, Invoke invoke, Reset reset)
{
    if(!config.time_kernel)
    {
        reset();
        invoke();
        return 0;
    }
    for(int i = 0; i < config.warmup; ++i)
    {
        reset();
        invoke();
    }
    hipEvent_t start, stop;
    hip_check_error(hipEventCreate(&start));
    hip_check_error(hipEventCreate(&stop));
    float total_ms = 0;
    for(int i = 0; i < config.repeat; ++i)
    {
        reset();
        hip_check_error(hipEventRecord(start, nullptr));
        invoke();
        hip_check_error(hipEventRecord(stop, nullptr));
        hip_check_error(hipEventSynchronize(stop));
        float elapsed_ms = 0;
        hip_check_error(hipEventElapsedTime(&elapsed_ms, start, stop));
        total_ms += elapsed_ms;
    }
    hip_check_error(hipEventDestroy(start));
    hip_check_error(hipEventDestroy(stop));
    return total_ms / static_cast<float>(config.repeat);
}

template <typename ALayout, typename BLayout, bool Fused>
bool run(const ProblemSize& p, const ExecutionConfig& config)
{
    const int m = p.m, n = p.n, k = p.k;
    using DsLayout = std::conditional_t<Fused, ck::Tuple<Row, Row>, ck::Tuple<>>;
    using DsType = std::conditional_t<Fused, ck::Tuple<F16, F16>, ck::Tuple<>>;
    using Op = std::conditional_t<Fused, ck::tensor_operation::element_wise::AddAddRelu,
                                 PassThrough>;
    using Gemm = DeviceGemm<ALayout, BLayout, DsLayout, DsType, Op>;
    Tensor<F16> a(descriptor<ALayout>(m, k, p.stride_a));
    Tensor<F16> b(descriptor<BLayout>(k, n, p.stride_b));
    Tensor<F16> d0(descriptor<Row>(m, n, p.stride_d0));
    Tensor<F16> d1(descriptor<Row>(m, n, p.stride_d1));
    Tensor<F16> expected(descriptor<Row>(m, n, p.stride_e));
    Tensor<F16> actual(expected.mDesc);
    Tensor<float> c(expected.mDesc);
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < k; ++j)
            a(i,j) = ck::type_convert<F16>(config.init_method == 0 ? 0.0f :
                config.init_method == 1 ? static_cast<float>((i % 5 + j % 5) % 5 - 2) :
                static_cast<float>(((i % 23) * 7 + (j % 23) * 3) % 23 - 11) / 16);
    for(int i = 0; i < k; ++i)
        for(int j = 0; j < n; ++j)
            b(i,j) = ck::type_convert<F16>(config.init_method == 0 ? 0.0f :
                config.init_method == 1 ? static_cast<float>((i % 5 + j % 5) % 5 - 2) :
                static_cast<float>(((i % 19) * 5 + (j % 19) * 11) % 19 - 9) / 16);
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
        {
            d0(i,j) = ck::type_convert<F16>(config.init_method == 0 ? 0.0f :
                static_cast<float>((i % 11 + (j % 11) * 3) % 11 - 5) /
                    (config.init_method == 1 ? 1 : 8));
            d1(i,j) = ck::type_convert<F16>(config.init_method == 0 ? 0.0f :
                static_cast<float>(((i % 7) * 3 + j % 7) % 7 - 3) /
                    (config.init_method == 1 ? 1 : 8));
        }

    ck::DeviceMem ad(a.mDesc.GetElementSpaceSize() * sizeof(F16));
    ck::DeviceMem bd(b.mDesc.GetElementSpaceSize() * sizeof(F16));
    ck::DeviceMem d0d(d0.mDesc.GetElementSpaceSize() * sizeof(F16));
    ck::DeviceMem d1d(d1.mDesc.GetElementSpaceSize() * sizeof(F16));
    ck::DeviceMem ed(actual.mDesc.GetElementSpaceSize() * sizeof(F16));
    ad.ToDevice(a.mData.data());
    bd.ToDevice(b.mData.data());
    d0d.ToDevice(d0.mData.data());
    d1d.ToDevice(d1.mData.data());
    std::array<const void*, Fused ? 2 : 0> ds{};
    std::array<ck::index_t, Fused ? 2 : 0> ds_strides{};
    if constexpr(Fused)
    {
        ds = {d0d.GetDeviceBuffer(), d1d.GetDeviceBuffer()};
        ds_strides = {p.stride_d0, p.stride_d1};
    }
    Gemm gemm;
    auto arg = gemm.MakeArgument(ad.GetDeviceBuffer(), bd.GetDeviceBuffer(), ds,
                                ed.GetDeviceBuffer(), m, n, k,
                                p.stride_a, p.stride_b,
                                ds_strides, p.stride_e, p.k_batch,
                                PassThrough{}, PassThrough{}, Op{});
    if(!gemm.IsSupportedArgument(arg))
    {
        std::cerr << "Unsupported problem: " << m << ',' << n << ',' << k << '\n';
        return false;
    }
    auto invoker = gemm.MakeInvoker();
    const float elapsed_ms = execute(config, [&] {
        invoker.Run(arg, StreamConfig{nullptr, false});
        hip_check_error(hipGetLastError());
    }, [&] { ed.SetZero(); });
    hip_check_error(hipStreamSynchronize(nullptr));

    bool ok = true;
    if(config.verify)
    {
        ed.FromDevice(actual.mData.data());
        using Ref = ck::tensor_operation::host::ReferenceGemm<
            F16, F16, float, float, PassThrough, PassThrough, PassThrough>;
        Ref ref;
        auto ref_arg = ref.MakeArgument(a, b, c, PassThrough{}, PassThrough{}, PassThrough{});
        ref.MakeInvoker().Run(ref_arg);
        for(int i = 0; i < m; ++i)
            for(int j = 0; j < n; ++j)
            {
                float v = c(i,j);
                if constexpr(Fused)
                    v = std::max(0.0f, v + ck::type_convert<float>(d0(i,j)) +
                                          ck::type_convert<float>(d1(i,j)));
                expected(i,j) = ck::type_convert<F16>(v);
            }
        // Split-K rounds each partial and each atomic sum to FP16; the unsplit
        // reference rounds only once. Keep its tolerance separate from fused GEMM.
        const double tolerance = p.k_batch == 1 ? 1e-3 : 5e-3;
        ok = ck::utils::check_err(actual, expected, "f16 MultiD mismatch", tolerance, tolerance);
    }
    if(config.time_kernel && (!std::isfinite(elapsed_ms) || elapsed_ms <= 0))
    {
        std::cerr << "Invalid GPU elapsed time: " << elapsed_ms << '\n';
        return false;
    }
    std::cout << (!config.verify ? "RUN " : ok ? "PASS " : "FAIL ")
              << (std::is_same_v<ALayout, Row> ? 'R' : 'C')
              << (std::is_same_v<BLayout, Row> ? 'R' : 'C') << "R "
              << (Fused ? "add_add_relu " : "gemm ")
              << m << 'x' << n << 'x' << k << " KBatch=" << p.k_batch
              << " strides=" << p.stride_a << ',' << p.stride_b << ',' << p.stride_e
              << " D=" << p.stride_d0 << ',' << p.stride_d1 << '\n';
    if(config.time_kernel && ok)
    {
        const double tflops = 2.0 * m * n * k / (1e9 * elapsed_ms);
        std::cout << "Perf: " << elapsed_ms << " ms, " << tflops
                  << " TFlops, warmup=" << config.warmup << " repeat=" << config.repeat
                  << ", " << gemm.GetTypeString() << '\n';
    }
    return ok;
}

template <typename ALayout, typename BLayout>
int run_examples()
{
    // K loop boundaries, M/N/K tail blocks, and independently strided D tensors.
    const std::array<std::array<int, 4>, 6> problems{{
        {256, 256, 128, 0}, {128, 128, 96, 0}, {2, 4, 96, 0},
        {130, 132, 72, 0}, {258, 260, 256, 8}, {130, 132, 136, 16}}};
    bool ok = true;
    for(const auto& p : problems)
    {
        const auto problem = make_problem<ALayout, BLayout>(p[0], p[1], p[2], p[3]);
        ok = run<ALayout, BLayout, false>(problem, {}) && ok;
        ok = run<ALayout, BLayout, true>(problem, {}) && ok;
    }
    ok = run<ALayout, BLayout, false>(make_problem<ALayout, BLayout>(128, 128, 256, 0, 2), {}) && ok;
    ok = run<ALayout, BLayout, false>(make_problem<ALayout, BLayout>(130, 132, 512, 8, 4), {}) && ok;
    return ok ? 0 : 1;
}

inline void usage(const char* name)
{
    std::cout << "Usage: " << name << " [--help | --test]\n"
              << "       " << name << " verify init time"
              << " [M N K StrideA StrideB StrideE KBatch"
              << " [fused StrideD0 StrideD1 [warmup repeat]]]\n"
              << "verify/time/fused: 0 or 1; init: 0=zeros, 1=integers, 2=fractions\n"
              << "fused=1: E=relu(A*B+D0+D1), requires KBatch=1\n"
              << "Defaults: 256x256x128, packed A/B/E, verify=1, init=2, time=0,"
              << " fused=0, KBatch=1, warmup=5, repeat=50\n"
              << "--test runs the 14-case correctness matrix for this layout.\n"
              << "Perf is GPU invoker time; split-K includes its internal output reset.\n";
}

inline int parse_integer(const char* text)
{
    std::size_t end = 0;
    const std::string value{text};
    const int result = std::stoi(value, &end);
    if(end != value.size())
        throw std::invalid_argument("not an integer: " + value);
    return result;
}

template <typename ALayout, typename BLayout>
int run_example(int argc, char* argv[])
{
    try
    {
        if(argc == 2 && std::string{argv[1]} == "--help")
        {
            usage(argv[0]);
            return 0;
        }
        if(argc == 2 && std::string{argv[1]} == "--test")
            return run_examples<ALayout, BLayout>();
        if(argc != 1 && argc != 4 && argc != 11 && argc != 14 && argc != 16)
            throw std::invalid_argument("unexpected argument count");
        auto p = make_problem<ALayout, BLayout>(256, 256, 128);
        ExecutionConfig config;
        if(argc >= 4)
        {
            const int verify = parse_integer(argv[1]);
            config.init_method = parse_integer(argv[2]);
            const int time = parse_integer(argv[3]);
            if((verify != 0 && verify != 1) || (time != 0 && time != 1) ||
               config.init_method < 0 || config.init_method > 2)
                throw std::invalid_argument("invalid verification, initialization or timing mode");
            config.verify = verify != 0;
            config.time_kernel = time != 0;
        }
        if(argc >= 11)
        {
            p = {parse_integer(argv[4]), parse_integer(argv[5]), parse_integer(argv[6]),
                 parse_integer(argv[7]), parse_integer(argv[8]), parse_integer(argv[9]),
                 parse_integer(argv[10]), parse_integer(argv[9]), parse_integer(argv[9])};
        }
        if(argc >= 14)
        {
            const int fused = parse_integer(argv[11]);
            if(fused != 0 && fused != 1)
                throw std::invalid_argument("fused must be 0 or 1");
            config.fused = fused != 0;
            p.stride_d0 = parse_integer(argv[12]);
            p.stride_d1 = parse_integer(argv[13]);
        }
        if(argc == 16)
        {
            config.warmup = parse_integer(argv[14]);
            config.repeat = parse_integer(argv[15]);
        }
        if(p.m <= 0 || p.n <= 0 || p.k <= 0 || p.k_batch <= 0 || p.k_batch > p.k ||
           config.warmup < 0 || config.repeat <= 0)
            throw std::invalid_argument("dimensions/KBatch/repeat must be positive; warmup >= 0");
        if(p.stride_a < (std::is_same_v<ALayout, Row> ? p.k : p.m) ||
           p.stride_b < (std::is_same_v<BLayout, Row> ? p.n : p.k) ||
           p.stride_e < p.n || p.stride_d0 < p.n || p.stride_d1 < p.n)
            throw std::invalid_argument("leading stride is smaller than the stored dimension");
        if(config.fused && p.k_batch != 1)
            throw std::invalid_argument("fused split-K is not supported by this example");
        return (config.fused ? run<ALayout, BLayout, true>(p, config)
                             : run<ALayout, BLayout, false>(p, config)) ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}

} // namespace hcu_f16_multi_d
