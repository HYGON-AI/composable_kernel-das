// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "universal_gemm_dispatch.hpp"
#include "ck_tile/host.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

namespace {
using ck_tile::index_t;
ck_tile::ArgParser arguments()
{
    ck_tile::ArgParser parser;
    parser.insert("m", "256", "M dimension")
        .insert("n", "256", "N dimension")
        .insert("k", "128", "K dimension")
        .insert("a_layout", "R", "A layout (R)")
        .insert("b_layout", "C", "B layout (R or C)")
        .insert("c_layout", "R", "C layout (R)")
        .insert("stride_a", "0", "A leading stride; 0 uses packed storage")
        .insert("stride_b", "0", "B leading stride; 0 uses packed storage")
        .insert("stride_c", "0", "C leading stride; 0 uses packed storage")
        .insert("prec", "fp16", "fp16 or bf16")
        .insert("pipeline", "compv3", "compv3, compv4 or mem")
        .insert("scheduler", "intrawave", "intrawave or interwave (Mem only)")
        .insert("epilogue", "default", "default or cshuffle")
        .insert("split_k", "1", "K partitions (1)")
        .insert("init", "0", "0 random, 1 linear, 2 ones")
        .insert("v", "1", "0 skip, 1 CPU reference, 2 GPU reference")
        .insert("timer", "gpu", "gpu or cpu")
        .insert("warmup", "0", "warmup iterations")
        .insert("repeat", "0", "timed iterations (0 launches once)")
        .insert("persistent", "0", "persistent mode (0)")
        .insert("flush_cache", "false", "cache flush (false)")
        .insert("rotating_count", "1", "number of buffer sets (1)");
    return parser;
}

UniversalGemmOptions parse(const ck_tile::ArgParser& parser, bool b_row)
{
    const auto pipeline = parser.get_str("pipeline");
    const auto scheduler = parser.get_str("scheduler");
    const auto epilogue = parser.get_str("epilogue");
    const auto timer = parser.get_str("timer");
    if(pipeline != "compv3" && pipeline != "compv4" && pipeline != "mem")
        throw std::invalid_argument("unknown pipeline");
    if(scheduler != "intrawave" && (scheduler != "interwave" || pipeline != "mem"))
        throw std::invalid_argument("interwave is supported only by Mem");
    if(epilogue != "default" && epilogue != "cshuffle")
        throw std::invalid_argument("unknown epilogue");
    if(timer != "gpu" && timer != "cpu")
        throw std::invalid_argument("unknown timer");
    if(parser.get_int("split_k") != 1 || parser.get_int("persistent") != 0 ||
       parser.get_bool("flush_cache") || parser.get_int("rotating_count") != 1)
        throw std::invalid_argument("requires split_k=1, persistent=0, flush_cache=false and rotating_count=1");

    const index_t m = parser.get_int("m");
    const index_t n = parser.get_int("n");
    const index_t k = parser.get_int("k");
    const auto stride = [&](const char* name, index_t packed) {
        const index_t value = parser.get_int(name);
        if(value != 0 && value < packed)
            throw std::invalid_argument(std::string("invalid leading stride ") + name);
        return value == 0 ? packed : value;
    };
    const int verify = parser.get_int("v");
    const int init = parser.get_int("init");
    const int warmup = parser.get_int("warmup");
    const int repeat = parser.get_int("repeat");
    if(m <= 0 || n <= 0 || k <= 0 || verify < 0 || verify > 2 || init < 0 || init > 2 ||
       warmup < 0 || repeat < 0)
        throw std::invalid_argument("invalid dimensions, verification, initialization or timing count");

    const auto kind = pipeline == "compv3" ? Pipeline::CompV3 :
                      pipeline == "compv4" ? Pipeline::CompV4 : Pipeline::Mem;
    return {m, n, k,
            stride("stride_a", k), stride("stride_b", b_row ? n : k), stride("stride_c", n),
            init, verify, kind,
            scheduler == "intrawave" ? Scheduler::Intrawave : Scheduler::Interwave,
            epilogue == "cshuffle",
            ck_tile::stream_config{nullptr, repeat > 0, 0, warmup, repeat, timer == "gpu"}};
}

template <typename T, typename BLayout>
__global__ void reference_on_device(const T* a,
                                    const T* b,
                                    T* c,
                                    index_t m,
                                    index_t n,
                                    index_t k,
                                    index_t lda,
                                    index_t ldb,
                                    index_t ldc)
{
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto row = i / n;
    const auto col = i % n;
    if(row >= static_cast<std::size_t>(m))
        return;
    float sum = 0;
    for(index_t x = 0; x < k; ++x)
    {
        const auto bi = std::is_same_v<BLayout, Row> ? x * ldb + col : col * ldb + x;
        sum += ck_tile::type_convert<float>(a[row * lda + x]) *
               ck_tile::type_convert<float>(b[bi]);
    }
    c[row * ldc + col] = ck_tile::type_convert<T>(sum);
}

template <typename T, typename BLayout>
bool run(const UniversalGemmOptions& options)
{
    using namespace ck_tile;
    const auto descriptor = [](index_t rows, index_t cols, index_t stride, bool row) {
        return HostTensorDescriptor({static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)},
                                    {static_cast<std::size_t>(row ? stride : 1),
                                     static_cast<std::size_t>(row ? 1 : stride)});
    };
    const auto [m, n, k, lda, ldb, ldc, init, verify, pipeline, scheduler, shuffle, stream] = options;
    const int initialization = init;
    HostTensor<T> a(descriptor(m, k, lda, true));
    HostTensor<T> b(descriptor(k, n, ldb, std::is_same_v<BLayout, Row>));
    const auto c_descriptor = descriptor(m, n, ldc, true);
    HostTensor<T> output(c_descriptor);
    std::mt19937 random(119);
    std::uniform_int_distribution<int> values(-8, 8);
    const auto fill = [&](auto& tensor, int offset) {
        for(std::size_t i = 0; i < tensor.size(); ++i)
        {
            const int value = initialization == 0 ? values(random) :
                              initialization == 1 ? static_cast<int>((i + offset) % 17) - 8 : 8;
            tensor.data()[i] = type_convert<T>(value * 0.125f);
        }
    };
    fill(a, 0);
    fill(b, 5);
    const auto sentinel = type_convert<T>(64.0f);
    std::fill(output.begin(), output.end(), sentinel);

    DeviceMem a_device(a.get_element_space_size_in_bytes());
    DeviceMem b_device(b.get_element_space_size_in_bytes());
    DeviceMem c_device(output.get_element_space_size_in_bytes());
    a_device.ToDevice(a.data());
    b_device.ToDevice(b.data());
    c_device.ToDevice(output.data());
    const UniversalGemmKernelArgs<> args{{a_device.GetDeviceBuffer()},
                                         {b_device.GetDeviceBuffer()},
                                         {},
                                         c_device.GetDeviceBuffer(),
                                         m, n, k, {lda}, {ldb}, {}, ldc, 1};
    const auto ms = shuffle ? launch<T, BLayout, true>(options, args) :
                             launch<T, BLayout, false>(options, args);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    if(stream.time_kernel_)
        std::cout << "time_ms=" << ms << '\n';
    if(verify == 0)
        return true;

    c_device.FromDevice(output.data());
    HostTensor<T> reference(c_descriptor);
    if(verify == 1)
        reference_gemm<T, T, float, T>(a, b, reference);
    else
    {
        DeviceMem reference_device(reference.get_element_space_size_in_bytes());
        constexpr unsigned int threads = 256;
        const dim3 grid((static_cast<std::size_t>(m) * n + threads - 1) / threads);
        reference_on_device<T, BLayout><<<grid, threads>>>(
            static_cast<const T*>(a_device.GetDeviceBuffer()),
            static_cast<const T*>(b_device.GetDeviceBuffer()),
            static_cast<T*>(reference_device.GetDeviceBuffer()),
            m, n, k, lda, ldb, ldc);
        HIP_CHECK_ERROR(hipGetLastError());
        reference_device.FromDevice(reference.data());
    }

    constexpr float atol = 1e-3f;
    constexpr float rtol = 1e-3f;
    float max_error = 0;
    std::size_t mismatches = 0;
    for(std::size_t i = 0; i < output.size(); ++i)
    {
        const float actual = type_convert<float>(output.data()[i]);
        const float expected = type_convert<float>(i % ldc < static_cast<std::size_t>(n) ?
                                                       reference.data()[i] : sentinel);
        const float error = std::abs(actual - expected);
        max_error = std::max(max_error, error);
        mismatches += !std::isfinite(actual) || !std::isfinite(expected) ||
                      error > atol + rtol * std::abs(expected);
    }
    std::cout << "max_abs_error=" << max_error << " mismatches=" << mismatches
              << " atol=" << atol << " rtol=" << rtol << '\n';
    return mismatches == 0;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto parser = arguments();
        if(!parser.parse(argc, argv))
            return EXIT_FAILURE;
        const auto b_layout = parser.get_str("b_layout");
        const auto precision = parser.get_str("prec");
        if(parser.get_str("a_layout") != "R" || parser.get_str("c_layout") != "R" ||
           (b_layout != "R" && b_layout != "C"))
            throw std::invalid_argument("supported layouts are RRR and RCR");
        if(precision != "fp16" && precision != "bf16")
            throw std::invalid_argument("supported precisions are fp16 and bf16");
        const bool b_row = b_layout == "R";
        const auto options = parse(parser, b_row);
        if(precision == "fp16")
            return (b_row ? run<ck_tile::half_t, Row>(options)
                          : run<ck_tile::half_t, Col>(options)) ? EXIT_SUCCESS : EXIT_FAILURE;
        return (b_row ? run<ck_tile::bf16_t, Row>(options)
                      : run<ck_tile::bf16_t, Col>(options)) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch(const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
