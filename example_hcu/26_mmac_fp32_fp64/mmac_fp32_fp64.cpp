// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/gpu/warp/mmac_gemm.hpp"
#include "ck/utility/amd_xdlops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

static void hip_check(hipError_t status)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A: row-major [16,K], B: column-major [K,16], C/D: column-major [16,16].
// Each wave computes a complete tile. With vstep=0 and no LTS, lane=m+16*(n%4),
// register=n/4, and each lane supplies one K element from a group of four.
template <typename T, int Api>
__global__ void mmac_tile(const T* a, const T* b, const T* c, T* d, int k_size)
{
    const int lane = threadIdx.x;
    const int row  = lane % 16;
    const int kg   = lane / 16;
    ck::vector_type<T, 4> acc;
    ck::static_for<0, 4, 1>{}([&](auto i) {
        acc.template AsType<T>()(i) = c[lane + 64 * i.value];
    });

    for(int k = 0; k < k_size; k += 4)
    {
        const T av = a[row * k_size + k + kg];
        const T bv = b[row * k_size + k + kg];
        if constexpr(std::is_same_v<T, float>)
        {
            if constexpr(Api == 0)
                ck::intrin_mmac_f32_16x16x4f32(av, bv, acc);
            else if constexpr(Api == 1)
                ck::intrin_mfma_f32_16x16x4f32<16, 16>::Run(av, bv, acc);
            else
                ck::mmac_type<ck::MmacInstr::mmac_f32_16x16x4f32>{}.run(av, bv, acc);
        }
        else
        {
            if constexpr(Api == 0)
                ck::intrin_mmac_16x16x4f64(av, bv, acc);
            else if constexpr(Api == 1)
                ck::intrin_mfma_f64_16x16x4f64<16, 16>::Run(av, bv, acc);
            else
                ck::mmac_type<ck::MmacInstr::mmac_16x16x4f64>{}.run(av, bv, acc);
        }
    }

    ck::static_for<0, 4, 1>{}([&](auto i) {
        d[lane + 64 * i.value] = acc.template AsType<T>()[i];
    });
}

template <typename T, int Api>
static bool run_case(int k_size, int pattern)
{
    std::vector<T> a(16 * k_size), b(16 * k_size), c(256), d(256);
    std::mt19937 rng(20260907 + k_size + pattern);
    std::uniform_int_distribution<int> values(-32, 32);
    auto sample = [&](int i) {
        T value = static_cast<T>(values(rng)) / 16;
        // Detect accidental FP32 evaluation in the FP64 path as well as layout errors.
        if constexpr(std::is_same_v<T, double>)
            value += std::ldexp(static_cast<double>(i + 1), -30);
        return value;
    };
    for(int i = 0; i < 16 * k_size; ++i)
    {
        a[i] = pattern == 0 ? T{0}
                           : pattern == 1 ? static_cast<T>((i / k_size) % 4 == i % k_size)
                                          : sample(i);
        b[i] = sample(i + 1);
    }
    for(int i = 0; i < 256; ++i)
        c[i] = sample(i + 2);

    T *da = nullptr, *db = nullptr, *dc = nullptr, *dd = nullptr;
    hip_check(hipMalloc(&da, a.size() * sizeof(T)));
    hip_check(hipMalloc(&db, b.size() * sizeof(T)));
    hip_check(hipMalloc(&dc, c.size() * sizeof(T)));
    hip_check(hipMalloc(&dd, d.size() * sizeof(T)));
    hip_check(hipMemcpy(da, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice));
    hip_check(hipMemcpy(db, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice));
    hip_check(hipMemcpy(dc, c.data(), c.size() * sizeof(T), hipMemcpyHostToDevice));
    hipLaunchKernelGGL((mmac_tile<T, Api>), dim3(1), dim3(64), 0, 0, da, db, dc, dd, k_size);
    hip_check(hipGetLastError());
    hip_check(hipDeviceSynchronize());
    hip_check(hipMemcpy(d.data(), dd, d.size() * sizeof(T), hipMemcpyDeviceToHost));
    hip_check(hipFree(dd));
    hip_check(hipFree(dc));
    hip_check(hipFree(db));
    hip_check(hipFree(da));

    double max_error = 0;
    int mismatches   = 0;
    for(int n = 0; n < 16; ++n)
        for(int m = 0; m < 16; ++m)
        {
            T expected = c[n * 16 + m];
            for(int k = 0; k < k_size; ++k)
                expected = std::fma(a[m * k_size + k], b[n * k_size + k], expected);
            const double actual = static_cast<double>(d[n * 16 + m]);
            const double error  = std::abs(actual - static_cast<double>(expected));
            const double limit  = std::is_same_v<T, float>
                                      ? 1e-5 + 2e-6 * std::abs(expected)
                                      : 1e-12 + 2e-13 * std::abs(expected);
            if(!std::isfinite(actual) || error > limit)
                ++mismatches;
            max_error = std::max(max_error, error);
        }
    std::printf("%s dtype=%s api=%d K=%d pattern=%d mismatches=%d max_error=%.17g\n",
                mismatches == 0 ? "PASS" : "FAIL",
                std::is_same_v<T, float> ? "fp32" : "fp64",
                Api,
                k_size,
                pattern,
                mismatches,
                max_error);
    return mismatches == 0;
}

template <typename T, int Api>
static int run_suite()
{
    int failures = 0;
    failures += !run_case<T, Api>(4, 0); // Preserve all four nonzero accumulator registers.
    failures += !run_case<T, Api>(4, 1); // Basis rows expose K/lane ownership.
    for(int k : {4, 8, 20, 64})
        failures += !run_case<T, Api>(k, 2);
    return failures;
}

int main()
{
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, 0));
    const std::string arch = properties.gcnArchName;
    if(arch.rfind("gfx936", 0) != 0 && arch.rfind("gfx938", 0) != 0)
    {
        std::fprintf(stderr, "This FP32/FP64 test requires gfx936 or gfx938, got %s\n",
                     arch.c_str());
        return EXIT_FAILURE;
    }
    std::printf("device=%s arch=%s\n", properties.name, arch.c_str());
    int failures = 0;
    failures += run_suite<float, 0>();
    failures += run_suite<float, 1>();
    failures += run_suite<float, 2>();
    failures += run_suite<double, 0>();
    failures += run_suite<double, 1>();
    failures += run_suite<double, 2>();
    std::printf("MMAC_FP32_FP64 %s cases=36 failures=%d\n", failures ? "FAIL" : "PASS", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
