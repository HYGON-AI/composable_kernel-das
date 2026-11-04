// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/device_memory.hpp"

namespace wmma_example {

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Spec = ck::tensor_operation::device::GemmSpecialization;
template <ck::index_t... X>
using S = ck::Sequence<X...>;

// The public parameter order and tuning values match upstream example/01_gemm.
// The HCU implementation normalizes them internally to a wave64 XDL/MMAC tile.
template <typename T, typename ALayout = Row, typename BLayout = Col,
          Spec Padding = Spec::MNKPadding, typename Shuffle = float,
          typename AOp = PassThrough, typename BOp = PassThrough, typename COp = PassThrough,
          typename CLayout = Row>
using Gemm = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout, BLayout, CLayout, T, T, T, float, Shuffle, AOp, BOp, COp, Padding,
    1, 128, 64, 128, 64, 2, 16, 16, 2, 4,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, std::is_same_v<ALayout, Row> ? 2 : 1,
    2, 2, true,
    S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, std::is_same_v<BLayout, Col> ? 2 : 1,
    2, 2, true, 1, 1, S<1, 32, 1, 4>, 8>;

inline void hip_check(hipError_t status)
{
    if(status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

struct Stream
{
    hipStream_t value;
    Stream() { hip_check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { static_cast<void>(hipStreamDestroy(value)); }
};

struct Scale
{
    float value = 1;
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        y = ck::type_convert<Y>(ck::type_convert<float>(x) * value);
    }
};

template <typename Layout>
std::size_t offset(int i, int j, int stride)
{
    return std::is_same_v<Layout, Row> ? std::size_t(i) * stride + j
                                     : std::size_t(j) * stride + i;
}

template <typename T, typename ALayout = Row, typename BLayout = Col,
          Spec Padding = Spec::MNKPadding, typename Shuffle = float,
          typename AOp = PassThrough, typename BOp = PassThrough, typename COp = PassThrough>
bool check(int m, int n, int k, int sa, int sb, int sc, bool pointer_api = false,
           bool timed = false, bool verify = true,
           AOp a_op = {}, BOp b_op = {}, COp c_op = {})
{
    using Device = Gemm<T, ALayout, BLayout, Padding, Shuffle, AOp, BOp, COp>;
    if(m <= 0 || n <= 0 || k <= 0 || sa < (std::is_same_v<ALayout, Row> ? k : m) ||
       sb < (std::is_same_v<BLayout, Row> ? n : k) || sc < n)
        throw std::runtime_error("Invalid example dimensions/strides");
    // Nonzero pointer offsets and poison in all padding detect descriptor/store mistakes.
    constexpr int guard = 16;
    const T poison = ck::type_convert<T>(-91.f);
    std::vector<T> a(guard * 2 + std::size_t(std::is_same_v<ALayout, Row> ? m : k) * sa, poison);
    std::vector<T> b(guard * 2 + std::size_t(std::is_same_v<BLayout, Row> ? k : n) * sb, poison);
    std::vector<T> c(guard * 2 + std::size_t(m) * sc, poison);
    for(int i = 0; i < m; ++i)
        for(int p = 0; p < k; ++p)
            a[guard + offset<ALayout>(i, p, sa)] = ck::type_convert<T>(((i * 7 + p * 3) % 17 - 8) / 16.f);
    for(int p = 0; p < k; ++p)
        for(int j = 0; j < n; ++j)
            b[guard + offset<BLayout>(p, j, sb)] = ck::type_convert<T>(((p * 5 + j * 11) % 19 - 9) / 16.f);

    ck::DeviceMem da(a.size() * sizeof(T)), db(b.size() * sizeof(T)), dc(c.size() * sizeof(T));
    Stream stream;
    hip_check(hipMemcpyAsync(da.GetDeviceBuffer(), a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice, stream.value));
    hip_check(hipMemcpyAsync(db.GetDeviceBuffer(), b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice, stream.value));
    hip_check(hipMemcpyAsync(dc.GetDeviceBuffer(), c.data(), c.size() * sizeof(T), hipMemcpyHostToDevice, stream.value));
    auto* pa = static_cast<T*>(da.GetDeviceBuffer()) + guard;
    auto* pb = static_cast<T*>(db.GetDeviceBuffer()) + guard;
    auto* pc = static_cast<T*>(dc.GetDeviceBuffer()) + guard;
    Device gemm;
    float ms;
    if(pointer_api)
    {
        // Exercise virtual dispatch through the public DeviceGemm base interface.
        using Base = ck::tensor_operation::device::DeviceGemm<ALayout, BLayout, Row, T, T, T,
                                                              AOp, BOp, COp>;
        Base& base = gemm;
        auto arg = base.MakeArgumentPointer(pa, pb, pc, m, n, k, sa, sb, sc, a_op, b_op, c_op);
        if(!base.IsSupportedArgument(arg.get()))
            throw std::runtime_error("Unexpected unsupported pointer argument");
        ms = base.MakeInvokerPointer()->Run(arg.get(), StreamConfig{stream.value, timed});
    }
    else
    {
        auto arg = gemm.MakeArgument(pa, pb, pc, m, n, k, sa, sb, sc, a_op, b_op, c_op);
        if(!gemm.IsSupportedArgument(arg))
            throw std::runtime_error("Unexpected unsupported argument");
        ms = gemm.MakeInvoker().Run(arg, StreamConfig{stream.value, timed});
    }
    hip_check(hipGetLastError());
    hip_check(hipMemcpyAsync(c.data(), dc.GetDeviceBuffer(), c.size() * sizeof(T), hipMemcpyDeviceToHost, stream.value));
    hip_check(hipStreamSynchronize(stream.value));

    bool passed = true;
    float max_error = 0;
    if(verify)
    {
        for(int i = 0; i < m; ++i)
            for(int j = 0; j < n; ++j)
            {
                float sum = 0;
                for(int p = 0; p < k; ++p)
                {
                    T av, bv;
                    a_op(av, a[guard + offset<ALayout>(i, p, sa)]);
                    b_op(bv, b[guard + offset<BLayout>(p, j, sb)]);
                    sum += ck::type_convert<float>(av) * ck::type_convert<float>(bv);
                }
                T expected;
                c_op(expected, ck::type_convert<Shuffle>(sum));
                const float ref = ck::type_convert<float>(expected);
                const float got = ck::type_convert<float>(c[guard + std::size_t(i) * sc + j]);
                const float error = std::abs(got - ref);
                max_error = std::max(max_error, error);
                const float tol = std::is_same_v<T, ck::half_t> ? 0.001f : 0.008f;
                passed &= std::isfinite(got) && error <= tol * (1 + std::abs(ref));
            }
        for(std::size_t x = 0; x < c.size(); ++x)
        {
            const bool logical = x >= guard && x < guard + std::size_t(m) * sc &&
                                 (x - guard) % sc < std::size_t(n);
            if(!logical)
                passed &= std::memcmp(&c[x], &poison, sizeof(T)) == 0;
        }
    }
    std::cout << (passed ? (verify ? "PASS " : "RUN ") : "FAIL ")
              << ALayout::name << '/' << BLayout::name << " M=" << m << " N=" << n << " K=" << k
              << " strides=" << sa << ',' << sb << ',' << sc
              << " api=" << (pointer_api ? "base-pointer" : "typed")
              << " max_error=" << max_error;
    if(timed) std::cout << " time_ms=" << ms;
    std::cout << '\n';
    return passed;
}

template <typename T, typename ALayout, typename BLayout>
bool layout_suite()
{
    bool ok = true;
    const int shapes[][3] = {{64, 128, 64}, {128, 256, 192}, {65, 129, 33},
                            {7, 9, 5}, {1, 1, 1}, {32, 48, 513}};
    int i = 0;
    for(const auto& shape : shapes)
    {
        const int m = shape[0], n = shape[1], k = shape[2];
        ok &= check<T, ALayout, BLayout>(m, n, k,
                  (std::is_same_v<ALayout, Row> ? k : m) + 3,
                  (std::is_same_v<BLayout, Row> ? n : k) + 5, n + 7, (++i % 2) == 0);
    }
    return ok;
}

template <typename T>
bool contracts()
{
    using Device = Gemm<T>;
    Device op;
    ck::DeviceMem mem(128 * 256 * sizeof(T));
    auto* ptr = static_cast<T*>(mem.GetDeviceBuffer());
    auto make = [&](int m, int n, int k, int sa, int sb, int sc) {
        return op.MakeArgument(ptr, ptr, ptr, m, n, k, sa, sb, sc, {}, {}, {});
    };
    bool ok = !op.IsSupportedArgument(make(0, 128, 64, 64, 64, 128)) &&
              !op.IsSupportedArgument(make(64, 0, 64, 64, 64, 128)) &&
              !op.IsSupportedArgument(make(64, 128, 0, 64, 64, 128)) &&
              !op.IsSupportedArgument(make(-1, 128, 64, 64, 64, 128)) &&
              !op.IsSupportedArgument(make(64, 128, 64, 63, 64, 128)) &&
              !op.IsSupportedArgument(make(64, 128, 64, 64, 63, 128)) &&
              !op.IsSupportedArgument(make(64, 128, 64, 64, 64, 127));
    auto null_arg = op.MakeArgument(nullptr, ptr, ptr, 64, 128, 64, 64, 64, 128, {}, {}, {});
    ok &= !op.IsSupportedArgument(null_arg);
    ok &= !op.IsSupportedArgument(static_cast<const ck::tensor_operation::device::BaseArgument*>(nullptr));
    bool threw = false;
    try { op.MakeInvoker().Run(null_arg); }
    catch(const std::runtime_error&) { threw = true; }
    ok &= threw;
    using Default = Gemm<T, Row, Col, Spec::Default>;
    Default def;
    auto exact = def.MakeArgument(ptr, ptr, ptr, 64, 128, 64, 64, 64, 128, {}, {}, {});
    ok &= def.IsSupportedArgument(exact);
    // Different specialization arguments must not pass a polymorphic type check.
    ok &= !op.IsSupportedArgument(&exact);
    threw = false;
    try { op.MakeInvoker().Run(&exact); }
    catch(const std::runtime_error&) { threw = true; }
    ok &= threw;
    auto reject = [&](int m, int n, int k) {
        return !def.IsSupportedArgument(def.MakeArgument(ptr, ptr, ptr, m, n, k, k, k, n, {}, {}, {}));
    };
    ok &= reject(63, 128, 64) && reject(64, 127, 64) && reject(64, 128, 63);
    using MN = Gemm<T, Row, Col, Spec::MNPadding>;
    MN mn;
    ok &= mn.IsSupportedArgument(mn.MakeArgument(ptr, ptr, ptr, 7, 9, 64, 64, 64, 9, {}, {}, {}));
    ok &= !mn.IsSupportedArgument(mn.MakeArgument(ptr, ptr, ptr, 7, 9, 63, 63, 63, 9, {}, {}, {}));
    using ColumnC = Gemm<T, Row, Col, Spec::MNKPadding, float, PassThrough, PassThrough, PassThrough, Col>;
    ColumnC col;
    ok &= !col.IsSupportedArgument(col.MakeArgument(ptr, ptr, ptr, 64, 128, 64, 64, 64, 128, {}, {}, {}));
    std::cout << (ok ? "PASS" : "FAIL") << " support/rejection contracts\n";
    return ok;
}

template <typename T>
int run(int argc, char* argv[])
{
    try
    {
        std::cout << "device=" << ck::get_device_name() << " " << Gemm<T>{}.GetTypeString() << '\n';
        if(argc == 2 && std::string(argv[1]) == "--test")
        {
            bool ok = layout_suite<T, Row, Row>();
            ok &= layout_suite<T, Row, Col>();
            ok &= layout_suite<T, Col, Row>();
            ok &= layout_suite<T, Col, Col>();
            ok &= check<T, Row, Col, Spec::Default>(64, 128, 64, 64, 64, 128);
            ok &= check<T, Row, Col, Spec::MNPadding>(7, 9, 64, 67, 69, 16, true);
            // Torch uses the input dtype for CShuffle; retain functor state too.
            ok &= check<T, Row, Col, Spec::MNKPadding, T>(65, 129, 33, 36, 38, 136, true);
            ok &= check<T, Row, Col, Spec::MNKPadding, float, Scale, Scale, Scale>(
                17, 23, 65, 68, 70, 30, false, false, true, Scale{0.5f}, Scale{1.5f}, Scale{0.25f});
            ok &= contracts<T>();
            std::cout << (ok ? "ALL_PASS" : "FAILED") << '\n';
            return ok ? 0 : 1;
        }
        if(argc != 1 && argc != 4 && argc != 10)
        {
            std::cerr << "Usage: " << argv[0] << " [verify init_method time [M N K StrideA StrideB StrideC]] | --test\n"
                      << "Uses deterministic fractional data; init_method is retained for CK CLI compatibility.\n";
            return 2;
        }
        int m = 64, n = 128, k = 64, sa = 64, sb = 64, sc = 128;
        if(argc == 10)
        {
            m = std::stoi(argv[4]); n = std::stoi(argv[5]); k = std::stoi(argv[6]);
            sa = std::stoi(argv[7]); sb = std::stoi(argv[8]); sc = std::stoi(argv[9]);
        }
        return check<T>(m, n, k, sa, sb, sc, false,
                        argc >= 4 && std::stoi(argv[3]) != 0,
                        argc < 4 || std::stoi(argv[1]) != 0) ? 0 : 1;
    }
    catch(const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}

} // namespace wmma_example
