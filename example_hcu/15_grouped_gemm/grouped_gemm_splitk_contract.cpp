// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "grouped_gemm_splitk_common.hpp"

namespace splitk_example {
template <typename T, typename AL, typename BL>
bool Layout()
{
    using Direct = TestOp<Ordinary<T, AL, BL, 1>>;
    using Staged = TestOp<TwoStage<T, T, Tuple<>, Tuple<>, Pass, AL, BL, 1>>;
    bool pass    = Run<Direct, T, T, AL, BL>({16, 32}, {32, 64}, {64, 64}, 1);
    pass &= Run<Staged, T, T, AL, BL>({16, 32}, {32, 64}, {64, 72}, 1);
    pass &= Run<Staged, T, T, AL, BL>({16, 32}, {32, 64}, {256, 512}, 4, true);
    return pass;
}

template <typename T>
bool TypeCases()
{
    bool pass = Layout<T, Row, Col>();
    pass &= Layout<T, Row, Row>();
    pass &= Layout<T, Col, Row>();
    pass &= Layout<T, Col, Col>();
    using Direct = TestOp<Ordinary<T>>;
    using Staged = TestOp<TwoStage<T>>;
    pass &= Run<Direct, T>({16, 16, 16, 16}, {32, 32, 32, 32}, {64, 64, 64, 64}, 1);
    pass &= Run<Staged, T>({16, 16}, {32, 32}, {64, 72}, 1);
    pass &= Run<Direct, T>({17, 63}, {68, 68}, {264, 264}, 1, true);
    pass &= Run<Staged, T>({17, 63}, {68, 68}, {264, 520}, 4, true, true);
    pass &= Run<Direct, T>({0, 16, 0}, {32, 32, 0}, {64, 64, 64}, 1);
    pass &= Run<Staged, T>({0, 16, 0}, {32, 32, 0}, {64, 64, 64}, 1);
    pass &= Run<Direct, T>({}, {}, {}, 1);
    pass &= Run<Staged, T>({0}, {32}, {64}, 4);
    pass &= Run<Direct, T>({16}, {32}, {64}, 1, false, false, true);
    pass &= Run<Direct, T>({16}, {32}, {64}, 4, true, false, false, false);
    pass &= Run<Direct, T>({16}, {32}, {64}, 4, false, false, false, !is_same_v<T, bhalf_t>);
    pass &= Run<Direct, T>({16}, {32}, {257}, 1, false, false, false, false);
    pass &= Run<Staged, T>({16, 16}, {32, 32}, {8, 256}, 1, false, false, false, false);
    pass &= Run<Direct, T>({16}, {67}, {64}, 1, false, false, false, false);
    for(const index_t batch : {1, 2, 4})
        pass &= Run<Staged, T>({64, 128, 192}, {128, 128, 128}, {256, 512, 768}, batch);
    pass &= Run<Direct, T>({17, 63}, {68, 68}, {264, 264}, 1, true, false, false, true, true);
    pass &= Run<Staged, T>({17, 63}, {68, 68}, {264, 520}, 4, true, false, false, true, true, 1);
    return pass;
}

template <typename Op>
void BufferLifecycle()
{
    constexpr index_t m = 16, n = 32, k = 64;
    Op op;
    std::vector<half_t> ha(m * k, type_convert<half_t>(0.25f));
    std::vector<half_t> hb(n * k, type_convert<half_t>(0.5f));
    std::vector<half_t> result(m * n);
    DeviceMem a(ha.size() * sizeof(half_t)), b(hb.size() * sizeof(half_t));
    a.ToDevice(ha.data());
    b.ToDevice(hb.data());
    DeviceMem b1(hb.size() * sizeof(half_t));
    std::fill(hb.begin(), hb.end(), type_convert<half_t>(0.25f));
    b1.ToDevice(hb.data());
    std::vector<const void*> pa{a.GetDeviceBuffer()}, pb{b.GetDeviceBuffer()},
        pb1{b1.GetDeviceBuffer()};
    std::vector<std::array<const void*, 0>> ds(1);
    std::vector<tensor_operation::device::GemmDesc> desc{{m, n, k, k, k, n, {}}};
    DeviceMem e0(m * n * sizeof(half_t)), e1(m * n * sizeof(half_t));
    std::vector<void*> pe0{e0.GetDeviceBuffer()}, pe1{e1.GetDeviceBuffer()};
    auto arg0 = op.MakeArgument(pa, pb, ds, pe0, desc, Pass{}, Pass{}, Pass{});
    auto arg1 = op.MakeArgument(pa, pb1, ds, pe1, desc, Pass{}, Pass{}, Pass{});
    DeviceMem ka0(op.GetDeviceKernelArgSize(&arg0)), ka1(op.GetDeviceKernelArgSize(&arg1));
    DeviceMem ws0(op.GetWorkSpaceSize(&arg0)), ws1(op.GetWorkSpaceSize(&arg1));
    hipStream_t s0, s1;
    hip_check_error(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
    hip_check_error(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    for(const index_t batch : {1, 4, 2, 1})
    {
        op.SetKBatchSize(&arg0, batch);
        op.SetKBatchSize(&arg1, batch);
        // Rebind independent buffers, including the opposite setter order.
        op.SetDeviceKernelArgs(&arg0, ka0.GetDeviceBuffer());
        op.SetWorkSpacePointer(&arg0, ws0.GetDeviceBuffer());
        op.SetWorkSpacePointer(&arg1, ws1.GetDeviceBuffer());
        op.SetDeviceKernelArgs(&arg1, ka1.GetDeviceBuffer());
        op.MakeInvoker().Run(arg0, StreamConfig{s0, false});
        op.MakeInvoker().Run(arg1, StreamConfig{s1, false});
        hip_check_error(hipStreamSynchronize(s0));
        hip_check_error(hipStreamSynchronize(s1));
        for(auto* e : {&e0, &e1})
        {
            e->FromDevice(result.data());
            for(const auto value : result)
                if(!(std::abs(type_convert<float>(value) - (e == &e0 ? 8.f : 4.f)) <= 0.f))
                    throw std::runtime_error("Buffer rebind/two-stream result mismatch");
        }
    }
    hip_check_error(hipStreamDestroy(s0));
    hip_check_error(hipStreamDestroy(s1));
    std::cout << "PASS buffer rebind, k_batch updates and two independent streams\n";
}

void InvalidArguments()
{
    using Op = Ordinary<half_t>;
    Op op;
    std::vector<const void*> a(1, reinterpret_cast<void*>(256)), b = a;
    std::vector<void*> e(1, reinterpret_cast<void*>(256));
    std::vector<std::array<const void*, 0>> ds(1);
    std::vector<tensor_operation::device::GemmDesc> d{{16, 32, 64, 64, 64, 32, {}}};
    auto expect_throw = [](auto run) {
        bool threw = false;
        try
        {
            run();
        }
        catch(const std::exception&)
        {
            threw = true;
        }
        if(!threw)
            throw std::runtime_error("Invalid input was accepted");
    };
    auto arg = op.MakeArgument(a, b, ds, e, d, Pass{}, Pass{}, Pass{});
    expect_throw([&] { op.SetKBatchSize(&arg, 0); });
    expect_throw([&] { op.SetKBatchSize(&arg, -1); });
    expect_throw([&] { op.MakeInvoker().Run(arg); });
    op.SetKBatchSize(&arg, 4);
    op.SetKBatchSize(&arg, 1);
    if(!op.IsSupportedArgument(arg))
        throw std::runtime_error("K batch update did not restore support");
    for(const index_t bad : {index_t(0), index_t(-1), std::numeric_limits<index_t>::max()})
    {
        d[0].K_ = bad;
        expect_throw([&] { op.MakeArgument(a, b, ds, e, d, Pass{}, Pass{}, Pass{}); });
    }
    d[0].K_ = 64;
    a.clear();
    expect_throw([&] { op.MakeArgument(a, b, ds, e, d, Pass{}, Pass{}, Pass{}); });
    a.push_back(reinterpret_cast<void*>(258));
    auto unaligned = op.MakeArgument(a, b, ds, e, d, Pass{}, Pass{}, Pass{});
    if(op.IsSupportedArgument(unaligned))
        throw std::runtime_error("Unaligned pointer accepted");
    a[0]          = reinterpret_cast<void*>(256);
    e[0]          = reinterpret_cast<void*>(258);
    auto scalar_e = TwoStage<half_t>::MakeArgument(a, b, ds, e, d, Pass{}, Pass{}, Pass{});
    if(!TwoStage<half_t>::IsSupportedArgument(scalar_e))
        throw std::runtime_error("Final E incorrectly requires FP32 workspace alignment");
    expect_throw([&] { TwoStage<half_t>::MakeInvoker().Run(scalar_e); });
    using DOp = TwoStage<half_t,
                         float,
                         Tuple<half_t, half_t>,
                         Tuple<Row, Row>,
                         tensor_operation::element_wise::AddAdd>;
    std::vector<std::array<const void*, 2>> dp(1);
    expect_throw([&] {
        DOp::MakeArgument(a, b, dp, e, d, Pass{}, Pass{}, tensor_operation::element_wise::AddAdd{});
    });
    std::cout << "PASS invalid arguments and unbound buffers\n";
}
} // namespace splitk_example

int main(int argc, char** argv)
{
    using namespace splitk_example;
    try
    {
        const std::string type = argc > 1 ? argv[1] : "fp16";
        bool pass;
        if(type == "fp16")
            pass = TypeCases<half_t>();
        else if(type == "bf16")
            pass = TypeCases<bhalf_t>();
        else if(type == "fp32")
            pass = TypeCases<float>();
        else
            throw std::invalid_argument("dtype must be fp16, bf16 or fp32");
        InvalidArguments();
        if(type == "fp16")
        {
            BufferLifecycle<Ordinary<half_t>>();
            BufferLifecycle<TwoStage<half_t>>();
            using AddAdd = tensor_operation::element_wise::AddAdd;
            pass &=
                Run<TestOp<TwoStage<half_t, float, Tuple<half_t, half_t>, Tuple<Row, Row>, AddAdd>,
                           AddAdd>,
                    half_t,
                    float,
                    Row,
                    Col,
                    2>({17, 63}, {68, 68}, {128, 256}, 1, true, true);
            pass &=
                Run<TestOp<TwoStage<half_t, float, Tuple<half_t, half_t>, Tuple<Row, Row>, AddAdd>,
                           AddAdd>,
                    half_t,
                    float,
                    Row,
                    Col,
                    2>({17, 63}, {68, 68}, {256, 512}, 4, true, true);
        }
        return !pass;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
