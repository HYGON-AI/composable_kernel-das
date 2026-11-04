// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp"

namespace splitk_example {
using namespace ck;
using Row  = tensor_layout::gemm::RowMajor;
using Col  = tensor_layout::gemm::ColumnMajor;
using Pass = tensor_operation::element_wise::PassThrough;
template <index_t... X>
using S = Sequence<X...>;
template <typename T, typename AL = Row, typename BL = Col, index_t V = 8>
using Ordinary = tensor_operation::device::DeviceGroupedGemmXdlSplitKCShuffle<
    AL,
    BL,
    Tuple<>,
    Row,
    T,
    T,
    float,
    T,
    Tuple<>,
    T,
    Pass,
    Pass,
    Pass,
    tensor_operation::device::GemmSpecialization::MNKPadding,
    1,
    256,
    64,
    128,
    32,
    8,
    8,
    16,
    16,
    2,
    4,
    S<1, 4, 64, 1>,
    S<0, 2, 1, 3>,
    S<0, 2, 1, 3>,
    is_same_v<AL, Row> ? 3 : 2,
    V,
    8,
    true,
    S<1, 4, 64, 1>,
    S<0, 2, 1, 3>,
    S<0, 2, 1, 3>,
    is_same_v<BL, Col> ? 3 : 2,
    V,
    8,
    true,
    1,
    1,
    S<1, 32, 1, 8>,
    4>;

template <typename T,
          typename E        = T,
          typename Ds       = Tuple<>,
          typename DL       = Tuple<>,
          typename Epilogue = Pass,
          typename AL       = Row,
          typename BL       = Col,
          index_t V         = 8>
using TwoStage = tensor_operation::device::DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage<
    AL,
    BL,
    DL,
    Row,
    T,
    T,
    float,
    float,
    Ds,
    E,
    Pass,
    Pass,
    Epilogue,
    tensor_operation::device::GemmSpecialization::MNKPadding,
    1,
    256,
    64,
    128,
    32,
    8,
    8,
    16,
    16,
    2,
    4,
    S<1, 4, 64, 1>,
    S<0, 2, 1, 3>,
    S<0, 2, 1, 3>,
    is_same_v<AL, Row> ? 3 : 2,
    V,
    8,
    true,
    S<1, 4, 64, 1>,
    S<0, 2, 1, 3>,
    S<0, 2, 1, 3>,
    is_same_v<BL, Col> ? 3 : 2,
    V,
    8,
    true,
    1,
    1,
    S<1, 32, 1, 8>,
    4>;

inline size_t Offset(bool row, index_t i, index_t j, index_t stride)
{ return row ? size_t(i) * stride + j : size_t(j) * stride + i; }

template <typename Op,
          typename T,
          typename E   = T,
          typename AL  = Row,
          typename BL  = Col,
          index_t NumD = 0>
bool Run(const std::vector<index_t>& ms,
         const std::vector<index_t>& ns,
         const std::vector<index_t>& ks,
         index_t batch,
         bool padded           = false,
         bool timing           = false,
         bool legacy_buffer    = false,
         bool expect_supported = true,
         bool random_input     = false,
         index_t output_offset = 0)
{
    constexpr bool ar = is_same_v<AL, Row>, br = is_same_v<BL, Row>;
    const size_t groups = ms.size();
    std::minstd_rand rng(1103);
    std::vector<tensor_operation::device::GemmDesc> descs;
    std::vector<const void*> pa, pb;
    std::vector<void*> pe;
    std::vector<std::array<const void*, NumD>> pd(groups);
    std::vector<std::vector<T>> ha(groups), hb(groups);
    std::vector<std::vector<E>> he(groups);
    std::vector<std::array<std::vector<T>, NumD>> hd(groups);
    std::vector<std::unique_ptr<DeviceMem>> da, db, de;
    std::vector<std::array<std::unique_ptr<DeviceMem>, NumD>> dd(groups);
    for(size_t g = 0; g < groups; ++g)
    {
        const index_t m = ms[g], n = ns[g], k = ks[g];
        const index_t sa = (ar ? k : m) + (padded ? 8 : 0), sb = (br ? n : k) + (padded ? 8 : 0);
        const index_t se = n + (padded ? 12 : 0);
        ha[g].resize(size_t(ar ? m : k) * sa);
        hb[g].resize(size_t(br ? k : n) * sb);
        he[g].assign(size_t(m) * se + 16 + output_offset, type_convert<E>(-13.f));
        for(index_t i = 0; i < m; ++i)
            for(index_t j = 0; j < k; ++j)
                ha[g][Offset(ar, i, j, sa)] =
                    type_convert<T>(random_input ? float(int(rng() % 65) - 32) / 128.f
                                                 : float(int((i * 3 + j + g) % 7) - 3) * 0.125f);
        for(index_t i = 0; i < k; ++i)
            for(index_t j = 0; j < n; ++j)
                hb[g][Offset(br, i, j, sb)] =
                    type_convert<T>(random_input ? float(int(rng() % 65) - 32) / 128.f
                                                 : float(int((i + j * 5 + g) % 9) - 4) * 0.125f);
        da.emplace_back(std::make_unique<DeviceMem>(std::max(size_t(1), ha[g].size() * sizeof(T))));
        db.emplace_back(std::make_unique<DeviceMem>(std::max(size_t(1), hb[g].size() * sizeof(T))));
        de.emplace_back(std::make_unique<DeviceMem>(he[g].size() * sizeof(E)));
        if(!ha[g].empty())
            da.back()->ToDevice(ha[g].data());
        if(!hb[g].empty())
            db.back()->ToDevice(hb[g].data());
        de.back()->ToDevice(he[g].data());
        pa.push_back(da.back()->GetDeviceBuffer());
        pb.push_back(db.back()->GetDeviceBuffer());
        pe.push_back(static_cast<E*>(de.back()->GetDeviceBuffer()) + output_offset);
        std::vector<index_t> sd;
        for(index_t d = 0; d < NumD; ++d)
        {
            sd.push_back(n + (padded ? 4 * (d + 1) : 0));
            hd[g][d].assign(size_t(m) * sd.back() + 16, type_convert<T>(-13.f));
            for(index_t i = 0; i < m; ++i)
                for(index_t j = 0; j < n; ++j)
                    hd[g][d][size_t(i) * sd.back() + j] = type_convert<T>(
                        0.25f * (d + 1) + float((i * 3 + j * 5 + d) % 11 - 5) / 64.f);
            dd[g][d] = std::make_unique<DeviceMem>(hd[g][d].size() * sizeof(T));
            dd[g][d]->ToDevice(hd[g][d].data());
            pd[g][d] = dd[g][d]->GetDeviceBuffer();
        }
        descs.push_back({m, n, k, sa, sb, se, sd});
    }
    Op op;
    auto argument =
        op.MakeArgument(pa, pb, pd, pe, descs, Pass{}, Pass{}, typename Op::ExampleEpilogue{});
    op.SetKBatchSize(&argument, batch);
    std::cout << "CASE dtype="
              << (is_same_v<T, half_t>    ? "fp16"
                  : is_same_v<T, bhalf_t> ? "bf16"
                                          : "fp32")
              << " layout=" << (ar ? 'R' : 'C') << (br ? 'R' : 'C') << " D=" << NumD
              << " k_batch=" << batch << " random=" << random_input;
    for(const auto& d : descs)
    {
        std::cout << " [" << d.M_ << ',' << d.N_ << ',' << d.K_ << ";strides=" << d.stride_A_ << ','
                  << d.stride_B_ << ',' << d.stride_C_;
        for(const auto s : d.stride_Ds_)
            std::cout << ',' << s;
        std::cout << ']';
    }
    std::cout << '\n';
    const bool supported = op.IsSupportedArgument(argument);
    if(supported != expect_supported)
        throw std::runtime_error("Unexpected support decision");
    if(!supported)
    {
        std::cout << "EXPECTED_UNSUPPORTED\n";
        return true;
    }
    DeviceMem args(std::max(size_t(1), op.GetDeviceKernelArgSize(&argument)));
    DeviceMem workspace(std::max(size_t(1), op.GetWorkSpaceSize(&argument)));
    if(!legacy_buffer)
        op.SetDeviceKernelArgs(&argument, args.GetDeviceBuffer());
    op.SetWorkSpacePointer(&argument, workspace.GetDeviceBuffer());
    hipStream_t stream;
    hip_check_error(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto invoker  = op.MakeInvoker();
    float ms_time = 0;
    for(int repeat = 0; repeat < 2; ++repeat)
        ms_time = invoker.Run(argument, StreamConfig{stream, timing, 0, 2, 3});
    hip_check_error(hipStreamSynchronize(stream));
    hip_check_error(hipStreamDestroy(stream));
    double max_error = 0;
    for(size_t g = 0; g < groups; ++g)
    {
        de[g]->FromDevice(he[g].data());
        const auto& d = descs[g];
        for(index_t i = 0; i < d.M_; ++i)
            for(index_t j = 0; j < d.stride_C_; ++j)
            {
                double ref = -13;
                if(j < d.N_)
                {
                    ref = 0;
                    for(index_t k = 0; k < d.K_; ++k)
                        ref += double(type_convert<float>(ha[g][Offset(ar, i, k, d.stride_A_)])) *
                               type_convert<float>(hb[g][Offset(br, k, j, d.stride_B_)]);
                    for(index_t z = 0; z < NumD; ++z)
                        ref += type_convert<float>(hd[g][z][size_t(i) * d.stride_Ds_[z] + j]);
                    ref = type_convert<float>(type_convert<E>(float(ref)));
                }
                double actual =
                    type_convert<float>(he[g][output_offset + size_t(i) * d.stride_C_ + j]);
                if(j >= d.N_ && std::abs(actual + 13.0) > 0.0)
                    throw std::runtime_error("Output stride padding overwritten");
                max_error        = std::max(max_error, std::abs(actual - ref));
                const double tol = is_same_v<E, float> ? 1e-4 : 0.02;
                if(!std::isfinite(actual) || std::abs(actual - ref) > tol + tol * std::abs(ref))
                {
                    std::cerr << "FAIL group=" << g << " i=" << i << " j=" << j
                              << " actual=" << actual << " ref=" << ref << '\n';
                    return false;
                }
            }
        for(index_t i = 0; i < output_offset; ++i)
            if(!(std::abs(type_convert<float>(he[g][i]) + 13.f) <= 0.f))
                throw std::runtime_error("Output prefix guard overwritten");
        for(size_t i = output_offset + size_t(d.M_) * d.stride_C_; i < he[g].size(); ++i)
            if(!(std::abs(type_convert<float>(he[g][i]) + 13.f) <= 0.f))
                throw std::runtime_error("Output guard overwritten");
    }
    std::cout << "PASS groups=" << groups << " k_batch=" << batch << " padded=" << padded
              << " timing=" << timing << " error=" << max_error << " ms=" << ms_time
              << " workspace=" << op.GetWorkSpaceSize(&argument) << " " << op.GetTypeString()
              << '\n';
    return true;
}

template <typename Base, typename Epilogue = Pass>
struct TestOp : Base
{
    using ExampleEpilogue = Epilogue;
};
} // namespace splitk_example
