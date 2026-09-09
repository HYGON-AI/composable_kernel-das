// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"

namespace {

struct ProblemSet
{
    std::vector<int> ms;
    std::vector<int> ns;
    std::vector<int> ks;
};

ProblemSet make_problem_set(int group_count, bool ragged)
{
    ProblemSet problems;
    problems.ms.reserve(group_count);
    problems.ns.reserve(group_count);
    problems.ks.reserve(group_count);

    for(int group = 0; group < group_count; ++group)
    {
        if(ragged)
        {
            problems.ms.push_back(33 + (group * 17) % 47);
            problems.ns.push_back(65 + (group * 19) % 61);
            problems.ks.push_back(67 + (group * 23) % 59);
        }
        else
        {
            problems.ms.push_back(64 * (1 + group % 2));
            problems.ns.push_back(128);
            problems.ks.push_back(64 * (2 + group % 2));
        }
    }
    return problems;
}

int make_stride(int logical_length, bool strided, int group)
{
    if(!strided)
        return logical_length;

    constexpr int vector_alignment = 8;
    const int aligned_length =
        ((logical_length + vector_alignment - 1) / vector_alignment) * vector_alignment;
    return aligned_length + vector_alignment * (1 + group % 3);
}

template <typename DataType>
DataType make_a_value(int group, int m, int k)
{
    const float value = static_cast<float>((group * 7 + m * 3 + k * 5) % 11 - 5) / 8.0f;
    return ck_tile::type_convert<DataType>(value);
}

template <typename DataType>
DataType make_b_value(int group, int k, int n)
{
    const float value = static_cast<float>((group * 3 + k * 2 + n * 7) % 13 - 6) / 8.0f;
    return ck_tile::type_convert<DataType>(value);
}

template <typename DataType>
bool run_c_api_case(int dtype,
                    const char* dtype_name,
                    const char* layout_name,
                    char a_layout,
                    char b_layout,
                    const char* mode,
                    const ProblemSet& problems,
                    bool strided,
                    float atol,
                    float rtol)
{
    const int group_count = static_cast<int>(problems.ms.size());
    const char* dispatch_name =
        std::string(mode) == "ragged_strided" ? "generic_v4_padding" : "generic_v4";
    const DataType zero   = ck_tile::type_convert<DataType>(0.0f);
    const DataType sentinel = ck_tile::type_convert<DataType>(-7.0f);

    std::vector<std::vector<DataType>> host_as(group_count);
    std::vector<std::vector<DataType>> host_bs(group_count);
    std::vector<std::vector<DataType>> host_cs(group_count);
    std::vector<std::vector<DataType>> references(group_count);
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_as;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_bs;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_cs;
    std::vector<ck_tile_hcu_grouped_gemm_desc> descs;

    device_as.reserve(group_count);
    device_bs.reserve(group_count);
    device_cs.reserve(group_count);
    descs.reserve(group_count);

    for(int group = 0; group < group_count; ++group)
    {
        const int m = problems.ms[group];
        const int n = problems.ns[group];
        const int k = problems.ks[group];
        const int stride_a = make_stride(a_layout == 'R' ? k : m, strided, group);
        const int stride_b = make_stride(b_layout == 'R' ? n : k, strided, group);
        const int stride_c = make_stride(n, strided, group);
        const std::size_t a_elements =
            static_cast<std::size_t>(a_layout == 'R' ? m : k) * stride_a;
        const std::size_t b_elements =
            static_cast<std::size_t>(b_layout == 'R' ? k : n) * stride_b;
        const std::size_t c_elements = static_cast<std::size_t>(m) * stride_c;

        host_as[group].assign(a_elements, zero);
        host_bs[group].assign(b_elements, zero);
        host_cs[group].assign(c_elements, sentinel);
        references[group].resize(static_cast<std::size_t>(m) * n);

        auto a_index = [&](int row, int reduction) {
            return a_layout == 'R' ? static_cast<std::size_t>(row) * stride_a + reduction
                                   : static_cast<std::size_t>(reduction) * stride_a + row;
        };
        auto b_index = [&](int reduction, int column) {
            return b_layout == 'R' ? static_cast<std::size_t>(reduction) * stride_b + column
                                   : static_cast<std::size_t>(column) * stride_b + reduction;
        };

        for(int row = 0; row < m; ++row)
            for(int reduction = 0; reduction < k; ++reduction)
                host_as[group][a_index(row, reduction)] =
                    make_a_value<DataType>(group, row, reduction);

        for(int reduction = 0; reduction < k; ++reduction)
            for(int column = 0; column < n; ++column)
                host_bs[group][b_index(reduction, column)] =
                    make_b_value<DataType>(group, reduction, column);

        for(int row = 0; row < m; ++row)
        {
            for(int column = 0; column < n; ++column)
            {
                float accumulator = 0.0f;
                for(int reduction = 0; reduction < k; ++reduction)
                {
                    accumulator +=
                        ck_tile::type_convert<float>(host_as[group][a_index(row, reduction)]) *
                        ck_tile::type_convert<float>(host_bs[group][b_index(reduction, column)]);
                }
                references[group][static_cast<std::size_t>(row) * n + column] =
                    ck_tile::type_convert<DataType>(accumulator);
            }
        }

        device_as.push_back(std::make_unique<ck_tile::DeviceMem>(a_elements * sizeof(DataType)));
        device_bs.push_back(std::make_unique<ck_tile::DeviceMem>(b_elements * sizeof(DataType)));
        device_cs.push_back(std::make_unique<ck_tile::DeviceMem>(c_elements * sizeof(DataType)));
        device_as.back()->ToDevice(host_as[group].data());
        device_bs.back()->ToDevice(host_bs[group].data());
        device_cs.back()->ToDevice(host_cs[group].data());

        descs.push_back({device_as.back()->GetDeviceBuffer(),
                         device_bs.back()->GetDeviceBuffer(),
                         device_cs.back()->GetDeviceBuffer(),
                         1,
                         m,
                         n,
                         k,
                         stride_a,
                         stride_b,
                         stride_c,
                         0,
                         nullptr,
                         nullptr});
    }

    ck_tile::DeviceMem workspace(ck_tile_hcu_grouped_gemm_workspace_size(group_count));
    const int rc = ck_tile_hcu_grouped_gemm_run(descs.data(),
                                                 group_count,
                                                 dtype,
                                                 a_layout,
                                                 b_layout,
                                                 workspace.GetDeviceBuffer(),
                                                 nullptr);
    if(rc != 0)
    {
        std::cerr << "CASE dtype=" << dtype_name << " layout=" << layout_name
                  << " mode=" << mode << " groups=" << group_count
                  << " result=FAIL reason=api_rc rc=" << rc << std::endl;
        return false;
    }

    const hipError_t sync_status = hipDeviceSynchronize();
    if(sync_status != hipSuccess)
    {
        std::cerr << "CASE dtype=" << dtype_name << " layout=" << layout_name
                  << " mode=" << mode << " groups=" << group_count
                  << " result=FAIL reason=hip_sync error=" << hipGetErrorString(sync_status)
                  << std::endl;
        return false;
    }

    for(int group = 0; group < group_count; ++group)
    {
        const int m = problems.ms[group];
        const int n = problems.ns[group];
        const int stride_c = make_stride(n, strided, group);
        device_cs[group]->FromDevice(host_cs[group].data());

        for(int row = 0; row < m; ++row)
        {
            for(int column = 0; column < n; ++column)
            {
                const float actual = ck_tile::type_convert<float>(
                    host_cs[group][static_cast<std::size_t>(row) * stride_c + column]);
                const float expected = ck_tile::type_convert<float>(
                    references[group][static_cast<std::size_t>(row) * n + column]);
                const float tolerance = atol + rtol * std::abs(expected);
                if(std::abs(actual - expected) > tolerance)
                {
                    std::cerr << "CASE dtype=" << dtype_name << " layout=" << layout_name
                              << " mode=" << mode << " groups=" << group_count
                              << " result=FAIL reason=value group=" << group << " row=" << row
                              << " column=" << column << " expected=" << expected
                              << " actual=" << actual << " tolerance=" << tolerance << std::endl;
                    return false;
                }
            }

            for(int column = n; column < stride_c; ++column)
            {
                const auto& padding_value =
                    host_cs[group][static_cast<std::size_t>(row) * stride_c + column];
                if(std::memcmp(&padding_value, &sentinel, sizeof(DataType)) != 0)
                {
                    std::cerr << "CASE dtype=" << dtype_name << " layout=" << layout_name
                              << " mode=" << mode << " groups=" << group_count
                              << " result=FAIL reason=output_padding_overwrite group=" << group
                              << " row=" << row << " column=" << column << std::endl;
                    return false;
                }
            }
        }
    }

    std::cout << "CASE dtype=" << dtype_name << " layout=" << layout_name << " mode=" << mode
              << " groups=" << group_count
              << " api=ck_tile_hcu_grouped_gemm_run dispatch=" << dispatch_name
              << " reference=host_fp32_accumulate_quantized_input_output"
              << " atol=" << atol << " rtol=" << rtol << " result=PASS" << std::endl;
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    bool fp16_only = false;
    for(int i = 1; i < argc; ++i)
    {
        if(std::string{argv[i]} == "--fp16-only")
            fp16_only = true;
        else
        {
            std::cerr << "unknown argument: " << argv[i] << std::endl;
            return 2;
        }
    }

    const std::vector<std::tuple<const char*, char, char>> layouts{
        {"NT", 'R', 'C'}, {"NN", 'R', 'R'}, {"TN", 'C', 'R'}};
    const std::vector<int> group_counts{1, 3, 4, 5, 8, 16};

    int bf16_cases = 0;
    int bf16_passed = 0;
    if(!fp16_only)
    {
        for(const auto& [layout_name, a_layout, b_layout] : layouts)
        {
            for(int group_count : group_counts)
            {
                const ProblemSet aligned = make_problem_set(group_count, false);
                const ProblemSet ragged  = make_problem_set(group_count, true);

                ++bf16_cases;
                bf16_passed += run_c_api_case<ck_tile::bf16_t>(CK_TILE_HCU_GROUPED_GEMM_BF16,
                                                                "bf16",
                                                                layout_name,
                                                                a_layout,
                                                                b_layout,
                                                                "aligned_packed",
                                                                aligned,
                                                                false,
                                                                0.25f,
                                                                0.02f);
                ++bf16_cases;
                bf16_passed += run_c_api_case<ck_tile::bf16_t>(CK_TILE_HCU_GROUPED_GEMM_BF16,
                                                                "bf16",
                                                                layout_name,
                                                                a_layout,
                                                                b_layout,
                                                                "aligned_strided",
                                                                aligned,
                                                                true,
                                                                0.25f,
                                                                0.02f);
                ++bf16_cases;
                bf16_passed += run_c_api_case<ck_tile::bf16_t>(CK_TILE_HCU_GROUPED_GEMM_BF16,
                                                                "bf16",
                                                                layout_name,
                                                                a_layout,
                                                                b_layout,
                                                                "ragged_strided",
                                                                ragged,
                                                                true,
                                                                0.25f,
                                                                0.02f);
            }
        }
    }

    int fp16_cases = 0;
    int fp16_passed = 0;
    for(const auto& [layout_name, a_layout, b_layout] : layouts)
    {
        const ProblemSet aligned = make_problem_set(3, false);
        const ProblemSet ragged  = make_problem_set(3, true);
        ++fp16_cases;
        fp16_passed += run_c_api_case<ck_tile::half_t>(CK_TILE_HCU_GROUPED_GEMM_FP16,
                                                        "fp16",
                                                        layout_name,
                                                        a_layout,
                                                        b_layout,
                                                        "aligned_packed",
                                                        aligned,
                                                        false,
                                                        0.05f,
                                                        0.01f);
        ++fp16_cases;
        fp16_passed += run_c_api_case<ck_tile::half_t>(CK_TILE_HCU_GROUPED_GEMM_FP16,
                                                        "fp16",
                                                        layout_name,
                                                        a_layout,
                                                        b_layout,
                                                        "ragged_strided",
                                                        ragged,
                                                        true,
                                                        0.05f,
                                                        0.01f);
    }

    std::cout << "SUMMARY bf16_passed=" << bf16_passed << " bf16_cases=" << bf16_cases
              << " fp16_passed=" << fp16_passed << " fp16_cases=" << fp16_cases
              << " fp16_only=" << (fp16_only ? 1 : 0) << std::endl;
    return bf16_passed == bf16_cases && fp16_passed == fp16_cases ? 0 : 1;
}
