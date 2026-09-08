// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include "grouped_gemm.hpp"

namespace {

constexpr std::size_t FullReferenceElementLimit = 64 * 1024;

struct Shape
{
    int m;
    int n;
    int k;
};

struct CaseSpec
{
    const char* name;
    int dtype;
    int layout;
    std::vector<Shape> shapes;
    bool variable_m = false;
    bool variable_k = false;
    bool may_have_empty_groups = false;
    std::uint32_t disabled = 0;
    std::uint32_t expected_instance_id = 0;
    bool blas_transposed_device_args = false;
    bool compare_old_c_abi = false;
    std::int64_t k_capacity = 0;
};

template <typename T>
T make_a_value(int group, int row, int reduction)
{
    const float value = static_cast<float>((group * 7 + row * 3 + reduction * 5) % 11 - 5) /
                        16.0f;
    return ck_tile::type_convert<T>(value);
}

template <typename T>
T make_b_value(int group, int reduction, int column)
{
    const float value =
        static_cast<float>((group * 3 + reduction * 2 + column * 7) % 13 - 6) / 16.0f;
    return ck_tile::type_convert<T>(value);
}

template <typename T>
bool equal_value(T actual, T expected, float atol, float rtol)
{
    const float a = ck_tile::type_convert<float>(actual);
    const float e = ck_tile::type_convert<float>(expected);
    return std::abs(a - e) <= atol + rtol * std::abs(e);
}

ck_tile_hcu_grouped_gemm_dimension_v1 common_dimension(std::int64_t extent)
{
    return {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, extent};
}

ck_tile_hcu_grouped_gemm_dimension_v1 device_dimension(std::int64_t extent)
{
    return {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1, 0, extent};
}

template <typename T>
bool run_case(const CaseSpec& spec, const char* dtype_name)
{
    const int group_count = static_cast<int>(spec.shapes.size());
    const char a_layout = spec.layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1 ? 'C' : 'R';
    const char b_layout = spec.layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1 ? 'C' : 'R';
    const T zero = ck_tile::type_convert<T>(0.0f);
    const T sentinel = ck_tile::type_convert<T>(-7.0f);

    std::vector<std::vector<T>> host_as(group_count);
    std::vector<std::vector<T>> host_bs(group_count);
    std::vector<std::vector<T>> host_cs(group_count);
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_as;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_bs;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> device_cs;
    std::vector<int> stride_as(group_count);
    std::vector<int> stride_bs(group_count);
    std::vector<int> stride_cs(group_count);
    std::vector<ck_tile::GemmTransKernelArg> kargs;
    std::vector<ck_tile_hcu_grouped_gemm_desc> old_descs;

    device_as.reserve(group_count);
    device_bs.reserve(group_count);
    device_cs.reserve(group_count);
    kargs.reserve(group_count);
    old_descs.reserve(group_count);

    for(int group = 0; group < group_count; ++group)
    {
        const auto [m, n, k] = spec.shapes[group];
        const int stride_a = a_layout == 'R' ? k : std::max(m, 1);
        const int stride_b = b_layout == 'R' ? n : k;
        const int stride_c = n;
        stride_as[group] = stride_a;
        stride_bs[group] = stride_b;
        stride_cs[group] = stride_c;

        const std::size_t a_elements = std::max<std::size_t>(
            1, static_cast<std::size_t>(a_layout == 'R' ? std::max(m, 1) : k) * stride_a);
        const std::size_t b_elements = std::max<std::size_t>(
            1, static_cast<std::size_t>(b_layout == 'R' ? k : n) * stride_b);
        const std::size_t c_elements =
            std::max<std::size_t>(1, static_cast<std::size_t>(std::max(m, 1)) * stride_c);

        host_as[group].assign(a_elements, zero);
        host_bs[group].assign(b_elements, zero);
        host_cs[group].assign(c_elements, sentinel);

        const auto a_index = [&](int row, int reduction) {
            return a_layout == 'R' ? static_cast<std::size_t>(row) * stride_a + reduction
                                   : static_cast<std::size_t>(reduction) * stride_a + row;
        };
        const auto b_index = [&](int reduction, int column) {
            return b_layout == 'R' ? static_cast<std::size_t>(reduction) * stride_b + column
                                   : static_cast<std::size_t>(column) * stride_b + reduction;
        };
        for(int row = 0; row < m; ++row)
            for(int reduction = 0; reduction < k; ++reduction)
                host_as[group][a_index(row, reduction)] =
                    make_a_value<T>(group, row, reduction);
        for(int reduction = 0; reduction < k; ++reduction)
            for(int column = 0; column < n; ++column)
                host_bs[group][b_index(reduction, column)] =
                    make_b_value<T>(group, reduction, column);

        device_as.push_back(std::make_unique<ck_tile::DeviceMem>(a_elements * sizeof(T)));
        device_bs.push_back(std::make_unique<ck_tile::DeviceMem>(b_elements * sizeof(T)));
        device_cs.push_back(std::make_unique<ck_tile::DeviceMem>(c_elements * sizeof(T)));
        device_as.back()->ToDevice(host_as[group].data());
        device_bs.back()->ToDevice(host_bs[group].data());
        device_cs.back()->ToDevice(host_cs[group].data());

        const void* device_a = device_as.back()->GetDeviceBuffer();
        const void* device_b = device_bs.back()->GetDeviceBuffer();
        void* device_c = device_cs.back()->GetDeviceBuffer();
        old_descs.push_back({device_a,
                             device_b,
                             device_c,
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

        if(spec.blas_transposed_device_args)
        {
            // The selected BF16 TN family consumes C^T = B^T A. The physical
            // A/B/C storage remains the caller's ordinary TN/row-major view.
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{device_b},
                                                                   {device_a},
                                                                   {},
                                                                   device_c,
                                                                   n,
                                                                   m,
                                                                   k,
                                                                   {stride_b},
                                                                   {stride_a},
                                                                   {},
                                                                   n,
                                                                   1});
        }
        else
        {
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{device_a},
                                                                   {device_b},
                                                                   {},
                                                                   device_c,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   {stride_a},
                                                                   {stride_b},
                                                                   {},
                                                                   stride_c,
                                                                   1});
        }
    }

    const auto max_m = std::max_element(spec.shapes.begin(), spec.shapes.end(),
                                        [](const Shape& lhs, const Shape& rhs) {
                                            return lhs.m < rhs.m;
                                        })->m;
    const auto max_k = std::max_element(spec.shapes.begin(), spec.shapes.end(),
                                        [](const Shape& lhs, const Shape& rhs) {
                                            return lhs.k < rhs.k;
                                        })->k;
    ck_tile_hcu_grouped_gemm_problem_v1 problem{};
    problem.struct_size = sizeof(problem);
    problem.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    problem.data_type = spec.dtype;
    problem.layout = spec.layout;
    problem.group_count = group_count;
    problem.problem_flags =
        (spec.variable_m || spec.variable_k ? CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1
                                            : 0u) |
        (spec.may_have_empty_groups
             ? CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1
             : 0u);
    problem.policy_disable_flags = spec.disabled;
    problem.m = spec.variable_m ? device_dimension(static_cast<std::int64_t>(max_m) * group_count)
                                : common_dimension(spec.shapes.front().m);
    problem.n = common_dimension(spec.shapes.front().n);
    problem.k = spec.variable_k
                    ? device_dimension(spec.k_capacity > 0 ? spec.k_capacity : max_k)
                    : common_dimension(spec.shapes.front().k);

    ck_tile_hcu_grouped_gemm_selection_v1 selected{};
    selected.struct_size = sizeof(selected);
    selected.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    int status = ck_tile_hcu_grouped_gemm_select_device_args_v1(&problem, &selected);
    if(status != 0 || selected.instance_id != spec.expected_instance_id)
    {
        std::cerr << "DEVICE_ARGS_CASE name=" << spec.name << " dtype=" << dtype_name
                  << " result=FAIL reason=selection status=" << status
                  << " expected_id=" << spec.expected_instance_id
                  << " actual_id=" << selected.instance_id << std::endl;
        return false;
    }

    ck_tile::DeviceMem device_kargs(kargs.size() * sizeof(ck_tile::GemmTransKernelArg));
    device_kargs.ToDevice(kargs.data());
    ck_tile_hcu_grouped_gemm_launch_v1 launch{};
    launch.struct_size = sizeof(launch);
    launch.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    launch.device_args = device_kargs.GetDeviceBuffer();
    launch.expected_instance_id = selected.instance_id;

    ck_tile_hcu_grouped_gemm_selection_v1 launched{};
    launched.struct_size = sizeof(launched);
    launched.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    status = ck_tile_hcu_grouped_gemm_run_device_args_v1(&problem, &launch, &launched);
    if(status != 0 || launched.instance_id != selected.instance_id)
    {
        std::cerr << "DEVICE_ARGS_CASE name=" << spec.name << " dtype=" << dtype_name
                  << " result=FAIL reason=launch status=" << status << std::endl;
        return false;
    }
    const hipError_t sync_status = hipDeviceSynchronize();
    if(sync_status != hipSuccess)
    {
        std::cerr << "DEVICE_ARGS_CASE name=" << spec.name
                  << " result=FAIL reason=hip_sync error=" << hipGetErrorString(sync_status)
                  << std::endl;
        return false;
    }

    std::vector<std::vector<T>> device_args_outputs(group_count);
    const float atol = spec.dtype == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1 ? 0.08f : 0.35f;
    const float rtol = spec.dtype == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1 ? 0.015f : 0.03f;
    for(int group = 0; group < group_count; ++group)
    {
        const auto [m, n, k] = spec.shapes[group];
        device_cs[group]->FromDevice(host_cs[group].data());
        device_args_outputs[group] = host_cs[group];
        if(m == 0)
        {
            if(std::memcmp(host_cs[group].data(), &sentinel, sizeof(T)) != 0)
            {
                std::cerr << "DEVICE_ARGS_CASE name=" << spec.name
                          << " result=FAIL reason=empty_group_write group=" << group << std::endl;
                return false;
            }
            continue;
        }

        const auto a_index = [&](int row, int reduction) {
            return a_layout == 'R'
                       ? static_cast<std::size_t>(row) * stride_as[group] + reduction
                       : static_cast<std::size_t>(reduction) * stride_as[group] + row;
        };
        const auto b_index = [&](int reduction, int column) {
            return b_layout == 'R'
                       ? static_cast<std::size_t>(reduction) * stride_bs[group] + column
                       : static_cast<std::size_t>(column) * stride_bs[group] + reduction;
        };
        const std::size_t output_count = static_cast<std::size_t>(m) * n;
        const std::size_t check_count =
            output_count <= FullReferenceElementLimit ? output_count : 257;
        for(std::size_t check = 0; check < check_count; ++check)
        {
            const std::size_t linear = output_count <= FullReferenceElementLimit
                                           ? check
                                           : (check * (output_count - 1)) / (check_count - 1);
            const int row = static_cast<int>(linear / n);
            const int column = static_cast<int>(linear % n);
            float reference = 0.0f;
            for(int reduction = 0; reduction < k; ++reduction)
            {
                reference += ck_tile::type_convert<float>(host_as[group][a_index(row, reduction)]) *
                             ck_tile::type_convert<float>(host_bs[group][b_index(reduction, column)]);
            }
            const T expected = ck_tile::type_convert<T>(reference);
            const T actual = host_cs[group][static_cast<std::size_t>(row) * stride_cs[group] + column];
            if(!equal_value(actual, expected, atol, rtol))
            {
                std::cerr << "DEVICE_ARGS_CASE name=" << spec.name
                          << " result=FAIL reason=value group=" << group << " row=" << row
                          << " column=" << column
                          << " expected=" << ck_tile::type_convert<float>(expected)
                          << " actual=" << ck_tile::type_convert<float>(actual) << std::endl;
                return false;
            }
        }
    }

    if(spec.compare_old_c_abi)
    {
        for(int group = 0; group < group_count; ++group)
        {
            std::fill(host_cs[group].begin(), host_cs[group].end(), sentinel);
            device_cs[group]->ToDevice(host_cs[group].data());
        }
        ck_tile::DeviceMem old_workspace(ck_tile_hcu_grouped_gemm_workspace_size(group_count));
        const int old_dtype = spec.dtype == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1
                                  ? CK_TILE_HCU_GROUPED_GEMM_FP16
                                  : CK_TILE_HCU_GROUPED_GEMM_BF16;
        status = ck_tile_hcu_grouped_gemm_run(old_descs.data(),
                                               group_count,
                                               old_dtype,
                                               a_layout,
                                               b_layout,
                                               old_workspace.GetDeviceBuffer(),
                                               nullptr);
        if(status != 0 || hipDeviceSynchronize() != hipSuccess)
        {
            std::cerr << "DEVICE_ARGS_CASE name=" << spec.name
                      << " result=FAIL reason=old_c_abi status=" << status << std::endl;
            return false;
        }
        for(int group = 0; group < group_count; ++group)
        {
            device_cs[group]->FromDevice(host_cs[group].data());
            if(host_cs[group].size() != device_args_outputs[group].size() ||
               std::memcmp(host_cs[group].data(),
                           device_args_outputs[group].data(),
                           host_cs[group].size() * sizeof(T)) != 0)
            {
                std::cerr << "DEVICE_ARGS_CASE name=" << spec.name
                          << " result=FAIL reason=c_abi_device_args_mismatch group=" << group
                          << std::endl;
                return false;
            }
        }
    }

    std::cout << "DEVICE_ARGS_CASE name=" << spec.name << " dtype=" << dtype_name
              << " layout=" << spec.layout << " groups=" << group_count
              << " instance_id=" << selected.instance_id
              << " instance_name=" << selected.instance_name
              << " parity=" << (spec.compare_old_c_abi ? "old_c_abi_exact" : "not_requested")
              << " result=PASS" << std::endl;
    return true;
}

bool run_negative_cases()
{
    ck_tile_hcu_grouped_gemm_problem_v1 problem{};
    problem.struct_size = sizeof(problem);
    problem.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    problem.data_type = CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1;
    problem.layout = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1;
    problem.group_count = 1;
    problem.m = common_dimension(64);
    problem.n = common_dimension(128);
    problem.k = common_dimension(128);
    ck_tile_hcu_grouped_gemm_selection_v1 selected{};
    selected.struct_size = sizeof(selected);
    selected.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    if(ck_tile_hcu_grouped_gemm_select_device_args_v1(&problem, &selected) != 0)
        return false;

    ck_tile::GemmTransKernelArg dummy(ck_tile::UniversalGemmKernelArgs<>{{nullptr},
                                                                         {nullptr},
                                                                         {},
                                                                         nullptr,
                                                                         64,
                                                                         128,
                                                                         128,
                                                                         {128},
                                                                         {128},
                                                                         {},
                                                                         128,
                                                                         1});
    ck_tile::DeviceMem device_args(sizeof(dummy));
    device_args.ToDevice(&dummy);
    ck_tile_hcu_grouped_gemm_launch_v1 launch{};
    launch.struct_size = sizeof(launch);
    launch.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    launch.device_args = device_args.GetDeviceBuffer();
    launch.expected_instance_id = selected.instance_id + 1;
    const int mismatch =
        ck_tile_hcu_grouped_gemm_run_device_args_v1(&problem, &launch, nullptr);

    problem.n.kind = CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1;
    const int unsupported =
        ck_tile_hcu_grouped_gemm_select_device_args_v1(&problem, &selected);
    const bool pass = mismatch == CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INSTANCE_MISMATCH_V1 &&
                      unsupported == CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
    std::cout << "DEVICE_ARGS_NEGATIVE mismatch=" << mismatch
              << " unsupported=" << unsupported << " result=" << (pass ? "PASS" : "FAIL")
              << std::endl;
    return pass;
}

} // namespace

int main(int argc, char* argv[])
{
    bool pmd_correctness = false;
    for(int i = 1; i < argc; ++i)
    {
        if(std::string{argv[i]} == "--pmd-correctness")
            pmd_correctness = true;
        else
        {
            std::cerr << "DEVICE_ARGS_SMOKE unknown_argument=" << argv[i] << std::endl;
            return 2;
        }
    }

    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        return 2;
    const std::string arch = props.gcnArchName;
    const bool gfx936 = arch.rfind("gfx936", 0) == 0;
    const bool gfx938 = arch.rfind("gfx938", 0) == 0;
    const bool gfx946 = arch.rfind("gfx946", 0) == 0;
    if(!gfx936 && !gfx938 && !gfx946)
    {
        std::cerr << "DEVICE_ARGS_SMOKE unsupported_arch=" << arch << std::endl;
        return 2;
    }

    constexpr std::uint32_t disable_bw =
        CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BW_FAMILY_V1;
    constexpr std::uint32_t disable_v3 =
        CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_V3_V1;
    constexpr std::uint32_t disable_dsreadm =
        CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_DSREADM_V1;
    constexpr std::uint32_t disable_mls =
        CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX938_MLS_V1;

    std::vector<CaseSpec> cases{
        {"fp16_nt_regular_parity", 1, 1, {{64, 128, 128}}, false, false, false, 0, 4002, false, true},
        {"fp16_nn_padding", 1, 2, {{65, 129, 129}}, false, false, false, 0, 4003},
        {"fp16_tn_regular", 1, 3, {{64, 128, 128}}, false, false, false, 0, 4002},
    };
    if(!pmd_correctness)
    {
        cases.insert(cases.end(), {
        {"bf16_nt_padding", 2, 1, {{65, 129, 129}}, false, false, false, 0, 4003},
        {"bf16_nn_regular", 2, 2, {{64, 128, 128}}, false, false, false, 0, 4002},
        {"bf16_tn_padding", 2, 3, {{65, 129, 129}}, false, false, false, 0, 4003},
        {"bf16_nt_ragged_empty", 2, 1, {{65, 129, 129}, {0, 129, 129}, {129, 129, 129}}, true, false, true, 0, 4003},
        {"v4_m_only_fallback", 1, 3, {{512, 512, 512}}, false, false, false,
         gfx936 ? disable_bw | disable_v3 : disable_mls, 4000},
        {"v4_full_padding_fallback", 2, 2, {{512, 512, 513}}, false, false, false,
         gfx936 ? disable_bw | disable_v3 : disable_mls, 4001},
        });
    }

    if(gfx936)
    {
        cases.push_back({"gfx936_v3", 1, 1, {{2048, 2048, 2048}}, false, false, false,
                         disable_bw, 2000});
        cases.push_back({"gfx936_dsreadm", 1, 2, {{2048, 2048, 2048}}, false, false, false,
                         disable_bw, 2001});
        cases.push_back({"gfx936_dsreadm_m128_mod_tail", 1, 2, {{1920, 2048, 2048}}, true,
                         false, false, disable_bw, 2001});
        cases.push_back({"gfx936_dsreadm_mixed_m_tail", 1, 2,
                         {{1920, 2048, 2048}, {2048, 2048, 2048}, {2176, 2048, 2048}}, true,
                         false, false, disable_bw, 2001});
        // Instance 2001 selects on K capacity; logical K=192 is three 64-wide
        // blocks (HasHotLoop=false, TailNumber::Odd).
        cases.push_back({"gfx936_dsreadm_k128_tn", 1, 3, {{2048, 2816, 128}}, false, true,
                         false, disable_bw, 2001, false, false, 2048});
        cases.push_back({"gfx936_dsreadm_k192_tn", 1, 3, {{2048, 2816, 192}}, false, true,
                         false, disable_bw, 2001, false, false, 2048});
        cases.push_back({"gfx936_dsreadm_k192_tn_ragged", 1, 3,
                         {{2048, 2816, 192}, {2048, 2816, 128}}, false, true, false,
                         disable_bw, 2001, false, false, 2048});
        cases.push_back({"gfx936_dsreadm_k192_nn", 1, 2, {{2048, 2816, 192}}, false, true,
                         false, disable_bw, 2001, false, false, 2048});
        cases.push_back({"gfx936_v3_default_backward", 2, 2, {{2048, 2048, 2048}}, false,
                         false, false, disable_bw | disable_dsreadm, 2002});
        cases.push_back({"gfx936_bw_selected", 1, 1, {{2048, 2048, 2048}}, false, false,
                         false, 0,
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
                         2000
#else
                         1000
#endif
        });
    }
    else if(gfx938)
    {
        cases.push_back({"gfx938_mls_small", 1, 1, {{512, 512, 512}}, false, false, false,
                         0, 3001});
        cases.push_back({"gfx938_mls_large", 2, 2, {{2048, 2048, 2048}}, false, false,
                         false, 0,
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
                         4000
#else
                         3000
#endif
        });
    }
    if(!pmd_correctness && (gfx936 || gfx938))
    {
        cases.push_back({"bw_bf16_tn_transposed", 2, 3, {{2048, 2048, 2048}}, false, false,
                         false, 0, 1003, true});
        cases.push_back({"bw_bf16_tn_logical_k_tail", 2, 3, {{2048, 2048, 2111}}, false, true,
                         false, 0, 1004, true});
    }

    int passed = 0;
    for(const auto& spec : cases)
    {
        const bool ok = spec.dtype == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1
                            ? run_case<ck_tile::half_t>(spec, "fp16")
                            : run_case<ck_tile::bf16_t>(spec, "bf16");
        passed += ok ? 1 : 0;
    }
    const bool negative_pass = run_negative_cases();
    std::cout << "DEVICE_ARGS_SMOKE arch=" << arch << " passed=" << passed
              << " cases=" << cases.size() << " negative=" << (negative_pass ? 1 : 0)
              << " pmd_correctness=" << (pmd_correctness ? 1 : 0)
              << " result="
              << (passed == static_cast<int>(cases.size()) && negative_pass ? "PASS" : "FAIL")
              << std::endl;
    return passed == static_cast<int>(cases.size()) && negative_pass ? 0 : 1;
}
