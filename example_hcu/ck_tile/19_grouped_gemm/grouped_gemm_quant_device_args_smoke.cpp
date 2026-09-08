// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/gemm_quant/grouped_gemm_quant_device_args.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct GroupStorage
{
    GroupStorage(ck_tile::index_t m, ck_tile::index_t n, ck_tile::index_t k)
        : M(m),
          N(n),
          K(k),
          a_host(static_cast<std::size_t>(M * K), ck_tile::type_convert<ck_tile::fp8_t>(1.0f)),
          b_host(static_cast<std::size_t>(K * N), ck_tile::type_convert<ck_tile::fp8_t>(1.0f)),
          c_host(static_cast<std::size_t>(M * N), ck_tile::type_convert<ck_tile::half_t>(0.0f)),
          aq_host(static_cast<std::size_t>(M), 1.0f),
          bq_host(static_cast<std::size_t>(N), 1.0f),
          a_device(a_host.size() * sizeof(ck_tile::fp8_t)),
          b_device(b_host.size() * sizeof(ck_tile::fp8_t)),
          c_device(c_host.size() * sizeof(ck_tile::half_t)),
          aq_device(aq_host.size() * sizeof(float)),
          bq_device(bq_host.size() * sizeof(float))
    {
        a_device.ToDevice(a_host.data());
        b_device.ToDevice(b_host.data());
        c_device.SetZero();
        aq_device.ToDevice(aq_host.data());
        bq_device.ToDevice(bq_host.data());
    }

    ck_tile::QuantGemmTransKernelArg MakeArgs() const
    {
        ck_tile::QuantGroupedGemmKernelArgs kernel_args{
            a_device.GetDeviceBuffer(),
            b_device.GetDeviceBuffer(),
            aq_device.GetDeviceBuffer(),
            bq_device.GetDeviceBuffer(),
            c_device.GetDeviceBuffer(),
            M,
            N,
            K,
            1,
            1,
        // TN: A is column-major [M,K], B is row-major [K,N].
            M,
            N,
            N,
            1,
            1,
            1};
        return ck_tile::QuantGemmTransKernelArg{std::move(kernel_args)};
    }

    bool Check(const std::string& label)
    {
        c_device.FromDevice(c_host.data());
        for(std::size_t i = 0; i < c_host.size(); ++i)
        {
            const float actual = ck_tile::type_convert<float>(c_host[i]);
            if(std::abs(actual - static_cast<float>(K)) > 0.01f)
            {
                std::cerr << label << " mismatch at " << i << ": expected " << K
                          << ", actual " << actual << std::endl;
                return false;
            }
        }
        return true;
    }

    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    std::vector<ck_tile::fp8_t> a_host;
    std::vector<ck_tile::fp8_t> b_host;
    std::vector<ck_tile::half_t> c_host;
    std::vector<float> aq_host;
    std::vector<float> bq_host;
    ck_tile::DeviceMem a_device;
    ck_tile::DeviceMem b_device;
    ck_tile::DeviceMem c_device;
    ck_tile::DeviceMem aq_device;
    ck_tile::DeviceMem bq_device;
};

bool run_case(std::int32_t mode, const char* label)
{
    std::vector<std::unique_ptr<GroupStorage>> groups;
    groups.emplace_back(std::make_unique<GroupStorage>(16, 16, 16));
    groups.emplace_back(std::make_unique<GroupStorage>(16, 16, 32));

    std::vector<ck_tile::QuantGemmTransKernelArg> host_args;
    for(const auto& group : groups)
        host_args.push_back(group->MakeArgs());
    ck_tile::DeviceMem device_args(host_args.size() * sizeof(host_args.front()));
    device_args.ToDevice(host_args.data());

    ck_tile_hcu_quant_grouped_gemm_problem_v1 problem{};
    problem.struct_size = sizeof(problem);
    problem.abi_version = CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    problem.a_data_type = CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E4M3_V1;
    problem.b_data_type = CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E4M3_V1;
    problem.c_data_type = CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP16_V1;
    problem.layout = CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_TN_V1;
    problem.quant_mode = mode;
    problem.group_count = static_cast<std::int32_t>(groups.size());
    problem.problem_flags = CK_TILE_HCU_QUANT_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1;
    problem.k_batch = 1;
    problem.m = {CK_TILE_HCU_QUANT_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, 16};
    problem.n = {CK_TILE_HCU_QUANT_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, 16};
    problem.k = {
        CK_TILE_HCU_QUANT_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1, 0, 32};
    problem.qk_a = 1;
    problem.qk_b = 1;
    problem.stride_aq = 1;
    problem.stride_bq = 1;

    ck_tile_hcu_quant_grouped_gemm_selection_v1 selected{};
    selected.struct_size = sizeof(selected);
    selected.abi_version = CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    int status = ck_tile_hcu_quant_grouped_gemm_select_device_args_v1(&problem, &selected);
    if(status != CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1 ||
       selected.instance_id !=
           CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_PERSISTENT_128X128X32_V1)
    {
        std::cerr << label << " selection failed, status=" << status
                  << ", instance=" << selected.instance_id << std::endl;
        return false;
    }

    ck_tile_hcu_quant_grouped_gemm_launch_v1 launch{};
    launch.struct_size = sizeof(launch);
    launch.abi_version = CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    launch.device_args = device_args.GetDeviceBuffer();
    launch.device_args_kind =
        CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_KIND_QUANT_GEMM_TRANS_V1;
    launch.device_args_stride_bytes = static_cast<std::uint32_t>(
        ck_tile_hcu_quant_grouped_gemm_device_args_size_v1(launch.device_args_kind));
    launch.expected_instance_id = selected.instance_id;

    status = ck_tile_hcu_quant_grouped_gemm_run_device_args_v1(&problem, &launch, nullptr);
    if(status != CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1)
    {
        std::cerr << label << " launch failed, status=" << status << std::endl;
        return false;
    }
    const auto hip_status = hipDeviceSynchronize();
    if(hip_status != hipSuccess)
    {
        std::cerr << label << " HIP failure: " << hipGetErrorString(hip_status) << std::endl;
        return false;
    }

    bool pass = true;
    for(auto& group : groups)
        pass = group->Check(label) && pass;
    std::cout << label << " instance=" << selected.instance_id << " name="
              << selected.instance_name << " " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

bool check_registry_architecture_mask()
{
    ck_tile_hcu_quant_grouped_gemm_candidate_info_v1 candidate{};
    candidate.struct_size = sizeof(candidate);
    candidate.abi_version = CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    const int status       = ck_tile_hcu_quant_grouped_gemm_query_instance_v1(
        CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_PERSISTENT_128X128X32_V1, &candidate);
    const std::uint32_t expected =
        CK_TILE_HCU_QUANT_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1 |
        CK_TILE_HCU_QUANT_GROUPED_GEMM_ARCHITECTURE_MASK_GFX946_V1;
    const bool pass = status == CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1 &&
                      candidate.architecture_mask == expected;
    std::cout << "quant device-args registry architecture_mask="
              << candidate.architecture_mask << " expected=" << expected << " "
              << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

} // namespace

int main()
{
    bool pass = check_registry_architecture_mask();
    pass = run_case(CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_TENSORWISE_V1,
                    "quant device-args TN variable-K tensorwise") &&
           pass;
    pass = run_case(CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_ROWWISE_V1,
                    "quant device-args TN variable-K rowwise") &&
           pass;
    return pass ? 0 : 1;
}
