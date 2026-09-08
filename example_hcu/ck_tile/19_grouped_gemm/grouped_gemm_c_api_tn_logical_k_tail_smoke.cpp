// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"

namespace {

using DataType = ck_tile::bf16_t;

constexpr int GroupCount = 16;
constexpr int M          = 2048;
constexpr int N          = 2048;
constexpr std::size_t GuardElements = 256;
constexpr std::array<int, GroupCount> GroupK = {
    2102, 2120, 2108, 2092, 2079, 2047, 1966, 2048,
    2015, 2006, 2017, 2014, 2050, 2079, 2008, 2006};

std::uint64_t fnv1a(const DataType* data, std::size_t elements)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    const std::size_t byte_count = elements * sizeof(DataType);
    std::uint64_t hash = 1469598103934665603ULL;
    for(std::size_t i = 0; i < byte_count; ++i)
    {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct GroupBuffers
{
    std::vector<DataType> host_a;
    std::vector<DataType> host_b;
    std::vector<DataType> host_c;
    std::unique_ptr<ck_tile::DeviceMem> device_a;
    std::unique_ptr<ck_tile::DeviceMem> device_b;
    std::unique_ptr<ck_tile::DeviceMem> device_c;
    DataType a_value;
    DataType b_value;
};

bool run_once(const char* mode,
              DataType input_guard,
              std::vector<GroupBuffers>& groups,
              std::vector<ck_tile_hcu_grouped_gemm_desc>& descs,
              void* workspace,
              std::vector<std::uint64_t>& hashes,
              float& elapsed_ms)
{
    const DataType output_guard = ck_tile::type_convert<DataType>(-7.0f);
    for(auto& group : groups)
    {
        std::fill(group.host_a.end() - GuardElements, group.host_a.end(), input_guard);
        std::fill(group.host_b.end() - GuardElements, group.host_b.end(), input_guard);
        std::fill(group.host_c.begin(), group.host_c.end(), output_guard);
        group.device_a->ToDevice(group.host_a.data());
        group.device_b->ToDevice(group.host_b.data());
        group.device_c->ToDevice(group.host_c.data());
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop  = nullptr;
    const hipError_t create_start_status = hipEventCreate(&start);
    const hipError_t create_stop_status  = hipEventCreate(&stop);
    if(create_start_status != hipSuccess || create_stop_status != hipSuccess)
    {
        if(start != nullptr)
        {
            [[maybe_unused]] const hipError_t destroy_status = hipEventDestroy(start);
        }
        if(stop != nullptr)
        {
            [[maybe_unused]] const hipError_t destroy_status = hipEventDestroy(stop);
        }
        return false;
    }
    const hipError_t record_start_status = hipEventRecord(start, nullptr);
    const int rc = ck_tile_hcu_grouped_gemm_run(descs.data(),
                                                 GroupCount,
                                                 CK_TILE_HCU_GROUPED_GEMM_BF16,
                                                 'C',
                                                 'R',
                                                 workspace,
                                                 nullptr);
    const hipError_t record_stop_status = hipEventRecord(stop, nullptr);
    const hipError_t sync_status = hipEventSynchronize(stop);
    const hipError_t elapsed_status = hipEventElapsedTime(&elapsed_ms, start, stop);
    const hipError_t destroy_start_status = hipEventDestroy(start);
    const hipError_t destroy_stop_status  = hipEventDestroy(stop);
    if(rc != 0 || record_start_status != hipSuccess ||
       record_stop_status != hipSuccess || sync_status != hipSuccess ||
       elapsed_status != hipSuccess || destroy_start_status != hipSuccess ||
       destroy_stop_status != hipSuccess)
    {
        std::cerr << "CASE mode=" << mode << " result=FAIL reason=launch rc=" << rc
                  << " record_start=" << hipGetErrorString(record_start_status)
                  << " record_stop=" << hipGetErrorString(record_stop_status)
                  << " sync=" << hipGetErrorString(sync_status)
                  << " elapsed=" << hipGetErrorString(elapsed_status)
                  << " destroy_start=" << hipGetErrorString(destroy_start_status)
                  << " destroy_stop=" << hipGetErrorString(destroy_stop_status)
                  << std::endl;
        return false;
    }

    hashes.clear();
    hashes.reserve(GroupCount);
    constexpr std::array<int, 3> sample_rows = {0, 1, M - 1};
    constexpr std::array<int, 3> sample_cols = {0, 1, N - 1};
    for(int group_id = 0; group_id < GroupCount; ++group_id)
    {
        auto& group = groups[group_id];
        group.device_c->FromDevice(group.host_c.data());
        const std::size_t output_elements = static_cast<std::size_t>(M) * N;
        hashes.push_back(fnv1a(group.host_c.data(), output_elements));

        const float expected = static_cast<float>(GroupK[group_id]) *
                               ck_tile::type_convert<float>(group.a_value) *
                               ck_tile::type_convert<float>(group.b_value);
        const float tolerance = 0.25f + 0.02f * std::abs(expected);
        for(int row : sample_rows)
        {
            for(int col : sample_cols)
            {
                const float actual = ck_tile::type_convert<float>(
                    group.host_c[static_cast<std::size_t>(row) * N + col]);
                if(std::abs(actual - expected) > tolerance)
                {
                    std::cerr << "CASE mode=" << mode
                              << " result=FAIL reason=value group=" << group_id
                              << " row=" << row << " col=" << col
                              << " expected=" << expected << " actual=" << actual
                              << " tolerance=" << tolerance << std::endl;
                    return false;
                }
            }
        }

        for(std::size_t i = output_elements; i < group.host_c.size(); ++i)
        {
            if(std::memcmp(&group.host_c[i], &output_guard, sizeof(DataType)) != 0)
            {
                std::cerr << "CASE mode=" << mode
                          << " result=FAIL reason=output_guard group=" << group_id
                          << " offset=" << (i - output_elements) << std::endl;
                return false;
            }
        }
    }

    std::cout << "CASE mode=" << mode << " groups=" << GroupCount
              << " M=" << M << " N=" << N << " logical_k_sum=32757"
              << " elapsed_ms=" << elapsed_ms << " result=PASS" << std::endl;
    return true;
}

} // namespace

int main()
{
    std::vector<GroupBuffers> groups;
    std::vector<ck_tile_hcu_grouped_gemm_desc> descs;
    groups.reserve(GroupCount);
    descs.reserve(GroupCount);

    for(int group_id = 0; group_id < GroupCount; ++group_id)
    {
        const std::size_t a_elements = static_cast<std::size_t>(M) * GroupK[group_id];
        const std::size_t b_elements = static_cast<std::size_t>(GroupK[group_id]) * N;
        const std::size_t c_elements = static_cast<std::size_t>(M) * N;
        const DataType a_value = ck_tile::type_convert<DataType>(
            static_cast<float>(1 + group_id % 4) / 8.0f);
        const DataType b_value = ck_tile::type_convert<DataType>(
            static_cast<float>(1 + (group_id * 3) % 4) / 8.0f);

        GroupBuffers group;
        group.host_a.assign(a_elements + GuardElements, a_value);
        group.host_b.assign(b_elements + GuardElements, b_value);
        group.host_c.resize(c_elements + GuardElements);
        group.device_a = std::make_unique<ck_tile::DeviceMem>(
            group.host_a.size() * sizeof(DataType));
        group.device_b = std::make_unique<ck_tile::DeviceMem>(
            group.host_b.size() * sizeof(DataType));
        group.device_c = std::make_unique<ck_tile::DeviceMem>(
            group.host_c.size() * sizeof(DataType));
        group.a_value = a_value;
        group.b_value = b_value;
        groups.push_back(std::move(group));

        auto& stored = groups.back();
        descs.push_back({stored.device_a->GetDeviceBuffer(),
                         stored.device_b->GetDeviceBuffer(),
                         stored.device_c->GetDeviceBuffer(),
                         1,
                         M,
                         N,
                         GroupK[group_id],
                         M,
                         N,
                         N,
                         0,
                         nullptr,
                         nullptr});
    }

    ck_tile::DeviceMem workspace(ck_tile_hcu_grouped_gemm_workspace_size(GroupCount));
    std::vector<std::uint64_t> selected_hashes_a;
    std::vector<std::uint64_t> selected_hashes_b;
    std::vector<std::uint64_t> rollback_hashes;
    float selected_ms_a = 0;
    float selected_ms_b = 0;
    float rollback_ms   = 0;

    unsetenv("CK_TILE_GROUPED_GEMM_DISABLE_GFX936_BF16_TN_LOGICAL_K_TAIL");
    if(!run_once("selected_guard_pos",
                 ck_tile::type_convert<DataType>(3.0f),
                 groups,
                 descs,
                 workspace.GetDeviceBuffer(),
                 selected_hashes_a,
                 selected_ms_a) ||
       !run_once("selected_guard_neg",
                 ck_tile::type_convert<DataType>(-3.0f),
                 groups,
                 descs,
                 workspace.GetDeviceBuffer(),
                 selected_hashes_b,
                 selected_ms_b) ||
       selected_hashes_a != selected_hashes_b)
    {
        std::cerr << "SUMMARY result=FAIL reason=selected_guard_influence" << std::endl;
        return 1;
    }

    setenv("CK_TILE_GROUPED_GEMM_DISABLE_GFX936_BF16_TN_LOGICAL_K_TAIL", "1", 1);
    if(!run_once("rollback_generic_padding",
                 ck_tile::type_convert<DataType>(5.0f),
                 groups,
                 descs,
                 workspace.GetDeviceBuffer(),
                 rollback_hashes,
                 rollback_ms) ||
       selected_hashes_a != rollback_hashes)
    {
        std::cerr << "SUMMARY result=FAIL reason=rollback_mismatch" << std::endl;
        return 1;
    }
    unsetenv("CK_TILE_GROUPED_GEMM_DISABLE_GFX936_BF16_TN_LOGICAL_K_TAIL");

    std::cout << "SUMMARY selected_guard_hash_equal=1 rollback_hash_equal=1"
              << " selected_ms_a=" << selected_ms_a
              << " selected_ms_b=" << selected_ms_b
              << " rollback_ms=" << rollback_ms
              << " result=PASS" << std::endl;
    return 0;
}
