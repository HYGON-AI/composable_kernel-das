// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/host/hip_check_error.hpp"
#include <string>
#include <type_traits>
#include <utility>

namespace ck_tile {
namespace detail {
template <typename Properties, typename = void>
struct has_num_xcc : std::false_type
{
};

template <typename Properties>
struct has_num_xcc<Properties, std::void_t<decltype(std::declval<Properties>().numXcc)>>
    : std::true_type
{
};

template <typename Properties>
inline int reported_xcd_count(const Properties& props)
{
    if constexpr(has_num_xcc<Properties>::value)
        return props.numXcc > 0 ? props.numXcc : 0;
    else
        return 0;
}
} // namespace detail

struct HipDeviceInfo
{
    int device_id = -1;
    std::string arch_name;
    // 0 means unavailable, not a physical device with zero XCDs.
    int num_xcds = 0;
};

// Runtime information for the selected HIP device (HIP_VISIBLE_DEVICES applies).
// No model environment variables, CU-count guesses or operator policy here.
// num_xcds does not describe the HBM address interleave or its granularity.
inline const HipDeviceInfo& get_hip_device_info()
{
    int device = 0;
    HIP_CHECK_ERROR(hipGetDevice(&device));
    static thread_local HipDeviceInfo cached{};
    if(device != cached.device_id)
    {
        hipDeviceProp_t props{};
        HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));
        cached.arch_name = props.gcnArchName;
        const auto suffix = cached.arch_name.find(':');
        if(suffix != std::string::npos)
            cached.arch_name.resize(suffix);
        cached.num_xcds = detail::reported_xcd_count(props);
        cached.device_id = device;
    }
    return cached;
}
} // namespace ck_tile
