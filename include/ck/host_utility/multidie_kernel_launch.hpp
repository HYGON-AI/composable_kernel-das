// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "ck/host_utility/kernel_launch.hpp"
#include "ck_tile/host/hip_device_info.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

// Host API availability is a build property. The selected HIP device, not a
// model environment variable or CU-count guess, decides the runtime route.
#ifndef CK_ENABLE_MULTIDIE
#define CK_ENABLE_MULTIDIE 0
#endif

namespace ck {
struct MultidieLaunchPolicy
{
    const char* mode_env;
    const char* chunk_env;
    const char* mask_env;
    const char* log_env;
    const char* name;
    const char* log_name;
    bool count_all_grid_dimensions;
    bool skip_empty_grid;
    bool ordinary_hip_launch;
};

namespace launch_policy {
inline constexpr MultidieLaunchPolicy layernorm{
    "CK_TILE_LAYERNORM_DIE_MODE", "CK_TILE_LAYERNORM_DIE_CHUNK", "CK_TILE_LAYERNORM_DIE_MASK",
    "CK_TILE_LAYERNORM_LOG_LAUNCH", "LayerNorm", "CK layernorm2d", true, false, false};
inline constexpr MultidieLaunchPolicy rmsnorm{
    "CK_TILE_RMSNORM_DIE_MODE", "CK_TILE_RMSNORM_DIE_CHUNK", "CK_TILE_RMSNORM_DIE_MASK",
    "CK_TILE_RMSNORM_LOG_LAUNCH", "RMSNorm", "CK rmsnorm2d", true, false, false};
inline constexpr MultidieLaunchPolicy w8a8{
    "CK_W8A8_DIE_MODE", "CK_W8A8_DIE_CHUNK", "CK_W8A8_DIE_MASK",
    "CK_W8A8_LOG_LAUNCH", "W8A8", "CK W8A8", true, false, false};
inline constexpr MultidieLaunchPolicy moe_sorting{
    "AITER_MOE_SORTING_DIE_MODE", "AITER_MOE_SORTING_DIE_CHUNK", "AITER_MOE_SORTING_DIE_MASK",
    "AITER_MOE_SORTING_LOG_LAUNCH", "MoE sorting", "AITER moe_sorting", false, true, true};
} // namespace launch_policy

namespace multidie_detail {
inline unsigned int read_unsigned(const char* name, unsigned int fallback)
{
    const char* value = std::getenv(name);
    if(value == nullptr)
        return fallback;
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(value, &end, 0);
    if(value[0] == '\0' || value[0] == '-' || end == value || *end != '\0' ||
       errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max())
        throw std::invalid_argument(std::string("invalid unsigned value for ") + name);
    return static_cast<unsigned int>(parsed);
}

#if CK_ENABLE_MULTIDIE
// The target SDK stores dieConfig by pointer in captured nodes. Bind its
// lifetime to the Graph/GraphExec, including clones, rather than this stack.
inline hipLaunchMultiDieConfig* retain_graph_config(hipStream_t stream,
                                                    const hipLaunchMultiDieConfig& config)
{
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    HIP_CHECK_ERROR(hipStreamGetCaptureInfo_v2(stream, &capture, nullptr, &graph));
    if(capture != hipStreamCaptureStatusActive)
        return nullptr;
    auto owned = std::make_unique<hipLaunchMultiDieConfig>(config);
    hipUserObject_t object = nullptr;
    HIP_CHECK_ERROR(hipUserObjectCreate(
        &object, owned.get(),
        [](void* ptr) { delete static_cast<hipLaunchMultiDieConfig*>(ptr); },
        1, hipUserObjectNoDestructorSync));
    auto* pointer = owned.release();
    const auto status = hipGraphRetainUserObject(graph, object, 1, hipGraphUserObjectMove);
    if(status != hipSuccess)
    {
        HIP_CHECK_ERROR(hipUserObjectRelease(object));
        HIP_CHECK_ERROR(status);
    }
    return pointer;
}
#endif

// Deduce argument types only from the kernel signature. Host values convert
// before HIP sees their addresses (in particular int32 -> int64 parameters).
template <typename T> struct kernel_argument { using type = T; };

// True means the dispatch was handled (multi-die, or an allowed empty grid).
// False leaves the ordinary launch/timing implementation to the caller.
template <typename... Args>
inline bool try_launch(const MultidieLaunchPolicy& policy, [[maybe_unused]] void (*kernel)(Args...),
                       dim3 grid, dim3 block, std::size_t lds, [[maybe_unused]] hipStream_t stream,
                       [[maybe_unused]] typename kernel_argument<Args>::type... args)
{
    const char* mode = std::getenv(policy.mode_env);
    if(mode == nullptr)
        mode = "auto";
    const bool automatic = std::strcmp(mode, "auto") == 0;
    const bool off = std::strcmp(mode, "off") == 0;
    if(!automatic && !off && std::strcmp(mode, "linear") != 0)
        throw std::invalid_argument(std::string(policy.mode_env) + " must be auto, off or linear");

    bool multi = false;
    int xcds = 0;
    unsigned int chunk = 0;
#if CK_ENABLE_MULTIDIE
    if(!off)
    {
        const auto& info = ck_tile::get_hip_device_info();
        xcds = info.num_xcds;
        multi = info.arch_name == "gfx946" && xcds >= 2;
        if(!automatic && !multi)
            throw std::runtime_error(std::string(policy.name) +
                " linear launch requires gfx946 with at least 2 reported XCDs");
        const std::uint64_t groups = policy.count_all_grid_dimensions
            ? std::uint64_t{grid.x} * grid.y * grid.z : grid.x;
        multi = multi && (!automatic || groups >= 2);
    }
#else
    if(!automatic && !off)
        throw std::runtime_error(std::string(policy.name) +
                                 " multi-die launch requires a build targeting gfx946");
#endif
    if(multi)
    {
        chunk = read_unsigned(policy.chunk_env, 1);
        if(chunk == 0)
            throw std::invalid_argument(std::string(policy.chunk_env) + " must be positive");
        if(read_unsigned(policy.mask_env, 0) != 0)
            throw std::invalid_argument(std::string(policy.mask_env) + " supports only 0 (all dies)");
    }
    // MoE sorting validates configuration even for an empty dispatch.
    if(policy.skip_empty_grid && (grid.x == 0 || grid.y == 0 || grid.z == 0))
        return true;
    const char* log = std::getenv(policy.log_env);
    if(log != nullptr && std::strcmp(log, "0") != 0)
        std::cerr << '[' << policy.log_name << " launch] mode=" << (multi ? "linear" : "ordinary")
                  << " grid=" << grid.x << ',' << grid.y << ',' << grid.z
                  << " block=" << block.x << ',' << block.y << ',' << block.z
                  << " shared=" << lds << " die_mask=0 chunk=" << chunk << " xcds=" << xcds << '\n';
#if CK_ENABLE_MULTIDIE
    if(multi)
    {
        hipLaunchMultiDieConfig config{};
        config.linearMode.mode = hipDispatchLinearMode;
        config.linearMode.dieMask = hipDieAll;
        config.linearMode.chunkSize = chunk;
        auto* graph_config = retain_graph_config(stream, config);
        void* parameters[] = {static_cast<void*>(&args)...};
        HIP_CHECK_ERROR(hipLaunchKernelMultiDie(reinterpret_cast<const void*>(kernel),
                                              grid, block, parameters, lds, stream,
                                              graph_config ? graph_config : &config));
        return true;
    }
#endif
    return false;
}

template <typename... Args>
inline void launch(const MultidieLaunchPolicy& policy, void (*kernel)(Args...),
                   dim3 grid, dim3 block, std::size_t lds, hipStream_t stream,
                   typename kernel_argument<Args>::type... args)
{
    if(try_launch(policy, kernel, grid, block, lds, stream, args...))
        return;
    if(policy.ordinary_hip_launch)
    {
        void* parameters[] = {static_cast<void*>(&args)...};
        HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<const void*>(kernel),
                                       grid, block, parameters, lds, stream));
    }
    else
        kernel<<<grid, block, lds, stream>>>(args...);
}
} // namespace multidie_detail

// Legacy CK keeps its existing ordinary launch/timer when this build does not
// opt in. Multi-die legacy timing is still unsupported, including mode=off.
template <typename Kernel, typename... Args>
float launch_and_time_multidie_kernel([[maybe_unused]] const MultidieLaunchPolicy& policy,
                                     const StreamConfig& config, Kernel kernel,
                                     dim3 grid, dim3 block, std::size_t lds, Args... args)
{
#if CK_ENABLE_MULTIDIE
    if(config.time_kernel_)
        throw std::runtime_error(std::string(policy.name) +
            " multi-die timing is not implemented; use an external benchmark");
    if(multidie_detail::try_launch(policy, kernel, grid, block, lds, config.stream_id_, args...))
        return 0;
#endif
    return launch_and_time_kernel(config, kernel, grid, block, lds, args...);
}
} // namespace ck

namespace ck_tile {
// Preserve these examples' kentry launch bounds and outer CK Tile timer. MoE
// P0/P1/P23 remain separate, ordered dispatches without a cross-die spin barrier.
template <int MaxThreadPerBlock = CK_TILE_MAX_THREAD_PER_BLOCK,
          int MinBlockPerCu = CK_TILE_MIN_BLOCK_PER_CU, typename Kernel, typename... Args>
inline auto make_multidie_kernel(const ck::MultidieLaunchPolicy& policy, Kernel,
                                 dim3 grid, dim3 block, std::size_t lds, Args... args)
{
    const auto kernel = kentry<MaxThreadPerBlock, MinBlockPerCu, Kernel, Args...>;
    return [=](const stream_config& s) {
        ck::multidie_detail::launch(policy, kernel, grid, block, lds, s.stream_id_, args...);
    };
}
} // namespace ck_tile
