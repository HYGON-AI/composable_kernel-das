// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/hip_device_info.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

// gfx946 builds enable both launch paths; runtime topology selects the path.
#ifndef CK_TILE_LAYERNORM_ENABLE_MULTIDIE
#define CK_TILE_LAYERNORM_ENABLE_MULTIDIE 0
#endif

namespace ck_tile {
namespace layernorm2d_detail {

inline bool is_gfx946() { return get_hip_device_info().arch_name == "gfx946"; }

inline unsigned int read_unsigned(const char* name, unsigned int default_value)
{
    const char* value = std::getenv(name);
    if(value == nullptr)
        return default_value;
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(value, &end, 0);
    if(value[0] == '\0' || value[0] == '-' || end == value || *end != '\0' ||
       errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max())
        throw std::invalid_argument(std::string("invalid unsigned value for ") + name);
    return static_cast<unsigned int>(parsed);
}

inline void log_launch(const char* mode, dim3 grid, unsigned int mask, unsigned int chunk,
                       int xcds = 0)
{
    const char* log = std::getenv("CK_TILE_LAYERNORM_LOG_LAUNCH");
    if(log != nullptr && std::strcmp(log, "0") != 0)
        std::cerr << "[CK layernorm2d launch] mode=" << mode << " grid=" << grid.x << ','
                  << grid.y << ',' << grid.z << " die_mask=" << mask << " chunk=" << chunk
                  << " xcds=" << xcds << '\n';
}

#if CK_TILE_LAYERNORM_ENABLE_MULTIDIE
// This SDK stores dieConfig as a pointer in captured kernel nodes. Tie its
// lifetime to the Graph (including instantiated/cloned Graphs), not this stack.
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
    auto* pointer = owned.release(); // user object now owns the allocation
    const auto status = hipGraphRetainUserObject(graph, object, 1, hipGraphUserObjectMove);
    if(status != hipSuccess)
    {
        HIP_CHECK_ERROR(hipUserObjectRelease(object));
        HIP_CHECK_ERROR(status);
    }
    return pointer;
}
#endif

template <typename Kernel, typename Kargs>
inline bool launch_multidie(Kernel kernel,
                           dim3 grid,
                           dim3 block,
                           std::size_t lds_bytes,
                           hipStream_t stream,
                           Kargs& args)
{
    const char* mode = std::getenv("CK_TILE_LAYERNORM_DIE_MODE");
    if(mode == nullptr)
        mode = "auto";
    const bool automatic = std::strcmp(mode, "auto") == 0;
    if(std::strcmp(mode, "off") == 0)
    {
        log_launch("ordinary", grid, 0, 0);
        return false;
    }
    if(!automatic && std::strcmp(mode, "linear") != 0)
        throw std::invalid_argument("CK_TILE_LAYERNORM_DIE_MODE must be auto, off or linear");

#if CK_TILE_LAYERNORM_ENABLE_MULTIDIE
    const auto& info = get_hip_device_info();
    if(info.arch_name != "gfx946")
    {
        if(!automatic)
            throw std::runtime_error("LayerNorm explicit multi-die launch requires gfx946");
        log_launch("ordinary", grid, 0, 0);
        return false;
    }
    const std::uint64_t groups = std::uint64_t{grid.x} * grid.y * grid.z;
    if(automatic && (groups < 2 || info.num_xcds < 2))
    {
        log_launch("ordinary", grid, 0, 0, info.num_xcds);
        return false;
    }
    if(info.num_xcds < 2)
        throw std::runtime_error("LayerNorm explicit multi-die launch requires the runtime to report at least 2 XCDs");
    const auto chunk = read_unsigned("CK_TILE_LAYERNORM_DIE_CHUNK", 1);
    const auto mask = read_unsigned("CK_TILE_LAYERNORM_DIE_MASK", 0); // hipDieAll
    if(chunk == 0)
        throw std::invalid_argument("CK_TILE_LAYERNORM_DIE_CHUNK must be positive");
    // The target sbx2 runtime ignores individual-die masks. Only the all-die
    // route is validated; reject unsupported isolation instead of hiding it.
    if(mask != 0)
        throw std::invalid_argument("CK_TILE_LAYERNORM_DIE_MASK currently supports only 0 (all dies)");
    hipLaunchMultiDieConfig config{};
    config.linearMode.mode = hipDispatchLinearMode;
    config.linearMode.dieMask = hipDieAll;
    config.linearMode.chunkSize = chunk;
    auto* graph_config = retain_graph_config(stream, config);
    void* parameters[] = {static_cast<void*>(&args)};
    log_launch("linear", grid, mask, chunk, info.num_xcds);
    HIP_CHECK_ERROR(hipLaunchKernelMultiDie(reinterpret_cast<const void*>(kernel),
                                          grid,
                                          block,
                                          parameters,
                                          lds_bytes,
                                          stream,
                                          graph_config == nullptr ? &config : graph_config));
    return true;
#else
    if(!automatic)
        throw std::runtime_error("LayerNorm multi-die launch requires a build targeting gfx946");
    log_launch("ordinary", grid, 0, 0);
    return false;
#endif
}
} // namespace layernorm2d_detail

// LayerNorm-only wrapper: preserve the existing kentry and ordinary HIP launch.
template <int MaxThreads, int MinBlocks, typename Kernel, typename Kargs>
auto make_layernorm2d_fwd_kernel(Kernel, dim3 grid, dim3 block, std::size_t lds, Kargs args)
{
    const auto kernel = kentry<MaxThreads, MinBlocks, Kernel, Kargs>;
    return [=](const stream_config& s) mutable {
        if(!layernorm2d_detail::launch_multidie(kernel, grid, block, lds, s.stream_id_, args))
            kernel<<<grid, block, lds, s.stream_id_>>>(args);
    };
}
} // namespace ck_tile
