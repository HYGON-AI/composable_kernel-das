// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "ck/host_utility/kernel_launch.hpp"
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
#ifndef CK_W8A8_ENABLE_MULTIDIE
#define CK_W8A8_ENABLE_MULTIDIE 0
#endif

namespace ck {
namespace w8a8_detail {

inline bool is_gfx946() { return ck_tile::get_hip_device_info().arch_name == "gfx946"; }

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
    const char* log = std::getenv("CK_W8A8_LOG_LAUNCH");
    if(log != nullptr && std::strcmp(log, "0") != 0)
        std::cerr << "[CK W8A8 launch] mode=" << mode << " grid=" << grid.x << ','
                  << grid.y << ',' << grid.z << " die_mask=" << mask << " chunk=" << chunk
                  << " xcds=" << xcds << '\n';
}

#if CK_W8A8_ENABLE_MULTIDIE
// This SDK stores dieConfig as a pointer in captured kernel nodes. Tie its
// lifetime to the Graph (including instantiated/cloned Graphs), not this stack.
inline hipLaunchMultiDieConfig* retain_graph_config(hipStream_t stream,
                                                    const hipLaunchMultiDieConfig& config)
{
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    hip_check_error(hipStreamGetCaptureInfo_v2(stream, &capture, nullptr, &graph));
    if(capture != hipStreamCaptureStatusActive)
        return nullptr;

    auto owned = std::make_unique<hipLaunchMultiDieConfig>(config);
    hipUserObject_t object = nullptr;
    hip_check_error(hipUserObjectCreate(
        &object, owned.get(),
        [](void* ptr) { delete static_cast<hipLaunchMultiDieConfig*>(ptr); },
        1, hipUserObjectNoDestructorSync));
    auto* pointer = owned.release(); // user object now owns the allocation
    const auto status = hipGraphRetainUserObject(graph, object, 1, hipGraphUserObjectMove);
    if(status != hipSuccess)
    {
        hip_check_error(hipUserObjectRelease(object));
        hip_check_error(status);
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
    const char* mode = std::getenv("CK_W8A8_DIE_MODE");
    if(mode == nullptr)
        mode = "auto";
    const bool automatic = std::strcmp(mode, "auto") == 0;
    if(std::strcmp(mode, "off") == 0)
    {
        log_launch("ordinary", grid, 0, 0);
        return false;
    }
    if(!automatic && std::strcmp(mode, "linear") != 0)
        throw std::invalid_argument("CK_W8A8_DIE_MODE must be auto, off or linear");

#if CK_W8A8_ENABLE_MULTIDIE
    const auto& info = ck_tile::get_hip_device_info();
    if(info.arch_name != "gfx946")
    {
        if(!automatic)
            throw std::runtime_error("W8A8 explicit multi-die launch requires gfx946");
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
        throw std::runtime_error("W8A8 explicit multi-die launch requires the runtime to report at least 2 XCDs");
    const auto chunk = read_unsigned("CK_W8A8_DIE_CHUNK", 1);
    const auto mask = read_unsigned("CK_W8A8_DIE_MASK", 0); // hipDieAll
    if(chunk == 0)
        throw std::invalid_argument("CK_W8A8_DIE_CHUNK must be positive");
    // The target sbx2 runtime ignores individual-die masks. Only the all-die
    // route is validated; reject unsupported isolation instead of hiding it.
    if(mask != 0)
        throw std::invalid_argument("CK_W8A8_DIE_MASK currently supports only 0 (all dies)");
    hipLaunchMultiDieConfig config{};
    config.linearMode.mode = hipDispatchLinearMode;
    config.linearMode.dieMask = hipDieAll;
    config.linearMode.chunkSize = chunk;
    auto* graph_config = retain_graph_config(stream, config);
    void* parameters[] = {static_cast<void*>(&args)};
    log_launch("linear", grid, mask, chunk, info.num_xcds);
    hip_check_error(hipLaunchKernelMultiDie(reinterpret_cast<const void*>(kernel),
                                          grid,
                                          block,
                                          parameters,
                                          lds_bytes,
                                          stream,
                                          graph_config == nullptr ? &config : graph_config));
    return true;
#else
    if(!automatic)
        throw std::runtime_error("W8A8 multi-die launch requires a build targeting gfx946");
    log_launch("ordinary", grid, 0, 0);
    return false;
#endif
}
} // namespace w8a8_detail


// Only the AITER W8A8 modules opt into this launch policy. Other CK consumers
// preserve the ordinary launch and timing implementation.
template <typename Kernel, typename Kargs>
float launch_and_time_w8a8_kernel(const StreamConfig& config, Kernel kernel,
                                 dim3 grid, dim3 block, std::size_t lds, Kargs args)
{
#if CK_W8A8_ENABLE_MULTIDIE
    if(config.time_kernel_)
        throw std::runtime_error("W8A8 multi-die timing is not implemented; use an external benchmark");
    if(w8a8_detail::launch_multidie(kernel, grid, block, lds, config.stream_id_, args))
        return 0;
#endif
    return launch_and_time_kernel(config, kernel, grid, block, lds, args);
}
} // namespace ck
