// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/stream_config.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace grouped_splitk_detail {

inline void
CheckShape(index_t m, index_t n, index_t k, index_t bm, index_t bn, index_t bk, index_t batch)
{
    if(m < 0 || n < 0 || k < 0 || batch <= 0)
        throw std::invalid_argument("Grouped Split-K: negative dimension or nonpositive k_batch");
    if(m == 0 || n == 0)
        return;
    if(k == 0)
        throw std::invalid_argument("Grouped Split-K: K=0 with nonempty output is unsupported");
    const uint64_t limit = std::numeric_limits<index_t>::max();
    const uint64_t mp    = (uint64_t(m) + bm - 1) / bm * bm;
    const uint64_t np    = (uint64_t(n) + bn - 1) / bn * bn;
    const uint64_t unit  = uint64_t(bk) * batch;
    const uint64_t kp    = (uint64_t(k) + unit - 1) / unit * unit;
    if(mp > limit || np > limit || kp > limit || mp * np > limit ||
       (mp / bm) * (np / bn) > limit / batch)
        throw std::overflow_error("Grouped Split-K: padded dimensions exceed index_t range");
}

template <typename Layout, typename T>
bool CheckMatrix(const void* p,
                 index_t rows,
                 index_t cols,
                 index_t stride,
                 index_t vector,
                 bool check_pointer = true)
{
    constexpr bool row       = is_same_v<Layout, tensor_layout::gemm::RowMajor>;
    const index_t contiguous = row ? cols : rows;
    const index_t outer      = row ? rows : cols;
    return stride >= contiguous && vector > 0 &&
           (!check_pointer ||
            (p != nullptr && reinterpret_cast<uintptr_t>(p) % (sizeof(T) * vector) == 0)) &&
           stride % vector == 0 &&
           (uint64_t(outer - 1) * stride + contiguous) <=
               uint64_t(std::numeric_limits<index_t>::max()) / sizeof(T);
}

// Time the entire operation, including reset, on the caller's stream. Each warmup
// and repeat must start from the same state even when CK_TIME_KERNEL is disabled.
template <typename Run>
float RunAndTime(const StreamConfig& config, Run run)
{
    if(!config.time_kernel_)
    {
        run();
        return 0;
    }
    if(config.cold_niters_ < 0 || config.nrepeat_ <= 0)
        throw std::invalid_argument("Grouped Split-K: invalid timing iteration count");
    for(int i = 0; i < config.cold_niters_; ++i)
        run();
    hipEvent_t start{}, stop{};
    hip_check_error(hipEventCreate(&start));
    try
    {
        hip_check_error(hipEventCreate(&stop));
        hip_check_error(hipEventRecord(start, config.stream_id_));
        for(int i = 0; i < config.nrepeat_; ++i)
            run();
        hip_check_error(hipEventRecord(stop, config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));
        float milliseconds = 0;
        hip_check_error(hipEventElapsedTime(&milliseconds, start, stop));
        static_cast<void>(hipEventDestroy(stop));
        static_cast<void>(hipEventDestroy(start));
        return milliseconds / config.nrepeat_;
    }
    catch(...)
    {
        if(stop)
            static_cast<void>(hipEventDestroy(stop));
        static_cast<void>(hipEventDestroy(start));
        throw;
    }
}

} // namespace grouped_splitk_detail
} // namespace device
} // namespace tensor_operation
} // namespace ck
