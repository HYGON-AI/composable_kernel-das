// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_prefill_launch.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_cumsum_kernel.hpp"

namespace gdn_example {
namespace {

template <typename DataType>
struct GdnCumsumTailKernel
{
    struct Kargs
    {
        const DataType* g;
        const float* a_log;
        const float* dt_bias;
        float* output;
        int t;
        int heads;
        float scale;
        bool use_gate;
        bool has_bias;
    };

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        const int lane = ck_tile::get_thread_local_1d_id() & 63;
        const int token = static_cast<int>(blockIdx.x) * 64 + lane;
        const int head = blockIdx.y;
        const bool valid = token < args.t;
        float value = 0.0f;
        if(valid)
        {
            const int64_t offset =
                static_cast<int64_t>(token) * args.heads + head;
            value = ck_tile::type_convert<float>(args.g[offset]);
            if(args.use_gate)
            {
                value += args.has_bias ? args.dt_bias[head] : 0.0f;
                value = -expf(args.a_log[head]) *
                    (value < 20.0f ? log1pf(expf(value)) : value);
            }
        }
        ck_tile::static_for<0, 6, 1>{}([&](auto i) {
            constexpr uint32_t offset = 1u << i;
            const float previous =
                ck_tile::warp_shuffle_up(value, offset);
            if(lane >= static_cast<int>(offset))
                value += previous;
        });
        if(valid)
            args.output[static_cast<int64_t>(token) * args.heads + head] =
                value * args.scale;
    }
};

void launch_cumsum_tail(const PrefillArguments& args, hipStream_t stream)
{
    using Kernel = GdnCumsumTailKernel<float>;
    const typename Kernel::Kargs kargs{
        args.g,
        args.gate_in_kernel ? args.a_log : nullptr,
        args.has_dt_bias ? args.dt_bias : nullptr,
        args.g_cum,
        args.t,
        args.h_v,
        args.state_use_exp2 && !args.gate_in_kernel ? 1.0f : ck_tile::log2e_v<float>,
        args.gate_in_kernel,
        args.has_dt_bias};
    const dim3 grid(
        static_cast<unsigned int>((args.t + 63) / 64),
        static_cast<unsigned int>(args.h_v),
        1);
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<64, 1>(
            Kernel{}, grid, dim3(64), 0, kargs));
}

} // namespace

void launch_cumsum(const PrefillArguments& args, hipStream_t stream)
{
    // All chunk stages consume g_cum, including KKT, recompute and output.
    // Disabling the scalar gate must therefore make their cumulative gate
    // neutral as well as selecting the ungated state update.
    if(!args.state_use_g)
    {
        HIP_CHECK_ERROR(hipMemsetAsync(args.g_cum,
                                      0,
                                      static_cast<size_t>(args.t) * args.h_v * sizeof(float),
                                      stream));
        return;
    }
    if(args.t % 64 != 0 && !args.is_varlen)
    {
        launch_cumsum_tail(args, stream);
        return;
    }
    using Policy = GdnCumsumDefaultPolicy<GdnCumsumProblemF32>;
    GdnCumsumFwdInvoker<Policy>::Run(stream,
                                     args.g,
                                     args.gate_in_kernel ? args.a_log : nullptr,
                                     args.has_dt_bias ? args.dt_bias : nullptr,
                                     args.g_cum,
                                     args.t,
                                     args.h_v,
                                     1,
                                     args.state_use_exp2 && !args.gate_in_kernel
                                         ? 1.0f : ck_tile::log2e_v<float>,
                                     false,
                                     args.gate_in_kernel,
                                     args.has_dt_bias,
                                     args.cu_seqlens,
                                     args.chunk_indices,
                                     args.num_chunks,
                                     args.is_varlen);
}

} // namespace gdn_example
