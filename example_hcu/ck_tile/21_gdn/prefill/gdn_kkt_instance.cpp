// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_prefill_launch.hpp"
#include <cstring>
#include "ck_tile/ops/gdn/kernel/gdn_ck_kkt_solve_kernel.hpp"

namespace gdn_example {

template <typename Problem, int NumWarps, bool SplitHeads = false>
void launch_kkt_impl(const PrefillArguments& args, hipStream_t stream)
{
    using Policy = GdnKktSolveDefaultPolicy<Problem>;
    GdnKktSolveFwdInvoker<Policy, NumWarps, SplitHeads>::Run(stream,
                                                 static_cast<const uint16_t*>(args.k),
                                                 args.g_cum,
                                                 args.beta,
                                                 args.a,
                                                 args.t,
                                                 args.h_qk,
                                                 args.h_v,
                                                 args.head_dim,
                                                 true,
                                                 args.cu_seqlens,
                                                 args.chunk_indices,
                                                 args.num_chunks,
                                                 args.is_varlen);
}

template <typename Problem>
void dispatch_kkt(const PrefillArguments& args, hipStream_t stream)
{
    // Small aligned BF16 head groups expose too few CTAs when four value
    // heads share one block. Reuse the existing independent-head schedule.
    const bool aligned_small_bf16 = args.is_bf16 && args.t <= 128 &&
        args.t % 64 == 0 && args.h_qk == 16 && args.h_v == 64;
    if(!args.is_varlen && args.t > 0 &&
       (args.t % 64 != 0 || aligned_small_bf16) && args.h_v > args.h_qk &&
       ((args.t + 63) / 64) * args.h_qk < 128)
    {
        launch_kkt_impl<Problem, 1, true>(args, stream);
        return;
    }
    switch(args.h_v / args.h_qk)
    {
    case 1: launch_kkt_impl<Problem, 1>(args, stream); break;
    case 2: launch_kkt_impl<Problem, 2>(args, stream); break;
    case 4: launch_kkt_impl<Problem, 4>(args, stream); break;
    }
}

void launch_kkt_bf16(const PrefillArguments& args, hipStream_t stream)
{
    // Independent value-head CTAs reduce KKT latency for the validated
    // 64-CU gfx938 fixed-length BF16 workloads. Other targets retain the
    // established grouped-head schedule.
    const bool measured_heads = (args.h_qk == 2 && args.h_v == 8) ||
                                (args.h_qk == 16 && args.h_v == 64);
    const bool measured_length = args.t == 64 || args.t == 256 ||
                                 args.t == 1024 || args.t == 4096;
    if(!args.is_varlen && args.num_sequences == 1 && measured_heads && measured_length)
    {
        int device = 0;
        hipDeviceProp_t properties{};
        if(hipGetDevice(&device) == hipSuccess &&
           hipGetDeviceProperties(&properties, device) == hipSuccess &&
           properties.multiProcessorCount == 64 &&
           std::strncmp(properties.gcnArchName, "gfx938", 6) == 0)
        {
            launch_kkt_impl<GdnKktSolveProblemBf16, 1, true>(args, stream);
            return;
        }
    }
    dispatch_kkt<GdnKktSolveProblemBf16>(args, stream);
}

void launch_kkt_fp16(const PrefillArguments& args, hipStream_t stream)
{
    dispatch_kkt<GdnKktSolveProblemF16>(args, stream);
}

} // namespace gdn_example
