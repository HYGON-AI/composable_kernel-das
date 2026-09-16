// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_prefill_launch.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_kkt_solve_kernel.hpp"

namespace gdn_example {

template <typename Problem, int NumWarps>
void launch_kkt_impl(const PrefillArguments& args, hipStream_t stream)
{
    using Policy = GdnKktSolveDefaultPolicy<Problem>;
    GdnKktSolveFwdInvoker<Policy, NumWarps>::Run(stream,
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
    switch(args.h_v / args.h_qk)
    {
    case 1: launch_kkt_impl<Problem, 1>(args, stream); break;
    case 2: launch_kkt_impl<Problem, 2>(args, stream); break;
    case 4: launch_kkt_impl<Problem, 4>(args, stream); break;
    }
}

void launch_kkt_bf16(const PrefillArguments& args, hipStream_t stream)
{
    dispatch_kkt<GdnKktSolveProblemBf16>(args, stream);
}

void launch_kkt_fp16(const PrefillArguments& args, hipStream_t stream)
{
    dispatch_kkt<GdnKktSolveProblemF16>(args, stream);
}

} // namespace gdn_example
