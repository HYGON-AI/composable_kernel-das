// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_prefill_launch.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_recompute_w_u_kernel.hpp"

namespace gdn_example {

template <typename Problem>
void launch_recompute_impl(const PrefillArguments& args, hipStream_t stream)
{
    using Policy = GdnRecomputeWUDefaultPolicy<Problem>;
    GdnRecomputeWUFwdInvoker<Policy>::Run(stream,
                                          static_cast<const uint16_t*>(args.k),
                                          static_cast<const uint16_t*>(args.v),
                                          args.beta,
                                          args.a,
                                          args.g_cum,
                                          args.w,
                                          args.u,
                                          args.t,
                                          args.h_qk,
                                          args.h_v,
                                          args.head_dim,
                                          args.head_dim,
                                          true,
                                          args.cu_seqlens,
                                          args.chunk_indices,
                                          args.num_chunks,
                                          args.is_varlen);
}

void launch_recompute_bf16(const PrefillArguments& args, hipStream_t stream)
{
    launch_recompute_impl<GdnRecomputeWUProblemBf16>(args, stream);
}

void launch_recompute_fp16(const PrefillArguments& args, hipStream_t stream)
{
    launch_recompute_impl<GdnRecomputeWUProblemFp16>(args, stream);
}

} // namespace gdn_example
