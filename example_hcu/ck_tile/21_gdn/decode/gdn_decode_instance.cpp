// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_decode_launch.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_fused_recurrent_kernel.hpp"

namespace gdn_decode_example {
namespace {

template <typename DataType, bool ScalarGHeadwiseFastPath = false>
void launch_impl(const GdnFusedRecurrentKargs& args, hipStream_t stream)
{
    using Problem =
        GdnFusedRecurrentProblem<DataType, DataType, kSupportedHeadDim>;
    using Policy = GdnFusedRecurrentDefaultPolicy<Problem>;
    using Kernel =
        GdnFusedRecurrentKernel<Policy, ScalarGHeadwiseFastPath>;

    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<Policy::kBlockSize,
                             Policy::kLaunchMinBlocks>(
            Kernel{},
            Kernel::GridSize(args),
            Kernel::BlockSize(),
            0,
            args));
}

template <typename DataType>
void dispatch(const GdnFusedRecurrentKargs& args, hipStream_t stream)
{
    const bool scalar_g_headwise_fast_path =
        args.key_dim == kSupportedHeadDim &&
        args.value_dim == kSupportedHeadDim &&
        args.g_dtype == gdn_tensor_dtype_code(GdnTensorDtype::Float32) &&
        args.beta_dtype == gdn_tensor_dtype_code(GdnTensorDtype::BFloat16) &&
        args.use_g && !args.use_gk && !args.use_gv &&
        args.beta_headwise && args.use_initial_state &&
        args.store_final_state && !args.use_qk_l2norm &&
        !args.use_exp2 && !args.transpose_state &&
        !args.variable_length && !args.gate_in_kernel;
    if(scalar_g_headwise_fast_path)
        launch_impl<DataType, true>(args, stream);
    else
        launch_impl<DataType>(args, stream);
}

} // namespace

void launch_bf16(const GdnFusedRecurrentKargs& args, hipStream_t stream)
{
    dispatch<ck_tile::bf16_t>(args, stream);
}

void launch_fp16(const GdnFusedRecurrentKargs& args, hipStream_t stream)
{
    dispatch<ck_tile::half_t>(args, stream);
}

} // namespace gdn_decode_example
