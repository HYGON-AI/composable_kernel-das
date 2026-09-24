// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_decode_launch.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_fused_recurrent_kernel.hpp"

namespace gdn_decode_example {
namespace {

template <typename DataType, bool ScalarGHeadwiseFastPath = false, bool VectorLayout = false, bool RawGateFastPath = false, bool BetaSigmoid = false, int ValueRepeats = 1, int NumWarps = 4, bool Normalize = false>
void launch_impl(const GdnFusedRecurrentKargs& args, hipStream_t stream)
{
    using Problem =
        GdnFusedRecurrentProblem<DataType, DataType, kSupportedHeadDim>;
    using Policy = std::conditional_t<VectorLayout, GdnFusedRecurrentVectorPolicy<Problem, ValueRepeats, NumWarps, Normalize>, GdnFusedRecurrentDefaultPolicy<Problem>>;
    using Kernel =
        GdnFusedRecurrentKernel<Policy, ScalarGHeadwiseFastPath, RawGateFastPath, BetaSigmoid>;

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

template <typename DataType, bool BetaSigmoid = false>
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
    constexpr bool bf = std::is_same_v<DataType, ck_tile::bf16_t>;
    const bool raw_gate_fast = args.transpose_state && args.key_dim == 128 && args.value_dim == 128 &&
        args.g_dtype == (bf ? 1 : 0) && args.dt_bias_dtype == (bf ? 1 : 0) &&
        args.beta_dtype == 2 && args.a_log_dtype == 2 &&
        args.use_g && !args.use_gk && !args.use_gv && args.beta_headwise &&
        args.use_initial_state && args.store_final_state && !args.use_exp2 &&
        args.gate_in_kernel && args.has_dt_bias;
    if(raw_gate_fast)
    {
        const auto heads = args.sequences * args.value_heads;
        if(args.use_qk_l2norm)
            launch_impl<DataType, false, true, true, BetaSigmoid, 2, 4, true>(args, stream);
        // Keep V16 and its grid, but amortize gate/reduction work over two
        // value rows per lane once enough CTAs are available. Tiny grids
        // retain four waves to avoid reducing latency-hiding parallelism.
        else if(heads >= 24 && heads <= 64)
            launch_impl<DataType, false, true, true, BetaSigmoid, 2, 2>(args, stream);
        else if(heads <= 64)
            launch_impl<DataType, false, true, true, BetaSigmoid, 1, 4>(args, stream);
        else if(heads <= 1024)
            launch_impl<DataType, false, true, true, BetaSigmoid, 2, 4>(args, stream);
        else
            launch_impl<DataType, false, true, true, BetaSigmoid, 4, 4>(args, stream);
    }
    else if(args.transpose_state && args.key_dim == 128 && args.value_dim == 128)
        launch_impl<DataType, false, true, false, BetaSigmoid>(args, stream);
    else if(scalar_g_headwise_fast_path)
        launch_impl<DataType, true, false, false, BetaSigmoid>(args, stream);
    else
        launch_impl<DataType, false, false, false, BetaSigmoid>(args, stream);
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


void launch_bf16_raw_beta(const GdnFusedRecurrentKargs& input, const float* raw_beta, hipStream_t stream)
{
    auto args = input;
    args.beta = raw_beta;
    args.beta_dtype = gdn_tensor_dtype_code(GdnTensorDtype::Float32);
    dispatch<ck_tile::bf16_t, true>(args, stream);
}
void launch_fp16_raw_beta(const GdnFusedRecurrentKargs& input, const float* raw_beta, hipStream_t stream)
{
    auto args = input;
    args.beta = raw_beta;
    args.beta_dtype = gdn_tensor_dtype_code(GdnTensorDtype::Float32);
    dispatch<ck_tile::half_t, true>(args, stream);
}

} // namespace gdn_decode_example
