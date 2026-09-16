// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_prefill_launch.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_fused_recurrent_kernel.hpp"
#include <stdexcept>

namespace gdn_example {
namespace {

template <typename DataType>
void launch_per_key_prefill(const PrefillArguments& args, hipStream_t stream)
{
    // Scalar-gate chunk factorization does not include the per-key decay in
    // KKT, recompute or output. Use the CK recurrent implementation for this
    // option so every token applies both gates before the delta update.
    if(args.cp_context)
        throw std::invalid_argument("per-key prefill does not support CP context");
    GdnFusedRecurrentKargs kargs{};
    kargs.q = args.q;
    kargs.k = args.k;
    kargs.v = args.v;
    kargs.g = args.g;
    kargs.gk = args.gk;
    kargs.beta = args.beta;
    kargs.a_log = args.a_log;
    kargs.dt_bias = args.dt_bias;
    kargs.initial_state = args.initial_state;
    kargs.cu_seqlens = args.cu_seqlens;
    kargs.output = args.output;
    kargs.final_state = args.final_state;
    kargs.batch = 1;
    kargs.time = args.t;
    kargs.qk_heads = args.h_qk;
    kargs.value_heads = args.h_v;
    kargs.key_dim = args.head_dim;
    kargs.value_dim = args.head_dim;
    kargs.sequences = args.num_sequences;
    kargs.scale = args.scale;
    kargs.g_dtype = gdn_tensor_dtype_code(GdnTensorDtype::Float32);
    kargs.gk_dtype = kargs.g_dtype;
    kargs.beta_dtype = kargs.g_dtype;
    kargs.a_log_dtype = kargs.g_dtype;
    kargs.dt_bias_dtype = kargs.g_dtype;
    kargs.use_g = args.state_use_g;
    kargs.use_gk = true;
    kargs.beta_headwise = true;
    kargs.use_initial_state = args.state_has_initial_state;
    kargs.store_final_state = args.state_store_final_state;
    kargs.use_exp2 = args.state_use_exp2;
    kargs.transpose_state = args.state_transpose_state;
    kargs.variable_length = args.is_varlen;
    kargs.gate_in_kernel = args.gate_in_kernel;
    kargs.has_dt_bias = args.has_dt_bias;
    using Problem = GdnFusedRecurrentProblem<DataType, DataType, kSupportedHeadDim>;
    using Policy = GdnFusedRecurrentDefaultPolicy<Problem>;
    using Kernel = GdnFusedRecurrentKernel<Policy>;
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<Policy::kBlockSize, Policy::kLaunchMinBlocks>(
            Kernel{}, Kernel::GridSize(kargs), Kernel::BlockSize(), 0, kargs));
}

} // namespace

void launch_prefill_bf16(const PrefillArguments& args, hipStream_t stream)
{
    launch_beta_sigmoid(args, stream);
    launch_l2norm_bf16(args, stream);
    if(args.state_use_gk)
    {
        launch_per_key_prefill<ck_tile::bf16_t>(args, stream);
        return;
    }
    launch_prepare_chunk_indices(args, stream);
    launch_cumsum(args, stream);
    launch_kkt_bf16(args, stream);
    launch_recompute_bf16(args, stream);
    launch_cp_context(args, stream);
    launch_state_bf16(args, stream);
    if(args.state_save_new_value)
        launch_output_bf16(args, stream);
}

void launch_prefill_fp16(const PrefillArguments& args, hipStream_t stream)
{
    launch_beta_sigmoid(args, stream);
    launch_l2norm_fp16(args, stream);
    if(args.state_use_gk)
    {
        launch_per_key_prefill<ck_tile::half_t>(args, stream);
        return;
    }
    launch_prepare_chunk_indices(args, stream);
    launch_cumsum(args, stream);
    launch_kkt_fp16(args, stream);
    launch_recompute_fp16(args, stream);
    launch_cp_context(args, stream);
    launch_state_fp16(args, stream);
    if(args.state_save_new_value)
        launch_output_fp16(args, stream);
}

} // namespace gdn_example
