// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_prefill_launch.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_beta_sigmoid_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_prepare_indices_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_l2norm_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_ck_pre_process_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/gdn_cp_context_kernel.hpp"

namespace gdn_example {

// 1. Beta Sigmoid
void launch_beta_sigmoid(const PrefillArguments& args, hipStream_t stream)
{
    if(!args.beta_sigmoid_in_kernel)
        return;
    const size_t count = static_cast<size_t>(args.t) * args.h_v;
    constexpr unsigned int block_size = 256;
    const dim3 grid(
        static_cast<unsigned int>((count + block_size - 1) / block_size));
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<block_size, 1>(
            ck_tile::BetaSigmoidKernel{},
            grid,
            dim3(block_size),
            0,
            args.beta_input,
            args.beta_processed,
            count));
}

// 2. Prepare Chunk Indices
void launch_prepare_chunk_indices(const PrefillArguments& args, hipStream_t stream)
{
    if(!args.is_varlen)
        return;
    ck_tile::PrepareChunkIndicesKernel::Kargs kargs{args.cu_seqlens,
                                                    args.chunk_offsets,
                                                    args.chunk_indices,
                                                    args.num_sequences,
                                                    64};
    constexpr int block_size = 256;
    auto callable = ck_tile::make_kernel<block_size, 1>(
        ck_tile::PrepareChunkIndicesKernel{},
        dim3(1, 1, 1),
        dim3(block_size, 1, 1),
        0,
        kargs);
    callable(ck_tile::stream_config{stream, false});
}

// 3. L2Norm
template <typename DataType>
void launch_l2norm_impl(const PrefillArguments& args, hipStream_t stream)
{
    if(!args.use_qk_l2norm)
        return;

    using Problem = GdnL2NormFwdProblem<DataType, DataType>;
    using Policy  = GdnL2NormFwdDefaultPolicy<Problem>;
    using Kernel  = GdnL2NormFwdKernel<Policy, true>;

    const auto launch_one = [&](const void* input,
                                void* output,
                                float* rstd) {
        auto kargs = Kernel::MakeKargs(
            input,
            output,
            rstd,
            static_cast<ck_tile::long_index_t>(args.t) * args.h_qk,
            args.head_dim,
            1.0e-6f);
        ck_tile::launch_kernel(
            ck_tile::stream_config{stream},
            ck_tile::make_kernel<Policy::kBlockSize, 1>(
                Kernel{},
                Kernel::GridSize(kargs),
                Kernel::BlockSize(),
                0,
                kargs));
    };

    launch_one(args.q_input, args.q_norm, args.q_rstd);
    launch_one(args.k_input, args.k_norm, args.k_rstd);
}

void launch_l2norm_bf16(const PrefillArguments& args, hipStream_t stream)
{
    launch_l2norm_impl<ck_tile::bf16_t>(args, stream);
}

void launch_l2norm_fp16(const PrefillArguments& args, hipStream_t stream)
{
    launch_l2norm_impl<ck_tile::half_t>(args, stream);
}

// 4. CP Context
namespace {

template <typename Problem>
void launch_preprocess(const PrefillArguments& args, hipStream_t stream)
{
    using Policy = GdnPreProcessDefaultPolicy<Problem>;
    GdnPreProcessFwdInvoker<Policy>::Run(
        stream,
        static_cast<const uint16_t*>(args.k),
        args.w,
        args.u,
        args.g_cum,
        args.cp_hm,
        args.t,
        args.h_qk,
        args.h_v,
        ck_tile::kCpDim,
        ck_tile::kCpDim,
        true);
}

} // namespace

bool launch_cp_context(const PrefillArguments& args, hipStream_t stream)
{
    if(!args.cp_context)
        return false;

    if(args.t % 64 != 0 || args.is_varlen)
        return false;

    if(args.is_bf16)
        launch_preprocess<GdnPreProcessProblemBf16>(args, stream);
    else
        launch_preprocess<GdnPreProcessProblemFp16>(args, stream);

    const size_t summary_elements =
        static_cast<size_t>(args.h_v) * ck_tile::kCpDim * (ck_tile::kCpDim + ck_tile::kCpDim);
    const size_t gathered_elements =
        summary_elements * static_cast<size_t>(args.cp_world_size);
    ck_tile::PopulateCpKargs populate_args{
        args.cp_hm, args.cp_ag_hm, args.h_v, args.cp_world_size};
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<ck_tile::kCpBlockSize, 1>(
            ck_tile::PopulateCpKernel{},
            dim3(static_cast<unsigned int>(
                     ck_tile::integer_divide_ceil(
                         gathered_elements,
                         static_cast<size_t>(ck_tile::kCpBlockSize))),
                 1,
                 1),
            dim3(ck_tile::kCpBlockSize, 1, 1),
            0,
            populate_args));

    ck_tile::MergeCpKargs merge_args{
        args.cp_ag_hm, args.cp_state, args.h_v, args.cp_rank};
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<ck_tile::kCpBlockSize, 1>(
            ck_tile::MergeCpKernel{},
            dim3(ck_tile::kCpDim / ck_tile::kCpValueTile,
                 static_cast<unsigned int>(args.h_v),
                 1),
            dim3(ck_tile::kCpBlockSize, 1, 1),
            0,
            merge_args));
    return true;
}

} // namespace gdn_example
