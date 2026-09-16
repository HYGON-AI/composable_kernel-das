// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_prefill_launch.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#include "ck_tile/ops/gdn/kernel/gdn_fwd_output_kernel.hpp"

namespace gdn_example {
namespace {

constexpr ck_tile::index_t kValueSplitCtaThreshold = 96;

ck_tile::index_t select_group4_value_split(ck_tile::index_t num_chunks,
                                           ck_tile::index_t num_qk_heads)
{
    return num_chunks * num_qk_heads <= kValueSplitCtaThreshold ? 2 : 1;
}

} // namespace

template <typename DataType, int GroupSize>
void launch_output_impl(const PrefillArguments& args, hipStream_t stream)
{
    const int num_chunks = args.is_varlen
        ? args.num_chunks
        : ck_tile::integer_divide_ceil(args.t, 64);
    const gdn::GdnOutputFwdKargs<DataType> kargs{
        static_cast<const DataType*>(args.q),
        static_cast<const DataType*>(args.k),
        reinterpret_cast<const DataType*>(args.v_new),
        reinterpret_cast<const DataType*>(args.h),
        args.g_cum,
        args.is_varlen ? args.cu_seqlens : nullptr,
        args.is_varlen ? args.chunk_indices : nullptr,
        reinterpret_cast<DataType*>(args.output),
        args.scale,
        args.t,
        num_chunks,
        args.num_sequences,
        args.h_qk,
        args.h_v,
        args.is_varlen};
    const bool parallel_vh = args.t <= 256;
    const bool preshuffled_h = args.t > 256 && args.h_v >= 16;
    const auto launch = [&](auto kernel, dim3 grid) {
        using Kernel = decltype(kernel);
        const dim3 block(Kernel::kBlockSize, 1, 1);
        auto callable = ck_tile::make_kernel<Kernel::kBlockSize, 1>(
            kernel, grid, block, 0, kargs);
        callable(ck_tile::stream_config{stream, false});
    };
    if(parallel_vh)
    {
        using Kernel = gdn::GdnOutputTiledMmacFwdKernel<DataType, 1>;
        launch(Kernel{},
               dim3(num_chunks * (Kernel::kChunkSize / Kernel::kRowTile),
                    args.h_v,
                    1));
    }
    else if constexpr(GroupSize == 1)
    {
        if(preshuffled_h)
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, true, true>{},
                   dim3(num_chunks, args.h_qk, 1));
        else
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 1, 1, false, true>{},
                   dim3(num_chunks, args.h_qk, 1));
    }
    else if constexpr(GroupSize == 2)
    {
        if(preshuffled_h)
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 2, 1, true>{},
                   dim3(num_chunks, args.h_qk, 1));
        else
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 2, 1, false>{},
                   dim3(num_chunks, args.h_qk, 1));
    }
    else
    {
        const auto value_split =
            select_group4_value_split(num_chunks, args.h_qk);
        if(value_split == 2)
        {
            if(preshuffled_h)
                launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 2, true>{},
                       dim3(num_chunks, args.h_qk, 2));
            else
                launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 2>{},
                       dim3(num_chunks, args.h_qk, 2));
        }
        else if(preshuffled_h)
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 1, true>{},
                   dim3(num_chunks, args.h_qk, 1));
        else
            launch(gdn::GdnOutputTiledMmacFwdKernel<DataType, 4, 1>{},
                   dim3(num_chunks, args.h_qk, 1));
    }
}

template <typename DataType>
void dispatch_output(const PrefillArguments& args, hipStream_t stream)
{
    switch(args.h_v / args.h_qk)
    {
    case 1: launch_output_impl<DataType, 1>(args, stream); break;
    case 2: launch_output_impl<DataType, 2>(args, stream); break;
    case 4: launch_output_impl<DataType, 4>(args, stream); break;
    }
}

void launch_output_bf16(const PrefillArguments& args, hipStream_t stream)
{
    dispatch_output<ck_tile::bf16_t>(args, stream);
}

void launch_output_fp16(const PrefillArguments& args, hipStream_t stream)
{
    dispatch_output<ck_tile::fp16_t>(args, stream);
}

} // namespace gdn_example
