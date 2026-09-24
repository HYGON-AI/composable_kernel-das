// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "gdn_prefill_launch.hpp"
#include "ck_tile/host.hpp"

#include "ck_tile/ops/gdn/kernel/chunk_delta_h_scan_kernel.hpp"
#include "ck_tile/ops/gdn/kernel/chunk_delta_h_cooperative_kernel.hpp"


namespace gdn {

template <typename DataType,
          ck_tile::index_t VTile,
          bool GuardTail,
          bool PreshuffledH>
void launch_state_scan_specialization(
    const ck_tile::ChunkDeltaHScanFwdKargs<DataType>& args,
    hipStream_t stream)
{
    using Policy = ck_tile::ChunkDeltaHScanPolicy<VTile, true, GuardTail>;
    using Traits = ck_tile::ChunkDeltaHScanFwdTraits<DataType>;
    using Kernel = ck_tile::ChunkDeltaHScanKernel<Traits, Policy, true, PreshuffledH>;

    const dim3 block(Policy::kBlockSize, 1, 1);
    const dim3 grid(
        ck_tile::integer_divide_ceil(
            ck_tile::index_t{gdn_example::kSupportedHeadDim}, VTile),
        args.num_sequences * args.num_value_heads,
        1);
    auto callable = ck_tile::make_kernel<Policy::kBlockSize, Policy::kBlockPerCu>(
        Kernel{}, grid, block, 0, args);
    callable(ck_tile::stream_config{stream, false});
}

template <typename DataType>
void launch_state_scan_dtype(
    const ck_tile::ChunkDeltaHScanFwdKargs<DataType>& args,
    ck_tile::index_t vtile,
    bool guard_tail,
    bool preshuffled_h,
    hipStream_t stream)
{
    // The cooperative mapping is the validated route for both dtypes
    // in this bounded small-head workload.
    if(vtile == 16 && !guard_tail && !args.is_varlen &&
       args.num_sequences == 1 && args.num_qk_heads == 2 &&
       args.num_value_heads == 8 && args.total_tokens > 0 &&
       (args.total_tokens <= 4096 ||
        (std::is_same_v<DataType, ck_tile::half_t> && args.total_tokens <= 8192)))
    {
        auto launch=[&](auto packed) {
            using Kernel=ck_tile::ChunkDeltaHCooperativeKernel<DataType,decltype(packed)::value>;
            auto callable=ck_tile::make_kernel<256,1>(Kernel{},dim3(8,args.num_value_heads,1),dim3(256,1,1),0,args);
            callable(ck_tile::stream_config{stream,false});
        };
        if(preshuffled_h)launch(std::true_type{});else launch(std::false_type{});
        return;
    }
    if(vtile == 64)
    {
        if(guard_tail)
        {
            if(preshuffled_h)
                launch_state_scan_specialization<DataType, 64, true, true>(args, stream);
            else
                launch_state_scan_specialization<DataType, 64, true, false>(args, stream);
        }
        else
        {
            if(preshuffled_h)
                launch_state_scan_specialization<DataType, 64, false, true>(args, stream);
            else
                launch_state_scan_specialization<DataType, 64, false, false>(args, stream);
        }
    }
    else if(vtile == 32)
    {
        if(guard_tail)
        {
            if(preshuffled_h)
                launch_state_scan_specialization<DataType, 32, true, true>(args, stream);
            else
                launch_state_scan_specialization<DataType, 32, true, false>(args, stream);
        }
        else
        {
            if(preshuffled_h)
                launch_state_scan_specialization<DataType, 32, false, true>(args, stream);
            else
                launch_state_scan_specialization<DataType, 32, false, false>(args, stream);
        }
    }
    else
    {
        if(guard_tail)
        {
            if(preshuffled_h)
                launch_state_scan_specialization<DataType, 16, true, true>(args, stream);
            else
                launch_state_scan_specialization<DataType, 16, true, false>(args, stream);
        }
        else
        {
            if(preshuffled_h)
                launch_state_scan_specialization<DataType, 16, false, true>(args, stream);
            else
                launch_state_scan_specialization<DataType, 16, false, false>(args, stream);
        }
    }
}

} // namespace gdn
