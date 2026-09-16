// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "gdn_prefill_launch.hpp"
#include "gdn_fwd_state_static.hpp"
#include "ck_tile/ops/gdn/pipeline/cp/chunk_delta_h_grouped_scan.hpp"

#include <algorithm>

namespace gdn_example {
namespace {

using CpDataType = ck_tile::bf16_t;
using CpTraits = ck_tile::ChunkDeltaHScanFwdTraits<CpDataType>;
using CpGroupedKargs = ck_tile::ChunkDeltaHGroupedScanKargs<CpDataType>;
constexpr ck_tile::index_t kCpPrefixVTile = 32;
constexpr ck_tile::index_t kCpSummaryBlockSize = 512;

ck_tile::index_t choose_cp_groups(ck_tile::index_t num_chunks,
                                  ck_tile::index_t value_heads,
                                  ck_tile::index_t num_cus)
{
    const auto one_summary_wave =
        (num_cus + 2 * value_heads - 1) / (2 * value_heads);
    const auto two_summary_waves =
        (2 * num_cus + 2 * value_heads - 1) / (2 * value_heads);

    if(4 * value_heads <= num_cus / 2)
    {
        if(num_chunks <= 16)
            return std::max<ck_tile::index_t>(2, one_summary_wave - 1);
        if(num_chunks <= 64)
            return one_summary_wave;
        const auto growing_two_wave_grid =
            one_summary_wave - 1 + (num_chunks + 15) / 16;
        return std::min(two_summary_waves, growing_two_wave_grid);
    }

    // The hybrid first stage also replays group 0. Leave one additional
    // group for the tail-replay grid so (groups - 1) * value_heads can fill
    // the device without a shape-specific override.
    const auto dense_summary_grid =
        1 + (num_cus + value_heads - 1) / value_heads;
    const auto short_sequence_groups = 1 + (num_chunks + 7) / 8;
    return std::min(dense_summary_grid, short_sequence_groups);
}

void launch_state_cp_bf16(const PrefillArguments& args, hipStream_t stream)
{
    const ck_tile::index_t num_groups = args.state_cp_groups;
    const ck_tile::index_t num_tail_summary_groups = num_groups - 2;
    const ck_tile::index_t num_hybrid_slots = num_groups - 1;
    auto make_scan_args = [&](CpDataType* summary,
                              const float* state,
                              CpDataType* h_out,
                              CpDataType* v_out) {
        return ck_tile::ChunkDeltaHScanFwdKargs<CpDataType>{
            static_cast<const CpDataType*>(args.k),
            reinterpret_cast<const CpDataType*>(args.w),
            reinterpret_cast<const CpDataType*>(args.u),
            args.g_cum,
            nullptr,
            state,
            nullptr,
            nullptr,
            h_out,
            v_out,
            nullptr,
            args.t,
            num_groups,
            0,
            args.h_qk,
            args.h_v,
            state != nullptr,
            false,
            v_out != nullptr,
            true,
            false,
            true,
            false,
            true,
            nullptr,
            nullptr,
            summary};
    };

    auto* group_a = reinterpret_cast<CpDataType*>(args.state_cp_group_a);
    auto* group_b = reinterpret_cast<CpDataType*>(args.state_cp_group_b);
    auto* group_start = args.state_cp_group_start;
    auto* h = reinterpret_cast<CpDataType*>(args.h);
    auto* v_new = reinterpret_cast<CpDataType*>(args.v_new);

    CpGroupedKargs summary_args{
        make_scan_args(group_a, nullptr, nullptr, nullptr),
        args.num_chunks,
        num_groups};
    summary_args.scan.h_local = group_b;

    CpGroupedKargs replay_args{
        make_scan_args(h, group_start, h, v_new),
        args.num_chunks,
        num_groups};
    replay_args.scan.final_state =
        args.state_store_final_state ? args.final_state : nullptr;

    const bool use_preshuffled_h =
        args.t > 256 && args.h_v >= 16 && args.h_v >= args.h_qk;
    using SummaryPolicy =
        ck_tile::ChunkDeltaHCpSummaryPolicy<kSupportedHeadDim, kCpSummaryBlockSize>;
    using FirstReplayPolicy =
        ck_tile::ChunkDeltaHScanPolicy<64, true, false>;
    using HybridKargs =
        ck_tile::ChunkDeltaHFirstReplayTailSummaryKargs<CpDataType>;
    const auto group_state_elements =
        static_cast<int64_t>(args.h_v) * kSupportedHeadDim * kSupportedHeadDim;
    HybridKargs hybrid_args{
        summary_args,
        args.state_has_initial_state ? args.initial_state : nullptr,
        h,
        v_new,
        group_start + group_state_elements,
        args.state_has_initial_state};

    const auto launch_hybrid = [&](auto preshuffled_h) {
        constexpr bool kPreshuffledH = decltype(preshuffled_h)::value;
        using HybridKernel =
            ck_tile::ChunkDeltaHFirstReplayTailSummaryKernel<
                CpTraits,
                SummaryPolicy,
                FirstReplayPolicy,
                kPreshuffledH>;
        const dim3 block(SummaryPolicy::kBlockSize);
        const dim3 grid(2, num_hybrid_slots * args.h_v);
        auto hybrid =
            ck_tile::make_kernel<SummaryPolicy::kBlockSize,
                                 1>(
                HybridKernel{}, grid, block, 0, hybrid_args);
        hybrid(ck_tile::stream_config{stream, false});
    };
    if(use_preshuffled_h)
        launch_hybrid(std::true_type{});
    else
        launch_hybrid(std::false_type{});

    constexpr ck_tile::index_t kPrefixBlockSize = kCpSummaryBlockSize;
    if(num_tail_summary_groups > 0)
    {
        const dim3 prefix_block(kPrefixBlockSize);
        const dim3 prefix_grid(kSupportedHeadDim / kCpPrefixVTile, args.h_v);
        const auto* group_a_tail = group_a + group_state_elements;
        const auto* group_b_tail = group_b + group_state_elements;
        auto* group_start_tail = group_start + group_state_elements;
        ck_tile::chunk_delta_h_group_prefix_kernel<
            CpDataType, kCpPrefixVTile, kPrefixBlockSize, false>
            <<<prefix_grid, prefix_block, 0, stream>>>(
                group_a_tail,
                group_b_tail,
                group_start_tail,
                group_start_tail,
                nullptr,
                num_tail_summary_groups,
                args.h_v,
                args.head_dim,
                true,
                false);
    }

    const auto launch_replay = [&](auto replay_policy, auto preshuffled_h) {
        using ReplayPolicy = decltype(replay_policy);
        constexpr bool kPreshuffledH = decltype(preshuffled_h)::value;
        using ReplayKernel =
            ck_tile::ChunkDeltaHGroupReplayKernel<
                CpTraits, ReplayPolicy, kPreshuffledH, 1>;
        const dim3 block(ReplayPolicy::kBlockSize);
        const dim3 grid(kSupportedHeadDim / ReplayPolicy::kVTile,
                        (num_groups - 1) * args.h_v);
        auto replay =
            ck_tile::make_kernel<ReplayPolicy::kBlockSize,
                                 ReplayPolicy::kBlockPerCu>(
                ReplayKernel{}, grid, block, 0, replay_args);
        replay(ck_tile::stream_config{stream, false});
    };
    if(use_preshuffled_h)
        launch_replay(
            ck_tile::ChunkDeltaHScanPolicy<kSupportedHeadDim, true, false>{},
            std::true_type{});
    else
        launch_replay(
            ck_tile::ChunkDeltaHScanPolicy<kSupportedHeadDim, true, false>{},
            std::false_type{});
}

struct TransposeStateKernel
{
    CK_TILE_DEVICE void operator()(const float* input,
                                   float* output,
                                   size_t matrices) const
    {
        const size_t matrix = blockIdx.x;
        if(matrix >= matrices)
            return;
        for(int index = threadIdx.x;
            index < kSupportedHeadDim * kSupportedHeadDim;
            index += blockDim.x)
        {
            const int row = index / kSupportedHeadDim;
            const int col = index % kSupportedHeadDim;
            output[matrix * kSupportedHeadDim * kSupportedHeadDim +
                   static_cast<size_t>(col) * kSupportedHeadDim + row] =
                input[matrix * kSupportedHeadDim * kSupportedHeadDim +
                      static_cast<size_t>(row) * kSupportedHeadDim + col];
        }
    }
};

void transpose_state(const float* input,
                     float* output,
                     size_t matrices,
                     hipStream_t stream)
{
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<256, 1>(
            TransposeStateKernel{},
            dim3(static_cast<unsigned int>(matrices)),
            dim3(256),
            0,
            input,
            output,
            matrices));
}

template <typename DataType, ck_tile::index_t VTile, bool GuardTail>
void launch_state_fallback(
    const ck_tile::ChunkDeltaHScanFwdKargs<DataType>& args,
    hipStream_t stream)
{
    using Policy = ck_tile::ChunkDeltaHScanPolicy<VTile, false, GuardTail>;
    using Traits = ck_tile::ChunkDeltaHScanFwdTraits<DataType>;
    using Kernel = ck_tile::ChunkDeltaHScanKernel<Traits, Policy, false, false>;
    const dim3 grid(
        ck_tile::integer_divide_ceil(ck_tile::index_t{kSupportedHeadDim}, VTile),
        args.num_sequences * args.num_value_heads,
        1);
    auto callable = ck_tile::make_kernel<Policy::kBlockSize, Policy::kBlockPerCu>(
        Kernel{}, grid, dim3(Policy::kBlockSize), 0, args);
    callable(ck_tile::stream_config{stream, false});
}

template <typename DataType>
void dispatch_state_fallback(
    const ck_tile::ChunkDeltaHScanFwdKargs<DataType>& args,
    ck_tile::index_t vtile,
    bool guard_tail,
    hipStream_t stream)
{
    if(vtile == 64)
    {
        if(guard_tail)
            launch_state_fallback<DataType, 64, true>(args, stream);
        else
            launch_state_fallback<DataType, 64, false>(args, stream);
    }
    else if(vtile == 32)
    {
        if(guard_tail)
            launch_state_fallback<DataType, 32, true>(args, stream);
        else
            launch_state_fallback<DataType, 32, false>(args, stream);
    }
    else
    {
        if(guard_tail)
            launch_state_fallback<DataType, 16, true>(args, stream);
        else
            launch_state_fallback<DataType, 16, false>(args, stream);
    }
}

} // namespace

int select_state_cp_groups(int total_tokens, int value_heads)
{
    int device = 0;
    hipDeviceProp_t properties{};
    if(hipGetDevice(&device) != hipSuccess ||
       hipGetDeviceProperties(&properties, device) != hipSuccess)
        return 0;

    const auto num_chunks =
        ck_tile::integer_divide_ceil(total_tokens, kChunkSize);
    const auto num_cus =
        static_cast<ck_tile::index_t>(properties.multiProcessorCount);
    const auto direct_vtile =
        value_heads == 16 ? ck_tile::index_t{32}
                          : ck_tile::index_t{16};
    const auto direct_blocks =
        ck_tile::integer_divide_ceil(
            ck_tile::index_t{kSupportedHeadDim}, direct_vtile) *
        value_heads;
    const bool direct_grid_underfills_half_device =
        2 * direct_blocks <= num_cus;
    const bool long_sequence_can_amortize_cp =
        num_chunks >= 64 && direct_blocks < num_cus;
    const bool use_cp =
        (num_chunks >= 32 && direct_grid_underfills_half_device) ||
        long_sequence_can_amortize_cp;
    if(!use_cp)
        return 0;

    auto num_groups = choose_cp_groups(num_chunks, value_heads, num_cus);
    return std::max<ck_tile::index_t>(
        2, std::min(num_groups, num_chunks));
}

template <typename DataType>
void launch_state_impl(const PrefillArguments& args, hipStream_t stream)
{
    constexpr int chunk_size = 64;
    const bool state_varlen = args.state_is_varlen;
    const int num_chunks =
        state_varlen ? 0 : ck_tile::integer_divide_ceil(args.t, chunk_size);
    const size_t matrices =
        static_cast<size_t>(args.num_sequences) * args.h_v;
    const float* state_input = args.initial_state;
    float* state_output = args.output_final_state ? args.final_state : nullptr;
    if(args.state_transpose_state && args.state_has_initial_state)
    {
        transpose_state(args.initial_state,
                        args.state_input_workspace,
                        matrices,
                        stream);
        state_input = args.state_input_workspace;
    }
    if(args.state_transpose_state && args.state_store_final_state)
        state_output = args.state_final_workspace;

    ck_tile::ChunkDeltaHScanFwdKargs<DataType> kargs{
        static_cast<const DataType*>(args.k),
        reinterpret_cast<const DataType*>(args.w),
        reinterpret_cast<const DataType*>(args.u),
        args.g_cum,
        args.state_use_gk ? args.gk : nullptr,
        state_input,
        state_varlen ? args.cu_seqlens : nullptr,
        state_varlen ? args.chunk_offsets : nullptr,
        reinterpret_cast<DataType*>(args.h),
        reinterpret_cast<DataType*>(args.v_new),
        state_output,
        args.t,
        args.num_sequences,
        num_chunks,
        args.h_qk,
        args.h_v,
        args.state_has_initial_state,
        args.state_store_final_state,
        args.state_save_new_value,
        args.state_use_g,
        args.state_use_gk,
        true, // g_cum is always converted to log2 for all chunk stages.
        args.state_transpose_state,
        state_varlen,
        nullptr,
        nullptr,
        reinterpret_cast<DataType*>(args.h)};
    const ck_tile::index_t vtile =
        args.h_v > 16 ? 64 : (args.h_v == 16 ? 32 : 16);
    const bool guard_tail = state_varlen || args.t % chunk_size != 0;
    const bool preshuffled_h =
        args.t > 256 && args.h_v >= 16 && args.h_v >= args.h_qk;
    if(!args.state_use_g || args.state_use_gk)
        dispatch_state_fallback(kargs, vtile, guard_tail, stream);
    else
        gdn::launch_state_scan_dtype(
            kargs, vtile, guard_tail, preshuffled_h, stream);

    if(args.state_transpose_state && args.state_store_final_state)
        transpose_state(args.state_final_workspace,
                        args.final_state,
                        matrices,
                        stream);
}

void launch_state_bf16(const PrefillArguments& args, hipStream_t stream)
{
    if(args.state_cp_groups > 1)
        launch_state_cp_bf16(args, stream);
    else
        launch_state_impl<ck_tile::bf16_t>(args, stream);
}

void launch_state_fp16(const PrefillArguments& args, hipStream_t stream)
{
    launch_state_impl<ck_tile::fp16_t>(args, stream);
}

} // namespace gdn_example
