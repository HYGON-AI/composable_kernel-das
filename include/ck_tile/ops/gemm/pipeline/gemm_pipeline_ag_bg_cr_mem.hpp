// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include <stdexcept>
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrMem
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    // TODO: Is this 32K value gfx9 arch specific?
    static constexpr index_t MinMemInFlyBytes = 32768;

    static constexpr index_t WgpPerCU =
        (4 * get_warp_size() / BlockSize) >= 1 ? 4 * get_warp_size() / BlockSize : 1;
    static constexpr index_t FullMemBandPrefetchStages = integer_divide_ceil(
        MinMemInFlyBytes / WgpPerCU,
        (MPerBlock * sizeof(ADataType) + NPerBlock * sizeof(BDataType)) * KPerBlock);
    static constexpr index_t PrefetchStages =
        FullMemBandPrefetchStages >= 2
            ? FullMemBandPrefetchStages <= 8 ? FullMemBandPrefetchStages : 8
            : 2;

    static constexpr index_t LocalPrefillStages = 1;
    static constexpr index_t GlobalBufferNum    = PrefetchStages;

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % PrefetchStages == 1)
        {
            return TailNumber::One;
        }
        else if(num_loop % PrefetchStages == 2)
        {
            return TailNumber::Two;
        }
        else if(num_loop % PrefetchStages == 3)
        {
            return TailNumber::Three;
        }
        else if(num_loop % PrefetchStages == 4)
        {
            return TailNumber::Four;
        }
        else if(num_loop % PrefetchStages == 5)
        {
            return TailNumber::Five;
        }
        else if(num_loop % PrefetchStages == 6)
        {
            return TailNumber::Six;
        }
        else if(num_loop % PrefetchStages == 7)
        {
            return TailNumber::Seven;
        }
        else
        {
            return TailNumber::Full;
        }
    }
    template <index_t Count = 1, typename Run>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const Run& run, bool has_hot_loop, TailNumber tail)
    {
        if(tail == TailNumber::Full)
        {
            if(has_hot_loop)
                return run(bool_constant<true>{}, constant<TailNumber::Full>{});
            return run(bool_constant<false>{}, constant<TailNumber::Full>{});
        }

        constexpr auto tails = std::array{TailNumber::One,
                                          TailNumber::Two,
                                          TailNumber::Three,
                                          TailNumber::Four,
                                          TailNumber::Five,
                                          TailNumber::Six,
                                          TailNumber::Seven};
        if(tail == tails[Count - 1])
        {
            if(has_hot_loop)
                return run(bool_constant<true>{}, constant<tails[Count - 1]>{});
            return run(bool_constant<false>{}, constant<tails[Count - 1]>{});
        }
        if constexpr(Count + 1 < PrefetchStages)
            return TailHandler<Count + 1>(run, has_hot_loop, tail);
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_unreachable();
#else
        throw std::logic_error("Invalid memory pipeline tail");
#endif
    }
};

template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy>
struct GemmPipelineAgBgCrMem : BaseGemmPipelineAgBgCrMem<Problem>
{
    using Base = BaseGemmPipelineAgBgCrMem<Problem>;
    using Windows = GemmPipelineAgBgCrImplBase<Problem, Policy>;
    using ADataType = typename Problem::ADataType;
    using BDataType = typename Problem::BDataType;
    using CDataType = typename Problem::CDataType;
    using ALayout = typename Problem::ALayout;
    using BLayout = typename Problem::BLayout;
    using CLayout = typename Problem::CLayout;
    using BlockGemmShape = typename Problem::BlockGemmShape;
    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    using Base::PrefetchStages;
    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool Preshuffle = false;
    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;
    static constexpr auto Scheduler = Problem::Scheduler;
    static_assert(Scheduler == GemmPipelineScheduler::Intrawave ||
                  Scheduler == GemmPipelineScheduler::Interwave);

    template <bool Wave32 = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, Wave32>();
    }

    template <bool Wave32 = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, Wave32>();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST static const char* GetName()
    {
        return Scheduler == GemmPipelineScheduler::Intrawave ? "Mem_Intrawave" : "Mem_Interwave";
    }

    template <typename AWindow, typename BWindow>
    CK_TILE_DEVICE auto operator()(const AWindow& a_window,
                                   const BWindow& b_window,
                                   index_t num_loop,
                                   void* lds) const
    {
        constexpr bool a_column = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool b_row = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
        constexpr auto a_step = a_column ? make_array(Base::KPerBlock, 0)
                                         : make_array(0, Base::KPerBlock);
        constexpr auto b_step = b_row ? make_array(Base::KPerBlock, 0)
                                      : make_array(0, Base::KPerBlock);
        constexpr auto a_transpose = bool_constant<Windows::is_a_load_tr>{};
        constexpr auto b_transpose = bool_constant<Windows::is_b_load_tr>{};
        constexpr auto a_dstr =
            make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_dstr =
            make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
        const Windows windows{};
        auto lds_views = windows.GetABLdsTensorViews(lds);
        auto a_windows = windows.GetAWindows(a_window, lds_views.at(number<0>{}), a_dstr);
        auto b_windows = windows.GetBWindows(b_window, lds_views.at(number<1>{}), b_dstr);
        auto& a_global = a_windows.get(number<0>{});
        auto& a_store = a_windows.get(number<1>{});
        auto& a_load = a_windows.get(number<2>{});
        auto& b_global = b_windows.get(number<0>{});
        auto& b_store = b_windows.get(number<1>{});
        auto& b_load = b_windows.get(number<2>{});

        using ATile = decltype(load_tile(a_global));
        using BTile = decltype(load_tile(b_global));
        tuple_array<ATile, PrefetchStages> a_prefetch;
        tuple_array<BTile, PrefetchStages> b_prefetch;
        auto block_gemm = BlockGemm{};
        auto c_tile = block_gemm.MakeCBlockTile();
        auto a_regs = make_static_distributed_tensor<ADataType>(a_dstr);
        auto b_regs = make_static_distributed_tensor<BDataType>(b_dstr);
        tile_elementwise_inout([](auto& c) { c = 0; }, c_tile);

        const auto prefetch = [&](auto slot) {
            a_prefetch.get(slot) = load_tile(a_global);
            b_prefetch.get(slot) = load_tile(b_global);
            move_tile_window(a_global, a_step);
            move_tile_window(b_global, b_step);
        };
        const auto load_operands = [&](auto slot) {
            if constexpr(a_column && !a_transpose())
            {
                auto a = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a, a_prefetch.get(slot));
                windows.LocalPrefill(a_store, a);
            }
            else
                windows.LocalPrefill(a_store, a_prefetch.get(slot));

            if constexpr(b_row && !b_transpose())
            {
                auto b = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b, b_prefetch.get(slot));
                windows.LocalPrefill(b_store, b);
            }
            else
                windows.LocalPrefill(b_store, b_prefetch.get(slot));

            block_sync_lds();
            windows.LocalPrefetch(a_regs, a_load, a_transpose);
            windows.LocalPrefetch(b_regs, b_load, b_transpose);
            block_sync_lds();
        };

        constexpr auto tail = Problem::TailNum;
        constexpr index_t remaining = tail == TailNumber::Full ? PrefetchStages :
                                      tail == TailNumber::One ? 1 :
                                      tail == TailNumber::Two ? 2 :
                                      tail == TailNumber::Three ? 3 :
                                      tail == TailNumber::Four ? 4 :
                                      tail == TailNumber::Five ? 5 :
                                      tail == TailNumber::Six ? 6 : 7;

        if constexpr(Problem::HasHotLoop)
        {
            static_for<0, PrefetchStages, 1>{}(prefetch);
            for(index_t tile = 0; tile < num_loop - remaining; tile += PrefetchStages)
            {
                static_for<0, PrefetchStages, 1>{}([&](auto slot) {
                    load_operands(slot);
                    if constexpr(Scheduler == GemmPipelineScheduler::Interwave)
                    {
                        if(tile + PrefetchStages + slot.value < num_loop)
                            prefetch(slot);
                    }
                    block_gemm(c_tile, a_regs, b_regs);
                    if constexpr(Scheduler == GemmPipelineScheduler::Intrawave)
                    {
                        if(tile + PrefetchStages + slot.value < num_loop)
                            prefetch(slot);
                    }
                });
            }
        }
        else
            static_for<0, remaining, 1>{}(prefetch);

        static_for<0, remaining, 1>{}([&](auto slot) {
            load_operands(slot);
            block_gemm(c_tile, a_regs, b_regs);
        });
        return c_tile;
    }
};

} // namespace ck_tile
