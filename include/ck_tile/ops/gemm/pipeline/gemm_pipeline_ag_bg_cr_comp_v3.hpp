// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

// Compute V3 uses one LDS buffer. Global reads are double-buffered in registers,
// while LDS is consumed before the next K tile overwrites it.
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV3
{
    static constexpr index_t PrefetchStages   = 2;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        if constexpr(Problem::BlockGemmShape::NumWarps == 8)
            return num_loop > 3;
        else
            return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(BlockHasHotloop(num_loop) || num_loop == 3)
        {
            if constexpr(Problem::BlockGemmShape::NumWarps == 8)
                return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
            else
                return BlockHasHotloop(num_loop) ? TailNumber::Even : TailNumber::Odd;
        }
        else if(num_loop == 2)
        {
            return TailNumber::Even;
        }
        else
        {
            return Problem::BlockGemmShape::NumWarps == 8 ? TailNumber::One : TailNumber::Odd;
        }
    }

    template <size_t I = 0, typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        const bool hot_loop      = amd_wave_read_first_lane(has_hot_loop);
        const TailNumber tail    = amd_wave_read_first_lane(tail_number);
        constexpr auto scenarios = []() {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_HOT_TAIL_ONLY)
            // Focused grouped-GEMM candidate: callers prove every descriptor
            // has a V3 hot loop, so avoid embedding the three unreachable
            // short-K pipeline bodies in the persistent kernel.
            return std::array<std::pair<bool, TailNumber>, 2>{
                std::make_pair(true, TailNumber::Even),
                std::make_pair(true, TailNumber::Odd)};
#else
            if constexpr(Problem::BlockGemmShape::NumWarps == 8)
            {
                return std::array<std::pair<bool, TailNumber>, 5>{
                    std::make_pair(false, TailNumber::One),
                    std::make_pair(false, TailNumber::Even),
                    std::make_pair(false, TailNumber::Odd),
                    std::make_pair(true, TailNumber::Even),
                    std::make_pair(true, TailNumber::Odd)};
            }
            else
            {
                return std::array<std::pair<bool, TailNumber>, 4>{
                    std::make_pair(true, TailNumber::Even),
                    std::make_pair(true, TailNumber::Odd),
                    std::make_pair(false, TailNumber::Odd),
                    std::make_pair(false, TailNumber::Even)};
            }
#endif
        }();

        if(hot_loop == scenarios[I].first && tail == scenarios[I].second)
            return run_func(bool_constant<scenarios[I].first>{}, constant<scenarios[I].second>{});
        else if constexpr(I + 1 < scenarios.size())
            return TailHandler<I + 1>(run_func, has_hot_loop, tail_number);

#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_unreachable();
#else
        throw std::logic_error("Invalid Compute V3 hot-loop/tail combination");
#endif
    }
};

template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy>
struct GemmPipelineAgBgCrCompV3 : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using CLayout        = remove_cvref_t<typename Problem::CLayout>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    using Base::PrefetchStages;
    using Base::UsePersistentKernel;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }

    static constexpr index_t GetVectorSizeC()
    {
        return Policy::template GetVectorSizeC<Problem>();
    }

    static constexpr index_t GetSmemPackA()
    {
        return Policy::template GetSmemPackA<Problem>();
    }

    static constexpr index_t GetSmemPackB()
    {
        return Policy::template GetSmemPackB<Problem>();
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        return "COMPUTE_V3";
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_',
                      "pipeline_AgBgCrCompV3",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK),
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <bool HasHotLoop,
              TailNumber TailNum,
              typename ADramBlockWindow,
              typename BDramBlockWindow>
    CK_TILE_DEVICE auto Run(const ADramBlockWindow& a_dram_block_window,
                            const BDramBlockWindow& b_dram_block_window,
                            index_t num_loop,
                            void* p_smem) const
    {
        static_assert(!DoubleSmemBuffer, "Compute V3 requires one LDS buffer");
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindow::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindow::DataType>>,
            "A/B block-window data types must match the pipeline problem");

        constexpr bool is_a_col_major =
            std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major =
            std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
        constexpr auto is_a_load_tr = bool_constant<PipelineImplBase::is_a_load_tr>{};
        constexpr auto is_b_load_tr = bool_constant<PipelineImplBase::is_b_load_tr>{};

        auto&& [a_lds_block, b_lds_block] =
            PipelineImplBase{}.GetABLdsTensorViews(p_smem);

        constexpr auto a_lds_dstr =
            make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_dstr =
            make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

        auto a_windows =
            PipelineImplBase{}.GetAWindows(a_dram_block_window, a_lds_block, a_lds_dstr);
        auto b_windows =
            PipelineImplBase{}.GetBWindows(b_dram_block_window, b_lds_block, b_lds_dstr);
        auto& a_dram_window      = a_windows.get(I0{});
        auto& a_lds_store_window = a_windows.get(I1{});
        auto& a_lds_gemm_window  = a_windows.get(number<2>{});
        auto& b_dram_window      = b_windows.get(I0{});
        auto& b_lds_store_window = b_windows.get(I1{});
        auto& b_lds_gemm_window  = b_windows.get(number<2>{});

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        constexpr ADramTileWindowStep a_step =
            is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
        constexpr BDramTileWindowStep b_step =
            is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

        auto a_reg_tile = load_tile(a_dram_window);
        auto b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);

        const auto store_a = [&](const auto& src) {
            if constexpr(is_a_col_major && !is_a_load_tr())
            {
                auto shuffled = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, src);
            }
        };
        const auto store_b = [&](const auto& src) {
            if constexpr(is_b_row_major && !is_b_load_tr())
            {
                auto shuffled = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, src);
            }
        };

        auto block_gemm   = BlockGemm{};
        auto c_block_tile = block_gemm.MakeCBlockTile();
        using ALdsTile =
            decltype(make_static_distributed_tensor<ADataType>(a_lds_dstr));
        using BLdsTile =
            decltype(make_static_distributed_tensor<BDataType>(b_lds_dstr));
        ALdsTile a_block_tile;
        BLdsTile b_block_tile;
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        store_a(a_reg_tile);
        store_b(b_reg_tile);

        a_reg_tile = load_tile(a_dram_window);
        b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);

        block_sync_lds();
        PipelineImplBase{}.LocalPrefetch(a_block_tile, a_lds_gemm_window, is_a_load_tr);
        PipelineImplBase{}.LocalPrefetch(b_block_tile, b_lds_gemm_window, is_b_load_tr);
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(HasHotLoop)
        {
            index_t i = 0;
            do
            {
                block_gemm(c_block_tile, a_block_tile, b_block_tile);
                block_sync_lds();

                store_a(a_reg_tile);
                store_b(b_reg_tile);

                a_reg_tile = load_tile(a_dram_window);
                b_reg_tile = load_tile(b_dram_window);
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);

                block_sync_lds();
                PipelineImplBase{}.LocalPrefetch(a_block_tile, a_lds_gemm_window, is_a_load_tr);
                PipelineImplBase{}.LocalPrefetch(b_block_tile, b_lds_gemm_window, is_b_load_tr);
                __builtin_amdgcn_sched_barrier(0);
                ++i;
            } while(i < num_loop - (TailNum == TailNumber::Even ? 2 : 1));
        }

        if constexpr(TailNum == TailNumber::Odd || TailNum == TailNumber::One)
        {
            block_gemm(c_block_tile, a_block_tile, b_block_tile);
        }
        else
        {
            block_gemm(c_block_tile, a_block_tile, b_block_tile);
            block_sync_lds();
            store_a(a_reg_tile);
            store_b(b_reg_tile);
            block_sync_lds();
            PipelineImplBase{}.LocalPrefetch(a_block_tile, a_lds_gemm_window, is_a_load_tr);
            PipelineImplBase{}.LocalPrefetch(b_block_tile, b_lds_gemm_window, is_b_load_tr);
            block_gemm(c_block_tile, a_block_tile, b_block_tile);
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto run = [&](auto hot_loop, auto tail) {
            return Run<hot_loop.value, tail.value>(
                a_dram_block_window, b_dram_block_window, num_loop, p_smem);
        };
        return Base::TailHandler(run, has_hot_loop, tail_number);
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem) const
    {
        const auto run = [&](auto hot_loop, auto tail) {
            return Run<hot_loop.value, tail.value>(
                a_dram_block_window, b_dram_block_window, num_loop, p_smem);
        };
        return Base::TailHandler(run, has_hot_loop, tail_number);
    }
};

} // namespace ck_tile
