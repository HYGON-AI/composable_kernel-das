// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"

namespace ck_tile {

// gfx936 W8 experiment: keep the 256x256x64 compute tile, but roll its K
// dimension through two K16 LDS slots (32 KiB total for FP16/BF16 A+B).
// A K16 tile read from LDS and the following K16 tile read from global memory
// are simultaneously live while MMAC consumes the former. This avoids the V3
// W8 lifetime overlap between C, a full K64 global tile and a full K64 LDS
// operand tile.
//
// This pipeline intentionally supports only the aligned RowMajor/ColumnMajor
// grouped-GEMM tuning target. It is independent of the selected V3/V4/DSReadM
// paths and must pass the zero-spill resource gate before production dispatch
// is considered.
template <typename Problem,
          typename Policy = GemmPipelineAgBgCrCompV3WarpRakedPolicy>
struct GemmPipelineAgBgCrCompV3W8Overlap
    : public BaseGemmPipelineAgBgCrCompV3<Problem>
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

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
    static constexpr index_t KPerStage = BlockGemmShape::WarpTile::at(number<2>{});
    static constexpr index_t NumKStages = KPerBlock / KPerStage;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    using StageBlockGemmShape =
        TileGemmShape<sequence<MPerBlock, NPerBlock, KPerStage>,
                      typename BlockGemmShape::BlockWarps,
                      typename BlockGemmShape::WarpTile,
                      BlockGemmShape::PermuteA,
                      BlockGemmShape::PermuteB>;

    struct StageProblem : Problem
    {
        using BlockGemmShape = StageBlockGemmShape;
    };

    using StageBlockGemm =
        remove_cvref_t<decltype(Policy::template GetBlockGemm<StageProblem>())>;

    using LdsBlockGemmShape =
        TileGemmShape<sequence<MPerBlock, NPerBlock, 2 * KPerStage>,
                      typename BlockGemmShape::BlockWarps,
                      typename BlockGemmShape::WarpTile,
                      BlockGemmShape::PermuteA,
                      BlockGemmShape::PermuteB>;

    struct LdsProblem : Problem
    {
        using BlockGemmShape = LdsBlockGemmShape;
    };

    using LdsPipelineImplBase = GemmPipelineAgBgCrImplBase<LdsProblem, Policy>;

    static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor> &&
                      std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>,
                  "W8 overlap currently supports aligned RC grouped GEMM only");
    static_assert(MPerBlock == 256 && NPerBlock == 256 && KPerBlock == 64,
                  "W8 overlap is specialized for a 256x256x64 block tile");
    static_assert(BlockGemmShape::NumWarps == 8,
                  "W8 overlap requires exactly eight compute waves");
    static_assert(!Problem::kPadM && !Problem::kPadN && !Problem::kPadK,
                  "W8 overlap target requires aligned M/N/K");
    static_assert(!Problem::DoubleSmemBuffer,
                  "W8 overlap uses two rolling K16 LDS slots");
    static_assert(NumKStages == 4,
                  "W8 overlap expects four K16 slots in every K64 LDS tile");

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
        return "COMPUTE_V3_W8_OVERLAP";
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        constexpr index_t WaveNumM =
            BlockGemmShape::BlockWarps::at(number<0>{});
        constexpr index_t WaveNumN =
            BlockGemmShape::BlockWarps::at(number<1>{});
        return concat('_',
                      "pipeline_AgBgCrCompV3W8Overlap",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', WaveNumM, WaveNumN),
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<LdsProblem>();
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto Run(const ADramBlockWindow& a_dram_block_window,
                            const BDramBlockWindow& b_dram_block_window,
                            index_t num_loop,
                            void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindow::DataType>> &&
                std::is_same_v<BDataType,
                               remove_cvref_t<typename BDramBlockWindow::DataType>>,
            "A/B block-window data types must match the pipeline problem");

        auto&& [a_lds_block, b_lds_block] =
            LdsPipelineImplBase{}.GetABLdsTensorViews(p_smem);

        constexpr auto a_stage_dram_dstr =
            Policy::template MakeADramTileDistribution<StageProblem>();
        constexpr auto b_stage_dram_dstr =
            Policy::template MakeBDramTileDistribution<StageProblem>();

        auto a_dram_window =
            make_tile_window(a_dram_block_window.get_bottom_tensor_view(),
                             make_tuple(number<MPerBlock>{}, number<KPerStage>{}),
                             a_dram_block_window.get_window_origin(),
                             a_stage_dram_dstr);
        auto b_dram_window =
            make_tile_window(b_dram_block_window.get_bottom_tensor_view(),
                             make_tuple(number<NPerBlock>{}, number<KPerStage>{}),
                             b_dram_block_window.get_window_origin(),
                             b_stage_dram_dstr);

        auto a_lds_store_window_0 =
            make_tile_window(a_lds_block,
                             make_tuple(number<MPerBlock>{}, number<KPerStage>{}),
                             {0, 0});
        auto a_lds_store_window_1 =
            make_tile_window(a_lds_block,
                             make_tuple(number<MPerBlock>{}, number<KPerStage>{}),
                             {0, KPerStage});
        auto b_lds_store_window_0 =
            make_tile_window(b_lds_block,
                             make_tuple(number<NPerBlock>{}, number<KPerStage>{}),
                             {0, 0});
        auto b_lds_store_window_1 =
            make_tile_window(b_lds_block,
                             make_tuple(number<NPerBlock>{}, number<KPerStage>{}),
                             {0, KPerStage});

        constexpr auto a_stage_gemm_dstr = make_static_tile_distribution(
            StageBlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_stage_gemm_dstr = make_static_tile_distribution(
            StageBlockGemm::MakeBBlockDistributionEncode());
        auto a_lds_gemm_window_0 =
            make_tile_window(a_lds_block,
                             make_tuple(number<MPerBlock>{}, number<KPerStage>{}),
                             {0, 0},
                             a_stage_gemm_dstr);
        auto a_lds_gemm_window_1 =
            make_tile_window(a_lds_block,
                             make_tuple(number<MPerBlock>{}, number<KPerStage>{}),
                             {0, KPerStage},
                             a_stage_gemm_dstr);
        auto b_lds_gemm_window_0 =
            make_tile_window(b_lds_block,
                             make_tuple(number<NPerBlock>{}, number<KPerStage>{}),
                             {0, 0},
                             b_stage_gemm_dstr);
        auto b_lds_gemm_window_1 =
            make_tile_window(b_lds_block,
                             make_tuple(number<NPerBlock>{}, number<KPerStage>{}),
                             {0, KPerStage},
                             b_stage_gemm_dstr);

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        constexpr ADramTileWindowStep a_stage_step = make_array(0, KPerStage);
        constexpr BDramTileWindowStep b_stage_step = make_array(0, KPerStage);

        // Fill both K16 LDS slots before allocating the long-lived C tile.
        // The braces are intentional: the initial global tiles die before C is
        // created, preventing their lifetime from extending into the hot loop.
        const auto initial_fill = [&](auto& a_lds_window, auto& b_lds_window) {
            auto a_initial = load_tile(a_dram_window);
            auto b_initial = load_tile(b_dram_window);
            move_tile_window(a_dram_window, a_stage_step);
            move_tile_window(b_dram_window, b_stage_step);
            store_tile(a_lds_window, a_initial);
            store_tile(b_lds_window, b_initial);
        };
        initial_fill(a_lds_store_window_0, b_lds_store_window_0);
        initial_fill(a_lds_store_window_1, b_lds_store_window_1);

        const auto stage_block_gemm = StageBlockGemm{};
        auto c_block_tile           = stage_block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        block_sync_lds();
        auto a_compute_tile = load_tile(a_lds_gemm_window_0);
        auto b_compute_tile = load_tile(b_lds_gemm_window_0);
        block_sync_lds();

        // Roll over the two K16 LDS slots. The next global slot is
        // issued before MMAC on the current LDS-resident slot and is consumed
        // only by the following LDS store. A pair of overlap steps advances
        // one K32 span without a runtime slot-selection branch.
        const index_t total_stages = NumKStages * num_loop;
        index_t prefetched_stages  = 2;
        while(prefetched_stages < total_stages)
        {
            const auto overlap_stage = [&](auto& a_lds_store_window,
                                           auto& b_lds_store_window,
                                           auto& a_next_gemm_window,
                                           auto& b_next_gemm_window) {
                auto a_next_global = load_tile(a_dram_window);
                auto b_next_global = load_tile(b_dram_window);
                move_tile_window(a_dram_window, a_stage_step);
                move_tile_window(b_dram_window, b_stage_step);

                stage_block_gemm(
                    c_block_tile, a_compute_tile, b_compute_tile);

                store_tile(a_lds_store_window, a_next_global);
                store_tile(b_lds_store_window, b_next_global);
                a_compute_tile = load_tile(a_next_gemm_window);
                b_compute_tile = load_tile(b_next_gemm_window);
                block_sync_lds();
            };

            overlap_stage(a_lds_store_window_0,
                          b_lds_store_window_0,
                          a_lds_gemm_window_1,
                          b_lds_gemm_window_1);
            ++prefetched_stages;

            if(prefetched_stages < total_stages)
            {
                overlap_stage(a_lds_store_window_1,
                              b_lds_store_window_1,
                              a_lds_gemm_window_0,
                              b_lds_gemm_window_0);
                ++prefetched_stages;
            }
        }

        stage_block_gemm(c_block_tile, a_compute_tile, b_compute_tile);
        a_compute_tile = load_tile(a_lds_gemm_window_1);
        b_compute_tile = load_tile(b_lds_gemm_window_1);
        block_sync_lds();
        stage_block_gemm(c_block_tile, a_compute_tile, b_compute_tile);

        return c_block_tile;
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return Run(a_dram_block_window, b_dram_block_window, num_loop, p_smem);
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   bool,
                                   TailNumber,
                                   void* p_smem) const
    {
        return Run(a_dram_block_window, b_dram_block_window, num_loop, p_smem);
    }
};

} // namespace ck_tile
