// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mls_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = GemmPipelineAgBgCrMlsDefaultPolicy>
struct GemmPipelineAgBgCrMls
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using CLayout        = remove_cvref_t<typename Problem::CLayout>;
    using AMlsLayout     = ALayout;
    using BMlsLayout     = std::conditional_t<
        std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>,
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
#if defined(CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE)
    static constexpr index_t NumLdsStages = 1;
#else
    static constexpr index_t NumLdsStages = 2;
#endif
    using WarpTile = typename BlockGemmShape::WarpTile;
    static constexpr index_t MmmacInterleave = WarpTile::at(number<0>{}) / 16;
    static constexpr index_t NmmacInterleave = WarpTile::at(number<1>{}) / 16;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool Preshuffle       = false;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;
    static constexpr index_t NumWaveGroups = 1;
    static constexpr bool IsMlsPipeline    = true;
#if defined(CK_TILE_GROUPED_GEMM_MLS_DS_SOFFSET)
    static constexpr bool UseDsM0 = false;
#else
    static constexpr bool UseDsM0 = true;
#endif

    static_assert((std::is_same_v<ALayout, tensor_layout::gemm::RowMajor> ||
                   std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>) &&
                      (std::is_same_v<BLayout, tensor_layout::gemm::RowMajor> ||
                       std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>),
                  "Grouped-GEMM MLS requires GEMM row-major or column-major inputs");
    static_assert(
        ((MPerBlock == 128 && NPerBlock == 128 &&
          (KPerBlock == 32 || KPerBlock == 64) && BlockSize == 256) ||
         (((KPerBlock == 32 || KPerBlock == 64) &&
           MPerBlock == 256 && NPerBlock == 256 &&
           (BlockSize == 512 || BlockSize == 1024)) ||
          (KPerBlock == 32 &&
           ((MPerBlock == 256 && NPerBlock == 128) ||
            (MPerBlock == 128 && NPerBlock == 256) ||
            (MPerBlock == 256 && NPerBlock == 512)) &&
           (BlockSize == 512 || BlockSize == 1024)))),
        "Grouped-GEMM MLS supports 128x128x32-or-64/256 and experimental "
        "256x256x32-or-64, 256x128x32, 128x256x32, or 256x512x32 geometries");

    using WaspHelper = wasp_helper<BlockSize / get_warp_size(), BlockSize, false>;
    static constexpr auto AMlsWindowLengths = sequence<32, 32>{};
    static constexpr auto BMlsWindowLengths = sequence<32, 32>{};

    template <bool = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeA()
    {
        return 1;
    }

    template <bool = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeB()
    {
        return 1;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC() { return 2; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPackA() { return 1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPackB() { return 1; }
    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC() { return bool_constant<false>{}; }

    CK_TILE_DEVICE static void PromoteMlsPriority()
    {
#if !defined(CK_TILE_GROUPED_GEMM_MLS_DISABLE_PRIORITY_TOGGLE) && \
    !defined(CK_TILE_GROUPED_GEMM_MLS_KEEP_PRIORITY)
        promote_prio();
#endif
    }

    CK_TILE_DEVICE static void RestoreMlsPriority()
    {
#if !defined(CK_TILE_GROUPED_GEMM_MLS_DISABLE_PRIORITY_TOGGLE) && \
    !defined(CK_TILE_GROUPED_GEMM_MLS_KEEP_PRIORITY)
        restore_prio();
#endif
    }

    CK_TILE_DEVICE static void BeginMlsPriorityRegion()
    {
#if defined(CK_TILE_GROUPED_GEMM_MLS_KEEP_PRIORITY)
        promote_prio();
#endif
    }

    CK_TILE_DEVICE static void EndMlsPriorityRegion()
    {
#if defined(CK_TILE_GROUPED_GEMM_MLS_KEEP_PRIORITY)
        restore_prio();
#endif
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetLdsByteSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr bool IsSupported(index_t num_loop)
    {
        return num_loop % NumLdsStages == 0 && num_loop / NumLdsStages > 1;
    }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop / NumLdsStages > 2;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t)
    {
        return TailNumber::Full;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber)
    {
        if(has_hot_loop)
        {
            return run_func(bool_constant<true>{},
                            integral_constant<TailNumber, TailNumber::Full>{});
        }
        return run_func(bool_constant<false>{},
                        integral_constant<TailNumber, TailNumber::Full>{});
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_',
                      "pipeline_AgBgCrMls",
                      concat('x', MPerBlock, NPerBlock, KPerBlock, BlockSize),
                      concat('x', NumLdsStages));
    }

    template <typename AView, typename BView>
    CK_TILE_DEVICE auto operator()(const AView& a_view,
                                   const BView& b_view,
                                   index_t block_origin_m,
                                   index_t block_origin_n,
                                   index_t stride_a,
                                   index_t stride_b,
                                   index_t k_remainder,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   void* p_smem) const
    {
        auto a_mls_window =
            make_tile_window_mls(a_view,
                                 sequence<number<MPerBlock>{}, number<KPerBlock>{}>{},
                                 AMlsWindowLengths,
                                 {block_origin_m, 0},
                                 AMlsLayout{},
                                 number<MmmacInterleave>{},
                                 WaspHelper{},
                                 stride_a,
                                 k_remainder);

        auto b_mls_window =
            make_tile_window_mls(b_view,
                                 sequence<number<NPerBlock>{}, number<KPerBlock>{}>{},
                                 BMlsWindowLengths,
                                 {block_origin_n, 0},
                                 BMlsLayout{},
                                 number<NmmacInterleave>{},
                                 WaspHelper{},
                                 stride_b,
                                 k_remainder);

        constexpr auto a_lds_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        constexpr auto b_lds_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        constexpr index_t a_lds_elements =
            Policy::template GetALdsElementSize<Problem>();
        constexpr index_t b_lds_elements =
            Policy::template GetBLdsElementSize<Problem>();

        auto a_lds_windows = lds_utils::AllocateLdsWindows<ADataType, NumLdsStages>(
            p_smem,
            a_lds_elements,
            0,
            KPerBlock,
            make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
            make_multi_index(0, 0),
            a_lds_desc);

        auto b_lds_windows = lds_utils::AllocateLdsWindows<BDataType, NumLdsStages>(
            p_smem,
            b_lds_elements,
            NumLdsStages * a_lds_elements,
            KPerBlock,
            make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
            make_multi_index(0, 0),
            b_lds_desc);

        if(has_hot_loop)
        {
            return Run<true>(
                a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
        }
        return Run<false>(
            a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
    }

    template <bool HasHotLoop,
              typename AMlsWindow,
              typename ALdsWindows,
              typename BMlsWindow,
              typename BLdsWindows>
    CK_TILE_DEVICE auto Run(AMlsWindow& a_mls_window,
                            ALdsWindows& a_lds_windows,
                            BMlsWindow& b_mls_window,
                            BLdsWindows& b_lds_windows,
                            index_t num_loop) const
    {
#if defined(CK_TILE_GROUPED_GEMM_MLS_SEQUENTIAL_PAIR_STAGGERED)
        static_assert(NumLdsStages == 2 && KPerBlock == 32,
                      "The staggered sequential-pair MLS pipeline requires two K32 LDS stages");
        return RunSequentialPairStaggered(
            a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
#elif defined(CK_TILE_GROUPED_GEMM_MLS_SEQUENTIAL_PAIR_EVEN)
        static_assert(NumLdsStages == 2 && KPerBlock == 32,
                      "The even sequential-pair MLS pipeline requires two K32 LDS stages");
        return RunSequentialPairEven(
            a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
#elif defined(CK_TILE_GROUPED_GEMM_MLS_SEQUENTIAL_PAIR_LATE_REFILL)
        static_assert(NumLdsStages == 2 && KPerBlock == 32,
                      "The late-refill MLS pipeline requires two K32 LDS stages");
        return RunSequentialPair<true>(
            a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
#elif defined(CK_TILE_GROUPED_GEMM_MLS_SEQUENTIAL_PAIR)
        static_assert(NumLdsStages == 2 && KPerBlock == 32,
                      "The sequential-pair MLS pipeline requires two K32 LDS stages");
        return RunSequentialPair<false>(
            a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
#elif defined(CK_TILE_GROUPED_GEMM_MLS_K64_SPLIT_REG)
        static_assert(NumLdsStages == 1 && KPerBlock == 64,
                      "K64 split-register MLS requires one K64 LDS stage");
        return RunK64SplitReg(
            a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
#elif defined(CK_TILE_GROUPED_GEMM_MLS_FUSED_PAIR)
        static_assert(NumLdsStages == 2 && KPerBlock == 32,
                      "The fused-pair MLS pipeline requires two K32 LDS stages");
        return RunFusedPair(
            a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
#else
        const auto block_gemm = Policy::template GetBlockGemm<Problem>();
        auto c_block_tile     = block_gemm.MakeCBlockTile();

        static_for<0, NumLdsStages, 1>{}([&](auto i) {
            a_mls_window.async_mls_load_asm(a_lds_windows(i).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(b_lds_windows(i).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
        });

        __builtin_amdgcn_sched_barrier(0);
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0);
        async_load_fence((a_mls_window.get_num_of_access() +
                          b_mls_window.get_num_of_access()) *
                         (NumLdsStages - 1));
        __builtin_amdgcn_sched_barrier(0);
        wg_sync();

        BeginMlsPriorityRegion();
        PromoteMlsPriority();
        auto a_warp_tensors =
            block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        auto b_warp_tensors =
            block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        RestoreMlsPriority();

        const auto main_loop = [&](auto i, auto pad_last_stage) {
            wg_sync_lds(bool_constant<true>{});
#if defined(CK_TILE_GROUPED_GEMM_MLS_OVERLAP_GLOBAL_LOAD)
            // The warp tensors for this stage have already been read from LDS.
            // Refill the consumed ping-pong buffer before MMAC so the global
            // MLS transfer can make progress while the register-resident tile
            // is being computed.
            constexpr bool pad_k =
                pad_last_stage.value && (i == NumLdsStages - 1);
            a_mls_window.async_mls_load_asm(a_lds_windows(i).get_buffer_ptr(),
                                            bool_constant<pad_k>{});
            b_mls_window.async_mls_load_asm(b_lds_windows(i).get_buffer_ptr(),
                                            bool_constant<pad_k>{});
            a_mls_window.advance();
            b_mls_window.advance();
#endif
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();
#if !defined(CK_TILE_GROUPED_GEMM_MLS_SKIP_POST_MMAC_SYNC)
            wg_sync();
#endif

#if !defined(CK_TILE_GROUPED_GEMM_MLS_OVERLAP_GLOBAL_LOAD)
            constexpr bool pad_k =
                pad_last_stage.value && (i == NumLdsStages - 1);
            a_mls_window.async_mls_load_asm(a_lds_windows(i).get_buffer_ptr(),
                                            bool_constant<pad_k>{});
            b_mls_window.async_mls_load_asm(b_lds_windows(i).get_buffer_ptr(),
                                            bool_constant<pad_k>{});
            a_mls_window.advance();
            b_mls_window.advance();
#endif

            __builtin_amdgcn_sched_barrier(0);
            async_load_fence((a_mls_window.get_num_of_access() +
                              b_mls_window.get_num_of_access()) *
                             (NumLdsStages - 1));
            __builtin_amdgcn_sched_barrier(0);
            wg_sync();

            PromoteMlsPriority();
            a_warp_tensors = block_gemm.GetAWarpTensors(
                a_lds_windows(number<(i + 1) % NumLdsStages>{}).get_buffer_ptr(),
                bool_constant<UseDsM0>{});
            b_warp_tensors = block_gemm.GetBWarpTensors(
                b_lds_windows(number<(i + 1) % NumLdsStages>{}).get_buffer_ptr(),
                bool_constant<UseDsM0>{});
            RestoreMlsPriority();
        };

        const auto tail_loop = [&](auto i) {
            wg_sync_lds(bool_constant<true>{});
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();
            wg_sync();

            if constexpr(i != NumLdsStages - 1)
            {
                __builtin_amdgcn_sched_barrier(0);
                buffer_load_fence(0);
                __builtin_amdgcn_sched_barrier(0);
                wg_sync();
                PromoteMlsPriority();
                a_warp_tensors = block_gemm.GetAWarpTensors(
                    a_lds_windows(number<(i + 1) % NumLdsStages>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                b_warp_tensors = block_gemm.GetBWarpTensors(
                    b_lds_windows(number<(i + 1) % NumLdsStages>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                RestoreMlsPriority();
            }
        };

        if constexpr(HasHotLoop)
        {
            index_t loop = 0;
            do
            {
                static_for<0, NumLdsStages, 1>{}(
                    [&](auto i) { main_loop(i, bool_constant<false>{}); });
                loop += NumLdsStages;
            } while(loop < num_loop - 2 * NumLdsStages);
        }

        static_for<0, NumLdsStages, 1>{}(
            [&](auto i) { main_loop(i, bool_constant<true>{}); });
        static_for<0, NumLdsStages, 1>{}([&](auto i) { tail_loop(i); });

        EndMlsPriorityRegion();
        return c_block_tile;
#endif
    }

#if defined(CK_TILE_GROUPED_GEMM_MLS_SEQUENTIAL_PAIR_STAGGERED)
    template <typename AMlsWindow,
              typename ALdsWindows,
              typename BMlsWindow,
              typename BLdsWindows>
    CK_TILE_DEVICE auto RunSequentialPairStaggered(AMlsWindow& a_mls_window,
                                                   ALdsWindows& a_lds_windows,
                                                   BMlsWindow& b_mls_window,
                                                   BLdsWindows& b_lds_windows,
                                                   index_t num_loop) const
    {
        const auto block_gemm = Policy::template GetBlockGemm<Problem>();
        auto c_block_tile     = block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        constexpr index_t accesses_per_stage =
            AMlsWindow::get_num_of_access() + BMlsWindow::get_num_of_access();

        static_for<0, NumLdsStages, 1>{}([&](auto i) {
            a_mls_window.async_mls_load_asm(a_lds_windows(i).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(b_lds_windows(i).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
        });
        // Stage 0 is ready while the younger stage-1 transfer may remain
        // outstanding. Each hot half-pair repeats this one-stage stagger.
        async_load_fence(accesses_per_stage);
        wg_sync();

        index_t pair = 0;
        while(pair + NumLdsStages < num_loop)
        {
            PromoteMlsPriority();
            auto a_warp_tensors =
                block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            auto b_warp_tensors =
                block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            RestoreMlsPriority();

            wg_sync_lds(bool_constant<true>{});
            a_mls_window.async_mls_load_asm(
                a_lds_windows(number<0>{}).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(
                b_lds_windows(number<0>{}).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();

            // Wait for current stage 1, but leave next stage 0 in flight.
            async_load_fence(accesses_per_stage);
            wg_sync();
            PromoteMlsPriority();
            a_warp_tensors =
                block_gemm.GetAWarpTensors(a_lds_windows(number<1>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            b_warp_tensors =
                block_gemm.GetBWarpTensors(b_lds_windows(number<1>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            RestoreMlsPriority();

            wg_sync_lds(bool_constant<true>{});
            a_mls_window.async_mls_load_asm(
                a_lds_windows(number<1>{}).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(
                b_lds_windows(number<1>{}).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();

            // Wait for next stage 0, but leave next stage 1 in flight.
            async_load_fence(accesses_per_stage);
            wg_sync();
            pair += NumLdsStages;
        }

        PromoteMlsPriority();
        auto a_warp_tensors =
            block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        auto b_warp_tensors =
            block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        RestoreMlsPriority();
        wg_sync_lds(bool_constant<true>{});
        PromoteMlsPriority();
        block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
        RestoreMlsPriority();

        async_load_fence(0);
        wg_sync();
        PromoteMlsPriority();
        a_warp_tensors =
            block_gemm.GetAWarpTensors(a_lds_windows(number<1>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        b_warp_tensors =
            block_gemm.GetBWarpTensors(b_lds_windows(number<1>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        RestoreMlsPriority();
        wg_sync_lds(bool_constant<true>{});
        PromoteMlsPriority();
        block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
        RestoreMlsPriority();

        return c_block_tile;
    }
#endif

#if defined(CK_TILE_GROUPED_GEMM_MLS_SEQUENTIAL_PAIR_EVEN)
    template <typename AMlsWindow,
              typename ALdsWindows,
              typename BMlsWindow,
              typename BLdsWindows>
    CK_TILE_DEVICE auto RunSequentialPairEven(AMlsWindow& a_mls_window,
                                              ALdsWindows& a_lds_windows,
                                              BMlsWindow& b_mls_window,
                                              BLdsWindows& b_lds_windows,
                                              index_t num_loop) const
    {
        const auto block_gemm = Policy::template GetBlockGemm<Problem>();
        auto c_block_tile     = block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        static_for<0, NumLdsStages, 1>{}([&](auto i) {
            a_mls_window.async_mls_load_asm(a_lds_windows(i).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(b_lds_windows(i).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
        });
        async_load_fence(0);
        wg_sync();

        index_t pair = 0;
        while(pair + NumLdsStages < num_loop)
        {
            PromoteMlsPriority();
            auto a_warp_tensors =
                block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            auto b_warp_tensors =
                block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            RestoreMlsPriority();

            wg_sync_lds(bool_constant<true>{});
            a_mls_window.async_mls_load_asm(
                a_lds_windows(number<0>{}).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(
                b_lds_windows(number<0>{}).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();

            PromoteMlsPriority();
            a_warp_tensors =
                block_gemm.GetAWarpTensors(a_lds_windows(number<1>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            b_warp_tensors =
                block_gemm.GetBWarpTensors(b_lds_windows(number<1>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            RestoreMlsPriority();

            wg_sync_lds(bool_constant<true>{});
            a_mls_window.async_mls_load_asm(
                a_lds_windows(number<1>{}).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(
                b_lds_windows(number<1>{}).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();

            async_load_fence(0);
            wg_sync();
            pair += NumLdsStages;
        }

        // IsSupported() guarantees an even num_loop >= 4, so exactly one
        // already-loaded pair remains and no odd-stage tail is possible.
        PromoteMlsPriority();
        auto a_warp_tensors =
            block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        auto b_warp_tensors =
            block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        RestoreMlsPriority();
        wg_sync_lds(bool_constant<true>{});
        PromoteMlsPriority();
        block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
        RestoreMlsPriority();

        PromoteMlsPriority();
        a_warp_tensors =
            block_gemm.GetAWarpTensors(a_lds_windows(number<1>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        b_warp_tensors =
            block_gemm.GetBWarpTensors(b_lds_windows(number<1>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        RestoreMlsPriority();
        wg_sync_lds(bool_constant<true>{});
        PromoteMlsPriority();
        block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
        RestoreMlsPriority();

        return c_block_tile;
    }
#endif

#if defined(CK_TILE_GROUPED_GEMM_MLS_SEQUENTIAL_PAIR) || \
    defined(CK_TILE_GROUPED_GEMM_MLS_SEQUENTIAL_PAIR_LATE_REFILL)
    template <bool LateRefill,
              typename AMlsWindow,
              typename ALdsWindows,
              typename BMlsWindow,
              typename BLdsWindows>
    CK_TILE_DEVICE auto RunSequentialPair(AMlsWindow& a_mls_window,
                                          ALdsWindows& a_lds_windows,
                                          BMlsWindow& b_mls_window,
                                          BLdsWindows& b_lds_windows,
                                          index_t num_loop) const
    {
        const auto block_gemm = Policy::template GetBlockGemm<Problem>();
        auto c_block_tile     = block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        a_mls_window.async_mls_load_asm(a_lds_windows(number<0>{}).get_buffer_ptr());
        b_mls_window.async_mls_load_asm(b_lds_windows(number<0>{}).get_buffer_ptr());
        a_mls_window.advance();
        b_mls_window.advance();
        if(num_loop > 1)
        {
            a_mls_window.async_mls_load_asm(a_lds_windows(number<1>{}).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(b_lds_windows(number<1>{}).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
        }
        async_load_fence(0);
        wg_sync();

        index_t pair = 0;
        while(pair + 1 < num_loop)
        {
            PromoteMlsPriority();
            auto a_warp_tensors =
                block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            auto b_warp_tensors =
                block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            RestoreMlsPriority();

            const bool has_tail = pair + 2 < num_loop;
            if constexpr(!LateRefill)
            {
                wg_sync_lds(bool_constant<true>{});
                if(has_tail)
                {
                    a_mls_window.async_mls_load_asm(
                        a_lds_windows(number<0>{}).get_buffer_ptr());
                    b_mls_window.async_mls_load_asm(
                        b_lds_windows(number<0>{}).get_buffer_ptr());
                    a_mls_window.advance();
                    b_mls_window.advance();
                }
            }
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();

            PromoteMlsPriority();
            a_warp_tensors =
                block_gemm.GetAWarpTensors(a_lds_windows(number<1>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            b_warp_tensors =
                block_gemm.GetBWarpTensors(b_lds_windows(number<1>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            RestoreMlsPriority();

            wg_sync_lds(bool_constant<true>{});
            const bool has_next_pair = pair + 3 < num_loop;
            if constexpr(LateRefill)
            {
                if(has_tail)
                {
                    a_mls_window.async_mls_load_asm(
                        a_lds_windows(number<0>{}).get_buffer_ptr());
                    b_mls_window.async_mls_load_asm(
                        b_lds_windows(number<0>{}).get_buffer_ptr());
                    a_mls_window.advance();
                    b_mls_window.advance();
                }
            }
            if(has_next_pair)
            {
                a_mls_window.async_mls_load_asm(
                    a_lds_windows(number<1>{}).get_buffer_ptr());
                b_mls_window.async_mls_load_asm(
                    b_lds_windows(number<1>{}).get_buffer_ptr());
                a_mls_window.advance();
                b_mls_window.advance();
            }
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();

            if(has_tail)
            {
                async_load_fence(0);
                wg_sync();
            }
            pair += 2;
        }

        if(pair < num_loop)
        {
            PromoteMlsPriority();
            auto a_warp_tensors =
                block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            auto b_warp_tensors =
                block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr(),
                                           bool_constant<UseDsM0>{});
            RestoreMlsPriority();
            wg_sync_lds(bool_constant<true>{});
            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            RestoreMlsPriority();
            wg_sync();
        }

        return c_block_tile;
    }
#endif

#if defined(CK_TILE_GROUPED_GEMM_MLS_K64_SPLIT_REG)
    template <typename AMlsWindow,
              typename ALdsWindows,
              typename BMlsWindow,
              typename BLdsWindows>
    CK_TILE_DEVICE auto RunK64SplitReg(AMlsWindow& a_mls_window,
                                       ALdsWindows& a_lds_windows,
                                       BMlsWindow& b_mls_window,
                                       BLdsWindows& b_lds_windows,
                                       index_t num_loop) const
    {
        constexpr index_t KFragmentsPerHalf = 2;
        const auto block_gemm = Policy::template GetBlockGemm<Problem>();
        static_assert(decltype(block_gemm)::KWarpIter == 2 * KFragmentsPerHalf,
                      "K64 split-register MLS expects four K16 warp fragments");

        auto c_block_tile = block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        a_mls_window.async_mls_load_asm(a_lds_windows(number<0>{}).get_buffer_ptr());
        b_mls_window.async_mls_load_asm(b_lds_windows(number<0>{}).get_buffer_ptr());
        a_mls_window.advance();
        b_mls_window.advance();
        async_load_fence(0);
        wg_sync();

        index_t loop = 0;
        do
        {
            {
                PromoteMlsPriority();
                auto a_warp_tensors = block_gemm.template GetAWarpTensorsSlice<
                    0,
                    KFragmentsPerHalf>(
                    a_lds_windows(number<0>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                auto b_warp_tensors = block_gemm.template GetBWarpTensorsSlice<
                    0,
                    KFragmentsPerHalf>(
                    b_lds_windows(number<0>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                RestoreMlsPriority();
                block_gemm.template RunKSlice<KFragmentsPerHalf>(
                    c_block_tile, a_warp_tensors, b_warp_tensors);
            }

            {
                PromoteMlsPriority();
                auto a_warp_tensors = block_gemm.template GetAWarpTensorsSlice<
                    KFragmentsPerHalf,
                    KFragmentsPerHalf>(
                    a_lds_windows(number<0>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                auto b_warp_tensors = block_gemm.template GetBWarpTensorsSlice<
                    KFragmentsPerHalf,
                    KFragmentsPerHalf>(
                    b_lds_windows(number<0>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                RestoreMlsPriority();

                // All waves must finish the second K32 DS read before MLS
                // overwrites the single K64 LDS stage. The following global
                // transfer then overlaps with the second-half MMAC.
                wg_sync_lds(bool_constant<true>{});
                const bool has_next = loop + 1 < num_loop;
                if(has_next)
                {
                    a_mls_window.async_mls_load_asm(
                        a_lds_windows(number<0>{}).get_buffer_ptr());
                    b_mls_window.async_mls_load_asm(
                        b_lds_windows(number<0>{}).get_buffer_ptr());
                    a_mls_window.advance();
                    b_mls_window.advance();
                }

                block_gemm.template RunKSlice<KFragmentsPerHalf>(
                    c_block_tile, a_warp_tensors, b_warp_tensors);

                if(has_next)
                {
                    async_load_fence(0);
                    wg_sync();
                }
            }
            ++loop;
        } while(loop < num_loop);

        return c_block_tile;
    }
#endif

#if defined(CK_TILE_GROUPED_GEMM_MLS_FUSED_PAIR)
    template <typename AMlsWindow,
              typename ALdsWindows,
              typename BMlsWindow,
              typename BLdsWindows>
    CK_TILE_DEVICE auto RunFusedPair(AMlsWindow& a_mls_window,
                                     ALdsWindows& a_lds_windows,
                                     BMlsWindow& b_mls_window,
                                     BLdsWindows& b_lds_windows,
                                     index_t num_loop) const
    {
        const auto block_gemm = Policy::template GetBlockGemm<Problem>();
        auto c_block_tile     = block_gemm.MakeCBlockTile();

        static_for<0, NumLdsStages, 1>{}([&](auto i) {
            a_mls_window.async_mls_load_asm(a_lds_windows(i).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(b_lds_windows(i).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
        });

        __builtin_amdgcn_sched_barrier(0);
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
        __builtin_amdgcn_sched_barrier(0);
        async_load_fence(0);
        __builtin_amdgcn_sched_barrier(0);
        wg_sync();

        BeginMlsPriorityRegion();
        PromoteMlsPriority();
        auto a_warp_tensors_0 =
            block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        auto b_warp_tensors_0 =
            block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        auto a_warp_tensors_1 =
            block_gemm.GetAWarpTensors(a_lds_windows(number<1>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        auto b_warp_tensors_1 =
            block_gemm.GetBWarpTensors(b_lds_windows(number<1>{}).get_buffer_ptr(),
                                       bool_constant<UseDsM0>{});
        RestoreMlsPriority();

        index_t pair = 0;
        do
        {
            // Both LDS stages are register-resident, so the complete 64 KiB
            // ping-pong allocation can be refilled while two K32 MMAC tiles
            // execute. This emulates a K64/two-stage software pipeline without
            // allocating the unsupported 128 KiB LDS footprint.
            wg_sync_lds(bool_constant<true>{});

            const bool has_next_pair = pair + NumLdsStages < num_loop;
            if(has_next_pair)
            {
                static_for<0, NumLdsStages, 1>{}([&](auto i) {
                    a_mls_window.async_mls_load_asm(
                        a_lds_windows(i).get_buffer_ptr());
                    b_mls_window.async_mls_load_asm(
                        b_lds_windows(i).get_buffer_ptr());
                    a_mls_window.advance();
                    b_mls_window.advance();
                });
            }

            PromoteMlsPriority();
            block_gemm(c_block_tile, a_warp_tensors_0, b_warp_tensors_0);
            block_gemm(c_block_tile, a_warp_tensors_1, b_warp_tensors_1);
            RestoreMlsPriority();

            if(has_next_pair)
            {
                __builtin_amdgcn_sched_barrier(0);
                async_load_fence(0);
                __builtin_amdgcn_sched_barrier(0);
                wg_sync();

                PromoteMlsPriority();
                a_warp_tensors_0 = block_gemm.GetAWarpTensors(
                    a_lds_windows(number<0>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                b_warp_tensors_0 = block_gemm.GetBWarpTensors(
                    b_lds_windows(number<0>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                a_warp_tensors_1 = block_gemm.GetAWarpTensors(
                    a_lds_windows(number<1>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                b_warp_tensors_1 = block_gemm.GetBWarpTensors(
                    b_lds_windows(number<1>{}).get_buffer_ptr(),
                    bool_constant<UseDsM0>{});
                RestoreMlsPriority();
            }

            pair += NumLdsStages;
        } while(pair < num_loop);

        EndMlsPriorityRegion();
        return c_block_tile;
    }
#endif
};

} // namespace ck_tile
