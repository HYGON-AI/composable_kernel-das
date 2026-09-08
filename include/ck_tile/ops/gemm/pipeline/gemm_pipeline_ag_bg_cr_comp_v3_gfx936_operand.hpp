// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"

namespace ck_tile {

// gfx936 aligned RC/NN/TN specialization for a 256x256x64/W8 lowering.
// One K64 global tile is prefetched in registers while the current K64 tile
// resides in LDS. Unlike generic V3, MMAC operands are loaded from LDS one K16
// slice at a time, so the full replicated A/B block tile never coexists with
// the 128-VGPR accumulator. The custom block policy updates C vectors in place.
template <typename Problem,
          typename Policy = GemmPipelineAgBgCrCompV3TritonOperandPolicy>
struct GemmPipelineAgBgCrCompV3TritonOperand
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

    using FullBlockGemm =
        remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using StageBlockGemm =
        remove_cvref_t<decltype(Policy::template GetBlockGemm<StageProblem>())>;

    // Register-only output epilogues obtain their accumulator distribution
    // from the stage block GEMM; K depth does not participate in C layout.
    using BlockGemm = StageBlockGemm;

    static constexpr bool IsARow =
        std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>;
    static constexpr bool IsBRow =
        std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
    static_assert(is_any_of<ALayout,
                            tensor_layout::gemm::RowMajor,
                            tensor_layout::gemm::ColumnMajor>::value &&
                      is_any_of<BLayout,
                                tensor_layout::gemm::RowMajor,
                                tensor_layout::gemm::ColumnMajor>::value,
                  "gfx936 operand pipeline requires a standard GEMM layout");
    static_assert(MPerBlock == 256 && NPerBlock == 256 && KPerBlock == 64 &&
                      KPerStage == 16 && NumKStages == 4 &&
                      BlockGemmShape::WarpTile::at(number<0>{}) == 16 &&
                      BlockGemmShape::WarpTile::at(number<1>{}) == 16,
                  "Triton operand pipeline requires a 256x256x64 block with "
                  "eight interleaved M repeats and four interleaved N repeats "
                  "of the native M16xN16xK16 MMAC tile per wave");
    static_assert(BlockGemmShape::NumWarps == 8,
                  "Triton operand pipeline requires eight waves");
    static_assert(!Problem::kPadM && !Problem::kPadN && !Problem::kPadK &&
                      !Problem::DoubleSmemBuffer,
                  "Triton operand pipeline requires aligned single-LDS-buffer input");

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
        return "COMPUTE_V3_TRITON_OPERAND";
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_',
                      "pipeline_AgBgCrCompV3TritonOperand",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <index_t Offset0, index_t Offset1, typename Vec4>
    CK_TILE_DEVICE static void StoreLdsPairSt64(index_t byte_address,
                                                const Vec4& x0,
                                                const Vec4& x1)
    {
        static_assert(sizeof(Vec4) == 8,
                      "ds_write2st64_b64 requires two 64-bit payloads");
        asm volatile("ds_write2st64_b64 %0, %1, %2 offset0:%3 offset1:%4"
                     :
                     : "v"(byte_address),
                       "v"(bit_cast<fp32x2_t>(x0)),
                       "v"(bit_cast<fp32x2_t>(x1)),
                       "n"(Offset0),
                       "n"(Offset1)
                     : "memory");
    }

    template <typename Vec4>
    CK_TILE_DEVICE static void StoreLdsB64(index_t byte_address, const Vec4& x)
    {
        static_assert(sizeof(Vec4) == 8, "ds_write_b64 requires a 64-bit payload");
        asm volatile("ds_write_b64 %0, %1"
                     :
                     : "v"(byte_address), "v"(bit_cast<fp32x2_t>(x))
                     : "memory");
    }

    template <index_t Offset0, index_t Offset1>
    CK_TILE_DEVICE static fp32x4_t LoadLdsPairSt64(index_t byte_address)
    {
        fp32x4_t payload;
        asm volatile("ds_read2st64_b64 %0, %1 offset0:%2 offset1:%3"
                     : "=v"(payload)
                     : "v"(byte_address), "n"(Offset0), "n"(Offset1)
                     : "memory");
        return payload;
    }

    template <index_t Offset>
    CK_TILE_DEVICE static fp32x4_t LoadLdsM32x16(index_t byte_address)
    {
        fp32x4_t payload;
        asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                     : "=v"(payload)
                     : "v"(byte_address), "n"(Offset)
                     : "memory");
        return payload;
    }

    template <index_t Offset>
    CK_TILE_DEVICE static fp32x2_t LoadLdsB64(index_t byte_address)
    {
        fp32x2_t payload;
        asm volatile("ds_read_b64 %0, %1 offset:%2"
                     : "=v"(payload)
                     : "v"(byte_address), "n"(Offset)
                     : "memory");
        return payload;
    }

    CK_TILE_DEVICE static void WaitLdsReads()
    {
        // Inline DS reads are opaque to the compiler's waitcnt dependency
        // tracker. Do not let the following MMAC consume their destination
        // VGPRs before lgkmcnt reaches zero. Once correctness is established,
        // this full wait is split and interleaved with MMAC like Triton.
#if !defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_SCHED_BARRIER)
        __builtin_amdgcn_sched_barrier(0);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#if !defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_SCHED_BARRIER)
        __builtin_amdgcn_sched_barrier(0);
#endif
    }

    template <index_t Count>
    CK_TILE_DEVICE static void WaitLdsReads(number<Count>)
    {
#if !defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_SCHED_BARRIER)
        __builtin_amdgcn_sched_barrier(0);
#endif
        asm volatile("s_waitcnt lgkmcnt(%0)"
                     :
                     : "n"(Count)
                     : "memory");
#if !defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_SCHED_BARRIER)
        __builtin_amdgcn_sched_barrier(0);
#endif
    }

    CK_TILE_DEVICE static void WaitRrBlasSteadyLdsReads()
    {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SAFE_SYNC) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FULL_LDS_WAIT)
        WaitLdsReads();
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LDS_WAIT_COUNT)
        WaitLdsReads(
            number<CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LDS_WAIT_COUNT>{});
#else
        WaitLdsReads(number<8>{});
#endif
    }

    CK_TILE_DEVICE static void WaitRrBlasBoundaryLdsReads()
    {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_LDS_WAIT_COUNT)
        WaitLdsReads(
            number<CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_LDS_WAIT_COUNT>{});
#else
        WaitRrBlasSteadyLdsReads();
#endif
    }

    template <typename ADramBlockWindow,
              typename BDramBlockWindow,
              typename FinalEpiloguePreStore>
    CK_TILE_DEVICE auto Run(const ADramBlockWindow& a_dram_block_window,
                            const BDramBlockWindow& b_dram_block_window,
                            index_t num_loop,
                            void* p_smem,
                            [[maybe_unused]]
                            FinalEpiloguePreStore&& final_epilogue_pre_store) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindow::DataType>> &&
                std::is_same_v<BDataType,
                               remove_cvref_t<typename BDramBlockWindow::DataType>>,
            "A/B block-window data types must match the pipeline problem");

        auto ab_lds_blocks = PipelineImplBase{}.GetABLdsTensorViews(p_smem);
        auto& a_lds_block  = ab_lds_blocks.get(number<0>{});
        auto& b_lds_block  = ab_lds_blocks.get(number<1>{});

        // Triton's global producer mapping is not CK's generic warp-raked
        // mapping. For both aligned RC operands, thread tid owns X vector
        // tid[2:0] and rows tid>>3 + {0,64,128,192}. Put the iteration factor
        // outside wave/lane in the logical Y dimension so producer registers,
        // the swizzled LDS addresses below, and MMAC consumers agree exactly.
        constexpr auto triton_full_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<4, 8, 8>, sequence<8, 8>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});

        auto a_windows =
            PipelineImplBase{}.GetAWindows(a_dram_block_window, a_lds_block, triton_full_dstr);
        auto b_windows =
            PipelineImplBase{}.GetBWindows(b_dram_block_window, b_lds_block, triton_full_dstr);
        auto& a_dram_window = a_windows.get(number<0>{});
        auto& b_dram_window = b_windows.get(number<0>{});

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        constexpr ADramTileWindowStep a_step =
            IsARow ? make_array(0, KPerBlock)
                   : make_array(KPerBlock, 0);
        constexpr BDramTileWindowStep b_step =
            IsBRow ? make_array(KPerBlock, 0)
                   : make_array(0, KPerBlock);

#if defined(CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
        // The BLAS gfx936 NN producer uses buffer_load_dwordx4 ... lds for
        // each 16-byte K8 fragment. Reuse the exact Triton LDS swizzle below
        // while bypassing both the 64-byte/thread producer register tile and
        // its explicit DS writes. This first candidate deliberately refills a
        // complete K64 tile only after the previous tile is consumed; the
        // default overlapped VGPR producer remains unchanged.
        static_assert(sizeof(ADataType) == 2 && sizeof(BDataType) == 2,
                      "gfx936 direct G2L candidate requires 16-bit operands");

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0)
        // The RR direct producer always writes the same lane-owned 16-byte
        // destination in each K16 slot. Scalarize the two wave-local LDS
        // bases once instead of rebuilding a vector pointer and issuing
        // v_readfirstlane_b32 before every direct load.
        const index_t rr_tid  = get_thread_id();
        const index_t rr_wave = rr_tid >> 6;
        constexpr index_t RrStageBytes   = 16 * 1024;
        constexpr index_t RrOperandBytes = 8 * 1024;
        constexpr index_t RrWaveBytes = 64 * 16;
        // Dynamic LDS begins at byte zero. The first lane of wave W therefore
        // owns byte W*1024. Deriving that address from the wave-uniform tid>>6
        // value avoids creating a lane-varying local pointer that LLVM later
        // has to scalarize again at every unrolled stage.
        const uint32_t rr_wave_s = __builtin_amdgcn_readfirstlane(
            static_cast<uint32_t>(rr_wave));
        const uint32_t rr_lane_lds_base_s =
            rr_wave_s * static_cast<uint32_t>(RrWaveBytes);
        const uint32_t rr_a_m0_base =
            rr_lane_lds_base_s | (((rr_wave_s & 1u) * 4u) << 16);
        const uint32_t rr_b_m0_base =
            (rr_lane_lds_base_s +
             static_cast<uint32_t>(RrOperandBytes)) |
            ((rr_wave_s & 1u) << 16);
        // Keep the two bases in reserved SGPRs for the complete kernel. A
        // generic C++ SSA value loses its wave-uniform register class after
        // inlining and causes the backend to insert readfirstlane before every
        // use. Fixed local register variables preserve the vendor-style
        // prologue-once / hot-loop-s_add contract.
        register uint32_t rr_a_m0_base_reg asm("s60") = rr_a_m0_base;
        register uint32_t rr_b_m0_base_reg asm("s61") = rr_b_m0_base;
#else
        constexpr uint32_t rr_a_m0_base_reg = 0;
        constexpr uint32_t rr_b_m0_base_reg = 0;
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_A_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_B_RESOURCE)
        // hipBLASLt advances both SRD bases immediately after every K16 load
        // pair. Keep the lane offsets invariant across the entire K loop and
        // carry the two mutable SGPR resources across the unrolled stage
        // calls, rather than rebuilding a stage-dependent vector voffset.
        [[maybe_unused]] int32x4_t rr_stream_a_resource{};
        [[maybe_unused]] int32x4_t rr_stream_b_resource{};
        [[maybe_unused]] index_t rr_stream_a_k_stride = 0;
        [[maybe_unused]] index_t rr_stream_a_n_stride = 0;
        [[maybe_unused]] index_t rr_stream_b_row_bytes = 0;
        [[maybe_unused]] index_t rr_stream_a_stage_bytes = 0;
        [[maybe_unused]] index_t rr_stream_b_stage_bytes = 0;
        if constexpr(IsARow && IsBRow)
        {
            const auto rr_a_view   = a_dram_window.get_bottom_tensor_view();
            const auto rr_b_view   = b_dram_window.get_bottom_tensor_view();
            const auto rr_a_origin = a_dram_window.get_window_origin();
            const auto rr_b_origin = b_dram_window.get_window_origin();
            const auto& rr_a_desc  = rr_a_view.get_tensor_descriptor();
            const auto& rr_b_desc  = rr_b_view.get_tensor_descriptor();
            const auto& rr_a_buf   = rr_a_view.get_buffer_view();
            const auto& rr_b_buf   = rr_b_view.get_buffer_view();

            const index_t rr_a_base_element =
                rr_a_desc.calculate_offset(
                    make_multi_index(rr_a_origin[0], rr_a_origin[1]));
            const index_t rr_b_base_element =
                rr_b_desc.calculate_offset(
                    make_multi_index(rr_b_origin[0], rr_b_origin[1]));
            rr_stream_a_k_stride =
                rr_b_desc.calculate_offset(make_multi_index(1, 0)) -
                rr_b_desc.calculate_offset(make_multi_index(0, 0));
            rr_stream_a_n_stride =
                rr_b_desc.calculate_offset(make_multi_index(0, 1)) -
                rr_b_desc.calculate_offset(make_multi_index(0, 0));
            rr_stream_b_row_bytes =
                (rr_a_desc.calculate_offset(make_multi_index(1, 0)) -
                 rr_a_desc.calculate_offset(make_multi_index(0, 0))) *
                sizeof(ADataType);
            rr_stream_a_stage_bytes =
                KPerStage * rr_stream_a_k_stride * sizeof(BDataType);
            rr_stream_b_stage_bytes = KPerStage * sizeof(ADataType);

            const auto* rr_stream_a_ptr =
                rr_b_buf.p_data_ + rr_b_base_element;
            const auto* rr_stream_b_ptr =
                rr_a_buf.p_data_ + rr_a_base_element;
            const uint32_t rr_stream_a_remaining =
                rr_b_buf.buffer_size_ * sizeof(BDataType) -
                rr_b_base_element * sizeof(BDataType);
            const uint32_t rr_stream_b_remaining =
                rr_a_buf.buffer_size_ * sizeof(ADataType) -
                rr_a_base_element * sizeof(ADataType);
            rr_stream_a_resource =
                make_wave_buffer_resource(rr_stream_a_ptr,
                                          rr_stream_a_remaining);
            rr_stream_b_resource =
                make_wave_buffer_resource(rr_stream_b_ptr,
                                          rr_stream_b_remaining,
                                          rr_stream_b_row_bytes);
        }

        const auto advance_rr_stream_resource =
            [](int32x4_t& resource,
               uint32_t byte_shift,
               bool shrink_byte_range) {
                const uint64_t old_address =
                    static_cast<uint32_t>(resource.x) |
                    (static_cast<uint64_t>(
                         static_cast<uint32_t>(resource.y))
                     << 32);
                const uint64_t new_address = old_address + byte_shift;
                resource.x = static_cast<int32_t>(new_address);
                resource.y = static_cast<int32_t>(new_address >> 32);
                if(shrink_byte_range)
                {
                    const uint32_t old_range =
                        static_cast<uint32_t>(resource.z);
                    resource.z = static_cast<int32_t>(
                        old_range > byte_shift ? old_range - byte_shift : 0);
                }
            };
#endif

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD)
        // Fixed scalar SRDs for the exact BLAS-style recurrence. The generic
        // int32x4_t variant above is lowered through VGPRs and reconstructs
        // its resources with v_readfirstlane in every K16 stage.
        [[maybe_unused]] int32x4_t rr_fixed_a_resource{};
        [[maybe_unused]] int32x4_t rr_fixed_b_resource{};
        [[maybe_unused]] index_t rr_fixed_a_k_stride    = 0;
        [[maybe_unused]] index_t rr_fixed_a_n_stride    = 0;
        [[maybe_unused]] index_t rr_fixed_a_stage_bytes = 0;
        [[maybe_unused]] index_t rr_fixed_b_stage_bytes = 0;
        if constexpr(IsARow && IsBRow)
        {
            const auto rr_a_view   = a_dram_window.get_bottom_tensor_view();
            const auto rr_b_view   = b_dram_window.get_bottom_tensor_view();
            const auto rr_a_origin = a_dram_window.get_window_origin();
            const auto rr_b_origin = b_dram_window.get_window_origin();
            const auto& rr_a_desc  = rr_a_view.get_tensor_descriptor();
            const auto& rr_b_desc  = rr_b_view.get_tensor_descriptor();
            const auto& rr_a_buf   = rr_a_view.get_buffer_view();
            const auto& rr_b_buf   = rr_b_view.get_buffer_view();

            const index_t rr_a_base_element =
                rr_a_desc.calculate_offset(
                    make_multi_index(rr_a_origin[0], rr_a_origin[1]));
            const index_t rr_b_base_element =
                rr_b_desc.calculate_offset(
                    make_multi_index(rr_b_origin[0], rr_b_origin[1]));
            rr_fixed_a_k_stride =
                rr_b_desc.calculate_offset(make_multi_index(1, 0)) -
                rr_b_desc.calculate_offset(make_multi_index(0, 0));
            rr_fixed_a_n_stride =
                rr_b_desc.calculate_offset(make_multi_index(0, 1)) -
                rr_b_desc.calculate_offset(make_multi_index(0, 0));
            const index_t rr_fixed_b_row_bytes =
                (rr_a_desc.calculate_offset(make_multi_index(1, 0)) -
                 rr_a_desc.calculate_offset(make_multi_index(0, 0))) *
                sizeof(ADataType);
            rr_fixed_a_stage_bytes =
                KPerStage * rr_fixed_a_k_stride * sizeof(BDataType);
            rr_fixed_b_stage_bytes = KPerStage * sizeof(ADataType);

            const auto* rr_fixed_a_ptr =
                rr_b_buf.p_data_ + rr_b_base_element;
            const auto* rr_fixed_b_ptr =
                rr_a_buf.p_data_ + rr_a_base_element;
            const uint32_t rr_fixed_a_remaining =
                rr_b_buf.buffer_size_ * sizeof(BDataType) -
                rr_b_base_element * sizeof(BDataType);
            const uint32_t rr_fixed_b_remaining =
                rr_a_buf.buffer_size_ * sizeof(ADataType) -
                rr_a_base_element * sizeof(ADataType);
            rr_fixed_a_resource =
                make_wave_buffer_resource(rr_fixed_a_ptr,
                                          rr_fixed_a_remaining);
            rr_fixed_b_resource =
                make_wave_buffer_resource(rr_fixed_b_ptr,
                                          rr_fixed_b_remaining,
                                          rr_fixed_b_row_bytes);
        }

        const index_t rr_fixed_tid         = get_thread_id();
        const index_t rr_fixed_lane        = rr_fixed_tid & 63;
        const index_t rr_fixed_wave        = rr_fixed_tid >> 6;
        const index_t rr_fixed_half_lane   = rr_fixed_lane >> 1;
        const index_t rr_fixed_a_m =
            ((rr_fixed_half_lane >> 3) << 4) +
            (rr_fixed_half_lane & 3) +
            (((rr_fixed_half_lane >> 2) & 1) << 3) +
            ((rr_fixed_wave & 1) << 2) + ((rr_fixed_wave >> 1) << 6);
        const index_t rr_fixed_b_k_linear = rr_fixed_tid >> 5;
        const index_t rr_fixed_b_n        = (rr_fixed_tid & 31) * 8;
        const index_t rr_fixed_a_voffset =
            (((rr_fixed_b_k_linear & ~3) +
              ((rr_fixed_b_k_linear & 1) << 1) +
              ((rr_fixed_b_k_linear & 2) >> 1)) *
                 rr_fixed_a_k_stride +
             rr_fixed_b_n * rr_fixed_a_n_stride) *
            sizeof(BDataType);
        const int32x2_t rr_fixed_b_offsets = {
            static_cast<int32_t>(rr_fixed_a_m),
            static_cast<int32_t>(
                (rr_fixed_lane & 1) * 8 * sizeof(ADataType))};
        const uint32_t rr_fixed_a_stage_bytes_s =
            __builtin_amdgcn_readfirstlane(
                static_cast<uint32_t>(rr_fixed_a_stage_bytes));
        const uint32_t rr_fixed_b_stage_bytes_s =
            __builtin_amdgcn_readfirstlane(
                static_cast<uint32_t>(rr_fixed_b_stage_bytes));
        register uint32_t rr_fixed_a_stage_bytes_reg asm("s62") =
            rr_fixed_a_stage_bytes_s;
        register uint32_t rr_fixed_b_stage_bytes_reg asm("s63") =
            rr_fixed_b_stage_bytes_s;
#define CK_TILE_RR_FIXED_STREAM_LOAD(IK_STAGE, A_M0_BASE, B_M0_BASE)               \
        do                                                                         \
        {                                                                          \
            asm volatile(                                                          \
                "s_add_u32 m0, s60, %4;\n\t"                                      \
                "buffer_load_dwordx4 %2, s[44:47], 0 offen offset:0 lds;\n\t"      \
                "s_add_u32 m0, s61, %4;\n\t"                                      \
                "buffer_load_dwordx4 %3, s[48:51], 0 idxen offen offset:0 lds;\n\t" \
                "s_add_u32 s44, s44, s62;\n\t"                                    \
                "s_addc_u32 s45, s45, 0;\n\t"                                     \
                "s_sub_u32 s46, s46, s62;\n\t"                                    \
                "s_add_u32 s48, s48, s63;\n\t"                                    \
                "s_addc_u32 s49, s49, 0;\n\t"                                     \
                : "+{s[44:47]}"(rr_fixed_a_resource),                              \
                  "+{s[48:51]}"(rr_fixed_b_resource)                               \
                : "v"(rr_fixed_a_voffset),                                         \
                  "v"(rr_fixed_b_offsets),                                         \
                  "n"((IK_STAGE) * RrStageBytes),                                  \
                  "{s62}"(rr_fixed_a_stage_bytes_reg),                             \
                  "{s63}"(rr_fixed_b_stage_bytes_reg),                             \
                  "{s60}"(rr_a_m0_base_reg),                                       \
                  "{s61}"(rr_b_m0_base_reg)                                        \
                : "memory");                                                       \
        } while(false)
#endif

        [[maybe_unused]] const auto direct_load_rr_stage =
            [&](auto iKStage,
                [[maybe_unused]] uint32_t rr_a_m0_base_arg,
                [[maybe_unused]] uint32_t rr_b_m0_base_arg) {
            if constexpr(IsARow && IsBRow)
            {
                const auto a_view   = a_dram_window.get_bottom_tensor_view();
                const auto b_view   = b_dram_window.get_bottom_tensor_view();
                const auto a_origin = a_dram_window.get_window_origin();
                const auto b_origin = b_dram_window.get_window_origin();
                const auto& a_desc  = a_view.get_tensor_descriptor();
                const auto& b_desc  = b_view.get_tensor_descriptor();
                const auto& a_buf   = a_view.get_buffer_view();
                const auto& b_buf   = b_view.get_buffer_view();

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_B_IDXEN) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_B_RESOURCE)
                // The custom WGM8 kernel supplies the row-major A source as
                // vindex + K-byte voffset through a structured resource. This
                // enables the same descriptor cache-swizzle path instead of
                // flattening row*K into one VALU-generated byte offset.
                const index_t b_const_stride =
                    a_desc.get_length(number<1>{}) * sizeof(ADataType);
                [[maybe_unused]] const auto a_resource =
                    make_wave_buffer_resource(a_buf.p_data_,
                                              a_buf.buffer_size_ * sizeof(ADataType),
                                              b_const_stride);
#else
                [[maybe_unused]] const auto a_resource =
                    make_wave_buffer_resource(a_buf.p_data_,
                                              a_buf.buffer_size_ * sizeof(ADataType));
#endif
                [[maybe_unused]] const auto b_resource =
                    make_wave_buffer_resource(b_buf.p_data_,
                                              b_buf.buffer_size_ * sizeof(BDataType));

                const index_t tid   = get_thread_id();
                const index_t lane  = tid & 63;
                const index_t wave  = tid >> 6;
                const index_t half_lane = lane >> 1;
                const index_t a_m =
                    ((half_lane >> 3) << 4) + (half_lane & 3) +
                    (((half_lane >> 2) & 1) << 3) + ((wave & 1) << 2) +
                    ((wave >> 1) << 6);
                const index_t a_k = iKStage * KPerStage + (lane & 1) * 8;
                const index_t b_k_linear = tid >> 5;
                const index_t b_k =
                    iKStage * KPerStage + (b_k_linear & ~3) +
                    ((b_k_linear & 1) << 1) + ((b_k_linear & 2) >> 1);
                const index_t b_n = (tid & 31) * 8;

                [[maybe_unused]] const index_t a_element_offset =
                    b_desc.calculate_offset(
                        make_multi_index(b_origin[0] + b_k,
                                         b_origin[1] + b_n));
                [[maybe_unused]] const index_t b_element_offset =
                    a_desc.calculate_offset(
                        make_multi_index(a_origin[0] + a_m,
                                         a_origin[1] + a_k));

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_A_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_B_RESOURCE)
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_A_RESOURCE)
                auto& direct_a_resource = rr_stream_a_resource;
                const index_t direct_a_element_offset =
                    ((b_k_linear & ~3) +
                     ((b_k_linear & 1) << 1) +
                     ((b_k_linear & 2) >> 1)) *
                        rr_stream_a_k_stride +
                    b_n * rr_stream_a_n_stride;
#else
                const auto& direct_a_resource = b_resource;
                const index_t direct_a_element_offset = a_element_offset;
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_B_RESOURCE)
                auto& direct_b_resource = rr_stream_b_resource;
                constexpr index_t direct_b_element_offset = 0;
#else
                const auto& direct_b_resource = a_resource;
                const index_t direct_b_element_offset = b_element_offset;
#endif
                [[maybe_unused]] constexpr index_t direct_a_soffset = 0;
                [[maybe_unused]] constexpr index_t direct_b_soffset = 0;
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_SOFFSET)
                // Keep the lane-varying voffset stable and move the uniform
                // K16 displacement out of it, matching the vendor kernel's
                // hot-loop address split. The outer K64 window origin remains
                // in voffset and is advanced by the existing tile-window
                // logic.
                const index_t a_stage_element_stride =
                    b_desc.calculate_offset(
                        make_multi_index(b_origin[0] + b_k + KPerStage,
                                         b_origin[1] + b_n)) -
                    a_element_offset;
                const index_t b_stage_element_stride =
                    a_desc.calculate_offset(
                        make_multi_index(a_origin[0] + a_m,
                                         a_origin[1] + a_k + KPerStage)) -
                    b_element_offset;
                const index_t a_resource_element_shift =
                    iKStage * a_stage_element_stride;
                const index_t b_resource_element_shift =
                    iKStage * b_stage_element_stride;
                const index_t a_resource_byte_shift =
                    a_resource_element_shift * sizeof(ADataType);
                const index_t b_resource_byte_shift =
                    b_resource_element_shift * sizeof(BDataType);
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_RESOURCE)
                const auto direct_a_resource = make_wave_buffer_resource(
                    reinterpret_cast<const char*>(b_buf.p_data_) +
                        a_resource_byte_shift,
                    b_buf.buffer_size_ * sizeof(BDataType) -
                        a_resource_byte_shift);
                const auto direct_b_resource = make_wave_buffer_resource(
                    reinterpret_cast<const char*>(a_buf.p_data_) +
                        b_resource_byte_shift,
                    a_buf.buffer_size_ * sizeof(ADataType) -
                        b_resource_byte_shift);
                const index_t direct_a_element_offset =
                    a_element_offset - a_resource_element_shift;
                const index_t direct_b_element_offset =
                    b_element_offset - b_resource_element_shift;
                [[maybe_unused]] constexpr index_t direct_a_soffset = 0;
                [[maybe_unused]] constexpr index_t direct_b_soffset = 0;
#else
                // Prefer the buffer instruction's scalar soffset field over
                // rebuilding four resource descriptors. A's K16 step is
                // 128 KiB for the production shape, so it cannot use the
                // instruction's 12-bit immediate offset field.
                const auto& direct_a_resource = b_resource;
                const auto& direct_b_resource = a_resource;
                const index_t direct_a_element_offset =
                    a_element_offset - a_resource_element_shift;
                const index_t direct_b_element_offset =
                    b_element_offset - b_resource_element_shift;
                const index_t direct_a_soffset = a_resource_byte_shift;
                const index_t direct_b_soffset = b_resource_byte_shift;
#endif
#else
                const auto& direct_a_resource = b_resource;
                const auto& direct_b_resource = a_resource;
                const index_t direct_a_element_offset = a_element_offset;
                const index_t direct_b_element_offset = b_element_offset;
                [[maybe_unused]] constexpr index_t direct_a_soffset = 0;
                [[maybe_unused]] constexpr index_t direct_b_soffset = 0;
#endif

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SCALAR_M0)
                // Do not route this through
                // hcu_async_buffer_load_asm_impl: that general helper
                // scalarizes m0 internally, which would recreate the exact
                // v_readfirstlane instruction this candidate removes.
                const index_t a_voffset =
                    direct_a_element_offset * sizeof(ADataType);
                [[maybe_unused]] const index_t b_voffset =
                    direct_b_element_offset * sizeof(BDataType);
                asm volatile(
                    "s_add_u32 m0, %0, %3; \n\t"
                    "buffer_load_dwordx4 %1, %2, %4 offen offset:0 lds;\n\t"
                    :
                    : "s"(rr_a_m0_base_arg), "v"(a_voffset),
                      "s"(direct_a_resource),
                      "n"(iKStage * RrStageBytes),
                      "s"(direct_a_soffset)
                    : "memory");
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_B_IDXEN) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_B_RESOURCE)
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_B_RESOURCE)
                const int32x2_t b_offsets = {
                    static_cast<int32_t>(a_m),
                    static_cast<int32_t>(
                        (lane & 1) * 8 * sizeof(ADataType))};
#else
                const int32x2_t b_offsets = {
                    static_cast<int32_t>(a_origin[0] + a_m),
                    static_cast<int32_t>(
                        (a_origin[1] + a_k) * sizeof(ADataType))};
#endif
                asm volatile(
                    "s_add_u32 m0, %0, %3; \n\t"
                    "buffer_load_dwordx4 %1, %2, %4 idxen offen offset:0 lds;\n\t"
                    :
                    : "s"(rr_b_m0_base_arg), "v"(b_offsets),
                      "s"(direct_b_resource),
                      "n"(iKStage * RrStageBytes),
                      "s"(direct_b_soffset)
                    : "memory");
#else
                asm volatile(
                    "s_add_u32 m0, %0, %3; \n\t"
                    "buffer_load_dwordx4 %1, %2, %4 offen offset:0 lds;\n\t"
                    :
                    : "s"(rr_b_m0_base_arg), "v"(b_voffset),
                      "s"(direct_b_resource),
                      "n"(iKStage * RrStageBytes),
                      "s"(direct_b_soffset)
                    : "memory");
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_A_RESOURCE)
                // Match the solution ELF: update raw-A base/range by the
                // K16*N byte stride and structured-B base by 32 bytes between
                // each load pair, while the current VMEM transactions are in
                // flight. The following LDS wait and vmcnt(4) provide the
                // same useful instruction distance as the vendor loop.
                advance_rr_stream_resource(
                    rr_stream_a_resource,
                    static_cast<uint32_t>(rr_stream_a_stage_bytes),
                    true);
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STREAM_B_RESOURCE)
                advance_rr_stream_resource(
                    rr_stream_b_resource,
                    static_cast<uint32_t>(rr_stream_b_stage_bytes),
                    false);
#endif
#else
                auto* lds_ptr = a_lds_block.get_buffer_view().p_data_;
                constexpr index_t StageBytes   = 16 * 1024;
                constexpr index_t OperandBytes = 8 * 1024;
                const index_t stage_byte       = iKStage * StageBytes;
                const index_t a_lds_wrap       = (wave & 1) * 4;
                const index_t b_lds_wrap       = wave & 1;
                hcu_async_buffer_load_asm<ADataType, 8, false>(
                    lds_ptr + (stage_byte + tid * 16) / sizeof(ADataType),
                    direct_a_resource,
                    direct_a_element_offset * sizeof(ADataType),
                    0,
                    a_lds_wrap,
                    true,
                    bool_constant<false>{});
                hcu_async_buffer_load_asm<BDataType, 8, false>(
                    lds_ptr +
                        (stage_byte + OperandBytes + tid * 16) /
                            sizeof(BDataType),
                    direct_b_resource,
                    direct_b_element_offset * sizeof(BDataType),
                    0,
                    b_lds_wrap,
                    true,
                    bool_constant<false>{});
#endif
            }
        };
#endif

        [[maybe_unused]] const auto direct_load_full_tile = [&]() {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL)
            if constexpr(IsARow && !IsBRow)
            {
                const auto a_view   = a_dram_window.get_bottom_tensor_view();
                const auto b_view   = b_dram_window.get_bottom_tensor_view();
                const auto a_origin = a_dram_window.get_window_origin();
                const auto b_origin = b_dram_window.get_window_origin();
                const auto& a_desc  = a_view.get_tensor_descriptor();
                const auto& b_desc  = b_view.get_tensor_descriptor();
                const auto& a_buf   = a_view.get_buffer_view();
                const auto& b_buf   = b_view.get_buffer_view();

                const auto a_resource =
                    make_wave_buffer_resource(a_buf.p_data_,
                                              a_buf.buffer_size_ * sizeof(ADataType));
                const auto b_resource =
                    make_wave_buffer_resource(b_buf.p_data_,
                                              b_buf.buffer_size_ * sizeof(BDataType));

                auto* a_lds_ptr = a_lds_block.get_buffer_view().p_data_;
                auto* b_lds_ptr = b_lds_block.get_buffer_view().p_data_;

                const index_t tid     = get_thread_id();
                const index_t row     = tid >> 1;
                const index_t k8_slot = tid & 1;

                // BLAS-style stage-major LDS: a K16 stage is 8 KiB and
                // thread tid owns row tid/2, K8 fragment tid%2. Within every
                // wave, the 64 dwordx4 destinations are therefore contiguous.
                static_for<0, NumKStages, 1>{}([&](auto iKStage) {
                    const index_t k =
                        iKStage * KPerStage + k8_slot * 8;
                    const index_t lds_byte =
                        iKStage * 8192 + tid * 16;

                    const index_t a_element_offset =
                        a_desc.calculate_offset(
                            make_multi_index(a_origin[0] + row,
                                             a_origin[1] + k));
                    const index_t b_element_offset =
                        b_desc.calculate_offset(
                            make_multi_index(b_origin[0] + row,
                                             b_origin[1] + k));

                    hcu_async_buffer_load_asm<ADataType, 8, false>(
                        a_lds_ptr + lds_byte / sizeof(ADataType),
                        a_resource,
                        a_element_offset * sizeof(ADataType),
                        0,
                        true,
                        bool_constant<false>{});
                    hcu_async_buffer_load_asm<BDataType, 8, false>(
                        b_lds_ptr + lds_byte / sizeof(BDataType),
                        b_resource,
                        b_element_offset * sizeof(BDataType),
                        0,
                        true,
                        bool_constant<false>{});
                });

                buffer_load_fence(0);
            }
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
            if constexpr(IsARow && IsBRow)
            {
                // hipBLASLt BF16 NN uses four 16-K stages. Each stage owns
                // 8 KiB of A followed by 8 KiB of B. The direct-load m0
                // bases reduce to stage_base + tid * 16 for A and
                // stage_base + 8 KiB + tid * 16 for B.
                static_for<0, NumKStages, 1>{}([&](auto iKStage) {
                    direct_load_rr_stage(
                        iKStage, rr_a_m0_base_reg, rr_b_m0_base_reg);
                });

                buffer_load_fence(0);
            }
#endif
        };
#endif

        [[maybe_unused]] constexpr auto b_stage_dstr =
            make_static_tile_distribution(StageBlockGemm::MakeBBlockDistributionEncode());
        [[maybe_unused]] constexpr auto a_stage_dstr =
            make_static_tile_distribution(StageBlockGemm::MakeABlockDistributionEncode());

        const auto store_packed_a = [&](const auto& a_tile) {
            // Exact Triton local_alloc swizzle for
            // vec=4/perPhase=1/maxPhase=16. The warp-raked global tiles hold
            // four half8 loads per operand; view them as eight half4 values
            // and pair the 8/16/24-KiB-separated destinations with st64.
            using AVec4 = ext_vector_t<ADataType, 4>;
            static_assert(remove_cvref_t<decltype(a_tile)>::get_thread_buffer_size() == 32,
                          "Triton input mapping requires 32 elements per "
                          "operand and thread");

            const auto& a_vecs =
                a_tile.get_thread_buffer().template get_as<AVec4>();
            const index_t tid = get_thread_id();
            const index_t store_base =
                ((tid << 4) & 8176) ^ (tid & 120);

            // CK's register buffer is [load0.low, load0.high, ...],
            // whereas Triton stores [all four lows, all four highs].
            StoreLdsPairSt64<0, 16>(store_base, a_vecs[0], a_vecs[2]);
            StoreLdsPairSt64<32, 48>(store_base, a_vecs[4], a_vecs[6]);
            StoreLdsPairSt64<0, 16>(store_base ^ 8, a_vecs[1], a_vecs[3]);
            StoreLdsPairSt64<32, 48>(store_base ^ 8, a_vecs[5], a_vecs[7]);
        };

        const auto store_packed_b = [&](const auto& b_tile) {
            using BVec4 = ext_vector_t<BDataType, 4>;
            static_assert(remove_cvref_t<decltype(b_tile)>::get_thread_buffer_size() == 32,
                          "Triton input mapping requires 32 elements per "
                          "operand and thread");
            const auto& b_vecs =
                b_tile.get_thread_buffer().template get_as<BVec4>();
            const index_t tid = get_thread_id();
            const index_t store_base =
                ((tid << 4) & 8176) ^ (tid & 120);
            // Fold B's 32-KiB half into the st64 immediate exactly as
            // Triton does. A and B now share the same two address VGPRs.
            StoreLdsPairSt64<64, 80>(
                store_base, b_vecs[0], b_vecs[2]);
            StoreLdsPairSt64<96, 112>(
                store_base, b_vecs[4], b_vecs[6]);
            StoreLdsPairSt64<64, 80>(
                store_base ^ 8, b_vecs[1], b_vecs[3]);
            StoreLdsPairSt64<96, 112>(
                store_base ^ 8, b_vecs[5], b_vecs[7]);
        };

        const auto store_a_transposed = [&](const auto& tile) {
            using Vec4 = ext_vector_t<ADataType, 4>;
            static_assert(remove_cvref_t<decltype(tile)>::get_thread_buffer_size() == 32,
                          "transposed producer requires eight K4 vectors");
            const auto& vecs = tile.get_thread_buffer().template get_as<Vec4>();
            const index_t tid = get_thread_id();
            const index_t m0  = (tid & 31) * 8;
            const index_t k   = (tid >> 5) * 4;
            static_for<0, 8, 1>{}([&](auto i) {
                const index_t m = m0 + i;
                const index_t concept_tid = (m % 64) * 8 + k / 8;
                const index_t base =
                    ((concept_tid << 4) & 8176) ^ (concept_tid & 120);
                const index_t byte_address =
                    (base ^ ((k % 8) / 4) * 8) + (m / 64) * 8192;
                StoreLdsB64(byte_address, vecs[i]);
            });
        };

        const auto store_b_transposed = [&](const auto& tile) {
            using Vec4 = ext_vector_t<BDataType, 4>;
            static_assert(remove_cvref_t<decltype(tile)>::get_thread_buffer_size() == 32,
                          "transposed producer requires eight K4 vectors");
            const auto& vecs = tile.get_thread_buffer().template get_as<Vec4>();
            const index_t tid = get_thread_id();
            const index_t n0  = (tid & 31) * 8;
            const index_t k   = (tid >> 5) * 4;
            static_for<0, 8, 1>{}([&](auto i) {
                const index_t n = n0 + i;
                const index_t concept_tid = (n % 64) * 8 + k / 8;
                const index_t base =
                    ((concept_tid << 4) & 8176) ^ (concept_tid & 120);
                const index_t byte_address =
                    32768 + (base ^ ((k % 8) / 4) * 8) +
                    (n / 64) * 8192;
                StoreLdsB64(byte_address, vecs[i]);
            });
        };

        const auto store_full_tile = [&](const auto& a_tile, const auto& b_tile) {
            if constexpr(IsARow)
            {
                store_packed_a(a_tile);
            }
            else
            {
                auto a_shuffled = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffled, a_tile);
                store_a_transposed(a_shuffled);
            }

            if constexpr(!IsBRow)
            {
                store_packed_b(b_tile);
            }
            else
            {
                auto b_shuffled = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffled, b_tile);
                store_b_transposed(b_shuffled);
            }
        };

        const auto stage_block_gemm = StageBlockGemm{};
        auto c_block_tile           = stage_block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS)
        // Reserve the same contiguous accumulator bank used by the
        // hipBLASLt WGM8 kernel. DTK clang crashes when a single 128-wide
        // inline-asm operand is lowered, so expose the bank as 32 fixed
        // fp32x4 operands instead.
        [[maybe_unused]] fp32x4_t rr_c00 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c01 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c02 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c03 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c04 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c05 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c06 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c07 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c08 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c09 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c10 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c11 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c12 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c13 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c14 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c15 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c16 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c17 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c18 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c19 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c20 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c21 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c22 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c23 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c24 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c25 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c26 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c27 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c28 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c29 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c30 = fp32x4_t{};
        [[maybe_unused]] fp32x4_t rr_c31 = fp32x4_t{};
#endif

#if defined(CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL)
        if constexpr(IsARow && !IsBRow)
        {
            direct_load_full_tile();
            move_tile_window(a_dram_window, a_step);
            move_tile_window(b_dram_window, b_step);
        }
        else
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
        if constexpr(IsARow && IsBRow)
        {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD)
            CK_TILE_RR_FIXED_STREAM_LOAD(
                number<0>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            CK_TILE_RR_FIXED_STREAM_LOAD(
                number<1>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            CK_TILE_RR_FIXED_STREAM_LOAD(
                number<2>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            CK_TILE_RR_FIXED_STREAM_LOAD(
                number<3>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            buffer_load_fence(0);
#else
            direct_load_full_tile();
#endif
            move_tile_window(a_dram_window, a_step);
            move_tile_window(b_dram_window, b_step);
        }
        else
#endif
        {
            auto a_current_global = load_tile(a_dram_window);
            auto b_current_global = load_tile(b_dram_window);
            move_tile_window(a_dram_window, a_step);
            move_tile_window(b_dram_window, b_step);
            store_full_tile(a_current_global, b_current_global);
        }
        block_sync_lds();

        [[maybe_unused]] const auto get_a_load_base = [&](auto iKStage) {
            const index_t tid = get_thread_id();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL)
            if constexpr(IsARow && !IsBRow)
            {
                const index_t row0 =
                    (tid & 15) + ((tid >> 8) & 1) * 16;
                const index_t k4 = ((tid >> 4) & 3) * 4;
                return iKStage * number<8192>{} + row0 * 32 + k4 * 2;
            }
            else
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
            if constexpr(IsARow && IsBRow)
            {
                const index_t lane = tid & 63;
                const index_t wave = tid >> 6;
                const index_t lane7 = lane & 7;
                index_t base =
                    ((lane7 & 3) << 3) +
                    ((lane7 >> 2) << 5) +
                    ((lane7 >> 2) << 9) +
                    (((lane & 15) >> 3) << 8) +
                    ((lane >> 4) << 10) +
                    ((wave & 1) << 5);
                return iKStage * number<16384>{} + base * 2;
            }
            else
#endif
            {
                return ((((tid & 15) << 7) | ((tid << 3) & 2168)) ^
                        ((tid >> 1) & 24)) ^
                       (iKStage * number<32>{});
            }
        };

        [[maybe_unused]] const auto get_b_load_base = [&](auto iKStage) {
            const index_t tid = get_thread_id();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL)
            if constexpr(IsARow && !IsBRow)
            {
                const index_t row0 =
                    (tid & 15) + ((tid >> 6) & 3) * 16;
                const index_t k4 = ((tid >> 4) & 3) * 4;
                return 32768 + iKStage * number<8192>{} +
                       row0 * 32 + k4 * 2;
            }
            else
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
            if constexpr(IsARow && IsBRow)
            {
                const index_t lane = tid & 63;
                const index_t wave = tid >> 6;
                const index_t lane7 = lane & 7;
                const index_t lane8 = lane7 >> 2;
                const index_t wave2 = wave >> 1;
                const index_t base =
                    ((lane7 & 3) << 4) +
                    (lane8 << 3) +
                    (lane8 << 9) +
                    (((lane >> 3) & 1) << 6) +
                    ((lane >> 4) << 2) +
                    ((wave2 & 1) << 8) +
                    ((wave2 >> 1) << 10);
                return iKStage * number<16384>{} + number<8192>{} +
                       base * 2;
            }
            else
#endif
            {
                return (((tid << 5) & 6144) |
                        (((tid & 15) << 3) ^ ((tid >> 1) & 24)) |
                        ((tid & 15) << 7)) ^
                       (iKStage * number<32>{});
            }
        };

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_LDS_BASES)
        // The vendor WGM8 loop keeps four LDS consumer addresses live across
        // the whole K loop. Compute the lane/wave swizzle and exceptional
        // wrap once, then add only the compile-time K16 stage displacement.
        [[maybe_unused]] const index_t rr_a_load_base0 = [&]() {
            if constexpr(IsARow && IsBRow)
            {
                return get_a_load_base(number<0>{});
            }
            else
            {
                return index_t{0};
            }
        }();
        [[maybe_unused]] const index_t rr_b_load_base0 = [&]() {
            if constexpr(IsARow && IsBRow)
            {
                return get_b_load_base(number<0>{});
            }
            else
            {
                return index_t{0};
            }
        }();
        [[maybe_unused]] const index_t rr_a_alt_base0 = [&]() {
            if constexpr(IsARow && IsBRow)
            {
                const index_t tid  = get_thread_id();
                const index_t lane = tid & 63;
                const index_t wave = tid >> 6;
                const bool a_wrap =
                    (((lane >> 2) & 3) == 3) && ((wave & 1) == 1);
                return rr_a_load_base0 - (a_wrap ? 1024 : 0);
            }
            else
            {
                return index_t{0};
            }
        }();
        [[maybe_unused]] const index_t rr_b_alt_base0 = [&]() {
            if constexpr(IsARow && IsBRow)
            {
                const index_t tid  = get_thread_id();
                const index_t lane = tid & 63;
                const index_t wave = tid >> 6;
                const bool b_wrap =
                    (lane == 47 || lane == 63) &&
                    (((wave >> 1) & 1) == 1);
                return rr_b_load_base0 - (b_wrap ? 1024 : 0);
            }
            else
            {
                return index_t{0};
            }
        }();
#endif

        [[maybe_unused]] const auto load_lds_stage =
            [&](auto iKStage,
                auto& a_stage_tile,
                auto& b_stage_tile) {
            static_assert(
                remove_cvref_t<decltype(a_stage_tile)>::get_thread_buffer_size() == 32 &&
                    remove_cvref_t<decltype(b_stage_tile)>::get_thread_buffer_size() == 16,
                "native M16xN16 stage requires eight A and four B half4 operands");

            // Exact Triton MMAC-consumer layout. A repeats are 4 KiB apart,
            // B repeats are 8 KiB apart, and successive K16 stages select
            // XOR-32 swizzle phases.
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_LDS_BASES)
            const index_t rr_stage_byte =
                iKStage * number<16384>{};
            const index_t a_load_base =
                rr_a_load_base0 + rr_stage_byte;
            const index_t b_load_base =
                rr_b_load_base0 + rr_stage_byte;
#else
            const index_t a_load_base = get_a_load_base(iKStage);
            const index_t b_load_base = get_b_load_base(iKStage);
#endif

            auto& a_pairs =
                a_stage_tile.get_thread_buffer().template get_as<fp32x4_t>();
            auto& b_pairs =
                b_stage_tile.get_thread_buffer().template get_as<fp32x4_t>();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL)
            if constexpr(IsARow && !IsBRow)
            {
                a_pairs[0] = LoadLdsPairSt64<0, 2>(a_load_base);
                a_pairs[1] = LoadLdsPairSt64<4, 6>(a_load_base);
                a_pairs[2] = LoadLdsPairSt64<8, 10>(a_load_base);
                a_pairs[3] = LoadLdsPairSt64<12, 14>(a_load_base);
                b_pairs[0] = LoadLdsPairSt64<0, 4>(b_load_base);
                b_pairs[1] = LoadLdsPairSt64<8, 12>(b_load_base);
            }
            else
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
            if constexpr(IsARow && IsBRow)
            {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_LDS_BASES)
                const index_t a_alt_base =
                    rr_a_alt_base0 + rr_stage_byte;
                const index_t b_alt_base =
                    rr_b_alt_base0 + rr_stage_byte;
#else
                const index_t tid = get_thread_id();
                const index_t lane = tid & 63;
                const index_t wave = tid >> 6;
                const bool a_wrap =
                    (((lane >> 2) & 3) == 3) && ((wave & 1) == 1);
                const bool b_wrap =
                    (lane == 47 || lane == 63) &&
                    (((wave >> 1) & 1) == 1);
                const index_t a_alt_base =
                    a_load_base - (a_wrap ? 1024 : 0);
                const index_t b_alt_base =
                    b_load_base - (b_wrap ? 1024 : 0);
#endif

                auto& b_quads =
                    b_stage_tile.get_thread_buffer().template get_as<fp32x2_t>();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LDS_ASM_V7)
                // Keep the complete next-stage LDS read packet opaque to the
                // machine scheduler. Stage parity selects the same two fixed
                // operand banks used by the validated V3/V6 consumer:
                // even A=v128:v143/B=v160:v167, odd
                // A=v144:v159/B=v168:v175.
                if constexpr(iKStage == number<0>{} ||
                             iKStage == number<2>{})
                {
                    asm volatile(
                        "ds_read_m32x16_b16 %0, %8 offset:0\n\t"
                        "ds_read_m32x16_b16 %1, %8 offset:128\n\t"
                        "ds_read_m32x16_b16 %2, %8 offset:256\n\t"
                        "ds_read_m32x16_b16 %3, %9 offset:384\n\t"
                        "ds_read_b64 %4, %10 offset:0\n\t"
                        "ds_read_b64 %5, %11 offset:256\n\t"
                        "ds_read_b64 %6, %10 offset:4096\n\t"
                        "ds_read_b64 %7, %11 offset:4352"
                        : "=&{v[128:131]}"(a_pairs[0]),
                          "=&{v[132:135]}"(a_pairs[1]),
                          "=&{v[136:139]}"(a_pairs[2]),
                          "=&{v[140:143]}"(a_pairs[3]),
                          "=&{v[160:161]}"(b_quads[0]),
                          "=&{v[162:163]}"(b_quads[1]),
                          "=&{v[164:165]}"(b_quads[2]),
                          "=&{v[166:167]}"(b_quads[3])
                        : "v"(a_load_base),
                          "v"(a_alt_base),
                          "v"(b_load_base),
                          "v"(b_alt_base)
                        : "memory");
                }
                else
                {
                    asm volatile(
                        "ds_read_m32x16_b16 %0, %8 offset:0\n\t"
                        "ds_read_m32x16_b16 %1, %8 offset:128\n\t"
                        "ds_read_m32x16_b16 %2, %8 offset:256\n\t"
                        "ds_read_m32x16_b16 %3, %9 offset:384\n\t"
                        "ds_read_b64 %4, %10 offset:0\n\t"
                        "ds_read_b64 %5, %11 offset:256\n\t"
                        "ds_read_b64 %6, %10 offset:4096\n\t"
                        "ds_read_b64 %7, %11 offset:4352"
                        : "=&{v[144:147]}"(a_pairs[0]),
                          "=&{v[148:151]}"(a_pairs[1]),
                          "=&{v[152:155]}"(a_pairs[2]),
                          "=&{v[156:159]}"(a_pairs[3]),
                          "=&{v[168:169]}"(b_quads[0]),
                          "=&{v[170:171]}"(b_quads[1]),
                          "=&{v[172:173]}"(b_quads[2]),
                          "=&{v[174:175]}"(b_quads[3])
                        : "v"(a_load_base),
                          "v"(a_alt_base),
                          "v"(b_load_base),
                          "v"(b_alt_base)
                        : "memory");
                }
#else
                a_pairs[0] = LoadLdsM32x16<0>(a_load_base);
                a_pairs[1] = LoadLdsM32x16<128>(a_load_base);
                a_pairs[2] = LoadLdsM32x16<256>(a_load_base);
                a_pairs[3] = LoadLdsM32x16<384>(a_alt_base);
                b_quads[0] = LoadLdsB64<0>(b_load_base);
                b_quads[1] = LoadLdsB64<256>(b_alt_base);
                b_quads[2] = LoadLdsB64<4096>(b_load_base);
                b_quads[3] = LoadLdsB64<4352>(b_alt_base);
#endif
            }
            else
#endif
            {
                a_pairs[0] = LoadLdsPairSt64<0, 8>(a_load_base);
                a_pairs[1] = LoadLdsPairSt64<16, 24>(a_load_base);
                a_pairs[2] = LoadLdsPairSt64<32, 40>(a_load_base);
                a_pairs[3] = LoadLdsPairSt64<48, 56>(a_load_base);
                b_pairs[0] = LoadLdsPairSt64<64, 80>(b_load_base);
                b_pairs[1] = LoadLdsPairSt64<96, 112>(b_load_base);
            }
        };

        [[maybe_unused]] const auto compute_lds_stage =
            [&](auto& a_stage_tile, auto& b_stage_tile) {
            stage_block_gemm(c_block_tile, a_stage_tile, b_stage_tile);
        };

        const auto compute_lds_tile =
            [&](auto&& late_global_load,
                [[maybe_unused]] auto is_final_tile) {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_THREE_SLOT_STREAMING) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_FOUR_SLOT_STREAMING)
            // Keep only one A pair and two B pairs live. This is the native
            // gfx936 operand lifetime for a 256x256/W8 tile: each A pair feeds
            // eight MMACs, then its four VGPRs are reused by the next M pair.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;
            using AVec = typename StageImpl::AVecType;
            using BVec = typename StageImpl::BVecType;
            static_assert(std::is_same_v<AVec, BVec>,
                          "three-slot streaming requires matching A/B vectors");

#if defined(CK_TILE_GROUPED_GEMM_GFX936_FOUR_SLOT_STREAMING)
            thread_buffer<fp32x4_t, 4> operand_slots;
#else
            thread_buffer<fp32x4_t, 3> operand_slots;
#endif

            const auto a_base = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                return (((((tid & 15) << 7) | ((tid << 3) & 2168)) ^
                         ((tid >> 1) & 24)) ^
                        (iKStage * number<32>{}));
            };
            const auto b_base = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                return (((tid << 5) & 6144) |
                        (((tid & 15) << 3) ^ ((tid >> 1) & 24)) |
                        ((tid & 15) << 7)) ^
                       (iKStage * number<32>{});
            };
            const auto load_a_pair =
                [&](auto iKStage, auto iMPair, auto iASlot) {
                if constexpr(iMPair == number<0>{})
                {
                    operand_slots[iASlot] =
                        LoadLdsPairSt64<0, 8>(a_base(iKStage));
                }
                else if constexpr(iMPair == number<1>{})
                {
                    operand_slots[iASlot] =
                        LoadLdsPairSt64<16, 24>(a_base(iKStage));
                }
                else if constexpr(iMPair == number<2>{})
                {
                    operand_slots[iASlot] =
                        LoadLdsPairSt64<32, 40>(a_base(iKStage));
                }
                else
                {
                    static_assert(iMPair == number<3>{});
                    operand_slots[iASlot] =
                        LoadLdsPairSt64<48, 56>(a_base(iKStage));
                }
            };
            const auto load_b_pairs = [&](auto iKStage) {
                operand_slots[number<1>{}] =
                    LoadLdsPairSt64<64, 80>(b_base(iKStage));
                operand_slots[number<2>{}] =
                    LoadLdsPairSt64<96, 112>(b_base(iKStage));
            };
            const auto compute_pair_head =
                [&](auto iMPair, auto iASlot) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs =
                    operand_slots.template get_as<AVec>();
                const auto& b_vecs =
                    operand_slots.template get_as<BVec>();
                static_for<0, 2, 1>{}([&](auto iNInPair) {
                    constexpr auto iMRepeat = iMPair * number<2>{};
                    constexpr auto c_index =
                        iMRepeat * number<4>{} + iNInPair;
                    StageImpl{}(c_vecs[c_index],
                                a_vecs[iASlot * number<2>{}],
                                b_vecs[number<2>{} + iNInPair]);
                });
            };
            const auto compute_pair_tail =
                [&](auto iMPair, auto iASlot) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs =
                    operand_slots.template get_as<AVec>();
                const auto& b_vecs =
                    operand_slots.template get_as<BVec>();
                static_for<0, 2, 1>{}([&](auto iNInPair) {
                    constexpr auto iMRepeat = iMPair * number<2>{};
                    constexpr auto iNRepeat = number<2>{} + iNInPair;
                    constexpr auto c_index =
                        iMRepeat * number<4>{} + iNRepeat;
                    StageImpl{}(c_vecs[c_index],
                                a_vecs[iASlot * number<2>{}],
                                b_vecs[number<4>{} + iNInPair]);
                });
                static_for<0, 4, 1>{}([&](auto iNRepeat) {
                    constexpr auto iMRepeat =
                        iMPair * number<2>{} + number<1>{};
                    constexpr auto c_index =
                        iMRepeat * number<4>{} + iNRepeat;
                    StageImpl{}(c_vecs[c_index],
                                a_vecs[iASlot * number<2>{} + number<1>{}],
                                b_vecs[number<2>{} + iNRepeat]);
                });
            };
            const auto compute_pair =
                [&](auto iMPair, auto iASlot) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs =
                    operand_slots.template get_as<AVec>();
                const auto& b_vecs =
                    operand_slots.template get_as<BVec>();
                static_for<0, 2, 1>{}([&](auto iMInPair) {
                    static_for<0, 4, 1>{}([&](auto iNRepeat) {
                        constexpr auto iMRepeat =
                            iMPair * number<2>{} + iMInPair;
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNRepeat;
                        StageImpl{}(c_vecs[c_index],
                                    a_vecs[iASlot * number<2>{} + iMInPair],
                                    b_vecs[number<2>{} + iNRepeat]);
                    });
                });
            };

            static_for<0, NumKStages, 1>{}([&](auto iKStage) {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_FOUR_SLOT_STREAMING)
                // Double-buffer A pairs: issue the next ds_read before the
                // current eight MMACs so LDS latency is hidden without
                // retaining all four A pairs.
                load_a_pair(iKStage, number<0>{}, number<0>{});
                load_b_pairs(iKStage);
                load_a_pair(iKStage, number<1>{}, number<3>{});
                WaitLdsReads(number<2>{});
                compute_pair_head(number<0>{}, number<0>{});
                WaitLdsReads(number<1>{});
                compute_pair_tail(number<0>{}, number<0>{});

                WaitLdsReads();
                load_a_pair(iKStage, number<2>{}, number<0>{});
                compute_pair(number<1>{}, number<3>{});

                WaitLdsReads();
                load_a_pair(iKStage, number<3>{}, number<3>{});
                compute_pair(number<2>{}, number<0>{});

                WaitLdsReads();
                if constexpr(iKStage == number<3>{})
                {
                    late_global_load();
#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE)
                    if constexpr(is_final_tile.value)
                    {
                        block_sync_lds();
                        final_epilogue_pre_store(c_block_tile);
                    }
#endif
                }
                compute_pair(number<3>{}, number<3>{});
#else
                load_a_pair(
                    iKStage, number<0>{}, number<0>{});
                load_b_pairs(iKStage);
                // A0 and B0 are ready while the second B read may remain.
                WaitLdsReads(number<1>{});
                compute_pair_head(number<0>{}, number<0>{});
                WaitLdsReads();
                compute_pair_tail(number<0>{}, number<0>{});

                static_for<1, 4, 1>{}([&](auto iMPair) {
                    load_a_pair(
                        iKStage, iMPair, number<0>{});
                    WaitLdsReads();
                    if constexpr(iKStage == number<3>{} &&
                                 iMPair == number<3>{})
                    {
                        // Hide the next K64 global read behind the last eight
                        // MMACs, matching the native persistent hot loop.
                        late_global_load();
#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE)
                        if constexpr(is_final_tile.value)
                        {
                            // All input LDS reads are complete. Round 0 is
                            // already final, so use dead LDS while pair 3 runs.
                            block_sync_lds();
                            final_epilogue_pre_store(c_block_tile);
                        }
#endif
                    }
                    compute_pair(iMPair, number<0>{});
                });
#endif
            });
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING)
            // Reproduce Triton's five-slot operand rotation. Each slot holds
            // one ds_read2st64_b64 result (two native MMAC operands). The
            // next K16 stage is read into a slot immediately after the
            // current operand dies, so only 20 operand VGPRs are live rather
            // than the 48 VGPRs required by two complete stage tensors.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;
            using AVec = typename StageImpl::AVecType;
            using BVec = typename StageImpl::BVecType;
            static_assert(std::is_same_v<AVec, BVec>,
                          "rolling operand slots require matching A/B vectors");

            thread_buffer<fp32x4_t, 5> operand_slots;

            const auto a_base = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                return (((((tid & 15) << 7) | ((tid << 3) & 2168)) ^
                         ((tid >> 1) & 24)) ^
                        (iKStage * number<32>{}));
            };
            const auto b_base = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                return (((tid << 5) & 6144) |
                        (((tid & 15) << 3) ^ ((tid >> 1) & 24)) |
                        ((tid & 15) << 7)) ^
                       (iKStage * number<32>{});
            };
            const auto load_a_slot =
                [&](auto iKStage, auto iMPair, auto iSlot) {
                    if constexpr(iMPair == number<0>{})
                    {
                        operand_slots[iSlot] =
                            LoadLdsPairSt64<0, 8>(a_base(iKStage));
                    }
                    else if constexpr(iMPair == number<1>{})
                    {
                        operand_slots[iSlot] =
                            LoadLdsPairSt64<16, 24>(a_base(iKStage));
                    }
                    else if constexpr(iMPair == number<2>{})
                    {
                        operand_slots[iSlot] =
                            LoadLdsPairSt64<32, 40>(a_base(iKStage));
                    }
                    else
                    {
                        static_assert(iMPair == number<3>{});
                        operand_slots[iSlot] =
                            LoadLdsPairSt64<48, 56>(a_base(iKStage));
                    }
                };
            const auto load_b0_slot = [&](auto iKStage, auto iSlot) {
                operand_slots[iSlot] =
                    LoadLdsPairSt64<64, 80>(b_base(iKStage));
            };
            const auto load_b1_slot = [&](auto iKStage, auto iSlot) {
                operand_slots[iSlot] =
                    LoadLdsPairSt64<96, 112>(b_base(iKStage));
            };
            const auto compute_pair_half =
                [&](auto iMPair, auto iASlot, auto iBSlot, auto iNPair) {
                    [[maybe_unused]] auto& c_vecs =
                        c_block_tile.get_thread_buffer()
                            .template get_as<typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        operand_slots.template get_as<AVec>();
                    const auto& b_vecs =
                        operand_slots.template get_as<BVec>();
                    static_for<0, 2, 1>{}([&](auto iMInPair) {
                        static_for<0, 2, 1>{}([&](auto iNInPair) {
                            constexpr auto iMRepeat =
                                iMPair * number<2>{} + iMInPair;
                            constexpr auto iNRepeat =
                                iNPair * number<2>{} + iNInPair;
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            StageImpl{}(
                                c_vecs[c_index],
                                a_vecs[iASlot * number<2>{} + iMInPair],
                                b_vecs[iBSlot * number<2>{} + iNInPair]);
                        });
                    });
                };
            const auto compute_pair =
                [&](auto iMPair,
                    auto iASlot,
                    auto iB0Slot,
                    auto iB1Slot) {
                    compute_pair_half(
                        iMPair, iASlot, iB0Slot, number<0>{});
                    compute_pair_half(
                        iMPair, iASlot, iB1Slot, number<1>{});
                };
            const auto compute_pair_head =
                [&](auto iMPair, auto iASlot, auto iB0Slot) {
                    auto& c_vecs = c_block_tile.get_thread_buffer()
                                       .template get_as<
                                           typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        operand_slots.template get_as<AVec>();
                    const auto& b_vecs =
                        operand_slots.template get_as<BVec>();
                    static_for<0, 2, 1>{}([&](auto iNInPair) {
                        constexpr auto iMRepeat = iMPair * number<2>{};
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNInPair;
                        StageImpl{}(
                            c_vecs[c_index],
                            a_vecs[iASlot * number<2>{}],
                            b_vecs[iB0Slot * number<2>{} + iNInPair]);
                    });
                };
            const auto compute_pair_tail =
                [&](auto iMPair,
                    auto iASlot,
                    auto iB0Slot,
                    auto iB1Slot) {
                    auto& c_vecs = c_block_tile.get_thread_buffer()
                                       .template get_as<
                                           typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        operand_slots.template get_as<AVec>();
                    const auto& b_vecs =
                        operand_slots.template get_as<BVec>();

                    // Triton's gfx936 hot loop starts after A0 and B0 are
                    // ready, then waits for B1 before consuming the remaining
                    // six native operands. Preserve that 2+6 issue order
                    // instead of CK's otherwise equivalent 4+4 grouping.
                    static_for<0, 2, 1>{}([&](auto iNInPair) {
                        constexpr auto iMRepeat = iMPair * number<2>{};
                        constexpr auto iNRepeat =
                            number<2>{} + iNInPair;
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNRepeat;
                        StageImpl{}(
                            c_vecs[c_index],
                            a_vecs[iASlot * number<2>{}],
                            b_vecs[iB1Slot * number<2>{} + iNInPair]);
                    });
                    static_for<0, 2, 1>{}([&](auto iNPair) {
                        static_for<0, 2, 1>{}([&](auto iNInPair) {
                            constexpr auto iMRepeat =
                                iMPair * number<2>{} + number<1>{};
                            constexpr auto iNRepeat =
                                iNPair * number<2>{} + iNInPair;
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            constexpr auto iBSlot =
                                iNPair == number<0>{} ? iB0Slot : iB1Slot;
                            StageImpl{}(
                                c_vecs[c_index],
                                a_vecs[iASlot * number<2>{} + number<1>{}],
                                b_vecs[iBSlot * number<2>{} + iNInPair]);
                        });
                    });
                };

            // Stage 0: S0=A0, S1=B0, S2=B1, S3=A1, S4=A2.
            load_a_slot(number<0>{}, number<0>{}, number<0>{});
            load_b0_slot(number<0>{}, number<1>{});
            load_b1_slot(number<0>{}, number<2>{});
            load_a_slot(number<0>{}, number<1>{}, number<3>{});
            load_a_slot(number<0>{}, number<2>{}, number<4>{});
            WaitLdsReads(number<3>{});
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING_EXACT_WAIT)
            compute_pair_head(
                number<0>{}, number<0>{}, number<1>{});
            WaitLdsReads(number<2>{});
            compute_pair_tail(
                number<0>{}, number<0>{}, number<1>{}, number<2>{});
#else
            compute_pair_half(
                number<0>{}, number<0>{}, number<1>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<0>{}, number<0>{}, number<2>{}, number<1>{});
#endif
            load_a_slot(number<0>{}, number<3>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<1>{}, number<3>{}, number<1>{}, number<2>{});
            load_a_slot(number<1>{}, number<0>{}, number<3>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<2>{}, number<4>{}, number<1>{}, number<2>{});
            load_b0_slot(number<1>{}, number<4>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<3>{}, number<0>{}, number<1>{}, number<0>{});
            load_b1_slot(number<1>{}, number<1>{});
            compute_pair_half(
                number<3>{}, number<0>{}, number<2>{}, number<1>{});
            load_a_slot(number<1>{}, number<1>{}, number<0>{});
            load_a_slot(number<1>{}, number<2>{}, number<2>{});

            // Stage 1: S3=A0, S4=B0, S1=B1, S0=A1, S2=A2.
            WaitLdsReads(number<3>{});
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING_EXACT_WAIT)
            compute_pair_head(
                number<0>{}, number<3>{}, number<4>{});
            WaitLdsReads(number<2>{});
            compute_pair_tail(
                number<0>{}, number<3>{}, number<4>{}, number<1>{});
#else
            compute_pair_half(
                number<0>{}, number<3>{}, number<4>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<0>{}, number<3>{}, number<1>{}, number<1>{});
#endif
            load_a_slot(number<1>{}, number<3>{}, number<3>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<1>{}, number<0>{}, number<4>{}, number<1>{});
            load_a_slot(number<2>{}, number<0>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<2>{}, number<2>{}, number<4>{}, number<1>{});
            load_b0_slot(number<2>{}, number<2>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<3>{}, number<3>{}, number<4>{}, number<0>{});
            load_b1_slot(number<2>{}, number<4>{});
            compute_pair_half(
                number<3>{}, number<3>{}, number<1>{}, number<1>{});
            load_a_slot(number<2>{}, number<1>{}, number<3>{});
            load_a_slot(number<2>{}, number<2>{}, number<1>{});

            // Stage 2: S0=A0, S2=B0, S4=B1, S3=A1, S1=A2.
            WaitLdsReads(number<3>{});
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING_EXACT_WAIT)
            compute_pair_head(
                number<0>{}, number<0>{}, number<2>{});
            WaitLdsReads(number<2>{});
            compute_pair_tail(
                number<0>{}, number<0>{}, number<2>{}, number<4>{});
#else
            compute_pair_half(
                number<0>{}, number<0>{}, number<2>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<0>{}, number<0>{}, number<4>{}, number<1>{});
#endif
            load_a_slot(number<2>{}, number<3>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<1>{}, number<3>{}, number<2>{}, number<4>{});
            load_a_slot(number<3>{}, number<0>{}, number<3>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<2>{}, number<1>{}, number<2>{}, number<4>{});
            load_b0_slot(number<3>{}, number<1>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<3>{}, number<0>{}, number<2>{}, number<0>{});
            load_b1_slot(number<3>{}, number<2>{});
            compute_pair_half(
                number<3>{}, number<0>{}, number<4>{}, number<1>{});
            load_a_slot(number<3>{}, number<1>{}, number<0>{});
            load_a_slot(number<3>{}, number<2>{}, number<4>{});

            // Stage 3: S3=A0, S1=B0, S2=B1, S0=A1, S4=A2.
            WaitLdsReads(number<3>{});
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING_EXACT_WAIT)
            compute_pair_head(
                number<0>{}, number<3>{}, number<1>{});
            WaitLdsReads(number<2>{});
            compute_pair_tail(
                number<0>{}, number<3>{}, number<1>{}, number<2>{});
#else
            compute_pair_half(
                number<0>{}, number<3>{}, number<1>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<0>{}, number<3>{}, number<2>{}, number<1>{});
#endif
            load_a_slot(number<3>{}, number<3>{}, number<3>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<1>{}, number<0>{}, number<1>{}, number<2>{});
            WaitLdsReads(number<1>{});
            compute_pair(
                number<2>{}, number<4>{}, number<1>{}, number<2>{});
            // On the split-global variant, match Triton's aligned hot loop:
            // issue the next B K64 tile only after the third A pair dies and
            // hide those four VMEM reads behind the final eight MMACs.
            late_global_load();
            WaitLdsReads();
            compute_pair(
                number<3>{}, number<3>{}, number<1>{}, number<2>{});
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_SINGLE_PARTIAL)
            // One operand set, consumed in issue order. This keeps the
            // 221-VGPR baseline footprint while replacing each all-six-read
            // wait with Triton-style partial waits.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;

            auto a_stage =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            const auto load_stage_interleaved = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                const index_t a_load_base =
                    (((((tid & 15) << 7) |
                       ((tid << 3) & 2168)) ^
                      ((tid >> 1) & 24)) ^
                     (iKStage * number<32>{}));
                const index_t b_load_base =
                    (((tid << 5) & 6144) |
                     (((tid & 15) << 3) ^
                      ((tid >> 1) & 24)) |
                     ((tid & 15) << 7)) ^
                    (iKStage * number<32>{});
                auto& a_pairs = a_stage.get_thread_buffer()
                                    .template get_as<fp32x4_t>();
                auto& b_pairs = b_stage.get_thread_buffer()
                                    .template get_as<fp32x4_t>();
                a_pairs[0] = LoadLdsPairSt64<0, 8>(a_load_base);
                b_pairs[0] = LoadLdsPairSt64<64, 80>(b_load_base);
                b_pairs[1] = LoadLdsPairSt64<96, 112>(b_load_base);
                a_pairs[1] = LoadLdsPairSt64<16, 24>(a_load_base);
                a_pairs[2] = LoadLdsPairSt64<32, 40>(a_load_base);
                a_pairs[3] = LoadLdsPairSt64<48, 56>(a_load_base);
            };

            const auto compute_m_pair = [&](auto iMPair) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs = a_stage.get_thread_buffer()
                                         .template get_as<typename StageImpl::AVecType>();
                const auto& b_vecs = b_stage.get_thread_buffer()
                                         .template get_as<typename StageImpl::BVecType>();
                static_for<0, 2, 1>{}([&](auto iMInPair) {
                    constexpr auto iMRepeat =
                        iMPair * number<2>{} + iMInPair;
                    static_for<0, 4, 1>{}([&](auto iNRepeat) {
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNRepeat;
                        StageImpl{}(c_vecs[c_index],
                                    a_vecs[iMRepeat],
                                    b_vecs[iNRepeat]);
                    });
                });
            };

            static_for<0, NumKStages, 1>{}([&](auto iKStage) {
                load_stage_interleaved(iKStage);
                if constexpr(iKStage == number<3>{})
                {
                    late_global_load();
                }
                WaitLdsReads(number<3>{});
                compute_m_pair(number<0>{});
                WaitLdsReads(number<2>{});
                compute_m_pair(number<1>{});
                WaitLdsReads(number<1>{});
                compute_m_pair(number<2>{});
                WaitLdsReads();
                compute_m_pair(number<3>{});
            });
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP)
            // Keep two operand sets, but begin stage-0 MMACs once A pair 0
            // and both B pairs are ready. Stage 1 is already in flight, so
            // descending waits reproduce Triton's lgkmcnt-driven consumption
            // instead of stalling for all twelve read2 instructions.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;

            auto a_stage0 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage0 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);
            auto a_stage1 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage1 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            const auto load_lds_stage_interleaved =
                [&](auto iKStage,
                    auto& a_stage_tile,
                    auto& b_stage_tile) {
                    const index_t a_load_base =
                        get_a_load_base(iKStage);
                    const index_t b_load_base =
                        get_b_load_base(iKStage);
                    auto& a_pairs = a_stage_tile.get_thread_buffer()
                                        .template get_as<fp32x4_t>();
                    auto& b_pairs = b_stage_tile.get_thread_buffer()
                                        .template get_as<fp32x4_t>();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL)
                    if constexpr(IsARow && !IsBRow)
                    {
                        a_pairs[0] =
                            LoadLdsPairSt64<0, 2>(a_load_base);
                        b_pairs[0] =
                            LoadLdsPairSt64<0, 4>(b_load_base);
                        b_pairs[1] =
                            LoadLdsPairSt64<8, 12>(b_load_base);
                        a_pairs[1] =
                            LoadLdsPairSt64<4, 6>(a_load_base);
                        a_pairs[2] =
                            LoadLdsPairSt64<8, 10>(a_load_base);
                        a_pairs[3] =
                            LoadLdsPairSt64<12, 14>(a_load_base);
                    }
                    else
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
                    if constexpr(IsARow && IsBRow)
                    {
                        const index_t tid = get_thread_id();
                        const index_t lane = tid & 63;
                        const index_t wave = tid >> 6;
                        const bool a_wrap =
                            (((lane >> 2) & 3) == 3) &&
                            ((wave & 1) == 1);
                        const bool b_wrap =
                            (lane == 47 || lane == 63) &&
                            (((wave >> 1) & 1) == 1);
                        const index_t a_alt_base =
                            a_load_base - (a_wrap ? 1024 : 0);
                        const index_t b_alt_base =
                            b_load_base - (b_wrap ? 1024 : 0);
                        auto& b_quads =
                            b_stage_tile.get_thread_buffer()
                                .template get_as<fp32x2_t>();

                        a_pairs[0] =
                            LoadLdsM32x16<0>(a_load_base);
                        b_quads[0] =
                            LoadLdsB64<0>(b_load_base);
                        b_quads[1] =
                            LoadLdsB64<256>(b_alt_base);
                        b_quads[2] =
                            LoadLdsB64<4096>(b_load_base);
                        b_quads[3] =
                            LoadLdsB64<4352>(b_alt_base);
                        a_pairs[1] =
                            LoadLdsM32x16<128>(a_load_base);
                        a_pairs[2] =
                            LoadLdsM32x16<256>(a_load_base);
                        a_pairs[3] =
                            LoadLdsM32x16<384>(a_alt_base);
                    }
                    else
#endif
                    {
                        a_pairs[0] =
                            LoadLdsPairSt64<0, 8>(a_load_base);
                        b_pairs[0] =
                            LoadLdsPairSt64<64, 80>(b_load_base);
                        b_pairs[1] =
                            LoadLdsPairSt64<96, 112>(b_load_base);
                        a_pairs[1] =
                            LoadLdsPairSt64<16, 24>(a_load_base);
                        a_pairs[2] =
                            LoadLdsPairSt64<32, 40>(a_load_base);
                        a_pairs[3] =
                            LoadLdsPairSt64<48, 56>(a_load_base);
                    }
                };

            const auto compute_m_pair =
                [&](auto iMPair,
                    auto& a_stage_tile,
                    auto& b_stage_tile) {
                    auto& c_vecs = c_block_tile.get_thread_buffer()
                                       .template get_as<
                                           typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        a_stage_tile.get_thread_buffer()
                            .template get_as<typename StageImpl::AVecType>();
                    const auto& b_vecs =
                        b_stage_tile.get_thread_buffer()
                            .template get_as<typename StageImpl::BVecType>();
                    static_for<0, 2, 1>{}([&](auto iMInPair) {
                        constexpr auto iMRepeat =
                            iMPair * number<2>{} + iMInPair;
                        static_for<0, 4, 1>{}([&](auto iNRepeat) {
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            StageImpl{}(c_vecs[c_index],
                                        a_vecs[iMRepeat],
                                        b_vecs[iNRepeat]);
                        });
                    });
                };

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
            // The custom direct-store epilogue consumes CK's M-major raw
            // accumulator buffer and performs the required 16x16 block
            // transpose only when choosing the global output coordinate.
            // Keep every K16 stage on this manual path: with TransposeC, the
            // generic StageBlockGemm update uses a different distributed
            // buffer order and would combine unrelated output subtiles.
            const auto compute_full_stage =
                [&](auto& a_stage_tile, auto& b_stage_tile) {
                    static_for<0, 4, 1>{}([&](auto iMPair) {
                        compute_m_pair(iMPair, a_stage_tile, b_stage_tile);
                    });
                };
#endif

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_EXACT_SCHEDULE)
            // Match the hipBLASLt WGM8 K16 steady-state schedule. Keep the
            // current operands resident while issuing the next stage's eight
            // LDS reads and the dead stage's two direct VMEM-to-LDS refills.
            // lgkmcnt(8) leaves the new stage in flight; on the following
            // stage it retires the older eight reads before their registers
            // are consumed. The K64 boundary remains conservatively fenced
            // by the outer loop until this schedule proves beneficial.
            load_lds_stage_interleaved(
                number<0>{}, a_stage0, b_stage0);
            WaitLdsReads();
            __builtin_amdgcn_s_barrier();

            load_lds_stage(
                number<1>{}, a_stage1, b_stage1);
            if constexpr(!is_final_tile.value)
            {
                direct_load_rr_stage(
                    number<0>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
                buffer_load_fence(4);
            }
            WaitLdsReads(number<8>{});
            __builtin_amdgcn_s_barrier();
            compute_full_stage(a_stage0, b_stage0);

            load_lds_stage(
                number<2>{}, a_stage0, b_stage0);
            if constexpr(!is_final_tile.value)
            {
                direct_load_rr_stage(
                    number<1>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
                buffer_load_fence(4);
            }
            WaitLdsReads(number<8>{});
            __builtin_amdgcn_s_barrier();
            compute_full_stage(a_stage1, b_stage1);

            load_lds_stage(
                number<3>{}, a_stage1, b_stage1);
            if constexpr(!is_final_tile.value)
            {
                direct_load_rr_stage(
                    number<2>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
                buffer_load_fence(4);
            }
            WaitLdsReads(number<8>{});
            __builtin_amdgcn_s_barrier();
            compute_full_stage(a_stage0, b_stage0);

            // Stage 3 has no same-K64 successor in this first exact-schedule
            // probe. Retire its reads across all waves, issue the final dead
            // stage refill, and hide that VMEM latency behind stage-3 MMAC.
            WaitLdsReads();
            __builtin_amdgcn_s_barrier();
            if constexpr(!is_final_tile.value)
            {
                direct_load_rr_stage(
                    number<3>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            }
            late_global_load();
            compute_full_stage(a_stage1, b_stage1);
#else
            load_lds_stage_interleaved(
                number<0>{}, a_stage0, b_stage0);
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE)
            WaitLdsReads(number<3>{});
            compute_m_pair(number<0>{}, a_stage0, b_stage0);
            load_lds_stage(number<1>{}, a_stage1, b_stage1);
            WaitLdsReads(number<8>{});
            compute_m_pair(number<1>{}, a_stage0, b_stage0);
            WaitLdsReads(number<7>{});
            compute_m_pair(number<2>{}, a_stage0, b_stage0);
            WaitLdsReads(number<6>{});
            compute_m_pair(number<3>{}, a_stage0, b_stage0);
#else
            load_lds_stage(number<1>{}, a_stage1, b_stage1);
            WaitLdsReads(number<9>{});
            compute_m_pair(number<0>{}, a_stage0, b_stage0);
            WaitLdsReads(number<8>{});
            compute_m_pair(number<1>{}, a_stage0, b_stage0);
            WaitLdsReads(number<7>{});
            compute_m_pair(number<2>{}, a_stage0, b_stage0);
            WaitLdsReads(number<6>{});
            compute_m_pair(number<3>{}, a_stage0, b_stage0);
#endif
            WaitLdsReads();

#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE) && \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
            if constexpr(!is_final_tile.value)
            {
                // Every wave has finished reading stage 0. Refill its dead
                // 16-KiB slot from the next K64 tile while stage 1 computes.
                block_sync_lds();
                direct_load_rr_stage(
                    number<0>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            }
#endif
            load_lds_stage(number<2>{}, a_stage0, b_stage0);
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
            compute_full_stage(a_stage1, b_stage1);
#else
            compute_lds_stage(a_stage1, b_stage1);
#endif
            WaitLdsReads();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE) && \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
            if constexpr(!is_final_tile.value)
            {
                block_sync_lds();
                direct_load_rr_stage(
                    number<1>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            }
#endif
            load_lds_stage(number<3>{}, a_stage1, b_stage1);
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
            compute_full_stage(a_stage0, b_stage0);
#else
            compute_lds_stage(a_stage0, b_stage0);
#endif
            late_global_load();
            WaitLdsReads();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE) && \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
            if constexpr(!is_final_tile.value)
            {
                block_sync_lds();
                direct_load_rr_stage(
                    number<2>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            }
#endif
#if defined(CK_TILE_GROUPED_GEMM_TRITON_EPILOGUE_PRE_SYNC)
            // All final-K16 operands now reside in VGPRs. Move the first
            // epilogue barrier here so no wave can overwrite input LDS while
            // another still reads it, while leaving final MMAC and output
            // conversion in one scheduler region.
            block_sync_lds();
#endif
#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE)
            if constexpr(is_final_tile.value)
            {
#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE_POST_COMPUTE)
                // Non-persistent codegen uses a different raw accumulator
                // register order. Finish all final-K16 updates before
                // materializing fused round 0, while retaining the epilogue's
                // one-round LDS/store elision.
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
                compute_full_stage(a_stage1, b_stage1);
#else
                compute_lds_stage(a_stage1, b_stage1);
#endif
                block_sync_lds();
                final_epilogue_pre_store(c_block_tile);
#else
                // All final K16 operands are resident in VGPRs. Once every
                // wave has stopped reading A/B LDS, finish the first M pair,
                // convert its 32 accumulator scalars, and write epilogue
                // round 0 into the dead compute allocation. The remaining
                // three M pairs cover the four LDS writes.
                block_sync_lds();
                compute_m_pair(number<0>{}, a_stage1, b_stage1);
                final_epilogue_pre_store(c_block_tile);
                compute_m_pair(number<1>{}, a_stage1, b_stage1);
                compute_m_pair(number<2>{}, a_stage1, b_stage1);
                compute_m_pair(number<3>{}, a_stage1, b_stage1);
#endif
            }
            else
            {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
                compute_full_stage(a_stage1, b_stage1);
#else
                compute_lds_stage(a_stage1, b_stage1);
#endif
            }
#else
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
            compute_full_stage(a_stage1, b_stage1);
#else
            compute_lds_stage(a_stage1, b_stage1);
#endif
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE) && \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
            if constexpr(!is_final_tile.value)
            {
                block_sync_lds();
                direct_load_rr_stage(
                    number<3>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
            }
#endif
#endif
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_INPLACE_OVERLAP)
            // Triton-like operand lifetime control. A pair is dead after its
            // eight MMACs (two M repeats x four N repeats), so immediately
            // overwrite those four VGPRs with the same pair from the next
            // K16 stage. Only B needs a second register set.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;

            auto a_stage =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage0 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);
            auto b_stage1 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            const auto load_b_stage = [&](auto iKStage,
                                          auto& b_stage_tile) {
                const index_t tid = get_thread_id();
                const index_t b_load_base =
                    (((tid << 5) & 6144) |
                     (((tid & 15) << 3) ^ ((tid >> 1) & 24)) |
                     ((tid & 15) << 7)) ^
                    (iKStage * number<32>{});
                auto& b_pairs = b_stage_tile.get_thread_buffer()
                                    .template get_as<fp32x4_t>();
                b_pairs[0] = LoadLdsPairSt64<64, 80>(b_load_base);
                b_pairs[1] = LoadLdsPairSt64<96, 112>(b_load_base);
            };

            const auto compute_a_pair = [&](auto iMPair,
                                             auto& b_stage_tile) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs = a_stage.get_thread_buffer()
                                         .template get_as<typename StageImpl::AVecType>();
                const auto& b_vecs = b_stage_tile.get_thread_buffer()
                                         .template get_as<typename StageImpl::BVecType>();
                static_for<0, 2, 1>{}([&](auto iMInPair) {
                    constexpr auto iMRepeat =
                        iMPair * number<2>{} + iMInPair;
                    static_for<0, 4, 1>{}([&](auto iNRepeat) {
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNRepeat;
                        StageImpl{}(c_vecs[c_index],
                                    a_vecs[iMRepeat],
                                    b_vecs[iNRepeat]);
                    });
                });
            };

            const auto reload_a_pair = [&](auto iKStage, auto iMPair) {
                const index_t tid = get_thread_id();
                const index_t a_load_base =
                    (((((tid & 15) << 7) | ((tid << 3) & 2168)) ^
                      ((tid >> 1) & 24)) ^
                     (iKStage * number<32>{})) +
                    iMPair * number<8192>{};
                auto& a_pairs =
                    a_stage.get_thread_buffer().template get_as<fp32x4_t>();
                a_pairs[iMPair] =
                    LoadLdsPairSt64<0, 8>(a_load_base);
            };

            const auto compute_and_prefetch_next =
                [&](auto iNextKStage,
                    auto& b_current,
                    auto& b_next) {
                    // B remains live for all 32 current-stage MMACs, so issue
                    // the two next-stage B reads into a small second set.
                    load_b_stage(iNextKStage, b_next);
                    static_for<0, 4, 1>{}([&](auto iMPair) {
                        compute_a_pair(iMPair, b_current);
                        reload_a_pair(iNextKStage, iMPair);
                    });
                    WaitLdsReads();
                };

            load_lds_stage(number<0>{}, a_stage, b_stage0);
            WaitLdsReads();
            compute_and_prefetch_next(
                number<1>{}, b_stage0, b_stage1);
            compute_and_prefetch_next(
                number<2>{}, b_stage1, b_stage0);
            compute_and_prefetch_next(
                number<3>{}, b_stage0, b_stage1);
            late_global_load();
            static_for<0, 4, 1>{}(
                [&](auto iMPair) { compute_a_pair(iMPair, b_stage1); });
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_OVERLAP)
            // Ping-pong two operand register sets. Once stage S is ready,
            // issue LDS reads for S+1 and cover their latency with the 32
            // native MMACs of S. Reuse a register set only after its MMACs
            // have consumed it, bounding the live range below the spill
            // threshold.
            auto a_stage0 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage0 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);
            auto a_stage1 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage1 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            load_lds_stage(number<0>{}, a_stage0, b_stage0);
            WaitLdsReads();

            load_lds_stage(number<1>{}, a_stage1, b_stage1);
            compute_lds_stage(a_stage0, b_stage0);
            WaitLdsReads();

            load_lds_stage(number<2>{}, a_stage0, b_stage0);
            compute_lds_stage(a_stage1, b_stage1);
            WaitLdsReads();

            load_lds_stage(number<3>{}, a_stage1, b_stage1);
            compute_lds_stage(a_stage0, b_stage0);
            late_global_load();
            WaitLdsReads();
#if defined(CK_TILE_GROUPED_GEMM_TRITON_EPILOGUE_PRE_SYNC)
            // All final-K16 operands now reside in VGPRs. Move the first
            // epilogue barrier here so no wave can overwrite input LDS while
            // another still reads it, while leaving final MMAC and output
            // conversion in one scheduler region.
            block_sync_lds();
#endif
            compute_lds_stage(a_stage1, b_stage1);
#else
            static_for<0, NumKStages, 1>{}([&](auto iKStage) {
                auto a_stage_tile =
                    make_static_distributed_tensor<ADataType>(a_stage_dstr);
                auto b_stage_tile =
                    make_static_distributed_tensor<BDataType>(b_stage_dstr);
                load_lds_stage(iKStage, a_stage_tile, b_stage_tile);
                WaitLdsReads();
                if constexpr(iKStage == number<3>{})
                {
                    late_global_load();
                }
                compute_lds_stage(a_stage_tile, b_stage_tile);
            });
#endif
        };

#if defined(CK_TILE_GROUPED_GEMM_GFX936_DIRECT_G2L_FULL)
        if constexpr(IsARow && !IsBRow)
        {
            index_t i = 1;
            while(i < num_loop)
            {
                compute_lds_tile([&]() {}, bool_constant<false>{});
                block_sync_lds();
                direct_load_full_tile();
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);
                block_sync_lds();
                ++i;
            }

            compute_lds_tile([&]() {}, bool_constant<true>{});
        }
        else
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_STAGE)
        if constexpr(IsARow && IsBRow)
        {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS)
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;

            auto a_stage0 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage0 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);
            auto a_stage1 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage1 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            const auto compute_exact_stage =
                [&]([[maybe_unused]] auto iOperandBuffer,
                    auto& a_stage_tile,
                    auto& b_stage_tile) {
                    [[maybe_unused]] auto& c_vecs =
                        c_block_tile.get_thread_buffer()
                            .template get_as<typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        a_stage_tile.get_thread_buffer()
                            .template get_as<typename StageImpl::AVecType>();
                    const auto& b_vecs =
                        b_stage_tile.get_thread_buffer()
                            .template get_as<typename StageImpl::BVecType>();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_OPERANDS_V3)
                    auto rr_b0 = b_vecs[number<0>{}];
                    auto rr_b1 = b_vecs[number<1>{}];
                    auto rr_b2 = b_vecs[number<2>{}];
                    auto rr_b3 = b_vecs[number<3>{}];
                    auto rr_a0 = a_vecs[number<0>{}];
                    auto rr_a1 = a_vecs[number<1>{}];
                    auto rr_a2 = a_vecs[number<2>{}];
                    auto rr_a3 = a_vecs[number<3>{}];
                    auto rr_a4 = a_vecs[number<4>{}];
                    auto rr_a5 = a_vecs[number<5>{}];
                    auto rr_a6 = a_vecs[number<6>{}];
                    auto rr_a7 = a_vecs[number<7>{}];
                    if constexpr(iOperandBuffer == number<0>{})
                    {
                        asm volatile(
                            ""
                            : "+{v[160:161]}"(rr_b0),
                              "+{v[162:163]}"(rr_b1),
                              "+{v[164:165]}"(rr_b2),
                              "+{v[166:167]}"(rr_b3),
                              "+{v[128:129]}"(rr_a0),
                              "+{v[130:131]}"(rr_a1),
                              "+{v[132:133]}"(rr_a2),
                              "+{v[134:135]}"(rr_a3),
                              "+{v[136:137]}"(rr_a4),
                              "+{v[138:139]}"(rr_a5),
                              "+{v[140:141]}"(rr_a6),
                              "+{v[142:143]}"(rr_a7));
                    }
                    else
                    {
                        asm volatile(
                            ""
                            : "+{v[168:169]}"(rr_b0),
                              "+{v[170:171]}"(rr_b1),
                              "+{v[172:173]}"(rr_b2),
                              "+{v[174:175]}"(rr_b3),
                              "+{v[144:145]}"(rr_a0),
                              "+{v[146:147]}"(rr_a1),
                              "+{v[148:149]}"(rr_a2),
                              "+{v[150:151]}"(rr_a3),
                              "+{v[152:153]}"(rr_a4),
                              "+{v[154:155]}"(rr_a5),
                              "+{v[156:157]}"(rr_a6),
                              "+{v[158:159]}"(rr_a7));
                    }
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_INLINE_MMAC_ORDER)
                    // Keep the vendor's B-major 8x4 operand reuse order at
                    // the machine-scheduler boundary. Separate volatile asm
                    // statements retain dataflow through each accumulator and
                    // prevent LLVM from regrouping independent MMACs by its
                    // register-allocation order.
                    static_for<0, 4, 1>{}([&](auto iNRepeat) {
                        static_for<0, 8, 1>{}([&](auto iMRepeat) {
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            auto& c_vec       = c_vecs[c_index];
                            const auto a_vec  = a_vecs[iMRepeat];
                            const auto b_vec  = b_vecs[iNRepeat];
                            if constexpr(Problem::TransposeC)
                            {
                                // Transposed-C MMAC swaps the native A/B
                                // operands; mirror StageImpl exactly while
                                // pinning only the issue order.
                                asm volatile(
                                    "v_mmac_f32_16x16x16_bf16 %0, %1, %2, %0"
                                    : "+&v"(c_vec)
                                    : "v"(b_vec), "v"(a_vec));
                            }
                            else
                            {
                                asm volatile(
                                    "v_mmac_f32_16x16x16_bf16 %0, %1, %2, %0"
                                    : "+&v"(c_vec)
                                    : "v"(a_vec), "v"(b_vec));
                            }
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SETPRIO_AFTER_FIRST)
                            if constexpr(iNRepeat == number<0>{} &&
                                         iMRepeat == number<0>{})
                            {
                                __builtin_amdgcn_s_setprio(1);
                            }
#endif
                        });
                    });
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SETPRIO_AFTER_FIRST)
                    __builtin_amdgcn_s_setprio(0);
#endif
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS)
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VENDOR_OPERAND_ORDER_V4)
                    static_assert(!Problem::TransposeC,
                                  "vendor operand order V4 uses the native "
                                  "non-transposed MMAC/output distribution");
#else
                    static_assert(Problem::TransposeC,
                                  "fixed accumulator probe is specialized "
                                  "for the RR TransposeC path");
#endif
                    // One asm region forces a contiguous v0:v127
                    // accumulator assignment and preserves the vendor's
                    // B-major operand reuse order. The early-clobber output
                    // prevents any of the twelve operand fragments from
                    // overlapping the accumulator bank.
                    asm volatile(
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2)
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VENDOR_OPERAND_ORDER_V4)
                        // Match the custom BLAS MMAC source-port contract:
                        // the eight-fragment A bank is source 0 and the
                        // four-fragment B bank is source 1. The surrounding
                        // producer and epilogue remain unchanged so this is
                        // an isolated operand-port/order correctness probe.
                        "v_mmac_f32_16x16x16_bf16 %0, %36, %32, %0\n\t"
                        "s_setprio 1\n\t"
                        "v_mmac_f32_16x16x16_bf16 %1, %37, %32, %1\n\t"
                        "v_mmac_f32_16x16x16_bf16 %2, %38, %32, %2\n\t"
                        "v_mmac_f32_16x16x16_bf16 %3, %39, %32, %3\n\t"
                        "v_mmac_f32_16x16x16_bf16 %4, %40, %32, %4\n\t"
                        "v_mmac_f32_16x16x16_bf16 %5, %41, %32, %5\n\t"
                        "v_mmac_f32_16x16x16_bf16 %6, %42, %32, %6\n\t"
                        "v_mmac_f32_16x16x16_bf16 %7, %43, %32, %7\n\t"
                        "v_mmac_f32_16x16x16_bf16 %8, %36, %33, %8\n\t"
                        "v_mmac_f32_16x16x16_bf16 %9, %37, %33, %9\n\t"
                        "v_mmac_f32_16x16x16_bf16 %10, %38, %33, %10\n\t"
                        "v_mmac_f32_16x16x16_bf16 %11, %39, %33, %11\n\t"
                        "v_mmac_f32_16x16x16_bf16 %12, %40, %33, %12\n\t"
                        "v_mmac_f32_16x16x16_bf16 %13, %41, %33, %13\n\t"
                        "v_mmac_f32_16x16x16_bf16 %14, %42, %33, %14\n\t"
                        "v_mmac_f32_16x16x16_bf16 %15, %43, %33, %15\n\t"
                        "v_mmac_f32_16x16x16_bf16 %16, %36, %34, %16\n\t"
                        "v_mmac_f32_16x16x16_bf16 %17, %37, %34, %17\n\t"
                        "v_mmac_f32_16x16x16_bf16 %18, %38, %34, %18\n\t"
                        "v_mmac_f32_16x16x16_bf16 %19, %39, %34, %19\n\t"
                        "v_mmac_f32_16x16x16_bf16 %20, %40, %34, %20\n\t"
                        "v_mmac_f32_16x16x16_bf16 %21, %41, %34, %21\n\t"
                        "v_mmac_f32_16x16x16_bf16 %22, %42, %34, %22\n\t"
                        "v_mmac_f32_16x16x16_bf16 %23, %43, %34, %23\n\t"
                        "v_mmac_f32_16x16x16_bf16 %24, %36, %35, %24\n\t"
                        "v_mmac_f32_16x16x16_bf16 %25, %37, %35, %25\n\t"
                        "v_mmac_f32_16x16x16_bf16 %26, %38, %35, %26\n\t"
                        "v_mmac_f32_16x16x16_bf16 %27, %39, %35, %27\n\t"
                        "v_mmac_f32_16x16x16_bf16 %28, %40, %35, %28\n\t"
                        "v_mmac_f32_16x16x16_bf16 %29, %41, %35, %29\n\t"
                        "v_mmac_f32_16x16x16_bf16 %30, %42, %35, %30\n\t"
                        "v_mmac_f32_16x16x16_bf16 %31, %43, %35, %31\n\t"
                        "s_setprio 0"
#else
                        // Pin logical vendor-order C(n,m) tuples directly to
                        // v[0:3], v[4:7], ..., v[124:127]. Unlike the first
                        // fixed-accumulator probe, these are physical tuple
                        // constraints rather than allocator observations.
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONSUMER_ASM_V6)
                        "s_waitcnt lgkmcnt(8)\n\t"
                        "s_waitcnt vmcnt(4)\n\t"
                        "s_barrier\n\t"
#endif
                        "v_mmac_f32_16x16x16_bf16 %0, %32, %36, %0\n\t"
                        "s_setprio 1\n\t"
                        "v_mmac_f32_16x16x16_bf16 %1, %32, %37, %1\n\t"
                        "v_mmac_f32_16x16x16_bf16 %2, %32, %38, %2\n\t"
                        "v_mmac_f32_16x16x16_bf16 %3, %32, %39, %3\n\t"
                        "v_mmac_f32_16x16x16_bf16 %4, %32, %40, %4\n\t"
                        "v_mmac_f32_16x16x16_bf16 %5, %32, %41, %5\n\t"
                        "v_mmac_f32_16x16x16_bf16 %6, %32, %42, %6\n\t"
                        "v_mmac_f32_16x16x16_bf16 %7, %32, %43, %7\n\t"
                        "v_mmac_f32_16x16x16_bf16 %8, %33, %36, %8\n\t"
                        "v_mmac_f32_16x16x16_bf16 %9, %33, %37, %9\n\t"
                        "v_mmac_f32_16x16x16_bf16 %10, %33, %38, %10\n\t"
                        "v_mmac_f32_16x16x16_bf16 %11, %33, %39, %11\n\t"
                        "v_mmac_f32_16x16x16_bf16 %12, %33, %40, %12\n\t"
                        "v_mmac_f32_16x16x16_bf16 %13, %33, %41, %13\n\t"
                        "v_mmac_f32_16x16x16_bf16 %14, %33, %42, %14\n\t"
                        "v_mmac_f32_16x16x16_bf16 %15, %33, %43, %15\n\t"
                        "v_mmac_f32_16x16x16_bf16 %16, %34, %36, %16\n\t"
                        "v_mmac_f32_16x16x16_bf16 %17, %34, %37, %17\n\t"
                        "v_mmac_f32_16x16x16_bf16 %18, %34, %38, %18\n\t"
                        "v_mmac_f32_16x16x16_bf16 %19, %34, %39, %19\n\t"
                        "v_mmac_f32_16x16x16_bf16 %20, %34, %40, %20\n\t"
                        "v_mmac_f32_16x16x16_bf16 %21, %34, %41, %21\n\t"
                        "v_mmac_f32_16x16x16_bf16 %22, %34, %42, %22\n\t"
                        "v_mmac_f32_16x16x16_bf16 %23, %34, %43, %23\n\t"
                        "v_mmac_f32_16x16x16_bf16 %24, %35, %36, %24\n\t"
                        "v_mmac_f32_16x16x16_bf16 %25, %35, %37, %25\n\t"
                        "v_mmac_f32_16x16x16_bf16 %26, %35, %38, %26\n\t"
                        "v_mmac_f32_16x16x16_bf16 %27, %35, %39, %27\n\t"
                        "v_mmac_f32_16x16x16_bf16 %28, %35, %40, %28\n\t"
                        "v_mmac_f32_16x16x16_bf16 %29, %35, %41, %29\n\t"
                        "v_mmac_f32_16x16x16_bf16 %30, %35, %42, %30\n\t"
                        "v_mmac_f32_16x16x16_bf16 %31, %35, %43, %31\n\t"
                        "s_setprio 0"
#endif
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS)
                        // Map logical C(m,n) tuples onto the allocator's
                        // observed physical tuple order, then issue n-major.
                        // This gives the vendor sequence v0,v4,...,v124 while
                        // retaining one B fragment for each run of eight
                        // MMACs. The epilogue applies the inverse tuple map.
                        "v_mmac_f32_16x16x16_bf16 %0, %32, %36, %0\n\t"
                        "s_setprio 1\n\t"
                        "v_mmac_f32_16x16x16_bf16 %1, %32, %37, %1\n\t"
                        "v_mmac_f32_16x16x16_bf16 %24, %32, %38, %24\n\t"
                        "v_mmac_f32_16x16x16_bf16 %25, %32, %39, %25\n\t"
                        "v_mmac_f32_16x16x16_bf16 %16, %32, %40, %16\n\t"
                        "v_mmac_f32_16x16x16_bf16 %10, %32, %41, %10\n\t"
                        "v_mmac_f32_16x16x16_bf16 %17, %32, %42, %17\n\t"
                        "v_mmac_f32_16x16x16_bf16 %11, %32, %43, %11\n\t"
                        "v_mmac_f32_16x16x16_bf16 %2, %33, %36, %2\n\t"
                        "v_mmac_f32_16x16x16_bf16 %14, %33, %37, %14\n\t"
                        "v_mmac_f32_16x16x16_bf16 %3, %33, %38, %3\n\t"
                        "v_mmac_f32_16x16x16_bf16 %26, %33, %39, %26\n\t"
                        "v_mmac_f32_16x16x16_bf16 %8, %33, %40, %8\n\t"
                        "v_mmac_f32_16x16x16_bf16 %9, %33, %41, %9\n\t"
                        "v_mmac_f32_16x16x16_bf16 %6, %33, %42, %6\n\t"
                        "v_mmac_f32_16x16x16_bf16 %18, %33, %43, %18\n\t"
                        "v_mmac_f32_16x16x16_bf16 %12, %34, %36, %12\n\t"
                        "v_mmac_f32_16x16x16_bf16 %13, %34, %37, %13\n\t"
                        "v_mmac_f32_16x16x16_bf16 %22, %34, %38, %22\n\t"
                        "v_mmac_f32_16x16x16_bf16 %23, %34, %39, %23\n\t"
                        "v_mmac_f32_16x16x16_bf16 %28, %34, %40, %28\n\t"
                        "v_mmac_f32_16x16x16_bf16 %29, %34, %41, %29\n\t"
                        "v_mmac_f32_16x16x16_bf16 %4, %34, %42, %4\n\t"
                        "v_mmac_f32_16x16x16_bf16 %5, %34, %43, %5\n\t"
                        "v_mmac_f32_16x16x16_bf16 %15, %35, %36, %15\n\t"
                        "v_mmac_f32_16x16x16_bf16 %27, %35, %37, %27\n\t"
                        "v_mmac_f32_16x16x16_bf16 %20, %35, %38, %20\n\t"
                        "v_mmac_f32_16x16x16_bf16 %21, %35, %39, %21\n\t"
                        "v_mmac_f32_16x16x16_bf16 %7, %35, %40, %7\n\t"
                        "v_mmac_f32_16x16x16_bf16 %19, %35, %41, %19\n\t"
                        "v_mmac_f32_16x16x16_bf16 %30, %35, %42, %30\n\t"
                        "v_mmac_f32_16x16x16_bf16 %31, %35, %43, %31\n\t"
                        "s_setprio 0"
#else
                        // DTK allocates these 32 live tuple outputs in a stable
                        // but non-source order for the RR epilogue. Issue in
                        // the inverse observed allocation order so the physical
                        // accumulator destinations are v0, v4, ..., v124 while
                        // retaining each tuple's original logical A/B product.
                        "v_mmac_f32_16x16x16_bf16 %31, %35, %43, %31\n\t"
                        "s_setprio 1\n\t"
                        "v_mmac_f32_16x16x16_bf16 %30, %35, %42, %30\n\t"
                        "v_mmac_f32_16x16x16_bf16 %23, %34, %43, %23\n\t"
                        "v_mmac_f32_16x16x16_bf16 %22, %34, %42, %22\n\t"
                        "v_mmac_f32_16x16x16_bf16 %27, %35, %39, %27\n\t"
                        "v_mmac_f32_16x16x16_bf16 %15, %33, %43, %15\n\t"
                        "v_mmac_f32_16x16x16_bf16 %26, %35, %38, %26\n\t"
                        "v_mmac_f32_16x16x16_bf16 %14, %33, %42, %14\n\t"
                        "v_mmac_f32_16x16x16_bf16 %19, %34, %39, %19\n\t"
                        "v_mmac_f32_16x16x16_bf16 %7, %32, %43, %7\n\t"
                        "v_mmac_f32_16x16x16_bf16 %18, %34, %38, %18\n\t"
                        "v_mmac_f32_16x16x16_bf16 %6, %32, %42, %6\n\t"
                        "v_mmac_f32_16x16x16_bf16 %29, %35, %41, %29\n\t"
                        "v_mmac_f32_16x16x16_bf16 %28, %35, %40, %28\n\t"
                        "v_mmac_f32_16x16x16_bf16 %11, %33, %39, %11\n\t"
                        "v_mmac_f32_16x16x16_bf16 %10, %33, %38, %10\n\t"
                        "v_mmac_f32_16x16x16_bf16 %21, %34, %41, %21\n\t"
                        "v_mmac_f32_16x16x16_bf16 %20, %34, %40, %20\n\t"
                        "v_mmac_f32_16x16x16_bf16 %3, %32, %39, %3\n\t"
                        "v_mmac_f32_16x16x16_bf16 %2, %32, %38, %2\n\t"
                        "v_mmac_f32_16x16x16_bf16 %13, %33, %41, %13\n\t"
                        "v_mmac_f32_16x16x16_bf16 %12, %33, %40, %12\n\t"
                        "v_mmac_f32_16x16x16_bf16 %25, %35, %37, %25\n\t"
                        "v_mmac_f32_16x16x16_bf16 %24, %35, %36, %24\n\t"
                        "v_mmac_f32_16x16x16_bf16 %5, %32, %41, %5\n\t"
                        "v_mmac_f32_16x16x16_bf16 %4, %32, %40, %4\n\t"
                        "v_mmac_f32_16x16x16_bf16 %17, %34, %37, %17\n\t"
                        "v_mmac_f32_16x16x16_bf16 %16, %34, %36, %16\n\t"
                        "v_mmac_f32_16x16x16_bf16 %9, %33, %37, %9\n\t"
                        "v_mmac_f32_16x16x16_bf16 %8, %33, %36, %8\n\t"
                        "v_mmac_f32_16x16x16_bf16 %1, %32, %37, %1\n\t"
                        "v_mmac_f32_16x16x16_bf16 %0, %32, %36, %0\n\t"
                        "s_setprio 0"
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2)
                        : "+&{v[0:3]}"(rr_c00),
                          "+&{v[4:7]}"(rr_c01),
                          "+&{v[8:11]}"(rr_c02),
                          "+&{v[12:15]}"(rr_c03),
                          "+&{v[16:19]}"(rr_c04),
                          "+&{v[20:23]}"(rr_c05),
                          "+&{v[24:27]}"(rr_c06),
                          "+&{v[28:31]}"(rr_c07),
                          "+&{v[32:35]}"(rr_c08),
                          "+&{v[36:39]}"(rr_c09),
                          "+&{v[40:43]}"(rr_c10),
                          "+&{v[44:47]}"(rr_c11),
                          "+&{v[48:51]}"(rr_c12),
                          "+&{v[52:55]}"(rr_c13),
                          "+&{v[56:59]}"(rr_c14),
                          "+&{v[60:63]}"(rr_c15),
                          "+&{v[64:67]}"(rr_c16),
                          "+&{v[68:71]}"(rr_c17),
                          "+&{v[72:75]}"(rr_c18),
                          "+&{v[76:79]}"(rr_c19),
                          "+&{v[80:83]}"(rr_c20),
                          "+&{v[84:87]}"(rr_c21),
                          "+&{v[88:91]}"(rr_c22),
                          "+&{v[92:95]}"(rr_c23),
                          "+&{v[96:99]}"(rr_c24),
                          "+&{v[100:103]}"(rr_c25),
                          "+&{v[104:107]}"(rr_c26),
                          "+&{v[108:111]}"(rr_c27),
                          "+&{v[112:115]}"(rr_c28),
                          "+&{v[116:119]}"(rr_c29),
                          "+&{v[120:123]}"(rr_c30),
                          "+&{v[124:127]}"(rr_c31)
#else
                        : "+&v"(rr_c00),
                          "+&v"(rr_c01),
                          "+&v"(rr_c02),
                          "+&v"(rr_c03),
                          "+&v"(rr_c04),
                          "+&v"(rr_c05),
                          "+&v"(rr_c06),
                          "+&v"(rr_c07),
                          "+&v"(rr_c08),
                          "+&v"(rr_c09),
                          "+&v"(rr_c10),
                          "+&v"(rr_c11),
                          "+&v"(rr_c12),
                          "+&v"(rr_c13),
                          "+&v"(rr_c14),
                          "+&v"(rr_c15),
                          "+&v"(rr_c16),
                          "+&v"(rr_c17),
                          "+&v"(rr_c18),
                          "+&v"(rr_c19),
                          "+&v"(rr_c20),
                          "+&v"(rr_c21),
                          "+&v"(rr_c22),
                          "+&v"(rr_c23),
                          "+&v"(rr_c24),
                          "+&v"(rr_c25),
                          "+&v"(rr_c26),
                          "+&v"(rr_c27),
                          "+&v"(rr_c28),
                          "+&v"(rr_c29),
                          "+&v"(rr_c30),
                          "+&v"(rr_c31)
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_OPERANDS_V3)
                        : "v"(rr_b0),
                          "v"(rr_b1),
                          "v"(rr_b2),
                          "v"(rr_b3),
                          "v"(rr_a0),
                          "v"(rr_a1),
                          "v"(rr_a2),
                          "v"(rr_a3),
                          "v"(rr_a4),
                          "v"(rr_a5),
                          "v"(rr_a6),
                          "v"(rr_a7));
#else
                        : "v"(b_vecs[number<0>{}]),
                          "v"(b_vecs[number<1>{}]),
                          "v"(b_vecs[number<2>{}]),
                          "v"(b_vecs[number<3>{}]),
                          "v"(a_vecs[number<0>{}]),
                          "v"(a_vecs[number<1>{}]),
                          "v"(a_vecs[number<2>{}]),
                          "v"(a_vecs[number<3>{}]),
                          "v"(a_vecs[number<4>{}]),
                          "v"(a_vecs[number<5>{}]),
                          "v"(a_vecs[number<6>{}]),
                          "v"(a_vecs[number<7>{}]));
#endif
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_SETPRIO)
                    // The custom hipBLASLt WGM8 kernel raises wave priority
                    // only for the 32-MMAC consumer interval. Keep this
                    // isolated behind a probe macro so the validated
                    // continuous schedule remains unchanged.
                    __builtin_amdgcn_s_setprio(1);
#endif
#if !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_INLINE_MMAC_ORDER) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS) && \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SETPRIO_AFTER_FIRST)
                    // The vendor ISA deliberately issues the first MMAC at
                    // normal priority, then raises priority for the remaining
                    // 31 consumer instructions. Preserve that exact boundary
                    // instead of prioritizing the LDS-to-MMAC transition.
                    StageImpl{}(
                        c_vecs[number<0>{}], a_vecs[number<0>{}], b_vecs[number<0>{}]);
                    __builtin_amdgcn_s_setprio(1);
                    static_for<1, 32, 1>{}([&](auto iLinear) {
                        constexpr auto iMRepeat = iLinear / number<4>{};
                        constexpr auto iNRepeat = iLinear % number<4>{};
                        StageImpl{}(c_vecs[iLinear],
                                    a_vecs[iMRepeat],
                                    b_vecs[iNRepeat]);
                    });
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_EARLY_AND_FINAL_INLINE_BARRIER)
                    // Prevent a wave from refilling circular LDS while another
                    // wave still consumes the current slot. Inline assembly
                    // keeps the barrier after all 32 register-only MMACs.
                    asm volatile("s_barrier" ::: "memory");
#endif
                    __builtin_amdgcn_s_setprio(0);
#elif !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_INLINE_MMAC_ORDER) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS) && \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_MMAC_ORDER)
                    // The hipBLASLt WGM8 ISA keeps one B fragment resident
                    // across all eight M repeats before advancing N. Match
                    // that operand-read/reuse order without changing the
                    // accumulator coordinate mapping.
                    static_for<0, 4, 1>{}([&](auto iNRepeat) {
                        static_for<0, 8, 1>{}([&](auto iMRepeat) {
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            StageImpl{}(c_vecs[c_index],
                                        a_vecs[iMRepeat],
                                        b_vecs[iNRepeat]);
                        });
                    });
#elif !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_INLINE_MMAC_ORDER) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS) && \
    !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS)
                    static_for<0, 8, 1>{}([&](auto iMRepeat) {
                        static_for<0, 4, 1>{}([&](auto iNRepeat) {
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            StageImpl{}(c_vecs[c_index],
                                        a_vecs[iMRepeat],
                                        b_vecs[iNRepeat]);
                        });
                    });
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_SETPRIO)
                    __builtin_amdgcn_s_setprio(0);
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONTINUOUS_STRICT)
                    // Prevent LLVM from hoisting the following stage's
                    // inline LDS reads into this 32-MMAC interval. The BLAS
                    // ISA keeps each consumer interval contiguous and only
                    // then issues the next eight reads.
                    __builtin_amdgcn_sched_barrier(0);
#endif
                };

            // The first K16 operands come from the prologue tile. Every
            // non-final stage thereafter issues the next LDS read and the
            // dead-slot direct refill before computing the current operands.
            load_lds_stage(number<0>{}, a_stage0, b_stage0);
            WaitLdsReads();
            __builtin_amdgcn_s_barrier();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC)
            // Keep the prologue synchronization on the same side of the
            // consumer interval as written. Without a compiler scheduling
            // fence, LLVM is free to sink s_barrier below register-only MMACs.
            __builtin_amdgcn_sched_barrier(0);
#endif

            const auto finish_refill_and_compute =
                [&](auto iOperandBuffer,
                    auto& a_stage_tile,
                    auto& b_stage_tile,
                    auto is_k64_boundary) {
                    // The partial lgkmcnt(8)/vmcnt(4) contract is sufficient
                    // for the validated persistent traversal, but exposed a
                    // non-deterministic cross-wave LDS hazard when every
                    // output tile was launched independently with M01=2.
                    // Sweep stricter counts independently to find the
                    // smallest safe fence without losing VMEM overlap.
#if !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONSUMER_ASM_V6)
                    if constexpr(is_k64_boundary.value)
                    {
                        WaitRrBlasBoundaryLdsReads();
                    }
                    else
                    {
                        WaitRrBlasSteadyLdsReads();
                    }
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_CONSUMER_ASM_V6)
                    // The fixed-accumulator asm starts with the complete
                    // steady consumer fence and barrier, keeping the wait to
                    // MMAC transition in one scheduler-opaque region.
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_VMEM_WAIT_COUNT)
                    if constexpr(decltype(is_k64_boundary)::value)
                    {
                        // V6 keeps the vendor vmcnt(4) inside its opaque
                        // consumer packet. The persistent K64 wrap requires
                        // one additional completed direct refill on gfx936;
                        // establish that boundary-only contract before
                        // entering the otherwise unchanged packet.
                        buffer_load_fence(
                            CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_VMEM_WAIT_COUNT);
                    }
#endif
                    compute_exact_stage(
                        iOperandBuffer, a_stage_tile, b_stage_tile);
#else
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_POSTCOMPUTE_SYNC)
                    constexpr bool use_postcompute_sync = true;
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_POSTCOMPUTE_SYNC)
                    constexpr bool use_postcompute_sync =
                        decltype(is_k64_boundary)::value;
#else
                    constexpr bool use_postcompute_sync = false;
#endif
                    if constexpr(use_postcompute_sync)
                    {
                        // In the vendor steady-state ISA, each direct-to-LDS
                        // refill remains in flight while the current 32 MMACs
                        // execute. Only after those MMACs does the kernel wait
                        // down to vmcnt(4) and synchronize waves before
                        // issuing the following stage's LDS reads/refill.
                        compute_exact_stage(
                            iOperandBuffer, a_stage_tile, b_stage_tile);
                        buffer_load_fence(4);
                        __builtin_amdgcn_s_barrier();
                    }
                    else
                    {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_SAFE_SYNC) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FULL_VMEM_WAIT)
                        buffer_load_fence(0);
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VMEM_WAIT_COUNT)
                        buffer_load_fence(
                            CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VMEM_WAIT_COUNT);
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_VMEM_WAIT_COUNT)
                        if constexpr(decltype(is_k64_boundary)::value)
                        {
                            buffer_load_fence(
                                CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BOUNDARY_VMEM_WAIT_COUNT);
                        }
                        else
                        {
                            buffer_load_fence(4);
                        }
#else
                        buffer_load_fence(4);
#endif
                        __builtin_amdgcn_s_barrier();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC)
                        // hipBLASLt executes lgkmcnt/vmcnt/barrier before
                        // every 32-MMAC interval. Pin the barrier here so
                        // LLVM cannot sink it to the end of the register-only
                        // consumer.
                        __builtin_amdgcn_sched_barrier(0);
#endif
                        compute_exact_stage(
                            iOperandBuffer, a_stage_tile, b_stage_tile);
                    }
#endif
                };

            index_t i = 1;
            while(i < num_loop)
            {
                load_lds_stage(number<1>{}, a_stage1, b_stage1);
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD)
                CK_TILE_RR_FIXED_STREAM_LOAD(
                    number<0>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
#else
                direct_load_rr_stage(
                    number<0>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
#endif
                finish_refill_and_compute(
                    number<0>{}, a_stage0, b_stage0, bool_constant<false>{});

                load_lds_stage(number<2>{}, a_stage0, b_stage0);
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD)
                CK_TILE_RR_FIXED_STREAM_LOAD(
                    number<1>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
#else
                direct_load_rr_stage(
                    number<1>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
#endif
                finish_refill_and_compute(
                    number<1>{}, a_stage1, b_stage1, bool_constant<false>{});

                load_lds_stage(number<3>{}, a_stage1, b_stage1);
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD)
                CK_TILE_RR_FIXED_STREAM_LOAD(
                    number<2>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
#else
                direct_load_rr_stage(
                    number<2>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
#endif
                finish_refill_and_compute(
                    number<0>{}, a_stage0, b_stage0, bool_constant<false>{});

                // Stage 0 of the following K64 tile was refilled three
                // compute stages ago. Read it into the now-dead buffer while
                // refilling stage 3, then carry these operands across the
                // dynamic K64 loop boundary without a full VMEM fence.
                load_lds_stage(number<0>{}, a_stage0, b_stage0);
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD)
                CK_TILE_RR_FIXED_STREAM_LOAD(
                    number<3>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
#else
                direct_load_rr_stage(
                    number<3>{}, rr_a_m0_base_reg, rr_b_m0_base_reg);
#endif
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_EARLY_K64_WINDOW_ADVANCE)
                // The stage-3 direct refill has already consumed the current
                // K64 global windows. Advance the next-tile coordinates in
                // the refill-to-wait latency slot, matching the vendor
                // kernel's useful scalar address work between VMEM issue and
                // waitcnt. No current-stage MMAC operand depends on these
                // window coordinates.
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);
                ++i;
#endif
                finish_refill_and_compute(
                    number<1>{}, a_stage1, b_stage1, bool_constant<true>{});

#if !defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_EARLY_K64_WINDOW_ADVANCE)
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);
                ++i;
#endif
            }

            // Final K64 tile: keep the same eight-read rolling dependency,
            // but stop issuing refills beyond the valid K range.
            load_lds_stage(number<1>{}, a_stage1, b_stage1);
            WaitRrBlasSteadyLdsReads();
            __builtin_amdgcn_s_barrier();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC)
            __builtin_amdgcn_sched_barrier(0);
#endif
            compute_exact_stage(number<0>{}, a_stage0, b_stage0);

            load_lds_stage(number<2>{}, a_stage0, b_stage0);
            WaitRrBlasSteadyLdsReads();
            __builtin_amdgcn_s_barrier();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC)
            __builtin_amdgcn_sched_barrier(0);
#endif
            compute_exact_stage(number<1>{}, a_stage1, b_stage1);

            load_lds_stage(number<3>{}, a_stage1, b_stage1);
            WaitRrBlasSteadyLdsReads();
            __builtin_amdgcn_s_barrier();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC)
            __builtin_amdgcn_sched_barrier(0);
#endif
            compute_exact_stage(number<0>{}, a_stage0, b_stage0);

            WaitLdsReads();
            __builtin_amdgcn_s_barrier();
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_BARRIER_BEFORE_MMAC)
            __builtin_amdgcn_sched_barrier(0);
#endif
            compute_exact_stage(number<1>{}, a_stage1, b_stage1);
#else
            index_t i = 1;
            while(i < num_loop)
            {
                compute_lds_tile([&]() {}, bool_constant<false>{});
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
                // Four stage-local direct loads were issued while the other
                // stages computed. Fence once before the circular slots are
                // consumed as the next K64 tile.
                buffer_load_fence(0);
#else
                block_sync_lds();
                direct_load_full_tile();
#endif
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);
                block_sync_lds();
                ++i;
            }

            compute_lds_tile([&]() {}, bool_constant<true>{});
#endif
        }
        else
#endif
        {
            index_t i = 1;
            while(i < num_loop)
            {
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_SPLIT_GLOBAL)
                // Triton issues A's next-tile VMEM loads before the K16
                // operand loop, but delays B until the final K16 MMAC stage.
                auto a_next_global = load_tile(a_dram_window);
                move_tile_window(a_dram_window, a_step);
                decltype(load_tile(b_dram_window)) b_next_global;

                compute_lds_tile(
                    [&]() {
                        b_next_global = load_tile(b_dram_window);
                        move_tile_window(b_dram_window, b_step);
                    },
                    bool_constant<false>{});
#else
                // Issue the next global K64 tile before consuming the current
                // LDS tile. Its registers stay live across four K16 steps.
                auto a_next_global = load_tile(a_dram_window);
                auto b_next_global = load_tile(b_dram_window);
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);

                compute_lds_tile([&]() {}, bool_constant<false>{});
#endif
                block_sync_lds();
                store_full_tile(a_next_global, b_next_global);
                block_sync_lds();
                ++i;
            }

            compute_lds_tile([&]() {}, bool_constant<true>{});
        }
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_STREAM_SRD)
        if constexpr(IsARow && IsBRow)
        {
            // Keep the two direct LDS bases live in their reserved SGPRs
            // across every inline-assembly load that names s60/s61 directly.
            asm volatile("" : "+{s60}"(rr_a_m0_base_reg),
                         "+{s61}"(rr_b_m0_base_reg));
        }
#undef CK_TILE_RR_FIXED_STREAM_LOAD
#endif
        // The candidate epilogue consumes the native MMAC accumulator layout
        // and performs the lane transpose through the now-dead A/B LDS. Keep
        // the raw tile here so the producer stores can use the packed MMAC
        // register order instead of first scalarizing CWarpOutputDstr.
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS_V2) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_FIXED_ACCUMULATORS) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS)
        if constexpr(IsARow && IsBRow)
        {
            using RrCVec = ext_vector_t<float, 4>;
            const RrCVec rr_vendor_vecs[32] = {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_LINEAR_ACCUMULATORS)
                rr_c00, rr_c01, rr_c24, rr_c25, rr_c16, rr_c10, rr_c17, rr_c11,
                rr_c02, rr_c14, rr_c03, rr_c26, rr_c08, rr_c09, rr_c06, rr_c18,
                rr_c12, rr_c13, rr_c22, rr_c23, rr_c28, rr_c29, rr_c04, rr_c05,
                rr_c15, rr_c27, rr_c20, rr_c21, rr_c07, rr_c19, rr_c30, rr_c31};
#else
                rr_c00, rr_c01, rr_c02, rr_c03, rr_c04, rr_c05, rr_c06, rr_c07,
                rr_c08, rr_c09, rr_c10, rr_c11, rr_c12, rr_c13, rr_c14, rr_c15,
                rr_c16, rr_c17, rr_c18, rr_c19, rr_c20, rr_c21, rr_c22, rr_c23,
                rr_c24, rr_c25, rr_c26, rr_c27, rr_c28, rr_c29, rr_c30, rr_c31};
#endif
            auto& c_vecs = c_block_tile.get_thread_buffer()
                               .template get_as<RrCVec>();
            static_for<0, 4, 1>{}([&](auto iNRepeat) {
                static_for<0, 8, 1>{}([&](auto iMRepeat) {
                    constexpr auto vendor_index =
                        iNRepeat * number<8>{} + iMRepeat;
                    constexpr auto ck_index =
                        iMRepeat * number<4>{} + iNRepeat;
                    c_vecs[ck_index] = rr_vendor_vecs[vendor_index];
                });
            });
        }
#endif
        return c_block_tile;
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const auto no_final_epilogue_pre_store = [](const auto&) {};
        return Run(a_dram_block_window,
                   b_dram_block_window,
                   num_loop,
                   p_smem,
                   no_final_epilogue_pre_store);
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   bool,
                                   TailNumber,
                                   void* p_smem) const
    {
        const auto no_final_epilogue_pre_store = [](const auto&) {};
        return Run(a_dram_block_window,
                   b_dram_block_window,
                   num_loop,
                   p_smem,
                   no_final_epilogue_pre_store);
    }

    template <typename ADramBlockWindow,
              typename BDramBlockWindow,
              typename FinalEpiloguePreStore>
    CK_TILE_DEVICE auto RunWithEpiloguePreStore(
        const ADramBlockWindow& a_dram_block_window,
        const BDramBlockWindow& b_dram_block_window,
        index_t num_loop,
        void* p_smem,
        FinalEpiloguePreStore&& final_epilogue_pre_store) const
    {
        return Run(a_dram_block_window,
                   b_dram_block_window,
                   num_loop,
                   p_smem,
                   static_cast<FinalEpiloguePreStore&&>(
                       final_epilogue_pre_store));
    }
};

} // namespace ck_tile
