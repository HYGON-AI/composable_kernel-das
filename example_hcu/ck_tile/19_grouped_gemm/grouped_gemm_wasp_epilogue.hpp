// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technologies Co., Ltd.
//
// Epilogue for the WASP (warp-specialized producer / consumer) NT pipeline.
//
// Ported line by line from the standalone warp-specialized reference
// implementation (`gemm_epilogue_writeback_via_lds_pc`).
//
// WHY THIS IS THE POINT OF THE PORT
// ---------------------------------
// The native MMAC output puts one lane's four results at columns
// {q, q+4, q+8, q+12} across 16 rows (q = lane >> 4), so direct scalar stores
// never coalesce: measured 33.2M VL1 write requests on m=8100 vs
// hipBLASLt's 8.4M, and because two CTAs overlap per CU that write flood
// congests TA/TC for the WHOLE kernel. Staging each 128-row slice through the
// (by then idle) 64 KB of LDS and reading it back with a wave-64-covers-
// 4-rows-x-128B layout turns 128 scalar stores into 16 vector stores per wave:
// write requests 33.2M -> ~1M, measured 1.100 ms / 247 TF -> 0.852 ms / 319 TF.
//
// WHAT THIS EPILOGUE MUST DO THAT CShuffleEpilogue CANNOT
// ------------------------------------------------------
//   1. ROLE AWARE. Only the 8 consumer waves (warp id >= 4) own output. The 4
//      producer waves fall straight through. CShuffle's kBlockSize is
//      MWave*NWave*warp_size = 512 and its thread-raked tile distribution
//      indexes by warp id, so waves 8..11 of a 768 thread workgroup would
//      compute out-of-range coordinates.
//   2. BARRIER FREE. CShuffle emits several unconditional block_sync_lds()
//      (s_waitcnt lgkmcnt(0) + s_barrier) -- see cshuffle_epilogue.hpp:881,
//      883, 900, 917, 934, 944, 965. A workgroup barrier the producer waves
//      never reach deadlocks, so this epilogue uses per-wave LDS slabs and only
//      the LDS drain it actually needs. The cross-tile drain lives at the
//      PIPELINE's entry instead (block_sync_lds()), where both roles arrive.
//
// LDS: 8 consumer waves x 8 KB (128 rows x 32 columns x 2 B) = the pipeline's
// whole 64 KB buffer, reused after the K loop drains. GetSmemSize() is 0 so the
// framework's max(GemmPipeline, EpiloguePipeline) keeps the pipeline's 65536.
//
// STAGE 3 of the port (see the pipeline header for stages 1-2): the writeback
// itself. Correctness and the coalescing win are in place; note the default
// LDS_XOR below is what removes the store-side bank conflicts.
#pragma once

#include "ck_tile/core.hpp"

#if defined(CK_TILE_GROUPED_GEMM_WASP_DEBUG_RECORD)
// Debug only: what each CTA's consumer waves actually saw when they reached the
// epilogue. Answers "did this CTA run at all" and "what origin/size did it
// think it had", which reasoning cannot settle. 5 slots per CTA.
//   [0] epilogue entries   [1] block_org[0]   [2] M   [3] N   [4] ldd
// static: internal linkage, so -Wmissing-variable-declarations (part of
// -Weverything) stays quiet and each TU that opts in gets its own copy.
//   per CTA: [0] reached  [1] block_org[0]  [2] M  [3] N  [4] ldd
//            [5] calculated origin offset  [6] p_data_ low  [7] d_ptr low
//            [8] col_base                 [9] consumer_id
__device__ static int g_wasp_dbg[128 * 10];
// One slot per (CTA, wave): a plain store, so it cannot lose the increment that
// a shared counter or a shared OR would.
__device__ static int g_wasp_dbg_waves[128 * 16];
#endif

#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
__device__ static uint64_t g_wasp_epi_cycles[1024 * 16 * 9];
#endif

template <typename AccDataType_, typename ODataType_, typename Pipeline_>
struct GroupedGemmWaspStoreEpilogue
{
    using AccDataType = ck_tile::remove_cvref_t<AccDataType_>;
    using ODataType   = ck_tile::remove_cvref_t<ODataType_>;
    using DsDataType  = ck_tile::tuple<>;
    using DsLayout    = ck_tile::tuple<>;

    // The pipeline owns the WARP SPLIT; the epilogue only needs the count so
    // that it can filter the consumer waves.
    using Pipeline = ck_tile::remove_cvref_t<Pipeline_>;

    using AccTile = typename Pipeline::AccTile;

    static constexpr ck_tile::index_t NumDTensor = 0;
    static constexpr ck_tile::memory_operation_enum MemoryOperation =
        ck_tile::memory_operation_enum::set;

    // v1 ships 2-byte output only (bf16/fp16 share this path). fp32 output uses
    // a different geometry entirely (EpiGeomF32/EpiPassF32) and is not
    // ported yet.
    static_assert(sizeof(ODataType) == 2, "WASP epilogue v1 is 2-byte output only");

    // ------------------------------------------------------------------
    // Geometry, mirroring EpiCoalescePC<WARP_N=32>.
    // ------------------------------------------------------------------
    static constexpr ck_tile::index_t kWaveM = 256;   // rows per consumer wave
    static constexpr ck_tile::index_t kWaveN = 32;    // columns per consumer wave

    static constexpr int kEpiHalvesPerWarp = 2;       // 128-row slabs, two passes
    static constexpr int kEpiElemsPerLane   = 8;      // 16 B per lane per store

    static constexpr int kColsPerRow     = kWaveN / kEpiElemsPerLane;      // 4
    static constexpr int kLog2ColsPerRow = 2;                             // log2(4)
    static constexpr int kRowsPerPass    = 64 / kColsPerRow;              // 16

    static constexpr ck_tile::index_t kRowsPerHalf = kWaveM / kEpiHalvesPerWarp;   // 128
    static constexpr int kPasses = kRowsPerHalf / kRowsPerPass;                    // 8

    static constexpr int kSubN        = kWaveN / 16;                  // 2 n sub-tiles
    static constexpr int kSubMPerHalf = kRowsPerHalf / 16;            // 8 m sub-tiles

    static constexpr int kLdsRowBytes = kWaveN * sizeof(ODataType);   // 64
    static constexpr int kLdsVectorBytes = 8;
    static constexpr int kLdsVectorsPerRow = kLdsRowBytes / kLdsVectorBytes;
    static constexpr ck_tile::index_t kSlabBytesPerWave =
        kRowsPerHalf * kWaveN * sizeof(ODataType);                    // 8192

    static_assert(kColsPerRow == 4 && kRowsPerPass == 16 && kPasses == 8, "epilogue geometry");
    static_assert(kLdsVectorsPerRow == 8, "the epilogue expects eight 8-byte chunks per row");
    // One half consumes kSubMPerHalf x kSubN = 16 accumulator slots; the two
    // halves together must cover the whole 16 x 2 = 32 slot wave tile.
    static_assert(kEpiHalvesPerWarp * kSubMPerHalf * kSubN ==
                      AccTile::kMIters * AccTile::kNIters,
                  "the two halves must cover the whole accumulator tile");

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetVectorSizeC()
    {
        return kEpiElemsPerLane;
    }

    template <ck_tile::index_t I>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t
    GetVectorSizeD(ck_tile::number<I>)
    {
        return 1;
    }

    // Reuse the pipeline's LDS buffer; the framework takes the max of the two.
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

#if defined(CK_TILE_GROUPED_GEMM_WASP_NO_BOUNDS_CHECK)
    // Debug switch: store without any M/N masking. Used to tell "the second
    // m-tile is not processed at all" apart from "the bounds check rejects every
    // row of it". Safe for the shapes the test drives, because the rejected rows
    // are still inside the allocated output buffer.
    static constexpr bool kBoundsCheck = false;
#else
    static constexpr bool kBoundsCheck = true;
#endif

    // One 16 B lane payload: two 8 B halves, or eight elements.
    union EpiVec8
    {
        int64_t    q[2];
        int32_t    d[4];
        ODataType  e[kEpiElemsPerLane];
    };

    CK_TILE_DEVICE static uint16_t FloatToBf16RneNonNan(float value)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx936__)
        const uint32_t bits    = ck_tile::bit_cast<uint32_t>(value);
        const uint32_t rounded = bits + 0x7fffU + ((bits >> 16) & 1U);
        return static_cast<uint16_t>(rounded >> 16);
#else
        return ck_tile::bit_cast<uint16_t>(
            ck_tile::type_convert<ck_tile::bf16_t>(value));
#endif
    }

    // ------------------------------------------------------------------
    // Keep the 8-byte XOR-swizzle used by the HIPC epilogue so the LDS reload
    // produces output-order vectors directly, without a VGPR permutation.
    // ------------------------------------------------------------------
    CK_TILE_DEVICE static int LdsOffset(int row_in_slab, int col)
    {
        const int vector = (col / 4) ^ (row_in_slab & (kLdsVectorsPerRow - 1));
        return row_in_slab * kLdsRowBytes + vector * kLdsVectorBytes +
               (col & 3) * int(sizeof(ODataType));
    }

    // ------------------------------------------------------------------
    // Store one 128-row half of the pre-converted wave tile into the LDS slab.
    //
    // The MMAC C layout is MEASURED, not assumed: accumulator slot c of lane
    // group g holds column 4*c + g (NOT 4*g + c -- the fp8-era convention was
    // never valid on gfx936). A lane's four values are therefore four columns
    // apart, so the four converted values are staged individually.
    // ------------------------------------------------------------------
    template <int HALF>
    CK_TILE_DEVICE static void StoreHalfToLds(ODataType* slab,
                                              ck_tile::index_t lane,
                                              const ODataType (*converted)[4])
    {
        const int lane_and_15 = static_cast<int>(lane) & 15;
        const int lane_lsr_4  = static_cast<int>(lane) >> 4;
        char* const base      = reinterpret_cast<char*>(slab);

        ck_tile::static_for<0, kSubMPerHalf, 1>{}([&](auto mm_local) {
            constexpr int mm_global = HALF * kSubMPerHalf + mm_local.value;
            const int row_in_slab = mm_local.value * 16 + lane_and_15;
            ck_tile::static_for<0, kSubN, 1>{}([&](auto nn) {
                constexpr int flat = mm_global * kSubN + nn.value;
                ck_tile::static_for<0, 4, 1>{}([&](auto c_index) {
                    const int c = c_index.value;
                    const int col = nn.value * 16 + 4 * c + lane_lsr_4;
                    *reinterpret_cast<ODataType*>(base + LdsOffset(row_in_slab, col)) =
                        converted[flat][c];
                });
            });
        });
    }

    // ------------------------------------------------------------------
    // Read the wave's slab back and write it to global with coalesced 16 B
    // stores: a wave-64 now covers 4 rows x 128 B, so one store per lane is a
    // whole cache line instead of 16 scattered line transactions.
    // ------------------------------------------------------------------
    template <int HALF>
    CK_TILE_DEVICE static void LoadLdsAndStoreGlobal(const ODataType* slab,
                                                     ODataType* d_ptr,
                                                     ck_tile::index_t ldd,
                                                     ck_tile::index_t row_base_global,
                                                     ck_tile::index_t col_base_global,
                                                     [[maybe_unused]] ck_tile::index_t M,
                                                     [[maybe_unused]] ck_tile::index_t N,
                                                     ck_tile::index_t lane)
    {
        const int row_in_pass = static_cast<int>(lane) >> kLog2ColsPerRow;
        const int chunk = static_cast<int>(lane) & (kColsPerRow - 1);
        const int col_in_wave = chunk * kEpiElemsPerLane;
        const ck_tile::index_t col_global = col_base_global + col_in_wave;

        // Read two adjacent 8-byte chunks per pass. The LDS swizzle maps them
        // directly to the final output order.
        EpiVec8 buf[kPasses];
#pragma unroll
        for(int pass = 0; pass < kPasses; ++pass)
        {
            const int row_in_slab = pass * kRowsPerPass + row_in_pass;
#pragma unroll
            for(int q = 0; q < 2; ++q)
            {
                const int col = col_in_wave + q * 4;
                const int off = LdsOffset(row_in_slab, col);
                const auto* addr = reinterpret_cast<const char*>(slab) + off;
                const auto lds_addr = static_cast<ck_tile::index_t>(
                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(addr)));
                uint64_t loaded_pair;
                ck_tile::smem_load<8>{}(loaded_pair, lds_addr, 0);
                buf[pass].q[q] = static_cast<int64_t>(loaded_pair);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);

        // Phase B: coalesced global stores.
        bool all_rows_in_bounds = true;
        if constexpr(kBoundsCheck)
        {
            const ck_tile::index_t last_row = row_base_global + HALF * kRowsPerHalf +
                                              kRowsPerHalf - kRowsPerPass + row_in_pass;
            all_rows_in_bounds = (last_row < M);
        }

        // IsSupportedArgument requires N to be a multiple of the 256-column
        // block tile. Each consumer owns 32 columns and stores aligned 8-BF16
        // vectors, so the column-tail/scalar-store path is unreachable.
        if(all_rows_in_bounds)
        {
#pragma unroll
            for(int pass = 0; pass < kPasses; ++pass)
            {
                const ck_tile::index_t row_global = row_base_global + HALF * kRowsPerHalf +
                                                   pass * kRowsPerPass + row_in_pass;
                *reinterpret_cast<EpiVec8*>(d_ptr + row_global * ldd + col_global) = buf[pass];
            }
        }
        else
        {
#pragma unroll
            for(int pass = 0; pass < kPasses; ++pass)
            {
                const ck_tile::index_t row_global = row_base_global + HALF * kRowsPerHalf +
                                                   pass * kRowsPerPass + row_in_pass;
                if constexpr(kBoundsCheck)
                {
                    if(row_global < M)
                    {
                        *reinterpret_cast<EpiVec8*>(d_ptr + row_global * ldd + col_global) =
                            buf[pass];
                    }
                }
                else
                {
                    *reinterpret_cast<EpiVec8*>(d_ptr + row_global * ldd + col_global) =
                        buf[pass];
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Entry point. Called by GroupedGemmKernel on EVERY wave of the workgroup
    // (grouped_gemm_kernel.hpp:943-947, no role guard there), so the filter is
    // ours. Contains no workgroup barrier on purpose.
    // ------------------------------------------------------------------
#if defined(CK_TILE_GROUPED_GEMM_WASP_DIRECT_STORE)
    // ------------------------------------------------------------------
    // Debug path: store the accumulator straight to DRAM with no LDS staging.
    // 128 scalar stores per lane, i.e. deliberately slow -- but it answers the
    // one question that bisects this port: is the ACCUMULATOR itself right?
    // Same failure pattern as the staged path  -> the compute is wrong.
    // Pattern disappears                        -> the staging is wrong.
    // Not for production; no coalescing, no XOR swizzle.
    // ------------------------------------------------------------------
    template <typename OWindow, typename AccTile_, typename DWindow>
    CK_TILE_DEVICE void operator()(OWindow& o_window,
                                   const AccTile_& acc_tile,
                                   const DWindow&,
                                   void* = nullptr) const
    {
        const ck_tile::index_t wid = ck_tile::get_warp_id();
        if(wid < Pipeline::kNumProducerWaves)
        {
            return;
        }
        const ck_tile::index_t consumer_id = wid - Pipeline::kNumProducerWaves;
        const ck_tile::index_t lane        = ck_tile::get_lane_id();

        auto o_view        = o_window.get_bottom_tensor_view();
        const auto& o_desc = o_view.get_tensor_descriptor();
        const auto block_org = o_window.get_window_origin();

        // Raw base, no window origin: see the staged path for why adding it
        // again double counts the tile's row offset.
        auto* d_ptr = o_view.get_buffer_view().p_data_;
        const ck_tile::index_t ldd = o_desc.calculate_offset(ck_tile::make_multi_index(1, 0)) -
                                     o_desc.calculate_offset(ck_tile::make_multi_index(0, 0));
        [[maybe_unused]] const ck_tile::index_t M = o_desc.get_lengths()[ck_tile::number<0>{}];
        [[maybe_unused]] const ck_tile::index_t N = o_desc.get_lengths()[ck_tile::number<1>{}];

        const ck_tile::index_t row_in_tile = lane & 15;
        const ck_tile::index_t col_quad    = lane >> 4;
        const ck_tile::index_t col_base    = block_org[1] + consumer_id * kWaveN;

#pragma unroll
        for(int m = 0; m < static_cast<int>(AccTile::kMIters); ++m)
        {
            const ck_tile::index_t row = block_org[0] + m * 16 + row_in_tile;
            if constexpr(kBoundsCheck)
            {
                if(row >= M)
                {
                    continue;
                }
            }
#pragma unroll
            for(int n = 0; n < static_cast<int>(AccTile::kNIters); ++n)
            {
#pragma unroll
                for(int c = 0; c < 4; ++c)
                {
                    const ck_tile::index_t col = col_base + n * 16 + 4 * c + col_quad;
                    if(!kBoundsCheck || col < N)
                    {
                        d_ptr[row * ldd + col] = ck_tile::type_convert<ODataType>(
                            acc_tile.v[m * static_cast<int>(AccTile::kNIters) + n][c]);
                    }
                }
            }
        }
    }
#else
    template <typename OWindow, typename AccTile_, typename DWindow>
    CK_TILE_DEVICE void operator()(OWindow& o_window,
                                   const AccTile_& acc_tile,
                                   const DWindow&,
                                   void* p_smem = nullptr) const
    {
        const ck_tile::index_t wid = ck_tile::get_warp_id();
        if(wid < Pipeline::kNumProducerWaves)
        {
            return;   // producer waves own no output
        }
        if((ck_tile::wasp_detail::wasp_ablate() & 16) != 0)
        {
            // Ablation (see gemm_pipeline_ag_bg_cr_wasp.hpp): run the whole GEMM
            // but throw the writeback away, so the epilogue's cost can be timed
            // on its own.
            return;
        }
        // Only used by the DEBUG_RECORD block below; store_raw recomputes its
        // own. Keep for that diagnostic without tripping -Wunused otherwise.
        [[maybe_unused]] const ck_tile::index_t consumer_id =
            wid - Pipeline::kNumProducerWaves;
        [[maybe_unused]] const ck_tile::index_t lane = ck_tile::get_lane_id();

        auto o_view           = o_window.get_bottom_tensor_view();
        const auto& o_desc    = o_view.get_tensor_descriptor();
        const auto block_org  = o_window.get_window_origin();

        // d_ptr is deliberately the RAW tensor base, WITHOUT the window origin:
        // row_global and col_global below are already GLOBAL (they start at
        // block_org), so adding the origin again double counts it. That is
        // invisible for the first tile (origin 0) and pushes every store of
        // every later tile out of the buffer -- which then looks like "the
        // second m-tile was never written".
        auto* d_ptr = o_view.get_buffer_view().p_data_;
        const ck_tile::index_t ldd = o_desc.calculate_offset(ck_tile::make_multi_index(1, 0)) -
                                     o_desc.calculate_offset(ck_tile::make_multi_index(0, 0));
        const ck_tile::index_t M = o_desc.get_lengths()[ck_tile::number<0>{}];
        const ck_tile::index_t N = o_desc.get_lengths()[ck_tile::number<1>{}];

#if defined(CK_TILE_GROUPED_GEMM_WASP_DEBUG_RECORD)
        {
            const int cta = static_cast<int>(ck_tile::get_block_id());
            if(lane == 0 && cta >= 0 && cta < 128)
            {
                const auto p_data_low = static_cast<int>(
                    reinterpret_cast<uintptr_t>(o_view.get_buffer_view().p_data_) & 0xFFFFFFFFu);
                const auto d_ptr_low = static_cast<int>(
                    reinterpret_cast<uintptr_t>(d_ptr) & 0xFFFFFFFFu);
                g_wasp_dbg_waves[cta * 16 + static_cast<int>(wid)] = 1;
                g_wasp_dbg[cta * 10 + 0] = 1;
                g_wasp_dbg[cta * 10 + 1] = static_cast<int>(block_org[0]);
                g_wasp_dbg[cta * 10 + 2] = static_cast<int>(M);
                g_wasp_dbg[cta * 10 + 3] = static_cast<int>(N);
                g_wasp_dbg[cta * 10 + 4] = static_cast<int>(ldd);
                g_wasp_dbg[cta * 10 + 5] = static_cast<int>(
                    o_desc.calculate_offset(ck_tile::make_multi_index(block_org[0], block_org[1])));
                g_wasp_dbg[cta * 10 + 6] = p_data_low;
                g_wasp_dbg[cta * 10 + 7] = d_ptr_low;
                g_wasp_dbg[cta * 10 + 8] = static_cast<int>(block_org[1]);
                g_wasp_dbg[cta * 10 + 9] = static_cast<int>(consumer_id);
                // Proof of life: can this CTA's epilogue reach its own output
                // region at all? If the host does not see 777 here, the global
                // store itself is being lost, and no amount of bound/offset
                // reasoning matters.
                d_ptr[0] = ck_tile::type_convert<ODataType>(777.0F);
            }
        }
#endif

        store_raw(d_ptr, ldd, M, N, block_org[0], block_org[1], acc_tile, p_smem);
    }

    // Raw-pointer store core: the staged writeback fed by SCALARS instead of an
    // output tile window. `d_ptr` is the RAW tensor base (NO tile origin folded
    // in) -- row/col below are global and already start at row_base/col_base, so
    // adding the origin again would double-count it. row_base/col_base are the
    // tile's global row/col origin (block_org[0]/[1]). The grouped kernel's
    // RunGemmWasp path calls this directly with kargs-derived scalars, so no
    // output tile-window object is built or kept live.
    //
    // The role filter and the ablation gate live HERE (not only in the window
    // operator() above): RunGemmWasp calls store_raw directly on EVERY wave of
    // the workgroup, so producer waves (wid < kNumProducerWaves) must be dropped
    // here, exactly as the window path drops them before extracting the window.
#endif   // CK_TILE_GROUPED_GEMM_WASP_DIRECT_STORE

    CK_TILE_DEVICE static void store_raw(ODataType* d_ptr,
                                         ck_tile::index_t ldd,
                                         ck_tile::index_t M,
                                         ck_tile::index_t N,
                                         ck_tile::index_t row_base,
                                         ck_tile::index_t col_base,
                                         const AccTile& acc_tile,
                                         void* p_smem)
    {
        const ck_tile::index_t wid = ck_tile::get_warp_id();
        if(wid < Pipeline::kNumProducerWaves)
        {
            return;   // producer waves own no output
        }
        if((ck_tile::wasp_detail::wasp_ablate() & 16) != 0)
        {
            return;   // ablation: run the GEMM but throw the writeback away
        }
        const ck_tile::index_t consumer_id = wid - Pipeline::kNumProducerWaves;
        const ck_tile::index_t lane        = ck_tile::get_lane_id();

        auto* const slab = reinterpret_cast<ODataType*>(static_cast<char*>(p_smem) +
                                                        consumer_id * kSlabBytesPerWave);

        const ck_tile::index_t row_base_global = row_base;
        const ck_tile::index_t col_base_global = col_base + consumer_id * kWaveN;

#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        uint64_t epi_times[10]{};
        asm volatile("s_memtime %0" : "=s"(epi_times[0]));
#endif
        // Convert the complete accumulator before LDS staging, as in the HIPC
        // epilogue. This deliberately keeps the converted tile live across both
        // halves instead of repeating conversion while issuing LDS stores.
        ODataType converted[AccTile::kMIters * AccTile::kNIters][4];
        ck_tile::static_for<0, AccTile::kMIters * AccTile::kNIters, 1>{}([&](auto flat) {
            ck_tile::static_for<0, 4, 1>{}([&](auto c) {
                if constexpr(std::is_same_v<ODataType, ck_tile::bf16_t>)
                {
                    converted[flat.value][c.value] =
                        ck_tile::bit_cast<ODataType>(
                            FloatToBf16RneNonNan(acc_tile.v[flat.value][c.value]));
                }
                else
                {
                    converted[flat.value][c.value] =
                        ck_tile::type_convert<ODataType>(acc_tile.v[flat.value][c.value]);
                }
            });
        });
#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        asm volatile("s_memtime %0" : "=s"(epi_times[1]));
#endif

        StoreHalfToLds<0>(slab, lane, converted);
#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        asm volatile("s_memtime %0" : "=s"(epi_times[2]));
#endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        asm volatile("s_memtime %0" : "=s"(epi_times[3]));
#endif
        LoadLdsAndStoreGlobal<0>(
            slab, d_ptr, ldd, row_base_global, col_base_global, M, N, lane);
#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        asm volatile("s_memtime %0" : "=s"(epi_times[4]));
#endif

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        asm volatile("s_memtime %0" : "=s"(epi_times[5]));
#endif
        StoreHalfToLds<1>(slab, lane, converted);
#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        asm volatile("s_memtime %0" : "=s"(epi_times[6]));
#endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        asm volatile("s_memtime %0" : "=s"(epi_times[7]));
#endif
        LoadLdsAndStoreGlobal<1>(
            slab, d_ptr, ldd, row_base_global, col_base_global, M, N, lane);
#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
        asm volatile("s_memtime %0" : "=s"(epi_times[8]));
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        asm volatile("s_memtime %0" : "=s"(epi_times[9]));
        if(lane == 0)
        {
            const ck_tile::index_t cta = ck_tile::get_block_id();
            if(cta < 1024)
            {
                const ck_tile::index_t index = (cta * 16 + wid) * 9;
                for(int stage = 0; stage < 9; ++stage)
                {
                    g_wasp_epi_cycles[index + stage] =
                        epi_times[stage + 1] - epi_times[stage];
                }
            }
        }
#endif
    }
};
