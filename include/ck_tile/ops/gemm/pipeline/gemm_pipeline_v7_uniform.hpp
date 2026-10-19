// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Uniform (non warp-specialized) Direct-to-LDS grouped-GEMM pipeline for the
// gfx936 / gfx938 "v7" NN (dgrad) and TN (wgrad) kernels, ported into the
// ck_tile pipeline vocabulary.
//
// WHY A SEPARATE PIPELINE FAMILY
// ------------------------------
// The two operand layouts this serves share one structure that no existing CK
// pipeline has: at least one operand is [K, feature]-major, so its per-lane MMAC
// fragment is NOT a contiguous 16 B run in global memory. The treatment (from
// the reference kernels, where it was measured as the only arrangement that fits
// this LDS budget) is
//
//   * fetch 16 B (8 features) of one reduction row per lane into LDS with
//     Direct-to-LDS (`buffer_load_dwordx4 ... offen lds`),
//   * let the hardware perform the K/fragment redistribution inside the LDS READ
//     (`ds_read_m32x16_b16`), and
//   * interleave the per-lane store order so the read is bank-conflict-free.
//
// NN  : A = [M, K] K-contiguous (cheap: loaded as a fragment, one
//       `ds_read_b64` per m-tile) and B = [K, N] N-contiguous (m32x16 read).
// TN  : BOTH operands are [K, feature]-major, so both go through m32x16; the one
//       read instruction serves A (a 64-column window, 2 reads) and B (a
//       128-column window, 4 reads) because the A and B fragments of the MMAC
//       are the same lane pattern.
//
// GEOMETRY (fixed by the reference kernels; the CK config must match)
// ------------------------------------------------------------------
//   block tile 256M x 256N x 16K, 512 threads = 8 waves of 64
//   wave grid  4m x 2n, wave tile 64M x 128N, 32 MMACs per k-stage
//   LDS        4-stage ring of 16 KB (8 KB A + 8 KB B), the whole 64 KB budget
//   phase      12-phase unroll, operand ring 3, prefetch distance 5 (ring + 1),
//               two-group wave interleave (waves 0-3 mmac-first, waves 4-7
//               load-first) -- see the phase-discipline note below.
//
// WHY THE KERNEL MUST RUN ONE TILE PER CTA (measured, not stylistic)
// -----------------------------------------------------------------
// Wrapping this pipeline's k-loop in a tile loop -- even a loop the compiler can
// see through -- costs 40 VGPRs (215 -> 255/256) and a scratch spill, which
// measured ~20% of the throughput. The scan over the group table and the per-tile
// workgroup barrier are both free; it is the loop itself, because the epilogue's
// live range then overlaps the next tile's prologue in the register allocator.
// The device-args entry therefore launches a FLAT grid (one CTA per tile, the
// grid sized from a host-side tile bound) and re-derives the (group, m, n) tile
// coordinates from the device kargs inside the kernel. Do not "simplify" this
// into a persistent loop.
//
// PHASE DISCIPLINE (carried over verbatim, with the measurements that set it)
// ---------------------------------------------------------------------------
//   * Two-group interleave, not a uniform order: the LDS bursts of waves 0-3 and
//     4-7 stay one MMAC phase apart, so a load's LDS write-back always finds a
//     quiet window. The uniform (blasLt-like) order measured -12%, the mirror of
//     the interleave (group-B first) is throughput-neutral.
//   * The load tail waits are vmcnt(4) / lgkmcnt(8): with 2 loads per step and a
//     4-slot ring, vmcnt(4) is 2 steps of lookback, which is the loosest wait
//     that still guards the slot being re-filled.
//   * s_setprio(3) around the MMAC burst, s_setprio(1) elsewhere (blasLt's
//     scheme; measured inside the run-to-run spread, kept as the reference
//     kernel has it).
//   * The per-step workgroup barrier is NOT optional: dropping it measured +1.35%
//     and corrupts the result (the ring's cross-wave visibility depends on it).
//
// TILING CONTRACT
// ---------------
//   NN : A = [M, K] row major (a_stride), B = [K, N] row major (b_stride, the
//        group's own panel), D = [M, N] row major (d_stride)
//   TN : A = [K, M] row major (a_stride, M contiguous), B = [K, N] row major
//        (b_stride), D = [M, N] row major (d_stride)
// Rows past M and columns past N are masked on the STORE only; the K extent must
// be a whole number of 16-wide k-stages and N a whole number of 256-wide tiles
// (both are contract-level requirements, enforced by the host-side selector).
// The A panel fetch reads a whole 256-column tile, so TN additionally requires
// addressable slack after the last group's A columns (the caller's obligation,
// documented by the standalone kernel's header).
#pragma once

#include "ck_tile/core.hpp"

// V7UniformKind, V7UniformTileParams (complete) and the pipeline template
// declaration: the shared copy every grouped-GEMM TU already carries.
#include "gemm_pipeline_v7_uniform_fwd.hpp"

// The address arithmetic below is a port of the reference kernels, which are
// built with -Wno-old-style-cast; every cast is a fixed-width unsigned/signed
// conversion in an address expression. CK's own arch/buffer-addressing headers
// handle it the same way (see cast_pointer_to_constant_address_space).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"

namespace ck_tile {

// ---------------------------------------------------------------------------
// Geometry.
// ---------------------------------------------------------------------------
struct V7UniformGeometry
{
    static constexpr index_t kBlockM       = 256;
    static constexpr index_t kBlockN       = 256;
    static constexpr index_t kStageK       = 16;
    static constexpr index_t kThreads      = 512;
    static constexpr index_t kWaves        = 8;
    static constexpr index_t kWaveSize     = 64;
    static constexpr index_t kWaveGridM    = 4;
    static constexpr index_t kWaveGridN    = 2;
    static constexpr index_t kWaveTileM    = kBlockM / kWaveGridM;   // 64
    static constexpr index_t kWaveTileN    = kBlockN / kWaveGridN;   // 128
    static constexpr index_t kMfmaM        = 16;
    static constexpr index_t kMfmaN        = 16;
    static constexpr index_t kMfmaK        = 16;
    static constexpr index_t kMTilesPerWave = kWaveTileM / kMfmaM;   // 4
    static constexpr index_t kNTilesPerWave = kWaveTileN / kMfmaN;   // 8
    static constexpr index_t kMmacPerStage  = kMTilesPerWave * kNTilesPerWave;   // 32

    static constexpr index_t kNumStages   = 4;
    static constexpr index_t kOpRing      = 3;
    static constexpr index_t kPhaseCycle  = 12;
    static constexpr index_t kWaveGroup   = kWaves / 2;
    static constexpr index_t kPrefetchDist = kNumStages + 1;   // 5

    static constexpr index_t kLdsBytesA        = kBlockM * kStageK * 2;   // 8 KB
    static constexpr index_t kLdsBytesB        = kStageK * kBlockN * 2;   // 8 KB
    static constexpr index_t kLdsBytesPerStage = kLdsBytesA + kLdsBytesB; // 16 KB
    static constexpr index_t kLdsBytes = kLdsBytesPerStage * kNumStages;  // 64 KB

    static constexpr index_t kAccVgprPerLane = kMmacPerStage * 4;   // 128

    static constexpr index_t kAccMTiles = kMTilesPerWave;
    static constexpr index_t kAccNTiles = kNTilesPerWave;

    static_assert(kThreads == kWaves * kWaveSize, "512 = 8 x 64");
    static_assert(kMfmaK == kStageK, "one MMAC covers exactly one k-stage");
    static_assert(kLdsBytes == 64 * 1024, "the ring is the whole CU LDS budget");
    static_assert(kPhaseCycle % kOpRing == 0 && kPhaseCycle % kNumStages == 0,
                  "the unroll must span whole operand sets and whole LDS rings");
    static_assert(kPrefetchDist > kNumStages, "distance ring+1 is the whole point");
};

// ---------------------------------------------------------------------------
// Accumulator tile: 4 m-iterations x 8 n-iterations x 4 fp32 = 128 VGPR/lane.
// A VECTOR quad per slot (not four scalars): that is the granularity the
// register allocator works with around every MMAC, and the arrangement the
// reference kernels use.
// ---------------------------------------------------------------------------
struct V7UniformAccTile
{
    using vec_t = float __attribute__((ext_vector_type(4)));
    vec_t v[V7UniformGeometry::kAccMTiles * V7UniformGeometry::kAccNTiles];
};

// V7UniformKind and the pipeline template declaration come from the forward
// header: it is the copy every grouped-GEMM TU already carries.
namespace v7u_detail {

using vec4_bf16 = unsigned short __attribute__((ext_vector_type(4)));
using vec8_bf16 = unsigned short __attribute__((ext_vector_type(8)));
using vec4_fp32 = float __attribute__((ext_vector_type(4)));
using u32x4     = int32_t __attribute__((ext_vector_type(4)));

CK_TILE_DEVICE void wait_all_warp_arrived()
{
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_barrier\n");
    __builtin_amdgcn_sched_barrier(0);
}

template <bool Sync>
CK_TILE_DEVICE void wait_all_buffer_data_arrived(int wait_count = 0)
{
    __builtin_amdgcn_sched_barrier(0);
    if constexpr(Sync)
    {
        asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier\n" ::"n"(wait_count));
    }
    else
    {
        asm volatile("s_waitcnt vmcnt(%0)\n" ::"n"(wait_count));
    }
    __builtin_amdgcn_sched_barrier(0);
}

template <bool Sync>
CK_TILE_DEVICE void wait_lds_data_arrived(int wait_count = 0)
{
    __builtin_amdgcn_sched_barrier(0);
    if constexpr(Sync)
    {
        asm volatile("s_waitcnt lgkmcnt(%0)\n\ts_barrier\n" ::"n"(wait_count));
    }
    else
    {
        asm volatile("s_waitcnt lgkmcnt(%0)\n" ::"n"(wait_count));
    }
    __builtin_amdgcn_sched_barrier(0);
}

template <int Prio>
CK_TILE_DEVICE void set_prio()
{
    static_assert(Prio >= 0 && Prio <= 3, "s_setprio operand must be 0..3");
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio %0" ::"n"(Prio));
    __builtin_amdgcn_sched_barrier(0);
}

// MMAC. The 4-element bf16 operands and the 4-element fp32 accumulator are the
// instruction's own fragment layout; the redistribution from LDS is the
// ds_read_m32x16's job (see the header note).
CK_TILE_DEVICE vec4_fp32 mmac_bf16(const vec4_bf16& a, const vec4_bf16& b, const vec4_fp32& c)
{
    return __builtin_hcu_mmac_f32_16x16x16_bf16(a, b, c);
}

using buffer_rsrc_t = int32_t __attribute__((ext_vector_type(4)));

// Buffer resource descriptor. STRIDE is the CACHE_SWIZZLE rotation period (bit
// 30) on the A side of NN -- the one arch-dependent field, gated on the build's
// arch macros. Everything else is layout-neutral.
CK_TILE_DEVICE buffer_rsrc_t make_buffer_rsrc(const void* ptr,
                                              uint32_t stride_bytes,
                                              uint32_t size_bytes,
                                              bool allow_swizzle)
{
    buffer_rsrc_t r;
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    r[0]                = (int32_t)(addr & 0xffffffffu);
    r[1]                = (int32_t)(addr >> 32);
    r[2]                = (int32_t)size_bytes;
    r[3]                = 0x00020000;
    uint32_t w1         = (stride_bytes & 0x3fffu) << 16;
#if defined(__gfx936__) || defined(__gfx938__)
    if(allow_swizzle && stride_bytes % 64 == 0)
    {
        const uint32_t s64 = stride_bytes / 64;
        if((s64 & (s64 - 1)) == 0) w1 |= 1u << 30;
    }
#else
    (void)allow_swizzle;   // CACHE_SWIZZLE does not exist on this part
#endif
    r[1] |= (int32_t)w1;
    return r;
}

// Global -> LDS 16 B per lane. `idxen` is the pair {index lo, offset hi} form;
// `offen` keeps the whole address in the vaddr and the k-advance in the soffset.
CK_TILE_DEVICE void buffer_load_dwordx4_offen_lds(buffer_rsrc_t rsrc,
                                                  uint32_t vaddr,
                                                  uint32_t soffset,
                                                  uint32_t m0)
{
    asm volatile("s_mov_b32 m0, %[m0]\n\t"
                 "buffer_load_dwordx4 %[v], %[rsrc], %[so] offen lds\n"
                 :
                 : [v] "v"(vaddr), [rsrc] "s"(rsrc), [so] "s"(soffset), [m0] "s"(m0)
                 : "memory");
}

// SLC (evict-first) is set on the NN A load on purpose: it is the 32-row
// scattered load that overruns the VL1 address/alloc path, and marking its lines
// evict-first measured +0.8%. The B side must NOT carry it (-6.2%).
CK_TILE_DEVICE void buffer_load_dwordx4_idxen_lds_t(buffer_rsrc_t rsrc,
                                                    uint64_t pair,
                                                    uint32_t soffset,
                                                    uint32_t m0)
{
    asm volatile("s_mov_b32 m0, %[m0]\n\t"
                 "buffer_load_dwordx4 %[pair], %[rsrc], %[so] idxen offen slc lds\n"
                 :
                 : [pair] "v"(pair), [rsrc] "s"(rsrc), [so] "s"(soffset), [m0] "s"(m0)
                 : "memory");
}

CK_TILE_DEVICE void buffer_load_dwordx4_idxen_lds(buffer_rsrc_t rsrc,
                                                  uint64_t pair,
                                                  uint32_t soffset,
                                                  uint32_t m0)
{
    asm volatile("s_mov_b32 m0, %[m0]\n\t"
                 "buffer_load_dwordx4 %[pair], %[rsrc], %[so] idxen offen lds\n"
                 :
                 : [pair] "v"(pair), [rsrc] "s"(rsrc), [so] "s"(soffset), [m0] "s"(m0)
                 : "memory");
}

// LDS reads. b64 x4 = one b64 per m-tile from a single base + immediates (the
// NN A side, whose per-lane fragment IS contiguous); m32x16 x8 / x4 = the
// hardware redistribution used for every [K, feature]-major operand.
template <uint32_t kOff = 0>
CK_TILE_DEVICE void ds_read_b64_x4(
    uint32_t base, vec4_bf16& f0, vec4_bf16& f1, vec4_bf16& f2, vec4_bf16& f3)
{
    asm volatile("ds_read_b64 %0, %4 offset:%5\n\t"
                 "ds_read_b64 %1, %4 offset:%6\n\t"
                 "ds_read_b64 %2, %4 offset:%7\n\t"
                 "ds_read_b64 %3, %4 offset:%8\n"
                 : "=&v"(f0), "=&v"(f1), "=&v"(f2), "=&v"(f3)
                 : "v"(base), "n"(kOff), "n"(kOff + 512u), "n"(kOff + 1024u),
                   "n"(kOff + 1536u)
                 : "memory");
}

template <uint32_t kOff = 0>
CK_TILE_DEVICE void ds_read_m32x16_x8(const uint32_t (&rb)[4], vec4_bf16 (&out)[8])
{
    vec8_bf16 d0, d1, d2, d3;
    asm volatile("ds_read_m32x16_b16 %0, %4 offset:%8\n\t"
                 "ds_read_m32x16_b16 %1, %5 offset:%8\n\t"
                 "ds_read_m32x16_b16 %2, %6 offset:%8\n\t"
                 "ds_read_m32x16_b16 %3, %7 offset:%8\n"
                 : "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3)
                 : "v"(rb[0]), "v"(rb[1]), "v"(rb[2]), "v"(rb[3]), "n"(kOff)
                 : "memory");
    out[0] = __builtin_shufflevector(d0, d0, 0, 1, 2, 3);
    out[1] = __builtin_shufflevector(d0, d0, 4, 5, 6, 7);
    out[2] = __builtin_shufflevector(d1, d1, 0, 1, 2, 3);
    out[3] = __builtin_shufflevector(d1, d1, 4, 5, 6, 7);
    out[4] = __builtin_shufflevector(d2, d2, 0, 1, 2, 3);
    out[5] = __builtin_shufflevector(d2, d2, 4, 5, 6, 7);
    out[6] = __builtin_shufflevector(d3, d3, 0, 1, 2, 3);
    out[7] = __builtin_shufflevector(d3, d3, 4, 5, 6, 7);
}

template <uint32_t kOff = 0>
CK_TILE_DEVICE void ds_read_m32x16_x4(const uint32_t (&rb)[2], vec4_bf16 (&out)[4])
{
    vec8_bf16 d0, d1;
    asm volatile("ds_read_m32x16_b16 %0, %2 offset:%4\n\t"
                 "ds_read_m32x16_b16 %1, %3 offset:%4\n"
                 : "=&v"(d0), "=&v"(d1)
                 : "v"(rb[0]), "v"(rb[1]), "n"(kOff)
                 : "memory");
    out[0] = __builtin_shufflevector(d0, d0, 0, 1, 2, 3);
    out[1] = __builtin_shufflevector(d0, d0, 4, 5, 6, 7);
    out[2] = __builtin_shufflevector(d1, d1, 0, 1, 2, 3);
    out[3] = __builtin_shufflevector(d1, d1, 4, 5, 6, 7);
}

}   // namespace v7u_detail

// Namespace alias for the asm-level helpers, declared where C++ allows one.
namespace v7u = v7u_detail;

// ---------------------------------------------------------------------------
// The pipeline. One instance serves one tile: `run_tile` runs the whole K loop
// and leaves the accumulator in `acc` for the epilogue.
// ---------------------------------------------------------------------------
template <typename Problem, V7UniformKind kKind>
struct GemmPipelineV7Uniform
{
    using G       = V7UniformGeometry;
    using AccTile = V7UniformAccTile;

    // The framework's trait for the raw (no tensor view, no tile window) kernel
    // path: see GroupedGemmKernel's is_grouped_gemm_v7_uniform_pipeline.
    static constexpr bool IsV7UniformPipeline = true;

    static constexpr V7UniformKind kind = kKind;

    // ---- framework-facing surface (what GroupedGemmKernel reads) --------
    // The kernel takes BlockSize as a VALUE, not a call: keep the member form.
    static constexpr index_t BlockSize = G::kThreads;

    using ADataType = remove_cvref_t<typename Problem::ADataType>;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;
    using ALayout   = remove_cvref_t<typename Problem::ALayout>;
    using BLayout   = remove_cvref_t<typename Problem::BLayout>;
    using CLayout   = remove_cvref_t<typename Problem::CLayout>;

    static constexpr index_t KPerBlock = remove_cvref_t<typename Problem::BlockGemmShape>::kK;

    // The one-CTA-per-tile structure the register budget depends on: the config
    // must be non-persistent, and nothing double-buffers.
    static constexpr bool UsePersistentKernel = false;
    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr bool Preshuffle          = false;

    // The LDS the ring occupies (the whole CU budget).
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return G::kLdsBytes; }

    using AOpBank = v7u::vec4_bf16[G::kOpRing][G::kMTilesPerWave];
    using BOpBank = v7u::vec4_bf16[G::kOpRing][G::kNTilesPerWave];

    // Per-thread scalar state: the descriptors / per-lane bases the k loop needs.
    struct Addr
    {
        v7u::buffer_rsrc_t a_rsrc, b_rsrc;
        uint32_t a_vaddr, b_vaddr;      // per-lane global byte offsets (TN, NN-B)
        uint64_t a_pair;                // NN A: {index, offset} idxen pair
        uint32_t a_rd;                  // NN A b64 read base
        uint32_t a_rb[2];               // TN A m32x16 read bases
        uint32_t b_rb[4];               // B m32x16 read bases
        uint32_t m0_a_base, m0_b_base;
        int b_stride_bytes;
        int a_stride_bytes;   // TN: the A panel's row pitch (the k advance);
                              // NN keeps the K-contiguous 2-byte advance and
                              // never reads this.
        int n_stages;
        int wid;
    };

    CK_TILE_DEVICE static void mmac32(AccTile& acc,
                                      const AOpBank& a_bank,
                                      const BOpBank& b_bank,
                                      int bank)
    {
        // The pragma-unrolled loop body is written out for bank 0 so the MMAC
        // burst starts with a `s_setprio 3` as in the reference kernel.
        acc.v[0] = v7u::mmac_bf16(a_bank[bank][0], b_bank[bank][0], acc.v[0]);
        v7u::set_prio<3>();
#pragma unroll
        for(index_t i = 1; i < G::kMmacPerStage; ++i)
        {
            const index_t m = i / G::kNTilesPerWave;
            const index_t n = i % G::kNTilesPerWave;
            acc.v[i]       = v7u::mmac_bf16(a_bank[bank][m], b_bank[bank][n], acc.v[i]);
        }
        v7u::set_prio<1>();
    }

    template <int kBank, uint32_t kReadSlot, uint32_t kLoadSlot>
    CK_TILE_DEVICE static void issue_load_and_read(Addr& s,
                                                   AOpBank& a_bank,
                                                   BOpBank& b_bank,
                                                   int kt)
    {
        if((kt + 2) < s.n_stages)
        {
            if constexpr(kKind == V7UniformKind::NN)
            {
                v7u::ds_read_m32x16_x8<kReadSlot>(s.b_rb, b_bank[kBank]);
                v7u::ds_read_b64_x4<kReadSlot>(
                    s.a_rd, a_bank[kBank][0], a_bank[kBank][1], a_bank[kBank][2], a_bank[kBank][3]);
            }
            else
            {
                v7u::ds_read_m32x16_x8<kReadSlot>(s.b_rb, b_bank[kBank]);
                v7u::ds_read_m32x16_x4<kReadSlot>(s.a_rb, a_bank[kBank]);
            }
        }
        // The k advance rides the soffsets in every form: the descriptor base is
        // never rewritten inside the loop.
        const int kStage   = (kt + G::kPrefetchDist < s.n_stages) ? kt + G::kPrefetchDist : 0;
        // A's k advance: NN walks a K-contiguous [M, K] operand (32 B per
        // 16-wide stage); TN walks the [K, M] panel row-major, so one stage is
        // a_stride_bytes (16 panel rows). The reference kernels advance
        // exactly this way (bf16_gemm_tn_hipc issue_load_and_read).
        const uint32_t so_a =
            (uint32_t)(kStage * G::kStageK *
                       (kKind == V7UniformKind::NN ? 2u : (uint32_t)s.a_stride_bytes));
        const uint32_t so_b = (uint32_t)(kStage * G::kStageK * s.b_stride_bytes);
        if constexpr(kKind == V7UniformKind::NN)
        {
            v7u::buffer_load_dwordx4_idxen_lds_t(
                s.a_rsrc, s.a_pair, so_a, s.m0_a_base + kLoadSlot);
            v7u::buffer_load_dwordx4_offen_lds(
                s.b_rsrc, s.b_vaddr, so_b, s.m0_b_base + kLoadSlot);
        }
        else
        {
            v7u::buffer_load_dwordx4_offen_lds(
                s.a_rsrc, s.a_vaddr, so_a, s.m0_a_base + kLoadSlot);
            v7u::buffer_load_dwordx4_offen_lds(
                s.b_rsrc, s.b_vaddr, so_b, s.m0_b_base + kLoadSlot);
        }
    }

    // One k-stage. kGroupA selects the phase order (compile-time: the divergent
    // runtime branch per step compiled to a full exec-mask prologue every step);
    // kBarrier=false on the last step of an unrolled cycle, where the caller
    // emits the barrier after its loop control so the trip test overlaps the
    // drain instead of being exposed behind it.
    template <int kPos,
              bool kGroupA  = false,
              bool kBarrier = true,
              int kBankCur  = kPos % G::kOpRing,
              uint32_t kBankNext = (kPos + 2) % G::kOpRing,
              uint32_t kReadSlot = (uint32_t)(((kPos + 2) % G::kNumStages) * G::kLdsBytesPerStage),
              uint32_t kLoadSlot =
                  (uint32_t)(((kPos + G::kPrefetchDist) % G::kNumStages) * G::kLdsBytesPerStage)>
    CK_TILE_DEVICE static void stage_step(Addr& s, AccTile& acc, AOpBank& a_bank, BOpBank& b_bank, int kt)
    {
        if constexpr(kGroupA)
        {
            mmac32(acc, a_bank, b_bank, kBankCur);
            issue_load_and_read<kBankNext, kReadSlot, kLoadSlot>(s, a_bank, b_bank, kt);
            v7u::wait_lds_data_arrived<false>(8);
        }
        else
        {
            issue_load_and_read<kBankNext, kReadSlot, kLoadSlot>(s, a_bank, b_bank, kt);
            v7u::wait_lds_data_arrived<false>(8);
            mmac32(acc, a_bank, b_bank, kBankCur);
        }
        v7u::wait_all_buffer_data_arrived<false>(4);
        if constexpr(kBarrier)
        {
            v7u::wait_all_warp_arrived();
        }
    }

    CK_TILE_DEVICE static void run_tile(const V7UniformTileParams& p, AccTile& acc, void* smem);
};

// ---------------------------------------------------------------------------
// The tile body. Written once, with `if constexpr` on the operand layout; the
// instruction order, the waits and the ring arithmetic are the reference
// kernels' (see the header note for what each one is worth).
// ---------------------------------------------------------------------------
template <typename Problem, V7UniformKind kKind>
CK_TILE_DEVICE void GemmPipelineV7Uniform<Problem, kKind>::run_tile(
    const V7UniformTileParams& p, AccTile& acc, void* smem)
{
    using namespace v7u_detail;
    (void)smem;   // the ring is the launcher's dynamic LDS allocation

    const index_t tid  = threadIdx.x;
    const int wid      = (int)(tid >> 6);
    const int lane     = (int)(tid & (G::kWaveSize - 1));
    const int wave_m   = wid >> 1;
    const int wave_n   = wid & 1;
    const int wave_m_start = wave_m * (int)G::kWaveTileM;

    const int m_block_start = p.m_start;
    const int n_block_start = p.n_start;

#pragma unroll
    for(index_t i = 0; i < G::kMmacPerStage; ++i) acc.v[i] = vec4_fp32{0, 0, 0, 0};

    Addr S;
    S.wid            = wid;
    S.n_stages       = (int)(p.k / G::kStageK);
    S.b_stride_bytes = (int)p.b_stride * 2;

    const uint32_t m0_a_base = __builtin_amdgcn_readfirstlane((uint32_t)(wid * 1024));
    const uint32_t m0_b_base =
        __builtin_amdgcn_readfirstlane((uint32_t)(G::kLdsBytesA + wid * 1024));
    S.m0_a_base = m0_a_base;
    S.m0_b_base = m0_b_base;

    if constexpr(kKind == V7UniformKind::NN)
    {
        // ---- A = [M, K], K-contiguous ------------------------------------
        const int a_stride_bytes = (int)p.a_stride * 2;
        // Lane pairs fetch one row's two 16 B k-halves; rows r = 4..7, 12..15
        // (mod 16) take the halves swapped, which spreads the b64 read phases
        // over all 8 bank groups (a plain row-major 32 B stride reaches 4 and
        // measured 2-way conflicts on every read phase).
        const int a_row_global = m_block_start + wid * 32 + (lane >> 1);
        const uint32_t a_kh =
            ((((lane >> 1) & 15) >> 2) & 1) ^ (uint32_t)(lane & 1);
        // Descriptor STRIDE = rotation period: the largest power-of-two multiple
        // of 64 that fits the 14-bit field. Non-power-of-two pitches keep the
        // rotation through the idxen split (index*STRIDE + row*rem + col is
        // byte-identical to row*pitch + col).
        S.a_stride_bytes    = a_stride_bytes;   // unused on the NN advance path
        uint32_t a_stride_desc = 64u;
        while(a_stride_desc * 2u <= (uint32_t)a_stride_bytes && a_stride_desc * 2u <= 0x3fffu)
        {
            a_stride_desc *= 2u;
        }
        const uint32_t a_off_rem = (uint32_t)a_stride_bytes - a_stride_desc;
        S.a_pair = (uint64_t)(uint32_t)a_row_global |
                   ((uint64_t)((uint32_t)(a_row_global * (int)a_off_rem + a_kh * 16)) << 32);
        const uint32_t a_size_bytes = (uint32_t)((int64_t)p.m * a_stride_bytes);
        S.a_rsrc = make_buffer_rsrc(p.a_ptr, a_stride_desc, a_size_bytes, /*swizzle=*/true);
        // Read base: row (lane&15) at +32*row, 8 B k-unit (lane>>4) at +du, with
        // du compensating the swapped rows' [kh1|kh0] placement.
        const uint32_t a_rho = (uint32_t)(lane & 15);
        const uint32_t a_uu  = (uint32_t)(lane >> 4);
        const uint32_t a_t   = (a_rho >> 2) & 1;
        const uint32_t a_du  = a_t ? ((a_uu < 2u) ? 16u + 8u * a_uu : 8u * a_uu - 16u) : 8u * a_uu;
        S.a_rd = (uint32_t)(wave_m_start * 32 + a_rho * 32 + a_du);
    }
    else
    {
        // ---- A = [K, M], M-contiguous (the mirrored NN-B case) ------------
        const int a_stride_bytes = (int)p.a_stride * 2;
        const int a_k_in_stage   = wid * 2 + (lane >> 5);
        const int a_m_global =
            m_block_start + 8 * (((lane >> 3) & 3) * 8 + ((lane & 7) ^ (4 * ((lane >> 5) & 1))));
        S.a_vaddr = (uint32_t)(a_k_in_stage * a_stride_bytes + a_m_global * 2);
        const uint32_t a_size_bytes = (uint32_t)((int64_t)p.k * a_stride_bytes);
        S.a_rsrc = make_buffer_rsrc(p.a_ptr, (uint32_t)a_stride_bytes, a_size_bytes, false);
        S.a_stride_bytes = a_stride_bytes;
    }

    {
        // ---- B = [K, N], N-contiguous (both layouts) ----------------------
        const int b_stride_bytes = S.b_stride_bytes;
        // Monotonic walk: lanes 0-31 walk row 2w's full 512 B in address order,
        // lanes 32-63 walk row 2w+1's; the odd row's lane->octet map is XORed by
        // 4 so the two rows land on complementary bank halves. That is the L1
        // request-merge shape (the merged walk measured 310.8 -> 369.1 TF).
        const int b_k_in_stage = wid * 2 + (lane >> 5);
        const int b_n_global =
            n_block_start + 8 * (((lane >> 3) & 3) * 8 + ((lane & 7) ^ (4 * ((lane >> 5) & 1))));
        S.b_vaddr     = (uint32_t)(b_k_in_stage * b_stride_bytes + b_n_global * 2);
        const uint32_t b_size_bytes = (uint32_t)((int64_t)p.k * b_stride_bytes);
        S.b_rsrc      = make_buffer_rsrc(p.b_ptr, (uint32_t)b_stride_bytes, b_size_bytes, false);
    }

    // Read anchors, one per-lane vaddr per read: lane l supplies read r's anchor
    // for octet o = base + 4r + (l&3), row parity rw = (l>>2)&1, window (l>>3):
    //   addr = 1024*(l>>3) + 512*rw + 16*(o ^ (4*rw))
    // (the ^4*rw is the same phase term the walk applies to the odd row, so the
    // two reads hit complementary bank halves). One register per read is what
    // makes "+4 octets" a free immediate; the per-lane anchors are why the read
    // forms take arrays.
    const uint32_t rw = (uint32_t)((lane >> 2) & 1);
    const uint32_t ph = 4u * rw;
    const uint32_t b_rd0 =
        (uint32_t)G::kLdsBytesA + (uint32_t)(1024 * (lane >> 3)) + 512u * rw;
#pragma unroll
    for(int r = 0; r < 4; ++r)
    {
        const uint32_t o = (uint32_t)(16 * wave_n) + 4u * (uint32_t)r + (uint32_t)(lane & 3);
        S.b_rb[r]        = b_rd0 + 16u * (o ^ ph);
    }
    const uint32_t a_rd0 = (uint32_t)(1024 * (lane >> 3)) + 512u * rw;
#pragma unroll
    for(int r = 0; r < 2; ++r)
    {
        const uint32_t o = (uint32_t)(8 * wave_m) + 4u * (uint32_t)r + (uint32_t)(lane & 3);
        S.a_rb[r]        = a_rd0 + 16u * (o ^ ph);
    }

    AOpBank a_bank;
    BOpBank b_bank;

    // Prologue: fill the ring, consume banks 0/1, prime the pre-loop slot.
    // so_a: NN walks K-contiguous A (32 B per stage); TN walks the [K, M]
    // panel row-major (a_stride_bytes per stage) -- see issue_load_and_read.
#pragma unroll
    for(index_t st = 0; st < G::kNumStages; ++st)
    {
        const uint32_t so_a =
            (uint32_t)(st * G::kStageK *
                       (kKind == V7UniformKind::NN ? 2u : (uint32_t)S.a_stride_bytes));
        const uint32_t so_b = (uint32_t)(st * G::kStageK * S.b_stride_bytes);
        if constexpr(kKind == V7UniformKind::NN)
        {
            buffer_load_dwordx4_idxen_lds_t(
                S.a_rsrc, S.a_pair, so_a, S.m0_a_base + (uint32_t)(st * G::kLdsBytesPerStage));
            buffer_load_dwordx4_offen_lds(
                S.b_rsrc, S.b_vaddr, so_b, S.m0_b_base + (uint32_t)(st * G::kLdsBytesPerStage));
        }
        else
        {
            buffer_load_dwordx4_offen_lds(
                S.a_rsrc, S.a_vaddr, so_a, S.m0_a_base + (uint32_t)(st * G::kLdsBytesPerStage));
            buffer_load_dwordx4_offen_lds(
                S.b_rsrc, S.b_vaddr, so_b, S.m0_b_base + (uint32_t)(st * G::kLdsBytesPerStage));
        }
    }

    wait_all_buffer_data_arrived<true>(2 * (G::kNumStages - 1));
    if constexpr(kKind == V7UniformKind::NN)
    {
        ds_read_b64_x4<0>(S.a_rd, a_bank[0][0], a_bank[0][1], a_bank[0][2], a_bank[0][3]);
    }
    else
    {
        ds_read_m32x16_x4<0>(S.a_rb, a_bank[0]);
    }
    ds_read_m32x16_x8<0>(S.b_rb, b_bank[0]);
    wait_all_buffer_data_arrived<true>(2 * (G::kNumStages - 2));
    if constexpr(kKind == V7UniformKind::NN)
    {
        ds_read_b64_x4<G::kLdsBytesPerStage>(
            S.a_rd, a_bank[1][0], a_bank[1][1], a_bank[1][2], a_bank[1][3]);
    }
    else
    {
        ds_read_m32x16_x4<G::kLdsBytesPerStage>(S.a_rb, a_bank[1]);
    }
    ds_read_m32x16_x8<G::kLdsBytesPerStage>(S.b_rb, b_bank[1]);
    wait_all_buffer_data_arrived<true>(2);
    // lgkmcnt(0), not 8: the pre-loop load below overwrites slot 0, so both
    // prologue bursts must be fully drained first.
    wait_lds_data_arrived<false>(0);
    wait_all_warp_arrived();
    {
        constexpr uint32_t kPreSlot =
            (uint32_t)(((G::kPrefetchDist - 1) % G::kNumStages) * G::kLdsBytesPerStage);
        // Runtime, not constexpr: it wraps to 0 when the K extent is exactly the
        // ring's own depth (K == kNumStages * kStageK), same as the reference.
        const int kPre4 = (S.n_stages > (int)G::kPrefetchDist - 1) ? (int)G::kPrefetchDist - 1 : 0;
        const uint32_t so_a =
            (uint32_t)(kPre4 * (int)G::kStageK *
                       (kKind == V7UniformKind::NN ? 2u : (uint32_t)S.a_stride_bytes));
        const uint32_t so_b = (uint32_t)(kPre4 * (int)G::kStageK * S.b_stride_bytes);
        if constexpr(kKind == V7UniformKind::NN)
        {
            buffer_load_dwordx4_idxen_lds_t(S.a_rsrc, S.a_pair, so_a, S.m0_a_base + kPreSlot);
            buffer_load_dwordx4_offen_lds(S.b_rsrc, S.b_vaddr, so_b, S.m0_b_base + kPreSlot);
        }
        else
        {
            buffer_load_dwordx4_offen_lds(S.a_rsrc, S.a_vaddr, so_a, S.m0_a_base + kPreSlot);
            buffer_load_dwordx4_offen_lds(S.b_rsrc, S.b_vaddr, so_b, S.m0_b_base + kPreSlot);
        }
    }

    // The k loop. One divergent branch for the whole kernel (not per step): the
    // exec-mask save/restore is paid once, and each arm is straight-line because
    // kGroupA is a compile-time parameter. The branch must stay DIVERGENT: the
    // uniform-branch variant corrupts on this toolchain (the scheduler reorders
    // the inline-asm waits).
    int kt = 0;
    if(S.wid < (int)G::kWaveGroup)
    {
#pragma unroll 1
        for(bool go = (kt + (int)G::kPhaseCycle <= S.n_stages); go;)
        {
            stage_step<0, true>(S, acc, a_bank, b_bank, kt + 0);
            stage_step<1, true>(S, acc, a_bank, b_bank, kt + 1);
            stage_step<2, true>(S, acc, a_bank, b_bank, kt + 2);
            stage_step<3, true>(S, acc, a_bank, b_bank, kt + 3);
            stage_step<4, true>(S, acc, a_bank, b_bank, kt + 4);
            stage_step<5, true>(S, acc, a_bank, b_bank, kt + 5);
            stage_step<6, true>(S, acc, a_bank, b_bank, kt + 6);
            stage_step<7, true>(S, acc, a_bank, b_bank, kt + 7);
            stage_step<8, true>(S, acc, a_bank, b_bank, kt + 8);
            stage_step<9, true>(S, acc, a_bank, b_bank, kt + 9);
            stage_step<10, true>(S, acc, a_bank, b_bank, kt + 10);
            stage_step<11, true, false>(S, acc, a_bank, b_bank, kt + 11);
            kt += (int)G::kPhaseCycle;
            go = (kt + (int)G::kPhaseCycle <= S.n_stages);
            wait_all_warp_arrived();
        }
        if(kt < S.n_stages) { stage_step<0, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<1, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<2, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<3, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<4, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<5, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<6, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<7, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<8, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<9, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<10, true>(S, acc, a_bank, b_bank, kt); ++kt; }
    }
    else
    {
#pragma unroll 1
        for(bool go = (kt + (int)G::kPhaseCycle <= S.n_stages); go;)
        {
            stage_step<0, false>(S, acc, a_bank, b_bank, kt + 0);
            stage_step<1, false>(S, acc, a_bank, b_bank, kt + 1);
            stage_step<2, false>(S, acc, a_bank, b_bank, kt + 2);
            stage_step<3, false>(S, acc, a_bank, b_bank, kt + 3);
            stage_step<4, false>(S, acc, a_bank, b_bank, kt + 4);
            stage_step<5, false>(S, acc, a_bank, b_bank, kt + 5);
            stage_step<6, false>(S, acc, a_bank, b_bank, kt + 6);
            stage_step<7, false>(S, acc, a_bank, b_bank, kt + 7);
            stage_step<8, false>(S, acc, a_bank, b_bank, kt + 8);
            stage_step<9, false>(S, acc, a_bank, b_bank, kt + 9);
            stage_step<10, false>(S, acc, a_bank, b_bank, kt + 10);
            stage_step<11, false, false>(S, acc, a_bank, b_bank, kt + 11);
            kt += (int)G::kPhaseCycle;
            go = (kt + (int)G::kPhaseCycle <= S.n_stages);
            wait_all_warp_arrived();
        }
        if(kt < S.n_stages) { stage_step<0, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<1, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<2, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<3, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<4, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<5, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<6, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<7, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<8, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<9, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<10, false>(S, acc, a_bank, b_bank, kt); ++kt; }
    }
}

}   // namespace ck_tile

#pragma clang diagnostic pop
