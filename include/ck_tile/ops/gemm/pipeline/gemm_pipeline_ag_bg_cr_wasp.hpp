// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Co., Ltd.
//
// Warp-specialized (producer / consumer) grouped-GEMM pipeline, ported from the
// standalone warp-specialized reference design into the ck_tile pipeline vocabulary.
//
// The structural fact that makes this expressible at all:
//
//     UniversalGemmKernel::kBlockSize == GemmPipeline::BlockSize
//
// The number of waves in the workgroup is the PIPELINE's business, while the
// warp grid that partitions the output tile is the CONFIG's business. So a
// 12-wave workgroup (4 producers + 8 consumers) carries a 1 x 8 wave grid that
// describes the CONSUMERS ONLY: the producer waves are invisible to
// TileGemmShape / TilePartitioner / the epilogue geometry.
//
//     block tile      256M x 256N x 64K
//     wave grid       M_Warp=1, N_Warp=8, warp tile 16x16x16
//                     -> MIterPerWarp=16, NIterPerWarp=2 (32 MMAC, 128 acc VGPR)
//     threads         12 waves x 64 = 768
//     producers       waves 0..3  : stage A, 64 rows each, into a 2-deep ring
//     consumers       waves 4..11 : each owns a 256M x 32N output stripe
//     LDS             64 KB, hand managed: 2 x 32 KB A ring, rounded out by the
//                     epilogue reusing the same buffer once the K loop drains.
//
// -----------------------------------------------------------------------------
// STAGE 1 (this revision): structure only.
//
// The role split, the A ring ownership, the barrier cadence and the fragment /
// accumulator mapping are in place. Deliberately NOT in place yet, each a
// separate measured step:
//
//   1. XOR K swizzle on the A slab. The current layout is plain row-major
//      (row stride 128 B), which is CORRECT but bank-conflicted: 128 B = 32
//      dwords = 0 mod 32 banks, so the 16 rows a wave reads in one ds_read
//      phase all land on the same bank (measured max_bank = 16 vs 8 with
//      the XOR). Expect this to be slow; it is the first thing to fix.
//   2. buffer_load_dwordx4 ... lds (SRD + m0) instead of a plain global load
//      followed by ds_write. Same bytes, fewer VGPRs, no register round trip.
//   3. The C / C' 1-quarter-stage phase stagger between the two consumer half
//      groups, and the peeled k-tail with its LDS zero-fill.
//   4. M / N tail masking: the A rows past M and the B columns past N are read
//      unguarded right now. The reference kernel handles them with a bit-31 vaddr (hardware
//      OOB zero-fill) and a register mask; the selector must therefore not
//      route ragged shapes here until that lands.
// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------
// ISSUE-SLOT BUDGET (this revision): the k loops carried a lot of VALU.
//
// A VALU instruction and an MMAC issued by DIFFERENT waves on the same SIMD
// cannot share a cycle, so every address the producer and consumer loops
// recomputed was an issue slot taken from somebody's MMAC. Both loops are now
// VALU-free by construction: per-lane addresses are built once per TILE, and
// every stage-varying term rides either the uniform SGPR soffset (the producer's
// K column and ring slot, the B K column) or the ds_read's own immediate field
// (the A ring slot and the m-iteration stride).
//
// Measured off the ISA, per k-stage per wave:
//
//   consumer : 31 VALU -> 0    (211 -> 175 instructions, 128 of them MMAC)
//   producer : 42.5 VALU -> 0  (the loop body is 8 DMA loads + ~20 scalar ops)
//
// Wall clock on the ported shape: 245.6 -> 285.0 TFLOPS, VGPRs 254 -> 241, still
// 0 spills. What it took, beyond hoisting the arithmetic, was making the hoisted
// values NON-REMARKABLE to LLVM's rematerialisation heuristic -- it prefers
// rebuilding a cheap loop-invariant address inside the loop over holding it in a
// register. See AProdSrc (8 panel vaddrs) and AReadBases (2 K-half bases).
// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------
// THE 1/4-PHASE STAGGER (the standalone kernel's C / C' personas).
//
// WHAT IT IS. Both consumer bodies execute the same items per k-stage, but C'
// issues them rotated by ONE ITEM: C pairs each read with its MMAC inside one
// interval, C' lets one read cross each barrier and consumes it at the top of
// the next interval. In items (one item = one 8-unit m-chunk = 32 MMACs):
//
//   C  I1 = [R c0 M c0  R c1 M c1]      C  I2 = [R c2 M c2  R c3 M c3]
//   C' I1 = [M c3(prev) R c0 M c0 R c1] C' I2 = [M c1 R c2 M c2 R c3]
//
// with a k-stage split into two intervals by BAR(k).mid, and c0/c1 the h0 chunks,
// c2/c3 the h1 chunks. C leads an interval with its ds_reads (LDS busy) while C'
// leads with its MMACs (matrix busy), and they swap at the tail. This is the
// same shape as the reference kernel's stage_body_C / stage_body_Cp; the
// granularity is one m-chunk instead of one K-half because the fragment has to
// fit (see below).
//
// MEASURED, this shape, two cadences:
//
//      cadence         body     split    TFLOPS  spill/scratch
//   (a) 1 work interval  C        -       286.4   0 / 0
//   (a) 1 work interval  C'       0       299.4   5 / 24 B
//   (a) 1 work interval  C/C'    4+4      188.1  16 / 52 B
//   (b) 2 intervals      C        -       247.7  16 / 52 B
//   (b) 2 intervals      C'       0       264.8  16 / 52 B
//   (b) 2 intervals      C/C'    4+4      172.0  16 / 52 B
//
// (b) is hipc's cadence and is what the rotation needs -- without the mid
// barrier there is no interval to rotate within. Read across the rows: the C'
// ORDER is worth +4..7% in both cadences, the mid-stage rendezvous costs 12%,
// and the rotation itself costs another 35% on top. The rotation is the only
// configuration that loses; every uniform body is better than any mix, and the
// loss is not codegen (the spill is 16 / 52 B in all three (b) rows).
//
// The mechanism is the rendezvous: a barrier releases when the SLOWEST wave
// arrives, and the two personas have different interval costs (C drains in
// place twice per interval, C' gets one cheap drain and one free one), so the
// workgroup runs at C's pace while C' idles -- and the LDS queue interleaving of
// two out-of-phase read streams makes both of them slower than either uniform
// stream. There is no phase offset to be had once a hard sync re-times every
// wave twice a stage.
//
// THE REGISTER SIDE of the same experiment: the reference kernel keeps a whole
// K-half fragment (64 VGPR) live across the barrier. Here that lands on the
// 256-VGPR cap with 260 registers spilled / 372 B/lane of scratch and measures
// 263.9 against 299.3 for the 8-unit chunk (spill 5 / 24 B). The reference
// kernel itself compiles at VGPR 256 / SGPR 104 / scratch 108 B/lane (read out
// of its own code object) -- it spills too, a quarter as much, because its body
// carries no framework state beyond the params struct. Its occupancy hint
// (amdgpu_waves_per_eu(1,1), mirrored here as CK_TILE_WAVES_PER_EU_HINT in
// host/kernel_launch.hpp) was tried and is NEUTRAL: the cap stays 256 and the
// fragment still spills. Chunking the fragment is what buys the difference.
// -----------------------------------------------------------------------------
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"

namespace ck_tile {

// ---------------------------------------------------------------------------
// Accumulator tile (16
// m-iterations x 2 n-iterations x 4 fp32 = 128 VGPR per lane.
// ---------------------------------------------------------------------------
struct WaspAccTile
{
    static constexpr index_t kMIters = 16;   // 256 M / 16
    static constexpr index_t kNIters = 2;    // 32 N  / 16

    // One 4-dword quad per (m,n) slot, as a VECTOR: the backend keeps the quad
    // as one value (the reference kernel's union_vec4_fp32 acc_o[]), instead of
    // promoting it into four independent scalars. That granularity is what the
    // register allocator works with around every MMAC.
    using acc_vec_t = float __attribute__((ext_vector_type(4)));
    acc_vec_t v[kMIters * kNIters];
};

namespace wasp_detail {

constexpr index_t kVmemSampleStage = 16;

struct WaspVmemSample
{
    uint64_t issue_to_wait_start = 0;
    uint64_t wait_cycles         = 0;
    uint64_t vmcnt4_cycles       = 0;
    uint64_t vmcnt2_cycles       = 0;
    uint32_t valid               = 0;
};

struct WaspVmemDiag
{
    uint64_t issued_at = 0;
    WaspVmemSample sample{};
    bool issued = false;
    bool valid  = false;
};

#if defined(CK_TILE_GROUPED_GEMM_WASP_VMEM_WAIT_COUNT)
__device__ static WaspVmemSample g_wasp_vmem_samples[1024 * 16];
#endif

CK_TILE_DEVICE uint64_t wasp_memtime()
{
    return static_cast<uint64_t>(__builtin_amdgcn_s_memtime());
}

using frag_a_t = unsigned short __attribute__((ext_vector_type(4)));   // 4 bf16 = 8 B
using frag_c_t = float __attribute__((ext_vector_type(4)));            // 4 fp32
using dword4_t = int32_t __attribute__((ext_vector_type(4)));          // 16 B

// 16 B staging unit: aliased as one dword4 or as two 8 B MMAC operand halves.
union alignas(16) stage16_t
{
    dword4_t  d4;
    frag_a_t  frag[2];
};
static_assert(sizeof(stage16_t) == 16, "16 B staging unit");

// NOTE: a plain `ds_read_b128(byte_addr)` helper used to live here, taking the
// address as a "v" operand. It is deliberately gone: that form makes the
// compiler materialise a fresh per-lane address for every ds_read, i.e. 2 v_add
// per m-iteration, 32 VALU per k-stage per consumer wave -- and on this part a
// VALU and an MMAC from different waves on the same SIMD cannot issue in the
// same cycle. Use the pipeline's a_read_half (one base VGPR + immediates).

#if defined(CK_TILE_GROUPED_GEMM_WASP_ABLATION)
// ---------------------------------------------------------------------------
// Ablation switch, set from the host with hipMemcpyToSymbol. Results are WRONG
// by construction -- the only thing being measured is the timing decomposition,
// because three separate hypotheses about where this kernel's time goes were
// each falsified by measurement and modelling is clearly not working.
//
// bit 0 : skip the B global loads
// bit 1 : skip the MMACs
// bit 2 : read every A fragment from row 0's address (same LDS traffic count,
//         degenerate address pattern -- isolates "16 distinct addresses" from
//         "the reads themselves")
// bit 3 : skip the workgroup barriers
// ---------------------------------------------------------------------------
__device__ static int g_wasp_ablate = 0;
#endif

CK_TILE_DEVICE int wasp_ablate()
{
#if defined(CK_TILE_GROUPED_GEMM_WASP_ABLATION)
    return g_wasp_ablate;
#else
    return 0;
#endif
}

CK_TILE_DEVICE void ds_write_b128(uint32_t byte_addr, const dword4_t& v)
{
    asm volatile("ds_write_b128 %0, %1" ::"v"(byte_addr), "v"(v) : "memory");
}

CK_TILE_DEVICE frag_c_t mmac_bf16(const frag_a_t& a,
                                                  const frag_a_t& b,
                                                  const frag_c_t& c)
{
    return __builtin_hcu_mmac_f32_16x16x16_bf16(a, b, c);
}

// Raw global 16 B read. STAGE 1 uses this instead of buffer_load ... lds; the
// bytes fetched are identical, only the path to LDS differs.
CK_TILE_DEVICE dword4_t global_load_b128(const void* p)
{
    return *reinterpret_cast<const dword4_t*>(p);
}

// ---------------------------------------------------------------------------
// STAGE 2: buffer_load_dwordx4 Direct-to-LDS.
//
// Ported from the reference producer (buffer_load_dwordx4_x8_to_lds_vaddr):
// each `buffer_load_dwordx4 vaddr, srsrc, soffset offen lds` streams one PANEL
// -- 64 lanes x 16 B = 1024 B -- straight from global into LDS at m0 + lane*16,
// with m0 advanced by 1024 between the 8 panels. This removes the VGPR round
// trip the STAGE-1 producer paid (8x global_load into 32 VGPR, then 8x
// ds_write): the memory system writes LDS directly, so the producer no longer
// holds the whole slab live in registers and the global->LDS hand-off is one
// instruction stream instead of two with a barrier-ordered clobber between.
//
// gfx936 / gfx938 only (the `lds` modifier and the CACHE_SWIZZLE descriptor
// bit). The STAGE-1 load+ds_write path is kept as the portable fallback.
// ---------------------------------------------------------------------------
using buffer_rsrc_t = int32_t __attribute__((ext_vector_type(4)));

// V4 buffer resource descriptor. STRIDE (word1 bits [29:16]) is programmed for
// the CACHE_SWIZZLE rotation period only -- `offen` does NOT scale the index by
// it. CACHE_SWIZZLE (bit 30) exists on gfx936/938 and needs stride/64 to be a
// power of two, which K=4096 (pitch 8192 = 64 x 128) and K=2048 (4096 = 64 x
// 64) both satisfy; it is left off otherwise.
CK_TILE_DEVICE buffer_rsrc_t wasp_make_buffer_rsrc(const void* ptr,
                                                   uint32_t stride_bytes,
                                                   uint32_t size_bytes,
                                                   bool allow_swizzle = true)
{
    buffer_rsrc_t r;
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    r[0]                = static_cast<int32_t>(addr & 0xffffffffu);
    r[1]                = static_cast<int32_t>(addr >> 32);
    r[2]                = static_cast<int32_t>(size_bytes);
    r[3]                = 0x00020000;

    uint32_t val = (stride_bytes & 0x3fffu) << 16;
#if defined(__gfx936__) || defined(__gfx938__)
    if(allow_swizzle && (stride_bytes % 64u) == 0u)
    {
        const uint32_t s64 = stride_bytes / 64u;
        if((s64 & (s64 - 1u)) == 0u)
        {
            val |= 1u << 30;
        }
    }
#else
    (void)allow_swizzle;
#endif
    r[1] |= static_cast<int32_t>(val);
    return r;
}

// 8 panels, each a dwordx4 (16 B) per lane, Direct-to-LDS. `lds_m0` is the LDS
// BYTE address of the first panel (p_smem is at an LDS offset whose numeric
// pointer value IS that address); `soffset` is the stage's K column byte offset
// shared by all 8 loads; each `v_off[i]` is this lane's per-panel byte offset
// (row * pitch + swizzled K within the stage), with bit 31 set for an
// out-of-range row so the hardware zero-fills it.
CK_TILE_DEVICE void wasp_buffer_load_x8_to_lds(buffer_rsrc_t rsrc,
                                               uint32_t lds_m0,
                                               uint32_t soffset,
                                               uint32_t v0,
                                               uint32_t v1,
                                               uint32_t v2,
                                               uint32_t v3,
                                               uint32_t v4,
                                               uint32_t v5,
                                               uint32_t v6,
                                               uint32_t v7)
{
    asm volatile(
        "s_mov_b32 m0, %1\n\t"
        "buffer_load_dwordx4 %0, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %2, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %3, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %4, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %5, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %6, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %7, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %8, %9, %10 offen lds\n"
        :
        : "v"(v0), "s"(lds_m0), "v"(v1), "v"(v2), "v"(v3), "v"(v4), "v"(v5), "v"(v6),
          "v"(v7), "s"(rsrc), "s"(soffset)
        : "memory");
}

// One 16 B consumer-B read through the CACHE_SWIZZLE descriptor (gfx936/938).
//
// `pair` packs {index in the LOW dword, byte offset in the HIGH dword}; with
// `idxen offen` the hardware forms addr = base + index * STRIDE + offset +
// soffset. STRIDE (the descriptor's row pitch, programmed in
// wasp_make_buffer_rsrc) also sets the CACHE_SWIZZLE VL1 tag-bank rotation
// period, which is the whole point: the plain pointer dereference the STAGE-1/2
// consumer used suffered a large VL1-Tag RAM stall on the B stream (every lane
// of a wave hitting the same tag bank). The index is the global B row (output
// column), the offset the in-row K byte (col_group*16 + k_half*64), and the
// stage's K column base rides the uniform SGPR `soffset`.
// FOUR buffer_load_dwordx4 in ONE asm block -- the reference kernel's B load
// form verbatim (buffer_load_dwordx4_x4_idxen_soff2 in bf16_gemm_nt_hipc.cpp).
//
// What matters about the shape:
//   * two IDXEN index pairs only (one per 16-wide n-row), NOT one per load;
//   * the K-half difference rides the instruction's IMMEDIATE field (+64 B =
//     kPassK * 2 B), so it costs no register at all;
//   * the stage's K column rides the uniform SGPR soffset;
//   * the per-lane in-row byte offset rides the top half of each pair.
// The port used to issue four separate loads, each with its own 64-bit pair
// carrying the k-half inside the offset: same bytes, 4x the index registers and
// 4 asm blocks instead of 1.
CK_TILE_DEVICE void wasp_buffer_load_b128_x4_idxen_soff2(buffer_rsrc_t rsrc,
                                                         uint64_t p0,
                                                         uint64_t p1,
                                                         uint32_t s_col_base,
                                                         dword4_t& o0,
                                                         dword4_t& o1,
                                                         dword4_t& o2,
                                                         dword4_t& o3)
{
    asm volatile("buffer_load_dwordx4 %0, %4, %6, %7 idxen offen offset:0\n\t"
                 "buffer_load_dwordx4 %1, %4, %6, %7 idxen offen offset:64\n\t"
                 "buffer_load_dwordx4 %2, %5, %6, %7 idxen offen offset:0\n\t"
                 "buffer_load_dwordx4 %3, %5, %6, %7 idxen offen offset:64\n"
                 : "=&v"(o0), "=&v"(o1), "=&v"(o2), "=&v"(o3)
                 : "v"(p0), "v"(p1), "s"(rsrc), "s"(s_col_base)
                 : "memory");
}

CK_TILE_DEVICE dword4_t wasp_buffer_load_b128_idxen(buffer_rsrc_t rsrc,
                                                    uint64_t pair,
                                                    uint32_t soffset)
{
    dword4_t o;
    asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 idxen offen\n"
                 : "=v"(o)
                 : "v"(pair), "s"(rsrc), "s"(soffset)
                 : "memory");
    return o;
}

// Workgroup barrier that actually publishes LDS.
//
// ck_tile::block_sync_lds() is NOT usable here: in its default configuration
// (CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM undefined) it expands
// to __syncthreads(), i.e. a bare s_barrier -- see core/arch/arch.hpp:121-134.
// s_barrier synchronizes EXECUTION, it does not drain posted LDS stores, so a
// producer's ds_write_b128 can still be in flight when a consumer's
// ds_read_b128 executes. The result is a race that is invisible to a single
// wave, intermittent across runs, and biased towards whichever lanes win the
// dispatch -- exactly the "same shape, different bad count" signature this port
// showed before the fix. LDS stores must be retired with s_waitcnt lgkmcnt(0)
// on the writer side BEFORE the barrier.
// Bare workgroup barrier -- the reference kernel's wait_all_warp_arrived()
// verbatim: sched_barrier(0), s_barrier, sched_barrier(0).
//
// NO LDS DRAIN. This is load bearing for the k-stage cadence: the C' persona
// issues its A(k,1) read at the very end of an interval and consumes it on the
// far side of the barrier. A draining barrier (s_waitcnt lgkmcnt(0) + s_barrier)
// forces that read to complete before ANY wave may pass, which is exactly the
// serialisation the 1/4-phase offset exists to remove -- and it also puts every
// wave's LDS latency on the critical path to the barrier. Each consumer waits
// for its own A data explicitly (the waits inside Q1/Q3), so nothing is lost by
// not draining here.
//
// The sched_barrier pair is the reference kernel's, not decoration: it stops the
// machine scheduler from hoisting a read across the barrier (which would break
// the deferral's ordering) or sinking an MMAC past it.
CK_TILE_DEVICE void wasp_barrier()
{
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_barrier\n");
    __builtin_amdgcn_sched_barrier(0);
}

CK_TILE_DEVICE void wasp_barrier_maybe()
{
    if((wasp_ablate() & 8) == 0)
    {
        wasp_barrier();
    }
}

CK_TILE_DEVICE void wasp_sync_lds()
{
    // Only the LGKM counter is drained here, matching the reference kernel.
    //
    // An earlier revision also drained vmcnt(0) on the theory that the A stage
    // (written with ds_write, an LDS-unit store) might be ordered by the global
    // counter. The ablation disproves the cost model behind that: the barriers
    // are ~20% of the runtime, and vmcnt(0) makes every wave -- including the 8
    // consumers, whose global loads are its own B operands and have nothing to
    // do with publishing A -- stall on its own outstanding global traffic at
    // each of the num_loop barriers. The producer's A writes are LDS stores and
    // are covered by lgkmcnt.
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
}

// Barrier with the ablation bit honored (see wasp_ablate). Uniform branch, so
// it costs one scalar compare when the ablation is off.
CK_TILE_DEVICE void wasp_sync_lds_maybe()
{
    if((wasp_ablate() & 8) == 0)
    {
        wasp_sync_lds();
    }
}

// Retire outstanding direct-to-LDS DMAs (vmcnt). The producer calls this before
// the barrier that publishes a stage, so the overlapping cadence can issue the
// next stage's DMA straight after a publish and let it run under the consumers'
// compute. All of a stage's 8 panels are in flight by the time this is reached,
// so it is one 8-wide MLP wait, not a per-panel serialization.
// s_setprio N (0..3): the CU's wave scheduler prefers a higher-priority wave
// when it picks which one to issue from next cycle; the hardware default is 1.
// The standalone kernel brackets every 64-MMAC chain with <3> ... <0>: a
// consumer held at priority 3 while its MMACs drain keeps the matrix pipe fed,
// and dropping back to 0 afterwards lets the producers' DMA and the consumers'
// own ds_reads through. The sched_barrier(0) pair pins the priority change in
// program order -- same form as the reference kernel's flash::set_prio.
template <int PRIO>
CK_TILE_DEVICE void wasp_set_prio()
{
    static_assert(PRIO >= 0 && PRIO <= 3, "s_setprio operand must be 0..3");
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio %0" ::"n"(PRIO));
    __builtin_amdgcn_sched_barrier(0);
}

CK_TILE_DEVICE void wasp_wait_vmcnt0()
{
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
}

CK_TILE_DEVICE void wasp_wait_vmcnt0(WaspVmemDiag& diag, index_t stage)
{
#if defined(CK_TILE_GROUPED_GEMM_WASP_VMEM_WAIT_COUNT)
    if(stage == kVmemSampleStage && diag.issued)
    {
        const uint64_t wait_start = wasp_memtime();
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        const uint64_t wait_end = wasp_memtime();
        diag.sample.issue_to_wait_start = wait_start - diag.issued_at;
        diag.sample.wait_cycles         = wait_end - wait_start;
        diag.sample.valid       = 1;
        diag.valid                      = true;
        return;
    }
#else
    (void)diag;
    (void)stage;
#endif
    wasp_wait_vmcnt0();
}

CK_TILE_DEVICE void wasp_wait_vmcnt0_producer(WaspVmemDiag& diag, index_t stage)
{
#if defined(CK_TILE_GROUPED_GEMM_WASP_VMEM_WAIT_COUNT)
    if(stage == kVmemSampleStage && diag.issued)
    {
        const uint64_t wait_start = wasp_memtime();
        asm volatile("s_waitcnt vmcnt(4)\n" ::: "memory");
        diag.sample.vmcnt4_cycles = wasp_memtime() - wait_start;
        asm volatile("s_waitcnt vmcnt(2)\n" ::: "memory");
        diag.sample.vmcnt2_cycles = wasp_memtime() - wait_start;
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        const uint64_t wait_end = wasp_memtime();
        diag.sample.issue_to_wait_start = wait_start - diag.issued_at;
        diag.sample.wait_cycles         = wait_end - wait_start;
        diag.sample.valid               = 1;
        diag.valid                      = true;
        return;
    }
#else
    (void)diag;
    (void)stage;
#endif
    wasp_wait_vmcnt0();
}

#if defined(CK_TILE_GROUPED_GEMM_WASP_DUMP_A_STAGE)
// ---------------------------------------------------------------------------
// DIAGNOSTIC: two snapshots of the raw A staging ring, taken by producer wave 0
// right after the barrier that publishes each of the first two stages.
//
// Comparing them on the host against the expected [256 rows x 64 k] layout
// settles the one question runtime evidence cannot: is the A operand wrong
// because the STAGE was never written correctly, or because the CONSUMER reads
// a correctly written stage wrongly?
//
// Raw 16-bit patterns, decoded on the host, so nothing is lost in translation.
// The snapshot has to happen HERE and not later: the epilogue borrows the whole
// 64 KB for its output staging and would overwrite both ring slots.
// ---------------------------------------------------------------------------
__device__ static unsigned short g_wasp_a_dump[2 * 256 * 64];

// One wave cooperatively copies a whole 32 KB stage: 64 lanes x 256 bf16 each.
//
// Deliberately NOT gated by a "first caller wins" flag: the copy is cooperative,
// so a flag that stops 63 of the 64 lanes would leave their regions of the dump
// at their zero-initialised value and read as an empty stage. Every wave-0 visit
// simply overwrites the previous snapshot, which is harmless because a correctly
// staged tile looks the same every time.
CK_TILE_DEVICE void take_a_stage_snapshot(int slot, const char* lds)
{
    const index_t         lane = ck_tile::get_lane_id();
    const unsigned short* src =
        reinterpret_cast<const unsigned short*>(lds + slot * 32768 + lane * 512);
    for(index_t i = 0; i < 256; ++i)
    {
        g_wasp_a_dump[slot * 256 * 64 + lane * 256 + i] = src[i];
    }
}
#endif

} // namespace wasp_detail

template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy>
struct GemmPipelineAgBgCrWasp : public GemmPipelineAgBgCrCompV3<Problem, Policy>
{
    using Base = GemmPipelineAgBgCrCompV3<Problem, Policy>;

    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;

    // ------------------------------------------------------------------
    // Wave split. The consumer count must equal the configured warp grid,
    // because those are the waves that own output.
    // ------------------------------------------------------------------
    static constexpr index_t kNumProducerWaves = 4;
    static constexpr index_t kNumConsumerWaves = 8;
    // The 1/4-phase stagger: consumer waves 0..3 walk the four quarters in the C
    // order, waves 4..7 in the C' order (the same four quarters rotated by one
    // quarter), so the two halves lead an interval with different pipes -- the
    // reference kernel's split verbatim (is_C_prime = consumer_id >= 4).
    //
    // MEASURED on this port, m=8152, with the k-stage structure exact: the C'
    // ORDER run uniformly is worth +4..7%; the ROTATION itself is a net loss,
    // because a barrier releases on the slowest wave and re-times the offset
    // twice a stage. Both personas are kept because the rotation is the point.
    static constexpr index_t kConsumerSplitWave = kNumConsumerWaves / 2;

    static constexpr index_t kNumWavesTotal    = kNumProducerWaves + kNumConsumerWaves;

    // SHADOWS the base's `Problem::kBlockSize` (== NumWarps * warp size == 512
    // for this geometry). UniversalGemmKernel reads the pipeline member, so the
    // launched workgroup is 768 threads while the tile grid stays 1 x 8.
    static constexpr index_t BlockSize = kNumWavesTotal * get_warp_size();

    static_assert(BlockGemmShape::NumWarps == kNumConsumerWaves,
                  "WASP: the tile warp grid must be exactly the consumer waves");
    static_assert(BlockSize == 768, "WASP workgroup is 12 waves of 64");

    // ------------------------------------------------------------------
    // LDS budget, hand managed (does not follow the ComputeV3 policy).
    //   A stage: 4 producer slabs x 64 rows x 64 K x 2 B = 32 KB
    //   2 stages                                        = 64 KB
    // ------------------------------------------------------------------
    static constexpr index_t kStageK        = 64;
    static constexpr index_t kPassK         = 32;
    static constexpr index_t kNumStages     = 2;
    static constexpr index_t kProdSlabRows  = 64;
    // Columns owned by one consumer wave. Every consumer reads/writes its OWN
    // stripe, so this offset has to be applied on both the B operand and the
    // store side -- forgetting it makes every consumer but the first compute the
    // wrong columns.
    static constexpr index_t kWaveN = BlockGemmShape::kN / kNumConsumerWaves;   // 256/8 = 32
    static constexpr index_t kProdSlabBytes = kProdSlabRows * kStageK * sizeof(uint16_t); // 8192
    static constexpr index_t kAStageBytes   = kNumProducerWaves * kProdSlabBytes;         // 32768
    static constexpr index_t kWaspLdsBytes  = kNumStages * kAStageBytes;                  // 65536

    static_assert(kProdSlabBytes * kNumProducerWaves == kAStageBytes, "A stage budget");
    static_assert(BlockGemmShape::kK == kStageK, "block K must equal the A-stage depth");
    static_assert(BlockGemmShape::kM == kNumProducerWaves * kProdSlabRows,
                  "A rows are split evenly across the producer waves");
    static_assert(BlockGemmShape::kN == kNumConsumerWaves * 32,
                  "each consumer wave owns a 32 wide N stripe");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kWaspLdsBytes; }

    using AccTile = WaspAccTile;

    // Marker for GroupedGemmKernel's compile-time dispatch (is_grouped_gemm_wasp
    // _pipeline): routes this pipeline through RunGemmWasp, the raw-pointer path
    // that feeds run_raw directly and keeps no tile-window state live across the
    // k loop. Absent on every other pipeline, so they keep the generic path.
    static constexpr bool IsWaspPipeline = true;

    // ------------------------------------------------------------------
    // Producer: waves 0..3, one 64-row slab each.
    //
    // STAGE 1 addressing (plain, see the header note):
    //   global  row r of the tile, k element      -> a_base + r*lda + k
    //   LDS     slab p, local row r, 16 B chunk c -> slab + r*128 + c*16
    //   lane l    owns row_group = l>>3, chunk = l&7   (no XOR yet)
    // ------------------------------------------------------------------
    // Producer A addressing: what is per-TILE and what is per-K-STAGE.
    //
    // The 8 panel vaddrs are `row * pitch + (chunk ^ row_group)*16` with
    // bit 31 set for a row past m_valid. Every term in that expression is
    // per-tile and per-lane only -- the k-stage does not appear anywhere in it
    // (the stage's K column rides the uniform SGPR `soffset`, and the LDS ring
    // slot rides `m0`). Rebuilding them inside the k loop therefore costs
    // 8 v_add + 8 v_cmp + 8 v_cndmask + 11 v_or per stage per producer wave,
    // measured off the ISA: 42.5 VALU per stage, against 8 useful DMA loads.
    //
    // That is the wrong trade on this part: an MMAC and a VALU instruction from
    // different waves on the SAME SIMD cannot issue in the same cycle, so 4
    // producer waves x ~40 VALU ride on the issue slots the 8 consumer waves'
    // MMACs want. (The producer's own register budget is tiny -- 9 VGPRs -- so
    // hoisting the values costs nothing there either.) Same idea as the
    // standalone kernel, which builds its vaddr array once before the loop and
    // then only bumps m0 and soffset, both scalar.
    // ------------------------------------------------------------------
    struct AProdSrc
    {
        wasp_detail::buffer_rsrc_t rsrc;      // SGPR pair set, built once
        uint32_t                   v_off[8];  // per-lane panel byte offsets
    };

    // a_base and lda are per-TILE, not per k-stage, so they are passed in rather
    // than re-derived here. The first version called calculate_offset on every
    // one of the num_loop calls -- 64 times per tile per wave -- and the ablation
    // says this kernel's time is dominated by exactly that kind of scalar
    // bookkeeping, not by the MMACs (removing every MMAC saves ~10%).
    template <typename ADataTypePtr>
    CK_TILE_DEVICE AProdSrc make_a_prod_src(const ADataTypePtr* a_base,
                                            index_t lda,
                                            index_t m_valid,
                                            index_t wid) const
    {
        using namespace wasp_detail;

        const index_t lane = get_lane_id();

        const index_t row_group = lane >> 3;   // 0..7
        const index_t chunk     = lane & 7;    // 0..7, 16 B units along K
#if defined(CK_TILE_GROUPED_GEMM_WASP_REVERSE_PRODUCER_SLAB)
        // Debug switch: hand the slabs out in reverse wave order. If the broken
        // rows follow producer wave 3 the fault is in that wave's execution; if
        // they stay at the same LDS addresses the fault is address dependent.
        const index_t slab_id = kNumProducerWaves - 1 - wid;
#else
        const index_t slab_id = wid;
#endif
        const index_t row_base = slab_id * kProdSlabRows;

        // One buffer descriptor over this tile's A. The size field bounds valid
        // rows to [0, m_valid): a row >= m_valid overruns num_records and reads
        // as 0, which (with the bit-31 mask below) is the zero-fill the
        // epilogue's row drop already assumes.
        const uint32_t pitch_bytes = static_cast<uint32_t>(lda) * sizeof(uint16_t);
        const uint32_t size_bytes =
            static_cast<uint32_t>((m_valid > 0 ? m_valid : 0)) * pitch_bytes;

        AProdSrc s;
        s.rsrc = wasp_make_buffer_rsrc(a_base, pitch_bytes, size_bytes);

        const uint32_t k_in_panel =
            static_cast<uint32_t>((chunk ^ row_group) * 16);   // swizzled K byte offset
#pragma unroll
        for(index_t i = 0; i < 8; ++i)
        {
            const index_t  row = row_base + i * 8 + row_group;
            const uint32_t v   = static_cast<uint32_t>(row) * pitch_bytes + k_in_panel;
            s.v_off[i]         = (row < m_valid) ? v : (v | 0x80000000u);
        }
        // ... and they have to SURVIVE the k loop rather than be rebuilt in it.
        // Left alone, LLVM's rematerialisation heuristic decides that 5 ALU ops
        // per panel per iteration are cheaper than 8 live VGPRs -- which is
        // exactly backwards for this kernel: the producer's own budget is ~20
        // VGPRs, and every one of those ALU ops is a VALU instruction competing
        // with a consumer wave's MMAC for the same SIMD issue slot. Measured on
        // the ISA before this: the loop body carried 11 v_add + 11 v_or + 8
        // v_cmp + 8 v_cndmask per stage, against 8 useful DMA loads. An asm
        // boundary makes the values non-rematerialisable, so they are computed
        // once per tile and then only read.
#pragma unroll
        for(index_t i = 0; i < 8; ++i)
        {
            asm volatile("" : "+v"(s.v_off[i]));
        }
        return s;
    }

    // One k-stage's DMA issue: 8 panels, 16 B per lane each, straight to LDS.
    // LDS m0 is a segment-relative byte offset, so its base is zero; passing
    // the shared-memory pointer here would force a VGPR-to-SGPR readfirstlane.
    // The only per-stage values are `soffset` (the stage's K column, uniform)
    // and `m0` (the ring slot plus producer slab, uniform).
    CK_TILE_DEVICE void producer_fill(const AProdSrc& src, index_t k_stage, index_t wid) const
    {
        using namespace wasp_detail;

        if((wasp_ablate() & 64) != 0)
        {
            return;   // ablation: idle producer (see wasp_ablate)
        }

        const index_t slot = (k_stage % kNumStages) * kAStageBytes;

        const uint32_t soffset = static_cast<uint32_t>(k_stage * kStageK) * sizeof(uint16_t);
        const uint32_t m0 =
            static_cast<uint32_t>(slot + wid * kProdSlabBytes);

        wasp_buffer_load_x8_to_lds(src.rsrc,
                                   m0,
                                   soffset,
                                   src.v_off[0],
                                   src.v_off[1],
                                   src.v_off[2],
                                   src.v_off[3],
                                   src.v_off[4],
                                   src.v_off[5],
                                   src.v_off[6],
                                   src.v_off[7]);
        // NOTE: the DMA is only ISSUED here, not retired. Direct-to-LDS
        // completion is on vmcnt, and the overlapping cadence in operator()
        // drains it (wasp_wait_vmcnt0) right before the publishing barrier -- so
        // that the NEXT stage's DMA, issued straight after a publish, overlaps
        // the consumers' compute of the stage just published.
    }

    // ------------------------------------------------------------------
    // Consumer: waves 4..11, one 256M x 32N stripe each.
    //
    // MMAC fragment contract (measured on gfx936, see the standalone reference kernel header):
    //   A  lane l holds rows m = l&15 within its 16-row sub-tile and
    //      k = 4*(l>>4) + 0..3 of each K16 operand
    //   C  accumulator slot c of lane group g holds column 4*c + g
    // The 16 B LDS unit for lane group g, K-half kh holds global
    // k = 8*(g + 4*kh) + 0..8, i.e. plain-layout offset (g + 4*kh)*16.
    // ------------------------------------------------------------------
    // One B buffer: [k_half][n_tile].
    using BBuf = wasp_detail::stage16_t[2][WaspAccTile::kNIters];

    // Prefetch a whole stage's B (2 K-halves x 2 n) into one buffer through the
    // CACHE_SWIZZLE descriptor. In-row K byte of unit (kh,n) is 16*col_group +
    // 64*kh; the stage K column base rides the SGPR soffset.
    // The per-lane addressing values the k-stage consumes, computed ONCE per
    // item -- the reference kernel's shape (a_base_h0 / a_base_h1 / pair_n0 /
    // pair_n1, bf16_gemm_nt_hipc.cpp:1272-1373) -- and handed to the interval
    // bodies. The k loop then carries them in registers and does NO addressing
    // arithmetic: no VALU in the quarters, the base/pairs are read where the
    // ds_read / buffer_load_dwordx4 need them.
    //
    // Per-lane because both are: the A read base is
    //     lds + row_in_tile*128 + ((col_group ^ (row_in_tile & 7)) * 16)
    // and the B IDXEN pair is {index = global B row, offset = in-row bytes}.
    struct AReadBases
    {
        uint32_t h0;
        uint32_t h1;
    };

    struct BPairs
    {
        uint64_t p0;
        uint64_t p1;
    };

    // The B prefetch: four buffer_load_dwordx4 in ONE asm block, the reference
    // kernel's form -- two IDXEN pairs, the K-half riding the instruction's
    // immediate (+0 / +64 B), the stage's K column riding the uniform SGPR
    // soffset. The pairs are the per-item values built in operator().
    CK_TILE_DEVICE void load_B_stage(wasp_detail::buffer_rsrc_t b_rsrc,
                                     const BPairs& bp,
                                     index_t k_stage,
                                     BBuf& buf,
                                     wasp_detail::WaspVmemDiag& vmem_diag) const
    {
        using namespace wasp_detail;
        const uint32_t s_col_base = static_cast<uint32_t>(k_stage * kStageK) * sizeof(uint16_t);
        if((wasp_ablate() & 1) != 0)
        {
            buf[0][0].d4 = 0;
            buf[1][0].d4 = 0;
            buf[0][1].d4 = 0;
            buf[1][1].d4 = 0;
            return;
        }
        // Destination order is the reference kernel's: (n0,kh0) (n0,kh1)
        // (n1,kh0) (n1,kh1) -- buf[kh][n], the immediate 0/64 choosing the half.
        wasp_buffer_load_b128_x4_idxen_soff2(b_rsrc, bp.p0, bp.p1, s_col_base,
                                             buf[0][0].d4,   // n0, kh0 (offset:0)
                                             buf[1][0].d4,   // n0, kh1 (offset:64)
                                             buf[0][1].d4,   // n1, kh0 (offset:0)
                                             buf[1][1].d4);  // n1, kh1 (offset:64)
#if defined(CK_TILE_GROUPED_GEMM_WASP_VMEM_WAIT_COUNT)
        if(k_stage == wasp_detail::kVmemSampleStage)
        {
            vmem_diag.issued_at = wasp_detail::wasp_memtime();
            vmem_diag.issued = true;
        }
#else
        (void)vmem_diag;
#endif
    }

    // The 16 B bytes one lane reads per (n-tile, K-half) live at global K =
    // 8*col + 32*khalf, i.e. col*16 B into the row; the K-half rides the
    // instruction immediate (+0 / +64 B) and the stage's K column rides the
    // uniform SGPR soffset -- the reference kernel's form.
    static constexpr uint32_t kBHalfBytes = 64;

    // ------------------------------------------------------------------
    // The A fragment is ONE WHOLE K-HALF: 16 x 16 B per lane = 64 VGPR, read with
    // ONE base register plus compile-time immediates, and consumed by ONE
    // 64-MMAC burst. This is the standalone kernel's shape AND grouping
    // (ds_read_a_half + mmac_half), and the k-stage body around it is unchanged:
    // two intervals per stage, one 16-unit read burst and one 64-MMAC burst in
    // each, in opposite order for the two consumer personas. See c_I1/c_I2 (C)
    // and cp_I1/cp_I2 (C').
    // ------------------------------------------------------------------

    // 16 consecutive ds_read_b128 off ONE base VGPR, with the m-iteration row
    // offsets (16 rows x 128 B = 0x800) AND the 2-deep A ring slot as
    // compile-time immediates. 32768 + 30720 = 63488 fits the 16-bit unsigned
    // offset field.
    //
    // The immediates are load bearing. The natural form -- a generic
    // ds_read_b128(byte_addr) helper taking the address as a "v" operand -- makes
    // the compiler materialise a fresh per-lane address for every ds_read, i.e.
    // 2 v_add per m-iteration, 32 VALU per k-stage per consumer wave, and those
    // ride on the same SIMD issue slots as the other waves' MMACs.
    // (row & 7) -- the XOR swizzle selector -- is invariant under +16 rows, so
    // the same base serves all 16 reads of a half.
    //
    // NOT SPLIT INTO 16 SINGLE-OUTPUT BLOCKS, and that was measured: the
    // per-statement "memory" clobber serialises the issue and it is 21% SLOWER
    // (171.7 -> 135.7 TFLOPS) with the SAME spill, so it buys the allocator
    // nothing either.
    // `token` is an UNUSED input operand (the accumulator element the previous
    // MMAC chain wrote). It exists only to create a data edge: the MMACs are
    // builtins, so nothing else orders them against these ds_reads, and without
    // the edge the reads can be issued first -- which makes the old fragment and
    // the new one overlap, and the allocator answers with a 64-register copy of
    // the old fragment. That copy is what pushes the window over the 256-VGPR
    // cap and gets the per-item A bases / B pairs spilled into `buffer_load_dword`
    // reloads inside the quarters. Zero instructions, one dependency.
    // 16 consecutive ds_read_b128 off ONE base VGPR, with the m-iteration row
    // offsets (16 rows x 128 B = 0x800) AND the 2-deep A ring slot as
    // compile-time immediates. 32768 + 30720 = 63488 fits the 16-bit unsigned
    // offset field.
    //
    // The outputs are dword4_t REFERENCES into the fragment's union objects (one
    // per m-tile), not copies: the MMACs read the same objects through their
    // bf16 view, so the fragment has ONE storage identity and the allocator
    // reuses the same 64 registers instead of materialising a second fragment at
    // the C' seam. That is the reference kernel's shape
    // (union_vec16_bf16 a_regs[16] + reinterpret_cast<dword4_t&>(a_regs[i].i32x4),
    // bf16_gemm_nt_hipc.cpp:1247 and :1281-1300).
    template <index_t SLOT>
    CK_TILE_DEVICE static void
    ds_read_b128_x16_imm(uint32_t base,
                         wasp_detail::dword4_t& o0,  wasp_detail::dword4_t& o1,  wasp_detail::dword4_t& o2,  wasp_detail::dword4_t& o3,
                         wasp_detail::dword4_t& o4,  wasp_detail::dword4_t& o5,  wasp_detail::dword4_t& o6,  wasp_detail::dword4_t& o7,
                         wasp_detail::dword4_t& o8,  wasp_detail::dword4_t& o9,  wasp_detail::dword4_t& o10, wasp_detail::dword4_t& o11,
                         wasp_detail::dword4_t& o12, wasp_detail::dword4_t& o13, wasp_detail::dword4_t& o14, wasp_detail::dword4_t& o15)
    {
        static_assert(SLOT + 30720 <= 65535,
                      "ds_read_b128 imm offset overflows the 16-bit unsigned field");
        asm volatile("ds_read_b128 %0,  %16 offset:%c17\n\t"
                     "ds_read_b128 %1,  %16 offset:%c18\n\t"
                     "ds_read_b128 %2,  %16 offset:%c19\n\t"
                     "ds_read_b128 %3,  %16 offset:%c20\n\t"
                     "ds_read_b128 %4,  %16 offset:%c21\n\t"
                     "ds_read_b128 %5,  %16 offset:%c22\n\t"
                     "ds_read_b128 %6,  %16 offset:%c23\n\t"
                     "ds_read_b128 %7,  %16 offset:%c24\n\t"
                     "ds_read_b128 %8,  %16 offset:%c25\n\t"
                     "ds_read_b128 %9,  %16 offset:%c26\n\t"
                     "ds_read_b128 %10, %16 offset:%c27\n\t"
                     "ds_read_b128 %11, %16 offset:%c28\n\t"
                     "ds_read_b128 %12, %16 offset:%c29\n\t"
                     "ds_read_b128 %13, %16 offset:%c30\n\t"
                     "ds_read_b128 %14, %16 offset:%c31\n\t"
                     "ds_read_b128 %15, %16 offset:%c32\n"
                     : "=&v"(o0),  "=&v"(o1),  "=&v"(o2),  "=&v"(o3),
                       "=&v"(o4),  "=&v"(o5),  "=&v"(o6),  "=&v"(o7),
                       "=&v"(o8),  "=&v"(o9),  "=&v"(o10), "=&v"(o11),
                       "=&v"(o12), "=&v"(o13), "=&v"(o14), "=&v"(o15)
                     : "v"(base),
                       "n"(SLOT + 0),     "n"(SLOT + 2048),  "n"(SLOT + 4096),  "n"(SLOT + 6144),
                       "n"(SLOT + 8192),  "n"(SLOT + 10240), "n"(SLOT + 12288), "n"(SLOT + 14336),
                       "n"(SLOT + 16384), "n"(SLOT + 18432), "n"(SLOT + 20480), "n"(SLOT + 22528),
                       "n"(SLOT + 24576), "n"(SLOT + 26624), "n"(SLOT + 28672), "n"(SLOT + 30720)
                     : "memory");
    }

    // One K-half of the 256M x K16 A operand for the whole wave tile. SLOT is
    // the A-ring slot ((k_stage % 2) * kAStageBytes), which the k loop unrolled
    // by parity makes compile-time.
    template <index_t SLOT, bool H1>
    CK_TILE_DEVICE void
    a_read_half(const AReadBases& b,
                wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters]) const
    {
        ds_read_b128_x16_imm<SLOT>(H1 ? b.h1 : b.h0,
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[0]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[1]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[2]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[3]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[4]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[5]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[6]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[7]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[8]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[9]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[10]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[11]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[12]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[13]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[14]),
                                   reinterpret_cast<wasp_detail::dword4_t&>(a_regs[15]));

    }

    // 64 MMACs for one K-half: kk(2) x m(16) x n(2). The standalone kernel's
    // mmac_half.
    //
    // kk stays OUTERMOST so consecutive MMACs write DIFFERENT accumulator
    // registers: the two writes to one slot end up 16 MMACs apart, which is what
    // hides the matrix pipe's accumulator pass-through behind 15 independent
    // MMACs. A (m, n, kk) order has a 4-deep RAW chain per slot instead.
    //
    // The wave-priority bracketing the standalone kernel wraps around this chain
    // (s_setprio 3 after one MMAC, 0 after the burst) was tried and it costs 4%
    // here: 293.5 -> 281.2 TFLOPS on this shape.
    //
    // A NOTE ON PARTIAL DRAINS: splitting this into two 16-MMAC halves fed by
    // halves of the A read (wait lgkmcnt(4) -> 16 MMACs -> wait lgkmcnt(0) ->
    // the other 16) was measured and it is SLOWER -- 256 TFLOPS against 299 with
    // the single drain. The LDS queue is shared with the other seven consumers,
    // so "wait for my first four reads" does not return any sooner; it just
    // takes two stalls instead of one.
    // The Q1 drain -- the reference kernel's `wait lgkmcnt(0); wait vmcnt(0)`:
    // lgkmcnt for A's LDS read, vmcnt for the B prefetch that Q1's MMACs consume.
    CK_TILE_DEVICE void wasp_wait_q1(wasp_detail::WaspVmemDiag& vmem_diag,
                                     index_t stage) const
    {
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        wasp_detail::wasp_wait_vmcnt0(vmem_diag, stage);
    }

    CK_TILE_DEVICE void mmac_half(const wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters],
                                  const wasp_detail::stage16_t (&b_half)[WaspAccTile::kNIters],
                                  AccTile& acc) const
    {
        using namespace wasp_detail;
        if((wasp_ablate() & 2) != 0)
        {
            return;   // ablation: skip MMACs
        }
        // The reference kernel peels the first MMAC out of the chain so that
        // set_prio<3> can be raised AFTER one has issued and set_prio<0> dropped
        // after the very last, with no per-iteration branch in the chain.
        //
        // (kk = 0, m = 0, n = 0)
        mmac_one<0, 0, 0>(a_regs, b_half, acc);
        wasp_set_prio<3>();

        // Keep every accumulator index compile-time. A dynamically indexed
        // vector array is lowered through private scratch by LLVM at this
        // register pressure, even when the loops are marked unrolled.
        static_for<1, WaspAccTile::kNIters, 1>{}([&](auto n) {
            mmac_one<0, 0, n.value>(a_regs, b_half, acc);
        });
        static_for<1, WaspAccTile::kMIters, 1>{}([&](auto m) {
            static_for<0, WaspAccTile::kNIters, 1>{}([&](auto n) {
                mmac_one<0, m.value, n.value>(a_regs, b_half, acc);
            });
        });
        static_for<0, WaspAccTile::kMIters, 1>{}([&](auto m) {
            static_for<0, WaspAccTile::kNIters, 1>{}([&](auto n) {
                mmac_one<1, m.value, n.value>(a_regs, b_half, acc);
            });
        });
        wasp_set_prio<0>();
    }

    // One 16x16x16 bf16 MMAC into acc.v[m*kNIters + n]. KK is literal so the
    // fragment indices stay compile-time; the peeling above only changes WHERE
    // the priority changes sit, never the order of the MMACs themselves.
    template <index_t KK, index_t M, index_t N>
    CK_TILE_DEVICE void mmac_one(const wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters],
                                 const wasp_detail::stage16_t (&b_half)[WaspAccTile::kNIters],
                                 AccTile& acc) const
    {
        using namespace wasp_detail;
        constexpr index_t slot = M * WaspAccTile::kNIters + N;
        acc.v[slot] = mmac_bf16(a_regs[M].frag[KK], b_half[N].frag[KK], acc.v[slot]);
    }

    // ------------------------------------------------------------------
    // PERSONA C's k-stage: (Q0,Q1) | BAR k.mid | (Q2,Q3) | BAR k.end.
    // Every read is drained inside the interval that issued it, so every A slot
    // is provably dead a full stage before the producer refills it.
    // ------------------------------------------------------------------
    template <index_t CUR>
    CK_TILE_DEVICE void c_I1(index_t consumer_id,
                             index_t k_stage,
                             index_t num_loop,
                             const AReadBases& a_rb,
                             wasp_detail::stage16_t (&b_regs)[2][2][WaspAccTile::kNIters],
                             wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters],
                             wasp_detail::WaspVmemDiag& vmem_diag,
                             AccTile& acc) const
    {
        using namespace wasp_detail;
        (void)consumer_id;
        (void)k_stage;
        (void)num_loop;
        constexpr index_t SLOT = CUR * kAStageBytes;
        // This interval issues no B prefetch and no stage-dependent addressing,
        // but keeps the same signature as its I2 sibling by design.
        (void)k_stage;
        (void)num_loop;
        if((wasp_ablate() & 32) != 0)
        {
            return;   // ablation: idle consumer
        }
        // Q0: A(k, h0).
        a_read_half<SLOT, false>(a_rb, a_regs);
        // Q1: A(h0) landed (lgkmcnt) + B(k) landed (vmcnt) -- B(k) was prefetched
        // in the previous stage and B(k+1) has not been issued yet.
        wasp_wait_q1(vmem_diag, k_stage);
        mmac_half(a_regs, b_regs[CUR][0], acc);
    }

    template <index_t CUR>
    CK_TILE_DEVICE void c_I2(index_t consumer_id,
                             index_t k_stage,
                             index_t num_loop,
                             const AReadBases& a_rb,
                             wasp_detail::buffer_rsrc_t b_rsrc,
                             const BPairs& b_pairs,
                             wasp_detail::stage16_t (&b_regs)[2][2][WaspAccTile::kNIters],
                             wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters],
                             wasp_detail::WaspVmemDiag& vmem_diag,
                             AccTile& acc) const
    {
        using namespace wasp_detail;
        constexpr index_t SLOT = CUR * kAStageBytes;
        (void)consumer_id;
        if((wasp_ablate() & 32) != 0)
        {
            return;   // ablation: idle consumer
        }
        // Q2: issue the B prefetch -- 4 x buffer_load_dwordx4 in ONE asm block,
        // the reference kernel's form (two IDXEN pairs, k-half on the immediate,
        // the stage's K column on the uniform SGPR soffset) -- and read A(k, h1).
        //
        // The MMACs of the next interval consume these bytes, so the Q1 drain
        // below is the reference kernel's `lgkmcnt(0); vmcnt(0)`.
        if(k_stage + 1 < num_loop)
        {
            load_B_stage(b_rsrc, b_pairs, k_stage + 1, b_regs[CUR ^ 1], vmem_diag);
        }
        a_read_half<SLOT, true>(a_rb, a_regs);
        // Q3: A(h1) landed. B(k)'s second half is resident, so NO vmcnt here --
        // draining it would stall on the B(k+1) prefetch just issued.
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        mmac_half(a_regs, b_regs[CUR][1], acc);
    }

    // ------------------------------------------------------------------
    // PERSONA C's k-stage, rotated by ONE QUARTER:
    // (Q3 of k-1, Q0) | BAR k.mid | (Q1, Q2) | BAR k.end.
    //
    // What crosses a barrier is the MMAC and the A fragment registers it
    // consumes. Each interval carries the same work as C's (one 16-unit read, one
    // 64-MMAC burst, one real drain): a rendezvous releases when the SLOWEST wave
    // arrives, so a heavier persona would just be paid for by the other half.
    // ------------------------------------------------------------------
    template <index_t CUR, bool IS_FIRST>
    CK_TILE_DEVICE void cp_I1(index_t consumer_id,
                              index_t k_stage,
                              index_t num_loop,
                              const AReadBases& a_rb,
                              wasp_detail::stage16_t (&b_regs)[2][2][WaspAccTile::kNIters],
                              wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters],
                              AccTile& acc) const
    {
        using namespace wasp_detail;
        (void)consumer_id;
        (void)k_stage;
        (void)num_loop;
        constexpr index_t SLOT = CUR * kAStageBytes;
        (void)k_stage;
        (void)num_loop;
        if((wasp_ablate() & 32) != 0)
        {
            return;   // ablation: idle consumer
        }
        if constexpr(!IS_FIRST)
        {
            // Q3 of the PREVIOUS stage: the quarter that crosses the barrier.
            // Its data was read at the end of the previous stage and is drained
            // HERE, so the wait is for a read issued a whole barrier ago -- that
            // is what the deferral buys. B(k-1) is still resident (its buffer is
            // CUR^1 here), which is why cp_I2's prefetch must not move earlier.
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            mmac_half(a_regs, b_regs[CUR ^ 1][1], acc);
        }
        // Q0: A(k, h0).
        a_read_half<SLOT, false>(a_rb, a_regs);
    }

    template <index_t CUR>
    CK_TILE_DEVICE void cp_I2(index_t consumer_id,
                              index_t k_stage,
                              index_t num_loop,
                              const AReadBases& a_rb,
                               wasp_detail::buffer_rsrc_t b_rsrc,
                              const BPairs& b_pairs,
                              wasp_detail::stage16_t (&b_regs)[2][2][WaspAccTile::kNIters],
                              wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters],
                               wasp_detail::WaspVmemDiag& vmem_diag,
                               AccTile& acc) const
    {
        using namespace wasp_detail;
        constexpr index_t SLOT = CUR * kAStageBytes;
        (void)consumer_id;
        if((wasp_ablate() & 32) != 0)
        {
            return;   // ablation: idle consumer
        }
        // Q1: A(k, h0) landed -- it was read before the mid barrier, so this wait
        // is cheap by construction -- and B(k) landed (prefetched one stage ago).
        wasp_wait_q1(vmem_diag, k_stage);
        mmac_half(a_regs, b_regs[CUR][0], acc);

        // Q2: issue the B prefetch -- 4 x buffer_load_dwordx4 in ONE asm block,
        // the reference kernel's form (two IDXEN pairs, k-half on the immediate,
        // the stage's K column on the uniform SGPR soffset) -- and read A(k, h1).
        // The bytes land in b_regs[CUR ^ 1] and are consumed by the next
        // interval's Q1 MMACs; that is what wasp_wait_q1()'s vmcnt half waits for.
        if(k_stage + 1 < num_loop)
        {
            load_B_stage(b_rsrc, b_pairs, k_stage + 1, b_regs[CUR ^ 1], vmem_diag);
        }
        a_read_half<SLOT, true>(a_rb, a_regs);
    }

    // Persona C driver: two intervals per k-stage, mid barrier between them.
    CK_TILE_DEVICE void consumer_run_c(index_t consumer_id,
                                                 index_t num_loop,
                                                 const AReadBases& a_rb,
                                                 wasp_detail::buffer_rsrc_t b_rsrc,
                                                 const BPairs& b_pairs,
                                                 wasp_detail::stage16_t (&b_regs)[2][2][WaspAccTile::kNIters],
                                                 wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters],
                                                 wasp_detail::WaspVmemDiag& vmem_diag,
                                                 AccTile& acc) const
    {
        for(index_t k = 0; k < num_loop; k += 2)
        {
            c_I1<0>(consumer_id, k, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
            wasp_detail::wasp_barrier_maybe();   // BAR(k).mid (bare, like the reference)
            c_I2<0>(consumer_id,
                    k,
                    num_loop,
                    a_rb,
                    b_rsrc,
                    b_pairs,
                    b_regs,
                    a_regs,
                    vmem_diag,
                    acc);
            wasp_detail::wasp_barrier_maybe();   // BAR(k).end
            if(k + 1 < num_loop)
            {
                c_I1<1>(
                    consumer_id, k + 1, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k+1).mid
                c_I2<1>(consumer_id,
                        k + 1,
                        num_loop,
                        a_rb,
                        b_rsrc,
                        b_pairs,
                        b_regs,
                        a_regs,
                        vmem_diag,
                        acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k+1).end
            }
        }
    }

    // Persona C' driver: the same barrier sequence, the rotated item order. The
    // first stage is peeled so IS_FIRST is compile-time (there is no stage -1 to
    // consume), and the rotation leaves exactly ONE MMAC owed after the loop. It
    // is executed here, without a barrier: the producer owes no extra rendezvous
    // for it either, so the cadence stays balanced.
    CK_TILE_DEVICE void consumer_run_cp(index_t consumer_id,
                                                   index_t num_loop,
                                                   const AReadBases& a_rb,
                                                   wasp_detail::buffer_rsrc_t b_rsrc,
                                                   const BPairs& b_pairs,
                                                   wasp_detail::stage16_t (&b_regs)[2][2][WaspAccTile::kNIters],
                                                   wasp_detail::stage16_t (&a_regs)[WaspAccTile::kMIters],
                                                   wasp_detail::WaspVmemDiag& vmem_diag,
                                                   AccTile& acc) const
    {
        if(num_loop == 0)
        {
            return;
        }
        if((num_loop & 1) != 0)
        {
            cp_I1<0, /*IS_FIRST=*/true>(consumer_id, 0, num_loop, a_rb, b_regs, a_regs, acc);
            wasp_detail::wasp_barrier_maybe();   // BAR(0).mid
            cp_I2<0>(
                consumer_id, 0, num_loop, a_rb, b_rsrc, b_pairs, b_regs, a_regs, vmem_diag, acc);
            wasp_detail::wasp_barrier_maybe();   // BAR(0).end
            for(index_t k = 1; k < num_loop; k += 2)
            {
                cp_I1<1, false>(consumer_id, k, num_loop, a_rb, b_regs, a_regs, acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k).mid
                cp_I2<1>(consumer_id,
                         k,
                         num_loop,
                         a_rb,
                         b_rsrc,
                         b_pairs,
                         b_regs,
                         a_regs,
                         vmem_diag,
                         acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k).end
                cp_I1<0, false>(consumer_id, k + 1, num_loop, a_rb, b_regs, a_regs, acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k+1).mid
                cp_I2<0>(consumer_id,
                         k + 1,
                         num_loop,
                         a_rb,
                         b_rsrc,
                         b_pairs,
                         b_regs,
                         a_regs,
                         vmem_diag,
                         acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k+1).end
            }
            if((wasp_detail::wasp_ablate() & 32) == 0)
            {
                // Odd num_loop leaves the final stage at CUR=0.
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
                mmac_half(a_regs, b_regs[0][1], acc);
            }
        }
        else
        {
            cp_I1<0, /*IS_FIRST=*/true>(consumer_id, 0, num_loop, a_rb, b_regs, a_regs, acc);
            wasp_detail::wasp_barrier_maybe();   // BAR(0).mid
            cp_I2<0>(
                consumer_id, 0, num_loop, a_rb, b_rsrc, b_pairs, b_regs, a_regs, vmem_diag, acc);
            wasp_detail::wasp_barrier_maybe();   // BAR(0).end

            for(index_t k = 1; k < num_loop - 1; k += 2)
            {
                cp_I1<1, false>(consumer_id, k, num_loop, a_rb, b_regs, a_regs, acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k).mid
                cp_I2<1>(consumer_id,
                         k,
                         num_loop,
                         a_rb,
                         b_rsrc,
                         b_pairs,
                         b_regs,
                         a_regs,
                         vmem_diag,
                         acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k).end
                cp_I1<0, false>(consumer_id, k + 1, num_loop, a_rb, b_regs, a_regs, acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k+1).mid
                cp_I2<0>(consumer_id,
                         k + 1,
                         num_loop,
                         a_rb,
                         b_rsrc,
                         b_pairs,
                         b_regs,
                         a_regs,
                         vmem_diag,
                         acc);
                wasp_detail::wasp_barrier_maybe();   // BAR(k+1).end
            }
            cp_I1<1, false>(consumer_id, num_loop - 1, num_loop, a_rb, b_regs, a_regs, acc);
            wasp_detail::wasp_barrier_maybe();   // BAR(k).mid
            cp_I2<1>(consumer_id,
                     num_loop - 1,
                     num_loop,
                     a_rb,
                     b_rsrc,
                     b_pairs,
                     b_regs,
                     a_regs,
                     vmem_diag,
                     acc);
            wasp_detail::wasp_barrier_maybe();   // BAR(k).end
            if((wasp_detail::wasp_ablate() & 32) == 0)
            {
                // Even num_loop leaves the final stage at CUR=1.
                asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
                mmac_half(a_regs, b_regs[1][1], acc);
            }
        }
    }
    // ------------------------------------------------------------------
    // Role-split entry point.
    //
    // Overlapping double-barrier cadence (two barriers per stage, A then B):
    //
    //   producer : issue(0)
    //              for k: wait vmcnt | BAR_A(k) | issue(k+1) | BAR_B(k)
    //   consumer : for k:             BAR_A(k) | consume(k)  | BAR_B(k)
    //
    // BAR_A(k) publishes stage k: the producer has drained the stage-k DMA
    // (vmcnt) before it, so the consumer reads a complete slab after it.
    // issue(k+1) is started straight after BAR_A(k), into the OTHER ring slot
    // ((k+1)%2 != k%2), so its global->LDS DMA runs UNDER the consumers' compute
    // of stage k. BAR_B(k) then proves the consumers are done reading slot k%2
    // before the producer refills it at issue(k+2) next iteration. Both roles
    // run exactly 2*num_loop barriers, in the same A,B order, so the split stays
    // legal.
    //
    // The two slots are the whole point of the ring: with a single slot the
    // producer could not stage k+1 while k is being read. The barrier carries
    // the double-buffer handshake; the DMA's own vmcnt carries completion.
    //
    // The entry barrier drains the PREVIOUS tile's epilogue, which borrows the
    // whole 64 KB buffer: without it this tile's issue(0) would overwrite A
    // staging while the previous consumers are still reading it back for stores.
    // ------------------------------------------------------------------
    // NOTE: GroupedGemmKernel invokes the pipeline with the SIX argument form
    // (num_loop, has_hot_loop, tail_number, smem) -- see
    // grouped_gemm_kernel.hpp:936-941 -- not the 4 argument one used by
    // UniversalGemmKernel's plain RunGemm. The hot-loop flag and the tail
    // number belong to the ComputeV3 TailHandler, which a role-split body does
    // not use: the k-tail is folded into producer_fill's mask instead (TODO,
    // see the header note). Both forms are provided so either caller works.
    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE AccTile operator()(const ADramBlockWindow& a_dram_block_window,
                                      const BDramBlockWindow& b_dram_block_window,
                                      index_t num_loop,
                                      bool /*has_hot_loop*/,
                                      TailNumber /*tail_number*/,
                                      void* p_smem) const
    {
        // ---- per-TILE tensor plumbing, hoisted out of the k loop ----
        // The windows are pure scalar carriers here: extract the base pointers,
        // strides, M/N lengths and the tile origin, then hand them to run_raw.
        // (The grouped kernel's RunGemmWasp path computes the SAME scalars
        // straight from kargs + tile origin and calls run_raw directly, so no
        // tile-window object lives across the k loop -- see run_raw's note.)
        const auto  a_view = a_dram_block_window.get_bottom_tensor_view();
        const auto& a_desc = a_view.get_tensor_descriptor();
        const auto  a_org  = a_dram_block_window.get_window_origin();
        const auto* a_base = a_view.get_buffer_view().p_data_ +
                             a_desc.calculate_offset(make_multi_index(a_org[0], a_org[1]));
        const index_t lda = a_desc.calculate_offset(make_multi_index(1, 0)) -
                            a_desc.calculate_offset(make_multi_index(0, 0));
        // M tail: a_org[0] is this tile's first GLOBAL row within the group's A,
        // get_lengths()[0] is the group's M, so valid tile-local rows = M - org.
        const index_t a_m     = a_desc.get_lengths()[number<0>{}];
        const index_t m_valid = a_m - a_org[0];

        const auto  b_view = b_dram_block_window.get_bottom_tensor_view();
        const auto& b_desc = b_view.get_tensor_descriptor();
        const auto  b_org  = b_dram_block_window.get_window_origin();
        const auto* b_base = b_view.get_buffer_view().p_data_ +
                             b_desc.calculate_offset(make_multi_index(b_org[0], b_org[1]));
        // B is ColumnMajor for the NT layout, so the view is (N, K): index 1 is
        // the n stride, i.e. the row pitch the reference kernel calls `b_stride`.
        const index_t b_n_stride = b_desc.calculate_offset(make_multi_index(1, 0)) -
                                   b_desc.calculate_offset(make_multi_index(0, 0));
        const index_t b_n_len = b_desc.get_lengths()[number<0>{}];

        return run_raw(a_base,
                       lda,
                       m_valid,
                       b_base,
                       b_n_stride,
                       b_n_len,
                       static_cast<index_t>(b_org[0]),
                       num_loop,
                       p_smem);
    }

    // Raw-pointer core: the whole k-loop body, fed by SCALARS instead of tile
    // windows. The framework's generic RunGemm builds tile-window objects for
    // A/B/C/D and keeps the C/D ones live across the entire k loop (the epilogue
    // needs them afterwards) -- ~30-40 VGPR of dead-weight live state stacked on
    // the 224-VGPR accumulator/fragment floor, which is what pushed this kernel
    // over the 256-VGPR cap and spilled the A fragment to scratch. The grouped
    // kernel's RunGemmWasp path computes these scalars from kargs + tile origin
    // and calls this directly, so no window state lives across the loop.
    //
    //   a_base/lda/m_valid : A row-major base (origin folded in), row stride,
    //                        valid tile-local rows
    //   b_base/b_n_stride  : B col-major base (origin folded in), n row pitch
    //   b_n_len/b_n_origin : B's N length and this tile's n origin (for the SRD
    //                        size bound (b_n_len - b_n_origin) * pitch)
    CK_TILE_DEVICE AccTile run_raw(const ADataType* a_base,
                                   index_t lda,
                                   index_t m_valid,
                                   const BDataType* b_base,
                                   index_t b_n_stride,
                                   index_t b_n_len,
                                   index_t b_n_origin,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        // Preserve the value-returning pipeline interface for generic CK
        // callers. The specialized grouped Wasp path uses run_raw_into so
        // producer waves never need a defined accumulator value.
        AccTile acc{};
        run_raw_into(a_base,
                     lda,
                     m_valid,
                     b_base,
                     b_n_stride,
                     b_n_len,
                     b_n_origin,
                     num_loop,
                     acc,
                     p_smem);
        return acc;
    }

    CK_TILE_DEVICE void run_raw_into(const ADataType* a_base,
                                     index_t lda,
                                     index_t m_valid,
                                     const BDataType* b_base,
                                     index_t b_n_stride,
                                     index_t b_n_len,
                                     index_t b_n_origin,
                                     index_t num_loop,
                                     AccTile& acc,
                                     void* p_smem) const
    {
        char* const lds   = static_cast<char*>(p_smem);
        const index_t wid = get_warp_id();

        wasp_detail::WaspVmemDiag vmem_diag{};

        // CACHE_SWIZZLE descriptor over this tile's B (built once per tile, not
        // per k-stage). STRIDE = the row pitch in bytes; size bounds the SRD to
        // the group's remaining N (b_n_len - b_n_origin).
        const uint32_t b_pitch_bytes = static_cast<uint32_t>(b_n_stride) * sizeof(uint16_t);
        const uint32_t b_size_bytes =
            static_cast<uint32_t>(b_n_len - b_n_origin) * b_pitch_bytes;
        const wasp_detail::buffer_rsrc_t b_rsrc =
            wasp_detail::wasp_make_buffer_rsrc(b_base, b_pitch_bytes, b_size_bytes,
                                               /*allow_swizzle=*/true);
        // Built once per tile, not per k-stage: the prefetch's SRD. Its data is
        // consumed by the k loop's MMACs one stage later (see the interval
        // bodies' Q1/Q2 ordering).

        wasp_detail::wasp_sync_lds_maybe();   // drain the previous tile's epilogue (both roles)

        if(wid < kNumProducerWaves)
        {
            // The 8 panel vaddrs and the SRD are per-TILE: build them once,
            // outside the k loop (see AProdSrc / make_a_prod_src).
            const auto a_src = make_a_prod_src(a_base, lda, m_valid, wid);

            // hipc's producer shape: issue first, then the barrier pair --
            // BAR(k).mid is a rendezvous, BAR(k).end is what PUBLISHES A(k+1)
            // (this role has waited for that DMA in front of it). The consumers'
            // stage k+1 starts on the far side of that barrier.
            producer_fill(a_src, 0, wid);                     // issue stage 0
            wasp_detail::wasp_barrier_maybe();               // BAR prologue.mid
            wasp_detail::wasp_wait_vmcnt0();                 // A(0) landed
            wasp_detail::wasp_barrier_maybe();               // BAR prologue.end: publish A(0)

            for(index_t k = 0; k < num_loop; ++k)
            {
                // Issuing k+1 here instead of after the barrier buys its DMA a
                // whole extra interval to land (it is confirmed below), and the
                // slot it writes -- (k+1)%2 -- was last read by the consumers in
                // stage k-1, i.e. one full interval before this issue.
                if(k + 1 < num_loop)
                {
                    producer_fill(a_src, k + 1, wid);         // issue k+1 -> slot (k+1)%2
#if defined(CK_TILE_GROUPED_GEMM_WASP_VMEM_WAIT_COUNT)
                    if(k + 1 == wasp_detail::kVmemSampleStage)
                    {
                        vmem_diag.issued_at = wasp_detail::wasp_memtime();
                        vmem_diag.issued = true;
                    }
#endif
                }
                wasp_detail::wasp_barrier_maybe();           // BAR(k).mid
                wasp_detail::wasp_wait_vmcnt0_producer(vmem_diag, k + 1); // A(k+1) landed
                wasp_detail::wasp_barrier_maybe();           // BAR(k).end: publish A(k+1)
            }
        }
        else
        {
            // Accumulators are consumed only by consumer waves. Keeping this
            // initialization inside their role branch avoids issuing 128
            // accumulator-zeroing moves on producer waves.
            acc = AccTile{};
            const index_t consumer_id = wid - kNumProducerWaves;
            // B double buffer [parity][k_half][n]. Prime stage 0's B, then walk
            // the k loop unrolled by parity so the live buffer index (CUR = k&1)
            // is a compile-time template arg.
            // The A fragment file is owned HERE, not by the per-stage body: the
            // C' persona's deferred MMAC consumes it one stage after the read
            // that filled it (see cp_I1).
            wasp_detail::stage16_t a_regs[WaspAccTile::kMIters];   // one whole K-half
            // Per-lane addressing values, computed once per item and shared by
            // the interval bodies, so the K loop carries only the reference
            // kernel's A bases and B index pairs.
            const uint32_t lane    = static_cast<uint32_t>(get_lane_id());
            const uint32_t row     = lane & 15u;
            const uint32_t r7      = row & 7u;
            const uint32_t colg    = lane >> 4;
            const uint32_t rowbase = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) +
                                     row * (kStageK * sizeof(uint16_t));
            const AReadBases a_rb{rowbase + ((colg ^ r7) * 16u),
                                  rowbase + (((colg + 4u) ^ r7) * 16u)};
            const uint32_t nrow  = static_cast<uint32_t>(consumer_id) * 32u + row;
            const uint32_t inrow = colg * 16u;
            const BPairs b_pr{(static_cast<uint64_t>(inrow) << 32) | nrow,
                              (static_cast<uint64_t>(inrow) << 32) | (nrow + 16u)};
            // Destination of the prefetch, double buffered [parity][k-half][n].
            wasp_detail::stage16_t b_regs[2][2][WaspAccTile::kNIters];
            // Prologue, mirroring the producer's: prime B(0), barrier, wait for
            // it, barrier. The second barrier is the one that publishes A(0) --
            // the producer reaches it only after waiting for that DMA.
            // Prime B(0); the first stage's MMACs consume it (see the wait below).
            if(num_loop > 0)
            {
                load_B_stage(b_rsrc, b_pr, 0, b_regs[0], vmem_diag);
            }
            wasp_detail::wasp_barrier_maybe();               // BAR prologue.mid
            wasp_detail::wasp_wait_vmcnt0();                 // B(0) landed
            wasp_detail::wasp_barrier_maybe();               // BAR prologue.end

            // waves 0..3 -> C, 4..7 -> C'
            if(consumer_id < kConsumerSplitWave)
            {
                consumer_run_c(
                    consumer_id, num_loop, a_rb, b_rsrc, b_pr, b_regs, a_regs, vmem_diag, acc);
            }
            else
            {
                consumer_run_cp(
                    consumer_id, num_loop, a_rb, b_rsrc, b_pr, b_regs, a_regs, vmem_diag, acc);
            }
        }

        // Drain barrier before the epilogue.
        //
        // The epilogue borrows the whole 64 KB buffer to stage its output: its
        // 8 per-consumer slabs tile the ENTIRE LDS, so consumer 0..3 write over
        // the very region (the first A stage) that consumer 4..7 are still
        // reading A from. Without this barrier a consumer that finishes its K
        // loop early corrupts the A operands of the consumers that have not,
        // which shows up as the second half of the N range being garbage while
        // the first half is exact.
        //
        // It has to live HERE rather than in the epilogue: the epilogue runs
        // only for the consumer waves, so a barrier inside it would never be
        // reached by the producers. Every wave arrives here, the producers then
        // fall through into the epilogue and return immediately.
        wasp_detail::wasp_sync_lds_maybe();

#if defined(CK_TILE_GROUPED_GEMM_WASP_VMEM_WAIT_COUNT)
        if(get_lane_id() == 0)
        {
            const index_t cta = get_block_id();
            if(cta < 1024 && wid < 16)
            {
                wasp_detail::g_wasp_vmem_samples[cta * 16 + wid] = vmem_diag.sample;
            }
        }
#endif

    }

    // 4 argument form, forwarded. Kept so that a caller that only knows the
    // plain RunGemm convention does not silently fall back to the inherited
    // ComputeV3 body (same-arity overloads in a base class are hidden, not
    // merged, by a derived declaration).
    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE AccTile operator()(const ADramBlockWindow& a_dram_block_window,
                                      const BDramBlockWindow& b_dram_block_window,
                                      index_t num_loop,
                                      void* p_smem) const
    {
        return (*this)(a_dram_block_window,
                       b_dram_block_window,
                       num_loop,
                       true,
                       TailNumber::Full,
                       p_smem);
    }
};

} // namespace ck_tile
