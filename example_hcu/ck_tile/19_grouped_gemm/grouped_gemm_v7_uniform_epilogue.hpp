// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Store epilogue for the uniform Direct-to-LDS NN/TN pipelines
// (gemm_pipeline_v7_uniform.hpp). Same writeback the reference kernels use,
// expressed in ck_tile terms.
//
// WHY NOT CShuffleEpilogue
// ------------------------
// The MMAC accumulator puts one lane's four results at columns {q, q+4, q+8,
// q+12} (q = lane >> 4) across 16 rows, so direct scalar stores never coalesce:
// measured 33.2M VL1 write requests on the reference shape against hipBLASLt's
// 8.4M, and because two CTAs overlap per CU that write flood congests TA/TC for
// the WHOLE kernel. Staging each 16-row slice through the (by then idle) ring
// LDS and reading it back so every lane holds one contiguous 16 B run turns 128
// scalar stores into 16 vector stores per wave: 33.2M -> ~1M requests, measured
// 1.100 ms / 247 TF -> 0.852 ms / 319 TF on the reference shape.
//
// Shuffle-free (per-wave LDS slabs, no block barrier) and 16 B stores are the two
// properties that matter; the layout below is the reference kernel's.
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp"

// The address arithmetic below is a port of the reference kernels, which are
// built with -Wno-old-style-cast; every cast is a fixed-width unsigned/signed
// conversion in an address expression. CK's own arch/buffer-addressing headers
// handle it the same way (see cast_pointer_to_constant_address_space). The
// 16 B store path reinterprets the output element pointer as one dwordx4
// vector -- an alignment-checked bit reinterpretation, not a type pun.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"

namespace ck_tile {

namespace v7u_epi_detail {

CK_TILE_DEVICE void ds_write_ele(uint32_t addr, uint16_t v)
{
    asm volatile("ds_write_b16 %0, %1" ::"v"(addr), "v"((uint32_t)v) : "memory");
}
CK_TILE_DEVICE void ds_write_ele(uint32_t addr, uint32_t v)
{
    asm volatile("ds_write_b32 %0, %1" ::"v"(addr), "v"(v) : "memory");
}

using u32x4 = int32_t __attribute__((ext_vector_type(4)));
CK_TILE_DEVICE u32x4 ds_read_b128(uint32_t addr)
{
    u32x4 v;
    asm volatile("ds_read_b128 %0, %1" : "=v"(v) : "v"(addr) : "memory");
    return v;
}

// fp32 -> output. The bf16 arm is the reference kernels' round-to-nearest-even
// shift (bit-identical to what the validated trees produce); fp16 goes through
// the hardware convert.
template <typename Element_O>
CK_TILE_DEVICE Element_O fp32_to(float f);
template <>
CK_TILE_DEVICE float fp32_to<float>(float f)
{
    return f;
}
template <>
CK_TILE_DEVICE bf16_t fp32_to<bf16_t>(float f)
{
    union
    {
        float fp32;
        uint32_t u32;
    } u = {f};
    u.u32 += 0x7fff + ((u.u32 >> 16) & 1);
    bf16_t r;
    *reinterpret_cast<uint16_t*>(&r) = (uint16_t)(u.u32 >> 16);
    return r;
}
template <>
CK_TILE_DEVICE half_t fp32_to<half_t>(float f)
{
    return __float2half(f);
}

}   // namespace v7u_epi_detail

// ---------------------------------------------------------------------------
// The epilogue. `store_tile` is called with the accumulator the pipeline filled
// and the same LDS the ring used (which the k loop has released by then).
// ---------------------------------------------------------------------------
template <typename Element_O_, typename AccTile_, typename Geometry = V7UniformGeometry>
struct GroupedGemmV7UniformStoreEpilogue
{
    using Element_O = remove_cvref_t<Element_O_>;
    using AccTile   = remove_cvref_t<AccTile_>;
    using G         = Geometry;

    // ---- framework-facing surface (what GroupedGemmKernel reads) --------
    using ODataType = Element_O;
    using DsDataType = ck_tile::tuple<>;   // no D tensors: single-stage store
    using DsLayout   = ck_tile::tuple<>;
    static constexpr index_t NumDTensor = 0;
    static constexpr memory_operation_enum MemoryOperation = memory_operation_enum::set;

    static constexpr int kE      = (int)sizeof(Element_O);       // 2 | 4
    static constexpr int kRowB   = (int)G::kWaveTileN * kE;      // 256 | 512
    static constexpr int kUnits  = kRowB / 16;                   // 16 | 32
    static constexpr int kSliceB = 16 * kRowB;                   // 4 KB | 8 KB

    // The epilogue reuses the pipeline's ring buffer, so it asks for none of its
    // own: the framework's max(pipeline, epilogue) keeps the pipeline's 64 KB.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    static_assert(kE == 2 || kE == 4, "2- or 4-byte output only");
    static_assert(kSliceB * (int)G::kWaves <= (int)G::kLdsBytes,
                  "the per-wave staging slabs must fit the ring LDS");

    // The store ALSO drains whatever loads the k tail left outstanding: their
    // LDS write-backs would otherwise land in the ring while the slabs are being
    // staged (observed as a few hundred bad elements, varying per run).
    CK_TILE_DEVICE static void store_tile(const V7UniformTileParams& p,
                                          const AccTile& acc,
                                          void* smem)
    {
        using namespace v7u_epi_detail;
        (void)smem;

        Element_O* d_out = reinterpret_cast<Element_O*>(p.d_ptr);
        const index_t d_stride = p.d_stride;

        const int tid  = (int)threadIdx.x;
        const int wid  = tid >> 6;
        const int lane = tid & 63;
        const int wave_m_start = (wid >> 1) * (int)G::kWaveTileM;
        const int wave_n_start = (wid & 1) * (int)G::kWaveTileN;

        v7u::wait_all_buffer_data_arrived<true>(0);

        const uint32_t epi_base = (uint32_t)(wid * kSliceB);
        const uint32_t row_q    = (uint32_t)(lane & 15);
        const uint32_t row_xor  = (uint32_t)(lane & 7);
        const uint32_t col_q    = (uint32_t)(lane >> 4);

#pragma unroll
        for(index_t mF = 0; mF < G::kMTilesPerWave; ++mF)
        {
#pragma unroll
            for(index_t nF = 0; nF < G::kNTilesPerWave; ++nF)
            {
                const index_t idx = mF * G::kNTilesPerWave + nF;
#pragma unroll
                for(int c = 0; c < 4; ++c)
                {
                    const uint32_t byte = (col_q + (uint32_t)(nF * 16 + 4 * c)) * (uint32_t)kE;
                    const uint32_t addr = epi_base + row_q * (uint32_t)kRowB +
                                          (((byte >> 4) ^ row_xor) << 4) + (byte & 15u);
                    const Element_O val = fp32_to<Element_O>(acc.v[idx][c]);
                    if constexpr(kE == 2)
                    {
                        ds_write_ele(addr, *reinterpret_cast<const uint16_t*>(&val));
                    }
                    else
                    {
                        ds_write_ele(addr, *reinterpret_cast<const uint32_t*>(&val));
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            // Read the whole slice back FIRST, then store: the ds_read results
            // are inline-asm VGPR defs, so the compiler will not insert an LGKM
            // wait before the consuming global stores.
            constexpr int kJ = kSliceB / 64 / 16;   // 4 | 8 reads per lane
            u32x4 vj[kJ];
#pragma unroll
            for(int j = 0; j < kJ; ++j)
            {
                const uint32_t t     = (uint32_t)lane + 64u * (uint32_t)j;
                const uint32_t r_in  = t / (uint32_t)kUnits;
                const uint32_t u     = t % (uint32_t)kUnits;
                const uint32_t addr  = epi_base + r_in * (uint32_t)kRowB +
                                       ((u ^ (r_in & 7u)) << 4);
                vj[j] = ds_read_b128(addr);
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for(int j = 0; j < kJ; ++j)
            {
                const uint32_t t    = (uint32_t)lane + 64u * (uint32_t)j;
                const uint32_t r_in = t / (uint32_t)kUnits;
                const uint32_t u    = t % (uint32_t)kUnits;
                const index_t row   = p.m_start + wave_m_start + mF * 16 + (index_t)r_in;
                if(row < p.m)
                {
                    const index_t col0 =
                        p.n_start + wave_n_start + (index_t)u * (16 / kE);
                    *reinterpret_cast<u32x4*>(&d_out[row * d_stride + col0]) = vj[j];
                }
            }
        }
    }
};

}   // namespace ck_tile

#pragma clang diagnostic pop
