// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac.hpp"

namespace ck_tile {
// Note:WT refers to Wave Tile Shape, MR refers to M Repeat, NR refers to N Repeat,
// MI refers to M Interleave, NI refers to N Interleave, KIterate refers to K Iterate.

// fp16
// FIXME:NR=2 X NI=4 = 16 X 8 = 128 not 64
using WarpGemmMmacF16F16F32_WT32x64x32_MR2NR2MI1NI4 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 2, 1, 4, 2>>;

using WarpGemmMmacF16F16F32_WT32x64x32_MR2NR1MI1NI4 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 1, 1, 4, 2>>;

using WarpGemmMmacF16F16F32_WT32x32x16_MR2NR2MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 2, 1, 1, 1>>;

using WarpGemmMmacF16F16F32_WT32x32x16_MR1NR1MI2NI2 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 2, 2, 1>>;

using WarpGemmMmacF16F16F32_WT16x64x32_MR1NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 4, 1, 1, 2>>;
using WarpGemmMmacF16F16F32_WT16x64x32_MR1NR1MI1NI4 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 4, 2>>;
using WarpGemmMmacF16F16F32_WT16x32x64_MR1NR2MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 2, 1, 1, 4>>;

// [HCU移植] 以下 _FMHA 别名为 FMHA 移植新增：与上面 QK(16x64x32)/PV(16x32x64) 形状相同，
// 但基于 WarpGemmAttributeMmacIterateKCRaw（裸硬件 C 布局），供 FMHA 流水线在寄存器内直接
// 消费；普通别名保留基类 C 布局给 GEMM example，两套不可混用。
// ---- FMHA fwd (gfx936/gfx928/gfx92a basic MMAC) warp tiles ----
// Same shapes as the QK (16x64x32) and PV (16x32x64) aliases above, but built on
// WarpGemmAttributeMmacIterateKCRaw, which carries the raw-hardware C register layout the
// FMHA pipeline consumes in-register. The plain aliases keep the base C layout that the
// standalone basic_gemm / grouped_gemm epilogue depends on; do not merge the two.
using WarpGemmMmacF16F16F32_WT16x64x32_MR1NR4MI1NI1_FMHA = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKCRaw<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 4, 1, 1, 2>>;
using WarpGemmMmacBF16BF16F32_WT16x64x32_MR1NR4MI1NI1_FMHA = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKCRaw<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 4, 1, 1, 2>>;
using WarpGemmMmacF16F16F32_WT16x32x64_MR1NR2MI1NI1_FMHA = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKCRaw<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 2, 1, 1, 4>>;
using WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_FMHA = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKCRaw<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 2, 1, 1, 4>>;

using WarpGemmMmacF16F16F32_WT32x64x32_MR2NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 4, 1, 1, 2>>;

// Triton-layout grouped-GEMM probe: keep the full 32x64 wave output tile, but
// consume one K16 operand set per call. This lets the block GEMM share each B
// LDS read across four N repeats instead of materializing two 32x32 calls.
using WarpGemmMmacF16F16F32_WT32x64x16_MR2NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 4, 1, 1, 1>>;

// Triton gfx936 uses one M128xN64 wave tile: eight M repeats and four N
// repeats share the same K16 B operand distribution.
using WarpGemmMmacF16F16F32_WT128x64x16_MR8NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 8, 4, 1, 1, 1>>;
using WarpGemmMmacF16F16F32_WT64x64x16_MR4NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 4, 4, 1, 1, 1>>;
using WarpGemmMmacF16F16F32_WT128x64x16_MR8NR4MI1NI1_NOuter = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKNOuter<
        WarpGemmAttributeMmacImplF16F16F32M16N16K16,
        8,
        4,
        1,
        1,
        1>>;
using WarpGemmMmacF16F16F32_WT64x64x16_MR4NR4MI1NI1_NOuter = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKNOuter<
        WarpGemmAttributeMmacImplF16F16F32M16N16K16,
        4,
        4,
        1,
        1,
        1>>;

using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR1MI1NI2 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 2, 8>>;

using WarpGemmMmacF16F16F32_WT16x16x128_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 1, 8>>;

// bf16
using WarpGemmMmacBF16BF16F32_WT32x32x16_MR2NR2MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16,
                                  2,
                                  2,
                                  1,
                                  1,
                                  1>>;

using WarpGemmMmacBF16BF16F32_WT32x32x16_MR1NR1MI2NI2 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 2, 2, 1>>;

using WarpGemmMmacBF16BF16F32_WT16x32x128_MR1NR1MI1NI2 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 2, 8>>;

using WarpGemmMmacBF16BF16F32_WT16x16x128_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 8>>;

// [HCU移植] 以下 16x16 系列别名为 FMHA BWD 移植新增（WIP）：BWD 的 5 个 GEMM 都切成
// 16x16 warp tile（K=32/16），_TRANSC 变体用于两个转置 GEMM（P^T*dO、dS^T*Q^T）。
// ---- FMHA BWD (gfx9 basic MMAC) warp tiles ----
// The BWD pipeline tiles every GEMM to a 16x16 warp tile with K=32 (kK0/kK2/kK4)
// or K=16 (kK1/kK3). These reuse the basic (non lit_lts) MMAC C register layout that
// was reverse-engineered/fixed for FWD, so non gfx938/946 archs avoid the TransC_GFX938
// path. The _TRANSC variants provide the transposed-C distribution used by the two
// transposed GEMMs (P^T*dO and dS^T*Q^T).
using WarpGemmMmacF16F16F32_WT16x16x32_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 1, 2>>;
using WarpGemmMmacF16F16F32_WT16x16x16_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 1, 1>>;
using WarpGemmMmacBF16BF16F32_WT16x16x32_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 2>>;
using WarpGemmMmacBF16BF16F32_WT16x16x16_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 1>>;

using WarpGemmMmacF16F16F32_WT16x16x32_MR1NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                        1, 1, 1, 1, 2>>;
using WarpGemmMmacF16F16F32_WT16x16x16_MR1NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                        1, 1, 1, 1, 1>>;
using WarpGemmMmacBF16BF16F32_WT16x16x32_MR1NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplBF16BF16F32M16N16K16TransC,
                                        1, 1, 1, 1, 2>>;
using WarpGemmMmacBF16BF16F32_WT16x16x16_MR1NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplBF16BF16F32M16N16K16TransC,
                                        1, 1, 1, 1, 1>>;

using WarpGemmMmacBF16BF16F32_WT16x64x32_MR1NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 4, 1, 1, 2>>;
using WarpGemmMmacBF16BF16F32_WT32x64x32_MR2NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 2, 4, 1, 1, 2>>;
using WarpGemmMmacBF16BF16F32_WT32x64x16_MR2NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 2, 4, 1, 1, 1>>;
using WarpGemmMmacBF16BF16F32_WT128x64x16_MR8NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 8, 4, 1, 1, 1>>;
using WarpGemmMmacBF16BF16F32_WT64x64x16_MR4NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 4, 4, 1, 1, 1>>;
using WarpGemmMmacBF16BF16F32_WT128x64x16_MR8NR4MI1NI1_NOuter = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKNOuter<
        WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16,
        8,
        4,
        1,
        1,
        1>>;
using WarpGemmMmacBF16BF16F32_WT64x64x16_MR4NR4MI1NI1_NOuter = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKNOuter<
        WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16,
        4,
        4,
        1,
        1,
        1>>;
using WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 2, 1, 1, 4>>;

// v2 refers to KIterate not continuous in KPerLane
using WarpGemmMmacF16F16F32_WT32x16x256_MR2NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v2<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                           2,
                                           1,
                                           1,
                                           1,
                                           16>>;
using WarpGemmMmacF16F16F32_WT32x16x256_MR2NR1MI1NI1_TRANSC_LEGACY = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v2<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransCLegacy,
                                           2,
                                           1,
                                           1,
                                           1,
                                           16>>;

// moe matrix B loaded into none swizzled lds
using WarpGemmMmacF16F16F32_WT16x16x128_MR1NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v2<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                           1,
                                           1,
                                           1,
                                           1,
                                           8>>;
using WarpGemmMmacF16F16F32_WT16x16x128_MR1NR1MI1NI1_TRANSC_LEGACY = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v2<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransCLegacy,
                                           1,
                                           1,
                                           1,
                                           1,
                                           8>>;

// moe matrix B loaded into swizzled lds
using WarpGemmMmacF16F16F32_WT16x16x64_MR1NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v3<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                           1,
                                           1,
                                           1,
                                           1,
                                           4>>; 
using WarpGemmMmacF16F16F32_WT16x16x64_MR1NR1MI1NI1_TRANSC_LEGACY = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v3<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransCLegacy,
                                           1,
                                           1,
                                           1,
                                           1,
                                           4>>;

// moe matrix B loaded into registers directly
using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR2MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                        1,
                                        2,
                                        1,
                                        1,
                                        8>>;
using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR2MI1NI1_TRANSC_LEGACY = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransCLegacy,
                                        1,
                                        2,
                                        1,
                                        1,
                                        8>>;

using WarpGemmMmacF16F16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                        1,
                                        2,
                                        1,
                                        1,
                                        4>>;
using WarpGemmMmacF16F16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC_LEGACY = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransCLegacy,
                                        1,
                                        2,
                                        1,
                                        1,
                                        4>>;

// gfx938/946 FMHA types: use LitLts impl with dedicated IterateKLitLts which provides
// the correct CWarpDstrEncoding for the lit_lts MMAC C register layout
// (kCMLane=4, kCNLane=16, kCM0PerLane=1, kCM1PerLane=4, kCNPerLane=1).
using WarpGemmMmacF16F16F32_WT16x64x32_MR1NR4MI1NI1_GFX938 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKLitLts<WarpGemmAttributeMmacImplF16F16F32M16N16K16LitLts,
                                         1,
                                         4,
                                         1,
                                         1,
                                         2>>;

using WarpGemmMmacF16F16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC_GFX938 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKLitLts<WarpGemmAttributeMmacImplF16F16F32M16N16K16LitLts,
                                         1,
                                         2,
                                         1,
                                         1,
                                         4>>;

using WarpGemmMmacBF16BF16F32_WT16x64x32_MR1NR4MI1NI1_GFX938 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKLitLts<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16LitLts,
                                         1,
                                         4,
                                         1,
                                         1,
                                         2>>;

using WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC_GFX938 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKLitLts<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16LitLts,
                                         1,
                                         2,
                                         1,
                                         1,
                                         4>>;
// bf16
using WarpGemmMmacBF16BF16F32_WT16x32x128_MR1NR2MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplBF16BF16F32M16N16K16TransC,
                                        1,
                                        2,
                                        1,
                                        1,
                                        8>>;

using WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplBF16BF16F32M16N16K16TransC,
                                        1,
                                        2,
                                        1,
                                        1,
                                        4>>;

// moe preshuffle matrix B loaded into registers directly
using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR1MI1NI2_Preshuffle = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKShuffle<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 2, 8>>;

using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR2MI1NI1_Preshuffle = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_Shuffle<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                                1,
                                                2,
                                                1,
                                                1,
                                                8>>;
using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR2MI1NI1_Preshuffle_LEGACY = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_Shuffle<
        WarpGemmAttributeMmacImplF16F16F32M16N16K16TransCLegacy,
        1,
        2,
        1,
        1,
        8>>;

//int8
using WarpGemmMmacI8I8I32_WT16x16x32_MR1NR1MI1NI1 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 1, 1>>;
using WarpGemmMmacI8I8I32_WT32x64x32_MR2NR1MI1NI4 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 2, 1, 1, 4, 1>>;
using WarpGemmMmacI8I8I32_WT32x64x64_MR2NR4MI1NI1 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 2, 4, 1, 1, 2>>;

// for MOE GEMM0
using WarpGemmMmacI8I8I32_WT16x16x64_MR1NR1MI1NI1 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 1, 2>>;
using WarpGemmMmacI8I8I32_WT16x16x128_MR1NR1MI1NI1 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 1, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x128_MR1NR1MI1NI2 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 2, 4>>;
using WarpGemmMmacI8I8I32_WT16x64x128_MR1NR1MI1NI4 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 4, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x128_MR1NR1MI1NI2_Preshuffle = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKShuffle<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 2, 4>>;

// including extra scales for matrix A and B
using WarpGemmMmacI8I8F32_WT16x16x32_MR1NR1MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 1, 1, 1, 1>>;
using WarpGemmMmacI8I8F32_WT16x16x64_MR1NR1MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 1, 1, 1, 2>>;
using WarpGemmMmacI8I8F32_WT16x32x64_MR1NR2MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 2, 1, 1, 2>>;
using WarpGemmMmacI8I8F32_WT16x32x128_MR1NR2MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 2, 1, 1, 4>>;
using WarpGemmMmacI8I8F32_WT16x64x64_MR1NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 4, 1, 1, 2>>;
using WarpGemmMmacI8I8F32_WT16x64x128_MR1NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 4, 1, 1, 4>>;
using WarpGemmMmacI8I8F32_WT16x64x32_MR1NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 4, 1, 1, 1>>;
using WarpGemmMmacI8I8F32_WT32x64x32_MR2NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 2, 4, 1, 1, 1>>;
using WarpGemmMmacI8I8F32_WT32x64x64_MR2NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 2, 4, 1, 1, 2>>;


// for MOE GEMM1
using WarpGemmMmacI8I8I32_WT16x16x64_MR1NR1MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 1, 1, 1, 2>>;
using WarpGemmMmacI8I8I32_WT16x32x64_MR1NR2MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 2>>;
using WarpGemmMmacI8I8I32_WT16x32x128_MR1NR2MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x256_MR1NR2MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 8>>;
using WarpGemmMmacI8I8I32_WT16x64x128_MR1NR4MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 4, 1, 1, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x128_MR1NR2MI1NI1_TRANSC_Preshuffle = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC_Shuffle<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x256_MR1NR2MI1NI1_TRANSC_Preshuffle = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC_Shuffle<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 8>>;

// int4
using WarpGemmMmacI4I4I32_WT16x16x64_MR1NR1MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 1, 1, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT16x64x64_MR1NR4MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 1, 4, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT16x64x64_MR1NR2MI1NI2 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 1, 2, 1, 2, 1>>;
using WarpGemmMmacI4I4I32_WT16x32x64_MR1NR2MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 1, 2, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT32x16x64_MR2NR1MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 2, 1, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT32x32x64_MR2NR2MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 2, 2, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT32x64x64_MR2NR4MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 2, 4, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT32x64x64_MR2NR2MI1NI2 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 2, 2, 1, 2, 1>>;

//fp8 fp8
using WarpGemmMmacfp8fp8f32_WT16x16x32_MR1NR1MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 1, 1, 1, 1>>;
using WarpGemmMmacfp8fp8f32_WT16x32x32_MR1NR2MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 2, 1, 1, 1>>;
using WarpGemmMmacfp8fp8f32_WT16x32x64_MR1NR2MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 2, 1, 1, 2>>;
using WarpGemmMmacfp8fp8f32_WT16x64x32_MR1NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 4, 1, 1, 1>>;
using WarpGemmMmacfp8fp8f32_WT16x64x64_MR1NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 4, 1, 1, 2>>;
using WarpGemmMmacfp8fp8f32_WT32x64x32_MR2NR1MI1NI4 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 2, 1, 1, 4, 1>>;
using WarpGemmMmacfp8fp8f32_WT32x64x32_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 2, 4, 1, 1, 1>>;
using WarpGemmMmacfp8fp8f32_WT32x64x64_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 2, 4, 1, 1, 2>>;

//fp8 fp8 scale
using WarpScaleGemmMmacfp8fp8f32_WT16x32x128_MR1NR2MI1NI1 = WarpFp8Bf8ScaleChannelGemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 2, 1, 1, 4>>;

//fp8 bf8
using WarpGemmMmacfp8bf8f32_WT32x64x32_MR2NR1MI1NI4 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_bf8, 2, 1, 1, 4, 1>>;
using WarpGemmMmacfp8bf8f32_WT32x64x32_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_bf8, 2, 4, 1, 1, 1>>;

//bf8 fp8
using WarpGemmMmacbf8fp8f32_WT32x64x32_MR2NR1MI1NI4 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_fp8, 2, 1, 1, 4, 1>>;
using WarpGemmMmacbf8fp8f32_WT32x64x32_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_fp8, 2, 4, 1, 1, 1>>;

//bf8 bf8
using WarpGemmMmacbf8bf8f32_WT32x64x32_MR2NR1MI1NI4 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_bf8, 2, 1, 1, 4, 1>>;
using WarpGemmMmacbf8bf8f32_WT32x64x32_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_bf8, 2, 4, 1, 1, 1>>;
using WarpGemmMmacbf8bf8f32_WT32x64x64_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_bf8, 2, 4, 1, 1, 2>>;

} // namespace ck_tile
