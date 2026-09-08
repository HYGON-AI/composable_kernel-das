// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Compiler architecture macros are only reliable in the HIP device pass. CMake supplies the
// corresponding CK_TILE_HCU_QUANT_GEMM_GFX* definition for host template instantiation.
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if defined(__gfx938__)
#define CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET 1
#else
#define CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET 0
#endif
#if defined(__gfx946__)
#define CK_TILE_HCU_QUANT_GEMM_GFX946_TARGET 1
#else
#define CK_TILE_HCU_QUANT_GEMM_GFX946_TARGET 0
#endif
#else
#if defined(CK_TILE_HCU_QUANT_GEMM_GFX938)
#define CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET 1
#else
#define CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET 0
#endif
#if defined(CK_TILE_HCU_QUANT_GEMM_GFX946)
#define CK_TILE_HCU_QUANT_GEMM_GFX946_TARGET 1
#else
#define CK_TILE_HCU_QUANT_GEMM_GFX946_TARGET 0
#endif
#endif

#define CK_TILE_HCU_QUANT_GEMM_TARGET \
    (CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET || CK_TILE_HCU_QUANT_GEMM_GFX946_TARGET)

#ifndef CK_TILE_ARCH_SUPPORTS_FP4
#define CK_TILE_ARCH_SUPPORTS_FP4 (!CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET)
#endif

#ifndef CK_TILE_ARCH_SUPPORTS_INT4
#define CK_TILE_ARCH_SUPPORTS_INT4 (!CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET)
#endif

#ifndef CK_TILE_ARCH_SUPPORTS_MXFP8
#define CK_TILE_ARCH_SUPPORTS_MXFP8 (!CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET)
#endif

#ifndef CK_TILE_ARCH_FORCE_OCP_FP8
#define CK_TILE_ARCH_FORCE_OCP_FP8 CK_TILE_HCU_QUANT_GEMM_TARGET
#endif

namespace ck_tile::hcu_quant_gemm {

constexpr bool is_gfx938_target() { return CK_TILE_HCU_QUANT_GEMM_GFX938_TARGET != 0; }
constexpr bool is_gfx946_target() { return CK_TILE_HCU_QUANT_GEMM_GFX946_TARGET != 0; }
constexpr bool is_quant_gemm_target() { return CK_TILE_HCU_QUANT_GEMM_TARGET != 0; }
constexpr bool arch_supports_fp4() { return CK_TILE_ARCH_SUPPORTS_FP4 != 0; }
constexpr bool arch_supports_int4() { return CK_TILE_ARCH_SUPPORTS_INT4 != 0; }
constexpr bool arch_supports_mxfp8() { return CK_TILE_ARCH_SUPPORTS_MXFP8 != 0; }
constexpr bool arch_uses_ocp_fp8() { return CK_TILE_ARCH_FORCE_OCP_FP8 != 0; }

} // namespace ck_tile::hcu_quant_gemm
