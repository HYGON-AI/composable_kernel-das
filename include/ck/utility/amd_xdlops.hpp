// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#ifndef CK_AMD_XDLOPS_HPP
#define CK_AMD_XDLOPS_HPP

#include "ck/ck.hpp"
#include "data_type.hpp"
#include "hcu_mmac.hpp"

namespace ck {

#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__)
__device__ inline int32x2_t intrin_mmac_repack_16x16_input_x2(int32_t packed_input)
{
    // Small-K AMD MFMA distributes K fragments across four 16-lane groups. Merge adjacent
    // source groups into HCU groups 0/1 and zero groups 2/3 for the doubled-K MMAC.
    const index_t lane_id          = __lane_id();
    const index_t mmac_group       = lane_id / 16;
    const index_t lane_in_group    = lane_id % 16;
    const index_t first_src_group  = (mmac_group % 2) * 2;
    const index_t first_src_lane   = first_src_group * 16 + lane_in_group;
    const index_t second_src_lane  = first_src_lane + 16;
    const int32x2_t gathered_input = {
        __builtin_amdgcn_ds_bpermute(first_src_lane << 2, packed_input),
        __builtin_amdgcn_ds_bpermute(second_src_lane << 2, packed_input)};

    return mmac_group < 2 ? gathered_input : int32x2_t{0, 0};
}
#endif

// Instantiated only when an unsupported MFMA size is compiled for HCU.
template <bool Supported = false, typename... Ts>
__device__ inline void intrin_mfma_unsupported_on_hcu(Ts&&...)
{
    static_assert(Supported,
                  "this MFMA size is not supported on HCU (no native instruction and cannot be "
                  "emulated); use a 16x16 MMAC size");
}

// fp32
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x1f32;

template <>
struct intrin_mfma_f32_32x32x1f32<64, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
        reg_c.template AsType<float32_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<1>{}], 1, 1, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <>
struct intrin_mfma_f32_32x32x1f32<32, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x1f32(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x2f32;

template <>
struct intrin_mfma_f32_32x32x2f32<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x2f32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x4f32;

template <>
struct intrin_mfma_f32_16x16x4f32<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
#if defined(__gfx926__) || defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || \
    defined(__gfx938__)
        intrin_mmac_f32_16x16x4f32(reg_a, reg_b, reg_c);
#elif defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x4f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x1f32;

template <>
struct intrin_mfma_f32_16x16x1f32<16, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x1f32(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 2, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x1f32;

template <>
struct intrin_mfma_f32_4x4x1f32<4, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <>
struct intrin_mfma_f32_4x4x1f32<8, 64>
{
    template <class FloatC>
    __device__ static void Run(const float& reg_a, const float& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
        reg_c.template AsType<float4_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_4x4x1f32(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<1>{}], 4, 1, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

// fp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4f16;

template <>
struct intrin_mfma_f32_32x32x4f16<64, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
        reg_c.template AsType<float32_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<1>{}], 1, 1, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <>
struct intrin_mfma_f32_32x32x4f16<32, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float32_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4f16(
            reg_a, reg_b, reg_c.template AsType<float32_t>()[Number<0>{}], 1, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x8f16;

template <>
struct intrin_mfma_f32_32x32x8f16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x8f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x16f16;

template <>
struct intrin_mfma_f32_16x16x16f16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__)
        intrin_mmac_f32_16x16x16f16(reg_a, reg_b, reg_c);
#elif defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x16f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x4f16;

template <>
struct intrin_mfma_f32_16x16x4f16<16, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x4f16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 2, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_4x4x4f16;

template <>
struct intrin_mfma_f32_4x4x4f16<4, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <>
struct intrin_mfma_f32_4x4x4f16<8, 64>
{
    template <class FloatC>
    __device__ static void Run(const half4_t& reg_a, const half4_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 4, 0, 0);
        reg_c.template AsType<float4_t>()(Number<1>{}) = __builtin_amdgcn_mfma_f32_4x4x4f16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<1>{}], 4, 1, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

// bfp16
template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x8bf16_1k;

template <>
struct intrin_mfma_f32_32x32x8bf16_1k<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_a, const bhalf4_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x16bf16_1k;

template <>
struct intrin_mfma_f32_16x16x16bf16_1k<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf4_t& reg_a, const bhalf4_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__) || \
    defined(__gfx946__)
        intrin_mmac_f32_16x16x16bf16(
            bit_cast<int16x4_t>(reg_a), bit_cast<int16x4_t>(reg_b), reg_c);
#elif defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_32x32x4bf16;

template <>
struct intrin_mfma_f32_32x32x4bf16<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_a, const bhalf2_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float16_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_32x32x4bf16(
            reg_a, reg_b, reg_c.template AsType<float16_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f32_16x16x8bf16;

template <>
struct intrin_mfma_f32_16x16x8bf16<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const bhalf2_t& reg_a, const bhalf2_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__)
        intrin_mmac_f32_16x16x16bf16(
            bit_cast<int16x4_t>(intrin_mmac_repack_16x16_input_x2(bit_cast<int32_t>(reg_a))),
            bit_cast<int16x4_t>(intrin_mmac_repack_16x16_input_x2(bit_cast<int32_t>(reg_b))),
            reg_c);
#elif defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<float4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f32_16x16x8bf16(
            reg_a, reg_b, reg_c.template AsType<float4_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_32x32x8i8;

template <>
struct intrin_mfma_i32_32x32x8i8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_a, const int8x4_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<int32x16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_32x32x8i8(bit_cast<int32_t>(reg_a),
                                                bit_cast<int32_t>(reg_b),
                                                reg_c.template AsType<int32x16_t>()[Number<0>{}],
                                                0,
                                                0,
                                                0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_16x16x16i8;

template <>
struct intrin_mfma_i32_16x16x16i8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const int8x4_t& reg_a, const int8x4_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__)
        intrin_mmac_i32_16x16x32i8(
            intrin_mmac_repack_16x16_input_x2(bit_cast<int32_t>(reg_a)),
            intrin_mmac_repack_16x16_input_x2(bit_cast<int32_t>(reg_b)),
            reg_c);
#elif defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_16x16x16i8(bit_cast<int32_t>(reg_a),
                                                 bit_cast<int32_t>(reg_b),
                                                 reg_c.template AsType<int32x4_t>()[Number<0>{}],
                                                 0,
                                                 0,
                                                 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_32x32x16i8;

template <>
struct intrin_mfma_i32_32x32x16i8<32, 32>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<int32x16_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_32x32x16_i8(bit_cast<int64_t>(reg_a),
                                                  bit_cast<int64_t>(reg_b),
                                                  reg_c.template AsType<int32x16_t>()[Number<0>{}],
                                                  0,
                                                  0,
                                                  0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_i32_16x16x32i8;

template <>
struct intrin_mfma_i32_16x16x32i8<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const int8x8_t& reg_a, const int8x8_t& reg_b, FloatC& reg_c)
    {
#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__)
        intrin_mmac_i32_16x16x32i8(bit_cast<int32x2_t>(reg_a),
                                   bit_cast<int32x2_t>(reg_b),
                                   reg_c);
#elif defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__HIP_DEVICE_COMPILE__)
        reg_c.template AsType<int32x4_t>()(Number<0>{}) =
            __builtin_amdgcn_mfma_i32_16x16x32i8(bit_cast<int64_t>(reg_a),
                                                 bit_cast<int64_t>(reg_b),
                                                 reg_c.template AsType<int32x4_t>()[Number<0>{}],
                                                 0,
                                                 0,
                                                 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};

template <index_t MPerWave, index_t NPerWave>
struct intrin_mfma_f64_16x16x4f64;

template <>
struct intrin_mfma_f64_16x16x4f64<16, 16>
{
    template <class FloatC>
    __device__ static void Run(const double& reg_a, const double& reg_b, FloatC& reg_c)
    {
#if defined(__gfx936__) || defined(__gfx938__)
        intrin_mmac_16x16x4f64(reg_a, reg_b, reg_c);
#elif defined(CK_ARCH_HCU)
        intrin_mfma_unsupported_on_hcu(reg_a, reg_b, reg_c);
#elif defined(__gfx90a__) || defined(__gfx940__)
        reg_c.template AsType<double4_t>()(Number<0>{}) = __builtin_amdgcn_mfma_f64_16x16x4f64(
            reg_a, reg_b, reg_c.template AsType<double4_t>()[Number<0>{}], 0, 0, 0);
#else
        swallow(reg_a, reg_b, reg_c);
#endif
    }
};
} // namespace ck
#endif
