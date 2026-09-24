// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core.hpp"

namespace ck_tile {

// GDN's FP32 accumulators must retain STANDARD BF16 rounding. Select in
// registers so every scalar conversion does not split/rejoin the wave's EXEC
// mask. Preserve the standard converter's NaN sign, payload and sticky bit.
CK_TILE_DEVICE inline uint16_t gdn_float_to_bf16_rne(float value)
{
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx936__) || defined(__gfx938__))
    const uint32_t bits = bit_cast<uint32_t>(value);
    const uint32_t rounded = bits + 0x7fffU + ((bits >> 16) & 1U);
    const uint32_t special = bits | ((bits & 0xffffU) != 0 ? 0x10000U : 0U);
    uint32_t result;
    // RNE's integer bias leaves either infinity unchanged after truncating
    // the low word. Only NaNs need payload/sticky selection.
    asm("v_cmp_u_f32 vcc, %1, %1\n"
        "v_cndmask_b32 %0, %2, %3, vcc\n"
        : "=v"(result)
        : "v"(value), "v"(rounded), "v"(special)
        : "vcc");
    return static_cast<uint16_t>(result >> 16);
#else
    return float_to_bf16_rtn_raw(value);
#endif
}

template <typename Dst, typename Src>
CK_TILE_DEVICE Dst gdn_type_convert(Src value)
{
#if (defined(__gfx936__) || defined(__gfx938__)) && CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT == 0
    if constexpr(std::is_same_v<Dst, bf16_t> && std::is_same_v<Src, float>)
    {
        // Keep literal zero/padding constants foldable; inline assembly must
        // only be used for runtime accumulators.
        if(__builtin_constant_p(value))
            return type_convert<Dst>(value);
        return bit_cast<bf16_t>(gdn_float_to_bf16_rne(value));
    }
    else
#endif
        return type_convert<Dst>(value);
}

template <typename Dst, typename Tensor>
CK_TILE_DEVICE auto gdn_cast_tile(const Tensor& src)
{
    if constexpr(std::is_same_v<Dst, bf16_t> &&
                 std::is_same_v<typename Tensor::DataType, float>)
        return tile_elementwise_in([](float x) { return gdn_type_convert<Dst>(x); }, src);
    else
        return cast_tile<Dst>(src);
}
} // namespace ck_tile
