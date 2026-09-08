// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
//
// Adapted for Hygon HCU: DPP8 instructions (v_dot2c_f32_f16_dpp, __builtin_amdgcn_mov_dpp8)
// are NOT available on HCU. HCU path falls back to regular inner_product().

#pragma once

#include "data_type.hpp"
#include "type_convert.hpp"
#include "inner_product.hpp"

namespace ck {

namespace dpp8 {

constexpr index_t lane_group_size = 8;

// DPP8 IntrinsicMaskDpp8 — kept for API compatibility, but unused on HCU
constexpr std::array<int, dpp8::lane_group_size> IntrinsicMaskDpp8 = {
    0,        // 0, 0, 0, 0, 0, 0, 0, 0
    2396745,  // 1, 1, 1, 1, 1, 1, 1, 1
    4793490,  // 2, 2, 2, 2, 2, 2, 2, 2
    7190235,  // 3, 3, 3, 3, 3, 3, 3, 3
    9586980,  // 4, 4, 4, 4, 4, 4, 4, 4
    11983725, // 5, 5, 5, 5, 5, 5, 5, 5
    14380470, // 6, 6, 6, 6, 6, 6, 6, 6
    16777215, // 7, 7, 7, 7, 7, 7, 7, 7
};

template <int SrcLaneIdx>
constexpr int get_dpp_sel_mask_broadcast()
{
    static_assert(SrcLaneIdx >= 0 && SrcLaneIdx < dpp8::lane_group_size,
                  "DPP8 src broadcast lane out of range <0, 7>.");
    return IntrinsicMaskDpp8[SrcLaneIdx];
}

// On AMD: v_dot2c_f32_f16_dpp inline asm with DPP8 lane broadcast
// On HCU: DPP8 not available — fall through to software inner_product
#if !defined(__gfx928__) && !defined(__gfx936__) && !defined(__gfx938__) && \
    !defined(__gfx92a__) && !defined(__gfx946__)

template <int SrcLaneIdx>
__device__ void inline_v_dot2c_dpp8_instr(const half2_t& a, const half2_t& b, float& c);

template <>
__device__ void inline_v_dot2c_dpp8_instr<0>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[0, 0, 0, 0, 0, 0, 0, 0]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<1>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[1, 1, 1, 1, 1, 1, 1, 1]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<2>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[2, 2, 2, 2, 2, 2, 2, 2]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<3>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[3, 3, 3, 3, 3, 3, 3, 3]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<4>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[4, 4, 4, 4, 4, 4, 4, 4]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<5>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[5, 5, 5, 5, 5, 5, 5, 5]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<6>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[6, 6, 6, 6, 6, 6, 6, 6]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<7>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[7, 7, 7, 7, 7, 7, 7, 7]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}

template <int SrcLaneIdx, bool ShareA>
__device__ void inline_v_dot2c_dpp8(const half2_t& a, const half2_t& b, float& c)
{
    static_assert(SrcLaneIdx >= 0 && SrcLaneIdx < dpp8::lane_group_size,
                  "DPP8 src broadcast lane out of range <0, 7>.");
    if constexpr(ShareA)
    {
        inline_v_dot2c_dpp8_instr<SrcLaneIdx>(a, b, c);
    }
    else
    {
        inline_v_dot2c_dpp8_instr<SrcLaneIdx>(b, a, c);
    }
}

template <int SrcLaneIdx>
__device__ void intrinsic_fdot2_impl(const half2_t& a, const half2_t& b, float& c)
{
    constexpr int sel_mask = get_dpp_sel_mask_broadcast<SrcLaneIdx>();
    const half2_t val_from_other_lane =
        bit_cast<half2_t>(__builtin_amdgcn_mov_dpp8(bit_cast<int>(a), sel_mask));
    inner_product(val_from_other_lane, b, c);
}

template <int SrcLaneIdx, bool ShareA>
__device__ void intrinsic_fdot2(const half2_t& a, const half2_t& b, float& c)
{
    if constexpr(ShareA)
    {
        intrinsic_fdot2_impl<SrcLaneIdx>(a, b, c);
    }
    else
    {
        intrinsic_fdot2_impl<SrcLaneIdx>(b, a, c);
    }
}

#endif // !HCU: AMD DPP path

// Common entry point: inner_product_dpp
// On AMD: uses DPP8 lane-group broadcast for dot product
// On HCU: falls back to regular inner_product() without lane sharing
template <typename TA, typename TB, typename TC, int SrcLaneIdx, bool ShareA>
__device__ void inner_product_dpp(const TA& a, const TB& b, TC& c)
{
#if defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || defined(__gfx938__) || \
    defined(__gfx946__)
    // HCU: DPP8 not available, use regular inner_product (no lane sharing optimization)
    inner_product(a, b, c);
#else
#if CK_USE_AMD_V_DOT_DPP8_INLINE_ASM
    inline_v_dot2c_dpp8<SrcLaneIdx, ShareA>(a, b, c);
#else
    intrinsic_fdot2<SrcLaneIdx, ShareA>(a, b, c);
#endif
#endif
}

} // namespace dpp8

} // namespace ck
