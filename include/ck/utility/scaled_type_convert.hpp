// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck/utility/type_convert.hpp"

namespace ck {

#ifndef CK_CODE_GEN_RTC

// Declare a template function for scaled conversion
template <typename Y, typename X>
__host__ __device__ constexpr Y scaled_type_convert(e8m0_bexp_t scale, X x)
{
    return type_convert<Y>(type_convert<float>(scale) * type_convert<float>(x));
}

template <>
inline __host__ __device__ constexpr float scaled_type_convert<float, float>(e8m0_bexp_t scale,
                                                                             float x)
{
    return type_convert<float>(scale) * x;
}

template <>
inline __host__ __device__ constexpr half_t scaled_type_convert<half_t, half_t>(
    e8m0_bexp_t scale, half_t x)
{
    return type_convert<half_t>(type_convert<float>(scale) * type_convert<float>(x));
}

template <>
inline __host__ __device__ constexpr bhalf_t scaled_type_convert<bhalf_t, bhalf_t>(
    e8m0_bexp_t scale, bhalf_t x)
{
    return type_convert<bhalf_t>(type_convert<float>(scale) * type_convert<float>(x));
}

#endif

} // namespace ck
