// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/f8_utils.hpp"
#include "ck/utility/mxf6_utils.hpp"
#include "ck/utility/random_gen.hpp"

#ifndef CK_USE_SR_F6_CONVERSION
#define CK_USE_SR_F6_CONVERSION 0
#endif

//WARNING: bf16 type convert in data_type.hpp, for now not included here.
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
namespace ck {

namespace utils {
namespace bf8_ocp_detail {

template <typename T, bool clip, bool stoch>
__host__ __device__ bf8_t run_cast_to_bf8(T x, uint32_t rng)
{
    constexpr bool is_half  = std::is_same<T, half_t>::value;
    constexpr bool is_float = std::is_same<T, float>::value;

    constexpr int bf8_exp  = 5;
    constexpr int bf8_mant = 2;

    constexpr int type_exp  = is_half ? 5 : 8;
    constexpr int type_mant = is_half ? 10 : 23;

    int exponent;
    uint32_t head, mantissa, sign;
    constexpr uint8_t qnan_code = 0x7D;
    constexpr uint8_t inf_code  = 0x7C;
    constexpr uint32_t nan_mask = is_half ? 0x7C00 : 0x7F800000;

    using T_bitwise = typename std::conditional<is_half, uint16_t, uint32_t>::type;
    T_bitwise x_bitwise = *(reinterpret_cast<T_bitwise*>(&x));

    if constexpr(is_float)
    {
        head     = x_bitwise & 0xFF800000;
        mantissa = x_bitwise & 0x7FFFFF;
        exponent = (head >> type_mant) & 0xFF;
        sign     = head >> (type_exp + type_mant);
    }
    else if constexpr(is_half)
    {
        head     = x_bitwise & 0xFC00;
        mantissa = x_bitwise & 0x3FF;
        exponent = (head >> type_mant) & 0x1F;
        sign     = head >> (type_exp + type_mant);
    }

    if((x_bitwise & nan_mask) == nan_mask)
    {
        return bf8_t(mantissa == 0 ? ((sign << (bf8_exp + bf8_mant)) | inf_code) : qnan_code);
    }

    if(x_bitwise == 0)
    {
        return bf8_t(0);
    }

    constexpr int max_exp = (1 << bf8_exp) - 2;
    constexpr int exp_low_cutoff = (1 << (type_exp - 1)) - (1 << (bf8_exp - 1));
    uint32_t drop_mask           = (1 << (type_mant - bf8_mant)) - 1;

    exponent -= exp_low_cutoff - 1;
    if(exponent <= 0)
    {
        drop_mask = (1 << (type_mant - bf8_mant + 1 - exponent)) - 1;
    }

    mantissa += 1 << type_mant;
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    if(mantissa >= (2 << type_mant))
    {
        mantissa >>= 1;
        exponent++;
    }
    mantissa >>= (type_mant - bf8_mant);

    if(exponent <= 0)
    {
        mantissa >>= 1 - exponent;
        exponent = 0;
    }
    else if(exponent > max_exp)
    {
        if(clip)
        {
            mantissa = (1 << bf8_mant) - 1;
            exponent = max_exp;
        }
        else
        {
            return bf8_t((sign << (bf8_exp + bf8_mant)) | inf_code);
        }
    }

    if(exponent == 0 && mantissa == 0)
    {
        return bf8_t(0);
    }

    mantissa &= (1 << bf8_mant) - 1;
    return bf8_t((sign << (bf8_exp + bf8_mant)) | (exponent << bf8_mant) | mantissa);
}

template <typename T>
__host__ __device__ T run_cast_from_bf8(bf8_t x)
{
    constexpr bool is_half  = std::is_same<T, half_t>::value;
    constexpr bool is_float = std::is_same<T, float>::value;

    constexpr int bf8_exp  = 5;
    constexpr int bf8_mant = 2;

    constexpr int type_exp  = is_half ? 5 : 8;
    constexpr int type_mant = is_half ? 10 : 23;

    constexpr uint8_t inf_exp = (1 << bf8_exp) - 1;
    T fNaN;
    if constexpr(is_half)
    {
        constexpr uint16_t ihNaN = 0x7C01;
        fNaN                     = *(reinterpret_cast<const half_t*>(&ihNaN));
    }
    else if constexpr(is_float)
    {
        constexpr uint32_t ifNaN = 0x7F800001;
        fNaN                     = *(reinterpret_cast<const float*>(&ifNaN));
    }

    const uint32_t raw = static_cast<uint8_t>(x);
    if(raw == 0)
    {
        return static_cast<T>(0);
    }

    uint32_t sign     = raw >> (bf8_exp + bf8_mant);
    uint32_t mantissa = raw & ((1 << bf8_mant) - 1);
    int exponent      = (raw & 0x7F) >> bf8_mant;

    if(exponent == 0 && mantissa == 0)
    {
        if constexpr(is_half)
        {
            constexpr uint16_t ihNeg0 = 0x8000;
            return sign ? *(reinterpret_cast<const half_t*>(&ihNeg0)) : static_cast<T>(0);
        }
        else if constexpr(is_float)
        {
            constexpr uint32_t ifNeg0 = 0x80000000;
            return sign ? *(reinterpret_cast<const float*>(&ifNeg0)) : static_cast<T>(0);
        }
    }

    if(exponent == inf_exp)
    {
        if(mantissa != 0)
            return fNaN;
        if constexpr(is_half)
        {
            constexpr uint16_t ihInf    = 0x7C00;
            constexpr uint16_t ihNegInf = 0xFC00;
            return *(reinterpret_cast<const half_t*>(sign ? &ihNegInf : &ihInf));
        }
        else if constexpr(is_float)
        {
            constexpr uint32_t ifInf    = 0x7F800000;
            constexpr uint32_t ifNegInf = 0xFF800000;
            return *(reinterpret_cast<const float*>(sign ? &ifNegInf : &ifInf));
        }
    }

    constexpr int exp_low_cutoff = (1 << (type_exp - 1)) - (1 << (bf8_exp - 1));
    typename std::conditional<is_half, uint16_t, uint32_t>::type retval;

    if(exponent == 0)
    {
        int sh = 1 + __builtin_clz(mantissa) - ((1 + type_exp + type_mant) - bf8_mant);
        mantissa <<= sh;
        mantissa &= ((1 << bf8_mant) - 1);
        exponent += 1 - sh;
    }

    exponent += exp_low_cutoff - 1;
    mantissa <<= type_mant - bf8_mant;

    if(exponent <= 0)
    {
        mantissa |= 1 << type_mant;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    retval = (sign << (type_exp + type_mant)) | (exponent << type_mant) | mantissa;
    return *(reinterpret_cast<const T*>(&retval));
}

template <typename T, bool clip, bool stoch>
__host__ __device__ bf8_t cast_to_bf8(T x, uint32_t rng)
{
    constexpr bool is_half  = std::is_same<T, half_t>::value;
    constexpr bool is_float = std::is_same<T, float>::value;
    static_assert(is_half || is_float, "Only half and float can be casted to bf8.");

    return run_cast_to_bf8<T, clip, stoch>(x, rng);
}

template <typename T>
__host__ __device__ T cast_from_bf8(bf8_t x)
{
    constexpr bool is_half  = std::is_same<T, half_t>::value;
    constexpr bool is_float = std::is_same<T, float>::value;
    static_assert(is_half || is_float, "Only half and float are supported.");

    return run_cast_from_bf8<T>(x);
}

} // namespace bf8_ocp_detail
} // namespace utils

// convert fp32 to fp8
template <>
inline __host__ __device__ fp8_t type_convert<fp8_t, float>(float x)
{
    constexpr bool negative_zero_nan = false;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::cast_to_f8<float, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
        x, rng);
}

// convert fp8 to fp32
template <>
inline __host__ __device__ float type_convert<float, fp8_t>(fp8_t x)
{
    constexpr bool negative_zero_nan = false;
    return utils::cast_from_f8<float, negative_zero_nan>(x);
}

// convert fp16 to fp8
template <>
inline __host__ __device__ fp8_t type_convert<fp8_t, half_t>(half_t x)
{
    constexpr bool negative_zero_nan = false;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::standard;
    constexpr uint32_t rng           = 0;
    return utils::cast_to_f8<half_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
        x, rng);
}

// convert fp8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, fp8_t>(fp8_t x)
{
    constexpr bool negative_zero_nan = false;
    return utils::cast_from_f8<half_t, negative_zero_nan>(x);
}

// convert fp32 to bf8
template <>
inline __host__ __device__ bf8_t type_convert<bf8_t, float>(float x)
{
    constexpr bool clip           = true;
    constexpr f8_rounding_mode rm = f8_rounding_mode::standard;
    constexpr uint32_t rng        = 0;
    return utils::bf8_ocp_detail::cast_to_bf8<float, clip, (rm == f8_rounding_mode::stochastic)>(
        x, rng);
}

// convert bf8 to fp32
template <>
inline __host__ __device__ float type_convert<float, bf8_t>(bf8_t x)
{
    return utils::bf8_ocp_detail::cast_from_bf8<float>(x);
}

// convert fp16 to bf8
template <>
inline __host__ __device__ bf8_t type_convert<bf8_t, half_t>(half_t x)
{
    constexpr bool clip           = true;
    constexpr f8_rounding_mode rm = f8_rounding_mode::standard;
    constexpr uint32_t rng        = 0;
    return utils::bf8_ocp_detail::cast_to_bf8<half_t, clip, (rm == f8_rounding_mode::stochastic)>(
        x, rng);
}

// convert bf8 to fp16
template <>
inline __host__ __device__ half_t type_convert<half_t, bf8_t>(bf8_t x)
{
    return utils::bf8_ocp_detail::cast_from_bf8<half_t>(x);
}

/**
 * @brief Converts a scalar value to f6_t with round-to-nearest-even.
 *
 * This HCU path currently supports the scalar f6_t surface only; packed/vector
 * F6 conversion remains a separate HCU audit item.
 */
template <typename X>
inline __host__ __device__ f6_t f6_convert_rne(X x, float scale = 1.0f)
{
    return utils::sat_convert_to_type<f6_t>(type_convert<float>(x) / scale);
}

/**
 * @brief Converts a scalar value to f6_t with stochastic rounding.
 */
template <typename X>
inline __host__ __device__ f6_t f6_convert_sr(X x, float scale = 1.0f)
{
    constexpr int seed = 42;
    uint32_t rng       = prand_generator<X, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::sat_convert_to_type_sr<f6_t>(type_convert<float>(x) / scale, rng);
}

/**
 * @brief Converts a scalar value to bf6_t with round-to-nearest-even.
 *
 * This HCU path currently supports the scalar bf6_t surface only; packed/vector
 * BF6 conversion remains a separate HCU audit item.
 */
template <typename X>
inline __host__ __device__ bf6_t bf6_convert_rne(X x, float scale = 1.0f)
{
    return utils::sat_convert_to_type<bf6_t>(type_convert<float>(x) / scale);
}

/**
 * @brief Converts a scalar value to bf6_t with stochastic rounding.
 */
template <typename X>
inline __host__ __device__ bf6_t bf6_convert_sr(X x, float scale = 1.0f)
{
    constexpr int seed = 42;
    uint32_t rng       = prand_generator<X, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::sat_convert_to_type_sr<bf6_t>(type_convert<float>(x) / scale, rng);
}

template <>
inline __host__ __device__ f6_t type_convert<f6_t, float>(float x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ float type_convert<float, f6_t>(f6_t x)
{
    return utils::to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
}

template <>
inline __host__ __device__ f6_t type_convert<f6_t, half_t>(half_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ half_t type_convert<half_t, f6_t>(f6_t x)
{
    return type_convert<half_t>(type_convert<float>(x));
}

template <>
inline __host__ __device__ f6_t type_convert<f6_t, bhalf_t>(bhalf_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return f6_convert_sr(x);
#else
    return f6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, f6_t>(f6_t x)
{
    return type_convert<bhalf_t>(type_convert<float>(x));
}

template <>
inline __host__ __device__ bf6_t type_convert<bf6_t, float>(float x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ float type_convert<float, bf6_t>(bf6_t x)
{
    return utils::to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), x);
}

template <>
inline __host__ __device__ bf6_t type_convert<bf6_t, half_t>(half_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ half_t type_convert<half_t, bf6_t>(bf6_t x)
{
    return type_convert<half_t>(type_convert<float>(x));
}

template <>
inline __host__ __device__ bf6_t type_convert<bf6_t, bhalf_t>(bhalf_t x)
{
#if CK_USE_SR_F6_CONVERSION
    return bf6_convert_sr(x);
#else
    return bf6_convert_rne(x);
#endif
}

template <>
inline __host__ __device__ bhalf_t type_convert<bhalf_t, bf6_t>(bf6_t x)
{
    return type_convert<bhalf_t>(type_convert<float>(x));
}

// Declare a template function for fp8 conversion using SR
template <typename Y, typename X>
__host__ __device__ constexpr Y f8_convert_sr(X x);

// convert fp32 to fp8 with stochastic rounding
template <>
inline __host__ __device__ fp8_t f8_convert_sr<fp8_t, float>(float x)
{
    constexpr bool negative_zero_nan = false;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 42;
    // as thread id is not available on host, use 0 for prn generation
    uint32_t rng = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::cast_to_f8<float, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
        x, rng);
}

// convert fp16 to fp8 with stochastic rounding
template <>
inline __host__ __device__ fp8_t f8_convert_sr<fp8_t, half_t>(half_t x)
{
    constexpr bool negative_zero_nan = false;
    constexpr bool clip              = true;
    constexpr f8_rounding_mode rm    = f8_rounding_mode::stochastic;
    constexpr int seed               = 42;
    // as thread id is not available on host, use 0 for prn generation
    uint32_t rng = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::cast_to_f8<half_t, negative_zero_nan, clip, (rm == f8_rounding_mode::stochastic)>(
        x, rng);
}

// convert fp32 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_t f8_convert_sr<bf8_t, float>(float x)
{
    constexpr bool clip           = true;
    constexpr f8_rounding_mode rm = f8_rounding_mode::stochastic;
    constexpr int seed            = 42;
    uint32_t rng                  = prand_generator<float, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::bf8_ocp_detail::cast_to_bf8<float, clip, (rm == f8_rounding_mode::stochastic)>(
        x, rng);
}

// convert fp16 to bf8 with stochastic rounding
template <>
inline __host__ __device__ bf8_t f8_convert_sr<bf8_t, half_t>(half_t x)
{
    constexpr bool clip           = true;
    constexpr f8_rounding_mode rm = f8_rounding_mode::stochastic;
    constexpr int seed            = 42;
    uint32_t rng                  = prand_generator<half_t, seed>(reinterpret_cast<uintptr_t>(&x), x);
    return utils::bf8_ocp_detail::cast_to_bf8<half_t, clip, (rm == f8_rounding_mode::stochastic)>(
        x, rng);
}

} // namespace ck
#endif
