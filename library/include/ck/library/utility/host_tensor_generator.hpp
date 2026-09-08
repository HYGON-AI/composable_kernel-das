// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cmath>
#include <numeric>
#include <random>

#include "ck/ck.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "ck/utility/data_type.hpp"

// host 侧 half 测试数据需要稳定的 fp16 bits；这里显式编码，避免 CK example
// 的 host 生成/校验路径与 device 看到的 half payload 不一致。
inline ck::half_t host_float_to_half(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    int32_t exp         = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant       = bits & 0x7fffff;

    uint16_t half_bits = sign;
    if(exp <= 0)
    {
        if(exp >= -10)
        {
            mant = mant | 0x800000;
            const uint32_t shift = static_cast<uint32_t>(14 - exp);
            half_bits |= static_cast<uint16_t>((mant + (1u << (shift - 1))) >> shift);
        }
    }
    else if(exp >= 31)
    {
        half_bits |= static_cast<uint16_t>(0x7c00 | (mant ? 1 : 0));
    }
    else
    {
        mant += 0x1000;
        if(mant & 0x800000)
        {
            mant = 0;
            ++exp;
        }

        if(exp >= 31)
            half_bits |= 0x7c00;
        else
            half_bits |= static_cast<uint16_t>((exp << 10) | (mant >> 13));
    }

    ck::half_t out;
    std::memcpy(&out, &half_bits, sizeof(half_bits));
    return out;
}

template <typename T>
struct GeneratorTensor_0
{
    template <typename... Is>
    T operator()(Is...)
    {
        return T{0};
    }
};

template <typename T>
struct GeneratorTensor_1
{
    T value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        return value;
    }
};

template <>
struct GeneratorTensor_1<ck::half_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::half_t operator()(Is...)
    {
        return host_float_to_half(value);
    }
};

template <>
struct GeneratorTensor_1<ck::bhalf_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        return ck::type_convert<ck::bhalf_t>(value);
    }
};

template <>
struct GeneratorTensor_1<int8_t>
{
    int8_t value = 1;

    template <typename... Is>
    int8_t operator()(Is...)
    {
        return value;
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
template <>
struct GeneratorTensor_1<ck::pk_i4_t>
{
    int8_t value = 1;

    template <typename... Is>
    ck::pk_i4_t operator()(Is...)
    {
        int t = value + 8;
        return ck::pk_i4_t{static_cast<ck::pk_i4_t::type>(((t << 4) + t) & 0xff)};
    }
};
#endif

#ifndef CK_CODE_GEN_RTC
template <>
struct GeneratorTensor_1<ck::e8m0_bexp_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::e8m0_bexp_t operator()(Is...)
    {
        return ck::e8m0_bexp_t{value};
    }
};

template <>
struct GeneratorTensor_1<ck::e4m3_scale_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::e4m3_scale_t operator()(Is...)
    {
        return ck::e4m3_scale_t{value};
    }
};

template <>
struct GeneratorTensor_1<ck::e5m3_scale_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::e5m3_scale_t operator()(Is...)
    {
        return ck::e5m3_scale_t{value};
    }
};
#endif

template <typename T>
struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        return static_cast<T>((std::rand() % (max_value - min_value)) + min_value);
    }
};

template <>
struct GeneratorTensor_2<ck::half_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::half_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return host_float_to_half(tmp);
    }
};

template <>
struct GeneratorTensor_2<ck::bhalf_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ck::bhalf_t>(tmp);
    }
};

template <>
struct GeneratorTensor_2<int8_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    int8_t operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
template <>
struct GeneratorTensor_2<ck::pk_i4_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::pk_i4_t operator()(Is...)
    {
        int hi = std::rand() % (max_value - min_value) + min_value + 8;
        int lo = std::rand() % (max_value - min_value) + min_value + 8;
        return ck::pk_i4_t{static_cast<ck::pk_i4_t::type>(((hi & 0xf) << 4) + (lo & 0xf))};
    }
};
#endif

#ifndef CK_CODE_GEN_RTC
template <>
struct GeneratorTensor_2<ck::e8m0_bexp_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::e8m0_bexp_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::e8m0_bexp_t{tmp};
    }
};

template <>
struct GeneratorTensor_2<ck::e4m3_scale_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::e4m3_scale_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::e4m3_scale_t{tmp};
    }
};

template <>
struct GeneratorTensor_2<ck::e5m3_scale_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::e5m3_scale_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::e5m3_scale_t{tmp};
    }
};
#endif

template <typename T>
struct GeneratorTensor_3
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        return static_cast<T>(min_value + tmp * (max_value - min_value));
    }
};

template <>
struct GeneratorTensor_3<ck::half_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::half_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return host_float_to_half(fp32_tmp);
    }
};

template <>
struct GeneratorTensor_3<ck::bhalf_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ck::bhalf_t>(fp32_tmp);
    }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
template <>
struct GeneratorTensor_3<ck::pk_i4_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::pk_i4_t operator()(Is...)
    {
        int hi = std::rand() % (max_value - min_value) + min_value + 8;
        int lo = std::rand() % (max_value - min_value) + min_value + 8;
        return ck::pk_i4_t{static_cast<ck::pk_i4_t::type>(((hi & 0xf) << 4) + (lo & 0xf))};
    }
};
#endif

#ifndef CK_CODE_GEN_RTC
template <>
struct GeneratorTensor_3<ck::e8m0_bexp_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::e8m0_bexp_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);
        return ck::e8m0_bexp_t{min_value + tmp * (max_value - min_value)};
    }
};

template <>
struct GeneratorTensor_3<ck::e4m3_scale_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::e4m3_scale_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);
        return ck::e4m3_scale_t{min_value + tmp * (max_value - min_value)};
    }
};

template <>
struct GeneratorTensor_3<ck::e5m3_scale_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::e5m3_scale_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);
        return ck::e5m3_scale_t{min_value + tmp * (max_value - min_value)};
    }
};
#endif

template <typename T>
struct GeneratorTensor_4
{
    std::mt19937 generator;
    std::normal_distribution<float> distribution;

    GeneratorTensor_4(float mean, float stddev, unsigned int seed = 1)
        : generator(seed), distribution(mean, stddev){};

    template <typename... Is>
    T operator()(Is...)
    {
        float tmp = distribution(generator);

        return ck::type_convert<T>(tmp);
    }
};

template <>
struct GeneratorTensor_4<ck::half_t>
{
    std::mt19937 generator;
    std::normal_distribution<float> distribution;

    GeneratorTensor_4(float mean, float stddev, unsigned int seed = 1)
        : generator(seed), distribution(mean, stddev){};

    template <typename... Is>
    ck::half_t operator()(Is...)
    {
        float tmp = distribution(generator);

        return host_float_to_half(tmp);
    }
};

struct GeneratorTensor_Checkboard
{
    template <typename... Ts>
    float operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {static_cast<ck::index_t>(Xs)...};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](bool init, ck::index_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

/**
 * @brief Is used to generate sequential values based on the specified dimension.
 *
 * @tparam Dim The specific dimension used for generation.
 *
 * GeneratorTensor_Sequential<1>{} will generate the following values for a 3x3 tensor:
 *
 * 0 1 2
 * 0 1 2
 * 0 1 2
 *
 * Essentially, the values generated are logical coordinates of the generated element that
 * correspond to dimension Dim. E.g. for 2-dimensional tensor and Dim=1, the values are the column
 * indices.
 *
 */
template <ck::index_t Dim>
struct GeneratorTensor_Sequential
{
    template <typename... Ts>
    float operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        return dims[Dim];
    }
};

template <typename T, size_t NumEffectiveDim = 2>
struct GeneratorTensor_Diagonal
{
    T value{1};

    template <typename... Ts>
    T operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        size_t start_dim                            = dims.size() - NumEffectiveDim;
        bool pred                                   = true;
        for(size_t i = start_dim + 1; i < dims.size(); i++)
        {
            pred &= (dims[start_dim] == dims[i]);
        }
        return pred ? value : T{0};
    }
};
