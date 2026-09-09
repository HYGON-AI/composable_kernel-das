// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck/utility/e4m3.hpp"
#include "ck/utility/e5m3.hpp"
#include "ck/utility/e8m0.hpp"
#include "ck/utility/statically_indexed_array.hpp"

namespace ck {

using tf32_t  = uint32_t;
using bhalf_t = ushort;
using half_t  = _Float16;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
using fp8_t   = _BitInt(8);
using bf8_t   = unsigned _BitInt(8);
using f8_t    = fp8_t;
using f8_ocp_t = fp8_t;
using int4_t  = _BitInt(4);
using uint4_t = unsigned _BitInt(4);
using f6_t    = _BitInt(6);
using bf6_t   = unsigned _BitInt(6);

struct pk_i4_t
{
    using type = int8_t;
    type data;
    __host__ __device__ constexpr pk_i4_t() : data{type{}} {}
    __host__ __device__ constexpr pk_i4_t(type init) : data{init} {}
};
#endif

template <typename T>
struct packed_type_info
{
    using element_type = remove_cvref_t<T>;
    static constexpr index_t packed_size = 1;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
template <>
struct packed_type_info<pk_i4_t>
{
    using element_type = int4_t;
    static constexpr index_t packed_size = 2;
};
#endif

template <typename T>
using element_type_t = typename packed_type_info<remove_cvref_t<T>>::element_type;

template <typename T>
inline constexpr index_t packed_size_v = packed_type_info<remove_cvref_t<T>>::packed_size;

template <typename T>
inline constexpr bool is_packed_type_v = packed_size_v<remove_cvref_t<T>> > 1;

// vector_type
template <typename T, index_t N>
struct vector_type;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<T __attribute__((ext_vector_type(V))), N>;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<vector_type<T, V>, N>;

// vector_type_maker
// This is the right way to handle "vector of vectors": making a bigger vector instead
template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<T __attribute__((ext_vector_type(N1))), N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
__host__ __device__ constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

// scalar_type
template <typename TV>
struct scalar_type;

// is_scalar_type
template <typename TV>
struct is_scalar_type
{
    static constexpr bool value = (scalar_type<remove_cvref_t<TV>>::vector_size == 1);
};

// has_same_scalar_type
template <typename X, typename Y>
using has_same_scalar_type = is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                     typename scalar_type<remove_cvref_t<Y>>::type>;

template <typename T, index_t N>
struct scalar_type<T __attribute__((ext_vector_type(N)))>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

//
template <>
struct scalar_type<double>
{
    using type                           = double;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<float>
{
    using type                           = float;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<tf32_t>
{
    using type                           = tf32_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<half_t>
{
    using type                           = half_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bhalf_t>
{
    using type                           = bhalf_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int32_t>
{
    using type                           = int32_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int8_t>
{
    using type                           = int8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<uint8_t>
{
    using type                           = uint8_t;
    static constexpr index_t vector_size = 1;
};

#ifndef CK_CODE_GEN_RTC
template <>
struct scalar_type<e8m0_bexp_t>
{
    using type                           = typename e8m0_bexp_t::type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<e4m3_scale_t>
{
    using type                           = e4m3_scale_t::type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<e5m3_scale_t>
{
    using type                           = e5m3_scale_t::type;
    static constexpr index_t vector_size = 1;
};
#endif

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
template <>
struct scalar_type<fp8_t>
{
    using type                           = fp8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_t>
{
    using type                           = bf8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int4_t>
{
    using type                           = int4_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<uint4_t>
{
    using type                           = uint4_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f6_t>
{
    using type                           = f6_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf6_t>
{
    using type                           = bf6_t;
    static constexpr index_t vector_size = 1;
};
#endif

//
template <typename T>
struct vector_type<T, 1>
{
    using d1_t = T;
    using type = d1_t;

    union
    {
        T d1_;
        StaticallyIndexedArray<T, 1> d1x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value, "wrong!");

        return data_.d1x1_;
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value, "wrong!");

        return data_.d1x1_;
    }
};

template <typename T>
struct vector_type<T, 2>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));

    using type = d2_t;

    union
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 4>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));

    using type = d4_t;

    union
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 8>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));

    using type = d8_t;

    union
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 16>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));

    using type = d16_t;

    union
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 32>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));

    using type = d32_t;

    union
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 64>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));

    using type = d64_t;

    union
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 128>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));

    using type = d128_t;

    union
    {
        d128_t d128_;
        StaticallyIndexedArray<d1_t, 128> d1x128_;
        StaticallyIndexedArray<d2_t, 64> d2x64_;
        StaticallyIndexedArray<d4_t, 32> d4x32_;
        StaticallyIndexedArray<d8_t, 16> d8x16_;
        StaticallyIndexedArray<d16_t, 8> d16x8_;
        StaticallyIndexedArray<d32_t, 4> d32x4_;
        StaticallyIndexedArray<d64_t, 2> d64x2_;
        StaticallyIndexedArray<d128_t, 1> d128x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
    }
};

template <typename T>
struct vector_type<T, 256>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));
    typedef T d256_t __attribute__((ext_vector_type(256)));

    using type = d256_t;

    union
    {
        d256_t d256_;
        StaticallyIndexedArray<d1_t, 256> d1x256_;
        StaticallyIndexedArray<d2_t, 128> d2x128_;
        StaticallyIndexedArray<d4_t, 64> d4x64_;
        StaticallyIndexedArray<d8_t, 32> d8x32_;
        StaticallyIndexedArray<d16_t, 16> d16x16_;
        StaticallyIndexedArray<d32_t, 8> d32x8_;
        StaticallyIndexedArray<d64_t, 4> d64x4_;
        StaticallyIndexedArray<d128_t, 2> d128x2_;
        StaticallyIndexedArray<d256_t, 1> d256x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
    }
};

// storage type traits
template <typename T>
struct storage_type
{
    using type = T;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
template <>
struct storage_type<int4_t>
{
    using type = uint8_t;
};

template <>
struct storage_type<uint4_t>
{
    using type = uint8_t;
};
#endif

using int64_t = long;

// fp64
using double2_t  = typename vector_type<double, 2>::type;
using double4_t  = typename vector_type<double, 4>::type;
using double8_t  = typename vector_type<double, 8>::type;
using double16_t = typename vector_type<double, 16>::type;
using double32_t = typename vector_type<double, 32>::type;
using double64_t = typename vector_type<double, 64>::type;

// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;
using half64_t = typename vector_type<half_t, 64>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;
using bhalf64_t = typename vector_type<bhalf_t, 64>::type;

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
// fp8
using fp8x2_t  = typename vector_type<fp8_t, 2>::type;
using fp8x4_t  = typename vector_type<fp8_t, 4>::type;
using fp8x8_t  = typename vector_type<fp8_t, 8>::type;
using fp8x16_t = typename vector_type<fp8_t, 16>::type;

// bf8
using bf8x2_t  = typename vector_type<bf8_t, 2>::type;
using bf8x4_t  = typename vector_type<bf8_t, 4>::type;
using bf8x8_t  = typename vector_type<bf8_t, 8>::type;
using bf8x16_t = typename vector_type<bf8_t, 16>::type;
#endif

// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

// i16
using int16x2_t = typename vector_type<int16_t, 2>::type;
using int16x4_t = typename vector_type<int16_t, 4>::type;
using int16x8_t = typename vector_type<int16_t, 8>::type;

// short
using short8_t = typename vector_type<short, 8>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

// u8
using uint8x2_t  = typename vector_type<uint8_t, 2>::type;
using uint8x4_t  = typename vector_type<uint8_t, 4>::type;
using uint8x8_t  = typename vector_type<uint8_t, 8>::type;
using uint8x16_t = typename vector_type<uint8_t, 16>::type;

// Convert X to Y
template <typename Y, typename X>
__host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}

// convert bfp16 to fp32
template <>
inline __host__ __device__ constexpr float type_convert<float, bhalf_t>(bhalf_t x)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(x) << 16};

    return u.fp32;
}

// convert fp32 to bfp16
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, float>(float x)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {x};

    // When the exponent bits are not all 1s, then the value is zero, normal,
    // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
    // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
    // This causes the bfloat16's mantissa to be incremented by 1 if the 16
    // least significant bits of the float mantissa are greater than 0x8000,
    // or if they are equal to 0x8000 and the least significant bit of the
    // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
    // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
    // has the value 0x7f, then incrementing it causes it to become 0x00 and
    // the exponent is incremented by one, which is the next higher FP value
    // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
    // with an exponent of 0x00 and a mantissa of 0x7f, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    bool flag0 = ~u.int32 & 0x7f800000;

    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bfloat16's mantissa bits are all 0.
    bool flag1 = !flag0 && (u.int32 & 0xffff);

    u.int32 += flag0 ? 0x7fff + ((u.int32 >> 16) & 1) : 0; // Round to nearest, round to even
    u.int32 |= flag1 ? 0x10000 : 0x0;                      // Preserve signaling NaN

    return uint16_t(u.int32 >> 16);
}

// convert fp16 to fp32
template <>
inline __host__ __device__ constexpr float type_convert<float, half_t>(half_t x)
{
#ifdef __HIP_DEVICE_COMPILE__
    return static_cast<float>(x);
#else
    union
    {
        half_t fp16;
        uint16_t int16;
    } in = {x};

    const uint32_t sign = static_cast<uint32_t>(in.int16 & 0x8000) << 16;
    uint32_t exp        = (in.int16 >> 10) & 0x1f;
    uint32_t mant       = in.int16 & 0x03ff;
    uint32_t bits       = 0;

    if(exp == 0)
    {
        if(mant == 0)
        {
            bits = sign;
        }
        else
        {
            exp = 127 - 15 + 1;
            while((mant & 0x0400) == 0)
            {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    }
    else if(exp == 0x1f)
    {
        bits = sign | 0x7f800000 | (mant << 13);
    }
    else
    {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    union
    {
        uint32_t int32;
        float fp32;
    } out = {bits};

    return out.fp32;
#endif
}

// convert fp32 to fp16
template <>
inline __host__ __device__ constexpr half_t type_convert<half_t, float>(float x)
{
#ifdef __HIP_DEVICE_COMPILE__
    return static_cast<half_t>(x);
#else
    union
    {
        float fp32;
        uint32_t int32;
    } in = {x};

    const uint32_t sign = (in.int32 >> 16) & 0x8000;
    const uint32_t fp32_exp = (in.int32 >> 23) & 0xff;
    int32_t exp             = static_cast<int32_t>(fp32_exp) - 127 + 15;
    uint32_t mant           = in.int32 & 0x7fffff;
    uint16_t bits           = 0;

    if(fp32_exp == 0xff)
    {
        bits = static_cast<uint16_t>(sign | 0x7c00 | (mant ? 0x0200 : 0));
    }
    else if(exp <= 0)
    {
        if(exp < -10)
        {
            bits = static_cast<uint16_t>(sign);
        }
        else
        {
            mant = (mant | 0x800000) >> (1 - exp);
            bits = static_cast<uint16_t>(sign | ((mant + 0x1000) >> 13));
        }
    }
    else if(exp >= 0x1f)
    {
        bits = static_cast<uint16_t>(sign | 0x7c00);
    }
    else
    {
        uint32_t half_mant = (mant + 0x1000) >> 13;

        if(half_mant == 0x400)
        {
            half_mant = 0;
            ++exp;
        }

        bits = exp >= 0x1f ? static_cast<uint16_t>(sign | 0x7c00)
                           : static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                                   half_mant);
    }

    union
    {
        uint16_t int16;
        half_t fp16;
    } out = {bits};

    return out.fp16;
#endif
}

// convert bfp16 to fp16 via fp32
template <>
inline __host__ __device__ constexpr half_t type_convert<half_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return type_convert<half_t>(x_fp32);
}

// convert fp16 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, half_t>(half_t x)
{
    float x_fp32 = type_convert<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

// convert bfp16 to int32 via fp32
template <>
inline __host__ __device__ constexpr int32_t type_convert<int32_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return static_cast<int32_t>(x_fp32);
}

// convert int32 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, int32_t>(int32_t x)
{
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

// convert bfp16 to int8 via fp32
template <>
inline __host__ __device__ constexpr int8_t type_convert<int8_t, bhalf_t>(bhalf_t x)
{
    float x_fp32 = type_convert<float>(x);

    return static_cast<int8_t>(x_fp32);
}

// convert int8 to bfp16 via fp32
template <>
inline __host__ __device__ constexpr bhalf_t type_convert<bhalf_t, int8_t>(int8_t x)
{
    float x_fp32 = static_cast<float>(x);

    return type_convert<bhalf_t>(x_fp32);
}

template <typename T>
struct NumericLimits
{
    __host__ __device__ static constexpr T Min() { return std::numeric_limits<T>::min(); }

    __host__ __device__ static constexpr T Max() { return std::numeric_limits<T>::max(); }

    __host__ __device__ static constexpr T Lowest() { return std::numeric_limits<T>::lowest(); }

    __host__ __device__ static constexpr T QuietNaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

    __host__ __device__ static constexpr T Infinity() { return std::numeric_limits<T>::infinity(); }
};

template <>
struct NumericLimits<half_t>
{
    static constexpr unsigned short binary_min    = 0x0400;
    static constexpr unsigned short binary_max    = 0x7BFF;
    static constexpr unsigned short binary_lowest = 0xFBFF;
    static constexpr unsigned short binary_qnan   = 0x7FFF;

    __host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }

    __host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }

    __host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }

    __host__ __device__ static constexpr half_t QuietNaN() { return bit_cast<half_t>(binary_qnan); }
};

#ifndef CK_CODE_GEN_RTC
template <>
struct NumericLimits<e8m0_bexp_t>
{
    static constexpr uint8_t binary_min      = 0x00;
    static constexpr uint8_t binary_max      = 0xFE;
    static constexpr uint8_t binary_qnan     = 0xFF;
    static constexpr uint8_t binary_1        = 0x7F;
    static constexpr uint8_t binary_2        = 0x80;
    static constexpr uint8_t binary_half     = 0x7E;
    static constexpr uint8_t binary_quarter  = 0x7D;
    static constexpr uint8_t binary_rcp_8    = 0x7C;
    static constexpr uint8_t binary_rcp_16   = 0x7B;
    static constexpr uint8_t binary_rcp_32   = 0x7A;
    static constexpr uint8_t binary_rcp_64   = 0x79;
    static constexpr uint8_t binary_rcp_128  = 0x78;
    static constexpr uint8_t binary_rcp_256  = 0x77;
    static constexpr uint8_t binary_rcp_512  = 0x76;
    static constexpr uint8_t binary_rcp_1024 = 0x75;

    __host__ __device__ static constexpr e8m0_bexp_t Min() { return e8m0_bexp_t(binary_min); }

    __host__ __device__ static constexpr e8m0_bexp_t Max() { return e8m0_bexp_t(binary_max); }

    __host__ __device__ static constexpr e8m0_bexp_t Lowest() { return e8m0_bexp_t(binary_min); }

    __host__ __device__ static constexpr e8m0_bexp_t QuietNaN()
    {
        return e8m0_bexp_t(binary_qnan);
    }

    __host__ __device__ static constexpr e8m0_bexp_t Binary_1()
    {
        return e8m0_bexp_t(binary_1);
    }
};

template <>
struct NumericLimits<e4m3_scale_t>
{
    static constexpr uint8_t binary_min    = 0x08;
    static constexpr uint8_t binary_max    = e4m3_scale_t::max_finite;
    static constexpr uint8_t binary_lowest = 0x00;
    static constexpr uint8_t binary_qnan   = e4m3_scale_t::nan_mask;

    __host__ __device__ static constexpr e4m3_scale_t Min()
    {
        return e4m3_scale_t(binary_min);
    }

    __host__ __device__ static constexpr e4m3_scale_t Max()
    {
        return e4m3_scale_t(binary_max);
    }

    __host__ __device__ static constexpr e4m3_scale_t Lowest()
    {
        return e4m3_scale_t(binary_lowest);
    }

    __host__ __device__ static constexpr e4m3_scale_t QuietNaN()
    {
        return e4m3_scale_t(binary_qnan);
    }
};

template <>
struct NumericLimits<e5m3_scale_t>
{
    static constexpr uint8_t binary_min    = 0x08;
    static constexpr uint8_t binary_max    = e5m3_scale_t::max_finite;
    static constexpr uint8_t binary_lowest = 0x00;
    static constexpr uint8_t binary_qnan   = e5m3_scale_t::nan_mask;

    __host__ __device__ static constexpr e5m3_scale_t Min()
    {
        return e5m3_scale_t(binary_min);
    }

    __host__ __device__ static constexpr e5m3_scale_t Max()
    {
        return e5m3_scale_t(binary_max);
    }

    __host__ __device__ static constexpr e5m3_scale_t Lowest()
    {
        return e5m3_scale_t(binary_lowest);
    }

    __host__ __device__ static constexpr e5m3_scale_t QuietNaN()
    {
        return e5m3_scale_t(binary_qnan);
    }
};
#endif

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
template <>
struct NumericLimits<fp8_t>
{
    // OCP E4M3 mode with exp bias = 7
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000 = 2^-6
    static constexpr uint8_t binary_max    = 0x7E; // 0b01111110 = 448
    static constexpr uint8_t binary_lowest = 0xFE; // 0b11111110 = -448
    static constexpr uint8_t binary_qnan   = 0x7F; // 0b01111111

    __host__ __device__ static constexpr fp8_t Min() { return fp8_t(binary_min); }

    __host__ __device__ static constexpr fp8_t Max() { return fp8_t(binary_max); }

    __host__ __device__ static constexpr fp8_t Lowest() { return fp8_t(binary_lowest); }

    __host__ __device__ static constexpr fp8_t QuietNaN() { return fp8_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_t>
{
    // OCP E5M2 mode with exp bias = 15
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100 = 2^-14
    static constexpr uint8_t binary_max    = 0x7B; // 0b01111011 = 57344
    static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011 = -57344
    static constexpr uint8_t binary_qnan   = 0x7D; // 0b01111101

    __host__ __device__ static constexpr bf8_t Min() { return bf8_t(binary_min); }

    __host__ __device__ static constexpr bf8_t Max() { return bf8_t(binary_max); }

    __host__ __device__ static constexpr bf8_t Lowest() { return bf8_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_t QuietNaN() { return bf8_t(binary_qnan); }
};

template <>
struct NumericLimits<int4_t>
{
    __host__ __device__ static constexpr int4_t Min() { return int4_t(-8); }

    __host__ __device__ static constexpr int4_t Max() { return int4_t(7); }

    __host__ __device__ static constexpr int4_t Lowest() { return int4_t(-8); }
};

template <>
struct NumericLimits<uint4_t>
{
    __host__ __device__ static constexpr uint4_t Min() { return uint4_t(0); }

    __host__ __device__ static constexpr uint4_t Max() { return uint4_t(15); }

    __host__ __device__ static constexpr uint4_t Lowest() { return uint4_t(0); }
};

template <>
struct NumericLimits<f6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08;
    static constexpr uint8_t binary_max_normal    = 0x1F;
    static constexpr uint8_t binary_lowest_normal = 0x3F;
    static constexpr uint8_t binary_min_subnorm   = 0x01;
    static constexpr uint8_t binary_max_subnorm   = 0x07;

    static constexpr float data_max_normal_number    = 7.5f;
    static constexpr float data_min_subnormal_number = 0.125f;

    __host__ __device__ static constexpr f6_t Min() { return f6_t(binary_min_normal & 0b111111); }

    __host__ __device__ static constexpr f6_t Max() { return f6_t(binary_max_normal & 0b111111); }

    __host__ __device__ static constexpr f6_t Lowest()
    {
        return f6_t(binary_lowest_normal & 0b111111);
    }

    __host__ __device__ static constexpr f6_t MinSubnorm()
    {
        return f6_t(binary_min_subnorm & 0b111111);
    }

    __host__ __device__ static constexpr f6_t MaxSubnorm()
    {
        return f6_t(binary_max_subnorm & 0b111111);
    }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }

    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<bf6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08;
    static constexpr uint8_t binary_max_normal    = 0x1F;
    static constexpr uint8_t binary_lowest_normal = 0x3F;
    static constexpr uint8_t binary_min_subnorm   = 0x01;
    static constexpr uint8_t binary_max_subnorm   = 0x03;

    static constexpr float data_max_normal_number    = 28.0f;
    static constexpr float data_min_subnormal_number = 0.0625f;

    __host__ __device__ static constexpr bf6_t Min() { return bf6_t(binary_min_normal); }

    __host__ __device__ static constexpr bf6_t Max() { return bf6_t(binary_max_normal); }

    __host__ __device__ static constexpr bf6_t Lowest() { return bf6_t(binary_lowest_normal); }

    __host__ __device__ static constexpr bf6_t MinSubnorm() { return bf6_t(binary_min_subnorm); }

    __host__ __device__ static constexpr bf6_t MaxSubnorm() { return bf6_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }

    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION

} // namespace ck
