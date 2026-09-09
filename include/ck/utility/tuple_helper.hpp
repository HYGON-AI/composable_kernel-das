// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "functional4.hpp"
#include "tuple.hpp"

namespace ck {

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tuple(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return make_tuple(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tie(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return tie(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

// Creates Tuple<Sequence<0>, Sequence<1>, ..., Sequence<N-1>>
namespace detail {
template <index_t... Is>
__host__ __device__ constexpr auto make_identity_sequences_impl(Sequence<Is...>)
{
    return make_tuple(Sequence<Is>{}...);
}
} // namespace detail

template <index_t N>
__host__ __device__ constexpr auto generate_identity_sequences()
{
    return detail::make_identity_sequences_impl(
        typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <index_t N>
__host__ __device__ constexpr auto generate_identity_sequences(Number<N>)
{
    return generate_identity_sequences<N>();
}

// tx and ty are tuple of references, return type of will tuple of referennce (not rvalue)
template <typename... X, typename... Y>
__host__ __device__ constexpr auto concat_tuple_of_reference(const Tuple<X&...>& tx,
                                                             const Tuple<Y&...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return Tuple<decltype(zs)...>{std::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

template <typename... X, typename... Y>
__host__ __device__ constexpr auto concat_tuple(const Tuple<X...>& tx, const Tuple<Y...>& ty)
{
    return unpack2(
        [&](auto... zs) { return Tuple<decltype(zs)...>{ck::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

// Support any number of tuples to concat (also 1)
template <typename... X>
__host__ __device__ constexpr auto concat_tuple(const Tuple<X...>& tx)
{
    return tx;
}

template <typename... X, typename... Tuples>
__host__ __device__ constexpr auto concat_tuple(const Tuple<X...>& tx, const Tuples&... tuples)
{
    return concat_tuple(tx, concat_tuple(tuples...));
}

namespace detail {

template <typename F, typename X, index_t... Is>
__host__ __device__ constexpr auto transform_tuples_impl(F f, const X& x, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, typename Z, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, const Z& z, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}), z.At(Number<Is>{}))...);
}

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x)
{
    return detail::transform_tuples_impl(
        f, x, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y)
{
    return detail::transform_tuples_impl(
        f, x, y, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y, typename Z>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y, const Z& z)
{
    return detail::transform_tuples_impl(
        f, x, y, z, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

} // namespace ck
