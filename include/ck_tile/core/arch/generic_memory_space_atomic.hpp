// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <cstdint>
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"

namespace ck_tile {

CK_TILE_HOST_DEVICE bf16_t add_bf16_t(const bf16_t& a, const bf16_t& b)
{
    return type_convert<bf16_t>(type_convert<float>(a) + type_convert<float>(b));
}

CK_TILE_HOST_DEVICE bf16x2_t add_bf16x2_t(const bf16x2_t& a, const bf16x2_t& b)
{
    bf16x2_t rtn;
    rtn[0] = add_bf16_t(a[0], b[0]);
    rtn[1] = add_bf16_t(a[1], b[1]);
    return rtn;
}

CK_TILE_HOST_DEVICE fp16_t add_fp16_t(const fp16_t& a, const fp16_t& b)
{
    return type_convert<fp16_t>(type_convert<float>(a) + type_convert<float>(b));
}

CK_TILE_HOST_DEVICE fp16x2_t add_fp16x2_t(const fp16x2_t& a, const fp16x2_t& b)
{
    fp16x2_t rtn;
    rtn[0] = add_fp16_t(a[0], b[0]);
    rtn[1] = add_fp16_t(a[1], b[1]);
    return rtn;
}

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to make the implementation of atomic_add explicit for
// each datatype.
template <typename X>
CK_TILE_DEVICE void atomic_add(X* p_dst, const X& x);

template <>
CK_TILE_DEVICE void atomic_add<bf16_t>(bf16_t* p_dst, const bf16_t& x)
{
    union U32BF162
    {
        uint32_t u32;
        bf16x2_t bf162;
    };

    const auto address = reinterpret_cast<std::uintptr_t>(p_dst);
    auto* dword_addr   = reinterpret_cast<uint32_t*>(address & ~std::uintptr_t{0x3});
    const index_t lane = (address >> 1) & 1;
    U32BF162 cur_v;
    U32BF162 new_;
    uint32_t old_v, new_v;
    cur_v.u32 = *dword_addr;

    do
    {
        old_v              = cur_v.u32;
        new_.bf162         = cur_v.bf162;
        new_.bf162[lane]   = add_bf16_t(cur_v.bf162[lane], x);
        new_v              = new_.u32;
        cur_v.u32          = atomicCAS(dword_addr, old_v, new_v);
    } while(cur_v.u32 != old_v);
}

template <>
CK_TILE_DEVICE void atomic_add<bf16x2_t>(bf16x2_t* p_dst, const bf16x2_t& x)
{
    union U32BF162_ADDR
    {
        uint32_t* u32_a;
        bf16x2_t* bf162_a;
    };

    union U32BF162
    {
        uint32_t u32;
        bf16x2_t bf162;
    };

    U32BF162_ADDR dword_addr;
    U32BF162 cur_v;
    U32BF162 new_;
    uint32_t old_v, new_v;
    dword_addr.bf162_a = p_dst;
    cur_v.u32          = *dword_addr.u32_a;

    do
    {
        old_v      = cur_v.u32;
        new_.bf162 = add_bf16x2_t(cur_v.bf162, x);
        new_v      = new_.u32;
        cur_v.u32  = atomicCAS(dword_addr.u32_a, old_v, new_v);
    } while(cur_v.u32 != old_v);
}

template <>
CK_TILE_DEVICE void atomic_add<fp16x2_t>(fp16x2_t* p_dst, const fp16x2_t& x)
{
    union U32FP162_ADDR
    {
        uint32_t* u32_a;
        fp16x2_t* fp162_a;
    };

    union U32FP162
    {
        uint32_t u32;
        fp16x2_t fp162;
    };

    U32FP162_ADDR dword_addr;
    U32FP162 cur_v;
    U32FP162 new_;
    uint32_t old_v, new_v;
    dword_addr.fp162_a = p_dst;
    cur_v.u32          = *dword_addr.u32_a;

    do
    {
        old_v      = cur_v.u32;
        new_.fp162 = add_fp16x2_t(cur_v.fp162, x);
        new_v      = new_.u32;
        cur_v.u32  = atomicCAS(dword_addr.u32_a, old_v, new_v);
    } while(cur_v.u32 != old_v);
}

template <>
CK_TILE_DEVICE void atomic_add<fp16_t>(fp16_t* p_dst, const fp16_t& x)
{
    union U32FP162
    {
        uint32_t u32;
        fp16x2_t fp162;
    };

    const auto address = reinterpret_cast<std::uintptr_t>(p_dst);
    auto* dword_addr   = reinterpret_cast<uint32_t*>(address & ~std::uintptr_t{0x3});
    const index_t lane = (address >> 1) & 1;
    U32FP162 cur_v;
    U32FP162 new_;
    uint32_t old_v, new_v;
    cur_v.u32 = *dword_addr;

    do
    {
        old_v              = cur_v.u32;
        new_.fp162         = cur_v.fp162;
        new_.fp162[lane]   = add_fp16_t(cur_v.fp162[lane], x);
        new_v              = new_.u32;
        cur_v.u32          = atomicCAS(dword_addr, old_v, new_v);
    } while(cur_v.u32 != old_v);
}

template <typename T, index_t N>
CK_TILE_DEVICE void atomic_add_g(T* p_dst, const thread_buffer<T, N>& x)
{
    static_assert((std::is_same<T, int32_t>::value && (N == 1)) ||
                      (std::is_same<T, uint32_t>::value && (N == 1)) ||
                      (std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (std::is_same<T, double>::value && (N == 1 || N == 2)) ||
                      (std::is_same<T, fp16_t>::value && (N == 1 || N == 2 || N == 4)) ||
                      (std::is_same<T, bf16_t>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");

    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};
    constexpr auto I2 = number<2>{};
    constexpr auto I3 = number<3>{};

    if constexpr(std::is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            atomicAdd(p_dst, bit_cast<float>(x));
        }
        else if constexpr(N == 2)
        {
            atomicAdd(c_style_pointer_cast<float*>(p_dst), x.template get_as<float>()[I0]);
            atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, x.template get_as<float>()[I1]);
        }
        else if constexpr(N == 4)
        {
            atomicAdd(c_style_pointer_cast<float*>(p_dst), x.template get_as<float>()[I0]);
            atomicAdd(c_style_pointer_cast<float*>(p_dst) + 1, x.template get_as<float>()[I1]);
            atomicAdd(c_style_pointer_cast<float*>(p_dst) + 2, x.template get_as<float>()[I2]);
            atomicAdd(c_style_pointer_cast<float*>(p_dst) + 3, x.template get_as<float>()[I3]);
        }
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            atomicAdd(p_dst, bit_cast<double>(x));
        }
        else if constexpr(N == 2)
        {
            atomicAdd(c_style_pointer_cast<double*>(p_dst), x.template get_as<double>()[I0]);
            atomicAdd(c_style_pointer_cast<double*>(p_dst) + 1, x.template get_as<double>()[I1]);
        }
    }
    else if constexpr(std::is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            atomicAdd(p_dst, bit_cast<int32_t>(x));
        }
    }
    else if constexpr(std::is_same<T, uint32_t>::value)
    {
        if constexpr(N == 1)
        {
            atomicAdd(p_dst, bit_cast<uint32_t>(x));
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value)
    {
        if constexpr(N == 1)
        {
            atomic_add(p_dst, bit_cast<bf16_t>(x));
        }
        else if constexpr(N == 2)
        {
            atomic_add(c_style_pointer_cast<bf16x2_t*>(p_dst), bit_cast<bf16x2_t>(x));
        }
        else if constexpr(N == 4)
        {
            atomic_add(c_style_pointer_cast<bf16x2_t*>(p_dst), x.template get_as<bf16x2_t>()[I0]);
            atomic_add(c_style_pointer_cast<bf16x2_t*>(p_dst) + 1,
                       x.template get_as<bf16x2_t>()[I1]);
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value)
    {
        if constexpr(N == 1)
        {
            atomic_add(p_dst, bit_cast<fp16_t>(x));
        }
        else if constexpr(N == 2)
        {
            atomic_add(c_style_pointer_cast<fp16x2_t*>(p_dst), bit_cast<fp16x2_t>(x));
        }
        else if constexpr(N == 4)
        {
            atomic_add(c_style_pointer_cast<fp16x2_t*>(p_dst), x.template get_as<fp16x2_t>()[I0]);
            atomic_add(c_style_pointer_cast<fp16x2_t*>(p_dst) + 1,
                       x.template get_as<fp16x2_t>()[I1]);
        }
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void atomic_max_g(T* p_dst, const thread_buffer<T, N>& x)
{
    static_assert((std::is_same<T, int32_t>::value && (N == 1)) ||
                      (std::is_same<T, uint32_t>::value && (N == 1)) ||
                      (std::is_same<T, float>::value && (N == 1 || N == 2)) ||
                      (std::is_same<T, double>::value && (N == 1)),
                  "wrong! not implemented");

    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};

    if constexpr(std::is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            atomicMax(p_dst, bit_cast<float>(x));
        }
        else if constexpr(N == 2)
        {
            atomicMax(c_style_pointer_cast<float*>(p_dst), x.template get_as<float>()[I0]);
            atomicMax(c_style_pointer_cast<float*>(p_dst) + 1, x.template get_as<float>()[I1]);
        }
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            atomicMax(p_dst, bit_cast<double>(x));
        }
    }
    else if constexpr(std::is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            atomicMax(p_dst, bit_cast<int32_t>(x));
        }
    }
    else if constexpr(std::is_same<T, uint32_t>::value)
    {
        if constexpr(N == 1)
        {
            atomicMax(p_dst, bit_cast<uint32_t>(x));
        }
    }
}

} // namespace ck_tile
