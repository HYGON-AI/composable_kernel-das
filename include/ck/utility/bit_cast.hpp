// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "ck/config.h"

#ifndef CK_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#endif

#ifndef CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST
#define CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST 1
#endif

namespace ck {

template <typename Y, typename X>
__host__ __device__ constexpr Y bit_cast(const X& x)
{
    static_assert(sizeof(X) == sizeof(Y),
                  "bit_cast requires source and destination to have the same size");

#if defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(Y, x);
#endif
#endif

#if defined(CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST) && CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST
    Y y;

    __builtin_memcpy(&y, &x, sizeof(X));

    return y;
#else
    union AsType
    {
        X x;
        Y y;
    };

    return AsType{x}.y;
#endif
}

} // namespace ck
