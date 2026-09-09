// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck/utility/get_shift.hpp"
#include "ck/utility/reduction_enums.hpp"

namespace ck {

struct float_equal_one
{
    template <class T>
    __host__ __device__ inline bool operator()(T x)
    {
        return x <= static_cast<T>(1.0f) and x >= static_cast<T>(1.0f);
    };
};

struct float_equal_zero
{
    template <class T>
    __host__ __device__ inline bool operator()(T x)
    {
        return x <= static_cast<T>(0.0f) and x >= static_cast<T>(0.0f);
    };
};

} // namespace ck
