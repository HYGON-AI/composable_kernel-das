// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck/utility/data_type.hpp"

template <typename T>
struct DtypeWrapper
{
    using type = T;
};

template <>
struct DtypeWrapper<ck::half_t>
{
    using type = ck_tile::fp16_t;
};

template <>
struct DtypeWrapper<ck::bhalf_t>
{
    using type = ck_tile::bf16_t;
};
