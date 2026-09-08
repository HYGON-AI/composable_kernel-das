// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"

namespace ck {

#ifndef CK_CODE_GEN_RTC
template <typename T>
struct nnvb_data_t_selector
{
    using type = typename scalar_type<T>::type;
};

template <>
struct nnvb_data_t_selector<e8m0_bexp_t>
{
    using type = e8m0_bexp_t::type;
};

template <>
struct nnvb_data_t_selector<e4m3_scale_t>
{
    using type = e4m3_scale_t::type;
};

template <>
struct nnvb_data_t_selector<e5m3_scale_t>
{
    using type = e5m3_scale_t::type;
};

// e8m0
using e8m0x4_bexp_t = typename vector_type<e8m0_bexp_t::type, 4>::type;
using e8m0x8_bexp_t = typename vector_type<e8m0_bexp_t::type, 8>::type;
// e4m3
using e4m3x4_scale_t = typename vector_type<e4m3_scale_t::type, 4>::type;
using e4m3x8_scale_t = typename vector_type<e4m3_scale_t::type, 8>::type;
// e5m3
using e5m3x4_scale_t = typename vector_type<e5m3_scale_t::type, 4>::type;
using e5m3x8_scale_t = typename vector_type<e5m3_scale_t::type, 8>::type;
#endif

} // namespace ck
