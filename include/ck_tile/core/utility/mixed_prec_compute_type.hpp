// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

#include <type_traits>

namespace ck_tile {

template <typename ComputeDataType, typename ADataType, typename BDataType>
using mixed_prec_compute_type_t =
    std::conditional_t<std::is_void_v<ComputeDataType>, ADataType, ComputeDataType>;

template <typename ThisDataType, typename OtherDataType, typename ComputeDataType>
using mixed_prec_compute_type_from_input_t =
    std::conditional_t<std::is_void_v<ThisDataType>, ComputeDataType, ThisDataType>;

} // namespace ck_tile
