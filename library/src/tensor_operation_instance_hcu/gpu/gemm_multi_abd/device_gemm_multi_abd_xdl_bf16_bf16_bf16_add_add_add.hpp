// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multi_abd_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = bhalf_t;
using F32  = float;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

template <index_t... Is>
using S = Sequence<Is...>;

using Add = element_wise::Add;

using device_gemm_multi_abd_xdl_bf16_bf16_bf16_add_add_add_instances = std::tuple<
    DeviceGemmMultiABD_Xdl_CShuffle_V3<Tuple<Row, Row>,
                                       Tuple<Col, Col>,
                                       Tuple<Row>,
                                       Row,
                                       Tuple<BF16, BF16>,
                                       Tuple<BF16, BF16>,
                                       Tuple<BF16>,
                                       BF16,
                                       F32,
                                       BF16,
                                       Add,
                                       Add,
                                       Add,
                                       GemmSpecialization::Default,
                                       256,
                                       128,
                                       128,
                                       32,
                                       8,
                                       8,
                                       16,
                                       16,
                                       4,
                                       4,
                                       S<4, 64, 1>,
                                       S<1, 0, 2>,
                                       S<1, 0, 2>,
                                       2,
                                       8,
                                       8,
                                       0,
                                       S<4, 64, 1>,
                                       S<1, 0, 2>,
                                       S<1, 0, 2>,
                                       2,
                                       8,
                                       8,
                                       0,
                                       1,
                                       1,
                                       S<1, 32, 1, 8>,
                                       S<4>,
                                       BlockGemmPipelineScheduler::Intrawave,
                                       BlockGemmPipelineVersion::v3>>;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
