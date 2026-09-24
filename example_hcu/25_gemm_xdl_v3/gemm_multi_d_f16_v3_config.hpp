// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

// Example instance configuration, independent of host tensors and CPU checking.
// Consumers instantiate the CK device template with their own supported tuning.
namespace hcu_f16_multi_d {

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using F16 = ck::half_t;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
template <ck::index_t... X>
using S = ck::Sequence<X...>;

// A is [M,K], B is [K,N], and all D/E tensors are [M,N]. Choose the
// vector dimension from the physical layout; the K-facing LDS layout is shared.
template <typename ALayout, typename BLayout, typename DsLayout,
          typename DsDataType, typename CDEOp>
using DeviceGemm = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
    ALayout, BLayout, DsLayout, Row, F16, F16, DsDataType, F16, float, float,
    PassThrough, PassThrough, CDEOp,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    256, 128, 128, 32, 8, 8, 16, 16, 4, 4,
    S<4,64,1>,
    std::conditional_t<std::is_same_v<ALayout, Row>, S<1,0,2>, S<0,2,1>>,
    std::conditional_t<std::is_same_v<ALayout, Row>, S<1,0,2>, S<0,2,1>>,
    std::is_same_v<ALayout, Row> ? 2 : 1,
    std::is_same_v<ALayout, Row> ? 8 : 2, 8, 0,
    S<4,64,1>,
    std::conditional_t<std::is_same_v<BLayout, Col>, S<1,0,2>, S<0,2,1>>,
    std::conditional_t<std::is_same_v<BLayout, Col>, S<1,0,2>, S<0,2,1>>,
    std::is_same_v<BLayout, Col> ? 2 : 1,
    std::is_same_v<BLayout, Col> ? 8 : 2, 8, 0,
    1, 1, S<1,32,1,8>, S<4>,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3>;

} // namespace hcu_f16_multi_d
