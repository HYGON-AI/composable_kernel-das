// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16
void add_device_gemm_ab_scale_xdl_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                           Col,
                                                           Empty_Tuple,
                                                           Row,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           F32,
                                                           Empty_Tuple,
                                                           BF16,
                                                           1,
                                                           128,
                                                           32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleD_ABScale<ALayout,
                                                              BLayout,
                                                              Empty_Tuple,
                                                              ELayout,
                                                              ADataType,
                                                              AScaleDataType,
                                                              BDataType,
                                                              BScaleDataType,
                                                              Empty_Tuple,
                                                              EDataType,
                                                              1,
                                                              128,
                                                              32,
                                                              PassThrough,
                                                              PassThrough,
                                                              PassThrough>>
{
    using DeviceOp = DeviceGemmMultipleD_ABScale<ALayout,
                                                 BLayout,
                                                 Empty_Tuple,
                                                 ELayout,
                                                 ADataType,
                                                 AScaleDataType,
                                                 BDataType,
                                                 BScaleDataType,
                                                 Empty_Tuple,
                                                 EDataType,
                                                 1,
                                                 128,
                                                 32,
                                                 PassThrough,
                                                 PassThrough,
                                                 PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_ENABLE_BF16
        if constexpr(is_same_v<ADataType, BF16> && is_same_v<BDataType, BF16> &&
                     is_same_v<EDataType, BF16> && is_same_v<AScaleDataType, F32> &&
                     is_same_v<BScaleDataType, F32>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_ab_scale_xdl_bf16_bf16_bf16_mk_nk_mn_instances(op_ptrs);
            }
        }
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
