// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modified by Hygon Information Technology Co., Ltd.
//
// Adapted for Hygon HCU: gemm_universal factory header.
// Wraps existing HCU V1 DeviceGemm instances into the DeviceGemmV2 interface
// used by upstream client code. Delegates to gemm.hpp (V1 factory).

#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_v1_to_v2_adapter.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmV2<ALayout,
                                               BLayout,
                                               CLayout,
                                               ADataType,
                                               BDataType,
                                               CDataType,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOpV2 = DeviceGemmV2<ALayout,
                                    BLayout,
                                    CLayout,
                                    ADataType,
                                    BDataType,
                                    CDataType,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough,
                                    ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        // Get V1 instances from the existing HCU factory
        auto v1_instances = DeviceOperationInstanceFactory<
            DeviceGemm<ALayout,
                       BLayout,
                       CLayout,
                       ADataType,
                       BDataType,
                       CDataType,
                       ck::tensor_operation::element_wise::PassThrough,
                       ck::tensor_operation::element_wise::PassThrough,
                       ck::tensor_operation::element_wise::PassThrough>>::GetInstances();

        // Wrap each V1 instance in the adapter
        std::vector<std::unique_ptr<DeviceOpV2>> op_ptrs;
        op_ptrs.reserve(v1_instances.size());
        for(auto& v1_op : v1_instances)
        {
            op_ptrs.push_back(
                std::make_unique<DeviceGemmV1ToV2Adapter<ALayout,
                                                         BLayout,
                                                         CLayout,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         ck::tensor_operation::element_wise::PassThrough,
                                                         ck::tensor_operation::element_wise::PassThrough,
                                                         ck::tensor_operation::element_wise::PassThrough>>(
                    std::move(v1_op)));
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
