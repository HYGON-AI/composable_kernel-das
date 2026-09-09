// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include <memory>
#include <type_traits>
#include <tuple>
#include <vector>

#include "ck/utility/functional2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename DeviceGemmOp>
struct DeviceGroupedConvBwdWeight_Explicit;

namespace instance {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename DeviceGemmV3Ops,
          typename BaseOp>
void add_explicit_gemm_device_operation_instances(
    std::vector<std::unique_ptr<BaseOp>>& op_instances)
{
    ck::static_for<0, std::tuple_size_v<DeviceGemmV3Ops>, 1>{}([&](auto i) {
        using DeviceGemmOp = std::tuple_element_t<i, DeviceGemmV3Ops>;

        if constexpr(std::is_same_v<DeviceGemmOp, std::nullptr_t>)
        {
            return;
        }
        else
        {
            using NewOpInstance = DeviceGroupedConvBwdWeight_Explicit<NDimSpatial,
                                                                      InLayout,
                                                                      WeiLayout,
                                                                      OutLayout,
                                                                      InDataType,
                                                                      WeiDataType,
                                                                      OutDataType,
                                                                      InElementwiseOperation,
                                                                      WeiElementwiseOperation,
                                                                      OutElementwiseOperation,
                                                                      DeviceGemmOp>;

            static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                          "NewOpInstance must derive from BaseOp");
            static_assert(std::is_default_constructible_v<NewOpInstance>,
                          "NewOpInstance must be default-constructible");

            op_instances.push_back(std::make_unique<NewOpInstance>());
        }
    });
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
