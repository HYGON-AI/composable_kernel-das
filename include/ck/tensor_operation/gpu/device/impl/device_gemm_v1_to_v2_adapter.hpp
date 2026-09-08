// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Adapted for Hygon HCU: wraps V1 DeviceGemm instances to satisfy the
// DeviceGemmV2 interface, bridging the gap without requiring a full
// V3 pipeline import.

#pragma once

#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGemmV1ToV2Adapter
    : public DeviceGemmV2<ALayout,
                          BLayout,
                          CLayout,
                          ADataType,
                          BDataType,
                          CDataType,
                          AElementwiseOperation,
                          BElementwiseOperation,
                          CElementwiseOperation>
{
    using V1DeviceOp = DeviceGemm<ALayout,
                                  BLayout,
                                  CLayout,
                                  ADataType,
                                  BDataType,
                                  CDataType,
                                  AElementwiseOperation,
                                  BElementwiseOperation,
                                  CElementwiseOperation>;

    explicit DeviceGemmV1ToV2Adapter(std::unique_ptr<V1DeviceOp> v1_op)
        : v1_op_(std::move(v1_op))
    {
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        ck::index_t /*KSplit*/,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) override
    {
        return v1_op_->MakeArgumentPointer(
            p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC, a_element_op, b_element_op, c_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return v1_op_->MakeInvokerPointer();
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return v1_op_->IsSupportedArgument(p_arg);
    }

    bool GetPermuteA() override { return false; }

    bool GetPermuteB() override { return false; }

    ck::index_t GetKPerBlock() override { return -1; }

private:
    std::unique_ptr<V1DeviceOp> v1_op_;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
