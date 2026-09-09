// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modified by Hygon Information Technology Co., Ltd.
//
// Adapted for Hygon HCU: gemm_mx stub.
// Upstream this depends on DeviceGemmMX + device_gemm_xdl_cshuffle_v3_mx.hpp
// which requires the MX type system (F4, F6, FP8, BF8, BF6 scalar types
// with E8M0PK scaling). Not yet ported to HCU.
//
// TODO: Import MX type system and V3 MX pipeline:
//   - MX scalar types (f4_t, f6_t, fp8_t, bf8_t, bf6_t) + type_convert
//   - include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp
//   - library/src/tensor_operation_instance/gpu/gemm_mx/ (18 files)

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// gemm_mx is not yet available on HCU.
// Requires MX type system import (F4/F6/FP8/BF8/BF6 scalar types).
// See gemm_universal.hpp for standard GEMM instances.

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
