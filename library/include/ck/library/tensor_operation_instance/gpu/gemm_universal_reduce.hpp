// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Adapted for Hygon HCU: gemm_universal_reduce stub.
// Upstream this depends on DeviceGemmV2R1 + device_gemm_xdl_cshuffle_v3r1.hpp
// which requires the V3R1 reduce pipeline (gridwise_gemm_xdl_cshuffle_v3r1,
// blockwise V3R1). Not yet ported to HCU.
//
// TODO: Import V3R1 reduce pipeline from upstream composable_kernel_github:
//   - include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3r1.hpp
//   - include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3r1.hpp

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// gemm_universal_reduce is not yet available on HCU.
// Requires V3R1 reduce pipeline import. See gemm_universal.hpp for the
// available non-reduce GEMM instances.

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
