// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Adapted for Hygon HCU: gemm_blockscale_wp stub.
// Upstream this depends on DeviceGemmMultipleD_BlockScale_BPreshuffle +
// device_gemm_multiple_d_xdl_cshuffle_v3_blockscale_bpreshuffle.hpp
// which requires the block-scale quantization pipeline + MoE variants.
// Not yet ported to HCU.
//
// TODO: Import block-scale quantization pipeline:
//   - include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_blockscale_bpreshuffle.hpp
//   - include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle.hpp
//   - MoE block-scale pipeline variants

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// gemm_blockscale_wp is not yet available on HCU.
// Requires block-scale quantization + MoE pipeline import.
// See gemm_universal.hpp for standard GEMM instances.

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
