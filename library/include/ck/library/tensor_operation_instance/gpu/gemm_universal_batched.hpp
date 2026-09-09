// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modified by Hygon Information Technology Co., Ltd.
//
// Adapted for Hygon HCU: gemm_universal_batched wrapper.
// Redirects to the existing HCU batched_gemm.hpp factory.
// Note: upstream uses DeviceBatchedGemmV2MultiD (V2 batched interface);
// HCU uses DeviceBatchedGemm (V1). A full V2 batched adapter would require
// importing device_batched_gemm_v2.hpp and the V3 batched multi-d pipeline.

#pragma once

#include "ck/library/tensor_operation_instance/gpu/batched_gemm.hpp"

// For API compatibility: upstream code that includes gemm_universal_batched.hpp
// will find the existing HCU batched GEMM instances via DeviceBatchedGemm (V1).
//
// To use the V2 interface (DeviceBatchedGemmV2MultiD), import the upstream
// device_batched_gemm_v2.hpp and device_batched_gemm_v1_to_v2_adapter.hpp,
// then provide a DeviceOperationInstanceFactory specialization similar to
// gemm_universal.hpp's pattern.
