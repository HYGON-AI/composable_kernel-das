// grouped_gemm_v7_bf16_nn_device_args.cpp — CK grouped-GEMM instance entry
// (InstanceId 5000) for the uniform Direct-to-LDS NN (dgrad) kernel.
//
// The whole kernel lives in ck_tile: pipeline (gemm_pipeline_v7_uniform.hpp), epilogue
// (grouped_gemm_v7_uniform_epilogue.hpp) and config, instantiated through the
// framework's own GroupedGemmKernel by grouped_gemm_v7_uniform_launch.hpp. What this
// entry adds is the ABI shape: it receives the device kargs array the CK
// device-args ABI builds (one ck_tile::GemmTransKernelArg per group, built on
// the device, never read back to the host) plus the host-side tile bound, and
// launches one flat grid over it.
//
// CONTRACT (enforced by the CK selector, not here): bf16 in, bf16 out, gfx936,
// NN layout (A [M_g, K] row-major, B [K, N] row-major, D [M_g, N] row-major),
// K % 16 == 0, K >= 64 (the 4-stage ring), N % 256 == 0,
// 1 <= G <= kGroupedGemmV7MaxGroups, positive group lengths. A group's M may
// be ragged -- the epilogue masks rows past M_g.
//
// This TU is compiled under the v7 codegen contract (see CMakeLists.txt: the
// -mllvm switches and -fno-finite-math-only the kernel's register budget was
// tuned under). Do not move it into the provider manifest, whose consumers
// compile with their own options.
//
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "grouped_gemm_v7_uniform_launch.hpp"

extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream,
    std::uint64_t tile_slots) {
    (void)num_cu;   // the grid is sized by tile_slots, not by the CU count
    if (device_args == nullptr || group_count <= 0 ||
        group_count > ck_tile::kGroupedGemmV7MaxGroups || tile_slots == 0) {
        return -1;
    }

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    return v7_uniform_launch::launch<GemmConfigV7NnBf16,
                              Row,
                              Row,
                              Row,
                              ck_tile::bf16_t,
                              ck_tile::bf16_t,
                              float,
                              ck_tile::bf16_t>(device_args,
                                               group_count,
                                               tile_slots,
                                               stream);
}
