// grouped_gemm_v8_bf16_tn_device_args.cpp — CK grouped-GEMM instance entry
// (InstanceId 5001) for the uniform Direct-to-LDS TN (wgrad) kernel.
//
// Same construction as grouped_gemm_v7_bf16_nn_device_args.cpp (read that
// file's header first: the ABI shape, the flat grid and the codegen contract
// are all there and are not repeated here). The TN-specific points:
//
//   * Layout mapping. This entry consumes the CK TN description DIRECTLY:
//       A (true A^T) : [k, sum(M_g)] row-major, per-group a_ptr into the panel,
//                      stride_a = the panel's row pitch in elements
//       B (true B)   : [k, n] row-major per group, stride_b = its row pitch
//       D            : [sum(M_g), n] row-major, stride_e = its row pitch
//     i.e. the group's own column origin travels in a_ptr. The pipeline's A
//     panel load covers a whole 256-wide column tile, so the caller must keep
//     256 elements of addressable slack past the last group's columns -- the
//     pad obligation the selector documents (ragged M is allowed; the caller
//     owns the pad).
//
//   * The accumulation order is the plain K-serial one over the whole K extent
//     of the problem; there is no k-split anywhere in the v7 path.
//
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "grouped_gemm_v7_uniform_launch.hpp"

extern "C" int ck_tile_hcu_grouped_gemm_v8_bf16_tn_device_args(
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
    return v7_uniform_launch::launch<GemmConfigV8TnBf16,
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
