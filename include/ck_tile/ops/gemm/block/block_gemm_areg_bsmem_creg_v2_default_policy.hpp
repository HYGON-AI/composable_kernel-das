// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
// [HCU移植] 非 gfx9 分支的默认 warp gemm 由上游 MFMA 32x32 TransposedCDistribution
// 改为 HCU MMAC 16x32x64 TransC，并按 fp16/bf16 分别返回对应 MMAC 别名。
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"

namespace ck_tile {

// Default policy for BlockGemmARegBSmemCRegV2
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmARegBSmemCRegV2DefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {

#if 0
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        static_assert(kBlockSize % get_warp_size() == 0, "wrong!");

        constexpr index_t NumWarp = kBlockSize / get_warp_size();

        // FIXME
        if constexpr(NumWarp == 4 && kMPerBlock % 128 == 0 &&
                     kNPerBlock % 128 == 0 % kKPerBlock % 16 == 0)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K8{}, 4, 1);
        }
        else
        {
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K8{}, 4, 1);
        }
#else
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMmacF16F16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC{}, 1, 1);
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC{}, 1, 1);
        }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
#endif
    }
};

} // namespace ck_tile
