// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
// [HCU移植] 非 gfx9 分支的默认 warp gemm 由上游 MFMA 32x32 TransposedCDistribution
// 改为 HCU MMAC 16x32x64 TransC（WarpGemmMmac*_WT16x32x64_MR1NR2MI1NI1_TRANSC，MWarp/NWarp=1）。
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"

namespace ck_tile {

// Default policy for BlockGemmARegBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmARegBSmemCRegV1DefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
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
            return make_tuple(WarpGemmMmacF16F16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC{}, 1, 1);
#endif
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
                          std::is_same_v<typename Problem::BDataType, bf16_t> &&
                          std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC{}, 1, 1);
        }
    }
};

} // namespace ck_tile
