// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_mmac_tn_asmem_bsmem_creg_mls_v1.hpp"

namespace ck_tile {

struct GemmPipelineAgBgCrMlsDefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        return make_naive_tensor_descriptor_packed(
            make_tuple(number<Problem::BlockGemmShape::kM>{},
                       number<Problem::BlockGemmShape::kK>{}),
            number<8>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        return make_naive_tensor_descriptor_packed(
            make_tuple(number<Problem::BlockGemmShape::kN>{},
                       number<Problem::BlockGemmShape::kK>{}),
            number<8>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetALdsElementSize()
    {
        return integer_least_multiple(
            MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
            Problem::BlockGemmShape::kK);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetBLdsElementSize()
    {
        return integer_least_multiple(
            MakeBLdsBlockDescriptor<Problem>().get_element_space_size(),
            Problem::BlockGemmShape::kK);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsByteSize()
    {
#if defined(CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE)
        constexpr index_t num_lds_stages = 1;
#else
        constexpr index_t num_lds_stages = 2;
#endif
        return num_lds_stages *
               (GetALdsElementSize<Problem>() * sizeof(typename Problem::ADataType) +
                GetBLdsElementSize<Problem>() * sizeof(typename Problem::BDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return BlockGemmMmacTNAsmemBsmemCregMlsV1<Problem>{};
    }
};

} // namespace ck_tile
