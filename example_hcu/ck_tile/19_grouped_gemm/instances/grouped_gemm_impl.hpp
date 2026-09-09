// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "../grouped_gemm.hpp"

namespace {

template <typename BaseKernel, int Tag>
struct GroupedGemmKernelTagged : BaseKernel
{
};

[[maybe_unused]] bool grouped_gemm_same_plain_descs(const ck_tile_hcu_grouped_gemm_desc* descs, int group_count)
{
    static std::vector<ck_tile_hcu_grouped_gemm_desc> cached;
    if(static_cast<int>(cached.size()) != group_count)
    {
        return false;
    }
    for(int i = 0; i < group_count; ++i)
    {
        const auto& d = descs[i];
        const auto& c = cached[i];
        if(d.a_ptr != c.a_ptr || d.b_ptr != c.b_ptr || d.c_ptr != c.c_ptr || d.M != c.M ||
           d.N != c.N || d.K != c.K || d.k_batch != c.k_batch || d.stride_A != c.stride_A ||
           d.stride_B != c.stride_B || d.stride_C != c.stride_C)
        {
            return false;
        }
    }
    return true;
}

[[maybe_unused]] void grouped_gemm_update_plain_desc_cache(const ck_tile_hcu_grouped_gemm_desc* descs,
                                          int group_count)
{
    static std::vector<ck_tile_hcu_grouped_gemm_desc> cached;
    cached.assign(descs, descs + group_count);
}

template <typename GemmConfig>
inline constexpr bool grouped_gemm_requires_min_two_k_loops =
    GemmConfig::DoubleSmemBuffer && GemmConfig::Pipeline == CK_TILE_PIPELINE_COMPUTE_V4;

template <typename GemmConfig, typename ADataType, typename GemmDesc>
[[maybe_unused]]
bool grouped_gemm_has_min_two_k_loops(const GemmDesc& desc)
{
    if constexpr(GemmConfig::Pipeline == CK_TILE_PIPELINE_MLS)
    {
        if(desc.K <= 0 || desc.k_batch <= 0)
        {
            return false;
        }

        constexpr ck_tile::index_t packed_size = ck_tile::numeric_traits<ADataType>::PackedSize;
        constexpr ck_tile::index_t k_warp_tile = GemmConfig::K_Warp_Tile;
        const auto k                           = static_cast<ck_tile::index_t>(desc.K);
        const auto k_batch                     = static_cast<ck_tile::index_t>(desc.k_batch);
        const auto k_packed                    = k / packed_size;
        const auto k_grain                     = k_batch * k_warp_tile;
        const auto k_read = ((k + k_grain - 1) / k_grain) * k_warp_tile / packed_size;

        for(ck_tile::index_t split = 0; split < k_batch; ++split)
        {
            const auto split_k = split + 1 < k_batch ? k_read
                                                     : k_packed - k_read * (k_batch - 1);
            if(split_k <= 0)
            {
                return false;
            }
            const auto num_loop =
                (split_k + GemmConfig::K_Tile - 1) / GemmConfig::K_Tile;
            if(num_loop < 4 || num_loop % 2 != 0)
            {
                return false;
            }
        }
        return true;
    }
    else if constexpr(!grouped_gemm_requires_min_two_k_loops<GemmConfig>)
    {
        return true;
    }
    else
    {
        if(desc.K <= 0 || desc.k_batch <= 0)
        {
            return false;
        }

        constexpr ck_tile::index_t packed_size = ck_tile::numeric_traits<ADataType>::PackedSize;
        const auto k                           = static_cast<ck_tile::index_t>(desc.K);
        const auto k_batch                     = static_cast<ck_tile::index_t>(desc.k_batch);
        const auto k_packed                    = k / packed_size;
        if(k_packed <= 0)
        {
            return false;
        }

        ck_tile::index_t min_split_k = k_packed;
        if(k_batch > 1)
        {
            constexpr ck_tile::index_t k_warp_tile = GemmConfig::K_Warp_Tile;
            const auto k_grain                     = k_batch * k_warp_tile;
            const auto k_read =
                ((k + k_grain - 1) / k_grain) * k_warp_tile;
            const auto k_read_packed = k_read / packed_size;
            const auto last_split_k  = k_packed - k_read_packed * (k_batch - 1);
            if(k_read_packed <= 0 || last_split_k <= 0)
            {
                return false;
            }
            min_split_k = std::min(k_read_packed, last_split_k);
        }

        const auto num_loop =
            (min_split_k + GemmConfig::K_Tile - 1) / GemmConfig::K_Tile;
        return num_loop >= 2;
    }
}

template <typename GemmConfig, typename ADataType, typename GemmDesc>
[[maybe_unused]]
bool grouped_gemm_all_groups_have_min_two_k_loops(const std::vector<GemmDesc>& descs)
{
    for(const auto& desc : descs)
    {
        if(!grouped_gemm_has_min_two_k_loops<GemmConfig, ADataType>(desc))
        {
            return false;
        }
    }
    return true;
}

template <typename GemmConfig, typename ALayout, typename BLayout, typename GemmDesc>
[[maybe_unused]]
bool grouped_gemm_has_safe_mls_k_stride(const GemmDesc& desc)
{
    if constexpr(GemmConfig::Pipeline != CK_TILE_PIPELINE_MLS)
    {
        return true;
    }
    else
    {
        if(desc.K <= 0 || desc.stride_A <= 0 || desc.stride_B <= 0)
        {
            return false;
        }

        constexpr auto k_tile      = GemmConfig::K_Tile;
        constexpr auto k_warp_tile = GemmConfig::K_Warp_Tile;
        const auto k_batch          = static_cast<ck_tile::index_t>(desc.k_batch);
        const auto k_grain          = k_batch * k_warp_tile;
        const auto k_read = ((desc.K + k_grain - 1) / k_grain) * k_warp_tile;
        ck_tile::index_t required_k_span = 0;
        for(ck_tile::index_t split = 0; split < k_batch; ++split)
        {
            const auto split_k = split + 1 < k_batch ? k_read
                                                     : desc.K - k_read * (k_batch - 1);
            if(split_k <= 0)
            {
                return false;
            }
            const auto padded_split_k = ((split_k + k_tile - 1) / k_tile) * k_tile;
            required_k_span =
                std::max(required_k_span, split * k_read + padded_split_k);
        }
        const bool a_stride_safe =
            std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>
                ? desc.stride_A >= required_k_span
                : desc.stride_A >= desc.M;
        const bool b_stride_safe =
            std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>
                ? desc.stride_B >= required_k_span
                : desc.stride_B >= desc.N;
        return a_stride_safe && b_stride_safe;
    }
}

template <typename GemmConfig, typename CDataType, typename CLayout, typename GemmDesc>
[[maybe_unused]]
std::vector<std::pair<void*, std::size_t>>
grouped_gemm_make_output_resets(const std::vector<GemmDesc>& descs)
{
    std::vector<std::pair<void*, std::size_t>> resets;
    if constexpr(!GemmConfig::SupportsSplitKOutputAtomic)
    {
        return resets;
    }
    resets.reserve(descs.size());
    for(const auto& desc : descs)
    {
        const auto rows = std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>
                              ? desc.M
                              : desc.N;
        resets.emplace_back(desc.e_ptr,
                            static_cast<std::size_t>(rows) * desc.stride_E * sizeof(CDataType));
    }
    return resets;
}

template <typename GemmConfig, typename ALayout, typename BLayout, typename GemmDesc>
[[maybe_unused]]
bool grouped_gemm_all_groups_have_safe_mls_k_stride(const std::vector<GemmDesc>& descs)
{
    for(const auto& desc : descs)
    {
        if(!grouped_gemm_has_safe_mls_k_stride<GemmConfig, ALayout, BLayout>(desc))
        {
            return false;
        }
    }
    return true;
}

[[maybe_unused, noreturn]] void grouped_gemm_throw_min_two_k_loop_error()
{
    throw std::runtime_error(
        "Current grouped_gemm pipeline does not support the requested per-split K-loop count.");
}

} // namespace

template <typename GemmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          std::uint32_t UniqueKernelTag = 0>
int run_grouped_gemm_c_impl(const ck_tile_hcu_grouped_gemm_desc* descs,
                            int group_count,
                            void* workspace,
                            hipStream_t stream)
{
    if(descs == nullptr || group_count <= 0)
    {
        return -1;
    }

    const int k_batch = descs[0].k_batch;
    const int num_d   = descs[0].num_d_tensors;
    if(k_batch <= 0)
    {
        return -2;
    }

    bool use_multi_d = (num_d > 0 && descs[0].d_ptrs != nullptr && descs[0].stride_Ds != nullptr);

    if(use_multi_d && num_d != 1 && num_d != 2)
    {
        return -6; // only 1 or 2 D tensors supported
    }

    ck_tile::DeviceMem owned_workspace;
    void* kargs_ptr = workspace;
    const auto stream_cfg = ck_tile::stream_config{stream, false, 0, 0, 1};

#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
    // The isolated M128xN64 operand-layout probe already consumes the full
    // 64-KiB LDS budget in its GEMM pipeline. Exclude D-tensor/CShuffle
    // instantiations from this target so that its no-D direct-store path can
    // be measured independently; normal grouped-GEMM targets retain them.
    if(use_multi_d)
    {
        return -7;
    }
#else
    if(use_multi_d)
    {
        if(num_d == 1)
        {
            // Single-D bias: C = GEMM(A,B) + D0
            std::vector<ck_tile::GroupedGemmHostArgsImpl<1>> gemm_descs_vec;
            gemm_descs_vec.reserve(group_count);
            std::vector<ck_tile::GemmTransKernelArgImpl<1>> kargs;
            kargs.reserve(group_count);
            for(int i = 0; i < group_count; ++i)
            {
                const auto& arg = descs[i];
                if(arg.a_ptr == nullptr || arg.b_ptr == nullptr || arg.c_ptr == nullptr || arg.M <= 0 ||
                   arg.N <= 0 || arg.K <= 0 || arg.k_batch != k_batch || arg.num_d_tensors != num_d ||
                   arg.d_ptrs == nullptr || arg.stride_Ds == nullptr)
                {
                    return -3;
                }
                if(!grouped_gemm_has_min_two_k_loops<GemmConfig, ADataType>(arg) ||
                   !grouped_gemm_has_safe_mls_k_stride<GemmConfig, ALayout, BLayout>(arg))
                {
                    return -3;
                }
                std::array<const void*, 1> ds_ptrs = {arg.d_ptrs[0]};
                std::array<ck_tile::index_t, 1> ds_strides = {arg.stride_Ds[0]};
                gemm_descs_vec.emplace_back(arg.a_ptr,
                                            arg.b_ptr,
                                            ds_ptrs,
                                            arg.c_ptr,
                                            arg.k_batch,
                                            arg.M,
                                            arg.N,
                                            arg.K,
                                            arg.stride_A,
                                            arg.stride_B,
                                            ds_strides,
                                            arg.stride_C);
                kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<1, 1, 1>{{arg.a_ptr},
                                                                              {arg.b_ptr},
                                                                              ds_ptrs,
                                                                              arg.c_ptr,
                                                                              arg.M,
                                                                              arg.N,
                                                                              arg.K,
                                                                              {arg.stride_A},
                                                                              {arg.stride_B},
                                                                              ds_strides,
                                                                              arg.stride_C,
                                                                              arg.k_batch});
            }

            if(kargs_ptr == nullptr)
            {
                owned_workspace.Realloc(group_count * sizeof(ck_tile::GemmTransKernelArgImpl<1>));
                kargs_ptr = owned_workspace.GetDeviceBuffer();
            }
            HIP_CHECK_ERROR(
                hipMemcpyWithStream(kargs_ptr,
                                    kargs.data(),
                                    kargs.size() * sizeof(ck_tile::GemmTransKernelArgImpl<1>),
                                    hipMemcpyHostToDevice,
                                    stream_cfg.stream_id_));

            grouped_gemm<GemmConfig,
                         ADataType,
                         BDataType,
                         ck_tile::tuple<CDataType>,
                         AccDataType,
                         CDataType,
                         ALayout,
                         BLayout,
                         ck_tile::tuple<CLayout>,
                         CLayout,
                         ck_tile::element_wise::Add>(
                gemm_descs_vec,
                stream_cfg,
                kargs_ptr);
        }
        else
        {
            // Two-D: C = GEMM(A,B) op D0 op D1
        std::vector<ck_tile::GroupedGemmHostArgsImpl<2>> gemm_descs_vec;
        gemm_descs_vec.reserve(group_count);
        std::vector<ck_tile::GemmTransKernelArgImpl<2>> kargs;
        kargs.reserve(group_count);
        for(int i = 0; i < group_count; ++i)
        {
            const auto& arg = descs[i];
            if(arg.a_ptr == nullptr || arg.b_ptr == nullptr || arg.c_ptr == nullptr || arg.M <= 0 ||
               arg.N <= 0 || arg.K <= 0 || arg.k_batch != k_batch || arg.num_d_tensors != num_d ||
               arg.d_ptrs == nullptr || arg.stride_Ds == nullptr)
            {
                return -3;
            }
            if(!grouped_gemm_has_min_two_k_loops<GemmConfig, ADataType>(arg) ||
               !grouped_gemm_has_safe_mls_k_stride<GemmConfig, ALayout, BLayout>(arg))
            {
                return -3;
            }
            std::array<const void*, 2> ds_ptrs = {arg.d_ptrs[0], arg.d_ptrs[1]};
            std::array<ck_tile::index_t, 2> ds_strides = {arg.stride_Ds[0], arg.stride_Ds[1]};
            gemm_descs_vec.emplace_back(arg.a_ptr,
                                        arg.b_ptr,
                                        ds_ptrs,
                                        arg.c_ptr,
                                        arg.k_batch,
                                        arg.M,
                                        arg.N,
                                        arg.K,
                                        arg.stride_A,
                                        arg.stride_B,
                                        ds_strides,
                                        arg.stride_C);
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<1, 1, 2>{{arg.a_ptr},
                                                                          {arg.b_ptr},
                                                                          ds_ptrs,
                                                                          arg.c_ptr,
                                                                          arg.M,
                                                                          arg.N,
                                                                          arg.K,
                                                                          {arg.stride_A},
                                                                          {arg.stride_B},
                                                                          ds_strides,
                                                                          arg.stride_C,
                                                                          arg.k_batch});
        }

        if(kargs_ptr == nullptr)
        {
            owned_workspace.Realloc(group_count * sizeof(ck_tile::GemmTransKernelArgImpl<2>));
            kargs_ptr = owned_workspace.GetDeviceBuffer();
        }
        HIP_CHECK_ERROR(
            hipMemcpyWithStream(kargs_ptr,
                                kargs.data(),
                                kargs.size() * sizeof(ck_tile::GemmTransKernelArgImpl<2>),
                                hipMemcpyHostToDevice,
                                stream_cfg.stream_id_));

        grouped_gemm<GemmConfig,
                     ADataType,
                     BDataType,
                     ck_tile::tuple<CDataType, CDataType>,
                     AccDataType,
                     CDataType,
                     ALayout,
                     BLayout,
                     ck_tile::tuple<CLayout, CLayout>,
                     CLayout,
                     ck_tile::element_wise::AddAdd>(
            gemm_descs_vec,
            stream_cfg,
            kargs_ptr);
        } // end else (num_d == 2)
    }
    else
#endif
    {
        std::vector<grouped_gemm_kargs> gemm_descs_vec;
        gemm_descs_vec.reserve(group_count);
        for(int i = 0; i < group_count; ++i)
        {
            const auto& arg = descs[i];
            if(arg.a_ptr == nullptr || arg.b_ptr == nullptr || arg.c_ptr == nullptr || arg.M <= 0 ||
               arg.N <= 0 || arg.K <= 0 || arg.k_batch != k_batch)
            {
                return -3;
            }
            if(!grouped_gemm_has_min_two_k_loops<GemmConfig, ADataType>(arg) ||
               !grouped_gemm_has_safe_mls_k_stride<GemmConfig, ALayout, BLayout>(arg))
            {
                return -3;
            }

            gemm_descs_vec.emplace_back(arg.a_ptr,
                                         arg.b_ptr,
                                         arg.c_ptr,
                                         arg.k_batch,
                                         arg.M,
                                         arg.N,
                                         arg.K,
                                         arg.stride_A,
                                         arg.stride_B,
                                         arg.stride_C);
        }

        if(kargs_ptr == nullptr)
        {
            owned_workspace.Realloc(ck_tile_hcu_grouped_gemm_workspace_size(group_count));
            kargs_ptr = owned_workspace.GetDeviceBuffer();
        }

        if constexpr(GemmConfig::Persistent)
        {
            std::vector<ck_tile::GemmTransKernelArg> kargs;
            kargs.reserve(group_count);
            const bool splitk = gemm_descs_vec[0].k_batch > 1;
            for(const auto& arg : gemm_descs_vec)
            {
                kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{arg.a_ptr},
                                                                      {arg.b_ptr},
                                                                      {/*arg.ds_ptr*/},
                                                                      arg.e_ptr,
                                                                      arg.M,
                                                                      arg.N,
                                                                      arg.K,
                                                                      {arg.stride_A},
                                                                      {arg.stride_B},
                                                                      {/*arg.stride_Ds*/},
                                                                      arg.stride_E,
                                                                      arg.k_batch});
            }
            if(!grouped_gemm_same_plain_descs(descs, group_count))
            {
                HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                                    kargs.data(),
                                                    kargs.size() * sizeof(ck_tile::GemmTransKernelArg),
                                                    hipMemcpyHostToDevice,
                                                    stream_cfg.stream_id_));
                grouped_gemm_update_plain_desc_cache(descs, group_count);
            }

            grouped_gemm_tileloop<GemmConfig,
                                  ALayout,
                                  BLayout,
                                  CLayout,
                                  ADataType,
                                  BDataType,
                                  AccDataType,
                                  CDataType,
                                  UniqueKernelTag>(stream_cfg,
                                             group_count,
                                             kargs_ptr,
                                             grouped_gemm_make_output_resets<GemmConfig,
                                                                             CDataType,
                                                                             CLayout>(gemm_descs_vec),
                                             splitk);
        }
        else
        {
            grouped_gemm<GemmConfig,
                         ADataType,
                         BDataType,
                         ck_tile::tuple<>,
                         AccDataType,
                         CDataType,
                         ALayout,
                         BLayout,
                         ck_tile::tuple<>,
                         CLayout,
                         ck_tile::element_wise::PassThrough>(gemm_descs_vec, stream_cfg, kargs_ptr);
        }
    }
    return 0;
}

template <typename GemmConfig,
          typename PrecType,
          std::uint32_t UniqueKernelTag = 0>
int dispatch_grouped_gemm_c_layout(const ck_tile_hcu_grouped_gemm_desc* descs,
                                   int group_count,
                                   char a_layout,
                                   char b_layout,
                                   void* workspace,
                                   hipStream_t stream)
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;

    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using CDataType   = typename Types::CDataType;

    if(a_layout == 'R' && b_layout == 'C')
    {
        return run_grouped_gemm_c_impl<GemmConfig,
                                       Row,
                                       Col,
                                       Row,
                                       ADataType,
                                       BDataType,
                                       AccDataType,
                                       CDataType,
                                       UniqueKernelTag>(descs,
                                                  group_count,
                                                  workspace,
                                                  stream);
    }
    if(a_layout == 'R' && b_layout == 'R')
    {
        if constexpr(GemmConfig::SupportsFastRowMajorB)
        {
            return run_grouped_gemm_c_impl<GemmConfig,
                                           Row,
                                           Row,
                                           Row,
                                           ADataType,
                                           BDataType,
                                           AccDataType,
                                           CDataType,
                                           UniqueKernelTag>(descs,
                                                      group_count,
                                                      workspace,
                                                      stream);
        }
        return -4;
    }
    if(a_layout == 'C' && b_layout == 'R')
    {
        if constexpr(GemmConfig::SupportsFastColumnMajorA &&
                     GemmConfig::SupportsFastRowMajorB)
        {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT)
            // The BLAS-compatible transposed view computes C^T = B^T A^T and
            // writes the same physical row-major result through a column-major
            // [N, M] descriptor.
            return run_grouped_gemm_c_impl<GemmConfig,
                                           Col,
                                           Row,
                                           Col,
                                           ADataType,
                                           BDataType,
                                           AccDataType,
                                           CDataType,
                                           UniqueKernelTag>(descs,
                                                      group_count,
                                                      workspace,
                                                      stream);
#else
            return run_grouped_gemm_c_impl<GemmConfig,
                                           Col,
                                           Row,
                                           Row,
                                           ADataType,
                                           BDataType,
                                           AccDataType,
                                           CDataType,
                                           UniqueKernelTag>(descs,
                                                      group_count,
                                                      workspace,
                                                      stream);
#endif
        }
        return -4;
    }
    if(a_layout == 'C' && b_layout == 'C')
    {
        if constexpr(GemmConfig::SupportsFastColumnMajorA)
        {
            return run_grouped_gemm_c_impl<GemmConfig,
                                           Col,
                                           Col,
                                           Row,
                                           ADataType,
                                           BDataType,
                                           AccDataType,
                                           CDataType,
                                           UniqueKernelTag>(descs,
                                                      group_count,
                                                      workspace,
                                                      stream);
        }
        return -4;
    }
    return -4;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float grouped_gemm(const std::vector<ck_tile::GroupedGemmHostArgsImpl<DsDataType::size()>>&
                       gemm_descs,
                   const ck_tile::stream_config& s,
                   void* kargs_ptr)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    if(!grouped_gemm_all_groups_have_min_two_k_loops<GemmConfig, ADataType>(gemm_descs))
    {
        grouped_gemm_throw_min_two_k_loop_error();
    }

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 GemmConfig::TransposeC,
                                                                 false,
                                                                 false,
                                                                 1,
                                                                 false,
                                                                 false,
                                                                 GemmConfig::SupportsFastRowMajorB,
                                                                 GemmConfig::SupportsFastColumnMajorA>;

    constexpr bool use_fixed_padding_vectors =
        GemmConfig::kPadM || GemmConfig::kPadN || GemmConfig::kPadK;
    constexpr ck_tile::index_t padding_vector_size = 2;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              GemmShape,
                                                              GemmUniversalTraits,
                                                              ADataType,
                                                              use_fixed_padding_vectors,
                                                              padding_vector_size,
                                                              padding_vector_size>;

    using BaseGemmPipeline = typename PipelineTypeTraits<
        GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

    const ck_tile::index_t k_grain = gemm_descs[0].k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split = (gemm_descs[0].K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);


    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = GemmConfig::Scheduler;
        [[maybe_unused]] constexpr auto memory_operation =
            memory_operation_.value;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           has_hot_loop_v,
                                                                           tail_number_v,
                                                                           ck_tile::null_type,
                                                                           ck_tile::null_type,
                                                                           use_fixed_padding_vectors,
                                                                           padding_vector_size,
                                                                           padding_vector_size>;


        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_TILE_DIRECT_STORE)
        using GemmEpilogue =
            GroupedGemmOutputTileDirectStoreEpilogue<GemmPipeline,
                                                      AccDataType,
                                                      CDataType>;
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
        using GemmEpilogue =
            GroupedGemmDirectStoreEpilogue<AccDataType,
                                           CDataType,
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VENDOR_OPERAND_ORDER_V4)
                                           false>;
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
                                           true,
                                           true>;
#else
                                           UniversalGemmProblem::TransposeC>;
#endif
#else
        using CShuffleGemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation,
                                             1,
                                             use_fixed_padding_vectors,
                                             padding_vector_size,
                                             GemmConfig::TiledMMAPermuteN,
                                             GemmConfig::CShuffleWarpGemmMRepeat == 0
                                                 ? GemmConfig::M_Warp_Tile / 16
                                                 : GemmConfig::CShuffleWarpGemmMRepeat,
                                             GemmConfig::CShuffleWarpGemmNRepeat == 0
                                                 ? GemmConfig::N_Warp_Tile / 16
                                                 : GemmConfig::CShuffleWarpGemmNRepeat,
                                             GemmConfig::CShuffleWarpGemmMInterleave == 0
                                                 ? 1
                                                 : GemmConfig::CShuffleWarpGemmMInterleave,
                                             GemmConfig::CShuffleWarpGemmNInterleave == 0
                                                 ? 1
                                                 : GemmConfig::CShuffleWarpGemmNInterleave>>;
#if defined(CK_TILE_GROUPED_GEMM_DIRECT_STORE_EPILOGUE)
        using GemmEpilogue = std::conditional_t<
            std::tuple_size_v<DsDataType> == 0 &&
                memory_operation == ck_tile::memory_operation_enum::set,
            GroupedGemmDirectStoreEpilogue<AccDataType,
                                           CDataType,
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VENDOR_OPERAND_ORDER_V4)
                                           false>,
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
                                           true,
                                           true>,
#else
                                           UniversalGemmProblem::TransposeC>,
#endif
            CShuffleGemmEpilogue>;
#else
        using GemmEpilogue = CShuffleGemmEpilogue;
#endif
#endif

        using KernelBase =
            ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
#if defined(CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG)
        // Macro-only schedules otherwise instantiate the same kentry type in
        // separate HIP translation units. Primus links with
        // --allow-multiple-definition, so the offload linker may silently
        // collapse those device symbols. The empty tag wrapper changes only
        // the launch type/mangled name and keeps KernelBase semantics intact.
        using Kernel = GroupedGemmKernelTagged<
            KernelBase,
            CK_TILE_GROUPED_GEMM_UNIQUE_KERNEL_TAG>;
#else
        using Kernel = KernelBase;
#endif
        auto kargs   = Kernel::MakeKargs(gemm_descs);(void)kargs;
        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Kernel arguments not supported!");
        }

        constexpr dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::GridSize(gemm_descs);

        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            get_workspace_size(gemm_descs),
                                            hipMemcpyHostToDevice,
                                            s.stream_id_));

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ave_time =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       gemm_descs.size()));

        return ave_time;
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(gemm_descs[0].k_batch != 1)
        {
            throw std::runtime_error("multi-D grouped_gemm does not support SplitK");
        }

        Run(has_hot_loop_,
            tail_number_,
            ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{});
    };

    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);

    return ave_time;
}

template <typename GemmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          int UniqueKernelTag,
          ck_tile::GroupedGemmMFilter MFilter,
          ck_tile::index_t MFilterAlign>
float grouped_gemm_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            void* kargs_ptr,
                            const std::vector<std::pair<void*, std::size_t>>& output_resets,
                            bool splitk,
                            std::uint32_t requested_num_cu)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmUniversalTraits =
        ck_tile::PersistentTileGemmUniversalTraits<GemmConfig::kPadM,
                               GemmConfig::kPadN,
                               GemmConfig::kPadK,
                               GemmConfig::DoubleSmemBuffer,
                                ALayout,
                                BLayout,
                                CLayout,
                                GemmConfig::TransposeC,
                                false,
                                false,
                                true,
                               GemmConfig::SupportsFastRowMajorB,
                               GemmConfig::SupportsFastColumnMajorA>;

    constexpr bool use_fixed_padding_vectors =
        GemmConfig::kPadM || GemmConfig::kPadN || GemmConfig::kPadK;
    constexpr ck_tile::index_t padding_vector_size = 2;

    float ave_time{0};

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto scheduler        = GemmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           true,
                                                                           ck_tile::TailNumber::Full,
                                                                           ck_tile::null_type,
                                                                           ck_tile::null_type,
                                                                           use_fixed_padding_vectors,
                                                                           padding_vector_size,
                                                                           padding_vector_size>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_OUTPUT_TILE_DIRECT_STORE)
        using GemmEpilogue =
            GroupedGemmOutputTileDirectStoreEpilogue<GemmPipeline,
                                                      AccDataType,
                                                      CDataType>;
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
        using GemmEpilogue =
            GroupedGemmDirectStoreEpilogue<AccDataType,
                                           CDataType,
#if defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_VENDOR_OPERAND_ORDER_V4)
                                           false>;
#elif defined(CK_TILE_GROUPED_GEMM_GFX936_RR_BLAS_RING)
                                           true,
                                           true>;
#else
                                           UniversalGemmProblem::TransposeC>;
#endif
#else
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation,
                                             1,
                                             use_fixed_padding_vectors,
                                             padding_vector_size,
                                             GemmConfig::TiledMMAPermuteN,
                                             GemmConfig::CShuffleWarpGemmMRepeat == 0
                                                 ? GemmConfig::M_Warp_Tile / 16
                                                 : GemmConfig::CShuffleWarpGemmMRepeat,
                                             GemmConfig::CShuffleWarpGemmNRepeat == 0
                                                 ? GemmConfig::N_Warp_Tile / 16
                                                 : GemmConfig::CShuffleWarpGemmNRepeat,
                                             GemmConfig::CShuffleWarpGemmMInterleave == 0
                                                 ? 1
                                                 : GemmConfig::CShuffleWarpGemmMInterleave,
                                              GemmConfig::CShuffleWarpGemmNInterleave == 0
                                                  ? 1
                                                  : GemmConfig::CShuffleWarpGemmNInterleave>>;
#endif
        using KernelBase =
            ck_tile::GroupedGemmKernel<TilePartitioner,
                                       GemmPipeline,
                                       GemmEpilogue,
                                       MFilter,
                                       MFilterAlign>;
        using Kernel = std::conditional_t<
            UniqueKernelTag == 0,
            KernelBase,
            GroupedGemmKernelTagged<KernelBase, UniqueKernelTag>>;
        constexpr dim3 blocks = Kernel::BlockSize();
        dim3 grids = Kernel::MaxOccupancyGridSize(s);
        if(requested_num_cu > 0)
        {
            grids.x = std::min(
                grids.x,
                static_cast<decltype(grids.x)>(
                    requested_num_cu * GemmConfig::kBlockPerCu));
        }
#if defined(CK_TILE_GROUPED_GEMM_GRID_MULTIPLIER_FIXED)
        // The promoted tag93714 uses a fixed two-block-per-CU traversal.
        // Extra block ids safely exit after scanning the group list, without
        // changing the public descriptor ABI.
        static_assert(CK_TILE_GROUPED_GEMM_GRID_MULTIPLIER_FIXED > 0 &&
                          CK_TILE_GROUPED_GEMM_GRID_MULTIPLIER_FIXED <= 128,
                      "fixed grouped-GEMM grid multiplier must be in [1,128]");
        grids.x *= CK_TILE_GROUPED_GEMM_GRID_MULTIPLIER_FIXED;
#elif defined(CK_TILE_GROUPED_GEMM_GRID_SWEEP_RUNTIME)
        // Performance-probe only: queue additional persistent blocks while
        // keeping the device kernel and its grid-stride tile traversal intact.
        // The default multiplier is one, so production behavior is unchanged.
        if(const char* value = std::getenv("CK_TILE_GROUPED_GEMM_GRID_MULTIPLIER"))
        {
            const long multiplier = std::strtol(value, nullptr, 10);
            if(multiplier > 1 && multiplier <= 16)
            {
                grids.x *= static_cast<unsigned int>(multiplier);
            }
        }
#endif

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        const auto kernel = ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
            Kernel{},
            grids,
            blocks,
            0,
            ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
            num_groups);
        if constexpr(memory_operation == ck_tile::memory_operation_enum::atomic_add &&
                     GemmConfig::SupportsSplitKOutputAtomic)
        {
            const auto reset_outputs = [output_resets](const ck_tile::stream_config& stream) {
                for(const auto& [ptr, bytes] : output_resets)
                {
                    HIP_CHECK_ERROR(hipMemsetAsync(ptr, 0, bytes, stream.stream_id_));
                }
            };
            ave_time = ck_tile::launch_kernel(s, reset_outputs, kernel);
        }
        else
        {
            ave_time = ck_tile::launch_kernel(s, kernel);
        }

        return ave_time;
    };

#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
    if(splitk)
    {
        throw std::runtime_error("Triton operand-layout probe does not support SplitK");
    }
    Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                   ck_tile::memory_operation_enum::set>{});
#else
    if constexpr(std::is_same_v<CDataType, ck_tile::half_t> ||
                 std::is_same_v<CDataType, ck_tile::bf16_t>)
    {
        if(splitk)
        {
            if constexpr(GemmConfig::SupportsSplitKOutputAtomic)
            {
                Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                               ck_tile::memory_operation_enum::atomic_add>{});
                return ave_time;
            }
            else
            {
                throw std::runtime_error("fp16/bf16 grouped_gemm output does not support SplitK");
            }
        }

        Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{});
    }
    else if(!splitk)
    {
        Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        throw std::runtime_error(
            "fp32/int32 grouped_gemm output does not support SplitK vector atomic");
    }
#endif

    return ave_time;
}

#include "../run_grouped_gemm_example.inc"

template <typename GemmConfig, typename PrecType>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;
    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using CDataType   = typename Types::CDataType;
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }
    const bool multiple_d      = arg_parser.get_bool("multiple_d");
    const std::string md_op_str = arg_parser.get_str("multiple_d_op");
    const bool md_multiply      = (md_op_str == "multiply");
    const bool bias             = arg_parser.get_bool("bias");

#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_D)
    (void)md_multiply;
    auto run_multi_d = [&](auto, auto, auto) { return -7; };
    auto run_bias    = [&](auto, auto, auto) { return -7; };
#else
    auto run_multi_d = [&](auto a_layout_tag, auto b_layout_tag, auto c_layout_tag) {
        if(md_multiply)
        {
            return run_grouped_gemm_multiple_d_example_with_layouts<GemmConfig,
                                                                    ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    AccDataType,
                                                                    ck_tile::element_wise::MultiplyMultiply>(
                argc, argv, a_layout_tag, b_layout_tag, c_layout_tag);
        }
        return run_grouped_gemm_multiple_d_example_with_layouts<GemmConfig,
                                                                ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                AccDataType,
                                                                ck_tile::element_wise::AddAdd>(
            argc, argv, a_layout_tag, b_layout_tag, c_layout_tag);
    };

    auto run_bias = [&](auto a_layout_tag, auto b_layout_tag, auto c_layout_tag) {
        return run_grouped_gemm_bias_example_with_layouts<GemmConfig,
                                                          ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          AccDataType>(
            argc, argv, a_layout_tag, b_layout_tag, c_layout_tag);
    };
#endif

#if defined(CK_TILE_GROUPED_GEMM_FAST_RC_ONLY)
    if(a_layout == "R" && b_layout == "C")
    {
        if(bias)
        {
            return run_bias(Row{}, Col{}, Row{});
        }
        if(multiple_d)
        {
            return run_multi_d(Row{}, Col{}, Row{});
        }
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Row{}, Col{}, Row{});
    }
    if constexpr(GemmConfig::SupportsFastRowMajorB)
    {
        if(a_layout == "R" && b_layout == "R")
        {
            if(bias || multiple_d)
            {
                throw std::runtime_error(
                    "bias/multiple-D currently supports A row-major and B column-major.");
            }
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
                argc, argv, Row{}, Row{}, Row{});
        }
    }
    if constexpr(GemmConfig::SupportsFastColumnMajorA &&
                 GemmConfig::SupportsFastRowMajorB)
    {
        if(a_layout == "C" && b_layout == "R")
        {
            if(bias || multiple_d)
            {
                throw std::runtime_error(
                    "bias/multiple-D currently supports A row-major and B column-major.");
            }
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT)
                argc, argv, Col{}, Row{}, Col{});
#else
                argc, argv, Col{}, Row{}, Row{});
#endif
        }
    }
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT)
    if constexpr(GemmConfig::SupportsFastColumnMajorA)
    {
        if(a_layout == "C" && b_layout == "C")
        {
            if(bias || multiple_d)
            {
                throw std::runtime_error(
                    "bias/multiple-D currently supports A row-major and B column-major.");
            }
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
                argc, argv, Col{}, Col{}, Col{});
        }
    }
#endif
    throw std::runtime_error(
        GemmConfig::SupportsFastColumnMajorA
            ? "Fast grouped_gemm target supports NT, NN, TN, and gated TT layouts only."
            : GemmConfig::SupportsFastRowMajorB
                  ? "Fast grouped_gemm target only supports A row-major with B row-major or column-major."
                  : "Fast grouped_gemm target only supports A row-major and B column-major.");
#else

    if(a_layout == "R" && b_layout == "C")
    {
        if(bias)
        {
            return run_bias(Row{}, Col{}, Row{});
        }
        if(multiple_d)
        {
            return run_multi_d(Row{}, Col{}, Row{});
        }
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Row{}, Col{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "R")
    {
        if constexpr(GemmConfig::SupportsFastRowMajorB)
        {
            if(bias || multiple_d)
            {
                throw std::runtime_error(
                    "bias/multiple-D currently supports A row-major and B column-major.");
            }
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
                argc, argv, Row{}, Row{}, Row{});
        }
        throw std::runtime_error("B row-major is not supported for this grouped_gemm config.");
    }
    else if(a_layout == "C" && b_layout == "R")
    {
        if constexpr(GemmConfig::SupportsFastColumnMajorA &&
                     GemmConfig::SupportsFastRowMajorB)
        {
            if(bias || multiple_d)
            {
                throw std::runtime_error(
                    "bias/multiple-D currently supports A row-major and B column-major.");
            }
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_COLUMN_MAJOR_IMMEDIATE_SCALAR_SHORT)
                argc, argv, Col{}, Row{}, Col{});
#else
                argc, argv, Col{}, Row{}, Row{});
#endif
        }
        throw std::runtime_error("B row-major is not supported for this grouped_gemm config.");
    }
    else if(a_layout == "C" && b_layout == "C")
    {
        if constexpr(GemmConfig::SupportsFastColumnMajorA)
        {
            if(bias || multiple_d)
            {
                throw std::runtime_error(
                    "bias/multiple-D currently supports A row-major and B column-major.");
            }
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         AccDataType>(argc,
                                                                      argv,
                                                                      Col{},
                                                                      Col{},
                                                                      Row{});
        }
        throw std::runtime_error("A column-major is not supported for this grouped_gemm config.");
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A and B tensors!");
    }
#endif
}
