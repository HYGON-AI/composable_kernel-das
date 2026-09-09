// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "fused_moegemm_api_traits.hpp"
#include "ck_tile/ops/fused_moe.hpp"
#include <iostream>

template <ck_tile::index_t... Is>
using S = ck_tile::sequence<Is...>;

// do not the define of this tepmlate function inside the _api.cpp, otherwise will block make -j
template <typename Ts_>
float fused_moegemm_(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave,
                                                 ck_tile::FusedMoeGemmWeightPermuteEnum::no_permute,
                                                 Ts_::swizzled,
                                                 true /* pad hidden size */,
                                                 false /* pad intermediate size */>;
    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits>;

    using f_pipeline = ck_tile::FusedMoeGemmPipeline_FlatmmEx<f_problem>;
    // using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<
        f_shape,
        Ts_::IntermediateTilesPerBlock,
        Ts_::DebugWriterCount>;
    using f_kernel      = ck_tile::FusedMoeGemmKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    float elapsed_ms = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));

    if constexpr(Ts_::IntermediateTilesPerBlock != 6)
    {
        if(a.topk == 1 && a.deterministic_tmp_ptr != nullptr)
        {
            using reduce_kernel = ck_tile::FusedMoeDeterministicReduceKernel<
                double,
                typename Ts_::ODataType>;
            constexpr ck_tile::index_t num_planes = 1;
            const ck_tile::long_index_t element_count =
                static_cast<ck_tile::long_index_t>(a.num_tokens) * a.stride_token;
            const typename reduce_kernel::Kargs reduce_args{
                reinterpret_cast<const double*>(a.deterministic_tmp_ptr),
                reinterpret_cast<typename Ts_::ODataType*>(a.o_ptr),
                element_count,
                element_count,
                num_planes};
            const dim3 reduce_grids  = reduce_kernel::GridSize(reduce_args);
            constexpr dim3 reduce_blocks = reduce_kernel::BlockSize();
            elapsed_ms += ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<reduce_blocks.x, 1>(
                    reduce_kernel{}, reduce_grids, reduce_blocks, 0, reduce_args));
        }
    }
    return elapsed_ms;
}

template <typename Ts_>
float fused_moegemm_swizzled_lds_(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave,
                                                 ck_tile::FusedMoeGemmWeightPermuteEnum::no_permute,
                                                 Ts_::swizzled>;
    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits>;

    using f_pipeline = ck_tile::FusedMoeGemmPipeline_Flatmm_GU_Regs_D_LDS<f_problem>;
    // using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}

template <typename Ts_>
float fused_moegemm_wreg_(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave>;
    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits>;

    using f_pipeline = ck_tile::FusedMoeGemmPipeline_Flatmm_Wreg<f_problem>;
    // using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}


template <typename Ts_>
float fused_moegemm_wreg_shuffle_(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave,
                                                 ck_tile::FusedMoeGemmWeightPermuteEnum::permute_b>;


    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits>;
    
    using f_pipeline = ck_tile::FusedMoeGemmPipeline_Flatmm_Wreg_shuffle<f_problem>;
    // using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmWShuffleKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}


template <typename Ts_>
float fused_moegemm_wt_bd_shuffle_(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave,
                                                 ck_tile::FusedMoeGemmWeightPermuteEnum::permute_bd>;


    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits>;
    
    using f_pipeline = ck_tile::FusedMoeGemmPipeline_Flatmm_Wt_shuffle<f_problem>;
    // using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmWWtBDShuffleKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}


template <typename Ts_>
float fused_moegemm_wt_shuffle_quant(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave,
                                                 ck_tile::FusedMoeGemmWeightPermuteEnum::permute_bd>;


    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits,
                                                           Ts_::FusedQuant,
                                                           Ts_::QuantBlockSizeN,
                                                           Ts_::QuantBlockSizeK>;
    
    using f_pipeline = ck_tile::FusedMoeGemmPipeline_Flatmm_Wt_shuffle_quant<f_problem>;
    // using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmWWtBDShuffleQuantKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}

template <typename Ts_>
float fused_moegemm_wt_w8a8_quant(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave,
                                                 ck_tile::FusedMoeGemmWeightPermuteEnum::no_permute>;


    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits,
                                                           Ts_::FusedQuant,
                                                           Ts_::QuantBlockSizeN,
                                                           Ts_::QuantBlockSizeK>;
    
    using f_pipeline = ck_tile::FusedMoeGemmPipeline_Flatmm_W8A8_quant<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmWWtW8A8QuantKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}


template <typename Ts_>
float fused_moegemm_wt_w4a16_quant(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave,
                                                 ck_tile::FusedMoeGemmWeightPermuteEnum::no_permute>;


    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits,
                                                           Ts_::FusedQuant,
                                                           Ts_::QuantBlockSizeN,
                                                           Ts_::QuantBlockSizeK>;
    
    using f_pipeline = ck_tile::FusedMoeGemmPipeline_Flatmm_W4A16_quant<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmWWtW4A16QuantKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}

template <typename Ts_>
float fused_moegemm_wt_w4a8_quant(const ck_tile::stream_config& s, fused_moegemm_args a)
{
    using f_traits = ck_tile::FusedMoeGemmTraits<Ts_::GateOnly,
                                                 Ts_::FusedQuant == 1,
                                                 2 /*atomic*/,
                                                 typename Ts_::GmemLoadVectorLengths,
                                                 typename Ts_::SmemStoreVectorLengths,
                                                 typename Ts_::SmemLoadVectorLengths,
                                                 Ts_::OGmemStoreVectorLength,
                                                 Ts_::Gemm0NInterleave,
                                                 Ts_::Gemm1NInterleave,
                                                 ck_tile::FusedMoeGemmWeightPermuteEnum::no_permute>;


    using f_shape  = ck_tile::FusedMoeGemmShape<typename Ts_::BlockTile_0,
                                               typename Ts_::WarpPerBlock_0,
                                               typename Ts_::WarpTile_0,
                                               typename Ts_::BlockTile_1,
                                               typename Ts_::WarpPerBlock_1,
                                               typename Ts_::WarpTile_1>;

    constexpr auto get_activation_ = []() {
        if constexpr(Ts_::Activation == 0)
        {
            return ck_tile::element_wise::Gelu{};
        }
        else
            return ck_tile::element_wise::Silu{};
    };
    using f_act_ = ck_tile::remove_cvref_t<decltype(get_activation_())>;

    using f_problem = ck_tile::FusedMoeGemmPipelineProblem<typename Ts_::ADataType,
                                                           typename Ts_::GDataType,
                                                           typename Ts_::DDataType,
                                                           typename Ts_::AccDataType,
                                                           typename Ts_::ODataType,
                                                           typename Ts_::AScaleDataType,
                                                           typename Ts_::GScaleDataType,
                                                           typename Ts_::DScaleDataType,
                                                           typename Ts_::GZeroPointDataType,
                                                           typename Ts_::DZeroPointDataType,
                                                           typename Ts_::YSmoothScaleDataType,
                                                           typename Ts_::TopkWeightDataType,
                                                           typename Ts_::IndexDataType,
                                                           f_act_, // TODO: hardcoded
                                                           f_shape,
                                                           f_traits,
                                                           Ts_::FusedQuant,
                                                           Ts_::QuantBlockSizeN,
                                                           Ts_::QuantBlockSizeK>;
    
    using f_pipeline = ck_tile::FusedMoeGemmPipeline_Flatmm_W4A8_quant<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmWWtW4A8QuantKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
}
