// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp"

namespace ck_tile {
namespace detail {

template <typename Partitioner, typename Pipeline, typename Epilogue>
struct SingleSplitGemmKernel : UniversalGemmKernel<Partitioner, Pipeline, Epilogue>
{
    using Base = UniversalGemmKernel<Partitioner, Pipeline, Epilogue>;
    using Args = typename Base::KernelArgs;
    static constexpr bool kUsesMinBlockPerCuLaunchApi = true;

    CK_TILE_HOST static bool IsSupportedArgument(const Args& args)
    {
        using Row = tensor_layout::gemm::RowMajor;
        constexpr bool a_row = std::is_same_v<typename Pipeline::ALayout, Row>;
        constexpr bool b_row = std::is_same_v<typename Pipeline::BLayout, Row>;
        constexpr bool c_row = std::is_same_v<typename Pipeline::CLayout, Row>;
        return args.k_batch == 1 && args.M > 0 && args.N > 0 && args.K > 0 &&
               args.as_ptr[0] && args.bs_ptr[0] && args.e_ptr &&
               args.stride_As[0] >= (a_row ? args.K : args.M) &&
               args.stride_Bs[0] >= (b_row ? args.N : args.K) &&
               args.stride_E >= (c_row ? args.N : args.M) &&
               args.stride_As[0] % Pipeline::template GetVectorSizeA<>() == 0 &&
               args.stride_Bs[0] % Pipeline::template GetVectorSizeB<>() == 0 &&
               args.stride_E % Epilogue::GetVectorSizeC() == 0 &&
               Base::IsSupportedArgument(args);
    }
};

} // namespace detail

template <typename Shape, index_t GroupNum, index_t M01, typename Problem, typename Policy, typename Epilogue>
struct GemmKernel<GemmSpatiallyLocalTilePartitioner<Shape, GroupNum, M01>,
                  GemmPipelineAgBgCrCompV3<Problem, Policy>,
                  Epilogue>
    : detail::SingleSplitGemmKernel<GemmSpatiallyLocalTilePartitioner<Shape, GroupNum, M01>,
                                    GemmPipelineAgBgCrCompV3<Problem, Policy>,
                                    Epilogue>
{
};

template <typename Shape, index_t GroupNum, index_t M01, typename Problem, typename Policy, typename Epilogue>
struct GemmKernel<GemmSpatiallyLocalTilePartitioner<Shape, GroupNum, M01>,
                  GemmPipelineAgBgCrCompV4<Problem, Policy>,
                  Epilogue>
    : detail::SingleSplitGemmKernel<GemmSpatiallyLocalTilePartitioner<Shape, GroupNum, M01>,
                                    GemmPipelineAgBgCrCompV4<Problem, Policy>,
                                    Epilogue>
{
};

template <typename Shape, index_t GroupNum, index_t M01, typename Problem, typename Policy, typename Epilogue>
struct GemmKernel<GemmSpatiallyLocalTilePartitioner<Shape, GroupNum, M01>,
                  GemmPipelineAgBgCrMem<Problem, Policy>,
                  Epilogue>
    : detail::SingleSplitGemmKernel<GemmSpatiallyLocalTilePartitioner<Shape, GroupNum, M01>,
                                    GemmPipelineAgBgCrMem<Problem, Policy>,
                                    Epilogue>
{
};
} // namespace ck_tile
