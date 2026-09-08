// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// The gfx938 large-shape default below uses the validated K32 single-stage MLS
// pipeline. Keep these definitions source-local so unrelated module_cpp_api
// kernels retain their existing compilation contract.
#if !defined(CK_TILE_GROUPED_GEMM_DISABLE_DEFAULT_MLS_PIPELINE) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_MLS_256) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_REGPREFETCH_256_W16_GROUP8) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_W8_OVERLAP) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE) && \
    !defined(CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
#ifndef CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE
#define CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE
#endif
#ifndef CK_TILE_GROUPED_GEMM_MLS_OVERLAP_GLOBAL_LOAD
#define CK_TILE_GROUPED_GEMM_MLS_OVERLAP_GLOBAL_LOAD
#endif
#ifndef CK_TILE_GROUPED_GEMM_MLS_SKIP_POST_MMAC_SYNC
#define CK_TILE_GROUPED_GEMM_MLS_SKIP_POST_MMAC_SYNC
#endif
#endif

#include "grouped_gemm_impl.hpp"

#if defined(CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND)
template <typename PrecType>
using GroupedGemmBf16Config =
#if defined(CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C_NONPERSISTENT)
    GemmConfigComputeV3TritonOperand256ExactTraversalTransposeCNonPersistent<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_NONPERSISTENT)
    GemmConfigComputeV3TritonOperand256ExactTraversalNonPersistent<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C)
    GemmConfigComputeV3TritonOperand256ExactTraversalTransposeC<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL)
    GemmConfigComputeV3TritonOperand256ExactTraversal<PrecType>;
#else
    GemmConfigComputeV3TritonOperand256Aligned<PrecType>;
#endif
template <typename PrecType>
using GroupedGemmBf16PaddedConfig =
#if defined(CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C_NONPERSISTENT)
    GemmConfigComputeV3TritonOperand256ExactTraversalTransposeCNonPersistent<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_NONPERSISTENT)
    GemmConfigComputeV3TritonOperand256ExactTraversalNonPersistent<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_RR_BLAS_TRANSPOSE_C)
    GemmConfigComputeV3TritonOperand256ExactTraversalTransposeC<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_EXACT_TILE_TRAVERSAL)
    GemmConfigComputeV3TritonOperand256ExactTraversal<PrecType>;
#else
    GemmConfigComputeV3TritonOperand256Aligned<PrecType>;
#endif
#elif defined(CK_TILE_GROUPED_GEMM_TUNE_V3_W8_OVERLAP)
template <typename PrecType>
using GroupedGemmBf16Config =
    GemmConfigComputeV3W8Overlap256Aligned<PrecType>;
template <typename PrecType>
using GroupedGemmBf16PaddedConfig =
    GemmConfigComputeV3W8Overlap256Aligned<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM)
template <typename PrecType>
using GroupedGemmBf16Config =
    GemmConfigComputeV3RegPrefetch256W16Group8Dsreadm<PrecType>;
template <typename PrecType>
using GroupedGemmBf16PaddedConfig =
    GemmConfigComputeV3RegPrefetch256W16Group8Dsreadm<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE)
template <typename PrecType>
using GroupedGemmBf16Config =
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_STAGE_NONPERSISTENT)
    GemmConfigComputeV3DsreadmStage256W8NonPersistent<PrecType>;
#else
    GemmConfigComputeV3DsreadmStage256W8<PrecType>;
#endif
template <typename PrecType>
using GroupedGemmBf16PaddedConfig =
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_STAGE_NONPERSISTENT)
    GemmConfigComputeV3DsreadmStage256W8NonPersistent<PrecType>;
#else
    GemmConfigComputeV3DsreadmStage256W8<PrecType>;
#endif
#elif defined(CK_TILE_GROUPED_GEMM_TUNE_V3_REGPREFETCH_256_W16_GROUP8)
template <typename PrecType>
using GroupedGemmBf16Config =
    GemmConfigComputeV3RegPrefetch256W16Group8Aligned<PrecType>;
template <typename PrecType>
using GroupedGemmBf16PaddedConfig =
    GemmConfigComputeV3RegPrefetch256W16Group8Aligned<PrecType>;
#elif defined(CK_TILE_GROUPED_GEMM_TUNE_MLS_256)
template <typename PrecType>
using GroupedGemmBf16Config = GemmConfigMls256<PrecType>;
template <typename PrecType>
using GroupedGemmBf16PaddedConfig = GemmConfigMls256<PrecType>;
#else
template <typename PrecType>
using GroupedGemmBf16Config = GemmConfigComputeV4<PrecType>;
template <typename PrecType>
using GroupedGemmBf16PaddedConfig = GemmConfigComputeV4Mpad<PrecType>;
#endif

#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY_ONLY)
int grouped_gemm_c_run_bf16(const ck_tile_hcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream)
{
#if !defined(CK_TILE_GROUPED_GEMM_TUNE_MLS_256) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_REGPREFETCH_256_W16_GROUP8) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_W8_OVERLAP) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE) && \
    !defined(CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || \
    defined(CK_TILE_GROUPED_GEMM_ENABLE_GFX936_SELECTED)
    if(ck_tile::get_hcu_target_enum() == ck_tile::hcu_target_enum::gfx936 &&
       ((a_layout == 'R' && b_layout == 'C') ||
        (a_layout == 'R' && b_layout == 'R') ||
        (a_layout == 'C' && b_layout == 'R')) &&
       grouped_gemm_prefers_gfx936_direct_config<
           GemmConfigComputeV3TritonOperand256ExactTraversal<ck_tile::bf16_t>>(
           descs, group_count))
    {
        return grouped_gemm_c_run_gfx936_bf16(
            descs, group_count, a_layout, b_layout, workspace, stream);
    }
#endif
    if(ck_tile::get_hcu_target_enum() == ck_tile::hcu_target_enum::gfx936 &&
       a_layout == 'R' && b_layout == 'C' &&
       group_count > 0 &&
       grouped_gemm_prefers_large_config<
           GemmConfigComputeV3RegPrefetch256W16Group8Aligned<ck_tile::bf16_t>>(
           descs, group_count, 2048))
    {
        return dispatch_grouped_gemm_c_layout<
            GemmConfigComputeV3RegPrefetch256W16Group8Aligned<ck_tile::bf16_t>,
            ck_tile::bf16_t>(
            descs, group_count, a_layout, b_layout, workspace, stream);
    }

    if(ck_tile::get_hcu_target_enum() == ck_tile::hcu_target_enum::gfx938 &&
       grouped_gemm_prefers_large_config<GemmConfigMls256<ck_tile::bf16_t>>(
           descs, group_count, 2048))
    {
        return dispatch_grouped_gemm_c_layout<GemmConfigMls256<ck_tile::bf16_t>,
                                              ck_tile::bf16_t>(
            descs, group_count, a_layout, b_layout, workspace, stream);
    }

    if(grouped_gemm_prefers_large_config<GemmConfigComputeV4SquareWide<ck_tile::bf16_t>>(
           descs, group_count))
    {
        const auto hcu_target = ck_tile::get_hcu_target_enum();
        if(hcu_target == ck_tile::hcu_target_enum::gfx936)
        {
            if(a_layout == 'R' && b_layout == 'R')
            {
                return dispatch_grouped_gemm_c_layout<
                    GemmConfigComputeV4SquareM4<ck_tile::bf16_t>,
                    ck_tile::bf16_t>(
                    descs, group_count, a_layout, b_layout, workspace, stream);
            }
            if(a_layout == 'C' && b_layout == 'R')
            {
                return dispatch_grouped_gemm_c_layout<
                    GemmConfigComputeV4SquareM4M01_4NonPersistent<ck_tile::bf16_t>,
                    ck_tile::bf16_t>(
                    descs, group_count, a_layout, b_layout, workspace, stream);
            }
            return dispatch_grouped_gemm_c_layout<
                GemmConfigComputeV4SquareM4M01_4<ck_tile::bf16_t>,
                ck_tile::bf16_t>(
                descs, group_count, a_layout, b_layout, workspace, stream);
        }
        if(hcu_target == ck_tile::hcu_target_enum::gfx938)
        {
            return dispatch_grouped_gemm_c_layout<GemmConfigComputeV4SquareM4<ck_tile::bf16_t>,
                                                  ck_tile::bf16_t>(
                descs, group_count, a_layout, b_layout, workspace, stream);
        }
        return dispatch_grouped_gemm_c_layout<GemmConfigComputeV4SquareWide<ck_tile::bf16_t>,
                                              ck_tile::bf16_t>(
            descs, group_count, a_layout, b_layout, workspace, stream);
    }
#endif
    return dispatch_grouped_gemm_c_layout<GroupedGemmBf16Config<ck_tile::bf16_t>,
                                          ck_tile::bf16_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_c_run_bf16_mpad(const ck_tile_hcu_grouped_gemm_desc* descs,
                                  int group_count,
                                  char a_layout,
                                  char b_layout,
                                  void* workspace,
                                  hipStream_t stream)
{
    return dispatch_grouped_gemm_c_layout<GroupedGemmBf16PaddedConfig<ck_tile::bf16_t>,
                                          ck_tile::bf16_t>(
        descs, group_count, a_layout, b_layout, workspace, stream);
}

int grouped_gemm_example_run_bf16(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    if(grouped_gemm_example_needs_padding<GroupedGemmBf16Config<ck_tile::bf16_t>>(argc, argv))
    {
        return run_gemm_example_prec_type<GroupedGemmBf16PaddedConfig<ck_tile::bf16_t>,
                                          ck_tile::bf16_t>(a_layout,
                                                           b_layout,
                                                           argc,
                                                           argv);
    }
#if !defined(CK_TILE_GROUPED_GEMM_TUNE_MLS_256) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_REGPREFETCH_256_W16_GROUP8) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_W8_OVERLAP) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_TRITON_OPERAND) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM) && \
    !defined(CK_TILE_GROUPED_GEMM_TUNE_V3_DSREADM_STAGE) && \
    !defined(CK_TILE_GROUPED_GEMM_FORCE_BASELINE)
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD)
    if(ck_tile::get_hcu_target_enum() == ck_tile::hcu_target_enum::gfx936 &&
       ((a_layout == "R" && b_layout == "C") ||
        (a_layout == "R" && b_layout == "R") ||
        (a_layout == "C" && b_layout == "R")) &&
       grouped_gemm_example_prefers_gfx936_direct_config<
           GemmConfigComputeV3TritonOperand256ExactTraversal<ck_tile::bf16_t>>(
           argc, argv))
    {
        return grouped_gemm_example_run_gfx936_bf16(
            a_layout, b_layout, argc, argv);
    }
#endif
    if(ck_tile::get_hcu_target_enum() == ck_tile::hcu_target_enum::gfx936 &&
       a_layout == "R" && b_layout == "C" &&
       grouped_gemm_example_prefers_large_config<
           GemmConfigComputeV3RegPrefetch256W16Group8Aligned<ck_tile::bf16_t>>(
           argc, argv, 2048))
    {
        return run_gemm_example_prec_type<
            GemmConfigComputeV3RegPrefetch256W16Group8Aligned<ck_tile::bf16_t>,
            ck_tile::bf16_t>(a_layout, b_layout, argc, argv);
    }

    if(ck_tile::get_hcu_target_enum() == ck_tile::hcu_target_enum::gfx938 &&
       grouped_gemm_example_prefers_large_config<GemmConfigMls256<ck_tile::bf16_t>>(
           argc, argv, 2048))
    {
        return run_gemm_example_prec_type<GemmConfigMls256<ck_tile::bf16_t>,
                                          ck_tile::bf16_t>(
            a_layout, b_layout, argc, argv);
    }

    if(grouped_gemm_example_prefers_large_config<GemmConfigComputeV4SquareWide<ck_tile::bf16_t>>(
           argc, argv))
    {
        const auto hcu_target = ck_tile::get_hcu_target_enum();
        if(hcu_target == ck_tile::hcu_target_enum::gfx936)
        {
            if(a_layout == "R" && b_layout == "R")
            {
                return run_gemm_example_prec_type<
                    GemmConfigComputeV4SquareM4<ck_tile::bf16_t>,
                    ck_tile::bf16_t>(a_layout, b_layout, argc, argv);
            }
            if(a_layout == "C" && b_layout == "R")
            {
                return run_gemm_example_prec_type<
                    GemmConfigComputeV4SquareM4M01_4NonPersistent<ck_tile::bf16_t>,
                    ck_tile::bf16_t>(a_layout, b_layout, argc, argv);
            }
            return run_gemm_example_prec_type<
                GemmConfigComputeV4SquareM4M01_4<ck_tile::bf16_t>,
                ck_tile::bf16_t>(a_layout, b_layout, argc, argv);
        }
        if(hcu_target == ck_tile::hcu_target_enum::gfx938)
        {
            return run_gemm_example_prec_type<GemmConfigComputeV4SquareM4<ck_tile::bf16_t>,
                                              ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        return run_gemm_example_prec_type<GemmConfigComputeV4SquareWide<ck_tile::bf16_t>,
                                          ck_tile::bf16_t>(
            a_layout, b_layout, argc, argv);
    }
#endif
    return run_gemm_example_prec_type<GroupedGemmBf16Config<ck_tile::bf16_t>, ck_tile::bf16_t>(
        a_layout, b_layout, argc, argv);
}
#endif

#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_REGISTRY)
#include "grouped_gemm_device_args_provider.hpp"

extern "C" int ck_tile_hcu_grouped_gemm_registry_bf16_device_args(
    std::uint32_t instance_id,
    std::uint32_t layout,
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return grouped_gemm_device_args_detail::launch_registry_candidate<ck_tile::bf16_t>(
        static_cast<ck_tile::GroupedGemmInstanceId>(instance_id),
        static_cast<ck_tile::GroupedGemmLayout>(layout),
        device_args,
        group_count,
        num_cu,
        stream);
}
#endif
