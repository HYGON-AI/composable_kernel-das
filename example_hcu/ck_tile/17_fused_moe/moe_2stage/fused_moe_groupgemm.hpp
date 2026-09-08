#pragma once

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/ops/fused_moe_2stage/kernel/fmoe_2stage_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp"



#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2
#define CK_TILE_PIPELINE_COMPUTE_V4 3
#define CK_TILE_PIPELINE_PRESHUFFLE_V2 4

#ifndef CK_TILE_PIPELINE_DEFAULT
#define CK_TILE_PIPELINE_DEFAULT CK_TILE_PIPELINE_COMPUTE_V3
#endif


template <typename DataType>
struct GemmTypeConfig;

template <>
struct GemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    // using CDataType   = ck_tile::half_t;     // gpu does not support half atomic add
    using CDataType   = float;
    using AccDataType = float;
    using AScaleDataType = ck_tile::null_type;
    using BScaleDataType = ck_tile::null_type;
};


struct Gemm1TypeConfigFp8W8A8
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    // using CDataType   = float;
    using CDataType = ck_tile::half_t;      //fused silu_and_mul
    using AScaleDataType = float;
    using BScaleDataType = float;
};

struct Gemm2TypeConfigFp8W8A8
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = float;
    using AScaleDataType = float;
    using BScaleDataType = float;
};

struct Gemm1TypeConfigHalf
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using CDataType   = ck_tile::half_t;
    using AccDataType = float;
    using AScaleDataType = ck_tile::null_type;
    using BScaleDataType = ck_tile::null_type;
};

struct Gemm1TypeConfigInt8W8A8
{
    using ADataType   = ck_tile::int8_t;
    using BDataType   = ck_tile::int8_t;
    using AccDataType = float;
    // using CDataType   = float;
    using CDataType = ck_tile::half_t;      //fused silu_and_mul
    using AScaleDataType = float;
    using BScaleDataType = float;
};

struct Gemm2TypeConfigInt8W8A8
{
    using ADataType   = ck_tile::int8_t;
    using BDataType   = ck_tile::int8_t;
    using AccDataType = float;
    using CDataType   = float;
    using AScaleDataType = float;
    using BScaleDataType = float;
};

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool UseABScale = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    // static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    // static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 1;
    static constexpr ck_tile::index_t TileParitionerM01      = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t Pipeline      = CK_TILE_PIPELINE_COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool Preshuffle                = false;
    static constexpr bool Persistent                = true;
    static constexpr bool DoubleSmemBuffer          = false;
};

template <typename PrecType>
struct GemmConfigComputeV4 : public GemmConfigBase
{
    // Compute V4 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 32;  //must be M_Warp_Tile * M_Warp
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeV5 : public GemmConfigBase
{
    // Compute V5 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool Persistent           = false;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
    
    static constexpr int kBlockPerCu = 1;
    static constexpr bool UseABScale = true;
};

template <typename PrecType>
struct GemmConfigComputeV6 : public GemmConfigBase
{
    // Compute V5 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 32;  //must be M_Warp_Tile * M_Warp
    static constexpr ck_tile::index_t N_Tile = 64;  // 不可大于量化粒度 block_shape_n
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool Persistent           = false;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
    
    static constexpr int kBlockPerCu = 1;

    static constexpr bool UseABScale = true;
};

template <typename PrecType>
struct GemmConfigComputeBlockShape64 : public GemmConfigBase
{
    // Compute V5 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 32;  //must be M_Warp_Tile * M_Warp
    static constexpr ck_tile::index_t N_Tile = 64;  // 不可大于量化粒度 block_shape_n
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = false;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;

    static constexpr bool UseABScale = true;
};


template <ck_tile::index_t PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};




template <typename GemmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AScaleDataType = ck_tile::null_type,
          typename BScaleDataType = ck_tile::null_type>
float moe_grouped_gemm1_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            fused_moegemm_stage1_args &args,
                            bool splitk)
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
                                                   false,
                                                   false,
                                                   GemmConfig::UseABScale,
                                                   GemmConfig::Persistent>;


    ck_tile::index_t* block_start_dev = nullptr;
    ck_tile::index_t* block_end_dev   = nullptr;
    ck_tile::index_t* pre_tokens_dev  = nullptr;
    ck_tile::index_t total_grid_size = 0;

    if constexpr(GemmConfig::Persistent == false) {
        
        if(num_groups > 0)
        {
            std::vector<ck_tile::index_t> tokens_per_expert_host(2 * static_cast<std::size_t>(num_groups));

            HIP_CHECK_ERROR(hipMemcpy(tokens_per_expert_host.data(),
                                    args.tokens_positions_per_expert_ptr,
                                    tokens_per_expert_host.size() * sizeof(ck_tile::index_t),
                                    hipMemcpyDeviceToHost));

            std::vector<ck_tile::index_t> block_start_host(num_groups);
            std::vector<ck_tile::index_t> block_end_host(num_groups);
            std::vector<ck_tile::index_t> pre_tokens_host(num_groups);
            
            ck_tile::index_t pre_tokens_num = 0;
            for(ck_tile::index_t i = 0; i < num_groups; ++i)
            {
                const ck_tile::index_t tokens = tokens_per_expert_host[static_cast<std::size_t>(i)];
                const ck_tile::index_t blocks =
                    ck_tile::index_t(TilePartitioner::GridSize(tokens, args.intermediate_size));

                block_start_host[static_cast<std::size_t>(i)] = total_grid_size;
                total_grid_size += blocks;
                block_end_host[static_cast<std::size_t>(i)] = total_grid_size;
                pre_tokens_host[static_cast<std::size_t>(i)] = pre_tokens_num;
                pre_tokens_num += tokens;
            }

            if(total_grid_size > 0)
            {
                HIP_CHECK_ERROR(hipMalloc(&block_start_dev,
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups)));
                HIP_CHECK_ERROR(hipMalloc(&block_end_dev,
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups)));
                HIP_CHECK_ERROR(hipMalloc(&pre_tokens_dev,
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups)));

                HIP_CHECK_ERROR(hipMemcpy(block_start_dev,
                                        block_start_host.data(),
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups),
                                        hipMemcpyHostToDevice));
                HIP_CHECK_ERROR(hipMemcpy(block_end_dev,
                                        block_end_host.data(),
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups),
                                        hipMemcpyHostToDevice));
                HIP_CHECK_ERROR(hipMemcpy(pre_tokens_dev,
                                        pre_tokens_host.data(),
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups),
                                        hipMemcpyHostToDevice));
            }

            args.block_start_ptr = block_start_dev;
            args.block_end_ptr   = block_end_dev;
            args.pre_tokens_ptr  = pre_tokens_dev;
        }
        else
        {
            args.block_start_ptr = nullptr;
            args.block_end_ptr   = nullptr;
            args.pre_tokens_ptr  = nullptr;
        }
    }

    float ave_time = 0.0f;

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto scheduler        = GemmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        // We create the GEMM pipeline without specifying hotloop or tailnumber.
        // These are automatically run inside the kernel based on the given input data.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                           BDataType,
                                           AccDataType,
                                           GemmShape,
                                           GemmUniversalTraits,
                                           scheduler,
                                           true,
                                           ck_tile::TailNumber::Full,
                                           AScaleDataType,
                                           BScaleDataType>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
        using GemmEpilogue = ck_tile::Moe2StageEpilogue<
            ck_tile::Moe2StageEpilogueProblem<ADataType,
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
                                             GemmConfig::UseABScale>,
            1>;
        using Kernel      = ck_tile::FusedMoeStage1Kernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        constexpr dim3 blocks = Kernel::BlockSize();
        auto kargs = Kernel::MakeKargs(args);

        if constexpr(GemmConfig::Persistent)
        {
            const dim3 grids = Kernel::MaxOccupancyGridSize(s);

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                        << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                        << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
            }

            ave_time = ck_tile::launch_kernel(s,
                                ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                Kernel{},
                                grids,
                                blocks,
                                0,
                                kargs,
                                num_groups));
        }
        else
        {
            const dim3 grids = dim3(total_grid_size, 1, 1);

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                        << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                        << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
            }

            ave_time = ck_tile::launch_kernel(s,
                                ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                Kernel{},
                                grids,
                                blocks,
                                0,
                                kargs,
                                num_groups));


            if(block_start_dev != nullptr)
            {
                HIP_CHECK_ERROR(hipFree(block_start_dev));
            }
            if(block_end_dev != nullptr)
            {
                HIP_CHECK_ERROR(hipFree(block_end_dev));
            }
            if(pre_tokens_dev != nullptr)
            {
                HIP_CHECK_ERROR(hipFree(pre_tokens_dev));
            }
        }

        return ave_time;
    };

    
    Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                    ck_tile::memory_operation_enum::set>{});
    
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
          typename AScaleDataType = ck_tile::null_type,
          typename BScaleDataType = ck_tile::null_type>
float moe_grouped_gemm2_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            fused_moegemm_stage2_args &args,
                            bool splitk)
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
                                                   false,
                                                   false,
                                                   GemmConfig::UseABScale,
                                                   GemmConfig::Persistent>;

    ck_tile::index_t* block_start_dev = nullptr;
    ck_tile::index_t* block_end_dev   = nullptr;
    ck_tile::index_t* pre_tokens_dev  = nullptr;
    ck_tile::index_t total_grid_size = 0;

    if constexpr(GemmConfig::Persistent == false) {

        if(num_groups > 0)
        {
            std::vector<ck_tile::index_t> tokens_per_expert_host(2 * static_cast<std::size_t>(num_groups));

            HIP_CHECK_ERROR(hipMemcpy(tokens_per_expert_host.data(),
                                    args.tokens_positions_per_expert_ptr,
                                    tokens_per_expert_host.size() * sizeof(ck_tile::index_t),
                                    hipMemcpyDeviceToHost));

            std::vector<ck_tile::index_t> block_start_host(num_groups);
            std::vector<ck_tile::index_t> block_end_host(num_groups);
            std::vector<ck_tile::index_t> pre_tokens_host(num_groups);
            
            ck_tile::index_t pre_tokens_num = 0;
            for(ck_tile::index_t i = 0; i < num_groups; ++i)
            {
                const ck_tile::index_t tokens = tokens_per_expert_host[static_cast<std::size_t>(i)];
                const ck_tile::index_t blocks =
                    ck_tile::index_t(TilePartitioner::GridSize(tokens, args.hidden_size));

                block_start_host[static_cast<std::size_t>(i)] = total_grid_size;
                total_grid_size += blocks;
                block_end_host[static_cast<std::size_t>(i)] = total_grid_size;
                pre_tokens_host[static_cast<std::size_t>(i)] = pre_tokens_num;
                pre_tokens_num += tokens;
            }

            if(total_grid_size > 0)
            {
                HIP_CHECK_ERROR(hipMalloc(&block_start_dev,
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups)));
                HIP_CHECK_ERROR(hipMalloc(&block_end_dev,
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups)));
                HIP_CHECK_ERROR(hipMalloc(&pre_tokens_dev,
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups)));

                HIP_CHECK_ERROR(hipMemcpy(block_start_dev,
                                        block_start_host.data(),
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups),
                                        hipMemcpyHostToDevice));
                HIP_CHECK_ERROR(hipMemcpy(block_end_dev,
                                        block_end_host.data(),
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups),
                                        hipMemcpyHostToDevice));
                HIP_CHECK_ERROR(hipMemcpy(pre_tokens_dev,
                                        pre_tokens_host.data(),
                                        sizeof(ck_tile::index_t) * static_cast<std::size_t>(num_groups),
                                        hipMemcpyHostToDevice));
            }

            args.block_start_ptr = block_start_dev;
            args.block_end_ptr   = block_end_dev;
            args.pre_tokens_ptr  = pre_tokens_dev;
        }
        else
        {
            args.block_start_ptr = nullptr;
            args.block_end_ptr   = nullptr;
            args.pre_tokens_ptr  = nullptr;
        }
    }

    float ave_time = 0.0f;

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto scheduler        = GemmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        // We create the GEMM pipeline without specifying hotloop or tailnumber.
        // These are automatically run inside the kernel based on the given input data.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                           BDataType,
                                           AccDataType,
                                           GemmShape,
                                           GemmUniversalTraits,
                                           scheduler,
                                           true,
                                           ck_tile::TailNumber::Full,
                                           AScaleDataType,
                                           BScaleDataType>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
        using GemmEpilogue = ck_tile::Moe2StageEpilogue<
            ck_tile::Moe2StageEpilogueProblem<ADataType,
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
                                             GemmConfig::UseABScale>,
            2>;
        using Kernel      = ck_tile::FusedMoeStage2Kernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        constexpr dim3 blocks = Kernel::BlockSize();
        auto kargs = Kernel::MakeKargs(args);

        if constexpr(GemmConfig::Persistent)
        {
            const dim3 grids  = Kernel::MaxOccupancyGridSize(s);
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
                                        kargs,
                                        num_groups));
        }
        else
        {
            const dim3 grids = dim3(total_grid_size, 1, 1);
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
                                        kargs,
                                        num_groups));
            
            if(block_start_dev != nullptr)
            {
                HIP_CHECK_ERROR(hipFree(block_start_dev));
            }
            if(block_end_dev != nullptr)
            {
                HIP_CHECK_ERROR(hipFree(block_end_dev));
            }
            if(pre_tokens_dev != nullptr)
            {
                HIP_CHECK_ERROR(hipFree(pre_tokens_dev));
            }
        }

        return ave_time;
    };

    Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                    ck_tile::memory_operation_enum::atomic_add>{});

    return ave_time;
}
