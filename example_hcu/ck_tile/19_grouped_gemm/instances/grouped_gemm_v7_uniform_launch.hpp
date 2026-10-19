// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Launch helper for the v7 uniform (NN / TN) grouped-GEMM instances.
//
// This is the CK-kernel path: the whole tile chain
// (TileGemmShape -> TilePartitioner -> GroupedGemmKernel -> pipeline -> epilogue)
// is instantiated from the config, exactly as the other CK instances do it. What
// differs is the GRID:
//
//   * the config is NON-persistent, so the kernel maps block id -> (group,
//     m tile, n tile) from the kargs' block ranges and runs exactly ONE tile.
//     The kernel cannot be made persistent: this pipeline's register budget
//     (215 VGPR, 0 spill, the reference kernel's numbers) only survives when no
//     loop wraps its k loop -- see the pipeline header.
//   * the grid therefore has to cover the tile count, which the device-args
//     provider cannot read (per-group M lives in device memory). The dispatch
//     passes a host-side upper bound, rounded up to a whole number of CTAs:
//     BlockIdRange() below is that bound, and the kernels' `t >= total` tiles
//     simply exit early when the bound over-covers.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"   // GroupedGemmProblem
#include "../grouped_gemm.hpp"
#include "../grouped_gemm_v7_uniform_epilogue.hpp"

// -----------------------------------------------------------------------------
// v7/v8 uniform NN / TN configs: 256x256x16 bf16, one tile per CTA,
// Direct-to-LDS. They live HERE (not in grouped_gemm.hpp) on purpose: the
// configs are consumed only by the v7/v8 translation units, so tuning them --
// the common edit while debugging -- recompiles these two TUs, not the whole
// grouped-GEMM archive.
//
// GEOMETRY vs the CONFIG. The pipeline is fixed at 256x256x16 with a 4m x 2n
// wave grid (64M x 128N per wave, 32 MMACs per k-stage) and a 4-stage 16 KB LDS
// ring; the numbers below are what the FRAMEWORK needs to size its tile grid and
// to derive (M,N) tile origins from the block id. K_Tile is 64 because the
// framework only uses it for tile/loop bookkeeping -- the pipeline's own k
// loop is the 4-stage ring, advanced by the 16-wide k-stage.
//
// Persistent = false is load-bearing, not taste: the non-persistent kernel path
// maps block id -> (group, m tile, n tile) from the kargs' block ranges and then
// runs ONE tile, which is the only structure this pipeline compiles to the
// reference kernel's register budget (215 VGPR / 0 spill; a tile loop costs 40
// VGPRs and ~20% of the throughput, see the pipeline header).
//
// kPadM/N/K stay false: ragged M and the N tail are handled by the epilogue's
// store mask and the selector's shape gates, not by padding the operands.
// -----------------------------------------------------------------------------
template <typename PrecType, ck_tile::index_t PipelineId>
struct GemmConfigV7Uniform : public GemmConfigBase
{
    static_assert(std::is_same_v<PrecType, ck_tile::bf16_t>,
                  "the v7/v8 uniform pipelines are bf16 kernels");

    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool Persistent       = false;
    static constexpr bool DoubleSmemBuffer = false;

    static constexpr bool SupportsFastRowMajorB    = true;
    static constexpr bool SupportsFastColumnMajorA = true;

    static constexpr ck_tile::index_t TileParitionerGroupNum = 1;
    static constexpr ck_tile::index_t TileParitionerM01      = 1;

    static constexpr int kBlockPerCu = 1;   // 64 KB ring => one CTA per CU

    static constexpr ck_tile::index_t Pipeline = PipelineId;
};

using GemmConfigV7NnBf16 =
    GemmConfigV7Uniform<ck_tile::bf16_t, CK_TILE_PIPELINE_COMPUTE_V7>;
using GemmConfigV8TnBf16 =
    GemmConfigV7Uniform<ck_tile::bf16_t, CK_TILE_PIPELINE_COMPUTE_V8>;

namespace v7_uniform_launch {

// Upper bound on the (group, m tile, n tile) tile count for the v7 layouts,
// rounded up to a multiple of `num_cu` (one CTA per tile, so the grid must be a
// whole number of CTA slots). `problem.m`/`n` carry either a logical common
// length or a host-visible capacity -- never device-read values -- so this is
// computable here and stays graph-capture safe.
//
// Device-resident M: the capacity is the backing storage, i.e. an upper bound on
// the summed group rows, and each group can round up to its own tile, so
// ceil(capacity / 256) + group_count is the bound on the summed row tiles.
inline std::uint64_t block_id_range(const ck_tile::GroupedGemmProblem& problem,
                                    std::uint32_t num_cu)
{
    const std::int64_t groups    = problem.group_count > 0 ? problem.group_count : 1;
    const std::int64_t n_tiles   = (problem.n.extent + 255) / 256;
    const bool device_m =
        problem.m.kind == ck_tile::GroupedGemmDimensionKind::device_lengths_with_capacity;
    const std::int64_t m_tiles = device_m ? (problem.m.extent + 255) / 256 + groups
                                          : (problem.m.extent + 255) / 256 * groups;
    const std::uint64_t tiles  = static_cast<std::uint64_t>(m_tiles * n_tiles);
    const std::uint64_t slots  = num_cu > 0 ? num_cu : 1;
    return (tiles + slots - 1) / slots * slots;
}

template <typename Config,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType>
inline int launch(const void* kargs_dev,
                  int group_count,
                  std::uint64_t tile_slots,
                  hipStream_t stream)
{
    using Shape = ck_tile::TileGemmShape<
        ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
        ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
        ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;

    using Traits = ck_tile::TileGemmUniversalTraits<Config::kPadM,
                                                    Config::kPadN,
                                                    Config::kPadK,
                                                    Config::DoubleSmemBuffer,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    Config::TransposeC,
                                                    false,
                                                    false,
                                                    1,
                                                    false,
                                                    false,
                                                    Config::SupportsFastRowMajorB,
                                                    Config::SupportsFastColumnMajorA>;

    using Problem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                          BDataType,
                                                          AccDataType,
                                                          Shape,
                                                          Traits,
                                                          Config::Scheduler,
                                                          true,
                                                          ck_tile::TailNumber::Full,
                                                          ck_tile::null_type,
                                                          ck_tile::null_type,
                                                          false,
                                                          2,
                                                          2>;

    using Pipeline = typename PipelineTypeTraits<Config::Pipeline>::template GemmPipeline<Problem>;
    using Epilogue =
        ck_tile::GroupedGemmV7UniformStoreEpilogue<CDataType, typename Pipeline::AccTile>;
    using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<Shape,
                                                                  Config::TileParitionerGroupNum,
                                                                  Config::TileParitionerM01>;
    using Kernel = ck_tile::GroupedGemmKernel<Partitioner, Pipeline, Epilogue>;

    if(kargs_dev == nullptr || group_count <= 0 || tile_slots == 0)
    {
        return -1;
    }

    const dim3 grid(static_cast<unsigned>(tile_slots));
    const dim3 block(Kernel::BlockSize());

    const auto kernel = ck_tile::make_kernel<512, Config::kBlockPerCu>(
        Kernel{},
        grid,
        block,
        Kernel::GetSmemSize(),
        ck_tile::cast_pointer_to_constant_address_space(kargs_dev),
        group_count);
    ck_tile::launch_kernel(ck_tile::stream_config{stream}, kernel);
    return 0;
}

}   // namespace v7_uniform_launch
