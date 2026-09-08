// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm_quant/grouped_gemm_quant_device_args.h"

#include "ck_tile/core.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || CK_TILE_HCU_QUANT_GEMM_TARGET

#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          ck_tile::QuantType QuantMode,
          ck_tile::index_t KTile>
int run_quant_grouped_gemm(void* device_args,
                           int group_count,
                           std::uint32_t num_cu,
                           hipStream_t stream)
{
    constexpr ck_tile::index_t MTile     = 128;
    constexpr ck_tile::index_t NTile     = 128;
    constexpr ck_tile::index_t MWarp     = 4;
    constexpr ck_tile::index_t NWarp     = 1;
    constexpr ck_tile::index_t KWarp     = 1;
    constexpr ck_tile::index_t MWarpTile = 32;
    constexpr ck_tile::index_t NWarpTile = 64;
    constexpr ck_tile::index_t KWarpTile = 32;

    using AccDataType = float;
    using CLayout     = Row;
    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<MTile, NTile, KTile>,
                               ck_tile::sequence<MWarp, NWarp, KWarp>,
                               ck_tile::sequence<MWarpTile, NWarpTile, KWarpTile>>;
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;
    using Traits = ck_tile::TileGemmQuantTraits<true,
                                                true,
                                                true,
                                                false,
                                                false,
                                                false,
                                                ALayout,
                                                BLayout,
                                                CLayout,
                                                QuantMode,
                                                Row,
                                                Col,
                                                false,
                                                false,
                                                true>;
    using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
    using BQuantGroupSize = std::conditional_t<
        std::is_same_v<ALayout, Row>,
        ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>,
        ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>>;
    using Problem = std::conditional_t<
        QuantMode == ck_tile::QuantType::ABQuantGrouped,
        ck_tile::GemmABQuantPipelineProblem<ADataType,
                                             AccDataType,
                                             BDataType,
                                             AccDataType,
                                             AccDataType,
                                             GemmShape,
                                             Traits,
                                             AQuantGroupSize,
                                             BQuantGroupSize,
                                             false>,
        ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
                                                       BDataType,
                                                       AccDataType,
                                                       AccDataType,
                                                       GemmShape,
                                                       Traits>>;
    using Pipeline = std::conditional_t<
        QuantMode == ck_tile::QuantType::ABQuantGrouped,
        ck_tile::ABQuantGemmPipelineAgBgCrCompV3<Problem>,
        ck_tile::GemmPipelineAgBgCrCompV3<Problem>>;
    using Epilogue = ck_tile::CShuffleEpilogue<
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
                                         MWarp,
                                         NWarp,
                                         MWarpTile,
                                         NWarpTile,
                                         KWarpTile,
                                         Problem::TransposeC,
                                         ck_tile::memory_operation_enum::set>>;
    using Kernel =
        ck_tile::QuantGroupedGemmKernel<TilePartitioner, Pipeline, Epilogue, QuantMode>;

    const ck_tile::stream_config stream_cfg{stream};
    dim3 grid = Kernel::MaxOccupancyGridSize(stream_cfg);
    if(num_cu != 0)
        grid.x = std::min(grid.x, num_cu);
    ck_tile::launch_kernel(
        stream_cfg,
        ck_tile::make_kernel<Kernel::kBlockSize, 1>(
            Kernel{},
            grid,
            Kernel::BlockSize(),
            0,
            ck_tile::cast_pointer_to_constant_address_space(device_args),
            ck_tile::type_convert<ck_tile::index_t>(group_count)));
    return CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1;
}

template <typename ADataType, typename BDataType, typename CDataType>
int dispatch_layout_and_mode(std::uint32_t layout,
                             std::uint32_t mode,
                             void* device_args,
                             int group_count,
                             std::uint32_t num_cu,
                             hipStream_t stream)
{
#define CK_TILE_RUN_QGG(ALayout, BLayout, QuantMode, KTile)                              \
    return run_quant_grouped_gemm<ADataType, BDataType, CDataType, ALayout, BLayout,    \
                                   QuantMode, KTile>(                                    \
        device_args, group_count, num_cu, stream)

    if(layout == CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_NN_V1)
    {
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_TENSORWISE_V1)
            CK_TILE_RUN_QGG(Row, Row, ck_tile::QuantType::TensorQuant, 32);
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_ROWWISE_V1)
            CK_TILE_RUN_QGG(Row, Row, ck_tile::QuantType::RowColQuant, 32);
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_BLOCKWISE_V1)
            CK_TILE_RUN_QGG(Row, Row, ck_tile::QuantType::ABQuantGrouped, 128);
    }
    if(layout == CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_NT_V1)
    {
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_TENSORWISE_V1)
            CK_TILE_RUN_QGG(Row, Col, ck_tile::QuantType::TensorQuant, 32);
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_ROWWISE_V1)
            CK_TILE_RUN_QGG(Row, Col, ck_tile::QuantType::RowColQuant, 32);
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_BLOCKWISE_V1)
            CK_TILE_RUN_QGG(Row, Col, ck_tile::QuantType::ABQuantGrouped, 128);
    }
    if(layout == CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_TN_V1)
    {
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_TENSORWISE_V1)
            CK_TILE_RUN_QGG(Col, Row, ck_tile::QuantType::TensorQuant, 32);
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_ROWWISE_V1)
            CK_TILE_RUN_QGG(Col, Row, ck_tile::QuantType::RowColQuant, 32);
        if(mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_BLOCKWISE_V1)
            CK_TILE_RUN_QGG(Col, Row, ck_tile::QuantType::ABQuantGrouped, 128);
    }
#undef CK_TILE_RUN_QGG
    return CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_V1;
}

template <typename ADataType, typename BDataType>
int dispatch_output(std::uint32_t c_type,
                    std::uint32_t layout,
                    std::uint32_t mode,
                    void* device_args,
                    int group_count,
                    std::uint32_t num_cu,
                    hipStream_t stream)
{
    if(c_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP16_V1)
        return dispatch_layout_and_mode<ADataType, BDataType, ck_tile::half_t>(
            layout, mode, device_args, group_count, num_cu, stream);
    if(c_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_BF16_V1)
        return dispatch_layout_and_mode<ADataType, BDataType, ck_tile::bf16_t>(
            layout, mode, device_args, group_count, num_cu, stream);
    return CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_V1;
}

} // namespace

extern "C" std::size_t ck_tile_hcu_quant_grouped_gemm_provider_device_args_size_v1()
{
    return sizeof(ck_tile::QuantGemmTransKernelArg);
}

extern "C" int ck_tile_hcu_quant_grouped_gemm_provider_run_v1(std::uint32_t a_type,
                                                               std::uint32_t b_type,
                                                               std::uint32_t c_type,
                                                               std::uint32_t layout,
                                                               std::uint32_t mode,
                                                               void* device_args,
                                                               int group_count,
                                                               std::uint32_t num_cu,
                                                               hipStream_t stream)
{
    if(a_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E4M3_V1)
    {
        if(b_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E4M3_V1)
            return dispatch_output<ck_tile::fp8_t, ck_tile::fp8_t>(
                c_type, layout, mode, device_args, group_count, num_cu, stream);
        if(b_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E5M2_V1)
            return dispatch_output<ck_tile::fp8_t, ck_tile::bf8_t>(
                c_type, layout, mode, device_args, group_count, num_cu, stream);
    }
    if(a_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E5M2_V1)
    {
        if(b_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E4M3_V1)
            return dispatch_output<ck_tile::bf8_t, ck_tile::fp8_t>(
                c_type, layout, mode, device_args, group_count, num_cu, stream);
        if(b_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E5M2_V1)
            return dispatch_output<ck_tile::bf8_t, ck_tile::bf8_t>(
                c_type, layout, mode, device_args, group_count, num_cu, stream);
    }
    return CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_V1;
}

#endif // !__HIP_DEVICE_COMPILE__ || CK_TILE_HCU_QUANT_GEMM_TARGET
