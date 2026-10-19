// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// HCU (gfx936 family) instance set for GEMM with per-block A/B dequantization
// scales. HCU MMAC only supports 16x16 XDL tiles; the tuning below is derived
// from the validated HCU bf16 16x16 GEMM configuration
// (MPerBlock=NPerBlock=128, KPerBlock=32, AK1=BK1=8).

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = bhalf_t;
using F32  = float;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

template <index_t... Is>
using S = Sequence<Is...>;

using PassThrough = element_wise::PassThrough;

template <GemmSpecialization GemmSpec>
using device_gemm_ab_scale_xdl_bf16_bf16_bf16_mk_nk_mn_instances = std::tuple<
    // clang-format off
    //############################| ALayout| BLayout|       DsLayout| ELayout|      AData|      AScale|      BData|      BScale|       DsData| EData| AccData| Cshuffle|           A|           B|           C|          GEMM| Block| Scale| Scale| Scale|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|                         Block-wiseGemm|               Block-wiseGemm|         Compute|
    //############################|        |        |               |        |       Type|       Type|       Type|       Type|        Type|  Type|    Type|     Type| Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|                               Pipeline|                     Pipeline|     TypeA|
    //############################|        |        |               |        |           |           |           |           |            |      |        |         |   Operation|   Operation|   Operation|              |      |     M|     N|     K|      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|                              Scheduler|                     Verision|           |
    //############################|        |        |               |        |           |           |           |           |            |      |        |         |            |            |            |              |      |      |      |      |      |      |      |    |    |    |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |                                       |                             |             |

    // Compute friendly
    DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3<    Row,     Col,    Tuple<>,    Row,   BF16,    F32,   BF16,    F32,    Tuple<>,   BF16,    F32,    F32, PassThrough, PassThrough, PassThrough,      GemmSpec,     256,     1,   128,    32,    128,   128,    32,   8,   8,  16,   16,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,               2,              8,              8,          0,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,     S<4>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1,   BF16>
    // clang-format on
    >;

void add_device_gemm_ab_scale_xdl_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
                                                           Col,
                                                           Tuple<>,
                                                           Row,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           F32,
                                                           Tuple<>,
                                                           BF16,
                                                           1,
                                                           128,
                                                           32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_ab_scale_xdl_bf16_bf16_bf16_mk_nk_mn_instances<
            GemmSpecialization::Default>{});
    add_device_operation_instances(
        instances,
        device_gemm_ab_scale_xdl_bf16_bf16_bf16_mk_nk_mn_instances<
            GemmSpecialization::MNKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
