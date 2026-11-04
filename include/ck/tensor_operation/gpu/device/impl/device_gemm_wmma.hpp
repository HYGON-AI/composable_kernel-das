// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include <stdexcept>
#include <type_traits>

#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/utility/loop_scheduler.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Source-compatible with the upstream WMMA interface (NumPrefetch precedes
// BlockSize, and KPerBlock is the complete K tile, not K0PerBlock).
// HCU uses the existing XDL/MMAC CShuffle backend. WMMA wave32 tuning parameters
// are accepted for source compatibility, but do not configure the HCU kernel.
// Its fixed wave64 tile is 64x64x32, with scalar global transfers for tails and
// non-packed strides. GemmSpec still gates the caller's M/N/K block multiples.
// This is a correctness-oriented compatibility path, not a WMMA performance port.
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t K1,
          ck::index_t MPerWMMA,
          ck::index_t NPerWMMA,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::v1>
struct DeviceGemmWmma_CShuffle : public DeviceGemm<ALayout,
                                                   BLayout,
                                                   CLayout,
                                                   ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   AElementwiseOperation,
                                                   BElementwiseOperation,
                                                   CElementwiseOperation>
{
    using Row = tensor_layout::gemm::RowMajor;
    using Col = tensor_layout::gemm::ColumnMajor;
    template <index_t... X>
    using S = Sequence<X...>;

    static_assert(is_same_v<ALayout, Row> || is_same_v<ALayout, Col>, "Invalid A layout");
    static_assert(is_same_v<BLayout, Row> || is_same_v<BLayout, Col>, "Invalid B layout");
    static_assert(is_same_v<ADataType, BDataType>, "HCU WMMA compatibility requires equal A/B types");
    static_assert((is_same_v<ADataType, half_t> || is_same_v<ADataType, bhalf_t>) &&
                      is_same_v<AccDataType, float> && is_same_v<CDataType, ADataType>,
                  "HCU WMMA compatibility supports FP16/BF16 with FP32 accumulation");
    static_assert(MPerBlock > 0 && NPerBlock > 0 && KPerBlock > 0,
                  "GEMM block dimensions must be positive");
    static_assert(GemmSpec == GemmSpecialization::Default ||
                      GemmSpec == GemmSpecialization::MPadding ||
                      GemmSpec == GemmSpecialization::NPadding ||
                      GemmSpec == GemmSpecialization::KPadding ||
                      GemmSpec == GemmSpecialization::MNPadding ||
                      GemmSpec == GemmSpecialization::MKPadding ||
                      GemmSpec == GemmSpecialization::NKPadding ||
                      GemmSpec == GemmSpecialization::MNKPadding,
                  "Unsupported GEMM specialization");

    // Keep the backend independent of AMD WMMA's wave layout and K1 choices.
    // C is row-major, as in the upstream WMMA contract. Using Row here lets an
    // unsupported C layout be rejected by IsSupportedArgument without a kernel
    // template failure. Preserve the caller's actual CShuffle data type.
    using Backend = DeviceGemm_Xdl_CShuffleV3<
        ALayout, BLayout, Row, ADataType, BDataType, CDataType, AccDataType,
        CShuffleDataType, AElementwiseOperation, BElementwiseOperation, CElementwiseOperation,
        GemmSpecialization::MNKPadding,
        256, 64, 64, 32, 8, 8, 16, 16, 2, 2,
        S<4, 64, 1>,
        std::conditional_t<is_same_v<ALayout, Row>, S<1, 0, 2>, S<0, 2, 1>>,
        std::conditional_t<is_same_v<ALayout, Row>, S<1, 0, 2>, S<0, 2, 1>>,
        is_same_v<ALayout, Row> ? 2 : 1, 1, 8, false,
        S<4, 64, 1>,
        std::conditional_t<is_same_v<BLayout, Col>, S<1, 0, 2>, S<0, 2, 1>>,
        std::conditional_t<is_same_v<BLayout, Col>, S<1, 0, 2>, S<0, 2, 1>>,
        is_same_v<BLayout, Col> ? 2 : 1, 1, 8, false,
        1, 1, S<1, 32, 1, 8>, 1,
        BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v1,
        ADataType, BDataType>;

    // Use a distinct argument type so the polymorphic interface rejects another
    // operator's argument, even when both use the same normalized backend.
    struct Argument : public Backend::Argument
    {
        using Backend::Argument::Argument;
    };

    static constexpr bool IsValidCompilationParameter()
    {
        return is_same_v<CLayout, Row> && NumPrefetch > 0 && BlockSize > 0 && K1 > 0 &&
               MPerWMMA == 16 && NPerWMMA == 16 && MRepeat > 0 && NRepeat > 0;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        const auto name = ck::get_device_name();
        if((name != "gfx936" && name != "gfx938") || !IsValidCompilationParameter())
            return false;
        if(arg.M <= 0 || arg.N <= 0 || arg.K <= 0 || arg.KBatch != 1 ||
           arg.p_a_grid == nullptr || arg.p_b_grid == nullptr || arg.p_c_grid == nullptr)
            return false;
        if(arg.StrideA < (is_same_v<ALayout, Row> ? arg.K : arg.M) ||
           arg.StrideB < (is_same_v<BLayout, Row> ? arg.N : arg.K) || arg.StrideC < arg.N)
            return false;

        constexpr bool pad_m = GemmSpec == GemmSpecialization::MPadding ||
                               GemmSpec == GemmSpecialization::MNPadding ||
                               GemmSpec == GemmSpecialization::MKPadding ||
                               GemmSpec == GemmSpecialization::MNKPadding;
        constexpr bool pad_n = GemmSpec == GemmSpecialization::NPadding ||
                               GemmSpec == GemmSpecialization::MNPadding ||
                               GemmSpec == GemmSpecialization::NKPadding ||
                               GemmSpec == GemmSpecialization::MNKPadding;
        constexpr bool pad_k = GemmSpec == GemmSpecialization::KPadding ||
                               GemmSpec == GemmSpecialization::MKPadding ||
                               GemmSpec == GemmSpecialization::NKPadding ||
                               GemmSpec == GemmSpecialization::MNKPadding;
        if((!pad_m && arg.M % MPerBlock != 0) || (!pad_n && arg.N % NPerBlock != 0) ||
           (!pad_k && arg.K % KPerBlock != 0))
            return false;
        return Backend::IsSupportedArgument(arg);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const auto* arg = dynamic_cast<const Argument*>(p_arg);
        return arg != nullptr && IsSupportedArgument(*arg);
    }

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!DeviceGemmWmma_CShuffle::IsSupportedArgument(arg))
                throw std::runtime_error("DeviceGemmWmma_CShuffle: unsupported HCU GEMM argument");
            // The public WMMA API has no SplitK. Launch only the Set kernels:
            // the general V3 invoker also instantiates AtomicAdd, which requires
            // vectorized half/BF16 stores and cannot serve this scalar tail path.
            using GridwiseGemm = typename Backend::GridwiseGemm64;
            const auto [gx, gy, gz] = GridwiseGemm::CalculateGridSize(arg.M, arg.N, 1);
            const auto& backend_arg = static_cast<const typename Backend::Argument&>(arg);
            if(stream_config.log_level_ > 0)
                backend_arg.Print();
            const index_t padded_k = ck::math::integer_divide_ceil(arg.K, 32) * 32;
            if(GridwiseGemm::CalculateHasMainKBlockLoop(padded_k))
                return launch_and_time_kernel(
                    stream_config,
                    kernel_gemm_xdl_cshuffle_v3<GridwiseGemm, true, InMemoryDataOperationEnum::Set>,
                    dim3(gx, gy, gz), dim3(256), 0, backend_arg);
            return launch_and_time_kernel(
                stream_config,
                kernel_gemm_xdl_cshuffle_v3<GridwiseGemm, false, InMemoryDataOperationEnum::Set>,
                dim3(gx, gy, gz), dim3(256), 0, backend_arg);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            const auto* arg = dynamic_cast<const Argument*>(p_arg);
            if(arg == nullptr)
                throw std::runtime_error("DeviceGemmWmma_CShuffle: argument type mismatch");
            return Run(*arg, stream_config);
        }
    };

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        // Construct directly: the V3 device factory default-constructs functors.
        // This interface must preserve state supplied by the caller.
        return Argument{p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC, 1, false,
                        a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(MakeArgument(static_cast<const ADataType*>(p_a),
                                                       static_cast<const BDataType*>(p_b),
                                                       static_cast<CDataType*>(p_c),
                                                       M, N, K, StrideA, StrideB, StrideC,
                                                       a_element_op, b_element_op, c_element_op));
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override
    {
        return "DeviceGemmWmma_CShuffle<HCU_XDL_Fallback, " +
               getGemmSpecializationString(GemmSpec) +
               ", BlockSize=256, BlockTile=64x64x32, XDL=16x16, Pipeline=v1>";
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
