// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// HCU MultiABD GEMM device impl。
//
// 当前接口形态：
//   E = CDEElementwiseOperation(Gemm(AElementwiseOperation(A0, A1, ...),
//                                   BElementwiseOperation(B0, B1, ...)),
//                               D0, D1, ...)
//   A/B/D 均使用 ck::Tuple 表达多输入，底层通过 HCU
//   GridwiseGemmMultiABD_xdl_cshuffle_v3 在 Global -> LDS 搬运阶段完成
//   A/B elementwise materialize，再进入 XDL/MMAC CShuffle GEMM 主体。
//
// 已验证的最小可用范围：
//   - BF16/BF16/BF16，Acc=float，CShuffle=BF16。
//   - RCR 布局：A 为 RowMajor tuple，B 为 ColumnMajor tuple，D/E 为 RowMajor。
//   - NumATensor=2、NumBTensor=2、NumDTensor=1。
//   - AElementwiseOperation=Add，BElementwiseOperation=Add，
//     CDEElementwiseOperation=Add。
//   - wave64 HCU 路径，BlockGemmPipelineVersion v1/v3；
//     example_hcu_gemm_multi_abd_xdl_bf16_v3 已在 gfx938 上通过 correctness。
//
// 限制：
//   - 所有 A layout 必须与 A0 相同，所有 B layout 必须与 B0 相同。
//   - DirectLoad 仅允许 singleton A/B tuple；NumATensor 或 NumBTensor > 1 时禁用。
//   - 当前 device impl 只启用 wave64 路径。
//   - KBatch > 1 受现有 HCU/BF16 atomic 支持限制。
//   - host-side partitioner 仅做 M 维切分：要求 A/D/C 为 RowMajor 可按 M 切，
//     且 B tuple descriptor 自身小于 2GB；尚未做 >2GB 分配运行验证。
//   - 这是 HCU first 的最小闭包
//     DeviceGemmMultipleABD factory/profiler/Python runtime schema。

#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_abd.hpp"
#include "ck/utility/common_header.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename LayoutTuple, typename Layout, index_t... Is>
static constexpr bool is_layout_tuple_uniform(std::integer_sequence<index_t, Is...>)
{
    return (true && ... && is_same_v<remove_cvref_t<tuple_element_t<Is, LayoutTuple>>, Layout>);
}

template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename CLayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          typename LDSTypeA                           = ComputeTypeA,
          typename LDSTypeB                           = ComputeTypeB>
struct DeviceGemmMultiABD_Xdl_CShuffle_V3
    : public DeviceGemmMultiABDV2R1<AsLayout,
                                    BsLayout,
                                    DsLayout,
                                    CLayout,
                                    AsDataType,
                                    BsDataType,
                                    DsDataType,
                                    CDataType,
                                    AElementwiseOperation,
                                    BElementwiseOperation,
                                    CElementwiseOperation>
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                      BlkGemmPipelineVer == BlockGemmPipelineVersion::v3,
                  "M2-2 HCU MultiABD correctness path currently enables v1/v3 pipelines");

    using ALayout = remove_cvref_t<tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<tuple_element_t<0, BsLayout>>;

    static_assert(is_layout_tuple_uniform<AsLayout, ALayout>(
                      std::make_integer_sequence<index_t, NumATensor>{}),
                  "HCU MultiABD V3 currently requires all A layouts to match A0");
    static_assert(is_layout_tuple_uniform<BsLayout, BLayout>(
                      std::make_integer_sequence<index_t, NumBTensor>{}),
                  "HCU MultiABD V3 currently requires all B layouts to match B0");

    static constexpr auto WarpTileConfig64 = GetWarpTileConfig<BlockSize,
                                                               MPerBlock,
                                                               NPerBlock,
                                                               MPerXDL,
                                                               NPerXDL,
                                                               MXdlPerWave,
                                                               CShuffleMXdlPerWavePerShuffle,
                                                               CShuffleNXdlPerWavePerShuffle,
                                                               true>();
    static constexpr auto WarpTileConfig32 = GetWarpTileConfig<BlockSize,
                                                               MPerBlock,
                                                               NPerBlock,
                                                               MPerXDL,
                                                               NPerXDL,
                                                               MXdlPerWave,
                                                               CShuffleMXdlPerWavePerShuffle,
                                                               CShuffleNXdlPerWavePerShuffle,
                                                               false>();
    static constexpr auto NXdlPerWave64    = WarpTileConfig64.At(Number<3>{});
    static constexpr auto NXdlPerWave32    = WarpTileConfig32.At(Number<3>{});

    template <typename WarpTileConfig>
    using GridwiseGemmBase = GridwiseGemmMultiABD_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        CLayout,
        AsDataType,
        BsDataType,
        GemmAccDataType,
        CShuffleDataType,
        DsDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        WarpTileConfig::At(Number<0>{}),
        WarpTileConfig::At(Number<1>{}),
        WarpTileConfig::At(Number<2>{}),
        WarpTileConfig::At(Number<3>{}),
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        WarpTileConfig::At(Number<4>{}),
        WarpTileConfig::At(Number<5>{}),
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        LDSTypeA,
        LDSTypeB>;

    using GridwiseGemm64 = GridwiseGemmBase<decltype(WarpTileConfig64)>;
    using GridwiseGemm32 = GridwiseGemmBase<decltype(WarpTileConfig32)>;
    using Argument       = typename GridwiseGemm64::Argument;

    struct Partitioner
    {
        using AsGridPointer = typename GridwiseGemm64::AsGridPointer;
        using BsGridPointer = typename GridwiseGemm64::BsGridPointer;
        using DsGridPointer = typename GridwiseGemm64::DsGridPointer;

        index_t M;
        index_t N;
        std::array<index_t, NumATensor> StrideAs;
        std::array<index_t, NumBTensor> StrideBs;
        std::array<index_t, NumDTensor> StrideDs;
        index_t StrideC;

        static constexpr long_index_t TwoGB    = INT32_MAX;
        static constexpr index_t PartitionSize = 256;

        Partitioner() = default;
        Partitioner(index_t M_,
                    index_t N_,
                    std::array<index_t, NumATensor> StrideAs_,
                    std::array<index_t, NumBTensor> StrideBs_,
                    std::array<index_t, NumDTensor> StrideDs_,
                    index_t StrideC_)
            : M{M_},
              N{N_},
              StrideAs{StrideAs_},
              StrideBs{StrideBs_},
              StrideDs{StrideDs_},
              StrideC{StrideC_}
        {
        }

        __host__ bool isPartitionable() const
        {
            bool row_major = is_same<CLayout, tensor_layout::gemm::RowMajor>::value;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ALayout_ = remove_cvref_t<tuple_element_t<i.value, AsLayout>>;
                row_major &= is_same<ALayout_, tensor_layout::gemm::RowMajor>::value;
            });
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout_ = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                row_major &= is_same<DLayout_, tensor_layout::gemm::RowMajor>::value;
            });

            return areBsDescriptorsSmallerThan2GB() &&
                   (row_major || areDescriptorsSmallerThan2GB());
        }

        __host__ bool areBsDescriptorsSmallerThan2GB() const
        {
            bool are_bs_descriptors_smaller_than_2gb = true;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                are_bs_descriptors_smaller_than_2gb &=
                    (static_cast<long_index_t>(N) * static_cast<long_index_t>(StrideBs[i]) *
                     sizeof(BDataType_)) <= TwoGB;
            });
            return are_bs_descriptors_smaller_than_2gb;
        }

        __host__ bool areDescriptorsSmallerThan2GB(index_t m) const
        {
            bool are_as_descriptors_smaller_than_2gb = true;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                are_as_descriptors_smaller_than_2gb &=
                    (static_cast<long_index_t>(m) * static_cast<long_index_t>(StrideAs[i]) *
                     sizeof(ADataType_)) <= TwoGB;
            });

            const bool is_c_descriptor_smaller_than_2gb =
                (static_cast<long_index_t>(m) * static_cast<long_index_t>(StrideC) *
                 sizeof(CDataType)) <= TwoGB;

            bool are_ds_descriptors_smaller_than_2gb = true;
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType_ = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                are_ds_descriptors_smaller_than_2gb &=
                    (static_cast<long_index_t>(m) * static_cast<long_index_t>(StrideDs[i]) *
                     sizeof(DDataType_)) <= TwoGB;
            });

            return are_as_descriptors_smaller_than_2gb && is_c_descriptor_smaller_than_2gb &&
                   are_ds_descriptors_smaller_than_2gb;
        }

        __host__ bool areDescriptorsSmallerThan2GB() const
        {
            return areDescriptorsSmallerThan2GB(M);
        }

        template <typename Argument_>
        __host__ static bool isDescriptorValidForGemm(const Argument_& arg)
        {
            bool are_as_descriptors_valid = true;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType_ = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                are_as_descriptors_valid &=
                    static_cast<long_index_t>(arg.M) * static_cast<long_index_t>(arg.K) *
                        sizeof(ADataType_) <=
                    TwoGB;
            });

            bool are_bs_descriptors_valid = true;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType_ = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                are_bs_descriptors_valid &=
                    static_cast<long_index_t>(arg.N) * static_cast<long_index_t>(arg.K) *
                        sizeof(BDataType_) <=
                    TwoGB;
            });

            return are_as_descriptors_valid && are_bs_descriptors_valid &&
                   static_cast<long_index_t>(arg.M) * static_cast<long_index_t>(arg.N) *
                           sizeof(CDataType) <=
                       TwoGB;
        }

        __host__ auto splitProblem(index_t m,
                                   AsGridPointer p_as_grid_left,
                                   DsGridPointer p_ds_grid_left,
                                   CDataType* p_c_grid_left) const
        {
            const index_t m_left  = ck::math::integer_least_multiple(m / 2, PartitionSize);
            const index_t m_right = m - m_left;

            const auto as_grid_right_ptr = generate_tuple(
                [&](auto i) {
                    const long_index_t as_right_offset =
                        static_cast<long_index_t>(m_left) * StrideAs[i];
                    return p_as_grid_left(i) + as_right_offset;
                },
                Number<NumATensor>{});

            const auto ds_grid_right_ptr = generate_tuple(
                [&](auto i) {
                    const long_index_t ds_right_offset =
                        static_cast<long_index_t>(m_left) * StrideDs[i];
                    return p_ds_grid_left(i) + ds_right_offset;
                },
                Number<NumDTensor>{});

            const long_index_t c_right_offset = static_cast<long_index_t>(m_left) * StrideC;

            return ck::make_tuple(
                m_left, m_right, as_grid_right_ptr, ds_grid_right_ptr, p_c_grid_left + c_right_offset);
        }

        template <index_t NumTensor, typename GridPointer>
        __host__ static auto makePointerArray(GridPointer p_grid)
        {
            std::array<const void*, NumTensor> pointers{};
            static_for<0, NumTensor, 1>{}([&](auto i) { pointers[i] = p_grid(i); });
            return pointers;
        }

        template <typename ArgumentIn, typename ArgumentOut = ArgumentIn>
        std::vector<ArgumentOut> partitionGemmProblem(const ArgumentIn& arg) const
        {
            static constexpr index_t InitialSubArgsSize = 32;

            std::vector<ArgumentOut> sub_arguments;
            sub_arguments.reserve(InitialSubArgsSize);

            std::queue<index_t> split_m({arg.M});
            std::queue<AsGridPointer> as_grid_ptrs_queue({arg.p_as_grid});
            std::queue<DsGridPointer> ds_grid_ptrs_queue({arg.p_ds_grid});
            std::queue<CDataType*> c_grid_ptrs_queue({arg.p_c_grid});

            const auto bs_grid_ptr_array = makePointerArray<NumBTensor>(arg.p_bs_grid);

            while(!split_m.empty())
            {
                index_t m                 = split_m.front();
                AsGridPointer as_grid_ptr = as_grid_ptrs_queue.front();
                DsGridPointer ds_grid_ptr = ds_grid_ptrs_queue.front();
                CDataType* c_grid_ptr     = c_grid_ptrs_queue.front();

                if(areDescriptorsSmallerThan2GB(m) || (m <= PartitionSize))
                {
                    ArgumentOut new_arg{makePointerArray<NumATensor>(as_grid_ptr),
                                        bs_grid_ptr_array,
                                        makePointerArray<NumDTensor>(ds_grid_ptr),
                                        c_grid_ptr,
                                        m,
                                        arg.N,
                                        arg.K,
                                        arg.StrideAs,
                                        arg.StrideBs,
                                        arg.StrideDs,
                                        arg.StrideC,
                                        arg.KBatch,
                                        arg.a_element_op,
                                        arg.b_element_op,
                                        arg.c_element_op};
                    sub_arguments.emplace_back(std::move(new_arg));
                }
                else
                {
                    index_t left_m, right_m;
                    AsGridPointer as_grid_right_ptr;
                    DsGridPointer ds_grid_right_ptr;
                    CDataType* c_grid_right_ptr;

                    ck::tie(
                        left_m, right_m, as_grid_right_ptr, ds_grid_right_ptr, c_grid_right_ptr) =
                        splitProblem(m, as_grid_ptr, ds_grid_ptr, c_grid_ptr);

                    split_m.push(left_m);
                    split_m.push(right_m);

                    as_grid_ptrs_queue.push(as_grid_ptr);
                    as_grid_ptrs_queue.push(as_grid_right_ptr);
                    ds_grid_ptrs_queue.push(ds_grid_ptr);
                    ds_grid_ptrs_queue.push(ds_grid_right_ptr);
                    c_grid_ptrs_queue.push(c_grid_ptr);
                    c_grid_ptrs_queue.push(c_grid_right_ptr);
                }

                split_m.pop();
                as_grid_ptrs_queue.pop();
                ds_grid_ptrs_queue.pop();
                c_grid_ptrs_queue.pop();
            }

            return sub_arguments;
        }
    };

    struct Invoker : public BaseInvoker
    {
        template <typename GridwiseGemm>
        float RunImpSinglePartition(const typename GridwiseGemm::Argument& arg,
                                    const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, arg.KBatch);

            const index_t k_grain = arg.KBatch * KPerBlock;
            const index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;
            const bool has_main_k_block_loop =
                GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            constexpr index_t minimum_occupancy = []() {
                if constexpr(BlkGemmPipeSched == BlockGemmPipelineScheduler::Interwave)
                {
                    return 2;
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    return (MPerBlock * NPerBlock / BlockSize <= 128) ? 2 : 1;
                }
                else
                {
                    return 1;
                }
            }();

            float ave_time = 0;
            const auto run_kernel = [&](const auto& kernel) {
                if(arg.KBatch > 1)
                {
                    hipGetErrorString(hipMemsetAsync(arg.p_c_grid,
                                                     0,
                                                     arg.M * arg.N * sizeof(CDataType),
                                                     stream_config.stream_id_));
                }

                ave_time = launch_and_time_kernel(
                    stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
            };

            if(has_main_k_block_loop)
            {
                if(arg.KBatch > 1)
                {
                    const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_abd<
                        GridwiseGemm,
                        true,
                        InMemoryDataOperationEnum::AtomicAdd,
                        minimum_occupancy>;
                    run_kernel(kernel);
                }
                else
                {
                    const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_abd<
                        GridwiseGemm,
                        true,
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy>;
                    run_kernel(kernel);
                }
            }
            else
            {
                if(arg.KBatch > 1)
                {
                    const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_abd<
                        GridwiseGemm,
                        false,
                        InMemoryDataOperationEnum::AtomicAdd,
                        minimum_occupancy>;
                    run_kernel(kernel);
                }
                else
                {
                    const auto kernel = kernel_gemm_xdl_cshuffle_v3_multi_abd<
                        GridwiseGemm,
                        false,
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy>;
                    run_kernel(kernel);
                }
            }

            return ave_time;
        }

        template <typename GridwiseGemm>
        float RunImp(const typename GridwiseGemm::Argument& arg,
                     const StreamConfig& stream_config = StreamConfig{})
        {
            Partitioner partitioner(
                arg.M, arg.N, arg.StrideAs, arg.StrideBs, arg.StrideDs, arg.StrideC);
            auto sub_arguments = partitioner.partitionGemmProblem(arg);

            return std::accumulate(sub_arguments.begin(),
                                   sub_arguments.end(),
                                   0.0f,
                                   [&](float sum, const auto& sub_arg) {
                                       return sum + RunImpSinglePartition<GridwiseGemm>(
                                                        sub_arg, stream_config);
                                   });
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            return RunImp<GridwiseGemm64>(arg, stream_config);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static auto MakeArgument(std::array<const void*, NumATensor> p_as,
                             std::array<const void*, NumBTensor> p_bs,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             std::array<index_t, NumATensor> StrideAs,
                             std::array<index_t, NumBTensor> StrideBs,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_as,
                        p_bs,
                        p_ds,
                        p_c,
                        M,
                        N,
                        K,
                        StrideAs,
                        StrideBs,
                        StrideDs,
                        StrideC,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(is_gfx11_supported() && arg.KBatch > 1)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: gfx11 does not support "
                             "KBatch > 1"
                          << std::endl;
            }
            return false;
        }

        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> && arg.KBatch > 1)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: bf16 atomic is not "
                             "supported for KBatch > 1"
                          << std::endl;
            }
            return false;
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: K=" << arg.K
                          << " must be divisible by AK1=" << AK1 << " and BK1=" << BK1
                          << " without K padding" << std::endl;
            }
            return false;
        }

        if(CDEShuffleBlockTransferScalarPerVectors{}[Number<0>{}] <= 1 && arg.KBatch > 1)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: CDE shuffle vector for "
                             "C must be greater than 1 when KBatch > 1"
                          << std::endl;
            }
            return false;
        }

        Partitioner partitioner(arg.M, arg.N, arg.StrideAs, arg.StrideBs, arg.StrideDs, arg.StrideC);
        if(!partitioner.isPartitionable())
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: descriptors are not "
                             "partitionable under 2GB limits"
                          << std::endl;
            }
            return false;
        }

        if(get_warp_size() != 64)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: M2-2 HCU path expects "
                             "wave64"
                          << std::endl;
            }
            return false;
        }

        if constexpr(NXdlPerWave64 > 0)
        {
            auto sub_arguments = partitioner.partitionGemmProblem(arg);
            for(std::size_t i = 0; i < sub_arguments.size(); ++i)
            {
                const auto& sub_arg = sub_arguments[i];
                if(!GridwiseGemm64::CheckValidity(sub_arg))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: "
                                     "GridwiseGemm64::CheckValidity failed for partition "
                                  << i << " with M=" << sub_arg.M << ", N=" << sub_arg.N
                                  << ", K=" << sub_arg.K << ", KBatch=" << sub_arg.KBatch
                                  << std::endl;
                    }
                    return false;
                }
                if(!Partitioner::isDescriptorValidForGemm(sub_arg))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: descriptor "
                                     "size check failed for partition "
                                  << i << std::endl;
                    }
                    return false;
                }
            }
            return true;
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "DeviceGemmMultiABD_Xdl_CShuffle_V3 rejected: no valid wave64 warp "
                             "tile"
                          << std::endl;
            }
            return false;
        }
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(std::array<const void*, NumATensor> p_as,
                                                      std::array<const void*, NumBTensor> p_bs,
                                                      std::array<const void*, NumDTensor> p_ds,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      std::array<index_t, NumATensor> StrideAs,
                                                      std::array<index_t, NumBTensor> StrideBs,
                                                      std::array<index_t, NumDTensor> StrideDs,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(MakeArgument(p_as,
                                                       p_bs,
                                                       p_ds,
                                                       p_c,
                                                       M,
                                                       N,
                                                       K,
                                                       StrideAs,
                                                       StrideBs,
                                                       StrideDs,
                                                       StrideC,
                                                       KBatch,
                                                       a_element_op,
                                                       b_element_op,
                                                       c_element_op));
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream{};
        str << "DeviceGemmMultiABD_Xdl_CShuffle_V3"
            << "<" << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0] << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0] << ">"
            << " A#: " << NumATensor << ", "
            << "B#: " << NumBTensor << ", "
            << "D#: " << NumDTensor << ", "
            << "BlkSize: " << BlockSize << ", "
            << "BlkTile: " << MPerBlock << "x" << NPerBlock << "x" << KPerBlock << ", "
            << "WaveTile: " << MPerXDL << "x" << NPerXDL << ", "
            << "WaveMap: " << MXdlPerWave << "x" << NXdlPerWave << ", "
            << "Pipeline: "
            << (BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ? "v1" : "v3");
        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
