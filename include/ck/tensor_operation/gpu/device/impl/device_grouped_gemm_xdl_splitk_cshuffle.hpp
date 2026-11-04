// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp"
#include "grouped_gemm_splitk_utils.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_gemm_xdl_splitk(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                       const index_t group_count)
{
#if (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
     defined(__gfx936__) || defined(__gfx938__))
    constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();
    __shared__ uint8_t p_shared[shared_size];

    const index_t block_id = get_block_1d_id();
    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    index_t left  = 0;
    index_t right = group_count;
    while(left < right)
    {
        const index_t mid = left + (right - left) / 2;
        if(block_id >= gemm_desc_ptr[mid].block_end_)
            left = mid + 1;
        else
            right = mid;
    }
    if(left == group_count || block_id < gemm_desc_ptr[left].block_start_)
        return;
    const index_t group_id = left;

    GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation>(
        gemm_desc_ptr[group_id].karg_,
        static_cast<void*>(p_shared),
        gemm_desc_ptr[group_id].block_2_ctile_map_);
#else
    ignore = gemm_descs_const;
    ignore = group_count;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler(),
          // Current implementation does not support multiple D fusions.
          enable_if_t<AK1 == BK1 && is_same_v<DsLayout, ck::Tuple<>> &&
                          is_same_v<DsDataType, ck::Tuple<>>,
                      bool> = false>
struct DeviceGroupedGemmXdlSplitKCShuffle : public DeviceGroupedGemmSplitK<ALayout,
                                                                           BLayout,
                                                                           DsLayout,
                                                                           ELayout,
                                                                           ADataType,
                                                                           BDataType,
                                                                           DsDataType,
                                                                           EDataType,
                                                                           AElementwiseOperation,
                                                                           BElementwiseOperation,
                                                                           CDEElementwiseOperation>
{
    static constexpr index_t NumDTensor = DsDataType::Size();
    static_assert(is_same_v<ADataType, BDataType>, "Split-K requires identical A/B types");
    static_assert(NumPrefetch == 1, "Split-K currently supports one prefetch stage");
    static_assert(LoopSched == LoopScheduler::Default,
                  "Grouped Split-K currently supports the default scheduler");
    static_assert(is_same_v<AElementwiseOperation, element_wise::PassThrough> &&
                      is_same_v<BElementwiseOperation, element_wise::PassThrough> &&
                      is_same_v<CDEElementwiseOperation, element_wise::PassThrough>,
                  "Use TwoStage for a post-reduction epilogue");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static_assert(KPerBlock % AK1 == 0);
    static constexpr index_t K0PerBlock = KPerBlock / AK1;

    using GridwiseGemm = GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        EDataType,
        ALayout,
        BLayout,
        ELayout,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        AK1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferScalarPerVector_NPerBlock,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        true>;

    using CGridDesc_M_N = typename GridwiseGemm::CGridDesc_M_N;
    using Block2ETileMapKSplit =
        BlockToCTileMap_KSplit_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>;
    // Block2CTileMap configuration parameter.
    static constexpr index_t B2E_M01 = 8;
    using GroupedGemmBlock2ETileMap  = OffsettedBlockToCTileMap<Block2ETileMapKSplit>;
    using KernelArgument             = typename GridwiseGemm::Argument;

    struct GemmTransKernelArg
    {
        KernelArgument karg_;
        GroupedGemmBlock2ETileMap block_2_ctile_map_;
        index_t block_start_, block_end_;

        GemmTransKernelArg() = default;
        GemmTransKernelArg(KernelArgument&& karg,
                           GroupedGemmBlock2ETileMap&& b2c_map,
                           index_t block_start,
                           index_t block_end)
            : karg_{karg},
              block_2_ctile_map_{b2c_map},
              block_start_{block_start},
              block_end_{block_end}
        {
        }
    };

    static constexpr index_t DefaultKBatch = 1;

    // Argument
    struct Argument : public BaseArgument
    {

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs)
            : Argument(p_As, p_Bs, p_Es, gemm_descs, DefaultKBatch)
        {
            // TODO: use occupancy api to calculate appropriate batch size.
        }

        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<void*>& p_Es,
                 std::vector<GemmDesc>& gemm_descs,
                 index_t kbatch)
            : K_BATCH{kbatch}
        {
            if(kbatch <= 0 || gemm_descs.size() > size_t(std::numeric_limits<index_t>::max()))
                throw std::invalid_argument("Grouped Split-K: invalid batch/group count");
            grid_size_   = 0;
            group_count_ = ck::type_convert<ck::index_t>(gemm_descs.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/c.size");
            }

            gemm_kernel_args_.reserve(group_count_);

            skipped_group_count_ = 0;

            for(std::size_t i = 0; i < gemm_descs.size(); ++i)
            {
                const index_t M = gemm_descs[i].M_;
                const index_t N = gemm_descs[i].N_;
                const index_t K = gemm_descs[i].K_;

                grouped_splitk_detail::CheckShape(
                    M, N, K, MPerBlock, NPerBlock, KPerBlock, K_BATCH);
                if(M == 0 || N == 0)
                {
                    skipped_group_count_++;
                    continue;
                }

                const index_t stride_a = gemm_descs[i].stride_A_;
                const index_t stride_b = gemm_descs[i].stride_B_;
                const index_t stride_c = gemm_descs[i].stride_C_;

                const index_t m_padded = GridwiseGemm::CalculateMPadded(M);
                const index_t n_padded = GridwiseGemm::CalculateNPadded(N);
                const index_t k_padded = GridwiseGemm::CalculateKPadded(K, K_BATCH);
                const index_t k0       = GridwiseGemm::CalculateK0(K, K_BATCH);

                const auto c_grid_desc_m_n =
                    GridwiseGemm::MakeCGridDescriptor_M_N(M, N, m_padded, n_padded, stride_c);

                const auto local_b2c_tile_map =
                    Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, K_BATCH};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);

                if(grid_size_grp > std::numeric_limits<index_t>::max() - grid_size_)
                    throw std::overflow_error("Grouped Split-K: total grid exceeds index_t");
                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;
                grid_size_ += grid_size_grp;

                // block-to-e-tile map
                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                auto karg = KernelArgument{type_convert<const ADataType*>(p_As[i]),
                                           type_convert<const BDataType*>(p_Bs[i]),
                                           type_convert<EDataType*>(p_Es[i]),
                                           M,
                                           N,
                                           K,
                                           stride_a,
                                           stride_b,
                                           stride_c,
                                           m_padded,
                                           n_padded,
                                           k_padded,
                                           k0,
                                           K_BATCH};

                gemm_kernel_args_.emplace_back(
                    std::move(karg), std::move(grouped_block_2_ctile_map), block_start, block_end);
            }
        }

        /**
         * @brief      Recalculate group grid size for all gemms and update B2C maps.
         *
         * @param[in]  kbatch  The new splitK parameter value.
         */
        void UpdateKBatch(index_t kbatch)
        {
            // Validate the whole update before mutating the argument.
            if(kbatch <= 0)
                throw std::invalid_argument("Grouped Split-K: k_batch must be positive");
            uint64_t total_blocks = 0;
            for(const auto& trans : gemm_kernel_args_)
            {
                const auto& a = trans.karg_;
                grouped_splitk_detail::CheckShape(
                    a.M, a.N, a.K, MPerBlock, NPerBlock, KPerBlock, kbatch);
                total_blocks += uint64_t(a.MPadded / MPerBlock) * (a.NPadded / NPerBlock) * kbatch;
            }
            if(total_blocks > uint64_t(std::numeric_limits<index_t>::max()))
                throw std::overflow_error("Grouped Split-K: total grid exceeds index_t");
            K_BATCH    = kbatch;
            grid_size_ = 0;

            for(std::size_t i = 0; i < gemm_kernel_args_.size(); ++i)
            {

                auto& karg = gemm_kernel_args_[i].karg_;

                const index_t k_padded = GridwiseGemm::CalculateKPadded(karg.K, K_BATCH);
                const index_t k0       = GridwiseGemm::CalculateK0(karg.K, K_BATCH);

                const auto c_grid_desc_m_n = GridwiseGemm::MakeCGridDescriptor_M_N(
                    karg.M, karg.N, karg.MPadded, karg.NPadded, karg.StrideC);

                const auto local_b2c_tile_map =
                    Block2ETileMapKSplit{c_grid_desc_m_n, B2E_M01, K_BATCH};
                const index_t grid_size_grp = local_b2c_tile_map.CalculateGridSize(c_grid_desc_m_n);

                if(grid_size_grp > std::numeric_limits<index_t>::max() - grid_size_)
                    throw std::overflow_error("Grouped Split-K: total grid exceeds index_t");
                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_grp;
                grid_size_ += grid_size_grp;

                // block-to-e-tile map
                auto grouped_block_2_ctile_map =
                    GroupedGemmBlock2ETileMap(local_b2c_tile_map, block_start);

                karg.KPadded                            = k_padded;
                karg.K0                                 = k0;
                karg.k_batch                            = K_BATCH;
                gemm_kernel_args_[i].block_2_ctile_map_ = grouped_block_2_ctile_map;
                gemm_kernel_args_[i].block_start_       = block_start;
                gemm_kernel_args_[i].block_end_         = block_end;
            }
        }

        //  private:
        index_t K_BATCH;
        index_t group_count_;
        index_t skipped_group_count_;

        std::vector<GemmTransKernelArg> gemm_kernel_args_;
        index_t grid_size_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!DeviceGroupedGemmXdlSplitKCShuffle::IsSupportedArgument(arg))
                throw std::invalid_argument("Grouped Split-K: unsupported argument");
            if(arg.gemm_kernel_args_.empty())
                return 0;
            if(!arg.p_workspace_)
                throw std::invalid_argument("Grouped Split-K: device argument buffer is not bound");
            hip_check_error(
                hipMemcpyAsync(arg.p_workspace_,
                               arg.gemm_kernel_args_.data(),
                               arg.gemm_kernel_args_.size() * sizeof(GemmTransKernelArg),
                               hipMemcpyHostToDevice,
                               stream_config.stream_id_));
            const bool main_loop =
                GridwiseGemm::CalculateHasMainK0BlockLoop(arg.gemm_kernel_args_.front().karg_.K0);
            const auto launch = [&](auto has_main, auto operation) {
                const auto kernel = kernel_grouped_gemm_xdl_splitk<GridwiseGemm,
                                                                   GemmTransKernelArg,
                                                                   has_main.value,
                                                                   operation.value>;
                return grouped_splitk_detail::RunAndTime(stream_config, [&] {
                    if constexpr(operation.value == InMemoryDataOperationEnum::AtomicAdd)
                        for(const auto& trans : arg.gemm_kernel_args_)
                        {
                            const auto& a = trans.karg_;
                            hip_check_error(hipMemsetAsync(a.p_c_grid,
                                                           0,
                                                           size_t(a.M) * a.N * sizeof(EDataType),
                                                           stream_config.stream_id_));
                        }
                    kernel<<<dim3(arg.grid_size_), dim3(BlockSize), 0, stream_config.stream_id_>>>(
                        cast_pointer_to_constant_address_space(arg.p_workspace_),
                        arg.gemm_kernel_args_.size());
                    hip_check_error(hipGetLastError());
                });
            };
            if(arg.K_BATCH > 1)
            {
                if constexpr(!is_same_v<EDataType, bhalf_t>)
                {
                    constexpr auto add = integral_constant<InMemoryDataOperationEnum,
                                                           InMemoryDataOperationEnum::AtomicAdd>{};
                    return main_loop ? launch(Number<true>{}, add) : launch(Number<false>{}, add);
                }
            }
            constexpr auto set =
                integral_constant<InMemoryDataOperationEnum, InMemoryDataOperationEnum::Set>{};
            return main_loop ? launch(Number<true>{}, set) : launch(Number<false>{}, set);
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg, bool check_c_pointer = true)
    {
        // This CShuffle transfer vectorizes the output along N.
        if constexpr(!is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            return false;
        if((ck::type_convert<ck::index_t>(arg.gemm_kernel_args_.size()) +
            arg.skipped_group_count_) != arg.group_count_)
        {
            return false;
        }

        const auto arch = get_device_name();
        if((arch != "gfx936" && arch != "gfx938" && arch != "gfx908" && arch != "gfx90a") ||
           !is_xdl_supported() || arg.K_BATCH <= 0 ||
           (is_same_v<EDataType, bhalf_t> && arg.K_BATCH > 1))
            return false;
        bool main_loop = false;
        if(!arg.gemm_kernel_args_.empty())
            main_loop =
                GridwiseGemm::CalculateHasMainK0BlockLoop(arg.gemm_kernel_args_[0].karg_.K0);
        for(const auto& trans : arg.gemm_kernel_args_)
        {
            const auto& a = trans.karg_;
            if(!GridwiseGemm::CheckValidity(a) ||
               main_loop != GridwiseGemm::CalculateHasMainK0BlockLoop(a.K0) ||
               !grouped_splitk_detail::CheckMatrix<ALayout, ADataType>(
                   a.p_a_grid, a.M, a.K, a.StrideA, ABlockTransferSrcScalarPerVector) ||
               !grouped_splitk_detail::CheckMatrix<BLayout, BDataType>(
                   a.p_b_grid, a.K, a.N, a.StrideB, BBlockTransferSrcScalarPerVector) ||
               !grouped_splitk_detail::CheckMatrix<ELayout, EDataType>(
                   a.p_c_grid,
                   a.M,
                   a.N,
                   a.StrideC,
                   CDEBlockTransferScalarPerVector_NPerBlock,
                   check_c_pointer))
                return false;
            if(arg.K_BATCH > 1 &&
               a.StrideC != (is_same_v<ELayout, tensor_layout::gemm::RowMajor> ? a.N : a.M))
                return false;
        }
        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<std::array<const void*, NumDTensor>>&,
                             std::vector<void*>& p_Es,
                             std::vector<GemmDesc> gemm_descs,
                             AElementwiseOperation,
                             BElementwiseOperation,
                             CDEElementwiseOperation)
    {
        return Argument{p_As, p_Bs, p_Es, gemm_descs};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<std::array<const void*, NumDTensor>>&,
                        std::vector<void*>& p_Es,
                        std::vector<GemmDesc>& gemm_descs,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation) override
    {
        return std::make_unique<Argument>(p_As, p_Bs, p_Es, gemm_descs);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedGemm_XdlSplitK"
            << "<"
            << std::string(ALayout::name)[0] << ","
            << std::string(BLayout::name)[0] << ","
            << std::string(ELayout::name)[0] << ","
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->gemm_kernel_args_.size() *
               sizeof(GemmTransKernelArg);
    }

    size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const override
    { return GetWorkSpaceSize(p_arg); }

    void SetDeviceKernelArgs(BaseArgument* p_arg, void* pointer) const override
    { this->SetWorkSpacePointer(p_arg, pointer); }

    static void SetKBatchSize(Argument& arg, index_t kbatch) { arg.UpdateKBatch(kbatch); }

    // polymorphic
    void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const override
    {
        return SetKBatchSize(*dynamic_cast<Argument*>(p_arg), kbatch);
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
