// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// HCU implementation of the upstream grouped Split-K two-stage contract.
#pragma once
#include "device_grouped_gemm_xdl_splitk_cshuffle.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DsLayout,
          typename DsDataType,
          typename ELayout,
          typename EDataType,
          typename CDEOperation>
__global__ void kernel_grouped_gemm_splitk_epilogue(const float* workspace,
                                                    EDataType* output,
                                                    index_t m,
                                                    index_t n,
                                                    index_t output_stride,
                                                    Array<const void*, DsDataType::Size()> ds,
                                                    Array<index_t, DsDataType::Size()> strides,
                                                    CDEOperation operation)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= size_t(m) * n)
        return;
    const index_t row = i / n, col = i % n;
    constexpr bool row_e  = is_same_v<ELayout, tensor_layout::gemm::RowMajor>;
    const size_t c_offset = row_e ? size_t(row) * n + col : size_t(col) * m + row;
    const size_t e_offset =
        row_e ? size_t(row) * output_stride + col : size_t(col) * output_stride + row;
    const auto values = generate_tuple(
        [&](auto d) {
            using T             = remove_cvref_t<tuple_element_t<d.value, DsDataType>>;
            using L             = remove_cvref_t<tuple_element_t<d.value, DsLayout>>;
            const size_t offset = is_same_v<L, tensor_layout::gemm::RowMajor>
                                      ? size_t(row) * strides[d] + col
                                      : size_t(col) * strides[d] + row;
            return static_cast<const T*>(ds[d])[offset];
        },
        Number<DsDataType::Size()>{});
    EDataType e;
    unpack([&](const auto&... d) { operation(e, workspace[c_offset], d...); }, values);
    output[e_offset] = e;
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
          ck::index_t NumGemmKPrefetchStage,
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
          typename ABlockTransferThreadClusterLengths_KBatch_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_KBatch_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          typename ComputeDataType    = EDataType,
          // TODO: change gridwise_gemm_v2r4r2 to support AK1 & BK1
          enable_if_t<AK1 == BK1, bool> = false>
struct DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage
    : public DeviceGroupedGemmSplitK<ALayout,
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
    static_assert(DsLayout::Size() == NumDTensor, "D layout/type count mismatch");
    static_assert(PipelineVer == PipelineVersion::v1, "HCU TwoStage supports the v1 interface");
    using Pass = element_wise::PassThrough;
    using Gemm = DeviceGroupedGemmXdlSplitKCShuffle<
        ALayout,
        BLayout,
        Tuple<>,
        ELayout,
        ADataType,
        BDataType,
        AccDataType,
        float,
        Tuple<>,
        float,
        AElementwiseOperation,
        BElementwiseOperation,
        Pass,
        GemmSpec,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_KBatch_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_KBatch_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;

    struct Argument : BaseArgument
    {
        static std::vector<GemmDesc> WorkspaceDescs(const std::vector<GemmDesc>& descs)
        {
            auto result = descs;
            for(auto& d : result)
                d.stride_C_ = is_same_v<ELayout, tensor_layout::gemm::RowMajor> ? d.N_ : d.M_;
            return result;
        }
        Argument(std::vector<const void*>& a,
                 std::vector<const void*>& b,
                 std::vector<std::array<const void*, NumDTensor>>& ds,
                 std::vector<void*>& e,
                 std::vector<GemmDesc>& descs,
                 AElementwiseOperation,
                 BElementwiseOperation,
                 CDEElementwiseOperation operation)
            : workspace_descs_(WorkspaceDescs(descs)),
              gemm_(a, b, e, workspace_descs_),
              descs_(descs),
              ds_(ds),
              es_(e),
              operation_(operation)
        {
            if(ds.size() != descs.size())
                throw std::invalid_argument("TwoStage: D group count mismatch");
            for(size_t i = 0; i < descs.size(); ++i)
            {
                if(descs[i].stride_Ds_.size() != NumDTensor)
                    throw std::invalid_argument("TwoStage: D stride count mismatch");
                if(descs[i].M_ != 0 && descs[i].N_ != 0)
                    groups_.push_back(i);
            }
        }
        size_t WorkspaceSize() const
        {
            size_t elements = 0;
            for(const auto& trans : gemm_.gemm_kernel_args_)
            {
                const auto& a      = trans.karg_;
                const size_t count = size_t(a.MPadded) * a.NPadded;
                if(count > std::numeric_limits<size_t>::max() / sizeof(float) - elements)
                    throw std::overflow_error("TwoStage: workspace size overflow");
                elements += count;
            }
            return elements * sizeof(float);
        }
        void BindWorkspace(void* pointer)
        {
            p_workspace_  = pointer;
            size_t offset = 0;
            for(auto& trans : gemm_.gemm_kernel_args_)
            {
                auto& a    = trans.karg_;
                a.p_c_grid = pointer ? static_cast<float*>(pointer) + offset : nullptr;
                offset += size_t(a.MPadded) * a.NPadded;
            }
        }
        std::vector<GemmDesc> workspace_descs_;
        typename Gemm::Argument gemm_;
        std::vector<GemmDesc> descs_;
        std::vector<std::array<const void*, NumDTensor>> ds_;
        std::vector<void*> es_;
        std::vector<size_t> groups_;
        CDEElementwiseOperation operation_;
    };
    static bool IsSupportedArgument(const Argument& a)
    {
        // Before binding, gemm_ contains the caller's E addresses as placeholders.
        // FP32 workspace alignment must not be imposed on those final E tensors.
        if(!Gemm::IsSupportedArgument(a.gemm_, a.p_workspace_ != nullptr))
            return false;
        for(const auto g : a.groups_)
        {
            const auto& d = a.descs_[g];
            if(!grouped_splitk_detail::CheckMatrix<ELayout, EDataType>(
                   a.es_[g], d.M_, d.N_, d.stride_C_, 1))
                return false;
            bool valid = true;
            static_for<0, NumDTensor, 1>{}([&](auto j) {
                using T = remove_cvref_t<tuple_element_t<j.value, DsDataType>>;
                using L = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;
                valid &= grouped_splitk_detail::CheckMatrix<L, T>(
                    a.ds_[g][j], d.M_, d.N_, d.stride_Ds_[j], 1);
            });
            if(!valid)
                return false;
        }
        return true;
    }
    bool IsSupportedArgument(const BaseArgument* a) override
    {
        auto p = dynamic_cast<const Argument*>(a);
        return p && IsSupportedArgument(*p);
    }
    struct Invoker : BaseInvoker
    {
        float Run(const Argument& a, const StreamConfig& config = StreamConfig{})
        {
            if(!IsSupportedArgument(a))
                throw std::invalid_argument("TwoStage: unsupported argument");
            if(a.groups_.empty())
                return 0;
            if(!a.p_workspace_ || !a.gemm_.p_workspace_)
                throw std::invalid_argument(
                    "TwoStage: workspace and device argument buffers are required");
            auto one         = config;
            one.time_kernel_ = false;
            return grouped_splitk_detail::RunAndTime(config, [&] {
                typename Gemm::Invoker{}.Run(a.gemm_, one);
                for(size_t i = 0; i < a.groups_.size(); ++i)
                {
                    const auto g  = a.groups_[i];
                    const auto& d = a.descs_[g];
                    Array<const void*, NumDTensor> pointers;
                    Array<index_t, NumDTensor> strides;
                    static_for<0, NumDTensor, 1>{}([&](auto j) {
                        pointers(j) = a.ds_[g][j];
                        strides(j)  = d.stride_Ds_[j];
                    });
                    kernel_grouped_gemm_splitk_epilogue<DsLayout, DsDataType, ELayout, EDataType>
                        <<<dim3((size_t(d.M_) * d.N_ + 255) / 256),
                           dim3(256),
                           0,
                           config.stream_id_>>>(a.gemm_.gemm_kernel_args_[i].karg_.p_c_grid,
                                                static_cast<EDataType*>(a.es_[g]),
                                                d.M_,
                                                d.N_,
                                                d.stride_C_,
                                                pointers,
                                                strides,
                                                a.operation_);
                    hip_check_error(hipGetLastError());
                }
            });
        }
        float Run(const BaseArgument* a, const StreamConfig& config = StreamConfig{}) override
        { return Run(*dynamic_cast<const Argument*>(a), config); }
    };
    static auto MakeArgument(std::vector<const void*>& a,
                             std::vector<const void*>& b,
                             std::vector<std::array<const void*, NumDTensor>>& ds,
                             std::vector<void*>& e,
                             std::vector<GemmDesc>& descs,
                             AElementwiseOperation ao,
                             BElementwiseOperation bo,
                             CDEElementwiseOperation eo)
    { return Argument(a, b, ds, e, descs, ao, bo, eo); }
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& a,
                        std::vector<const void*>& b,
                        std::vector<std::array<const void*, NumDTensor>>& ds,
                        std::vector<void*>& e,
                        std::vector<GemmDesc>& descs,
                        AElementwiseOperation ao,
                        BElementwiseOperation bo,
                        CDEElementwiseOperation eo) override
    { return std::make_unique<Argument>(a, b, ds, e, descs, ao, bo, eo); }
    static auto MakeInvoker() { return Invoker{}; }
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    { return std::make_unique<Invoker>(); }
    size_t GetDeviceKernelArgSize(const BaseArgument* a) const override
    { return Gemm{}.GetDeviceKernelArgSize(&dynamic_cast<const Argument*>(a)->gemm_); }
    void SetDeviceKernelArgs(BaseArgument* a, void* pointer) const override
    { Gemm{}.SetDeviceKernelArgs(&dynamic_cast<Argument*>(a)->gemm_, pointer); }
    size_t GetWorkSpaceSize(const BaseArgument* a) const override
    { return dynamic_cast<const Argument*>(a)->WorkspaceSize(); }
    void SetWorkSpacePointer(BaseArgument* a, void* pointer) const override
    { dynamic_cast<Argument*>(a)->BindWorkspace(pointer); }
    static void SetKBatchSize(Argument& a, index_t batch) { a.gemm_.UpdateKBatch(batch); }
    void SetKBatchSize(BaseArgument* a, index_t batch) const override
    { SetKBatchSize(*dynamic_cast<Argument*>(a), batch); }
    static constexpr bool IsValidCompilationParameter()
    { return Gemm::IsValidCompilationParameter(); }
    std::string GetTypeString() const override
    { return "DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage/" + Gemm{}.GetTypeString(); }
};
} // namespace device
} // namespace tensor_operation
} // namespace ck
