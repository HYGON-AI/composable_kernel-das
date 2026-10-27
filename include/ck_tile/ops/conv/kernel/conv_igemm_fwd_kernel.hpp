// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/conv/kernel/conv_host_args.hpp"
#include "ck_tile/ops/conv/utility/conv_fwd_spec.hpp"
#include "ck_tile/ops/conv/utility/conv_fwd_to_gemm_transformer.hpp"
#include "ck_tile/ops/conv/utility/grouped_conv_ptr_offset.hpp"

namespace ck_tile {

template <typename TilePartitioner_,
          typename IgemmPipeline_,
          typename EpiloguePipeline_,
          ConvFwdSpec Spec>
struct ConvIgemmFwdKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using IgemmPipeline    = remove_cvref_t<IgemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    static constexpr index_t NDimSpatial = IgemmPipeline::NDimSpatial;

    static constexpr index_t BlockSize = IgemmPipeline::BlockSize;

    static constexpr index_t AVectorLength = IgemmPipeline::AVectorLength;
    static constexpr index_t BVectorLength = IgemmPipeline::BVectorLength;
    static constexpr index_t CVectorLength = IgemmPipeline::CVectorLength;

    static constexpr index_t MPerBlock = IgemmPipeline::MPerBlock;
    static constexpr index_t NPerBlock = IgemmPipeline::NPerBlock;
    static constexpr index_t KPerBlock = IgemmPipeline::KPerBlock;

    using ALayout = typename IgemmPipeline::ALayout;
    using BLayout = typename IgemmPipeline::BLayout;
    using CLayout = typename IgemmPipeline::CLayout;

    using ADataType = typename IgemmPipeline::ADataType;
    using BDataType = typename IgemmPipeline::BDataType;
    using CDataType = typename IgemmPipeline::CDataType;

    using AElementwiseOp = typename IgemmPipeline::AElementwiseOp;
    using BElementwiseOp = typename IgemmPipeline::BElementwiseOp;
    using CElementwiseOp = typename IgemmPipeline::CElementwiseOp;

    using HostArgs = ConvFwdHostArgs<NDimSpatial>;

    static constexpr index_t NDim = HostArgs::NDim;

    static constexpr auto transformer =
        ConvFwdToGemmTransformer<NDimSpatial, AVectorLength, BVectorLength, CVectorLength, Spec>{};

    static constexpr auto padder = MatrixPadder_v2<true, true, true, index_t, index_t, index_t>{
        MPerBlock, NPerBlock, KPerBlock};

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    CK_TILE_HOST static auto MakeADescriptor(const HostArgs& args)
    {
        const auto in_gemmmraw_gemmkraw_desc =
            transformer.template MakeADescriptor_M_K<ALayout>(args.input.lengths,
                                                              args.input.strides,
                                                              args.weight.lengths,
                                                              args.weight.strides,
                                                              args.output.lengths,
                                                              args.output.strides,
                                                              args.conv_filter_strides,
                                                              args.conv_filter_dilations,
                                                              args.input_left_pads,
                                                              args.input_right_pads);

        const auto in_gemmm_gemmk_desc = padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);
        const auto gemm_m              = in_gemmm_gemmk_desc.get_length(I0);
        const auto gemm_k              = in_gemmm_gemmk_desc.get_length(I1);

        const auto in_b_gemmm_gemmk_desc =
            transform_tensor_descriptor(in_gemmm_gemmk_desc,
                                        make_tuple(make_pass_through_transform(gemm_m),
                                                   make_unmerge_transform(make_tuple(I1, gemm_k))),
                                        make_tuple(sequence<0>{}, sequence<1>{}),
                                        make_tuple(sequence<1>{}, sequence<0, 2>{}));

        return in_b_gemmm_gemmk_desc;
    }

    CK_TILE_HOST static auto MakeBDescriptor(const HostArgs& args)
    {
        const auto wei_gemmnraw_gemmkraw_desc = transformer.template MakeBDescriptor_N_K<BLayout>(
            args.weight.lengths, args.weight.strides);

        const auto wei_gemmn_gemmk_desc = padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);
        const auto gemm_n               = wei_gemmn_gemmk_desc.get_length(I0);
        const auto gemm_k               = wei_gemmn_gemmk_desc.get_length(I1);

        const auto wei_b_gemmn_gemmk_desc =
            transform_tensor_descriptor(wei_gemmn_gemmk_desc,
                                        make_tuple(make_pass_through_transform(gemm_n),
                                                   make_unmerge_transform(make_tuple(I1, gemm_k))),
                                        make_tuple(sequence<0>{}, sequence<1>{}),
                                        make_tuple(sequence<1>{}, sequence<0, 2>{}));

        return wei_b_gemmn_gemmk_desc;
    }

    CK_TILE_HOST static auto MakeCDescriptor(const HostArgs& args)
    {
        const auto out_gemmmraw_gemmnraw_desc = transformer.template MakeCDescriptor_M_N<CLayout>(
            args.output.lengths, args.output.strides);

        const auto out_gemmm_gemmn_desc = padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);

        return out_gemmm_gemmn_desc;
    }

    using ADesc = decltype(MakeADescriptor({}));
    using BDesc = decltype(MakeBDescriptor({}));
    using CDesc = decltype(MakeCDescriptor({}));

    struct KernelArgs
    {
        const ADataType* a_ptr;
        const BDataType* b_ptr;
        CDataType* c_ptr;

        index_t num_group;

        ADesc a_desc;
        BDesc b_desc;
        CDesc c_desc;

        AElementwiseOp a_element_op;
        BElementwiseOp b_element_op;
        CElementwiseOp c_element_op;

        TilePartitioner tile_partitioner;

        GroupedConvPtrOffset grouped_conv_ptr_offset;
    };

    CK_TILE_HOST static constexpr auto
    MakeHostArgs(const void* input_ptr_,
                 const void* weight_ptr_,
                 void* output_ptr_,
                 const std::array<index_t, NDim>& in_g_n_c_wis_lengths,
                 const std::array<index_t, NDim>& in_g_n_c_wis_strides,
                 const std::array<index_t, NDim>& wei_g_k_c_xs_lengths,
                 const std::array<index_t, NDim>& wei_g_k_c_xs_strides,
                 const std::array<index_t, NDim>& out_g_n_k_wos_lengths,
                 const std::array<index_t, NDim>& out_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial>& strides,
                 const std::array<index_t, NDimSpatial>& dilations,
                 const std::array<index_t, NDimSpatial>& left_pads,
                 const std::array<index_t, NDimSpatial>& right_pads)
    {
        return HostArgs(input_ptr_,
                        weight_ptr_,
                        output_ptr_,
                        in_g_n_c_wis_lengths,
                        in_g_n_c_wis_strides,
                        wei_g_k_c_xs_lengths,
                        wei_g_k_c_xs_strides,
                        out_g_n_k_wos_lengths,
                        out_g_n_k_wos_strides,
                        strides,
                        dilations,
                        left_pads,
                        right_pads);
    }

    CK_TILE_HOST static constexpr auto MakeKernelArgs(const HostArgs& hargs)
    {
        const auto a_desc_b_m_k = MakeADescriptor(hargs);
        const auto b_desc_b_n_k = MakeBDescriptor(hargs);
        const auto c_desc_m_n   = MakeCDescriptor(hargs);

        const index_t num_group = hargs.input.lengths[0];

        // TODO: consider TG swizzle
        return KernelArgs{
            static_cast<const ADataType*>(hargs.input_ptr),
            static_cast<const BDataType*>(hargs.weight_ptr),
            static_cast<CDataType*>(hargs.output_ptr),
            num_group,
            a_desc_b_m_k,
            b_desc_b_n_k,
            c_desc_m_n,
            AElementwiseOp{},
            BElementwiseOp{},
            CElementwiseOp{},
            {c_desc_m_n.get_length(I0), c_desc_m_n.get_length(I1), 1, 1, 1},
            {hargs.input.strides[0], hargs.weight.strides[0], hargs.output.strides[0]}};
    }

    CK_TILE_HOST static bool IsSupportedArgument(const HostArgs& hargs, const KernelArgs& kargs)
    {
        [[maybe_unused]] const auto& in_lengths  = hargs.input.lengths;
        [[maybe_unused]] const auto& in_strides  = hargs.input.strides;
        [[maybe_unused]] const auto& wei_lengths = hargs.weight.lengths;
        [[maybe_unused]] const auto& wei_strides = hargs.weight.strides;
        [[maybe_unused]] const auto& out_lengths = hargs.output.lengths;
        [[maybe_unused]] const auto& out_strides = hargs.output.strides;

        const auto unsupported = [](const char* msg) {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(msg);
            }
        };

        // a stride is only meaningful when the extent of its own dimension is not 1
        const auto stride_matches = [](index_t length, index_t stride, index_t expected) {
            return length == 1 || stride == expected;
        };

        // check specialization: the Filter1x1 descriptors ignore filter extents and pads,
        // and Filter1x1Stride1Pad0 additionally assumes unit conv strides
        if constexpr(Spec == ConvFwdSpec::Filter1x1Stride1Pad0 ||
                     Spec == ConvFwdSpec::Filter1x1Pad0)
        {
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                if(wei_lengths[i + 3] != 1 || hargs.input_left_pads[i] != 0 ||
                   hargs.input_right_pads[i] != 0)
                {
                    unsupported("ConvFwdSpec::Filter1x1* needs a 1x1 filter and no padding!");
                    return false;
                }
                if constexpr(Spec == ConvFwdSpec::Filter1x1Stride1Pad0)
                {
                    if(hargs.conv_filter_strides[i] != 1)
                    {
                        unsupported(
                            "ConvFwdSpec::Filter1x1Stride1Pad0 needs unit conv filter strides!");
                        return false;
                    }
                }
            }
        }

        // check vector access of A (input tensor), mirroring MakeADescriptor_M_K<ALayout>
        if constexpr(std::is_same_v<ALayout, tensor_layout::convolution::NGCHW>)
        {
            // descriptor (N, C, Hi, Wi) vectorizes along Wi
            if(in_strides[4] != 1)
            {
                unsupported("input tensor is not contiguous along Wi for NGCHW layout!");
                return false;
            }
            if(in_lengths[4] % AVectorLength != 0)
            {
                unsupported("Wi is not a multiple of vector load size for A tensor!");
                return false;
            }
        }
        else if constexpr(std::is_same_v<ALayout, tensor_layout::convolution::NHWGC>)
        {
            // descriptor (N, Hi, Wi, C) vectorizes along C and hardcodes the C stride to 1
            if(in_strides[2] != 1)
            {
                unsupported("input tensor is not contiguous along C for NHWGC layout!");
                return false;
            }
            if(in_lengths[2] % AVectorLength != 0)
            {
                unsupported("C is not a multiple of vector load size for A tensor!");
                return false;
            }
        }
        else if constexpr(std::is_base_of_v<tensor_layout::convolution::BaseNGCHWc, ALayout>)
        {
            // innermost extent is the compile-time packed size ALayout::x
            if constexpr(ALayout::x % AVectorLength != 0)
            {
                unsupported(
                    "packed size of A layout is not a multiple of vector load size for A tensor!");
                return false;
            }
        }

        // check vector access of B (weight tensor), mirroring MakeBDescriptor_N_K<BLayout>;
        // the descriptors are built packed, so the weight must actually be packed
        if constexpr(std::is_same_v<BLayout, tensor_layout::convolution::GKCYX>)
        {
            const index_t K = wei_lengths[1];
            const index_t C = wei_lengths[2];
            const index_t Y = wei_lengths[3];
            const index_t X = wei_lengths[4];

            if(X % BVectorLength != 0)
            {
                unsupported("X is not a multiple of vector load size for B tensor!");
                return false;
            }
            if(!stride_matches(X, wei_strides[4], 1) || !stride_matches(Y, wei_strides[3], X) ||
               !stride_matches(C, wei_strides[2], Y * X) ||
               !stride_matches(K, wei_strides[1], C * Y * X))
            {
                unsupported("weight tensor is not packed for GKCYX layout!");
                return false;
            }
        }
        else if constexpr(std::is_same_v<BLayout, tensor_layout::convolution::GKYXC>)
        {
            const index_t K = wei_lengths[1];
            const index_t Y = wei_lengths[3];
            const index_t X = wei_lengths[4];
            const index_t C = wei_lengths[2];

            if(C % BVectorLength != 0)
            {
                unsupported("C is not a multiple of vector load size for B tensor!");
                return false;
            }
            if(!stride_matches(C, wei_strides[2], 1) || !stride_matches(X, wei_strides[4], C) ||
               !stride_matches(Y, wei_strides[3], X * C) ||
               !stride_matches(K, wei_strides[1], Y * X * C))
            {
                unsupported("weight tensor is not packed for GKYXC layout!");
                return false;
            }
        }
        else if constexpr(std::is_base_of_v<tensor_layout::convolution::BaseGKCYXc, BLayout>)
        {
            constexpr index_t Cx = BLayout::x;

            const index_t K = wei_lengths[1];
            const index_t C = wei_lengths[2];
            const index_t Y = wei_lengths[3];
            const index_t X = wei_lengths[4];

            if constexpr(Cx % BVectorLength != 0)
            {
                unsupported(
                    "packed size of B layout is not a multiple of vector load size for B tensor!");
                return false;
            }
            if(!stride_matches(X, wei_strides[4], Cx) ||
               !stride_matches(Y, wei_strides[3], X * Cx) ||
               !stride_matches(C, wei_strides[2], Y * X * Cx) ||
               !stride_matches(K, wei_strides[1], C * Y * X * Cx))
            {
                unsupported("weight tensor is not packed for GKCYXc layout!");
                return false;
            }
        }

        // check vector access of C (output tensor), mirroring MakeCDescriptor_M_N<CLayout>
        if constexpr(std::is_same_v<CLayout, tensor_layout::convolution::NGKHW>)
        {
            // descriptor (N, K, Ho, Wo) vectorizes along Wo
            if(out_strides[4] != 1)
            {
                unsupported("output tensor is not contiguous along Wo for NGKHW layout!");
                return false;
            }
            if(out_lengths[4] % CVectorLength != 0)
            {
                unsupported("Wo is not a multiple of vector store size for C tensor!");
                return false;
            }
        }
        else if constexpr(std::is_same_v<CLayout, tensor_layout::convolution::NHWGK>)
        {
            // descriptor (N * Ho * Wo, K) vectorizes along K and hardcodes the K stride to 1;
            // the flattened gemm-M dimension also assumes (N, Ho, Wo) are uniformly strided
            const index_t N  = out_lengths[1];
            const index_t Ho = out_lengths[3];
            const index_t Wo = out_lengths[4];

            if(out_strides[2] != 1)
            {
                unsupported("output tensor is not contiguous along K for NHWGK layout!");
                return false;
            }
            if(out_lengths[2] % CVectorLength != 0)
            {
                unsupported("K is not a multiple of vector store size for C tensor!");
                return false;
            }
            if(!stride_matches(Ho, out_strides[3], Wo * out_strides[4]) ||
               !stride_matches(N, out_strides[1], Ho * out_strides[3]))
            {
                unsupported("output tensor is not uniformly strided for the flattened M "
                            "dimension of NHWGK layout!");
                return false;
            }
        }
        else if constexpr(std::is_base_of_v<tensor_layout::convolution::BaseNGCHWc, CLayout>)
        {
            // innermost extent is the compile-time packed size CLayout::x
            if constexpr(CLayout::x % CVectorLength != 0)
            {
                unsupported(
                    "packed size of C layout is not a multiple of vector store size for C tensor!");
                return false;
            }
        }

        // gemm compatibility check on the padded descriptors: a zero extent (e.g. an empty
        // tensor) cannot form even one block tile
        const index_t gemm_m = kargs.a_desc.get_length(number<1>{});
        const index_t gemm_n = kargs.b_desc.get_length(number<1>{});
        const index_t gemm_k = kargs.b_desc.get_length(number<2>{});

        if(gemm_m < MPerBlock || gemm_n < NPerBlock || gemm_k < KPerBlock)
        {
            unsupported("problem extents are too small to form one block tile!");
            return false;
        }

        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemByteSize()
    {
        return max(IgemmPipeline::GetSmemByteSize(), EpiloguePipeline::GetSmemByteSize());
    }

    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        // compute grouped conv ptr offset
        const index_t num_blocks_per_group =
            __builtin_amdgcn_readfirstlane(get_grid_size() / kargs.num_group);
        const index_t g_idx =
            __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_group);

        const long_index_t a_ptr_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(kargs.grouped_conv_ptr_offset.GetAPtrOffset(g_idx)));
        const long_index_t b_ptr_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(kargs.grouped_conv_ptr_offset.GetBPtrOffset(g_idx)));
        const long_index_t c_ptr_offset = __builtin_amdgcn_readfirstlane(
            static_cast<long_index_t>(kargs.grouped_conv_ptr_offset.GetCPtrOffset(g_idx)));

        auto a_tensor_view =
            make_tensor_view<address_space_enum::global>(kargs.a_ptr + a_ptr_offset, kargs.a_desc);
        auto b_tensor_view =
            make_tensor_view<address_space_enum::global>(kargs.b_ptr + b_ptr_offset, kargs.b_desc);
        auto c_tensor_view =
            make_tensor_view<address_space_enum::global>(kargs.c_ptr + c_ptr_offset, kargs.c_desc);

        // allocate LDS
        __shared__ char p_smem[GetSmemByteSize()];

        // block work idx
        const auto block_work_idx = kargs.tile_partitioner.GetOutputTileIndex(get_block_1d_id());

        const index_t block_work_idx_m =
            __builtin_amdgcn_readfirstlane(block_work_idx[number<1>{}] * MPerBlock);
        const index_t block_work_idx_n =
            __builtin_amdgcn_readfirstlane(block_work_idx[number<2>{}] * NPerBlock);

        // A DRAM tile window
        // for load
        auto a_copy_dram_window =
            make_tile_window(a_tensor_view,
                             make_tuple(number<1>{}, number<MPerBlock>{}, number<KPerBlock>{}),
                             {0, block_work_idx_m, 0});

        // B DRAM tile window
        // for load
        auto b_copy_dram_window =
            make_tile_window(b_tensor_view,
                             make_tuple(number<1>{}, number<NPerBlock>{}, number<KPerBlock>{}),
                             {0, block_work_idx_n, 0});

        index_t num_loop = kargs.a_desc.get_length(number<2>{}) / KPerBlock;

        // Implicit gemm pipeline
        const auto c_block_tensor = IgemmPipeline{}(a_copy_dram_window,
                                                    kargs.a_element_op,
                                                    b_copy_dram_window,
                                                    kargs.b_element_op,
                                                    num_loop,
                                                    p_smem);

        // Epilogue pipeline
        auto c_store_dram_window =
            make_tile_window(c_tensor_view,
                             make_tuple(number<EpiloguePipeline::MBlockTilePerIter>{},
                                        number<EpiloguePipeline::NBlockTilePerIter>{}),
                             {block_work_idx_m, block_work_idx_n});
        EpiloguePipeline{}(c_store_dram_window, c_block_tensor, kargs.c_element_op, p_smem);
    }
};

} // namespace ck_tile
