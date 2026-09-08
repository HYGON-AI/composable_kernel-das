// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// [HCU移植总述] BWD pipeline 在 HCU MMAC 上的核心改动：
//
// 1. output-layout 体系：HCU MMAC 的 C 寄存器 lane 布局与 AMD MFMA 不同，
//    内部 C layout 不能被后续标量公式（softmax、dS=P*(dP-D)）直接当作逻辑
//    [M,N] 消费。解决方案：对每个 GEMM 的 C accumulator 调用
//    gemm_X.MakeCOutputBlockTile(c_acc)，将内部 C layout 转为逻辑 output-layout，
//    在 output-layout 中完成 softmax、dS 计算等标量操作。
//
// 2. LSE/D 广播：LSE 和 D 是 1D 行向量，需要沿 N 维广播到 2D [M,N] tile。
//    上游在内部 C layout 中用 lse[idx0]/d[idx0] 逐行广播，但 output-layout
//    中 idx0 不再对应逻辑行。解决方案：用 stride-0 的 2D LDS naive descriptor
//    （strides=(1,0)，即 (m,n) 都读 LDS[m]）构造广播视图，再以 output-layout
//    的 tile distribution 做 load_tile，得到与 output-layout 同分布的广播 tile。
//
// 3. G1(P^T@dO) 和 G3(dS^T*Q^T) 转置：上游用 TransposeC=true 让 warp GEMM
//    输出转置 C 分布，HCU 改为 TransposeC=false（HCU MMAC TransC 布局不兼容，
//    导致 dV/dK 约 83% wrong）。转置改为通过 LDS 中转：
//    - P^T：store p_out_gemm_main → ds_lds → pt_lds_read_window（转置读视图）
//    - dS^T：store ds_gemm → ds_lds → dst_lds_read_window（转置读视图）
//
// 4. 最终梯度输出：dQ/dK/dV 的 C accumulator 均通过 MakeCOutputBlockTile 转为
//    output-layout 后再 store/update_tile 到 DRAM。
template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdDQDKDVPipelineKRKTRVR
{
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType          = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType             = remove_cvref_t<typename Problem::DDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType         = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType         = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType         = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType         = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType      = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using FmhaDropout           = remove_cvref_t<typename Problem::FmhaDropout>;
    using HotLoopScheduler      = typename Policy::template HotLoopScheduler<Problem>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ      = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK      = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ     = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV     = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum         = Problem::BiasEnum;
    static constexpr bool kHasBiasGrad     = Problem::kHasBiasGrad;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentQGrad = 1;
    static constexpr index_t kAlignmentKGrad =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentKGrad<Problem>();
    static constexpr index_t kAlignmentVGrad =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentVGrad<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetTransposedAlignmentBias<Problem>();

    static constexpr const char* name = "kr_ktr_vr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp,
              typename BiasGradDramBlockWindowTmp,
              typename DebugDumpDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               const RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
               const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
               const DDramBlockWindowTmp& d_dram_block_window_tmp,
               const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               const BiasGradDramBlockWindowTmp& dbias_dram_block_window_tmp,
               const DebugDumpDramBlockWindowTmp& debug_dump_dram_block_window_tmp,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float raw_scale,
               float scale,
               float rp_undrop,
               float scale_rp_undrop,
               void* smem_ptr,
               FmhaDropout& dropout,
               ck_tile::index_t debug_dump_mode = 0) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                std::is_same_v<OGradDataType,
                               remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                std::is_same_v<LSEDataType,
                               remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                std::is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasGradDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        // init VGrad & KGrad
        auto dv_acc = decltype(gemm_1.MakeCBlockTile()){};
        auto dk_acc = decltype(gemm_3.MakeCBlockTile()){};

        // K, HBM ->LDS ->Reg
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             k_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());

        const auto k_origin = k_dram_window.get_window_origin();
        // Early termination
        const auto [seqlen_q_start, seqlen_q_end] =
            mask.GetTileRangeAlongY(k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_q_end - seqlen_q_start, kM0);

        // check early exit if masked and no work to do.
        if constexpr(FmhaMask::IsMasking)
        {
            if(num_total_loop <= 0)
            {
                // Note: here dk_acc&dv_acc are all cleard, return it
                // Note: v loaded but no fence, ignore it.
                return make_tuple(gemm_3.MakeCOutputBlockTile(dk_acc),
                                  gemm_1.MakeCOutputBlockTile(dv_acc));
            }
        }
        KDataType* k_lds_ptr =
            static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());

        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKRegBlockDescriptor<Problem>());

        auto k_reg_tensor = make_static_distributed_tensor<KDataType>(
            Policy::template MakeKRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // V, HBM ->LDS ->Reg
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             v_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeVDramTileDistribution<Problem>());

        VDataType* v_lds_ptr =
            static_cast<VDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kK2>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // KT, Reg ->LDS ->Reg
        auto shuffled_k_block_tile = make_static_distributed_tensor<KDataType>(
            Policy::template MakeShuffledKRegWriteBlockDescriptor<Problem>());

        KDataType* kt_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));

        auto shuffled_k_lds_write = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeShuffledKLdsWriteBlockDescriptor<Problem>());

        auto shuffled_k_lds_write_window = make_tile_window(
            shuffled_k_lds_write, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto kt_lds_read = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeKTLdsReadBlockDescriptor<Problem>());

        auto kt_lds_read_window =
            make_tile_window(kt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

        //------------------------------------------------------------------
        // Pre-Load KV into Registers
        auto k_block_tile = load_tile(k_dram_window);
        auto v_block_tile = load_tile(v_dram_window);

        store_tile(k_lds_write_window, k_block_tile);
        shuffle_tile(shuffled_k_block_tile, k_block_tile);
        store_tile(shuffled_k_lds_write_window, shuffled_k_block_tile);

        block_sync_lds();
        k_reg_tensor = load_tile(k_lds_read_window);
        block_sync_lds();

        auto kt_reg_tensor = load_tile(kt_lds_read_window);

        store_tile(v_lds_write_window, v_block_tile);

        block_sync_lds();

        auto v_reg_tensor = load_tile(v_lds_read_window);
        block_sync_lds();
        //---------------------------- Loop Load in ----------------------------//
        // Q: HBM ->Reg ->LDS
        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeQDramTileDistribution<Problem>());

        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>()));

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        auto pt_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakePTRegSliceBlockDescriptor<Problem>());
        // QT: Reg -> Reg-> LDS
        auto shuffled_q_block_tile = make_static_distributed_tensor<QDataType>(
            Policy::template MakeShuffledQRegWriteBlockDescriptor<Problem>());

        QDataType* qt_lds_ptr =
            static_cast<QDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto shuffled_q_lds_write = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeShuffledQLdsWriteBlockDescriptor<Problem>());

        auto shuffled_q_lds_write_window = make_tile_window(
            shuffled_q_lds_write, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto qt_lds_read = make_tensor_view<address_space_enum::lds>(
            qt_lds_ptr, Policy::template MakeQTLdsReadBlockDescriptor<Problem>());

        auto qt_lds_read_window =
            make_tile_window(qt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeQTRegSliceBlockDescriptor<Problem>());

        // dO: HBM ->Reg ->LDS
        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, 0},
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>()));

        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());

        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto do_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());
        // dOT: Reg ->Reg ->LDS
        auto shuffled_do_block_tile = make_static_distributed_tensor<OGradDataType>(
            Policy::template MakeShuffledOGradRegWriteBlockDescriptor<Problem>());

        OGradDataType* dot_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>()));

        auto shuffled_do_lds_write = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeShuffledOGradLdsWriteBlockDescriptor<Problem>());

        auto shuffled_do_lds_write_window = make_tile_window(
            shuffled_do_lds_write, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto dot_read_lds = make_tensor_view<address_space_enum::lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsReadBlockDescriptor<Problem>());

        auto dot_lds_read_window =
            make_tile_window(dot_read_lds,
                             make_tuple(number<kVHeaddim>{}, number<kM0>{}),
                             {0, 0},
                             Policy::template MakeOGradTRegSliceBlockDescriptor<Problem>());

        // dS: Reg -> Reg -> LDS
        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto ds_lds = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());

        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto ds_lds_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK4>{}),
                             ds_lds_window.get_window_origin(),
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());

        // [HCU移植] dS/dS^T/P^T 共享同一块 ds_lds 的转置读视图。
        // dst_lds_read 是 ds_lds 的转置版本：(M,N) → (N,M)，用于：
        // - pt_lds_read_window：G1 输入 P^T（从 p_out_gemm_main 写入 ds_lds 后转置读出）
        // - dst_lds_read_window：G3 输入 dS^T（从 ds_gemm 写入 ds_lds 后转置读出）
        // 这是 TransposeC=false 替代方案的核心：转置通过 LDS 物理转置而非 warp GEMM TransC。
        auto dst_lds_read = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr,
            transform_tensor_descriptor(
                Policy::template MakeSGradLdsBlockDescriptor<Problem>(),
                make_tuple(make_pass_through_transform(number<kN0>{}),
                           make_pass_through_transform(number<kM0>{})),
                make_tuple(sequence<1>{}, sequence<0>{}),
                make_tuple(sequence<0>{}, sequence<1>{})));

        auto pt_lds_read_window =
            make_tile_window(dst_lds_read,
                             make_tuple(number<kN0>{}, number<kK1>{}),
                             {0, 0},
                             Policy::template MakePTRegSliceBlockDescriptor<Problem>());

        auto dst_lds_read_window =
            make_tile_window(dst_lds_read,
                             make_tuple(number<kN0>{}, number<kK3>{}),
                             {0, 0},
                             Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());

        auto dst_reg_tensor = make_static_distributed_tensor<GemmDataType>(
            Policy::template MakeSGradTRegSliceBlockDescriptor<Problem>());
        // Bias: HBM ->Reg ->Reg ->LDS
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();

        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, bias_origin.at(number<1>{})},
                             Policy::template MakeBiasTileDistribution<Problem>());

        BiasDataType* bias_lds_ptr = static_cast<BiasDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto bias_lds = make_tensor_view<address_space_enum::lds>(
            bias_lds_ptr, Policy::template MakeBiasLdsBlockDescriptor<Problem>());

        auto bias_lds_write_window =
            make_tile_window(bias_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto bias_s_lds_read_window =
            make_tile_window(bias_lds_write_window.get_bottom_tensor_view(),
                             bias_lds_write_window.get_window_lengths(),
                             bias_lds_write_window.get_window_origin(),
                             Policy::template MakeBiasSTileDistribution<decltype(gemm_0)>());

        static_assert(std::is_same_v<BiasDataType, BiasGradDataType>,
                      "BiasDataType and BiasGradDataType should be the same!");

        // LSE: HBM -> LDS ->Reg
        auto lse_dram_window = make_tile_window(
            lse_dram_block_window_tmp.get_bottom_tensor_view(),
            lse_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_start},
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        LSEDataType* lse_lds_ptr = static_cast<LSEDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>()));

        auto lse_lds = make_tensor_view<address_space_enum::lds>(
            lse_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto lse_lds_write_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0});

        auto lse_lds_read_window = make_tile_window(
            lse_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        // D: HBM ->Reg
        auto d_dram_window = make_tile_window(
            d_dram_block_window_tmp.get_bottom_tensor_view(),
            d_dram_block_window_tmp.get_window_lengths(),
            {seqlen_q_start},
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        DDataType* d_lds_ptr = static_cast<DDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() + Policy::template GetSmemSizeLSE<Problem>()));

        auto d_lds = make_tensor_view<address_space_enum::lds>(
            d_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto d_lds_write_window = make_tile_window(d_lds, make_tuple(number<kM0>{}), {0});

        // RandVal: HBM ->Reg
        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0), false>(
            randval_dram_block_window_tmp, seqlen_q_start);

        // BiasGrad
        // Reg ->LDS ->Reg ->HBM
        const auto dbias_origin = dbias_dram_block_window_tmp.get_window_origin();

        auto dbias_dram_window =
            make_tile_window(dbias_dram_block_window_tmp.get_bottom_tensor_view(),
                             dbias_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, dbias_origin.at(number<1>{})}); // M/N

        auto debug_dump_dram_window =
            make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                             debug_dump_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_q_start, dbias_origin.at(number<1>{})}); // M/N

        auto dbias_lds_read_window =
            make_tile_window(bias_lds,
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeShuffledBiasTileDistribution<Problem>());

        // ----------------------------Loop write out------------------------------//
        auto dq_dram_window = make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dq_dram_block_window_tmp.get_window_lengths(),
                                               {seqlen_q_start, 0});

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType  = decltype(gemm_4.MakeCBlockTile());

        index_t i_total_loops = 0;
        index_t seqlen_q_step = seqlen_q_start;
        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kM0 == kK1, "kM0 should equal to kK1");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        static_assert(kM0 == kK3, "kM0 should equal to kK3");
        constexpr index_t k4_loops = kN0 / kK4;

        clear_tile(dv_acc);
        clear_tile(dk_acc);

        if(debug_dump_mode == 6)
        {
            auto debug_dump_v_dram_window =
                make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<kN0>{}, number<kK2>{}),
                                 {dbias_origin.at(number<1>{}), 0});
            store_tile(debug_dump_v_dram_window, cast_tile<BiasGradDataType>(v_reg_tensor));
            return make_tuple(gemm_3.MakeCOutputBlockTile(dk_acc),
                              gemm_1.MakeCOutputBlockTile(dv_acc));
        }

        if(debug_dump_mode == 15)
        {
            auto debug_dump_k_dram_window =
                make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<kN0>{}, number<kK0>{}),
                                 {dbias_origin.at(number<1>{}), 0});
            store_tile(debug_dump_k_dram_window, cast_tile<BiasGradDataType>(k_reg_tensor));
            return make_tuple(gemm_3.MakeCOutputBlockTile(dk_acc),
                              gemm_1.MakeCOutputBlockTile(dv_acc));
        }

        __builtin_amdgcn_sched_barrier(0);
        // Hot loop
        while(i_total_loops < num_total_loop)
        {
            auto q_block_tile = load_tile(q_dram_window);
            move_tile_window(q_dram_window, {kM0, 0});

            auto lse_block_tile = load_tile(lse_dram_window);
            move_tile_window(lse_dram_window, {kM0});

            store_tile(q_lds_window, q_block_tile);
            shuffle_tile(shuffled_q_block_tile, q_block_tile);
            store_tile(shuffled_q_lds_write_window, shuffled_q_block_tile);

            store_tile(lse_lds_write_window, lse_block_tile);

            block_sync_lds();

            auto q_reg_tensor = load_tile(q_lds_read_window);
            auto lse          = load_tile(lse_lds_read_window);

            block_sync_lds();

            if(debug_dump_mode == 14)
            {
                auto debug_dump_q_dram_window =
                    make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                     make_tuple(number<kM0>{}, number<kK0>{}),
                                     {seqlen_q_step, 0});
                store_tile(debug_dump_q_dram_window, cast_tile<BiasGradDataType>(q_reg_tensor));
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            // STAGE 1, Q@K Gemm0
            auto s_acc = SPBlockTileType{};

            clear_tile(s_acc);
            gemm_0(s_acc, q_reg_tensor, k_reg_tensor);

            // STAGE 2, Scale, Add bias, Mask, Softmax, Dropout
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const auto bias_tile    = load_tile(bias_dram_window);
                auto shuffled_bias_tile = make_static_distributed_tensor<BiasDataType>(
                    Policy::template MakeShuffledBiasTileDistribution<Problem>());
                shuffle_tile(shuffled_bias_tile, bias_tile);
                store_tile(bias_lds_write_window, shuffled_bias_tile);
                block_sync_lds();
                auto bias_s_tile = load_tile(bias_s_lds_read_window);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
                        x = scale * x + log2e_v<AccDataType> * type_convert<AccDataType>(y);
                    },
                    s_acc,
                    bias_s_tile);
                move_tile_window(bias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    seqlen_q_step, k_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = seqlen_q_step + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }

            if(debug_dump_mode == 3)
            {
                const auto debug_tile = tile_elementwise_in(
                    [&raw_scale](const auto& x) {
                        return ck_tile::type_convert<BiasGradDataType>(raw_scale * x);
                    },
                    s_acc);
                store_tile(debug_dump_dram_window, debug_tile);
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            static const auto get_validated_lse = [](LSEDataType raw_lse) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_lse == -numeric<LSEDataType>::infinity()
                               ? type_convert<LSEDataType>(0.f)
                               : raw_lse;
                }
                else
                {
                    return raw_lse;
                }
            };

            auto p                 = SPBlockTileType{};
            constexpr auto p_spans = decltype(p)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_lse         = log2e_v<LSEDataType> * get_validated_lse(lse[i_idx]);

                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p(i_j_idx) = exp2(s_acc[i_j_idx] - row_lse);
                    }
                    else
                    {
                        p(i_j_idx) = exp2(scale * s_acc[i_j_idx] - row_lse);
                    }
                });
            });

            if constexpr(FmhaDropout::IsDropout)
            {
                dropout.template Run<decltype(gemm_0), RandValOutputDataType>(
                    seqlen_q_step, k_origin.at(number<0>{}), p, randval_dram_window);
            }
            const auto p_gemm = [&]() {
                if constexpr(FmhaDropout::IsDropout)
                {
                    return tile_elementwise_in(
                        [](const auto& x) { return type_convert<GemmDataType>(x > 0.f ? x : 0.f); },
                        p);
                }
                else
                {
                    return cast_tile<GemmDataType>(p);
                }
            }();

            // [HCU移植] output-layout softmax：HCU MMAC 内部 C layout 不能直接做行规约，
            // 因此在 output-layout 中重新计算 P（而非使用内部 layout 的 p/p_gemm）。
            // 1. s_acc → s_out_main（output-layout）
            // 2. LSE 通过 stride-0 2D LDS 视图广播到 s_out_main 分布
            // 3. p_out_main = exp2(scale * s_out_main - log2e * lse_bcast_main)
            // 4. p_out_gemm_main = cast_tile<GemmDataType>(p_out_main)，供后续 G1/dS 使用
            const auto s_out_main = gemm_0.MakeCOutputBlockTile(s_acc);
            auto lse_bcast_lds_main = make_tensor_view<address_space_enum::lds>(
                lse_lds_ptr,
                make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN0>{}),
                                             make_tuple(number<1>{}, number<0>{}),
                                             number<1>{},
                                             number<1>{}));
            auto lse_bcast_lds_window_main =
                make_tile_window(lse_bcast_lds_main,
                                 make_tuple(number<kM0>{}, number<kN0>{}),
                                 {0, 0},
                                 s_out_main.get_tile_distribution());
            const auto lse_bcast_main = load_tile(lse_bcast_lds_window_main);
            auto p_out_main           = decltype(s_out_main){};
            constexpr auto p_out_main_spans = decltype(p_out_main)::get_distributed_spans();
            sweep_tile_span(p_out_main_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_out_main_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    const auto row_lse =
                        log2e_v<LSEDataType> * get_validated_lse(lse_bcast_main[i_j_idx]);
                    p_out_main(i_j_idx) = exp2(scale * s_out_main[i_j_idx] - row_lse);
                });
            });
            const auto p_out_gemm_main = cast_tile<GemmDataType>(p_out_main);

            if(debug_dump_mode == 1)
            {
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(p_gemm));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            if(debug_dump_mode == 13)
            {
                const auto s_out = gemm_0.MakeCOutputBlockTile(s_acc);
                const auto debug_tile = tile_elementwise_in(
                    [&raw_scale](const auto& x) {
                        return ck_tile::type_convert<BiasGradDataType>(raw_scale * x);
                    },
                    s_out);
                store_tile(debug_dump_dram_window, debug_tile);
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            if(debug_dump_mode == 8)
            {
                const auto s_out = gemm_0.MakeCOutputBlockTile(s_acc);
                auto lse_bcast_lds = make_tensor_view<address_space_enum::lds>(
                    lse_lds_ptr,
                    make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN0>{}),
                                                 make_tuple(number<1>{}, number<0>{}),
                                                 number<1>{},
                                                 number<1>{}));
                auto lse_bcast_lds_window =
                    make_tile_window(lse_bcast_lds,
                                     make_tuple(number<kM0>{}, number<kN0>{}),
                                     {0, 0},
                                     s_out.get_tile_distribution());
                const auto lse_bcast = load_tile(lse_bcast_lds_window);
                auto p_out           = decltype(s_out){};
                constexpr auto p_out_spans = decltype(p_out)::get_distributed_spans();
                sweep_tile_span(p_out_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_out_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        const auto row_lse =
                            log2e_v<LSEDataType> * get_validated_lse(lse_bcast[i_j_idx]);
                        p_out(i_j_idx) = exp2(scale * s_out[i_j_idx] - row_lse);
                    });
                });
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(p_out));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            if(debug_dump_mode == 11)
            {
                const auto p_out = gemm_0.MakeCOutputBlockTile(cast_tile<AccDataType>(p_gemm));
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(p_out));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            // STAGE 3, P^T@OGrad^T Gemm1
            auto do_block_tile = load_tile(do_dram_window);
            move_tile_window(do_dram_window, {kM0, 0});

            auto d_block_tile = load_tile(d_dram_window);
            move_tile_window(d_dram_window, {kM0});

            store_tile(do_lds_window, do_block_tile);
            shuffle_tile(shuffled_do_block_tile, do_block_tile);
            store_tile(shuffled_do_lds_write_window, shuffled_do_block_tile);

            store_tile(d_lds_write_window, d_block_tile);

            block_sync_lds();

            auto dot_reg_tensor = load_tile(dot_lds_read_window);

            block_sync_lds();

            // [HCU移植] G1 输入桥：P^T 通过 LDS 转置读获得。
            // 上游用 PTFromGemm0CToGemm1A 直接从 MFMA C 寄存器转置，
            // HCU 改为 store p_out_gemm_main → ds_lds → pt_lds_read_window（转置读）。
            // pt_lds_read_window 复用 dst_lds_read 转置视图，distribution 使用
            // MakePTRegSliceBlockDescriptor，与 G1 的 A warp 分布一致。
            store_tile(ds_lds_window, p_out_gemm_main);
            block_sync_lds();
            pt_reg_tensor = load_tile(pt_lds_read_window);
            block_sync_lds();

            if(debug_dump_mode == 16)
            {
                auto debug_dump_pt_dram_window =
                    make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                     make_tuple(number<kN0>{}, number<kK1>{}),
                                     {dbias_origin.at(number<1>{}), seqlen_q_step});
                store_tile(debug_dump_pt_dram_window, cast_tile<BiasGradDataType>(pt_reg_tensor));
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            gemm_1(dv_acc, pt_reg_tensor, dot_reg_tensor);

            // STAGE 4, OGrad@V Gemm2
            auto do_reg_tensor = load_tile(do_lds_read_window);
            block_sync_lds();

            if(debug_dump_mode == 5)
            {
                auto debug_dump_do_dram_window =
                    make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                     make_tuple(number<kM0>{}, number<kK2>{}),
                                     {seqlen_q_step, 0});
                store_tile(debug_dump_do_dram_window,
                           cast_tile<BiasGradDataType>(do_reg_tensor));
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            auto dp_acc = SPGradBlockTileType{};

            dp_acc = gemm_2(do_reg_tensor, v_reg_tensor);

            if(debug_dump_mode == 4)
            {
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(dp_acc));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            if(debug_dump_mode == 7)
            {
                const auto dp_out = gemm_2.MakeCOutputBlockTile(dp_acc);
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(dp_out));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            if(debug_dump_mode == 9)
            {
                const auto s_out = gemm_0.MakeCOutputBlockTile(s_acc);
                auto lse_bcast_lds = make_tensor_view<address_space_enum::lds>(
                    lse_lds_ptr,
                    make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN0>{}),
                                                 make_tuple(number<1>{}, number<0>{}),
                                                 number<1>{},
                                                 number<1>{}));
                auto lse_bcast_lds_window =
                    make_tile_window(lse_bcast_lds,
                                     make_tuple(number<kM0>{}, number<kN0>{}),
                                     {0, 0},
                                     s_out.get_tile_distribution());
                const auto lse_bcast = load_tile(lse_bcast_lds_window);
                auto p_out           = decltype(s_out){};
                constexpr auto p_out_spans = decltype(p_out)::get_distributed_spans();
                sweep_tile_span(p_out_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(p_out_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        const auto row_lse =
                            log2e_v<LSEDataType> * get_validated_lse(lse_bcast[i_j_idx]);
                        p_out(i_j_idx) = exp2(scale * s_out[i_j_idx] - row_lse);
                    });
                });
                const auto dp_out = gemm_2.MakeCOutputBlockTile(dp_acc);
                auto d_bcast_lds  = make_tensor_view<address_space_enum::lds>(
                    d_lds_ptr,
                    make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN0>{}),
                                                 make_tuple(number<1>{}, number<0>{}),
                                                 number<1>{},
                                                 number<1>{}));
                auto d_bcast_lds_window =
                    make_tile_window(d_bcast_lds,
                                     make_tuple(number<kM0>{}, number<kN0>{}),
                                     {0, 0},
                                     dp_out.get_tile_distribution());
                const auto d_bcast = load_tile(d_bcast_lds_window);
                auto ds_out       = decltype(dp_out){};
                constexpr auto ds_out_spans = decltype(ds_out)::get_distributed_spans();
                sweep_tile_span(ds_out_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(ds_out_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        bool undrop_flag = p_out[i_j_idx] >= 0;
                        ds_out(i_j_idx) =
                            p_out[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                  ? (dp_out[i_j_idx] - d_bcast[i_j_idx])
                                                  : d_bcast[i_j_idx]);
                    });
                });
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(ds_out));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            if(debug_dump_mode == 10)
            {
                const auto dp_out = gemm_2.MakeCOutputBlockTile(dp_acc);
                auto d_bcast_lds  = make_tensor_view<address_space_enum::lds>(
                    d_lds_ptr,
                    make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN0>{}),
                                                 make_tuple(number<1>{}, number<0>{}),
                                                 number<1>{},
                                                 number<1>{}));
                auto d_bcast_lds_window =
                    make_tile_window(d_bcast_lds,
                                     make_tuple(number<kM0>{}, number<kN0>{}),
                                     {0, 0},
                                     dp_out.get_tile_distribution());
                const auto d_out = load_tile(d_bcast_lds_window);
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(d_out));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            if(debug_dump_mode == 12)
            {
                const auto p_out  = gemm_0.MakeCOutputBlockTile(cast_tile<AccDataType>(p_gemm));
                const auto dp_out = gemm_2.MakeCOutputBlockTile(dp_acc);
                auto d_bcast_lds  = make_tensor_view<address_space_enum::lds>(
                    d_lds_ptr,
                    make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN0>{}),
                                                 make_tuple(number<1>{}, number<0>{}),
                                                 number<1>{},
                                                 number<1>{}));
                auto d_bcast_lds_window =
                    make_tile_window(d_bcast_lds,
                                     make_tuple(number<kM0>{}, number<kN0>{}),
                                     {0, 0},
                                     dp_out.get_tile_distribution());
                const auto d_bcast = load_tile(d_bcast_lds_window);
                auto ds_out       = decltype(dp_out){};
                constexpr auto ds_out_spans = decltype(ds_out)::get_distributed_spans();
                sweep_tile_span(ds_out_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(ds_out_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        bool undrop_flag       = p_out[i_j_idx] >= 0;
                        ds_out(i_j_idx) =
                            p_out[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                  ? (dp_out[i_j_idx] - d_bcast[i_j_idx])
                                                  : d_bcast[i_j_idx]);
                    });
                });
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(ds_out));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            // STAGE 5, dS = P * (dP - D) 在 output-layout 中计算
            // [HCU移植] 核心改动：
            // 1. dp_acc → dp_out：通过 MakeCOutputBlockTile 将 G2 C accumulator
            //    从内部 C layout 转为逻辑 output-layout（已验证 dpout valid:y）
            // 2. D 广播：用 stride-0 2D LDS 视图（strides=(1,0)），以 dp_out 的
            //    tile distribution 做 load_tile 得到 d_bcast，与 dp_out 同分布
            // 3. ds_out = p_out_main * (dp_out - d_bcast)：全部在 output-layout 中计算
            // 注意：此前在内部 C layout 中直接用 d[idx0] 广播 D，idx0 在 output-layout
            // 下不再对应逻辑行，导致 dS 约 93% wrong。stride-0 LDS 广播解决了此问题。
            const auto dp_out = gemm_2.MakeCOutputBlockTile(dp_acc);
            auto d_bcast_lds  = make_tensor_view<address_space_enum::lds>(
                d_lds_ptr,
                make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN0>{}),
                                             make_tuple(number<1>{}, number<0>{}),
                                             number<1>{},
                                             number<1>{}));
            auto d_bcast_lds_window =
                make_tile_window(d_bcast_lds,
                                 make_tuple(number<kM0>{}, number<kN0>{}),
                                 {0, 0},
                                 dp_out.get_tile_distribution());
            const auto d_bcast = load_tile(d_bcast_lds_window);

            auto ds_out       = decltype(dp_out){};
            constexpr auto ds_out_spans = decltype(ds_out)::get_distributed_spans();
            sweep_tile_span(ds_out_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(ds_out_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    bool undrop_flag       = p_out_main[i_j_idx] >= 0;
                    ds_out(i_j_idx) =
                        p_out_main[i_j_idx] * (!FmhaDropout::IsDropout || undrop_flag
                                                   ? (dp_out[i_j_idx] - d_bcast[i_j_idx])
                                                   : d_bcast[i_j_idx]);
                });
            });

            if constexpr(kHasBiasGrad)
            {
                const auto dbias = [&]() {
                    if constexpr(FmhaDropout::IsDropout)
                    {
                        return tile_elementwise_in(
                            [&rp_undrop](const auto& x) {
                                return type_convert<BiasGradDataType>(x * rp_undrop);
                            },
                            ds_out);
                    }
                    else
                    {
                        return cast_tile<BiasGradDataType>(ds_out);
                    }
                }();
                store_tile(bias_lds_write_window, dbias);
                block_sync_lds();
                auto shuffled_dbias_tile = load_tile(dbias_lds_read_window);
                auto dbias_tile          = make_static_distributed_tensor<BiasGradDataType>(
                    Policy::template MakeBiasTileDistribution<Problem>());
                shuffle_tile(dbias_tile, shuffled_dbias_tile);
                store_tile(dbias_dram_window, dbias_tile);
                move_tile_window(dbias_dram_window, {kM0, 0});
                __builtin_amdgcn_sched_barrier(0);
            }

            // STAGE 6, dS^T@Q^T Gemm3
            // [HCU移植] dS^T 输入桥：ds_gemm（output-layout ds_out 的半精度版本）
            // 先 store 到 ds_lds，再通过 dst_lds_read_window（转置读视图）读出 dS^T。
            // 这是 G3 TransposeC=false 替代方案的核心。
            auto qt_reg_tensor = load_tile(qt_lds_read_window);
            block_sync_lds();

            const auto ds_gemm = cast_tile<GemmDataType>(ds_out);

            if(debug_dump_mode == 2)
            {
                store_tile(debug_dump_dram_window, cast_tile<BiasGradDataType>(ds_gemm));
                move_tile_window(debug_dump_dram_window, {kM0, 0});
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            store_tile(ds_lds_window, ds_gemm);
            block_sync_lds();

            dst_reg_tensor = load_tile(dst_lds_read_window);

            if(debug_dump_mode == 17)
            {
                auto debug_dump_dst_dram_window =
                    make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                     make_tuple(number<kN0>{}, number<kK3>{}),
                                     {dbias_origin.at(number<1>{}), seqlen_q_step});
                store_tile(debug_dump_dst_dram_window,
                           cast_tile<BiasGradDataType>(dst_reg_tensor));
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }

            gemm_3(dk_acc, dst_reg_tensor, qt_reg_tensor);

            auto ds_reg_tensor      = load_tile(ds_lds_read_window);
            auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
            move_tile_window(ds_lds_read_window, {0, kK4});

            // STAGE7 SGrad@K^T Gemm4
            auto dq_acc = QGradBlockTileType{};
            clear_tile(dq_acc);

            static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor_next = load_tile(ds_lds_read_window);
                    move_tile_window(ds_lds_read_window, {0, kK4});
                }
                auto kt_reg_tensor_slice = get_slice_tile(kt_reg_tensor,
                                                          sequence<0, i_k4 * kK4>{},
                                                          sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);

                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
                }
            });
            move_tile_window(ds_lds_read_window, {0, -kN0});
            // QGrad Scale
            if constexpr(FmhaDropout::IsDropout)
            {
                tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                       dq_acc);
            }
            else
            {
                tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
            }
            if(debug_dump_mode == 20)
            {
                auto debug_dump_dq_dram_window =
                    make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                     make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                     {seqlen_q_step, 0});
                store_tile(debug_dump_dq_dram_window, cast_tile<BiasGradDataType>(dq_acc));
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }
            if(debug_dump_mode == 23)
            {
                auto debug_dump_dq_dram_window =
                    make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                     make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                     {seqlen_q_step, 0});
                const auto dq_out = gemm_4.MakeCOutputBlockTile(dq_acc);
                store_tile(debug_dump_dq_dram_window, cast_tile<BiasGradDataType>(dq_out));
                i_total_loops += 1;
                seqlen_q_step += kM0;
                continue;
            }
            // [HCU移植] dQ 输出：dq_acc 通过 MakeCOutputBlockTile 转为 output-layout
            // 后再 store/update_tile 到 DRAM。如果不做此转换，内部 C layout 的元素
            // 位置与逻辑 [M,K] 不对应，直接 store 会导致 dQ 约 88% wrong。
            if constexpr(kIsDeterministic)
            {
                const auto dq_out = gemm_4.MakeCOutputBlockTile(dq_acc);
                store_tile(dq_dram_window, dq_out);
            }
            else
            {
                const auto dq_out = gemm_4.MakeCOutputBlockTile(dq_acc);
                update_tile(dq_dram_window, dq_out);
            }
            move_tile_window(dq_dram_window, {kM0, 0});

            i_total_loops += 1;
            seqlen_q_step += kM0;
        }

        // Results Scale
        if constexpr(FmhaDropout::IsDropout)
        {
            tile_elementwise_inout([&scale_rp_undrop](auto& x) { x = x * scale_rp_undrop; },
                                   dk_acc);
            tile_elementwise_inout([&rp_undrop](auto& x) { x = x * rp_undrop; }, dv_acc);
        }
        else
        {
            tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dk_acc);
        }

        if(debug_dump_mode == 18)
        {
            auto debug_dump_dv_dram_window =
                make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                                 {dbias_origin.at(number<1>{}), 0});
            store_tile(debug_dump_dv_dram_window, cast_tile<BiasGradDataType>(dv_acc));
        }
        if(debug_dump_mode == 19)
        {
            auto debug_dump_dk_dram_window =
                make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                                 {dbias_origin.at(number<1>{}), 0});
            store_tile(debug_dump_dk_dram_window, cast_tile<BiasGradDataType>(dk_acc));
        }
        if(debug_dump_mode == 21)
        {
            auto debug_dump_dv_dram_window =
                make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<kN0>{}, number<kVHeaddim>{}),
                                 {dbias_origin.at(number<1>{}), 0});
            const auto dv_out = gemm_1.MakeCOutputBlockTile(dv_acc);
            store_tile(debug_dump_dv_dram_window, cast_tile<BiasGradDataType>(dv_out));
        }
        if(debug_dump_mode == 22)
        {
            auto debug_dump_dk_dram_window =
                make_tile_window(debug_dump_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<kN0>{}, number<kQKHeaddim>{}),
                                 {dbias_origin.at(number<1>{}), 0});
            const auto dk_out = gemm_3.MakeCOutputBlockTile(dk_acc);
            store_tile(debug_dump_dk_dram_window, cast_tile<BiasGradDataType>(dk_out));
        }
        // [HCU移植] dk_acc/dv_acc 通过 MakeCOutputBlockTile 转为 output-layout 后返回。
        // dQ 已在主循环中直接 store/update_tile 到 DRAM（需要 split-accumulate，不在这里返回）。
        // dK/dV 这里返回给 caller（在 fmha_bwd_kernel.hpp 中 scale 后最终写回）。
        return make_tuple(gemm_3.MakeCOutputBlockTile(dk_acc),
                          gemm_1.MakeCOutputBlockTile(dv_acc));
    }
};

// Generic name expected by the BWD codegen; on the HCU port the only dq_dk_dv
// pipeline variant is the KR/KTR/VR one.
template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
using BlockFmhaBwdDQDKDVPipeline = BlockFmhaBwdDQDKDVPipelineKRKTRVR<Problem, Policy>;

} // namespace ck_tile
