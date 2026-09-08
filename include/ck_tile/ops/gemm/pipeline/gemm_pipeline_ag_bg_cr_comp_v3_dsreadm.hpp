// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
#include "ck_tile/ops/conv/block/block_gemm_mmac_nt_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"

namespace ck_tile {

// Adapter from the universal GEMM shape to the HCU ds_read_m32x16 block GEMM
// contract already used by CK Tile convolution. The b16 DS load is exposed as
// an FP16 or BF16 fragment so the following MMAC keeps the input dtype.
template <typename Problem>
struct GemmDsreadmBlockProblem
{
    using ADataType   = remove_cvref_t<typename Problem::ADataType>;
    using BDataType   = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType = remove_cvref_t<typename Problem::CDataType>;
    using Shape       = remove_cvref_t<typename Problem::BlockGemmShape>;
    using BlockWarps  = typename Shape::BlockWarps;
    using WarpTile    = typename Shape::WarpTile;

    static_assert(is_any_of<ADataType, fp16_t, bf16_t>::value &&
                      std::is_same_v<ADataType, BDataType>,
                  "gfx936 dsreadm grouped GEMM supports matching FP16/BF16 inputs");
    static_assert(WarpTile::at(number<0>{}) == 32 &&
                      WarpTile::at(number<1>{}) == 32 &&
                      WarpTile::at(number<2>{}) == 16,
                  "dsreadm adapter currently requires a 32x32x16 warp tile");

    static constexpr index_t MPerBlock = Shape::kM;
    static constexpr index_t NPerBlock = Shape::kN;
    static constexpr index_t KPerBlock = Shape::kK;
    static constexpr index_t MWarps    = BlockWarps::at(number<0>{});
    static constexpr index_t NWarps    = BlockWarps::at(number<1>{});
    static constexpr index_t MmmacIter = 2;
    static constexpr index_t NmmacIter = 2;
    static constexpr index_t MPermmac  = 16;
    static constexpr index_t NPermmac  = 16;
    static constexpr index_t MmmacInterleave = 1;
    static constexpr index_t NmmacInterleave = 1;
    static constexpr index_t MWarpIter = MPerBlock / (MWarps * 32);
    static constexpr index_t NWarpIter = NPerBlock / (NWarps * 32);
    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr bool TransposeC   = Problem::TransposeC;
};

template <typename Problem>
struct GemmDsreadmBlockGemm
    : public BlockGemmMmacNTAsmemBSmemCregV1<GemmDsreadmBlockProblem<Problem>>
{
    using Base = BlockGemmMmacNTAsmemBSmemCregV1<GemmDsreadmBlockProblem<Problem>>;
    using WarpGemm = typename Base::WG;
};

// Variant used only by the staged-operand tuning pipeline below. Instead of
// retaining every K16 operand fragment for a full K64 block, it exposes one
// compile-time K stage at a time. Two such stage buffers can then overlap the
// next ds_read_m32x16 with MMAC on the current operands without changing the
// established full-K DSReadM pipeline.
template <typename Problem>
struct GemmDsreadmStageBlockGemm : public GemmDsreadmBlockGemm<Problem>
{
    using Base = GemmDsreadmBlockGemm<Problem>;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM)
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_N_OUTER)
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
    using WG = std::conditional_t<
        std::is_same_v<remove_cvref_t<typename Problem::ADataType>, bf16_t>,
        WarpGemmMmacBF16BF16F32_WT64x64x16_MR4NR4MI1NI1_NOuter,
        WarpGemmMmacF16F16F32_WT64x64x16_MR4NR4MI1NI1_NOuter>;
#else
    using WG = std::conditional_t<
        std::is_same_v<remove_cvref_t<typename Problem::ADataType>, bf16_t>,
        WarpGemmMmacBF16BF16F32_WT128x64x16_MR8NR4MI1NI1_NOuter,
        WarpGemmMmacF16F16F32_WT128x64x16_MR8NR4MI1NI1_NOuter>;
#endif
#else
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
    using WG = std::conditional_t<
        std::is_same_v<remove_cvref_t<typename Problem::ADataType>, bf16_t>,
        WarpGemmMmacBF16BF16F32_WT64x64x16_MR4NR4MI1NI1,
        WarpGemmMmacF16F16F32_WT64x64x16_MR4NR4MI1NI1>;
#else
    using WG = std::conditional_t<
        std::is_same_v<remove_cvref_t<typename Problem::ADataType>, bf16_t>,
        WarpGemmMmacBF16BF16F32_WT128x64x16_MR8NR4MI1NI1,
        WarpGemmMmacF16F16F32_WT128x64x16_MR8NR4MI1NI1>;
#endif
#endif
#else
    using WG   = typename Base::WG;
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_A_ALT_OPERAND)
    // The vendor grad-X kernel consumes B in the native ds_read_b64 order.
    // Pair it with the alternate DSReadM A register layout so both MMAC
    // operands use the same K-lane permutation without any VALU shuffle.
    using ADsreadm =
        WarpDsreadmDispatcher<typename Problem::ADataType, 1, 16, 2>;
#else
    using ADsreadm =
        WarpDsreadmDispatcher<typename Problem::ADataType,
                              GemmDsreadmBlockProblem<Problem>::MmmacIter,
                              GemmDsreadmBlockProblem<Problem>::MPermmac,
                              GemmDsreadmBlockProblem<Problem>::MmmacInterleave>;
#endif
    using BDsreadm =
        WarpDsreadmDispatcher<typename Problem::BDataType,
                              GemmDsreadmBlockProblem<Problem>::NmmacIter,
                              GemmDsreadmBlockProblem<Problem>::NPermmac,
                              GemmDsreadmBlockProblem<Problem>::NmmacInterleave>;

    static constexpr index_t MWarps = GemmDsreadmBlockProblem<Problem>::MWarps;
    static constexpr index_t NWarps = GemmDsreadmBlockProblem<Problem>::NWarps;
    static constexpr index_t MWarpIter =
        GemmDsreadmBlockProblem<Problem>::MWarpIter;
    static constexpr index_t NWarpIter =
        GemmDsreadmBlockProblem<Problem>::NWarpIter;
    static constexpr index_t KWarpIter = Base::KWarpIter;
    static constexpr index_t MPerBlockPerIter = Base::MPerBlockPerIter;
    static constexpr index_t NPerBlockPerIter = Base::NPerBlockPerIter;

    template <index_t KIter, typename ALdsWindow>
    CK_TILE_DEVICE auto LdsLoadAStage(const ALdsWindow& a_lds_window) const
    {
        static_assert(KIter < KWarpIter, "invalid staged A operand index");
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT)
        statically_indexed_array<
            decltype(LdsLoadAStageFragment<KIter, 0>(a_lds_window)),
            MWarpIter>
            warp_tensors;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_CONSUMER) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_ASM) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_ASM_RELAXED)
        static_assert(MWarpIter == 4,
                      "vendor A immediate consumer requires MIWT8_4");
        // The selected hipBLASLt solution computes one A LDS base and encodes
        // the four M-repeat displacements directly in ds_read_m32x16. Keep
        // the exceptional last-repeat wrap on a second base, matching its
        // v188/v192 address pair, instead of materializing four address VGPRs.
        const index_t lane    = get_thread_id() & 63;
        const index_t wave_id = get_thread_id() >> 6;
        const index_t lane_b2 = (lane >> 2) & 1;
        const index_t common_element =
            ((lane & 3) << 3) + (lane_b2 << 5) +
            (lane_b2 << 9) + (((lane >> 3) & 1) << 8) +
            ((lane >> 4) << 10);
        const index_t wave_id_m = wave_id & 1;
        const index_t base_byte =
            KIter * 0x4000 +
            ((common_element + (wave_id_m << 5)) << 1);
        const bool use_alt_base =
            (((lane >> 2) & 3) == 3) && (wave_id_m == 1);
        const auto* lds_base =
            reinterpret_cast<const char*>(
                a_lds_window.get_bottom_tensor_view()
                    .get_buffer_view()
                    .p_data_);
        const auto* a_base = lds_base + base_byte;
        const auto* a_alt_base =
            a_base - (use_alt_base ? 0x400 : 0);

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_ASM) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_ASM_RELAXED)
        const index_t a_base_addr =
            reinterpret_cast<uintptr_t>(a_base);
        const index_t a_alt_base_addr =
            reinterpret_cast<uintptr_t>(a_alt_base);
        thread_buffer<typename Problem::ADataType, 8> raw0;
        thread_buffer<typename Problem::ADataType, 8> raw1;
        thread_buffer<typename Problem::ADataType, 8> raw2;
        thread_buffer<typename Problem::ADataType, 8> raw3;
        auto& raw0_vec = raw0.template get_as<fp32x4_t>();
        auto& raw1_vec = raw1.template get_as<fp32x4_t>();
        auto& raw2_vec = raw2.template get_as<fp32x4_t>();
        auto& raw3_vec = raw3.template get_as<fp32x4_t>();
        asm volatile("ds_read_m32x16_b16 %0, %4 offset:0\n\t"
                     "ds_read_m32x16_b16 %1, %4 offset:128\n\t"
                     "ds_read_m32x16_b16 %2, %4 offset:256\n\t"
                     "ds_read_m32x16_b16 %3, %5 offset:384"
                     : "=v"(raw0_vec[number<0>{}]),
                       "=v"(raw1_vec[number<0>{}]),
                       "=v"(raw2_vec[number<0>{}]),
                       "=v"(raw3_vec[number<0>{}])
                     : "v"(a_base_addr), "v"(a_alt_base_addr)
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_A_IMMEDIATE_ASM_RELAXED)
                     );
#else
                     : "memory");
#endif
#else
        const auto raw0 = ADsreadm{}(
            reinterpret_cast<const typename Problem::ADataType*>(
                a_base));
        const auto raw1 = ADsreadm{}(
            reinterpret_cast<const typename Problem::ADataType*>(
                a_base + 0x80));
        const auto raw2 = ADsreadm{}(
            reinterpret_cast<const typename Problem::ADataType*>(
                a_base + 0x100));
        const auto raw3 = ADsreadm{}(
            reinterpret_cast<const typename Problem::ADataType*>(
                a_alt_base + 0x180));
#endif
        warp_tensors(number<0>{}) =
            make_static_distributed_tensor<typename Problem::ADataType>(
                make_static_tile_distribution(
                    typename ADsreadm::WarpStoreDstrEncoding{}),
                raw0.template get_as<typename Problem::ADataType>());
        warp_tensors(number<1>{}) =
            make_static_distributed_tensor<typename Problem::ADataType>(
                make_static_tile_distribution(
                    typename ADsreadm::WarpStoreDstrEncoding{}),
                raw1.template get_as<typename Problem::ADataType>());
        warp_tensors(number<2>{}) =
            make_static_distributed_tensor<typename Problem::ADataType>(
                make_static_tile_distribution(
                    typename ADsreadm::WarpStoreDstrEncoding{}),
                raw2.template get_as<typename Problem::ADataType>());
        warp_tensors(number<3>{}) =
            make_static_distributed_tensor<typename Problem::ADataType>(
                make_static_tile_distribution(
                    typename ADsreadm::WarpStoreDstrEncoding{}),
                raw3.template get_as<typename Problem::ADataType>());
#else
        static_for<0, MWarpIter, 1>{}([&](auto m_warp_iter) {
            warp_tensors(m_warp_iter) =
                LdsLoadAStageFragment<KIter, m_warp_iter>(
                    a_lds_window);
        });
#endif
        return warp_tensors;
#else
        const index_t warp_id_m = ck_tile::get_warp_id() / NWarps;
        auto warp_window        = make_tile_window(
            a_lds_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            a_lds_window.get_window_origin() +
                multi_index<2>{warp_id_m * WG::kM, KIter * WG::kK},
            make_static_tile_distribution(typename ADsreadm::WarpLoadDstrEncoding{}));

        statically_indexed_array<
            decltype(load_tile_by_dsreadm(ADsreadm{}, warp_window)),
            MWarpIter>
            warp_tensors;
        static_for<0, MWarpIter, 1>{}([&](auto m_warp_iter) {
            warp_tensors(m_warp_iter) =
                load_tile_by_dsreadm(ADsreadm{}, warp_window);
            if constexpr(m_warp_iter != MWarpIter - 1)
            {
                move_tile_window(warp_window,
                                 {MPerBlockPerIter, 0});
            }
        });
        return warp_tensors;
#endif
    }

    template <index_t KIter, index_t MIter, typename ALdsWindow>
    CK_TILE_DEVICE auto
    LdsLoadAStageFragment(const ALdsWindow& a_lds_window) const
    {
        static_assert(KIter < KWarpIter && MIter < MWarpIter);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT)
        // Exact grad-W hipBLASLt DSReadM addressing. The paired producer uses
        // four 16-KiB [A8KiB|B8KiB] slots and the m0 wave-pair swizzle bit.
        const index_t lane    = get_thread_id() & 63;
        const index_t wave_id = get_thread_id() >> 6;
        const index_t lane_b2 = (lane >> 2) & 1;
        const index_t common_element =
            ((lane & 3) << 3) + (lane_b2 << 5) +
            (lane_b2 << 9) + (((lane >> 3) & 1) << 8) +
            ((lane >> 4) << 10);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM)
        // Keep hipBLASLt's physical 2x4 wave ownership. Its wave id is n*2+m,
        // so the low bit selects the A/M wave.
        const index_t wave_id_m = wave_id & 1;
#else
        // hipBLASLt numbers the 2x4 waves as n*2+m, while CK numbers them as
        // m*4+n. Keep the BLAS lane swizzle but translate the logical M-wave
        // coordinate to CK's ordering.
        const index_t wave_id_m = wave_id >> 2;
#endif
        const index_t base_byte =
            (common_element + (wave_id_m << 5)) << 1;
        const bool use_alt_base = MIter == 3 &&
                                  (((lane >> 2) & 3) == 3) &&
                                  (wave_id_m == 1);
        const index_t address_byte =
            KIter * 0x4000 + base_byte + MIter * 0x80 -
            (use_alt_base ? 0x400 : 0);
        const auto* lds_base =
            reinterpret_cast<const char*>(
                a_lds_window.get_bottom_tensor_view()
                    .get_buffer_view()
                    .p_data_);
        const auto raw = ADsreadm{}(
            reinterpret_cast<const typename Problem::ADataType*>(
                lds_base + address_byte));
        return make_static_distributed_tensor<
            typename Problem::ADataType>(
            make_static_tile_distribution(
                typename ADsreadm::WarpStoreDstrEncoding{}),
            raw.template get_as<typename Problem::ADataType>());
#else
        const index_t warp_id_m = ck_tile::get_warp_id() / NWarps;
        auto warp_window        = make_tile_window(
            a_lds_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            a_lds_window.get_window_origin() +
                multi_index<2>{warp_id_m * WG::kM +
                                   MIter * MPerBlockPerIter,
                               KIter * WG::kK},
            make_static_tile_distribution(
                typename ADsreadm::WarpLoadDstrEncoding{}));
        return load_tile_by_dsreadm(ADsreadm{}, warp_window);
#endif
    }

    template <index_t KIter,
              typename ALdsWindow,
              typename AWarpTensor>
    CK_TILE_DEVICE auto
    LdsLoadAStageWithFirst(const ALdsWindow& a_lds_window,
                           const AWarpTensor& first) const
    {
        static_assert(KIter < KWarpIter && MWarpIter == 4);
        using WarpTensor = remove_cvref_t<AWarpTensor>;
        statically_indexed_array<WarpTensor, MWarpIter> warp_tensors;
        warp_tensors(number<0>{}) = first;
        static_for<1, MWarpIter, 1>{}([&](auto m_warp_iter) {
            warp_tensors(m_warp_iter) =
                LdsLoadAStageFragment<KIter, m_warp_iter>(
                    a_lds_window);
        });
        return warp_tensors;
    }

    template <index_t KIter, index_t NIter, typename BLdsWindow>
    CK_TILE_DEVICE auto
    LdsLoadBStageFragment(const BLdsWindow& b_lds_window) const
    {
        static_assert(KIter < KWarpIter && NIter < NWarpIter);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT)
        const index_t lane    = get_thread_id() & 63;
        const index_t wave_id = get_thread_id() >> 6;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_CONSUMER)
        // The Ailk_Bljk hipBLASLt kernel does not consume the DTL-produced B
        // tile with DSReadM. Each wave reads four half4 operands with
        // ds_read_b64; the two NIter fragments below preserve that exact
        // repeat order so RunStage can concatenate them into WG::BWarpTensor.
        const index_t lane7 = lane & 7;
        const index_t lane8 = lane7 >> 2;
        const index_t wave2 = wave_id >> 1;
        const index_t common_b_element =
            ((lane7 & 3) << 4) + (lane8 << 3) +
            (lane8 << 9) + (((lane >> 3) & 1) << 6) +
            ((lane >> 4) << 2) + ((wave2 & 1) << 8) +
            ((wave2 >> 1) << 10);

        const auto* b_legacy_base =
            reinterpret_cast<const char*>(
                b_lds_window.get_bottom_tensor_view()
                    .get_buffer_view()
                    .p_data_);
        const auto* combined_base = b_legacy_base - 0x8000;
        const index_t b_load_base =
            reinterpret_cast<uintptr_t>(combined_base) + 0x2000 +
            KIter * 0x4000 + common_b_element * 2;
        const bool b_wrap =
            (lane == 47 || lane == 63) && ((wave2 & 1) == 1);
        const index_t b_alt_base =
            b_load_base - (b_wrap ? 0x400 : 0);

        thread_buffer<typename Problem::BDataType, 8> raw;
        auto& quads = raw.template get_as<fp32x2_t>();
        if constexpr(NIter == 0)
        {
            asm volatile("ds_read_b64 %0, %2 offset:0\n\t"
                         "ds_read_b64 %1, %3 offset:256"
                         : "=v"(quads[0]), "=v"(quads[1])
                         : "v"(b_load_base), "v"(b_alt_base)
                         : "memory");
        }
        else
        {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_DEFER_PERMUTE) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_PIPELINED_WAIT4) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
            asm volatile("ds_read_b64 %0, %2 offset:4096\n\t"
                         "ds_read_b64 %1, %3 offset:4352"
                         : "=v"(quads[0]), "=v"(quads[1])
                         : "v"(b_load_base), "v"(b_alt_base)
                         : "memory");
#else
            asm volatile("ds_read_b64 %0, %2 offset:4096\n\t"
                         "ds_read_b64 %1, %3 offset:4352\n\t"
                         "s_waitcnt lgkmcnt(0)"
                         : "=v"(quads[0]), "=v"(quads[1])
                         : "v"(b_load_base), "v"(b_alt_base)
                         : "memory");
#endif
        }
#if !defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_OPERAND) && \
    !defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_DEFER_PERMUTE) && \
    !defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_PIPELINED_WAIT4) && \
    !defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE) && \
    !defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED) && \
    !defined(CK_TILE_GROUPED_GEMM_DSREADM_B_VGPR_PREPERMUTE)
        // Generic DSReadM presents each four-half B fragment as [0, 2, 1, 3].
        // Keep that established CK contract unless the isolated raw-operand
        // gate requests the vendor's native asymmetric A/B MMAC ordering.
        const auto b1 = raw[number<1>{}];
        raw[number<1>{}] = raw[number<2>{}];
        raw[number<2>{}] = b1;
        const auto b5 = raw[number<5>{}];
        raw[number<5>{}] = raw[number<6>{}];
        raw[number<6>{}] = b5;
#endif
        return make_static_distributed_tensor<
            typename Problem::BDataType>(
            make_static_tile_distribution(
                typename BDsreadm::WarpStoreDstrEncoding{}),
            raw);
#else
        const index_t lane_b2 = (lane >> 2) & 1;
        const index_t common_element =
            ((lane & 3) << 3) + (lane_b2 << 5) +
            (lane_b2 << 9) + (((lane >> 3) & 1) << 8) +
            ((lane >> 4) << 10);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM)
        // hipBLASLt's physical wave id is n*2+m.
        const index_t wave_id_n = wave_id >> 1;
#else
        // Translate CK's m*4+n wave numbering to the N-wave coordinate used
        // by the reconstructed BLAS LDS layout.
        const index_t wave_id_n = wave_id & 3;
#endif
        const index_t base_byte =
            (common_element + (wave_id_n << 5)) << 1;
        const bool use_alt_base =
            NIter == 1 && (((lane >> 2) & 3) == 3) &&
            (wave_id_n == 3);
        const index_t address_byte =
            0x2000 + KIter * 0x4000 + base_byte +
            NIter * 0x100 - (use_alt_base ? 0x400 : 0);
        const auto* b_legacy_base =
            reinterpret_cast<const char*>(
                b_lds_window.get_bottom_tensor_view()
                    .get_buffer_view()
                    .p_data_);
        const auto* combined_base = b_legacy_base - 0x8000;
        const auto raw = BDsreadm{}(
            reinterpret_cast<const typename Problem::BDataType*>(
                combined_base + address_byte));
        return make_static_distributed_tensor<
            typename Problem::BDataType>(
            make_static_tile_distribution(
                typename BDsreadm::WarpStoreDstrEncoding{}),
            raw.template get_as<typename Problem::BDataType>());
#endif
#else
        const index_t warp_id_n = ck_tile::get_warp_id() % NWarps;
        auto warp_window        = make_tile_window(
            b_lds_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_lds_window.get_window_origin() +
                multi_index<2>{warp_id_n * WG::kN +
                                   NIter * NPerBlockPerIter,
                               KIter * WG::kK},
            make_static_tile_distribution(
                typename BDsreadm::WarpLoadDstrEncoding{}));
        return load_tile_by_dsreadm(BDsreadm{}, warp_window);
#endif
    }

    template <index_t KIter, typename BLdsWindow>
    CK_TILE_DEVICE auto LdsLoadBStage(const BLdsWindow& b_lds_window) const
    {
        static_assert(KIter < KWarpIter, "invalid staged B operand index");
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT)
        statically_indexed_array<
            decltype(LdsLoadBStageFragment<KIter, 0>(b_lds_window)),
            NWarpIter>
            warp_tensors;
        static_for<0, NWarpIter, 1>{}([&](auto n_warp_iter) {
            warp_tensors(n_warp_iter) =
                LdsLoadBStageFragment<KIter, n_warp_iter>(
                    b_lds_window);
        });
        return warp_tensors;
#else
        const index_t warp_id_n = ck_tile::get_warp_id() % NWarps;
        auto warp_window        = make_tile_window(
            b_lds_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_lds_window.get_window_origin() +
                multi_index<2>{warp_id_n * WG::kN, KIter * WG::kK},
            make_static_tile_distribution(typename BDsreadm::WarpLoadDstrEncoding{}));

        statically_indexed_array<
            decltype(load_tile_by_dsreadm(BDsreadm{}, warp_window)),
            NWarpIter>
            warp_tensors;
        static_for<0, NWarpIter, 1>{}([&](auto n_warp_iter) {
            warp_tensors(n_warp_iter) =
                load_tile_by_dsreadm(BDsreadm{}, warp_window);
            if constexpr(n_warp_iter != NWarpIter - 1)
            {
                move_tile_window(warp_window,
                                 {NPerBlockPerIter, 0});
            }
        });
        return warp_tensors;
#endif
    }

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
    template <bool WaitFully = true, typename BWarpTensors>
    CK_TILE_DEVICE auto
    PrepareVendorB64Stage(BWarpTensors b_warp_tensors) const
    {
        // The following stage's four B64 reads were intentionally left
        // outstanding across the previous 32-MMAC interval. Retire them and
        // restore CK's half4 contract before issuing this stage's A reads and
        // the next B look-ahead. This keeps all eight v_perm producers ahead
        // of lgkmcnt(4), rather than letting LLVM split them through MMAC.
        if constexpr(WaitFully)
        {
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        }
        static_for<0, NWarpIter, 1>{}([&](auto n_warp_iter) {
            auto& b_fragment =
                b_warp_tensors[n_warp_iter].get_thread_buffer();
            const auto b1 = b_fragment[number<1>{}];
            b_fragment[number<1>{}] = b_fragment[number<2>{}];
            b_fragment[number<2>{}] = b1;
            const auto b5 = b_fragment[number<5>{}];
            b_fragment[number<5>{}] = b_fragment[number<6>{}];
            b_fragment[number<6>{}] = b5;
            // HCU clang cannot tie a StaticBuffer aggregate to a VGPR asm
            // output. Read the four permuted dwords into direct values and
            // make them inputs to a compiler barrier instead. This still
            // forces every permutation producer above the following LDS
            // issue while leaving the buffer values unchanged.
            const auto b_ready_1 = b_fragment[number<1>{}];
            const auto b_ready_2 = b_fragment[number<2>{}];
            const auto b_ready_5 = b_fragment[number<5>{}];
            const auto b_ready_6 = b_fragment[number<6>{}];
            asm volatile(""
                         :
                         : "v"(b_ready_1),
                           "v"(b_ready_2),
                           "v"(b_ready_5),
                           "v"(b_ready_6)
                         : "memory");
        });
        return b_warp_tensors;
    }
#endif

    template <index_t OperandBuffer = 0,
              typename CBlockTensor,
              typename AWarpTensors,
              typename BWarpTensors>
    CK_TILE_DEVICE void RunStage(CBlockTensor& c_block_tensor,
                                 const AWarpTensors& a_warp_tensors,
                                 const BWarpTensors& b_warp_tensors) const
    {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_WAVE_GEMM)
        // The selected hipBLASLt solution issues one MIWT8_4 warp GEMM per
        // K16 stage. The existing DSReadM fragments already contain exactly
        // eight A operand pairs and four B operand pairs; concatenate them in
        // MMAC repeat order and update the 128 raw accumulators in one call.
        typename WG::AWarpTensor a_warp_tensor;
        typename WG::BWarpTensor b_warp_tensor;
        typename WG::CWarpTensor c_warp_tensor;

        using AFragment =
            remove_cvref_t<decltype(a_warp_tensors[number<0>{}])>;
        using BFragment =
            remove_cvref_t<decltype(b_warp_tensors[number<0>{}])>;
        static_assert(AFragment::get_thread_buffer_size() * MWarpIter ==
                          WG::AWarpTensor::get_thread_buffer_size(),
                      "BLAS wave GEMM A fragment packing mismatch");
        static_assert(BFragment::get_thread_buffer_size() * NWarpIter ==
                          WG::BWarpTensor::get_thread_buffer_size(),
                      "BLAS wave GEMM B fragment packing mismatch");
        static_assert(
            remove_cvref_t<CBlockTensor>::get_thread_buffer_size() ==
                WG::CWarpTensor::get_thread_buffer_size(),
            "BLAS wave GEMM accumulator packing mismatch");

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_AB_MMAC) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_B_MMAC)
        // The exact lgkm/vmem/barrier boundary is emitted in the following
        // single 32-MMAC asm region. Keeping it there prevents LLVM from
        // moving the first consumer above the block rendezvous.
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_BUILTIN_WAIT4)
        // Encode lgkmcnt(4) through LLVM's waitcnt builtin so the scheduler
        // updates its LDS scoreboard. The inline-asm form emits the same ISA
        // instruction but LLVM then redundantly inserts lgkmcnt(3/2/1/0)
        // before the first four A consumers.
        __builtin_amdgcn_s_waitcnt(0xc47f);
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED)
        // Two complete A/B operand stages stay live. Retire the older eight
        // LDS reads and preserve the following eight exactly like the selected
        // hipBLASLt grad-X schedule. Once both the A/B operands and the full
        // MMAC region use fixed registers, the single asm dependency boundary
        // is sufficient to stop LLVM from reintroducing progressive waits;
        // retain the vendor's literal lgkmcnt(8) instead of allowing the
        // waitcnt pass to strengthen it to lgkmcnt(4).
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_AB_REGS) && \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_C_ASM)
        // The exact BLAS-style operand pipeline retires LDS before its VMEM
        // wait and barrier at the direct-load synchronization point below.
#else
        __builtin_amdgcn_s_waitcnt(0xc87f);
#endif
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_PIPELINED_WAIT4) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
        // B-only look-ahead leaves exactly four ds_read_b64 operations for the
        // following K16 stage behind the current four A reads. Retire the
        // current A/B operands while preserving those four independent B
        // operations across the 32-MMAC consumer interval.
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_DEFER_PERMUTE)
        // Inline-asm output dependencies prevent register use from moving
        // above ds_read_b64, but they do not satisfy the LDS scoreboard.
        // Wait at the consumption boundary so the full operand pipeline can
        // overlap B reads with intervening global/LDS work without exposing
        // partially completed fragments to the compatibility permutation.
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#endif
        static_for<0, MWarpIter, 1>{}([&](auto m_warp_iter) {
            static_for<0, AFragment::get_thread_buffer_size(), 1>{}(
                [&](auto i) {
                    constexpr index_t dst =
                        decltype(m_warp_iter)::value *
                            AFragment::get_thread_buffer_size() +
                        decltype(i)::value;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_A_CONSUMER_HALF4_REMAP)
                    // Vendor B64 exposes contiguous K half4 order, whereas
                    // generic A DSReadM exposes [0,2,1,3]. Remap A only at the
                    // compile-time packing boundary so both raw operands use
                    // the same K order without changing the global producer.
                    constexpr index_t src =
                        (decltype(i)::value % 4 == 1)
                            ? decltype(i)::value + 1
                            : ((decltype(i)::value % 4 == 2)
                                   ? decltype(i)::value - 1
                                   : decltype(i)::value);
#else
                    constexpr index_t src = decltype(i)::value;
#endif
                    a_warp_tensor.get_thread_buffer()[number<dst>{}] =
                        a_warp_tensors[m_warp_iter]
                            .get_thread_buffer()[number<src>{}];
                });
        });
        static_for<0, NWarpIter, 1>{}([&](auto n_warp_iter) {
            static_for<0, BFragment::get_thread_buffer_size(), 1>{}(
                [&](auto i) {
                    constexpr index_t dst =
                        decltype(n_warp_iter)::value *
                            BFragment::get_thread_buffer_size() +
                        decltype(i)::value;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_DEFER_PERMUTE) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_PIPELINED_WAIT4)
                    // Keep ds_read_b64 results untouched while they are in
                    // flight, then restore CK's established [0,2,1,3]
                    // operand contract only at the MMAC packing boundary.
                    constexpr index_t src =
                        (decltype(i)::value % 4 == 1)
                            ? decltype(i)::value + 1
                            : ((decltype(i)::value % 4 == 2)
                                   ? decltype(i)::value - 1
                                   : decltype(i)::value);
#else
                    constexpr index_t src = decltype(i)::value;
#endif
                    b_warp_tensor.get_thread_buffer()[number<dst>{}] =
                        b_warp_tensors[n_warp_iter]
                            .get_thread_buffer()[number<src>{}];
                });
        });
        static_for<0, WG::CWarpTensor::get_thread_buffer_size(), 1>{}(
            [&](auto i) {
                c_warp_tensor.get_thread_buffer()[i] =
                    c_block_tensor.get_thread_buffer()[i];
            });

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_AB_REGS)
        static_assert(OperandBuffer == 0 || OperandBuffer == 1,
                      "BLAS fixed A/B register stage requires buffer 0 or 1");
        using WarpGemmAttribute = typename WG::WarpGemmAttribute;
        using WarpGemmImpl      = typename WarpGemmAttribute::Impl;
        auto& c_vecs =
            c_warp_tensor.get_thread_buffer()
                .template get_as<typename WarpGemmImpl::CVecType>();
        auto a_vecs =
            a_warp_tensor.get_thread_buffer()
                .template get_as<typename WarpGemmImpl::AVecType>();
        auto b_vecs =
            b_warp_tensor.get_thread_buffer()
                .template get_as<typename WarpGemmImpl::BVecType>();
        static_assert(decltype(a_vecs)::size() == 8 &&
                          decltype(b_vecs)::size() == 4,
                      "BLAS fixed A/B registers require MIWT8_4 operands");
        if constexpr(OperandBuffer == 0)
        {
            asm volatile(""
                         : "+{v[160:161]}"(b_vecs(number<0>{})),
                           "+{v[162:163]}"(b_vecs(number<1>{})),
                           "+{v[164:165]}"(b_vecs(number<2>{})),
                           "+{v[166:167]}"(b_vecs(number<3>{})),
                           "+{v[128:129]}"(a_vecs(number<0>{})),
                           "+{v[130:131]}"(a_vecs(number<1>{})),
                           "+{v[132:133]}"(a_vecs(number<2>{})),
                           "+{v[134:135]}"(a_vecs(number<3>{})),
                           "+{v[136:137]}"(a_vecs(number<4>{})),
                           "+{v[138:139]}"(a_vecs(number<5>{})),
                           "+{v[140:141]}"(a_vecs(number<6>{})),
                           "+{v[142:143]}"(a_vecs(number<7>{}))
                         :
                         : "memory");
        }
        else
        {
            asm volatile(""
                         : "+{v[168:169]}"(b_vecs(number<0>{})),
                           "+{v[170:171]}"(b_vecs(number<1>{})),
                           "+{v[172:173]}"(b_vecs(number<2>{})),
                           "+{v[174:175]}"(b_vecs(number<3>{})),
                           "+{v[144:145]}"(a_vecs(number<0>{})),
                           "+{v[146:147]}"(a_vecs(number<1>{})),
                           "+{v[148:149]}"(a_vecs(number<2>{})),
                           "+{v[150:151]}"(a_vecs(number<3>{})),
                           "+{v[152:153]}"(a_vecs(number<4>{})),
                           "+{v[154:155]}"(a_vecs(number<5>{})),
                           "+{v[156:157]}"(a_vecs(number<6>{})),
                           "+{v[158:159]}"(a_vecs(number<7>{}))
                         :
                         : "memory");
        }
        WarpGemmAttribute{}(c_vecs, a_vecs, b_vecs);
#else
        WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
#endif

        static_for<0, WG::CWarpTensor::get_thread_buffer_size(), 1>{}(
            [&](auto i) {
                c_block_tensor.get_thread_buffer()[i] =
                    c_warp_tensor.get_thread_buffer()[i];
            });
#else
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_TRAVERSAL_SWAP)
        using SFC = space_filling_curve<sequence<MWarpIter, NWarpIter>,
                                        sequence<1, 0>,
                                        sequence<1, 1>>;
#else
        using SFC = space_filling_curve<sequence<MWarpIter, NWarpIter>,
                                        sequence<0, 1>,
                                        sequence<1, 1>>;
#endif
        constexpr auto num_access = SFC::get_num_of_access();
        static_for<0, num_access, 1>{}([&](auto access_id) {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_MMAC_TRAVERSAL_REVERSE)
            constexpr auto mapped_access_id =
                number<num_access - 1 - decltype(access_id)::value>{};
#else
            constexpr auto mapped_access_id = access_id;
#endif
            constexpr auto idx         = SFC::get_index(mapped_access_id);
            constexpr auto m_warp_iter = idx.at(number<0>{});
            constexpr auto n_warp_iter = idx.at(number<1>{});
            using CWarpDstr             = typename WG::CWarpDstr;
            using CWarpTensor           = typename WG::CWarpTensor;
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros =
                uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            CWarpTensor c_warp_tensor;
            c_warp_tensor.get_thread_buffer() =
                c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                    c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
            WG{}(c_warp_tensor,
                 a_warp_tensors[m_warp_iter],
                 b_warp_tensors[n_warp_iter]);
            c_block_tensor.set_y_sliced_thread_data(
                merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                c_warp_tensor.get_thread_buffer());
        });
#endif
    }

    // Convert the raw MMAC accumulator distribution to the logical [M, N]
    // output distribution entirely in registers.  This gives the grouped-GEMM
    // epilogue a zero-LDS path and avoids the CShuffle transpose/barrier tail.
    template <typename CBlockTensor>
    CK_TILE_DEVICE static constexpr auto
    MakeCOutputBlockTile(const CBlockTensor& c_block_tensor)
    {
        using CDataType = remove_cvref_t<typename Problem::CDataType>;

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MWarpIter, MWarps>,
                  sequence<NWarpIter, NWarps>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_output_dstr_encode =
            detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding,
                typename WG::CWarpOutputDstrEncoding{});
        constexpr auto c_block_output_dstr =
            make_static_tile_distribution(c_block_output_dstr_encode);
        auto c_block_output_tensor =
            make_static_distributed_tensor<CDataType>(c_block_output_dstr);

        using CWarpDstr         = typename WG::CWarpDstr;
        using CWarpOutputDstr   = typename WG::CWarpOutputDstr;
        using CWarpTensor       = typename WG::CWarpTensor;
        using CWarpOutputTensor = typename WG::CWarpOutputTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros =
            uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_output_y_lengths =
            to_sequence(
                CWarpOutputDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_output_y_index_zeros =
            uniform_sequence_gen_t<CWarpOutputDstr::NDimY, 0>{};

        static_for<0, MWarpIter, 1>{}([&](auto m_warp_iter) {
            static_for<0, NWarpIter, 1>{}([&](auto n_warp_iter) {
                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() =
                    c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(
                            sequence<m_warp_iter, n_warp_iter>{},
                            c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{},
                                        c_warp_y_lengths));

                const CWarpOutputTensor c_warp_output_tensor =
                    WG{}.MakeCOutputLayout(c_warp_tensor);
                c_block_output_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                    c_warp_output_y_index_zeros),
                    merge_sequences(sequence<1, 1>{},
                                    c_warp_output_y_lengths),
                    c_warp_output_tensor.get_thread_buffer());
            });
        });

        return c_block_output_tensor;
    }
};

struct GemmPipelineAgBgCrCompV3DsreadmPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV3DsreadmPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return GemmDsreadmBlockGemm<Problem>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t M = Problem::BlockGemmShape::kM;
        constexpr index_t K = Problem::BlockGemmShape::kK;
        constexpr auto packed_k_m = make_naive_tensor_descriptor_packed(
            make_tuple(number<K>{}, number<M>{}), number<8>{});
        return transform_tensor_descriptor(
            packed_k_m,
            make_tuple(make_pass_through_transform(number<K>{}),
                       make_pass_through_transform(number<M>{})),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t N = Problem::BlockGemmShape::kN;
        constexpr index_t K = Problem::BlockGemmShape::kK;
        constexpr auto packed_k_n = make_naive_tensor_descriptor_packed(
            make_tuple(number<K>{}, number<N>{}), number<8>{});
        return transform_tensor_descriptor(
            packed_k_n,
            make_tuple(make_pass_through_transform(number<K>{}),
                       make_pass_through_transform(number<N>{})),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr auto desc = MakeALdsBlockDescriptor<Problem>();
        return integer_least_multiple(
            sizeof(typename Problem::ADataType) * desc.get_element_space_size(), 16);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr auto desc = MakeBLdsBlockDescriptor<Problem>();
        return integer_least_multiple(
            sizeof(typename Problem::BDataType) * desc.get_element_space_size(), 16);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
        // The BLAS-compatible physical producer keeps four 16-KiB
        // [A8KiB|B8KiB] slots even though the logical M128 A tile uses only
        // the first 4 KiB of each A half-slot.
        return 65536;
#else
        return GetSmemSizeA<Problem>() + GetSmemSizeB<Problem>();
#endif
    }
};

// Compute-V3 scheduling with the convolution dsreadm block GEMM as the LDS
// consumer. The production V3/V4 classes are left unchanged.
template <typename Problem, bool StageOperandOverlap = false>
struct GemmPipelineAgBgCrCompV3Dsreadm : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using Policy = GemmPipelineAgBgCrCompV3DsreadmPolicy;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using CLayout        = remove_cvref_t<typename Problem::CLayout>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using BlockGemm = std::conditional_t<
        StageOperandOverlap,
        GemmDsreadmStageBlockGemm<Problem>,
        remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>>;
    using DistributionBlockGemm = remove_cvref_t<
        decltype(GemmPipelineAgBgCrCompV4DefaultPolicy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;
    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    using Base::PrefetchStages;
    using Base::UsePersistentKernel;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }

    static constexpr index_t GetVectorSizeC()
    {
        return Policy::template GetVectorSizeC<Problem>();
    }

    static constexpr index_t GetSmemPackA()
    {
        return Policy::template GetSmemPackA<Problem>();
    }

    static constexpr index_t GetSmemPackB()
    {
        return Policy::template GetSmemPackB<Problem>();
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        return StageOperandOverlap ? "COMPUTE_V3_DSREADM_STAGE"
                                   : "COMPUTE_V3_DSREADM";
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_',
                      "pipeline_AgBgCrCompV3Dsreadm",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      Problem::kBlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK),
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename CBlockTile, typename ALdsWindow, typename BLdsWindow>
    CK_TILE_DEVICE void RunOperandStages(CBlockTile& c_block_tile,
                                         const ALdsWindow& a_lds_gemm_window,
                                         const BLdsWindow& b_lds_gemm_window) const
    {
        static_assert(StageOperandOverlap,
                      "staged operands are only valid for the isolated pipeline");
        static_assert(BlockGemm::KWarpIter == 4,
                      "staged DSReadM currently requires a K64/K16 block");
        const auto block_gemm = BlockGemm{};

        auto a_even =
            block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
        auto b_even =
            block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
        auto a_odd =
            block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
        auto b_odd =
            block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        block_gemm.RunStage(c_block_tile, a_even, b_even);
        a_even = block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
        b_even = block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        block_gemm.RunStage(c_block_tile, a_odd, b_odd);
        a_odd = block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
        b_odd = block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        block_gemm.RunStage(c_block_tile, a_even, b_even);
        block_gemm.RunStage(c_block_tile, a_odd, b_odd);
    }

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN)
    // hipBLASLt's gfx936 BF16 TN kernel writes each K16 x 256 operand stage
    // directly from the global buffer into LDS with one dwordx4 per thread.
    // The existing DSReadM descriptors are physically packed as [K, M] and
    // [K, N], so tid = k * 32 + free8 maps exactly to
    //   element_offset = k * 256 + free8.
    //
    // This isolated correctness/resource gate deliberately refills a complete
    // K64 tile only after the current tile has been consumed. Once validated,
    // the same producer can be moved into the per-K16 MMAC schedule without
    // changing its coordinate or LDS contracts.
    template <bool HasHotLoop,
              TailNumber TailNum,
              bool PackA = false,
              typename ADramBlockWindow,
              typename BDramBlockWindow>
    CK_TILE_DEVICE auto RunStageOverlapDirectTn(
        const ADramBlockWindow& a_dram_block_window,
        const BDramBlockWindow& b_dram_block_window,
        index_t num_loop,
        void* p_smem,
        ADataType* packed_a_ptr = nullptr,
        index_t packed_a_stride = 0,
        index_t packed_a_stage_group = 0) const
    {
        static_assert(StageOperandOverlap);
        static_assert(!Problem::DoubleSmemBuffer,
                      "direct TN DSReadM requires one LDS buffer");
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS)
        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor> &&
                          (std::is_same_v<BLayout, tensor_layout::gemm::RowMajor> ||
                           std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>),
                      "direct DSReadM requires C,R or gated C,C operands");
#else
        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor> &&
                          std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>,
                      "direct TN DSReadM is restricted to the grad-W C,R layout");
#endif
        static_assert(sizeof(ADataType) == 2 && sizeof(BDataType) == 2,
                      "direct TN DSReadM requires 16-bit operands");
        static_assert(
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
                          MPerBlock == 128 &&
#else
                          MPerBlock == 256 &&
#endif
                          NPerBlock == 256 &&
                          KPerBlock == 64 && BlockSize == 512,
                      "direct TN DSReadM mapping requires the selected tile/512");

        auto&& [a_lds_block, b_lds_block] =
            PipelineImplBase{}.GetABLdsTensorViews(p_smem);
        constexpr auto a_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeBBlockDistributionEncode());
        auto a_windows =
            PipelineImplBase{}.GetAWindows(a_dram_block_window,
                                           a_lds_block,
                                           a_lds_dstr);
        auto b_windows =
            PipelineImplBase{}.GetBWindows(b_dram_block_window,
                                           b_lds_block,
                                           b_lds_dstr);
        auto& a_lds_gemm_window = a_windows.get(number<2>{});
        auto& b_lds_gemm_window = b_windows.get(number<2>{});

        auto a_dram_window = a_dram_block_window;
        auto b_dram_window = b_dram_block_window;

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        [[maybe_unused]] constexpr ADramTileWindowStep a_step =
            make_array(KPerBlock, 0);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS)
        [[maybe_unused]] constexpr BDramTileWindowStep b_step =
            std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>
                ? make_array(KPerBlock, 0)
                : make_array(0, KPerBlock);
#else
        [[maybe_unused]] constexpr BDramTileWindowStep b_step =
            make_array(KPerBlock, 0);
#endif
        auto* a_lds_ptr = a_lds_block.get_buffer_view().p_data_;
        [[maybe_unused]] auto* b_lds_ptr =
            b_lds_block.get_buffer_view().p_data_;

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR)
        // Reproduce the vendor four-slot shadow schedule at the K16
        // granularity. Slots 0/1/2 are primed before the loop. While MMAC
        // consumes slot s, the producer fills (s + 3) % 4, keeping at most
        // four newer VMEM operations outstanding. The barrier is block-wide:
        // it makes an overwritten slot safe even when waves do not finish
        // MMAC at exactly the same cycle.
        const auto a_view   = a_dram_window.get_bottom_tensor_view();
        const auto b_view   = b_dram_window.get_bottom_tensor_view();
        const auto a_origin = a_dram_window.get_window_origin();
        const auto b_origin = b_dram_window.get_window_origin();
        const auto& a_desc  = a_view.get_tensor_descriptor();
        const auto& b_desc  = b_view.get_tensor_descriptor();
        const auto& a_buf   = a_view.get_buffer_view();
        const auto& b_buf   = b_view.get_buffer_view();
        [[maybe_unused]] const auto a_resource =
            make_wave_buffer_resource(a_buf.p_data_,
                                      a_buf.buffer_size_ * sizeof(ADataType));
        [[maybe_unused]] const auto b_resource =
            make_wave_buffer_resource(b_buf.p_data_,
                                      b_buf.buffer_size_ * sizeof(BDataType));

        const index_t tid   = get_thread_id();
        const index_t k16   = tid >> 5;
        const index_t free8 = (tid & 31) * 8;
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
        // Preserve the established DTL thread-to-LDS scatter. The M128 tile
        // uses the low 16 M vectors from every K16 cohort; compacting those
        // 256 loads into tid[0:255] changes the physical LDS permutation.
        const index_t a_k16   = k16;
        const index_t a_free8 = free8;
#else
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_A_K16_PREPERMUTE)
        // Pair the native DSReadM A register order with raw ds_read_b64 B.
        // Swap K bits 0/1 once in the producer address instead of permuting
        // eight B vectors in every consumer wave and every K16 stage.
        const index_t a_k16 =
            (k16 & ~3) | ((k16 & 1) << 1) | ((k16 & 2) >> 1);
#else
        const index_t a_k16 = k16;
#endif
        const index_t a_free8 = free8;
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE)
        // Match the vendor address split more closely: keep the lane-varying
        // K16/free8 offset fixed, and move each uniform K16 stage displacement
        // into the scalar buffer resource. Rebuilding the resource is an
        // intentional compile probe; make_wave_buffer_resource scalarizes all
        // four dwords, so the hot loop should contain no per-slot VALU
        // recurrence for global addresses.
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT)
        // The BLAS ELF packs each K16 stage as [A 8 KiB | B 8 KiB],
        // advances stages by 16 KiB, and sets m0[18] for odd waves. That
        // high bit selects the paired-wave LDS swizzle used by its
        // conflict-free ds_read_m32x16_b16 addresses.
        const uint32_t wave_pair_m0 =
            (__builtin_amdgcn_readfirstlane(tid >> 6) & 1u) << 18;
        const index_t a_lds_tid =
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_A_LDS_K16_PREPERMUTE)
            (tid & 31) |
            (((k16 & ~3) | ((k16 & 1) << 1) | ((k16 & 2) >> 1))
             << 5);
#else
            tid;
#endif
        const auto a_m0_base =
            __builtin_amdgcn_readfirstlane(
                reinterpret_cast<uintptr_t>(a_lds_ptr +
                                            a_lds_tid * 8)) |
            wave_pair_m0;
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS)
        constexpr bool is_b_k_contiguous =
            std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>;
        const auto b_m0_base = [&]() {
            if constexpr(is_b_k_contiguous)
            {
                // The custom hipBLASLt Ailk_Bljk solution uses DTL bit 16
                // for a K-contiguous B vector. Lane pairs own K[0:7]/K[8:15]
                // for one N coordinate; the hardware scatters that dwordx4
                // into the DSReadM B layout without a VGPR transpose.
                const index_t b_lds_lane_element =
                    (tid >> 1) * 16 + (tid & 1) * 8;
                const uint32_t b_dtl_m0 =
                    (__builtin_amdgcn_readfirstlane(tid >> 6) & 1u) << 16;
                return __builtin_amdgcn_readfirstlane(
                           reinterpret_cast<uintptr_t>(
                               a_lds_ptr + 0x2000 / sizeof(BDataType) +
                               b_lds_lane_element)) |
                       b_dtl_m0;
            }
            else
            {
                return __builtin_amdgcn_readfirstlane(
                           reinterpret_cast<uintptr_t>(
                               a_lds_ptr + 0x2000 / sizeof(BDataType) +
                               tid * 8)) |
                       wave_pair_m0;
            }
        }();
#else
        const auto b_m0_base =
            __builtin_amdgcn_readfirstlane(
                reinterpret_cast<uintptr_t>(
                    a_lds_ptr + 0x2000 / sizeof(BDataType) +
                    tid * 8)) |
            wave_pair_m0;
#endif
#else
        const auto a_m0_base = __builtin_amdgcn_readfirstlane(
            reinterpret_cast<uintptr_t>(a_lds_ptr + tid * 8));
        const auto b_m0_base = __builtin_amdgcn_readfirstlane(
            reinterpret_cast<uintptr_t>(b_lds_ptr + tid * 8));
#endif
        const index_t a_element_offset0 =
            a_desc.calculate_offset(
                make_multi_index(a_origin[0] + a_k16,
                                 a_origin[1] + a_free8));
        const index_t b_element_offset0 =
            b_desc.calculate_offset(
                make_multi_index(b_origin[0] + k16,
                                 b_origin[1] + free8));
        const index_t a_element_offset1 =
            a_desc.calculate_offset(
                make_multi_index(a_origin[0] + 16 + a_k16,
                                 a_origin[1] + a_free8));
        const index_t b_element_offset1 =
            b_desc.calculate_offset(
                make_multi_index(b_origin[0] + 16 + k16,
                                 b_origin[1] + free8));
        const index_t a_voffset =
            a_element_offset0 * sizeof(ADataType);
        const index_t b_voffset =
            b_element_offset0 * sizeof(BDataType);
        const index_t a_stage_byte_stride =
            (a_element_offset1 - a_element_offset0) * sizeof(ADataType);
        const index_t b_stage_byte_stride =
            (b_element_offset1 - b_element_offset0) * sizeof(BDataType);
        const index_t a_buffer_bytes =
            a_buf.buffer_size_ * sizeof(ADataType);
        const index_t b_buffer_bytes =
            b_buf.buffer_size_ * sizeof(BDataType);

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS)
        const index_t lane       = tid & 63;
        const index_t wave       = tid >> 6;
        const index_t half_lane  = lane >> 1;
        const index_t lane_part8 = half_lane & 7;
        const index_t b_free =
            (half_lane >> 3) * 16 + (lane_part8 & 3) +
            (lane_part8 >> 2) * 8 + (wave & 1) * 4 +
            (wave >> 1) * 64;
        const index_t b_k8 = (lane & 1) * 8;
        const index_t b_stride_elements =
            b_desc.calculate_offset(
                make_multi_index(b_origin[0] + 1, b_origin[1])) -
            b_desc.calculate_offset(
                make_multi_index(b_origin[0], b_origin[1]));
#if !defined(CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_B_VGPR_PREPERMUTE) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_B_VGPR_PREPERMUTE_OVERLAP)
        const auto b_k_contiguous_resource =
            make_wave_buffer_resource(
                b_buf.p_data_,
                b_buffer_bytes,
                b_stride_elements * sizeof(BDataType));
#endif
#endif

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE)
        auto a_stage_resource_running =
            make_wave_buffer_resource(a_buf.p_data_, a_buffer_bytes);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS)
        auto b_stage_k_contiguous_resource_running =
            make_wave_buffer_resource(
                reinterpret_cast<const char*>(b_buf.p_data_) +
                    b_origin[1] * sizeof(BDataType),
                b_buffer_bytes,
                b_stride_elements * sizeof(BDataType));
#endif
        const auto advance_scalar_resource =
            [](int32x4_t& resource, index_t byte_stride) {
                asm volatile("s_add_u32 %0, %0, %2\n\t"
                             "s_addc_u32 %1, %1, 0"
                             : "+s"(resource.x), "+s"(resource.y)
                             : "s"(byte_stride));
            };
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_POINTER_RECURRENCE)
        auto a_stage_pointer_running =
            reinterpret_cast<const char*>(a_buf.p_data_);
        index_t a_stage_bytes_remaining = a_buffer_bytes;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE)
        const auto a_stage_resource_fixed =
            make_wave_buffer_resource(a_buf.p_data_, a_buffer_bytes);
        index_t a_stage_voffset_running = a_voffset;
#endif

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
        const auto direct_load_stage_impl =
            [&](auto iSlot, index_t global_stage, auto tail_mask) {
            constexpr bool apply_tail_mask =
                remove_cvref_t<decltype(tail_mask)>::value;
#else
        const auto direct_load_stage = [&](auto iSlot, index_t global_stage) {
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0)
            // Register variables cannot be captured by a C++ lambda. Declare
            // them in this fully inlined stage body so the scalar bases remain
            // explicit inputs to the two DTL instructions.
            register uint32_t a_m0_base_reg asm("s92") = a_m0_base;
            [[maybe_unused]] register uint32_t b_m0_base_reg asm("s93") =
                b_m0_base;
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT)
            constexpr index_t slot_byte_offset = iSlot * 0x4000;
#else
            constexpr index_t slot_byte_offset =
                iSlot * 16 * MPerBlock * sizeof(ADataType);
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
            const index_t logical_k_a = a_desc.get_length(number<0>{});
            const index_t logical_k_b = b_desc.get_length(number<0>{});
            const index_t stage_k     = global_stage * 16;
            [[maybe_unused]] const bool a_stage_has_any =
                a_origin[0] + stage_k < logical_k_a;
            const bool b_stage_has_any = b_origin[0] + stage_k < logical_k_b;
            // Do not materialize an SRD base beyond a compact allocation for
            // the fully padded K16 stages in the final K64 tile. Zero-fill
            // skips those VMEM operations; vendor-OOB issues only voffset -1.
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
            const index_t b_stage_base =
                !apply_tail_mask || b_stage_has_any
                    ? global_stage * b_stage_byte_stride
                    : 0;
#else
            const index_t b_stage_base =
                b_stage_has_any ? global_stage * b_stage_byte_stride : 0;
#endif
#else
            const index_t b_stage_base = global_stage * b_stage_byte_stride;
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE)
            const auto& a_stage_resource = a_stage_resource_running;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_POINTER_RECURRENCE)
            const auto a_stage_resource = make_wave_buffer_resource(
                a_stage_pointer_running, a_stage_bytes_remaining);
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE)
            const auto& a_stage_resource = a_stage_resource_fixed;
#else
            const index_t a_stage_base =
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
                (!apply_tail_mask || a_stage_has_any)
                    ? global_stage * a_stage_byte_stride
                    : 0;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB)
                a_stage_has_any ? global_stage * a_stage_byte_stride : 0;
#else
                global_stage * a_stage_byte_stride;
#endif
            const auto a_stage_resource = make_wave_buffer_resource(
                reinterpret_cast<const char*>(a_buf.p_data_) + a_stage_base,
                a_buffer_bytes - a_stage_base);
#endif
            const auto b_stage_resource = make_wave_buffer_resource(
                reinterpret_cast<const char*>(b_buf.p_data_) + b_stage_base,
                b_buffer_bytes - b_stage_base);

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
            const bool a_logical_k_valid =
                a_origin[0] + stage_k + a_k16 < logical_k_a;
            const bool b_logical_k_valid =
                b_origin[0] + stage_k + k16 < logical_k_b;
            [[maybe_unused]] const bool whole_stage_valid =
                a_origin[0] + stage_k + 15 < logical_k_a &&
                b_origin[0] + stage_k + 15 < logical_k_b;
#endif

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL)
            // Hygon direct-to-LDS buffer loads can fault instead of returning
            // zero when a lane uses an out-of-range SRD offset. Zero the whole
            // physical slot before a partial/empty K16 stage, synchronize the
            // zero fill, and let only logically valid lanes issue VMEM. Valid
            // DTL writes then overlay the exact established BLAS LDS mapping.
            if(!whole_stage_valid)
            {
                const int32x4_t zero128 = {0, 0, 0, 0};
                const index_t lane_byte = tid * 8 * sizeof(ADataType);
                const index_t a_zero_address =
                    reinterpret_cast<uintptr_t>(a_lds_ptr) +
                    slot_byte_offset + lane_byte;
                const index_t b_zero_address =
                    reinterpret_cast<uintptr_t>(a_lds_ptr) + 0x2000 +
                    slot_byte_offset + lane_byte;
                asm volatile("ds_write_b128 %0, %1"
                             :
                             : "v"(a_zero_address), "v"(zero128)
                             : "memory");
                asm volatile("ds_write_b128 %0, %1"
                             :
                             : "v"(b_zero_address), "v"(zero128)
                             : "memory");
                block_sync_lds();
            }
#endif

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB)
            // Keep the original aligned-stage instruction stream on every
            // full K16 stage. Only the final partial/empty stages pay the
            // lane-varying compare/cndmask cost required by the vendor tail
            // convention. `whole_stage_valid` is wave-uniform, so the hot
            // path avoids introducing divergent EXEC around DTL.
            const index_t a_current_voffset =
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE)
                a_stage_voffset_running;
#else
                a_voffset;
#endif
            index_t a_issue_voffset = a_current_voffset;
            index_t b_issue_voffset = b_voffset;
            if(!whole_stage_valid)
            {
                a_issue_voffset =
                    a_logical_k_valid ? a_current_voffset : index_t{-1};
                b_issue_voffset =
                    b_logical_k_valid ? b_voffset : index_t{-1};
            }
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
            // This compile-time branch is instantiated twice. The full-stage
            // instance preserves the original aligned offset path, while the
            // peeled final-K64 instance applies the vendor -1 convention.
            const index_t a_current_voffset =
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE)
                a_stage_voffset_running;
#else
                a_voffset;
#endif
            const index_t a_issue_voffset = [&]() {
                if constexpr(apply_tail_mask)
                {
                    return a_logical_k_valid ? a_current_voffset
                                             : index_t{-1};
                }
                else
                {
                    return a_current_voffset;
                }
            }();
            const index_t b_issue_voffset = [&]() {
                if constexpr(apply_tail_mask)
                {
                    return b_logical_k_valid ? b_voffset : index_t{-1};
                }
                else
                {
                    return b_voffset;
                }
            }();
#endif

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0)
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL)
            if(a_logical_k_valid)
            {
#endif
            asm volatile(
                "s_add_u32 m0, %0, %3\n\t"
                "buffer_load_dwordx4 %1, %2, 0 offen offset:0 lds"
                :
                : "{s92}"(a_m0_base_reg),
                  "v"(
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
                      a_issue_voffset
#else
                      a_voffset
#endif
                  ),
                  "s"(a_stage_resource),
                  "n"(slot_byte_offset)
                : "memory");
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL)
            }
#endif
#else
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
            if((tid & 31) < 16)
            {
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL)
                if(a_logical_k_valid)
                {
#endif
                    hcu_async_buffer_load_asm_impl<ADataType, 8>(
                    a_m0_base + slot_byte_offset,
                    a_stage_resource,
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE)
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
                    a_issue_voffset,
#else
                    a_stage_voffset_running,
#endif
#else
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
                    a_issue_voffset,
#else
                    a_voffset,
#endif
#endif
                    0);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL)
                }
#endif
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
            }
#endif
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS)
            if constexpr(is_b_k_contiguous)
            {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_B_VGPR_PREPERMUTE)
                // Load one K8 slice per thread, apply the MMAC K-lane
                // permutation once at the producer, then write the exact DTL
                // physical layout. Eight consumer waves otherwise repeat this
                // permutation for every K16 stage.
                auto b_payload =
                    hcu_struct_buffer_load_asm_impl<BDataType, 8>(
                        b_k_contiguous_resource,
                        b_origin[0] + b_free,
                        (b_origin[1] + global_stage * 16 + b_k8) *
                            sizeof(BDataType),
                        0);
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                const auto b1 = b_payload[number<1>{}];
                b_payload[number<1>{}] = b_payload[number<2>{}];
                b_payload[number<2>{}] = b1;
                const auto b5 = b_payload[number<5>{}];
                b_payload[number<5>{}] = b_payload[number<6>{}];
                b_payload[number<6>{}] = b5;

                const index_t b_group64 = b_free >> 6;
                const index_t b_q       = (b_free >> 2) & 15;
                const index_t b_r       = b_free & 3;
                const index_t b_column_base =
                    b_group64 * 1024 + (b_q >> 1) * 64 + b_r * 16 +
                    (b_q & 1) * 520;
                const bool b_wrap =
                    ((b_free & 63) == 63) && (b_k8 == 8);
                const index_t b_physical_element =
                    b_column_base + b_k8 - (b_wrap ? 512 : 0);
                const index_t b_lds_byte_address =
                    reinterpret_cast<uintptr_t>(a_lds_ptr) + 0x2000 +
                    slot_byte_offset +
                    b_physical_element * sizeof(BDataType);
                const auto b_vec =
                    b_payload.template get_as<fp32x4_t>()[number<0>{}];
                asm volatile("ds_write_b128 %0, %1"
                             :
                             : "v"(b_lds_byte_address), "v"(b_vec)
                             : "memory");
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#else
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_B_SCALAR_STAGE_RESOURCE)
                // Match the hipBLASLt grad-X address split: advance the
                // strided buffer base uniformly for each K16 stage, while
                // keeping both lane-varying operands (N index and K[0/8])
                // fixed in VGPRs.
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE)
                const auto& b_stage_k_contiguous_resource =
                    b_stage_k_contiguous_resource_running;
#else
                const index_t b_stage_byte =
                    (b_origin[1] + global_stage * 16) *
                    sizeof(BDataType);
                const auto b_stage_k_contiguous_resource =
                    make_wave_buffer_resource(
                        reinterpret_cast<const char*>(b_buf.p_data_) +
                            b_stage_byte,
                        b_buffer_bytes,
                        b_stride_elements * sizeof(BDataType));
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_M0)
                const int32x2_t b_offsets = {
                    static_cast<int32_t>(b_origin[0] + b_free),
                    static_cast<int32_t>(b_k8 * sizeof(BDataType))};
                asm volatile(
                    "s_add_u32 m0, %0, %3\n\t"
                    "buffer_load_dwordx4 %1, %2, 0 idxen offen "
                    "offset:0 lds"
                    :
                    : "{s93}"(b_m0_base_reg),
                      "v"(b_offsets),
                      "s"(b_stage_k_contiguous_resource),
                      "n"(slot_byte_offset)
                    : "memory");
#else
                hcu_async_struct_buffer_load_asm_impl<BDataType, 8>(
                    b_m0_base + slot_byte_offset,
                    b_stage_k_contiguous_resource,
                    b_origin[0] + b_free,
                     b_k8 * sizeof(BDataType),
                     0);
#endif
#else
                hcu_async_struct_buffer_load_asm_impl<BDataType, 8>(
                    b_m0_base + slot_byte_offset,
                    b_k_contiguous_resource,
                    b_origin[0] + b_free,
                    (b_origin[1] + global_stage * 16 + b_k8) *
                        sizeof(BDataType),
                     0);
#endif
#endif
            }
            else
#endif
            {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL)
                if(b_logical_k_valid)
                {
#endif
                    hcu_async_buffer_load_asm_impl<BDataType, 8>(
                        b_m0_base + slot_byte_offset,
                        b_stage_resource,
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
                        b_issue_voffset,
#else
                        b_voffset,
#endif
                        0);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_ZERO_FILL)
                }
#endif
            }
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_RESOURCE_RECURRENCE)
            advance_scalar_resource(
                a_stage_resource_running, a_stage_byte_stride);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS)
            if constexpr(is_b_k_contiguous)
            {
                advance_scalar_resource(
                    b_stage_k_contiguous_resource_running,
                    16 * sizeof(BDataType));
            }
#endif
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_POINTER_RECURRENCE)
            a_stage_pointer_running += a_stage_byte_stride;
            a_stage_bytes_remaining -= a_stage_byte_stride;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_SCALAR_VOFFSET_RECURRENCE)
            a_stage_voffset_running += a_stage_byte_stride;
#endif
        };
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
        const auto direct_load_stage = [&](auto iSlot, index_t global_stage) {
            direct_load_stage_impl(
                iSlot, global_stage, bool_constant<false>{});
        };
        const auto direct_load_tail_stage =
            [&](auto iSlot, index_t global_stage) {
                direct_load_stage_impl(
                    iSlot, global_stage, bool_constant<true>{});
            };
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_B_VGPR_PREPERMUTE_OVERLAP)
        // Split the producer-side B transpose into issue/commit halves.  The
        // coalesced global load is issued before the current 32-MMAC stage and
        // remains live in four VGPRs while that stage executes.  Commit waits
        // for VMEM only after the MMAC shadow, performs the two K-lane swaps,
        // and writes the exact DTL-compatible physical LDS layout.
        const auto issue_overlap_stage =
            [&](auto iSlot, index_t global_stage) {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT)
                constexpr index_t slot_byte_offset = iSlot * 0x4000;
#else
                constexpr index_t slot_byte_offset =
                    iSlot * 16 * MPerBlock * sizeof(ADataType);
#endif
                const index_t a_stage_base =
                    global_stage * a_stage_byte_stride;
                const auto a_stage_resource = make_wave_buffer_resource(
                    reinterpret_cast<const char*>(a_buf.p_data_) +
                        a_stage_base,
                    a_buffer_bytes - a_stage_base);
                hcu_async_buffer_load_asm_impl<ADataType, 8>(
                    a_m0_base + slot_byte_offset,
                    a_stage_resource,
                    a_voffset,
                    0);

                return hcu_struct_buffer_load_asm_impl<BDataType, 8>(
                    b_k_contiguous_resource,
                    b_origin[0] + b_free,
                    (b_origin[1] + global_stage * 16 + b_k8) *
                        sizeof(BDataType),
                    0);
            };

        const auto commit_overlap_stage =
            [&](auto iSlot, auto b_payload) {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_LDS_LAYOUT)
                constexpr index_t slot_byte_offset = iSlot * 0x4000;
#else
                constexpr index_t slot_byte_offset =
                    iSlot * 16 * MPerBlock * sizeof(ADataType);
#endif
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                const auto b1 = b_payload[number<1>{}];
                b_payload[number<1>{}] = b_payload[number<2>{}];
                b_payload[number<2>{}] = b1;
                const auto b5 = b_payload[number<5>{}];
                b_payload[number<5>{}] = b_payload[number<6>{}];
                b_payload[number<6>{}] = b5;

                const index_t b_group64 = b_free >> 6;
                const index_t b_q       = (b_free >> 2) & 15;
                const index_t b_r       = b_free & 3;
                const index_t b_column_base =
                    b_group64 * 1024 + (b_q >> 1) * 64 + b_r * 16 +
                    (b_q & 1) * 520;
                const bool b_wrap =
                    ((b_free & 63) == 63) && (b_k8 == 8);
                const index_t b_physical_element =
                    b_column_base + b_k8 - (b_wrap ? 512 : 0);
                const index_t b_lds_byte_address =
                    reinterpret_cast<uintptr_t>(a_lds_ptr) + 0x2000 +
                    slot_byte_offset +
                    b_physical_element * sizeof(BDataType);
                const auto b_vec =
                    b_payload.template get_as<fp32x4_t>()[number<0>{}];
                asm volatile("ds_write_b128 %0, %1"
                             :
                             : "v"(b_lds_byte_address), "v"(b_vec)
                             : "memory");
                asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                             "s_barrier"
                             :
                             :
                             : "memory");
            };
#endif
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_M0_ROLLING)
        // The BLAS kernel keeps the per-wave LDS base in SGPRs and advances
        // only its scalar buffer resources. Preserve that important part of
        // the ISA contract here: scalarize the lane-zero LDS destination once,
        // then use a single rolling global byte offset for each operand.
        //
        // The global resources remain unchanged in this isolated candidate.
        // Rolling two voffsets still removes the compiler's eight independent
        // per-slot address recurrences, while keeping the change layout-safe.
        const auto a_m0_base = __builtin_amdgcn_readfirstlane(
            reinterpret_cast<uintptr_t>(a_lds_ptr + tid * 8));
        const auto b_m0_base = __builtin_amdgcn_readfirstlane(
            reinterpret_cast<uintptr_t>(b_lds_ptr + tid * 8));

        const index_t a_element_offset0 =
            a_desc.calculate_offset(
                make_multi_index(a_origin[0] + k16,
                                 a_origin[1] + free8));
        const index_t b_element_offset0 =
            b_desc.calculate_offset(
                make_multi_index(b_origin[0] + k16,
                                 b_origin[1] + free8));
        const index_t a_element_offset1 =
            a_desc.calculate_offset(
                make_multi_index(a_origin[0] + 16 + k16,
                                 a_origin[1] + free8));
        const index_t b_element_offset1 =
            b_desc.calculate_offset(
                make_multi_index(b_origin[0] + 16 + k16,
                                 b_origin[1] + free8));

        index_t a_byte_offset = a_element_offset0 * sizeof(ADataType);
        index_t b_byte_offset = b_element_offset0 * sizeof(BDataType);
        const index_t a_stage_byte_stride =
            (a_element_offset1 - a_element_offset0) * sizeof(ADataType);
        const index_t b_stage_byte_stride =
            (b_element_offset1 - b_element_offset0) * sizeof(BDataType);

        const auto direct_load_stage = [&](auto iSlot, index_t) {
            constexpr index_t slot_byte_offset =
                iSlot * 16 * MPerBlock * sizeof(ADataType);
            hcu_async_buffer_load_asm_impl<ADataType, 8>(
                a_m0_base + slot_byte_offset,
                a_resource,
                a_byte_offset,
                0);
            hcu_async_buffer_load_asm_impl<BDataType, 8>(
                b_m0_base + slot_byte_offset,
                b_resource,
                b_byte_offset,
                0);
            a_byte_offset += a_stage_byte_stride;
            b_byte_offset += b_stage_byte_stride;
        };
#else
        const auto direct_load_stage = [&](auto iSlot, index_t global_stage) {
            const index_t k = global_stage * 16 + k16;
            const index_t a_element_offset =
                a_desc.calculate_offset(
                    make_multi_index(a_origin[0] + k,
                                     a_origin[1] + free8));
            const index_t b_element_offset =
                b_desc.calculate_offset(
                    make_multi_index(b_origin[0] + k,
                                     b_origin[1] + free8));
            const index_t lds_element =
                iSlot * 16 * MPerBlock + tid * 8;

            hcu_async_buffer_load_asm<ADataType, 8, false>(
                a_lds_ptr + lds_element,
                a_resource,
                a_element_offset * sizeof(ADataType),
                0,
                true,
                bool_constant<false>{});
            hcu_async_buffer_load_asm<BDataType, 8, false>(
                b_lds_ptr + lds_element,
                b_resource,
                b_element_offset * sizeof(BDataType),
                0,
                true,
                bool_constant<false>{});
        };
#endif

        const auto block_gemm = BlockGemm{};

#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_PREPASS)
        if constexpr(PackA)
        {
            // Keep the transpose temporaries disjoint from the 128 FP32
            // accumulators. Each output-N workgroup owns one group of sixteen
            // K16 stages, so this prepass writes every packed element once.
            const index_t first_pack_stage = packed_a_stage_group * 16;
            const index_t last_pack_stage =
                min(first_pack_stage + 16, packed_a_stride / 16);
            for(index_t global_stage = first_pack_stage;
                global_stage < last_pack_stage;
                ++global_stage)
            {
                const index_t a_stage_base =
                    global_stage * a_stage_byte_stride;
                const auto a_stage_resource = make_wave_buffer_resource(
                    reinterpret_cast<const char*>(a_buf.p_data_) +
                        a_stage_base,
                    a_buffer_bytes - a_stage_base);
                hcu_async_buffer_load_asm_impl<ADataType, 8>(
                    a_m0_base, a_stage_resource, a_voffset, 0);
                block_sync_lds_direct_load();

                // The penultimate K16 stage is repaired by the established
                // scalar tail path in GroupedGemmKernel.
                if(global_stage * 16 != packed_a_stride - 32)
                {
                    const auto a_pack_stage =
                        block_gemm.template LdsLoadAStage<0>(
                            a_lds_gemm_window);
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_WAVE_SGPR)
                    const index_t wave =
                        __builtin_amdgcn_readfirstlane(tid >> 6);
#else
                    const index_t wave = tid >> 6;
#endif
                    const index_t wave_m = wave & 1;
                    const index_t wave_n = wave >> 1;
                    const index_t lane   = tid & 63;
                    const index_t lane_m = lane & 15;
                    const index_t lane_k = lane >> 4;
                    const index_t k_base =
                        a_origin[0] + global_stage * 16 + lane_k * 4;

                    static_for<0, BlockGemm::MWarpIter, 1>{}(
                        [&](auto m_warp_iter) {
                            if(wave_n == decltype(m_warp_iter)::value)
                            {
                                const auto& raw =
                                    a_pack_stage[m_warp_iter]
                                        .get_thread_buffer();
                                static_for<0, 2, 1>{}([&](auto m_repeat) {
                                    constexpr index_t src_base =
                                        decltype(m_repeat)::value * 4;
                                    thread_buffer<ADataType, 4> packed4;
                                    packed4(number<0>{}) =
                                        raw[number<src_base + 0>{}];
                                    packed4(number<1>{}) =
                                        raw[number<src_base + 2>{}];
                                    packed4(number<2>{}) =
                                        raw[number<src_base + 1>{}];
                                    packed4(number<3>{}) =
                                        raw[number<src_base + 3>{}];
                                    const index_t m =
                                        a_origin[1] + wave_m * 32 +
                                        decltype(m_warp_iter)::value * 64 +
                                        decltype(m_repeat)::value * 16 +
                                        lane_m;
                                    auto* dst =
                                        reinterpret_cast<fp32x2_t*>(
                                            packed_a_ptr +
                                            m * packed_a_stride +
                                            k_base);
                                    *dst =
                                        packed4.template get_as<fp32x2_t>()[
                                            number<0>{}];
                                });
                            }
                        });
                }
                block_sync_lds_direct_load();
            }
        }
#endif

        auto c_block_tile = block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        direct_load_stage(number<0>{}, 0);
        direct_load_stage(number<1>{}, 1);
        direct_load_stage(number<2>{}, 2);
        block_sync_lds_direct_load();

        index_t next_global_stage = 3;
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DSREADM)
        index_t packed_global_stage = 0;
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_FROM_B_TRANSPOSED_C) && \
    defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_BUFFER_STORE)
#if !defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_LATE_BUFFER_RESOURCE)
        const auto packed_a_resource =
            make_wave_buffer_resource(packed_a_ptr);
#endif
#endif
        const auto store_packed_a_stage =
            [&](const auto& operand_stage,
                index_t global_stage,
                index_t pack_half) {
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_FROM_B_TRANSPOSED_C)
                if constexpr(PackA)
                {
                    if(global_stage / 16 != packed_a_stage_group)
                    {
                        return;
                    }
                    if(global_stage * 16 == packed_a_stride - 32)
                    {
                        return;
                    }
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_BUFFER_STORE) && \
    defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_LATE_BUFFER_RESOURCE)
                    // Keep the packed-output SRD out of the long-lived
                    // direct-load/MMAC state. Only the workgroup and K-stage
                    // that actually owns packed output materializes it.
                    const auto packed_a_resource =
                        make_wave_buffer_resource(packed_a_ptr);
#endif
                    const index_t wave   = tid >> 6;
                    const index_t wave_m = wave & 1;
                    const index_t wave_n = wave >> 1;
                    const index_t lane   = tid & 63;
                    const index_t lane_n = lane & 15;
                    const index_t lane_k = lane >> 4;
                    const index_t k_base =
                        b_origin[0] + global_stage * 16 + lane_k * 4;

                    // In the transposed-C grad-W view, B is the original
                    // row-major grad_out[K,N]. The two staged B fragments
                    // cover the full N256 tile. Emit B^T[N,K] directly into
                    // the packed allocation. Physical wave numbering is
                    // n*2+m. Use that paired M-wave bit to select one of the
                    // two N fragments, so all eight waves own disjoint rows.
                    static_for<0, BlockGemm::NWarpIter, 1>{}(
                        [&](auto n_warp_iter) {
                            if(wave_m ==
                               decltype(n_warp_iter)::value)
                            {
                                const auto& raw =
                                    operand_stage[n_warp_iter]
                                        .get_thread_buffer();
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DWORDX4_BATCHED_REPEATS)
                                (void)pack_half;
                                thread_buffer<BDataType, 4> packed4_0;
                                thread_buffer<BDataType, 4> packed4_1;
                                packed4_0(number<0>{}) =
                                    raw[number<0>{}];
                                packed4_0(number<1>{}) =
                                    raw[number<2>{}];
                                packed4_0(number<2>{}) =
                                    raw[number<1>{}];
                                packed4_0(number<3>{}) =
                                    raw[number<3>{}];
                                packed4_1(number<0>{}) =
                                    raw[number<4>{}];
                                packed4_1(number<1>{}) =
                                    raw[number<6>{}];
                                packed4_1(number<2>{}) =
                                    raw[number<5>{}];
                                packed4_1(number<3>{}) =
                                    raw[number<7>{}];
                                const auto packed_local_0 =
                                    packed4_0
                                        .template get_as<int32x2_t>()[
                                            number<0>{}];
                                const auto packed_local_1 =
                                    packed4_1
                                        .template get_as<int32x2_t>()[
                                            number<0>{}];
                                const auto packed_remote_0 =
                                    warp_shuffle(
                                        packed_local_0, lane ^ 16);
                                const auto packed_remote_1 =
                                    warp_shuffle(
                                        packed_local_1, lane ^ 16);
                                if((lane_k & 1) == 0)
                                {
                                    const int32x4_t packed8_0 = {
                                        packed_local_0[0],
                                        packed_local_0[1],
                                        packed_remote_0[0],
                                        packed_remote_0[1]};
                                    const int32x4_t packed8_1 = {
                                        packed_local_1[0],
                                        packed_local_1[1],
                                        packed_remote_1[0],
                                        packed_remote_1[1]};
                                    const index_t n_0 =
                                        b_origin[1] + wave_n * 32 +
                                        decltype(n_warp_iter)::value * 128 +
                                        lane_n;
                                    const index_t byte_offset_0 =
                                        (n_0 * packed_a_stride + k_base) *
                                        sizeof(ADataType);
                                    const index_t byte_offset_1 =
                                        byte_offset_0 +
                                        16 * packed_a_stride *
                                            sizeof(ADataType);
                                    llvm_amdgcn_raw_buffer_store_i32x4(
                                        packed8_0,
                                        packed_a_resource,
                                        byte_offset_0,
                                        0,
                                        0);
                                    llvm_amdgcn_raw_buffer_store_i32x4(
                                        packed8_1,
                                        packed_a_resource,
                                        byte_offset_1,
                                        0,
                                        0);
                                }
#else
                                static_for<0, 2, 1>{}([&](auto n_repeat) {
                                    if(pack_half >= 0 &&
                                       pack_half !=
                                           decltype(n_repeat)::value)
                                    {
                                        return;
                                    }
                                    constexpr index_t src_base =
                                        decltype(n_repeat)::value * 4;
                                    thread_buffer<BDataType, 4> packed4;
                                    packed4(number<0>{}) =
                                        raw[number<src_base + 0>{}];
                                    packed4(number<1>{}) =
                                        raw[number<src_base + 2>{}];
                                    packed4(number<2>{}) =
                                        raw[number<src_base + 1>{}];
                                    packed4(number<3>{}) =
                                        raw[number<src_base + 3>{}];
                                    const index_t n =
                                        b_origin[1] + wave_n * 32 +
                                        decltype(n_warp_iter)::value * 128 +
                                        decltype(n_repeat)::value * 16 +
                                        lane_n;
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_BUFFER_STORE)
                                    const index_t byte_offset =
                                        (n * packed_a_stride + k_base) *
                                        sizeof(ADataType);
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DWORDX4_LANE_PAIR)
                                    const auto packed_local =
                                        packed4
                                            .template get_as<int32x2_t>()[
                                                number<0>{}];
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DS_SWIZZLE_XOR16)
                                    const int32x2_t packed_remote = {
                                        __builtin_amdgcn_ds_swizzle(
                                            packed_local[0], 0x401f),
                                        __builtin_amdgcn_ds_swizzle(
                                            packed_local[1], 0x401f)};
#else
                                    const auto packed_remote =
                                        warp_shuffle(
                                            packed_local, lane ^ 16);
#endif
                                    if((lane_k & 1) == 0)
                                    {
                                        const int32x4_t packed8 = {
                                            packed_local[0],
                                            packed_local[1],
                                            packed_remote[0],
                                            packed_remote[1]};
                                        llvm_amdgcn_raw_buffer_store_i32x4(
                                            packed8,
                                            packed_a_resource,
                                            byte_offset,
                                            0,
                                            0);
                                    }
#else
                                    llvm_amdgcn_raw_buffer_store_i32x2(
                                        packed4
                                            .template get_as<int32x2_t>()[
                                                number<0>{}],
                                        packed_a_resource,
                                        byte_offset,
                                        0,
                                        0);
#endif
#else
                                    auto* dst =
                                        reinterpret_cast<fp32x2_t*>(
                                            packed_a_ptr +
                                            n * packed_a_stride + k_base);
                                    *dst =
                                        packed4.template get_as<fp32x2_t>()[
                                            number<0>{}];
#endif
                                });
#endif
                            }
                        });
                }
                else
                {
                    (void)operand_stage;
                    (void)global_stage;
                }
#else
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_PREPASS)
                (void)operand_stage;
                (void)global_stage;
#else
                if constexpr(PackA)
                {
                    if(global_stage / 16 != packed_a_stage_group)
                    {
                        return;
                    }
                    if(global_stage * 16 == packed_a_stride - 32)
                    {
                        return;
                    }
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_WAVE_SGPR)
                    // wave_m/wave_n are uniform within a wave.  Materialize
                    // them in SGPRs so the static MWarpIter selector lowers
                    // to scalar control flow instead of repeated
                    // v_cmp/saveexec regions around every packed store.
                    const index_t wave =
                        __builtin_amdgcn_readfirstlane(tid >> 6);
#else
                    const index_t wave = tid >> 6;
#endif
                    const index_t wave_m = wave & 1;
                    const index_t wave_n = wave >> 1;
                    const index_t lane   = tid & 63;
                    const index_t lane_m = lane & 15;
                    const index_t lane_k = lane >> 4;
                    const index_t k_base =
                        a_origin[0] + global_stage * 16 + lane_k * 4;

#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_SINGLE_N_WAVE)
                    // The A operand is identical across the four N waves.
                    // One N wave can therefore emit all four M-repeat slices
                    // without changing the packed tensor mapping or bytes.
                    // This replaces the four-way per-store selector that
                    // drove the fused kernel into VGPR/SGPR spills.
                    if(wave_n != 0)
                    {
                        return;
                    }
#endif
                    static_for<0, BlockGemm::MWarpIter, 1>{}(
                        [&](auto m_warp_iter) {
#if !defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_SINGLE_N_WAVE)
                            if(wave_n == decltype(m_warp_iter)::value)
#endif
                            {
                                const auto& raw =
                                    operand_stage[m_warp_iter]
                                        .get_thread_buffer();
                                static_for<0, 2, 1>{}([&](auto m_repeat) {
                                    if(pack_half >= 0 &&
                                       pack_half !=
                                           decltype(m_repeat)::value)
                                    {
                                        return;
                                    }
                                    constexpr index_t src_base =
                                        decltype(m_repeat)::value * 4;
                                    thread_buffer<ADataType, 4> packed4;
                                    packed4(number<0>{}) =
                                        raw[number<src_base + 0>{}];
                                    packed4(number<1>{}) =
                                        raw[number<src_base + 2>{}];
                                    packed4(number<2>{}) =
                                        raw[number<src_base + 1>{}];
                                    packed4(number<3>{}) =
                                        raw[number<src_base + 3>{}];
                                    const index_t m =
                                        a_origin[1] + wave_m * 32 +
                                        decltype(m_warp_iter)::value * 64 +
                                        decltype(m_repeat)::value * 16 +
                                        lane_m;
                                    auto* dst =
                                        reinterpret_cast<fp32x2_t*>(
                                            packed_a_ptr +
                                            m * packed_a_stride +
                                            k_base);
                                    *dst =
                                        packed4.template get_as<fp32x2_t>()[
                                            number<0>{}];
                                });
                            }
                        });
                }
                else
                {
                    (void)operand_stage;
                    (void)global_stage;
                }
#endif
#endif
            };
        const auto store_packed_operand_stage =
            [&](const auto& a_stage,
                const auto& b_stage,
                index_t global_stage,
                index_t pack_half = -1) {
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_FROM_B_TRANSPOSED_C)
                (void)a_stage;
                store_packed_a_stage(
                    b_stage, global_stage, pack_half);
#else
                (void)b_stage;
                store_packed_a_stage(
                    a_stage, global_stage, pack_half);
#endif
            };
#else
        (void)packed_a_ptr;
        (void)packed_a_stride;
        (void)packed_a_stage_group;
#endif
        const auto run_stage_with_optional_pack =
            [&](const auto& a_stage,
                const auto& b_stage) {
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DSREADM) && \
    defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_SPLIT_MMAC_STORE)
                store_packed_operand_stage(
                    a_stage, b_stage, packed_global_stage, 0);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
                store_packed_operand_stage(
                    a_stage, b_stage, packed_global_stage++, 1);
#elif defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DSREADM) && \
    defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_PRE_MMAC_STORE)
                store_packed_operand_stage(
                    a_stage, b_stage, packed_global_stage++);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
#elif defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DSREADM)
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
                store_packed_operand_stage(
                    a_stage, b_stage, packed_global_stage++);
#else
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
#endif
            };
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_B_VGPR_PREPERMUTE_OVERLAP)
        if constexpr(is_b_k_contiguous)
        {
            for(index_t i = 0; i < num_loop - 1; ++i)
            {
                {
                    auto a_stage =
                        block_gemm.template LdsLoadAStage<0>(
                            a_lds_gemm_window);
                    auto b_stage =
                        block_gemm.template LdsLoadBStage<0>(
                            b_lds_gemm_window);
                    auto b_payload =
                        issue_overlap_stage(number<3>{},
                                            next_global_stage++);
                    block_gemm.RunStage(c_block_tile, a_stage, b_stage);
                    commit_overlap_stage(number<3>{}, b_payload);
                }
                {
                    auto a_stage =
                        block_gemm.template LdsLoadAStage<1>(
                            a_lds_gemm_window);
                    auto b_stage =
                        block_gemm.template LdsLoadBStage<1>(
                            b_lds_gemm_window);
                    auto b_payload =
                        issue_overlap_stage(number<0>{},
                                            next_global_stage++);
                    block_gemm.RunStage(c_block_tile, a_stage, b_stage);
                    commit_overlap_stage(number<0>{}, b_payload);
                }
                {
                    auto a_stage =
                        block_gemm.template LdsLoadAStage<2>(
                            a_lds_gemm_window);
                    auto b_stage =
                        block_gemm.template LdsLoadBStage<2>(
                            b_lds_gemm_window);
                    auto b_payload =
                        issue_overlap_stage(number<1>{},
                                            next_global_stage++);
                    block_gemm.RunStage(c_block_tile, a_stage, b_stage);
                    commit_overlap_stage(number<1>{}, b_payload);
                }
                {
                    auto a_stage =
                        block_gemm.template LdsLoadAStage<3>(
                            a_lds_gemm_window);
                    auto b_stage =
                        block_gemm.template LdsLoadBStage<3>(
                            b_lds_gemm_window);
                    auto b_payload =
                        issue_overlap_stage(number<2>{},
                                            next_global_stage++);
                    block_gemm.RunStage(c_block_tile, a_stage, b_stage);
                    commit_overlap_stage(number<2>{}, b_payload);
                }
            }

            direct_load_stage(number<3>{}, next_global_stage);
            block_sync_lds_direct_load();
            RunOperandStages(c_block_tile,
                             a_lds_gemm_window,
                             b_lds_gemm_window);
            return c_block_tile;
        }

        // The same translation unit still instantiates non-C,C,C dispatch
        // lanes. Keep their established circular schedule so this narrowly
        // gated experiment does not impose the K-contiguous B contract on
        // unrelated layouts.
        for(index_t i = 0; i < num_loop - 1; ++i)
        {
            {
                auto a_stage =
                    block_gemm.template LdsLoadAStage<0>(
                        a_lds_gemm_window);
                auto b_stage =
                    block_gemm.template LdsLoadBStage<0>(
                        b_lds_gemm_window);
                direct_load_stage(number<3>{}, next_global_stage++);
                block_sync_load_raw(4);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
            }
            {
                auto a_stage =
                    block_gemm.template LdsLoadAStage<1>(
                        a_lds_gemm_window);
                auto b_stage =
                    block_gemm.template LdsLoadBStage<1>(
                        b_lds_gemm_window);
                direct_load_stage(number<0>{}, next_global_stage++);
                block_sync_load_raw(4);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
            }
            {
                auto a_stage =
                    block_gemm.template LdsLoadAStage<2>(
                        a_lds_gemm_window);
                auto b_stage =
                    block_gemm.template LdsLoadBStage<2>(
                        b_lds_gemm_window);
                direct_load_stage(number<1>{}, next_global_stage++);
                block_sync_load_raw(4);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
            }
            {
                auto a_stage =
                    block_gemm.template LdsLoadAStage<3>(
                        a_lds_gemm_window);
                auto b_stage =
                    block_gemm.template LdsLoadBStage<3>(
                        b_lds_gemm_window);
                direct_load_stage(number<2>{}, next_global_stage++);
                block_sync_load_raw(4);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
            }
        }

        direct_load_stage(number<3>{}, next_global_stage);
        block_sync_lds_direct_load();
        RunOperandStages(c_block_tile,
                         a_lds_gemm_window,
                         b_lds_gemm_window);
        return c_block_tile;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_B_A0_OPERAND_PIPELINED)
        // Keep the B-only winner and add just one look-ahead A fragment. The
        // other three A fragments stay just-in-time, avoiding the live-range
        // expansion of the rejected full A+B pipeline.
        auto b_current =
            block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
        auto a_first_current =
            block_gemm.template LdsLoadAStageFragment<0, 0>(
                a_lds_gemm_window);

        for(index_t i = 0; i < num_loop - 1; ++i)
        {
            auto a_stage =
                block_gemm.template LdsLoadAStageWithFirst<0>(
                    a_lds_gemm_window,
                    a_first_current);
            direct_load_stage(number<3>{}, next_global_stage++);
            block_sync_load_raw(4);
            auto b_next =
                block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
            auto a_first_next =
                block_gemm.template LdsLoadAStageFragment<1, 0>(
                    a_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_stage, b_current);
            b_current       = b_next;
            a_first_current = a_first_next;

            a_stage =
                block_gemm.template LdsLoadAStageWithFirst<1>(
                    a_lds_gemm_window,
                    a_first_current);
            direct_load_stage(number<0>{}, next_global_stage++);
            block_sync_load_raw(4);
            b_next =
                block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
            a_first_next =
                block_gemm.template LdsLoadAStageFragment<2, 0>(
                    a_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_stage, b_current);
            b_current       = b_next;
            a_first_current = a_first_next;

            a_stage =
                block_gemm.template LdsLoadAStageWithFirst<2>(
                    a_lds_gemm_window,
                    a_first_current);
            direct_load_stage(number<1>{}, next_global_stage++);
            block_sync_load_raw(4);
            b_next =
                block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
            a_first_next =
                block_gemm.template LdsLoadAStageFragment<3, 0>(
                    a_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_stage, b_current);
            b_current       = b_next;
            a_first_current = a_first_next;

            a_stage =
                block_gemm.template LdsLoadAStageWithFirst<3>(
                    a_lds_gemm_window,
                    a_first_current);
            direct_load_stage(number<2>{}, next_global_stage++);
            block_sync_load_raw(4);
            b_next =
                block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
            a_first_next =
                block_gemm.template LdsLoadAStageFragment<0, 0>(
                    a_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_stage, b_current);
            b_current       = b_next;
            a_first_current = a_first_next;
        }

        direct_load_stage(number<3>{}, next_global_stage);
        block_sync_lds_direct_load();
        auto a_stage =
            block_gemm.template LdsLoadAStageWithFirst<0>(
                a_lds_gemm_window,
                a_first_current);
        auto b_next =
            block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
        auto a_first_next =
            block_gemm.template LdsLoadAStageFragment<1, 0>(
                a_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_stage, b_current);
        b_current       = b_next;
        a_first_current = a_first_next;
        a_stage =
            block_gemm.template LdsLoadAStageWithFirst<1>(
                a_lds_gemm_window,
                a_first_current);
        b_next =
            block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
        a_first_next =
            block_gemm.template LdsLoadAStageFragment<2, 0>(
                a_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_stage, b_current);
        b_current       = b_next;
        a_first_current = a_first_next;
        a_stage =
            block_gemm.template LdsLoadAStageWithFirst<2>(
                a_lds_gemm_window,
                a_first_current);
        b_next =
            block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
        a_first_next =
            block_gemm.template LdsLoadAStageFragment<3, 0>(
                a_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_stage, b_current);
        a_stage =
            block_gemm.template LdsLoadAStageWithFirst<3>(
                a_lds_gemm_window,
                a_first_next);
        block_gemm.RunStage(c_block_tile, a_stage, b_next);
        return c_block_tile;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_A_OPERAND_PIPELINED)
        // Symmetric ablation of the B-only winner: retain the four A DSReadM
        // fragments for the following K16 stage while B remains just-in-time.
        auto a_current =
            block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);

        for(index_t i = 0; i < num_loop - 1; ++i)
        {
            auto b_stage =
                block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
            direct_load_stage(number<3>{}, next_global_stage++);
            block_sync_load_raw(4);
            auto a_next =
                block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_current, b_stage);
            a_current = a_next;

            b_stage =
                block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
            direct_load_stage(number<0>{}, next_global_stage++);
            block_sync_load_raw(4);
            a_next =
                block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_current, b_stage);
            a_current = a_next;

            b_stage =
                block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
            direct_load_stage(number<1>{}, next_global_stage++);
            block_sync_load_raw(4);
            a_next =
                block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_current, b_stage);
            a_current = a_next;

            b_stage =
                block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
            direct_load_stage(number<2>{}, next_global_stage++);
            block_sync_load_raw(4);
            a_next =
                block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_current, b_stage);
            a_current = a_next;
        }

        direct_load_stage(number<3>{}, next_global_stage);
        block_sync_lds_direct_load();
        auto b_stage =
            block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
        auto a_next =
            block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_current, b_stage);
        a_current = a_next;
        b_stage =
            block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
        a_next =
            block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_current, b_stage);
        a_current = a_next;
        b_stage =
            block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
        a_next =
            block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_current, b_stage);
        b_stage =
            block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_next, b_stage);
        return c_block_tile;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED2)
        // Keep two future raw-B fragments live. The one-stage raw-B pipeline
        // removes the compatibility v_perm work but exposes substantially
        // more LDS wait cycles because each B fragment has only one 32-MMAC
        // interval to mature. A second look-ahead fragment preserves the
        // coalesced producer and compile-time A half4 remap while giving LDS
        // two complete MMAC intervals of independent work.
        auto b_current =
            block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
        auto b_lookahead =
            block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);

        for(index_t i = 0; i < num_loop - 1; ++i)
        {
            auto a_stage =
                block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
            direct_load_stage(number<3>{}, next_global_stage++);
            block_sync_load_raw(4);
            auto b_next =
                block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_stage, b_current);
            b_current   = b_lookahead;
            b_lookahead = b_next;

            a_stage =
                block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
            direct_load_stage(number<0>{}, next_global_stage++);
            block_sync_load_raw(4);
            b_next =
                block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_stage, b_current);
            b_current   = b_lookahead;
            b_lookahead = b_next;

            a_stage =
                block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
            direct_load_stage(number<1>{}, next_global_stage++);
            block_sync_load_raw(4);
            b_next =
                block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_stage, b_current);
            b_current   = b_lookahead;
            b_lookahead = b_next;

            a_stage =
                block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
            direct_load_stage(number<2>{}, next_global_stage++);
            block_sync_load_raw(4);
            b_next =
                block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
            block_gemm.RunStage(c_block_tile, a_stage, b_current);
            b_current   = b_lookahead;
            b_lookahead = b_next;
        }

        direct_load_stage(number<3>{}, next_global_stage);
        block_sync_lds_direct_load();
        auto a_stage =
            block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
        auto b_next =
            block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_stage, b_current);
        b_current   = b_lookahead;
        b_lookahead = b_next;
        a_stage =
            block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
        b_next =
            block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_stage, b_current);
        b_current   = b_lookahead;
        b_lookahead = b_next;
        a_stage =
            block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_stage, b_current);
        b_current = b_lookahead;
        a_stage =
            block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
        block_gemm.RunStage(c_block_tile, a_stage, b_current);
        return c_block_tile;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED)
        // Retain only the two B DSReadM fragments for the following K16
        // stage. A remains just-in-time, limiting the extra live range while
        // giving the scheduler independent LDS work behind current MMACs.
        auto b_current =
            block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
        b_current = block_gemm.PrepareVendorB64Stage(b_current);
#endif
        const auto sync_b_pipeline_direct_load = [&]() {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_B_MMAC)
            // The waitcnt/vmem/barrier boundary is part of the following
            // relaxed full-MMAC asm region.
#else
            block_sync_load_raw(4);
#endif
        };

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DISABLE_MAIN_LOOP_UNROLL)
#pragma clang loop unroll(disable)
#endif
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
        for(index_t i = 0; i < num_loop - 2; ++i)
#else
        for(index_t i = 0; i < num_loop - 1; ++i)
#endif
        {
            auto a_stage =
                block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
            direct_load_stage(number<3>{}, next_global_stage++);
            sync_b_pipeline_direct_load();
            auto b_next =
                block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
            run_stage_with_optional_pack(a_stage, b_current);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
            b_current = block_gemm.PrepareVendorB64Stage(b_next);
#else
            b_current = b_next;
#endif

            a_stage =
                block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
            direct_load_stage(number<0>{}, next_global_stage++);
            sync_b_pipeline_direct_load();
            b_next =
                block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
            run_stage_with_optional_pack(a_stage, b_current);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
            b_current = block_gemm.PrepareVendorB64Stage(b_next);
#else
            b_current = b_next;
#endif

            a_stage =
                block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
            direct_load_stage(number<1>{}, next_global_stage++);
            sync_b_pipeline_direct_load();
            b_next =
                block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
            run_stage_with_optional_pack(a_stage, b_current);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
            b_current = block_gemm.PrepareVendorB64Stage(b_next);
#else
            b_current = b_next;
#endif

            a_stage =
                block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
            direct_load_stage(number<2>{}, next_global_stage++);
            sync_b_pipeline_direct_load();
            b_next =
                block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
            run_stage_with_optional_pack(a_stage, b_current);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
            b_current = block_gemm.PrepareVendorB64Stage(b_next);
#else
            b_current = b_next;
#endif
        }

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_LOGICAL_K_TAIL_VENDOR_OOB_PEELED)
        // Peel the final transition iteration. Its first refill is the last
        // full K16 stage of the penultimate K64 tile; the remaining three
        // refills and the post-loop refill belong to the partial final tile.
        {
            auto a_stage =
                block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
            direct_load_stage(number<3>{}, next_global_stage++);
            sync_b_pipeline_direct_load();
            auto b_next =
                block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
            run_stage_with_optional_pack(a_stage, b_current);
            b_current = b_next;

            a_stage =
                block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
            direct_load_tail_stage(number<0>{}, next_global_stage++);
            sync_b_pipeline_direct_load();
            b_next =
                block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
            run_stage_with_optional_pack(a_stage, b_current);
            b_current = b_next;

            a_stage =
                block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
            direct_load_tail_stage(number<1>{}, next_global_stage++);
            sync_b_pipeline_direct_load();
            b_next =
                block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
            run_stage_with_optional_pack(a_stage, b_current);
            b_current = b_next;

            a_stage =
                block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
            direct_load_tail_stage(number<2>{}, next_global_stage++);
            sync_b_pipeline_direct_load();
            b_next =
                block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
            run_stage_with_optional_pack(a_stage, b_current);
            b_current = b_next;
        }

        direct_load_tail_stage(number<3>{}, next_global_stage);
#else
        direct_load_stage(number<3>{}, next_global_stage);
#endif
        block_sync_lds_direct_load();
        auto a_stage =
            block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
        auto b_next =
            block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
        run_stage_with_optional_pack(a_stage, b_current);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
        b_current = block_gemm.PrepareVendorB64Stage(b_next);
#else
        b_current = b_next;
#endif
        a_stage =
            block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
        b_next =
            block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
        run_stage_with_optional_pack(a_stage, b_current);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
        b_current = block_gemm.PrepareVendorB64Stage(b_next);
#else
        b_current = b_next;
#endif
        a_stage =
            block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
        b_next =
            block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
        run_stage_with_optional_pack(a_stage, b_current);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
        b_next = block_gemm.PrepareVendorB64Stage(b_next);
#endif
        a_stage =
            block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_PIPELINED_WAIT4) || \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_EARLY_PERMUTE_PIPELINED)
        // No B look-ahead remains for the last K16 stage. Drain both the
        // previously issued B fragment and this final A fragment completely.
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#endif
        run_stage_with_optional_pack(a_stage, b_next);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_C_RESTORE)
        // Native vendor B64 order presents logical N repeats as [0, 2, 1, 3].
        // Keep that order throughout the K loop, then exchange the two middle
        // accumulator columns once.  This replaces eight v_perm instructions
        // per K16 stage with eight fp32x4 SSA swaps per output tile.
        static_assert(
            remove_cvref_t<decltype(c_block_tile)>::get_thread_buffer_size() ==
                128,
            "raw-B C restore is specialized for MIWT8_4 fp32 accumulators");
        auto& c_vec =
            c_block_tile.get_thread_buffer().template get_as<fp32x4_t>();
        static_for<0, 8, 1>{}([&](auto iM) {
            constexpr index_t c1 = decltype(iM)::value * 4 + 1;
            constexpr index_t c2 = decltype(iM)::value * 4 + 2;
            const auto        tmp = c_vec[number<c1>{}];
            c_vec(number<c1>{})   = c_vec[number<c2>{}];
            c_vec(number<c2>{})   = tmp;
        });
#endif
        return c_block_tile;
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_OPERAND_PIPELINED)
        // Keep two K16 operand fragments live and issue the following LDS
        // reads while MMAC consumes the current pair. This mirrors the vendor
        // schedule more closely than the one-stage-at-a-time loop below:
        // each steady-state K64 block enters with stages 0/1 already in
        // registers, reads stage 2 behind stage-0 MMAC, reads stage 3 behind
        // stage-1 MMAC, then preloads stages 0/1 of the next K64 block.
        auto a_even =
            block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
        auto b_even =
            block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
        auto a_odd =
            block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
        auto b_odd =
            block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        const auto sync_direct_load_for_current_operands = [&]() {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FUSED_SYNC_AB_MMAC)
            // Fused into BLAS_RELAXED_FULL_MMAC_ASM so the waitcnt pass cannot
            // split the first MMAC across this synchronization boundary.
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED) && \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_AB_REGS) && \
    defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_FIXED_C_ASM)
            // Match the selected hipBLASLt stage boundary exactly: retire
            // only the current eight LDS reads, retain the next eight, then
            // wait for the direct-to-LDS producers and rendezvous all waves.
            asm volatile("s_waitcnt lgkmcnt(8)\n\t"
                         "s_waitcnt vmcnt(4)\n\t"
                         "s_barrier"
                         :
                         :
                         : "memory");
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
            // Retire the older A+B stage while retaining the following eight
            // LDS reads, then synchronize the direct-to-LDS producers.
            __builtin_amdgcn_s_waitcnt(0xc87f);
            block_sync_load_raw(4);
#else
            block_sync_load_raw(4);
#endif
        };

        for(index_t i = 0; i < num_loop - 1; ++i)
        {
            direct_load_stage(number<3>{}, next_global_stage++);
            sync_direct_load_for_current_operands();
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
            b_even = block_gemm.template PrepareVendorB64Stage<false>(b_even);
#endif
            block_gemm.template RunStage<0>(
                c_block_tile, a_even, b_even);
            a_even =
                block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
            b_even =
                block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);

            direct_load_stage(number<0>{}, next_global_stage++);
            sync_direct_load_for_current_operands();
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
            b_odd = block_gemm.template PrepareVendorB64Stage<false>(b_odd);
#endif
            block_gemm.template RunStage<1>(
                c_block_tile, a_odd, b_odd);
            a_odd =
                block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
            b_odd =
                block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);

            direct_load_stage(number<1>{}, next_global_stage++);
            sync_direct_load_for_current_operands();
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
            b_even = block_gemm.template PrepareVendorB64Stage<false>(b_even);
#endif
            block_gemm.template RunStage<0>(
                c_block_tile, a_even, b_even);
            a_even =
                block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
            b_even =
                block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);

            direct_load_stage(number<2>{}, next_global_stage++);
            sync_direct_load_for_current_operands();
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
            b_odd = block_gemm.template PrepareVendorB64Stage<false>(b_odd);
#endif
            block_gemm.template RunStage<1>(
                c_block_tile, a_odd, b_odd);
            a_odd =
                block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
            b_odd =
                block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
        }

        // Finish the last K64 after draining the remaining two direct loads.
        direct_load_stage(number<3>{}, next_global_stage);
        block_sync_lds_direct_load();
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
        b_even = block_gemm.template PrepareVendorB64Stage<false>(b_even);
#endif
        block_gemm.template RunStage<0>(
            c_block_tile, a_even, b_even);
        a_even = block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
        b_even = block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
        b_odd = block_gemm.template PrepareVendorB64Stage<false>(b_odd);
#endif
        block_gemm.template RunStage<1>(
            c_block_tile, a_odd, b_odd);
        a_odd = block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
        b_odd = block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
        __builtin_amdgcn_s_waitcnt(0xc87f);
        b_even = block_gemm.template PrepareVendorB64Stage<false>(b_even);
#endif
        block_gemm.template RunStage<0>(
            c_block_tile, a_even, b_even);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_RAW_AB_PIPELINED)
        // Only the final stage remains; no following operand set can be kept
        // outstanding at its consumption boundary.
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#elif defined(CK_TILE_GROUPED_GEMM_DSREADM_VENDOR_B64_AB_PIPELINED_PERMUTE)
        __builtin_amdgcn_s_waitcnt(0xc07f);
        b_odd = block_gemm.template PrepareVendorB64Stage<false>(b_odd);
#endif
        block_gemm.template RunStage<1>(
            c_block_tile, a_odd, b_odd);
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_BLAS_RAW_C_RESTORE)
        // BLAS accumulates in N-major/M-minor order so its 32 MMAC
        // destinations are consecutive. Restore CK's M-major/N-minor thread
        // buffer contract once, after the K loop, with six in-place cycles.
        static_assert(
            remove_cvref_t<decltype(c_block_tile)>::get_thread_buffer_size() ==
                128,
            "BLAS raw-C restore is specialized for MIWT8_4 fp32 accumulators");
        auto& c_vec =
            c_block_tile.get_thread_buffer().template get_as<fp32x4_t>();
        const auto rotate5 = [&](auto i0, auto i1, auto i2, auto i3, auto i4) {
            const auto tmp = c_vec[i0];
            c_vec[i0]      = c_vec[i1];
            c_vec[i1]      = c_vec[i2];
            c_vec[i2]      = c_vec[i3];
            c_vec[i3]      = c_vec[i4];
            c_vec[i4]      = tmp;
        };
        rotate5(number<1>{},
                number<8>{},
                number<2>{},
                number<16>{},
                number<4>{});
        rotate5(number<3>{},
                number<24>{},
                number<6>{},
                number<17>{},
                number<12>{});
        rotate5(number<5>{},
                number<9>{},
                number<10>{},
                number<18>{},
                number<20>{});
        rotate5(number<7>{},
                number<25>{},
                number<14>{},
                number<19>{},
                number<28>{});
        rotate5(number<11>{},
                number<26>{},
                number<22>{},
                number<21>{},
                number<13>{});
        rotate5(number<15>{},
                number<27>{},
                number<30>{},
                number<23>{},
                number<29>{});
#endif
        return c_block_tile;
#else
        for(index_t i = 0; i < num_loop - 1; ++i)
        {
            {
                auto a_stage =
                    block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
                auto b_stage =
                    block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
                direct_load_stage(number<3>{}, next_global_stage++);
                block_sync_load_raw(4);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
            }
            {
                auto a_stage =
                    block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
                auto b_stage =
                    block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
                direct_load_stage(number<0>{}, next_global_stage++);
                block_sync_load_raw(4);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
            }
            {
                auto a_stage =
                    block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
                auto b_stage =
                    block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
                direct_load_stage(number<1>{}, next_global_stage++);
                block_sync_load_raw(4);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
            }
            {
                auto a_stage =
                    block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
                auto b_stage =
                    block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
                direct_load_stage(number<2>{}, next_global_stage++);
                block_sync_load_raw(4);
                block_gemm.RunStage(c_block_tile, a_stage, b_stage);
            }
        }

        // The main loop intentionally leaves the last K16 stage out of the
        // four-operation shadow window. Complete the final K64 tile with a
        // full wait, which also handles num_loop == 1.
        direct_load_stage(number<3>{}, next_global_stage);
        block_sync_lds_direct_load();
        RunOperandStages(c_block_tile,
                         a_lds_gemm_window,
                         b_lds_gemm_window);
        return c_block_tile;
#endif
#else
        const auto direct_load_full_tile = [&]() {
            const auto a_view   = a_dram_window.get_bottom_tensor_view();
            const auto b_view   = b_dram_window.get_bottom_tensor_view();
            const auto a_origin = a_dram_window.get_window_origin();
            const auto b_origin = b_dram_window.get_window_origin();
            const auto& a_desc  = a_view.get_tensor_descriptor();
            const auto& b_desc  = b_view.get_tensor_descriptor();
            const auto& a_buf   = a_view.get_buffer_view();
            const auto& b_buf   = b_view.get_buffer_view();

            const auto a_resource =
                make_wave_buffer_resource(a_buf.p_data_,
                                          a_buf.buffer_size_ * sizeof(ADataType));
            const auto b_resource =
                make_wave_buffer_resource(b_buf.p_data_,
                                          b_buf.buffer_size_ * sizeof(BDataType));

            const index_t tid   = get_thread_id();
            const index_t k16   = tid >> 5;
            const index_t free8 = (tid & 31) * 8;

            static_for<0, KPerBlock / 16, 1>{}([&](auto iKStage) {
                const index_t k = iKStage * 16 + k16;
                const index_t a_element_offset =
                    a_desc.calculate_offset(
                        make_multi_index(a_origin[0] + k,
                                         a_origin[1] + free8));
                const index_t b_element_offset =
                    b_desc.calculate_offset(
                        make_multi_index(b_origin[0] + k,
                                         b_origin[1] + free8));
                const index_t lds_element =
                    iKStage * 16 * MPerBlock + tid * 8;

                hcu_async_buffer_load_asm<ADataType, 8, false>(
                    a_lds_ptr + lds_element,
                    a_resource,
                    a_element_offset * sizeof(ADataType),
                    0,
                    true,
                    bool_constant<false>{});
                hcu_async_buffer_load_asm<BDataType, 8, false>(
                    b_lds_ptr + lds_element,
                    b_resource,
                    b_element_offset * sizeof(BDataType),
                    0,
                    true,
                    bool_constant<false>{});
            });

            buffer_load_fence(0);
        };

        auto c_block_tile = BlockGemm{}.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        direct_load_full_tile();
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);
        block_sync_lds();

        if constexpr(HasHotLoop)
        {
            index_t i = 0;
            do
            {
                RunOperandStages(c_block_tile,
                                 a_lds_gemm_window,
                                 b_lds_gemm_window);
                block_sync_lds();
                direct_load_full_tile();
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);
                block_sync_lds();
                ++i;
            } while(i < num_loop - (TailNum == TailNumber::Even ? 2 : 1));
        }

        RunOperandStages(c_block_tile,
                         a_lds_gemm_window,
                         b_lds_gemm_window);
        if constexpr(TailNum == TailNumber::Even)
        {
            block_sync_lds();
            direct_load_full_tile();
            block_sync_lds();
            RunOperandStages(c_block_tile,
                             a_lds_gemm_window,
                             b_lds_gemm_window);
        }
        else if constexpr(!HasHotLoop && TailNum == TailNumber::Odd &&
                          Problem::BlockGemmShape::NumWarps == 8)
        {
            // 8-wave num_loop==3: prologue loaded tile 0 and advanced the
            // window to tile 1. The Even-only epilogue would drop tiles 1
            // and 2 (K in [129, 192] when KPerBlock=64).
            block_sync_lds();
            direct_load_full_tile();
            move_tile_window(a_dram_window, a_step);
            move_tile_window(b_dram_window, b_step);
            block_sync_lds();
            RunOperandStages(c_block_tile,
                             a_lds_gemm_window,
                             b_lds_gemm_window);
            block_sync_lds();
            direct_load_full_tile();
            block_sync_lds();
            RunOperandStages(c_block_tile,
                             a_lds_gemm_window,
                             b_lds_gemm_window);
        }
        return c_block_tile;
#endif
    }
#endif

    template <bool HasHotLoop,
              TailNumber TailNum,
              bool PackA = false,
              typename ADramBlockWindow,
              typename BDramBlockWindow>
    CK_TILE_DEVICE auto RunStageOverlap(
        const ADramBlockWindow& a_dram_block_window,
        const BDramBlockWindow& b_dram_block_window,
        index_t num_loop,
        void* p_smem,
        ADataType* packed_a_ptr = nullptr,
        index_t packed_a_stride = 0,
        index_t packed_a_stage_group = 0) const
    {
        static_assert(StageOperandOverlap);
        static_assert(!Problem::DoubleSmemBuffer,
                      "staged Compute V3 DSReadM requires one LDS buffer");

        constexpr bool is_a_col_major =
            std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major =
            std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN)
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_B_K_CONTIGUOUS)
        if constexpr(is_a_col_major &&
                     (is_b_row_major ||
                      std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>))
#else
        if constexpr(is_a_col_major && is_b_row_major)
#endif
        {
            return RunStageOverlapDirectTn<HasHotLoop, TailNum, PackA>(
                a_dram_block_window,
                b_dram_block_window,
                num_loop,
                p_smem,
                packed_a_ptr,
                packed_a_stride,
                packed_a_stage_group);
        }
        else
#endif
        {
        (void)packed_a_ptr;
        (void)packed_a_stride;
        (void)packed_a_stage_group;
        auto&& [a_lds_block, b_lds_block] =
            PipelineImplBase{}.GetABLdsTensorViews(p_smem);
        constexpr auto a_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeBBlockDistributionEncode());
        auto a_windows =
            PipelineImplBase{}.GetAWindows(a_dram_block_window,
                                           a_lds_block,
                                           a_lds_dstr);
        auto b_windows =
            PipelineImplBase{}.GetBWindows(b_dram_block_window,
                                           b_lds_block,
                                           b_lds_dstr);
        auto& a_dram_window      = a_windows.get(I0{});
        auto& a_lds_store_window = a_windows.get(I1{});
        auto& a_lds_gemm_window  = a_windows.get(number<2>{});
        auto& b_dram_window      = b_windows.get(I0{});
        auto& b_lds_store_window = b_windows.get(I1{});
        auto& b_lds_gemm_window  = b_windows.get(number<2>{});

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        constexpr ADramTileWindowStep a_step =
            is_a_col_major ? make_array(KPerBlock, 0)
                           : make_array(0, KPerBlock);
        constexpr BDramTileWindowStep b_step =
            is_b_row_major ? make_array(KPerBlock, 0)
                           : make_array(0, KPerBlock);

        auto a_reg_tile = load_tile(a_dram_window);
        auto b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);

        const auto store_a = [&](const auto& src) {
            if constexpr(is_a_col_major)
            {
                auto shuffled = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, src);
            }
        };
        const auto store_b = [&](const auto& src) {
            if constexpr(is_b_row_major)
            {
                auto shuffled = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, src);
            }
        };

        auto c_block_tile = BlockGemm{}.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        store_a(a_reg_tile);
        store_b(b_reg_tile);
        a_reg_tile = load_tile(a_dram_window);
        b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);
        block_sync_lds();

        if constexpr(HasHotLoop)
        {
            index_t i = 0;
            do
            {
                RunOperandStages(c_block_tile,
                                 a_lds_gemm_window,
                                 b_lds_gemm_window);
                block_sync_lds();
                store_a(a_reg_tile);
                store_b(b_reg_tile);
                a_reg_tile = load_tile(a_dram_window);
                b_reg_tile = load_tile(b_dram_window);
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);
                block_sync_lds();
                ++i;
            } while(i < num_loop - (TailNum == TailNumber::Even ? 2 : 1));
        }

        RunOperandStages(c_block_tile,
                         a_lds_gemm_window,
                         b_lds_gemm_window);
        if constexpr(TailNum == TailNumber::Even)
        {
            block_sync_lds();
            store_a(a_reg_tile);
            store_b(b_reg_tile);
            block_sync_lds();
            RunOperandStages(c_block_tile,
                             a_lds_gemm_window,
                             b_lds_gemm_window);
        }
        else if constexpr(!HasHotLoop && TailNum == TailNumber::Odd &&
                          Problem::BlockGemmShape::NumWarps == 8)
        {
            // 8-wave staged overlap classifies num_loop==3 as Odd without a
            // hot loop. Prologue stored tile 0 (computed above), prefetched
            // tile 1 into registers, and advanced the window to tile 2. The
            // Even-only epilogue would drop tiles 1 and 2, which is the
            // K in [129, 192] (KPerBlock=64) correctness hole.
            block_sync_lds();
            store_a(a_reg_tile);
            store_b(b_reg_tile);
            a_reg_tile = load_tile(a_dram_window);
            b_reg_tile = load_tile(b_dram_window);
            block_sync_lds();
            RunOperandStages(c_block_tile,
                             a_lds_gemm_window,
                             b_lds_gemm_window);
            block_sync_lds();
            store_a(a_reg_tile);
            store_b(b_reg_tile);
            block_sync_lds();
            RunOperandStages(c_block_tile,
                             a_lds_gemm_window,
                             b_lds_gemm_window);
        }
        return c_block_tile;
        }
    }

    template <bool HasHotLoop,
              TailNumber TailNum,
              typename ADramBlockWindow,
              typename BDramBlockWindow>
    CK_TILE_DEVICE auto Run(const ADramBlockWindow& a_dram_block_window,
                            const BDramBlockWindow& b_dram_block_window,
                            index_t num_loop,
                            void* p_smem) const
    {
        static_assert(!Problem::DoubleSmemBuffer,
                      "Compute V3 dsreadm requires one LDS buffer");

        constexpr bool is_a_col_major =
            std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major =
            std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        auto&& [a_lds_block, b_lds_block] =
            PipelineImplBase{}.GetABLdsTensorViews(p_smem);

        constexpr auto a_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeBBlockDistributionEncode());

        auto a_windows =
            PipelineImplBase{}.GetAWindows(a_dram_block_window, a_lds_block, a_lds_dstr);
        auto b_windows =
            PipelineImplBase{}.GetBWindows(b_dram_block_window, b_lds_block, b_lds_dstr);
        auto& a_dram_window      = a_windows.get(I0{});
        auto& a_lds_store_window = a_windows.get(I1{});
        auto& a_lds_gemm_window  = a_windows.get(number<2>{});
        auto& b_dram_window      = b_windows.get(I0{});
        auto& b_lds_store_window = b_windows.get(I1{});
        auto& b_lds_gemm_window  = b_windows.get(number<2>{});

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        constexpr ADramTileWindowStep a_step =
            is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
        constexpr BDramTileWindowStep b_step =
            is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

        auto a_reg_tile = load_tile(a_dram_window);
        auto b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);

        const auto store_a = [&](const auto& src) {
            if constexpr(is_a_col_major)
            {
                auto shuffled = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, src);
            }
        };
        const auto store_b = [&](const auto& src) {
            if constexpr(is_b_row_major)
            {
                auto shuffled = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, src);
            }
        };

        auto block_gemm   = BlockGemm{};
        auto c_block_tile = block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        store_a(a_reg_tile);
        store_b(b_reg_tile);

        a_reg_tile = load_tile(a_dram_window);
        b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);

        block_sync_lds();
        auto a_warp_tiles = block_gemm.LdsLoadA(a_lds_gemm_window);
        auto b_warp_tiles = block_gemm.LdsLoadB(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(HasHotLoop)
        {
            index_t i = 0;
            do
            {
                block_gemm(c_block_tile, a_warp_tiles, b_warp_tiles);
                block_sync_lds();

                store_a(a_reg_tile);
                store_b(b_reg_tile);

                a_reg_tile = load_tile(a_dram_window);
                b_reg_tile = load_tile(b_dram_window);
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);

                block_sync_lds();
                a_warp_tiles = block_gemm.LdsLoadA(a_lds_gemm_window);
                b_warp_tiles = block_gemm.LdsLoadB(b_lds_gemm_window);
                __builtin_amdgcn_sched_barrier(0);
                ++i;
            } while(i < num_loop - (TailNum == TailNumber::Even ? 2 : 1));
        }

        if constexpr(TailNum == TailNumber::Odd || TailNum == TailNumber::One)
        {
            block_gemm(c_block_tile, a_warp_tiles, b_warp_tiles);
        }
        else
        {
            block_gemm(c_block_tile, a_warp_tiles, b_warp_tiles);
            block_sync_lds();
            store_a(a_reg_tile);
            store_b(b_reg_tile);
            block_sync_lds();
            a_warp_tiles = block_gemm.LdsLoadA(a_lds_gemm_window);
            b_warp_tiles = block_gemm.LdsLoadB(b_lds_gemm_window);
            block_gemm(c_block_tile, a_warp_tiles, b_warp_tiles);
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto run = [&](auto hot_loop, auto tail) {
            if constexpr(StageOperandOverlap)
            {
                return RunStageOverlap<hot_loop.value, tail.value, false>(
                    a_dram_block_window,
                    b_dram_block_window,
                    num_loop,
                    p_smem);
            }
            else
            {
                return Run<hot_loop.value, tail.value>(
                    a_dram_block_window,
                    b_dram_block_window,
                    num_loop,
                    p_smem);
            }
        };
        return Base::TailHandler(run, has_hot_loop, tail_number);
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem) const
    {
#if defined(CK_TILE_GROUPED_GEMM_DSREADM_FIXED_HOT_EVEN)
        (void)has_hot_loop;
        (void)tail_number;
        static_assert(StageOperandOverlap,
                      "fixed hot/even gate is specialized for staged DSReadM");
        return RunStageOverlap<true, TailNumber::Even, false>(
            a_dram_block_window,
            b_dram_block_window,
            num_loop,
            p_smem);
#else
        const auto run = [&](auto hot_loop, auto tail) {
            if constexpr(StageOperandOverlap)
            {
                return RunStageOverlap<hot_loop.value, tail.value, false>(
                    a_dram_block_window,
                    b_dram_block_window,
                    num_loop,
                    p_smem);
            }
            else
            {
                return Run<hot_loop.value, tail.value>(
                    a_dram_block_window,
                    b_dram_block_window,
                    num_loop,
                    p_smem);
            }
        };
        return Base::TailHandler(run, has_hot_loop, tail_number);
#endif
    }

#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DSREADM)
    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem,
                                   ADataType* packed_a_ptr,
                                   index_t packed_a_stride,
                                   index_t packed_a_stage_group) const
    {
        const auto run = [&](auto hot_loop, auto tail) {
            static_assert(StageOperandOverlap,
                          "fused DSReadM packing requires staged operands");
            return RunStageOverlap<hot_loop.value, tail.value, true>(
                a_dram_block_window,
                b_dram_block_window,
                num_loop,
                p_smem,
                packed_a_ptr,
                packed_a_stride,
                packed_a_stage_group);
        };
        return Base::TailHandler(run, has_hot_loop, tail_number);
    }
#endif
};

} // namespace ck_tile
