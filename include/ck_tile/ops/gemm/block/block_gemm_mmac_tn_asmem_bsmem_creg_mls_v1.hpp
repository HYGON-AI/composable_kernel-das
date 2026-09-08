// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_dispatcher_v2.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"

namespace ck_tile {

// GEMM-local block GEMM for the gfx938 MLS path. A/B are staged by MLS in the
// packed [M/N, K] LDS layout consumed by DS Matrix reads.
template <typename Problem>
struct BlockGemmMmacTNAsmemBsmemCregMlsV1
{
    using ADataType     = remove_cvref_t<typename Problem::ADataType>;
    using BDataType     = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType   = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static_assert(std::is_same_v<ADataType, BDataType> &&
                      (std::is_same_v<ADataType, half_t> ||
                       std::is_same_v<ADataType, bf16_t>) &&
                      std::is_same_v<AccDataType, float>,
                  "Grouped-GEMM MLS supports matching FP16/BF16 inputs and FP32 accumulation");

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
    static constexpr index_t BlockSize = Problem::kBlockSize;

    using BlockWarps = typename BlockGemmShape::BlockWarps;
    using WarpTile   = typename BlockGemmShape::WarpTile;

    static constexpr index_t MWarps = BlockWarps::at(number<0>{});
    static constexpr index_t NWarps = BlockWarps::at(number<1>{});

    static constexpr index_t MPerMmac = 16;
    static constexpr index_t NPerMmac = 16;
    static constexpr index_t MmmacIter = 1;
    static constexpr index_t NmmacIter = 1;
    static constexpr index_t MmmacInterleave = WarpTile::at(number<0>{}) / MPerMmac;
    static constexpr index_t NmmacInterleave = WarpTile::at(number<1>{}) / NPerMmac;
    static constexpr index_t MWarpIter =
        MPerBlock / (MWarps * MmmacIter * MPerMmac * MmmacInterleave);
    static constexpr index_t NWarpIter =
        NPerBlock / (NWarps * NmmacIter * NPerMmac * NmmacInterleave);

    static constexpr index_t MPerMlsLoad = MmmacIter * MPerMmac * MmmacInterleave;
    static constexpr index_t NPerMlsLoad = NmmacIter * NPerMmac * NmmacInterleave;

    static constexpr bool AUsesTransMls =
        std::is_same_v<typename Problem::ALayout, tensor_layout::gemm::RowMajor>;
    static constexpr bool BUsesTransMls =
        std::is_same_v<typename Problem::BLayout, tensor_layout::gemm::ColumnMajor>;

    using AWarpDsreadmFormat = std::conditional_t<
        AUsesTransMls,
        WarpDsreadmFormatDispatcherV2<
            sizeof(ADataType), MPerMmac, MPerMlsLoad, MmmacInterleave, true>,
        WarpDsreadmFormatDispatcherV2<
            sizeof(ADataType), MPerMlsLoad, MPerMmac, MmmacInterleave, false>>;
    using BWarpDsreadmFormat = std::conditional_t<
        BUsesTransMls,
        WarpDsreadmFormatDispatcherV2<
            sizeof(BDataType), NPerMmac, NPerMlsLoad, NmmacInterleave, true>,
        WarpDsreadmFormatDispatcherV2<
            sizeof(BDataType), NPerMlsLoad, NPerMmac, NmmacInterleave, false>>;

    static constexpr index_t ALdsWarpElemOffsetM = MPerMlsLoad * KPerBlock;
    static constexpr index_t BLdsWarpElemOffsetN = NPerMlsLoad * KPerBlock;
    static constexpr index_t ALdsImmedOffsetM =
        ALdsWarpElemOffsetM * MWarps * sizeof(ADataType);
    static constexpr index_t BLdsImmedOffsetN =
        BLdsWarpElemOffsetN * NWarps * sizeof(BDataType);
    static constexpr index_t ALdsImmedOffsetK =
        AWarpDsreadmFormat::Impl::kK *
        (AUsesTransMls ? 1 : MPerMlsLoad) * sizeof(ADataType);
    static constexpr index_t BLdsImmedOffsetK =
        BWarpDsreadmFormat::Impl::kK *
        (BUsesTransMls ? 1 : NPerMlsLoad) * sizeof(BDataType);

#if defined(CK_TILE_GROUPED_GEMM_MLS_USE_LIT_LTS)
    using MmacImpl = std::conditional_t<
        std::is_same_v<ADataType, half_t>,
        WarpGemmAttributeMmacImplF16F16F32M16N16K16LitLts,
        WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16LitLts>;
    using WarpGemmK1 =
        WarpGemmImpl<WarpGemmAttributeMmacIterateKLitLts<MmacImpl,
                                                         MmmacIter,
                                                         NmmacIter,
                                                         MmmacInterleave,
                                                         NmmacInterleave,
                                                         1>>;
#else
    using MmacImpl = std::conditional_t<
        std::is_same_v<ADataType, half_t>,
        WarpGemmAttributeMmacImplF16F16F32M16N16K16,
        WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16>;
    using WarpGemmK1 =
        WarpGemmImpl<WarpGemmAttributeMmacIterateK<MmacImpl,
                                                   MmmacIter,
                                                   NmmacIter,
                                                   MmmacInterleave,
                                                   NmmacInterleave,
                                                   1>>;
#endif

    static constexpr index_t KIter =
        (AWarpDsreadmFormat::kKStoreLane * AWarpDsreadmFormat::kKStorePerLane) /
        WarpGemmK1::kK;

#if defined(CK_TILE_GROUPED_GEMM_MLS_USE_LIT_LTS)
    using WarpGemm =
        WarpGemmImpl<WarpGemmAttributeMmacIterateKLitLts<MmacImpl,
                                                         MmmacIter,
                                                         NmmacIter,
                                                         MmmacInterleave,
                                                         NmmacInterleave,
                                                         KIter>>;
#else
    using WarpGemm =
        WarpGemmImpl<WarpGemmAttributeMmacIterateK<MmacImpl,
                                                   MmmacIter,
                                                   NmmacIter,
                                                   MmmacInterleave,
                                                   NmmacInterleave,
                                                   KIter>>;
#endif

    static constexpr index_t KWarpIter = KPerBlock / WarpGemm::kK;

    using AWarpDsreadmDstr = remove_cvref_t<decltype(make_static_tile_distribution(
        typename AWarpDsreadmFormat::WarpStoreDstrEncoding{}))>;
    using BWarpDsreadmDstr = remove_cvref_t<decltype(make_static_tile_distribution(
        typename BWarpDsreadmFormat::WarpStoreDstrEncoding{}))>;

    static_assert(std::is_same_v<AWarpDsreadmDstr, typename WarpGemm::AWarpDstr>,
                  "MLS A DS Matrix fragment distribution must match WarpGemm");
    static_assert(std::is_same_v<BWarpDsreadmDstr, typename WarpGemm::BWarpDstr>,
                  "MLS B DS Matrix fragment distribution must match WarpGemm");

    CK_TILE_DEVICE BlockGemmMmacTNAsmemBsmemCregMlsV1()
    {
        static_assert(BlockSize == MWarps * NWarps * get_warp_size(),
                      "BlockSize must equal MWarps * NWarps * wave size");
        static_assert(MPerBlock %
                              (MWarpIter * MmmacIter * MPerMmac * MmmacInterleave) ==
                          0,
                      "Invalid M MLS block decomposition");
        static_assert(NPerBlock %
                              (NWarpIter * NmmacIter * NPerMmac * NmmacInterleave) ==
                          0,
                      "Invalid N MLS block decomposition");

        const index_t warp_id_m = get_warp_id() / NWarps;
        const index_t warp_id_n = get_warp_id() % NWarps;
        a_warp_lds_elem_offset  = warp_id_m * ALdsWarpElemOffsetM;
        b_warp_lds_elem_offset  = warp_id_n * BLdsWarpElemOffsetN;
    }

    template <bool UseM0 = true>
    CK_TILE_DEVICE auto GetAWarpTensors(CK_TILE_LDS_ADDR ADataType* smem_ptr,
                                        bool_constant<UseM0> = {}) const
    {
        using vector_t = ext_vector_t<ADataType, AWarpDsreadmFormat::kVectorLength>;

        return generate_tuple(
            [&](auto i) {
                constexpr auto k_warp_iter = i % KWarpIter;
                constexpr auto m_warp_iter = i / KWarpIter;
                constexpr auto immed_offset =
                    number<m_warp_iter * ALdsImmedOffsetM +
                           k_warp_iter * ALdsImmedOffsetK>{};

                vector_t vec_value = bit_cast<vector_t>(AWarpDsreadmFormat{}(
                    smem_ptr + a_warp_lds_elem_offset,
                    immed_offset,
                    bool_constant<UseM0>{}));

                auto warp_tensor =
                    make_static_distributed_tensor<ADataType>(make_static_tile_distribution(
                        typename AWarpDsreadmFormat::WarpStoreDstrEncoding{}));
                warp_tensor.get_thread_buffer().template get_as<vector_t>()(number<0>{}) =
                    vec_value;
                return warp_tensor;
            },
            number<MWarpIter * KWarpIter>{});
    }

    template <index_t KBegin, index_t KCount, bool UseM0 = true>
    CK_TILE_DEVICE auto GetAWarpTensorsSlice(CK_TILE_LDS_ADDR ADataType* smem_ptr,
                                             bool_constant<UseM0> = {}) const
    {
        static_assert(KBegin >= 0 && KCount > 0 && KBegin + KCount <= KWarpIter,
                      "Invalid A MLS K-fragment slice");
        using vector_t = ext_vector_t<ADataType, AWarpDsreadmFormat::kVectorLength>;

        return generate_tuple(
            [&](auto i) {
                constexpr auto k_warp_iter = KBegin + i % KCount;
                constexpr auto m_warp_iter = i / KCount;
                constexpr auto immed_offset =
                    number<m_warp_iter * ALdsImmedOffsetM +
                           k_warp_iter * ALdsImmedOffsetK>{};
#if defined(CK_TILE_GROUPED_GEMM_MLS_REUSE_DS_M0)
                constexpr bool skip_mov_m0 = (i != 0);
#else
                constexpr bool skip_mov_m0 = false;
#endif

                vector_t vec_value = bit_cast<vector_t>(AWarpDsreadmFormat{}(
                    smem_ptr + a_warp_lds_elem_offset,
                    immed_offset,
                    bool_constant<UseM0>{},
                    bool_constant<skip_mov_m0>{}));

                auto warp_tensor =
                    make_static_distributed_tensor<ADataType>(make_static_tile_distribution(
                        typename AWarpDsreadmFormat::WarpStoreDstrEncoding{}));
                warp_tensor.get_thread_buffer().template get_as<vector_t>()(number<0>{}) =
                    vec_value;
                return warp_tensor;
            },
            number<MWarpIter * KCount>{});
    }

    template <bool UseM0 = true>
    CK_TILE_DEVICE auto GetBWarpTensors(CK_TILE_LDS_ADDR BDataType* smem_ptr,
                                        bool_constant<UseM0> = {}) const
    {
        using vector_t = ext_vector_t<BDataType, BWarpDsreadmFormat::kVectorLength>;

        return generate_tuple(
            [&](auto i) {
                constexpr auto k_warp_iter = i % KWarpIter;
                constexpr auto n_warp_iter = i / KWarpIter;
                constexpr auto immed_offset =
                    number<n_warp_iter * BLdsImmedOffsetN +
                           k_warp_iter * BLdsImmedOffsetK>{};

                vector_t vec_value = bit_cast<vector_t>(BWarpDsreadmFormat{}(
                    smem_ptr + b_warp_lds_elem_offset,
                    immed_offset,
                    bool_constant<UseM0>{}));

                auto warp_tensor =
                    make_static_distributed_tensor<BDataType>(make_static_tile_distribution(
                        typename BWarpDsreadmFormat::WarpStoreDstrEncoding{}));
                warp_tensor.get_thread_buffer().template get_as<vector_t>()(number<0>{}) =
                    vec_value;
                return warp_tensor;
            },
            number<NWarpIter * KWarpIter>{});
    }

    template <index_t KBegin, index_t KCount, bool UseM0 = true>
    CK_TILE_DEVICE auto GetBWarpTensorsSlice(CK_TILE_LDS_ADDR BDataType* smem_ptr,
                                             bool_constant<UseM0> = {}) const
    {
        static_assert(KBegin >= 0 && KCount > 0 && KBegin + KCount <= KWarpIter,
                      "Invalid B MLS K-fragment slice");
        using vector_t = ext_vector_t<BDataType, BWarpDsreadmFormat::kVectorLength>;

        return generate_tuple(
            [&](auto i) {
                constexpr auto k_warp_iter = KBegin + i % KCount;
                constexpr auto n_warp_iter = i / KCount;
                constexpr auto immed_offset =
                    number<n_warp_iter * BLdsImmedOffsetN +
                           k_warp_iter * BLdsImmedOffsetK>{};
#if defined(CK_TILE_GROUPED_GEMM_MLS_REUSE_DS_M0)
                constexpr bool skip_mov_m0 = (i != 0);
#else
                constexpr bool skip_mov_m0 = false;
#endif

                vector_t vec_value = bit_cast<vector_t>(BWarpDsreadmFormat{}(
                    smem_ptr + b_warp_lds_elem_offset,
                    immed_offset,
                    bool_constant<UseM0>{},
                    bool_constant<skip_mov_m0>{}));

                auto warp_tensor =
                    make_static_distributed_tensor<BDataType>(make_static_tile_distribution(
                        typename BWarpDsreadmFormat::WarpStoreDstrEncoding{}));
                warp_tensor.get_thread_buffer().template get_as<vector_t>()(number<0>{}) =
                    vec_value;
                return warp_tensor;
            },
            number<NWarpIter * KCount>{});
    }

    template <typename CBlockTensor, typename AWarpTensors, typename BWarpTensors>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const AWarpTensors& a_warp_tensors,
                                   const BWarpTensors& b_warp_tensors) const
    {
        using SFC = space_filling_curve<sequence<MWarpIter, NWarpIter, KWarpIter>,
                                        sequence<0, 1, 2>,
                                        sequence<1, 1, 1>>;

        static_for<0, SFC::get_num_of_access(), 1>{}([&](auto access_id) {
            constexpr auto idx         = SFC::get_index(access_id);
            constexpr auto m_warp_iter = idx.at(number<0>{});
            constexpr auto n_warp_iter = idx.at(number<1>{});
            constexpr auto k_warp_iter = idx.at(number<2>{});

            using CWarpDstr   = typename WarpGemm::CWarpDstr;
            using CWarpTensor = typename WarpGemm::CWarpTensor;
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros =
                uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            CWarpTensor c_warp_tensor;
            c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            WarpGemm{}(c_warp_tensor,
                       a_warp_tensors[number<m_warp_iter * KWarpIter + k_warp_iter>{}],
                       b_warp_tensors[number<n_warp_iter * KWarpIter + k_warp_iter>{}]);

            c_block_tensor.set_y_sliced_thread_data(
                merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                c_warp_tensor.get_thread_buffer());
        });
    }

    template <index_t KCount,
              typename CBlockTensor,
              typename AWarpTensors,
              typename BWarpTensors>
    CK_TILE_DEVICE void RunKSlice(CBlockTensor& c_block_tensor,
                                  const AWarpTensors& a_warp_tensors,
                                  const BWarpTensors& b_warp_tensors) const
    {
        static_assert(KCount > 0 && KCount <= KWarpIter, "Invalid MLS K compute slice");
        using SFC = space_filling_curve<sequence<MWarpIter, NWarpIter, KCount>,
                                        sequence<0, 1, 2>,
                                        sequence<1, 1, 1>>;

        static_for<0, SFC::get_num_of_access(), 1>{}([&](auto access_id) {
            constexpr auto idx         = SFC::get_index(access_id);
            constexpr auto m_warp_iter = idx.at(number<0>{});
            constexpr auto n_warp_iter = idx.at(number<1>{});
            constexpr auto k_warp_iter = idx.at(number<2>{});

            using CWarpDstr   = typename WarpGemm::CWarpDstr;
            using CWarpTensor = typename WarpGemm::CWarpTensor;
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros =
                uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            CWarpTensor c_warp_tensor;
            c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            WarpGemm{}(c_warp_tensor,
                       a_warp_tensors[number<m_warp_iter * KCount + k_warp_iter>{}],
                       b_warp_tensors[number<n_warp_iter * KCount + k_warp_iter>{}]);

            c_block_tensor.set_y_sliced_thread_data(
                merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                c_warp_tensor.get_thread_buffer());
        });
    }

    CK_TILE_DEVICE static auto MakeCBlockTile()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MWarpIter, MWarps>, sequence<NWarpIter, NWarps>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encoding = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr =
            make_static_tile_distribution(c_block_dstr_encoding);
        return make_static_distributed_tensor<AccDataType>(c_block_dstr);
    }

    index_t a_warp_lds_elem_offset;
    index_t b_warp_lds_elem_offset;
};

} // namespace ck_tile
