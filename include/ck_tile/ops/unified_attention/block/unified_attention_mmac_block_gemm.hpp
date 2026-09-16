// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/ops/unified_attention/pipeline/unified_attention_mmac_problem.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_mmac_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/mmac_block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"

namespace ck_tile {

template <typename BlockWarps_, typename WarpGemm_>
struct LocalARegBSmemPolicy
{
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpGemm   = remove_cvref_t<WarpGemm_>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemm{}, BlockWarps::at(number<0>{}), BlockWarps::at(number<1>{}));
    }
};

template <typename Problem_, typename Policy_>
struct LocalBlockGemmARegBSmemCReg
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    CK_TILE_DEVICE auto MakeABlockTile() const
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;
        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});
        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        return make_static_distributed_tensor<ADataType>(a_block_dstr);
    }

    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iNWarp = get_warp_id() % NWarp;

        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});
        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        auto a_block_tensor = make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);
        a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_block_window_tmp.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        using AWarpDstr   = typename WG::AWarpDstr;
        using CWarpDstr   = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }
};

template <typename Problem_, typename Policy_>
struct LocalBlockGemmARegBRegCReg
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    CK_TILE_DEVICE static constexpr auto MakeABlockTile()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;
        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});
        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        return make_static_distributed_tensor<ADataType>(a_block_dstr);
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockTile()
    {
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;
        constexpr auto b_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});
        constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);
        return make_static_distributed_tensor<BDataType>(b_block_dstr);
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return make_static_distributed_tensor<CDataType>(c_block_dstr);
    }

    template <typename CBlockTensor, typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BBlockTensor::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        using AWarpDstr = typename WG::AWarpDstr;
        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                AWarpTensor a_warp_tensor;
                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    template <typename CBlockTensor>
    CK_TILE_DEVICE auto MakeOuputLayout(const CBlockTensor& c_block_tensor) const
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        using CWarpDstr         = typename WG::CWarpDstr;
        using CWarpOutputDstr   = typename WG::CWarpOutputDstr;
        using CWarpTensor       = typename WG::CWarpTensor;
        using CWarpOutputTensor = typename WG::CWarpOutputTensor;

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        constexpr auto c_block_out_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpOutputDstrEncoding{});
        constexpr auto c_output_block_dstr = make_static_tile_distribution(c_block_out_dstr_encode);
        auto c_block_output_tensor = make_static_distributed_tensor<CDataType>(c_output_block_dstr);

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_output_y_lengths =
            to_sequence(CWarpOutputDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_output_y_index_zeros =
            uniform_sequence_gen_t<CWarpOutputDstr::NDimY, 0>{};

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                CWarpOutputTensor c_warp_output_tensor = WG{}.MakeCOutputLayout(c_warp_tensor);
                c_block_output_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_output_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_output_y_lengths),
                    c_warp_output_tensor.get_thread_buffer());
            });
        });

        return c_block_output_tensor;
    }
};


template <int NIter, typename OutTensor, typename InTensor>
CK_TILE_DEVICE void repack_qk16x64_p_to_pv32_areg(OutTensor& out, const InTensor& in)
{
    using OutDataType = typename OutTensor::DataType;
    constexpr index_t kElementsPerLaneGroup = 8;

    const index_t lane     = get_lane_id();
    const index_t row_lane = lane & 15;
    const index_t dst_grp  = (lane >> 4) & 3;

    static_for<0, 1, 1>{}([&](auto m_iter) {
        static_for<0, 4, 1>{}([&](auto group) {
            static_for<0, kElementsPerLaneGroup, 1>{}([&](auto e) {
                constexpr index_t col64 =
                    number<NIter>{} * 32 + group * kElementsPerLaneGroup + e;
                constexpr index_t src_grp = col64 % 4;
                constexpr index_t i_nr    = col64 / 16;
                constexpr index_t i_cn0   = (col64 / 4) % 4;
                constexpr index_t src_off = m_iter * 16 + i_nr * 4 + i_cn0;
                constexpr index_t dst_off = m_iter * kElementsPerLaneGroup + e;

                const auto local_v =
                    type_convert<OutDataType>(in.get_thread_buffer()[number<src_off>{}]);
                const auto remote_v = warp_shuffle(local_v, row_lane + 16 * src_grp);
                if (dst_grp == group) {
                    out.get_thread_buffer()(number<dst_off>{}) = remote_v;
                }
            });
        });
    });
}

template <int NIter, typename OutTensor, typename InTensor>
CK_TILE_DEVICE void repack_qk16x64_p_to_pv64_areg(OutTensor& out, const InTensor& in)
{
    using OutDataType = typename OutTensor::DataType;
    constexpr index_t kElementsPerLaneGroup = 8;

    const index_t lane     = get_lane_id();
    const index_t row_lane = lane & 15;
    const index_t dst_grp  = (lane >> 4) & 3;

    // QK output for a 16x64 warp:
    //   source lane group owns columns iNR * 16 + iCN0 * 4 + src_grp.
    // PV AReg for a K=64 slice:
    //   destination lane group owns columns k_iter * 32 + dst_grp * 8 + e.
    static_for<0, 1, 1>{}([&](auto m_iter) {
        static_for<0, 2, 1>{}([&](auto k_iter) {
            static_for<0, 4, 1>{}([&](auto group) {
                static_for<0, kElementsPerLaneGroup, 1>{}([&](auto e) {
                    constexpr index_t col64 =
                        k_iter * 32 + group * kElementsPerLaneGroup + e;
                    constexpr index_t src_grp = col64 % 4;
                    constexpr index_t i_nr    = col64 / 16;
                    constexpr index_t i_cn0   = (col64 / 4) % 4;
                    constexpr index_t src_off = ((m_iter + number<NIter>{}) * 16) +
                                                i_nr * 4 + i_cn0;
                    constexpr index_t dst_off =
                        (m_iter * 2 + k_iter) * kElementsPerLaneGroup + e;

                    const auto local_v =
                        type_convert<OutDataType>(in.get_thread_buffer()[number<src_off>{}]);
                    const auto remote_v = warp_shuffle(local_v, row_lane + 16 * src_grp);
                    if (dst_grp == group) {
                        out.get_thread_buffer()(number<dst_off>{}) = remote_v;
                    }
                });
            });
        });
    });
}

} // namespace ck_tile

template <typename Problem, int D, typename PipelinePolicy>
using UnifiedAttentionQkBlockGemm = ck_tile::MmacBlockGemmASmemBSmemCRegV1<
    UnifiedAttentionQkBlockGemmProblem<Problem, D, PipelinePolicy>,
    ck_tile::BlockGemmASmemBSmemCRegV1CustomPolicy<
        typename Problem::QDataType, typename Problem::KDataType, typename Problem::AccDataType,
        typename PipelinePolicy::template QkBlockWarps<D>,
        typename PipelinePolicy::QkWarpGemm>>;


template <typename Problem, int NKey, int D, typename PipelinePolicy>
using UnifiedAttentionQkBlockGemmN = ck_tile::MmacBlockGemmASmemBSmemCRegV1<
    UnifiedAttentionQkBlockGemmProblemN<Problem, NKey, D, PipelinePolicy>,
    ck_tile::BlockGemmASmemBSmemCRegV1CustomPolicy<
        typename Problem::QDataType, typename Problem::KDataType, typename Problem::AccDataType,
        typename PipelinePolicy::template QkBlockWarpsN<NKey>,
        typename PipelinePolicy::QkWarpGemm32>>;

template <typename Problem, int N, typename PipelinePolicy>
using UnifiedAttentionPvBlockGemm = ck_tile::MmacBlockGemmASmemBSmemCRegV1<
    UnifiedAttentionPvBlockGemmProblem<Problem, N, PipelinePolicy>,
    ck_tile::BlockGemmASmemBSmemCRegV1CustomPolicy<
        typename Problem::QDataType, typename Problem::VDataType, typename Problem::AccDataType,
        typename PipelinePolicy::template PvBlockWarps<N>,
        typename PipelinePolicy::PvWarpGemm>>;

template <int D, typename PipelinePolicy>
using Phase1ARegBSmemBlockGemm = ck_tile::LocalBlockGemmARegBSmemCReg<
    UnifiedAttentionQkBlockGemmProblem<typename PipelinePolicy::Problem, D, PipelinePolicy>,
    ck_tile::LocalARegBSmemPolicy<
        typename PipelinePolicy::template QkBlockWarps<D>,
        typename PipelinePolicy::QkWarpGemm>>;


template <typename Problem, int NKey, int D, typename PipelinePolicy>
using Phase1ARegBSmemBlockGemmNForProblem = ck_tile::LocalBlockGemmARegBSmemCReg<
    UnifiedAttentionQkBlockGemmProblemN<Problem, NKey, D, PipelinePolicy>,
    ck_tile::LocalARegBSmemPolicy<
        typename PipelinePolicy::template QkBlockWarpsN<NKey>,
        typename PipelinePolicy::QkWarpGemm32>>;

template <int NKey, int D, typename PipelinePolicy>
using Phase1ARegBSmemBlockGemmN = Phase1ARegBSmemBlockGemmNForProblem<
    typename PipelinePolicy::Problem, NKey, D, PipelinePolicy>;

template <int N, typename PipelinePolicy>
using Phase2PvARegBSmemBlockGemm = ck_tile::LocalBlockGemmARegBSmemCReg<
    UnifiedAttentionPvBlockGemmProblem<typename PipelinePolicy::Problem, N, PipelinePolicy>,
    ck_tile::LocalARegBSmemPolicy<
        typename PipelinePolicy::template PvBlockWarps<N>,
        typename PipelinePolicy::PvWarpGemm>>;

template <typename Problem, int N, int K, typename PipelinePolicy>
using Phase2PvARegBSmemBlockGemmKForProblem = ck_tile::LocalBlockGemmARegBSmemCReg<
    UnifiedAttentionPvBlockGemmProblemK<Problem, N, K, PipelinePolicy>,
    ck_tile::LocalARegBSmemPolicy<
        typename PipelinePolicy::template PvBlockWarps<N>,
        typename PipelinePolicy::PvWarpGemm>>;

template <int N, int K, typename PipelinePolicy>
using Phase2PvARegBSmemBlockGemmK = Phase2PvARegBSmemBlockGemmKForProblem<
    typename PipelinePolicy::Problem, N, K, PipelinePolicy>;

template <typename Problem, int N, int K, typename PipelinePolicy>
using Phase2PvARegBRegBlockGemmKForProblem = ck_tile::LocalBlockGemmARegBRegCReg<
    UnifiedAttentionPvBlockGemmProblemK<Problem, N, K, PipelinePolicy>,
    ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<
        typename Problem::QDataType, typename Problem::VDataType, typename Problem::AccDataType,
        typename PipelinePolicy::template PvBlockWarps<N>,
        typename PipelinePolicy::PvWarpGemm>>;

template <int N, int K, typename PipelinePolicy>
using Phase2PvARegBRegBlockGemmK = Phase2PvARegBRegBlockGemmKForProblem<
    typename PipelinePolicy::Problem, N, K, PipelinePolicy>;
