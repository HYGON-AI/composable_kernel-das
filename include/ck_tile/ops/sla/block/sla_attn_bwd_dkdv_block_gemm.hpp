// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Copied from the fast fwd CK path and renamed for the bwd dK/dV experiment.

#pragma once

#include "ck_tile/core.hpp"
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_>
struct SlaLocalBlockGemmARegBRegCReg
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

    template <index_t KIter,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor>
    CK_TILE_DEVICE void RunKIter(CBlockTensor& c_block_tensor,
                                const ABlockTensor& a_block_tensor,
                                const BBlockTensor& b_block_tensor,
                                number<KIter>) const
    {
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
        static_assert(KIter < KIterPerWarp, "invalid dK/dV K fragment");

        using AWarpDstr   = typename WG::AWarpDstr;
        using BWarpDstr   = typename WG::BWarpDstr;
        using CWarpDstr   = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros =
            uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros =
            uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros =
            uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            AWarpTensor a_warp_tensor;
            a_warp_tensor.get_thread_buffer() =
                a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, KIter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                BWarpTensor b_warp_tensor;
                b_warp_tensor.get_thread_buffer() =
                    b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<nIter, KIter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() =
                    c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.get_thread_buffer());
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

template <typename Problem_, typename Policy_>
struct SlaLocalBlockGemmARegBRegCRegInterleavedB
    : SlaLocalBlockGemmARegBRegCReg<Problem_, Policy_>
{
    using Base           = SlaLocalBlockGemmARegBRegCReg<Problem_, Policy_>;
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();
    static constexpr index_t NIterPerWarp =
        BlockGemmShape::kN / (NWarp * WG::kN);
    static constexpr index_t KIterPerWarp = BlockGemmShape::kK / WG::kK;

    using BWarpDstr       = typename WG::BWarpDstr;
    using BWarpTensor     = typename WG::BWarpTensor;
    using BWarpThreadDesc = typename BWarpTensor::ThreadTensorDesc;
    static constexpr index_t BWarpThreadSize = BWarpTensor::get_thread_buffer_size();

    struct InterleavedBBlockTile
    {
        thread_buffer<uint32_t, KIterPerWarp * BWarpThreadSize> packed;

        template <index_t KIter, index_t WaitCount>
        CK_TILE_DEVICE void FenceKIter()
        {
            static_assert(BWarpThreadSize == 4,
                          "interleaved B wait assumes a bf16x4 warp fragment");
            buffer_load_fence(
                WaitCount,
                packed(number<KIter * BWarpThreadSize + 0>{}),
                packed(number<KIter * BWarpThreadSize + 1>{}),
                packed(number<KIter * BWarpThreadSize + 2>{}),
                packed(number<KIter * BWarpThreadSize + 3>{}));
        }
    };

    CK_TILE_DEVICE auto LoadBBlockTileDword(const BDataType* __restrict__ x,
                                            index_t row_stride,
                                            index_t n_base,
                                            index_t start_k) const
    {
        static_assert(MWarp == 1 && NWarp == 4,
                      "interleaved B path requires a 1M x 4N wave arrangement");
        static_assert(NIterPerWarp == 2 && KIterPerWarp == 2,
                      "interleaved B path is specialized for N128 x K32");
        static_assert(sizeof(BDataType) == 2,
                      "interleaved B dword path requires 16-bit inputs");

        InterleavedBBlockTile b_block;
        const index_t i_nwarp = get_warp_id() % NWarp;
        const auto resource = make_wave_buffer_resource(x);

        static_for<0, KIterPerWarp, 1>{}([&](auto k_iter) {
            constexpr auto spans = BWarpTensor::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx_n) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx_k) {
                    constexpr auto idx = make_tuple(idx_n, idx_k);
                    constexpr auto y_idx =
                        BWarpDstr{}.get_y_indices_from_distributed_indices(decltype(idx){});
                    constexpr index_t fragment_offset =
                        BWarpThreadDesc{}.calculate_offset(y_idx);
                    const auto x_idx =
                        get_x_indices_from_distributed_indices(BWarpDstr{}, idx);
                    const index_t n_lane = x_idx.at(number<0>{});
                    const index_t k_lane = x_idx.at(number<1>{});

                    const index_t physical_n =
                        n_base + i_nwarp * 32 + n_lane * 2;
                    const index_t physical_k =
                        start_k + k_iter * WG::kK + k_lane;
                    const index_t byte_offset =
                        (physical_k * row_stride + physical_n) *
                        static_cast<index_t>(sizeof(BDataType));

                    const auto x2 = hcu_buffer_load_asm_impl<BDataType, 2>(
                        resource, byte_offset, 0);
                    b_block.packed(
                        number<k_iter * BWarpThreadSize + fragment_offset>{}) =
                        bit_cast<uint32_t>(x2);
                });
            });
        });

        return b_block;
    }

    template <typename BLdsDesc>
    CK_TILE_DEVICE auto LoadBBlockTileDwordFromLds(
        const BDataType* __restrict__ x,
        const BLdsDesc& lds_desc,
        index_t n_wave_chunk_stride) const
    {
        static_assert(MWarp == 1 && NWarp == 4,
                      "interleaved LDS B path requires a 1M x 4N wave arrangement");
        static_assert(NIterPerWarp == 2 && KIterPerWarp == 2,
                      "interleaved LDS B path is specialized for N128 x K32");
        static_assert(sizeof(BDataType) == 2,
                      "interleaved LDS B dword path requires 16-bit inputs");

        InterleavedBBlockTile b_block;
        const index_t i_nwarp = get_warp_id() % NWarp;

        static_for<0, KIterPerWarp, 1>{}([&](auto k_iter) {
            constexpr auto spans = BWarpTensor::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx_n) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx_k) {
                    constexpr auto idx = make_tuple(idx_n, idx_k);
                    constexpr auto y_idx =
                        BWarpDstr{}.get_y_indices_from_distributed_indices(decltype(idx){});
                    constexpr index_t fragment_offset =
                        BWarpThreadDesc{}.calculate_offset(y_idx);
                    const auto x_idx =
                        get_x_indices_from_distributed_indices(BWarpDstr{}, idx);
                    const index_t n_lane = x_idx.at(number<0>{});
                    const index_t k_lane = x_idx.at(number<1>{});

                    const index_t logical_k = k_iter * WG::kK + k_lane;
                    const index_t logical_n = n_lane * 2;
                    const index_t lds_offset =
                        lds_desc.calculate_offset(make_multi_index(logical_k, logical_n)) +
                        i_nwarp * n_wave_chunk_stride;

                    __builtin_memcpy(&b_block.packed(number<k_iter * BWarpThreadSize + fragment_offset>{}), x + lds_offset, sizeof(uint32_t));
                });
            });
        });

        return b_block;
    }

    CK_TILE_DEVICE auto MakeInterleavedBBlockTile() const
    {
        return InterleavedBBlockTile{};
    }

    template <index_t KIter, typename BLdsDesc>
    CK_TILE_DEVICE void LoadBBlockTileDwordFromLdsKIter(
        InterleavedBBlockTile& b_block,
        const BDataType* __restrict__ x,
        const BLdsDesc& lds_desc,
        index_t n_wave_chunk_stride) const
    {
        static_assert(MWarp == 1 && NWarp == 4,
                      "interleaved LDS B path requires a 1M x 4N wave arrangement");
        static_assert(NIterPerWarp == 2 && KIterPerWarp == 2,
                      "interleaved LDS B path is specialized for N128 x K32");
        static_assert(KIter < KIterPerWarp, "invalid interleaved LDS B K fragment");
        static_assert(sizeof(BDataType) == 2,
                      "interleaved LDS B dword path requires 16-bit inputs");

        const index_t i_nwarp = get_warp_id() % NWarp;
        constexpr auto spans = BWarpTensor::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx_n) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx_k) {
                constexpr auto idx = make_tuple(idx_n, idx_k);
                constexpr auto y_idx =
                    BWarpDstr{}.get_y_indices_from_distributed_indices(decltype(idx){});
                constexpr index_t fragment_offset =
                    BWarpThreadDesc{}.calculate_offset(y_idx);
                const auto x_idx =
                    get_x_indices_from_distributed_indices(BWarpDstr{}, idx);
                const index_t n_lane = x_idx.at(number<0>{});
                const index_t k_lane = x_idx.at(number<1>{});

                const index_t logical_k = KIter * WG::kK + k_lane;
                const index_t logical_n = n_lane * 2;
                const index_t lds_offset =
                    lds_desc.calculate_offset(make_multi_index(logical_k, logical_n)) +
                    i_nwarp * n_wave_chunk_stride;

                __builtin_memcpy(&b_block.packed(number<KIter * BWarpThreadSize + fragment_offset>{}), x + lds_offset, sizeof(uint32_t));
            });
        });
    }

    template <index_t KIter, typename CBlockTensor, typename ABlockTensor>
    CK_TILE_DEVICE void RunKIterWithBWarpFragments(CBlockTensor& c_block_tensor,
                                                   const ABlockTensor& a_block_tensor,
                                                   const BWarpTensor& b0,
                                                   const BWarpTensor& b1) const
    {
        static_assert(KIter < KIterPerWarp, "invalid interleaved B K fragment");

        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        static_assert(NIterPerWarp == 2 && NPerBlock == NWarp * WG::kN * 2,
                      "direct interleaved B path expects two N iterations");

        using AWarpDstr   = typename WG::AWarpDstr;
        using CWarpDstr   = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros =
            uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros =
            uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, MIterPerWarp, 1>{}([&](auto m_iter) {
            AWarpTensor a_warp;
            a_warp.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                merge_sequences(sequence<m_iter, KIter>{}, a_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

            auto run_n_iter = [&](auto n_iter, const BWarpTensor& b_warp) {
                CWarpTensor c_warp;
                c_warp.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                WG{}(c_warp, a_warp, b_warp);
                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<m_iter, n_iter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp.get_thread_buffer());
            };

            run_n_iter(number<0>{}, b0);
            run_n_iter(number<1>{}, b1);
        });
    }

    template <index_t KIter>
    CK_TILE_DEVICE void PrepareBBlockTileDwordKIter(InterleavedBBlockTile& b_packed) const
    {
        static_assert(KIter < KIterPerWarp, "invalid interleaved B K fragment");
        static_assert(BWarpThreadSize == 4,
                      "prepared interleaved B path expects four BF16 values per fragment");

        BWarpTensor b0;
        BWarpTensor b1;
        constexpr auto spans = BWarpTensor::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx_n) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx_k) {
                constexpr auto idx = make_tuple(idx_n, idx_k);
                constexpr auto y_idx =
                    BWarpDstr{}.get_y_indices_from_distributed_indices(decltype(idx){});
                constexpr index_t fragment_offset =
                    BWarpThreadDesc{}.calculate_offset(y_idx);
                const auto x2 = bit_cast<thread_buffer<BDataType, 2>>(
                    b_packed.packed[
                        number<KIter * BWarpThreadSize + fragment_offset>{}]);
                b0(idx) = x2[number<0>{}];
                b1(idx) = x2[number<1>{}];
            });
        });

        const auto b0_packed =
            bit_cast<thread_buffer<uint32_t, BWarpThreadSize / 2>>(b0.get_thread_buffer());
        const auto b1_packed =
            bit_cast<thread_buffer<uint32_t, BWarpThreadSize / 2>>(b1.get_thread_buffer());
        static_for<0, BWarpThreadSize / 2, 1>{}([&](auto i) {
            constexpr index_t I = decltype(i)::value;
            b_packed.packed(number<KIter * BWarpThreadSize + I>{}) = b0_packed[i];
            b_packed.packed(number<KIter * BWarpThreadSize + BWarpThreadSize / 2 + I>{}) =
                b1_packed[i];
        });
    }

    template <index_t KIter, typename CBlockTensor, typename ABlockTensor>
    CK_TILE_DEVICE void RunPreparedKIter(CBlockTensor& c_block_tensor,
                                        const ABlockTensor& a_block_tensor,
                                        const InterleavedBBlockTile& b_prepared,
                                        number<KIter>) const
    {
        static_assert(KIter < KIterPerWarp, "invalid interleaved B K fragment");
        static_assert(BWarpThreadSize == 4,
                      "prepared interleaved B path expects four BF16 values per fragment");

        thread_buffer<uint32_t, BWarpThreadSize / 2> b0_packed;
        thread_buffer<uint32_t, BWarpThreadSize / 2> b1_packed;
        static_for<0, BWarpThreadSize / 2, 1>{}([&](auto i) {
            constexpr index_t I = decltype(i)::value;
            b0_packed[i] = b_prepared.packed[number<KIter * BWarpThreadSize + I>{}];
            b1_packed[i] = b_prepared.packed[
                number<KIter * BWarpThreadSize + BWarpThreadSize / 2 + I>{}];
        });

        BWarpTensor b0;
        BWarpTensor b1;
        b0.get_thread_buffer() = bit_cast<thread_buffer<BDataType, BWarpThreadSize>>(b0_packed);
        b1.get_thread_buffer() = bit_cast<thread_buffer<BDataType, BWarpThreadSize>>(b1_packed);
        RunKIterWithBWarpFragments<KIter>(c_block_tensor, a_block_tensor, b0, b1);
    }

    template <index_t KIter, typename CBlockTensor, typename ABlockTensor>
    CK_TILE_DEVICE void RunKIter(CBlockTensor& c_block_tensor,
                                const ABlockTensor& a_block_tensor,
                                const InterleavedBBlockTile& b_packed,
                                number<KIter>) const
    {
        static_assert(KIter < KIterPerWarp, "invalid interleaved B K fragment");

        BWarpTensor b0;
        BWarpTensor b1;
        constexpr auto spans = BWarpTensor::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx_n) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx_k) {
                constexpr auto idx = make_tuple(idx_n, idx_k);
                constexpr auto y_idx =
                    BWarpDstr{}.get_y_indices_from_distributed_indices(decltype(idx){});
                constexpr index_t fragment_offset =
                    BWarpThreadDesc{}.calculate_offset(y_idx);
                const auto x2 = bit_cast<thread_buffer<BDataType, 2>>(
                    b_packed.packed[
                        number<KIter * BWarpThreadSize + fragment_offset>{}]);
                b0(idx) = x2[number<0>{}];
                b1(idx) = x2[number<1>{}];
            });
        });

        RunKIterWithBWarpFragments<KIter>(c_block_tensor, a_block_tensor, b0, b1);
    }
};

template <typename Problem_, typename Policy_>
struct SlaLocalBlockGemmASmemBRegCReg
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    CK_TILE_DEVICE static constexpr auto MakeBBlockTile()
    {
        return SlaLocalBlockGemmARegBRegCReg<Problem, Policy>::MakeBBlockTile();
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        return SlaLocalBlockGemmARegBRegCReg<Problem, Policy>::MakeCBlockTile();
    }

    template <typename CBlockTensor>
    CK_TILE_DEVICE auto MakeOuputLayout(const CBlockTensor& c_block_tensor) const
    {
        return SlaLocalBlockGemmARegBRegCReg<Problem, Policy>{}.MakeOuputLayout(c_block_tensor);
    }

    template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockTensorTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockWindowTmp& a_block_window_tmp,
                                   const BBlockTensorTmp& b_block_tensor_tmp) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockWindowTmp::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BBlockTensorTmp::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockTensorTmp{}.get_lengths()[number<0>{}];
        constexpr index_t KPerBlock = BBlockTensorTmp{}.get_lengths()[number<1>{}];

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
        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iMWarp = get_warp_id() / NWarp;

        auto a_warp_window_tmp = make_tile_window(
            a_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            a_block_window_tmp.get_window_origin() + multi_index<2>{iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;
                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        using BWarpDstr   = typename WG::BWarpDstr;
        using CWarpDstr   = typename WG::CWarpDstr;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor_tmp.get_y_sliced_thread_data(
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
};

} // namespace ck_tile
