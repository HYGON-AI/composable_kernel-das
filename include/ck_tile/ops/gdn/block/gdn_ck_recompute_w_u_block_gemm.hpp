// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Modified by Hygon Information Technology Co., Ltd.
#pragma once
//
// GDN recompute_w_u custom block GEMM policies.
// Local CK tile block GEMM helper adapted for recompute_w_u shapes.
//
// Recompute GEMM: [M=64, K=64] x [K=64, N=64] -> [M=64, N=64].

#include "ck_tile/ops/gdn/pipeline/gdn_ck_recompute_w_u_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"


namespace ck_tile {

template <typename DataType>
struct GdnRecomputeWUDsreadmMN32K16
{
    using RawType = DataType;
    using RetType = ext_vector_t<DataType, 8>;

    static constexpr index_t kMN = 32;
    static constexpr index_t kK  = 16;

    static constexpr index_t kMNLoadLane    = 4;
    static constexpr index_t kKLoadLane     = 16;
    static constexpr index_t kMNLoadPerLane = 8;
    static constexpr index_t kKLoadPerLane  = 1;

    static constexpr index_t kMNStoreLane     = 16;
    static constexpr index_t kKStoreLane      = 4;
    static constexpr index_t kMN0StorePerLane = 2;
    static constexpr index_t kMN1StorePerLane = 1;
    static constexpr index_t kKStorePerLane   = 4;
    static constexpr index_t kMNInterleave    = 1;

    CK_TILE_DEVICE void operator()(const RawType* lds_ptr, RetType& ret) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
        if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            ret = bit_cast<RetType>(__builtin_hcu_ds_read_m32x16_bf16(
                reinterpret_cast<short*>(const_cast<RawType*>(lds_ptr))));
        }
        else
        {
            ret = bit_cast<RetType>(__builtin_hcu_ds_read_m32x16_f16(
                reinterpret_cast<__fp16*>(const_cast<RawType*>(lds_ptr))));
        }
#else
        detail::swallow{lds_ptr, ret};
#endif
    }
};

// Policy adapter for block GEMM.

template <typename BlockWarps_, typename WarpGemm_>
struct GdnRecomputeWUARegBSmemPolicy
{
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpGemm   = remove_cvref_t<WarpGemm_>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemm{},
                          BlockWarps::at(number<0>{}),
                          BlockWarps::at(number<1>{}));
    }
};


// ARegBReg block GEMM. A and B are loaded into CK distributed tensors before
// the warp MMAC calls.

template <typename Problem_, typename Policy_>
struct GdnRecomputeWUBlockGemmARegBReg
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

    template <typename BLdsView>
    CK_TILE_DEVICE auto LoadBByDsreadm(const BLdsView& b_lds_view) const
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t KIterPerWarp = BlockGemmShape::kK / WG::kK;
        using BVec8 = ext_vector_t<BDataType, 8>;

        statically_indexed_array<typename WG::BWarpTensor, KIterPerWarp> b_warp_tensors;
        const index_t lane = get_lane_id();
        const index_t n_lane_offset = (lane % 4) * 8;
        const index_t k_lane_offset = lane / 4;
        const BDataType* b_lds = b_lds_view.get_buffer_view().p_data_;

        static_for<0, KIterPerWarp, 1>{}([&](auto k_iter) {
            constexpr index_t k_base = k_iter * WG::kK;
            const BDataType* lane_ptr =
                b_lds + (k_base + k_lane_offset) * BlockGemmShape::kN + n_lane_offset;
            BVec8 b_lo;
            BVec8 b_hi;
            GdnRecomputeWUDsreadmMN32K16<BDataType>{}(lane_ptr, b_lo);
            GdnRecomputeWUDsreadmMN32K16<BDataType>{}(lane_ptr + 32, b_hi);
            b_warp_tensors(k_iter).get_thread_buffer().template get_as<BVec8>()(number<0>{}) =
                b_lo;
            b_warp_tensors(k_iter).get_thread_buffer().template get_as<BVec8>()(number<1>{}) =
                b_hi;
        });
        return b_warp_tensors;
    }

    template <typename CBlockTensor, typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
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

        using AWarpDstr = typename WG::AWarpDstr;
        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
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


    template <typename CBlockTensor, typename ABlockTensor, typename BLdsView>
    CK_TILE_DEVICE void RunWithStreamingDsreadmB(CBlockTensor& c_block_tensor,
                                                 const ABlockTensor& a_block_tensor,
                                                 const BLdsView& b_lds_view) const
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t KPerBlock = BlockGemmShape::kK;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        using AWarpDstr   = typename WG::AWarpDstr;
        using CWarpDstr   = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;
        using BVec8       = ext_vector_t<BDataType, 8>;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        const index_t lane = get_lane_id();
        const index_t n_lane_offset = (lane % 4) * 8;
        const index_t k_lane_offset = lane / 4;
        const BDataType* b_lds = b_lds_view.get_buffer_view().p_data_;

        static_for<0, KIterPerWarp, 1>{}([&](auto k_iter) {
            constexpr index_t k_base = k_iter * WG::kK;
            const BDataType* lane_ptr =
                b_lds + (k_base + k_lane_offset) * BlockGemmShape::kN + n_lane_offset;

            BVec8 b_lo;
            BVec8 b_hi;
            GdnRecomputeWUDsreadmMN32K16<BDataType>{}(lane_ptr, b_lo);
            GdnRecomputeWUDsreadmMN32K16<BDataType>{}(lane_ptr + 32, b_hi);
            BWarpTensor b_warp_tensor;
            b_warp_tensor.get_thread_buffer().template get_as<BVec8>()(number<0>{}) = b_lo;
            b_warp_tensor.get_thread_buffer().template get_as<BVec8>()(number<1>{}) = b_hi;

            static_for<0, MIterPerWarp, 1>{}([&](auto m_iter) {
                AWarpTensor a_warp_tensor;
                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<m_iter, k_iter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<m_iter, 0>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<m_iter, 0>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.get_thread_buffer());
            });
        });
    }

    template <typename CBlockTensor, typename ABlockTensor, typename BLdsView>
    CK_TILE_DEVICE void RunLowerWithPrefetch2DsreadmB(CBlockTensor& c_block_tensor,
                                                      const ABlockTensor& a_block_tensor,
                                                      const BLdsView& b_lds_view) const
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = BlockGemmShape::kK / WG::kK;
        static_assert(KIterPerWarp % 2 == 0);

        using AWarpDstr   = typename WG::AWarpDstr;
        using CWarpDstr   = typename WG::CWarpDstr;
        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;
        using BVec8       = ext_vector_t<BDataType, 8>;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        const index_t lane = get_lane_id();
        const index_t warp = get_warp_id();
        const index_t n_lane_offset = (lane % 4) * 8;
        const index_t k_lane_offset = lane / 4;
        const BDataType* b_lds = b_lds_view.get_buffer_view().p_data_;

        auto run_k = [&](auto k_iter, const BWarpTensor& b_warp_tensor) {
            static_for<0, MIterPerWarp, 1>{}([&](auto m_iter) {
                AWarpTensor a_warp_tensor;
                a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<m_iter, k_iter>{}, a_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<m_iter, 0>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<m_iter, 0>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.get_thread_buffer());
            });
        };

        auto run_pair = [&](auto k_iter0, auto has_second) {
            constexpr index_t k_iter1 = decltype(k_iter0)::value + 1;
            const BDataType* lane_ptr0 =
                b_lds + (k_iter0 * WG::kK + k_lane_offset) * BlockGemmShape::kN +
                n_lane_offset;
            const BDataType* lane_ptr1 =
                b_lds + (k_iter1 * WG::kK + k_lane_offset) * BlockGemmShape::kN +
                n_lane_offset;

            BVec8 b0_lo;
            BVec8 b0_hi;
            BVec8 b1_lo;
            BVec8 b1_hi;
            BWarpTensor b0;
            BWarpTensor b1;
            GdnRecomputeWUDsreadmMN32K16<BDataType>{}(lane_ptr0, b0_lo);
            GdnRecomputeWUDsreadmMN32K16<BDataType>{}(lane_ptr0 + 32, b0_hi);
            b0.get_thread_buffer().template get_as<BVec8>()(number<0>{}) = b0_lo;
            b0.get_thread_buffer().template get_as<BVec8>()(number<1>{}) = b0_hi;
            if constexpr(decltype(has_second)::value)
            {
                GdnRecomputeWUDsreadmMN32K16<BDataType>{}(lane_ptr1, b1_lo);
                GdnRecomputeWUDsreadmMN32K16<BDataType>{}(lane_ptr1 + 32, b1_hi);
                b1.get_thread_buffer().template get_as<BVec8>()(number<0>{}) = b1_lo;
                b1.get_thread_buffer().template get_as<BVec8>()(number<1>{}) = b1_hi;
            }

            run_k(k_iter0, b0);
            if constexpr(decltype(has_second)::value)
            {
                run_k(number<k_iter1>{}, b1);
            }
        };

        auto run_count = [&](auto k_count) {
            constexpr index_t KCount = decltype(k_count)::value;
            run_pair(number<0>{}, bool_constant<(KCount > 1)>{});
            if constexpr(KCount > 2)
            {
                run_pair(number<2>{}, bool_constant<(KCount > 3)>{});
            }
        };

        if(warp == 0)
            run_count(number<1>{});
        else if(warp == 1)
            run_count(number<2>{});
        else if(warp == 2)
            run_count(number<3>{});
        else
            run_count(number<4>{});
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

} // namespace ck_tile
