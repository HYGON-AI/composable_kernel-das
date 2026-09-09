// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// A is block distributed tensor
// B is block distributed tensor
// C is block distributed tensor

namespace detail {

template <typename T, typename = void>
struct has_scale_support : std::false_type
{
};

template <typename T>
struct has_scale_support<T, std::void_t<typename T::AScaleType, typename T::BScaleType>>
    : std::true_type
{
};

template <typename T, typename = void>
struct use_direct_c_buffer_mmac : std::false_type
{
};

template <typename T>
struct use_direct_c_buffer_mmac<T, std::void_t<decltype(T::DirectCBufferMmac)>>
    : std::bool_constant<T::DirectCBufferMmac>
{
};

// template <bool Enabled, typename WarpGemm>
// struct warp_scale_traits
// {
//     using AScaleVec = ck_tile::null_type;
//     using BScaleVec = ck_tile::null_type;
// };

// template <typename WarpGemm>
// struct warp_scale_traits<true, WarpGemm>
// {
//     using AScaleVec = typename WarpGemm::AScaleVec;
//     using BScaleVec = typename WarpGemm::BScaleVec;
// };

} // namespace detail

template <typename Problem_,
          typename Policy_,
          bool TransposeC_ = false>
struct BlockGemmARegBRegCRegV1
{
    private:
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem        = remove_cvref_t<PipelineProblem_>;
        using Policy         = remove_cvref_t<GemmPolicy_>;
        using ADataType      = remove_cvref_t<typename Problem::ADataType>;
        using BDataType      = remove_cvref_t<typename Problem::BDataType>;
        using CDataType      = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;

        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

        static constexpr bool UseABScale = detail::has_scale_support<WarpGemm>::value;

        static constexpr index_t MWarp        = config.template at<1>();    // 2
        static constexpr index_t NWarp        = config.template at<2>();    // 2
        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);     //128/(2*32)=2
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);     //128/(2*32)=2
        static constexpr index_t KIterPerWarp = KPerBlock * ck_tile::numeric_traits<ADataType>::PackedSize / WarpGemm::kK;               // e.g. pk_int4: 64*2/64=2

        static constexpr index_t KPack = WarpGemm::kKPerThread;
    };

    public:
    using Problem                    = remove_cvref_t<Problem_>;
    using Policy                     = remove_cvref_t<Policy_>;
    static constexpr bool TransposeC = TransposeC_;

    using Traits = GemmTraits_<Problem, Policy>;

    using WarpGemm       = typename Traits::WarpGemm;
    using BlockGemmShape = typename Traits::BlockGemmShape;

    using ADataType = remove_cvref_t<typename Traits::ADataType>;
    using BDataType = remove_cvref_t<typename Traits::BDataType>;
    using CDataType = remove_cvref_t<typename Traits::CDataType>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr bool UseABScale = Traits::UseABScale;

    static constexpr index_t MWarp            = Traits::MWarp;
    static constexpr index_t NWarp            = Traits::NWarp;
    static constexpr bool UseDefaultScheduler = (Problem::NumWaveGroups != 1);  //False (Problem::NumWaveGroups = TileGemmUniversalTraits::NumWaveGroups == 1)

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto a_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<NWarp>,
                                           tuple<sequence<MIterPerWarp>, sequence<KIterPerWarp>>,
                                           tuple<>,
                                           tuple<>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{};

            constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

            return a_block_dstr_encode;
        }
        else
        {
            constexpr auto a_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<NWarp>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

            return a_block_dstr_encode;
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode()
    {
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto b_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<MWarp>,
                                           tuple<sequence<NIterPerWarp>, sequence<KIterPerWarp>>,
                                           tuple<>,
                                           tuple<>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{};
            constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

            return b_block_dstr_encode;
        }
        else
        {
            constexpr auto b_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<0, 1>>,
                tuple<sequence<0, 1>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                b_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

            return b_block_dstr_encode;
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockDistributionEncode()
    {
        using c_distr_ys_major = std::conditional_t<TransposeC, sequence<2, 1>, sequence<1, 2>>;
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
                tuple<>,
                tuple<>,
                c_distr_ys_major,
                sequence<0, 0>>{};
            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

            return c_block_dstr_encode;
        }
        else
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 1>>,
                c_distr_ys_major,
                sequence<0, 0>>{};
            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

            return c_block_dstr_encode;
        }
    }

    template <bool Enabled = UseABScale, typename = std::enable_if_t<Enabled>>
    CK_TILE_DEVICE static constexpr auto MakeAScaleBlockDistributionEncode()
    {
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto a_scale_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<NWarp>,
                                           tuple<sequence<MIterPerWarp>, sequence<1>>,
                                           tuple<>,
                                           tuple<>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{};

            constexpr auto a_scale_block_dstr_encode =
                detail::make_embed_tile_distribution_encoding(
                    a_scale_block_outer_dstr_encoding,
                    typename WarpGemm::AScaleWarpDstrEncoding{});

            return a_scale_block_dstr_encode;
        }
        else
        {
            constexpr auto a_scale_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<NWarp>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<1, 2>,
                sequence<0, 0>>{};

            constexpr auto a_scale_block_dstr_encode =
                detail::make_embed_tile_distribution_encoding(
                    a_scale_block_outer_dstr_encoding,
                    typename WarpGemm::AScaleWarpDstrEncoding{});

            return a_scale_block_dstr_encode;
        }
    }

    template <bool Enabled = UseABScale, typename = std::enable_if_t<Enabled>>
    CK_TILE_DEVICE static constexpr auto MakeBScaleBlockDistributionEncode()
    {
        if constexpr(UseDefaultScheduler)
        {
            constexpr auto b_scale_block_outer_dstr_encoding =
                tile_distribution_encoding<sequence<MWarp>,
                                           tuple<sequence<NIterPerWarp>, sequence<1>>,
                                           tuple<>,
                                           tuple<>,
                                           sequence<1, 2>,
                                           sequence<0, 0>>{};

            constexpr auto b_scale_block_dstr_encode =
                detail::make_embed_tile_distribution_encoding(
                    b_scale_block_outer_dstr_encoding,
                    typename WarpGemm::BScaleWarpDstrEncoding{});

            return b_scale_block_dstr_encode;
        }
        else
        {
            constexpr auto b_scale_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<0, 1>>,
                tuple<sequence<0, 1>>,
                sequence<1, 2>,
                sequence<0, 0>>{};

            constexpr auto b_scale_block_dstr_encode =
                detail::make_embed_tile_distribution_encoding(
                    b_scale_block_outer_dstr_encoding,
                    typename WarpGemm::BScaleWarpDstrEncoding{});

            return b_scale_block_dstr_encode;
        }
    }

    template <typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename AScaleBlockTensor,
              typename BScaleBlockTensor>
    CK_TILE_DEVICE void ComputeGemm(CBlockTensor& c_block_tensor,
                                    const ABlockTensor& a_block_tensor,
                                    const BBlockTensor& b_block_tensor,
                                    const AScaleBlockTensor* a_scale_block_tensor,
                                    const BScaleBlockTensor* b_scale_block_tensor) const
    {
        static_assert(std::is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                          std::is_same_v<BDataType, remove_cv_t<typename BBlockTensor::DataType>> &&
                          std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        // check ABC-block-distribution
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeABlockDistributionEncode())>,
                           remove_cvref_t<decltype(ABlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "A distribution is wrong!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeBBlockDistributionEncode())>,
                           remove_cvref_t<decltype(BBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "B distribution is wrong!");
        static_assert(
            std::is_same_v<remove_cvref_t<decltype(MakeCBlockDistributionEncode())>,
                           remove_cvref_t<decltype(CBlockTensor::get_tile_distribution()
                                                       .get_static_tile_distribution_encoding())>>,
            "C distribution is wrong!");

        using AWarpDstr   = typename WarpGemm::AWarpDstr;
        using BWarpDstr   = typename WarpGemm::BWarpDstr;
        using CWarpDstr   = typename WarpGemm::CWarpDstr;
        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_warp_y_lengths =
            to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto b_warp_y_lengths =
            to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        constexpr bool kHasScale = UseABScale;

        // using ScaleTraits = detail::warp_scale_traits<kHasScale, WarpGemm>;
        // using AScaleVec   = typename ScaleTraits::AScaleVec;
        // using BScaleVec   = typename ScaleTraits::BScaleVec;

        constexpr auto a_scale_warp_y_lengths = [&]() {
            if constexpr(kHasScale)
            {
                using AScaleWarpDstr = typename WarpGemm::AScaleWarpDstr;
                return to_sequence(AScaleWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();

        constexpr auto b_scale_warp_y_lengths = [&]() {
            if constexpr(kHasScale)
            {
                using BScaleWarpDstr = typename WarpGemm::BScaleWarpDstr;
                return to_sequence(BScaleWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();

        constexpr auto a_scale_warp_y_index_zeros = [&]() {
            if constexpr(kHasScale)
            {
                using AScaleWarpDstr = typename WarpGemm::AScaleWarpDstr;
                return uniform_sequence_gen_t<AScaleWarpDstr::NDimY, 0>{};
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();

        constexpr auto b_scale_warp_y_index_zeros = [&]() {
            if constexpr(kHasScale)
            {
                using BScaleWarpDstr = typename WarpGemm::BScaleWarpDstr;
                return uniform_sequence_gen_t<BScaleWarpDstr::NDimY, 0>{};
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();

        constexpr auto a_scale_block_encode = [&]() {
            if constexpr(kHasScale)
            {
                return MakeAScaleBlockDistributionEncode();
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();

        constexpr auto b_scale_block_encode = [&]() {
            if constexpr(kHasScale)
            {
                return MakeBScaleBlockDistributionEncode();
            }
            else
            {
                return ck_tile::null_type{};
            }
        }();

        if constexpr(kHasScale)
        {
            static_assert(!std::is_same_v<AScaleBlockTensor, ck_tile::null_type> &&
                              !std::is_same_v<BScaleBlockTensor, ck_tile::null_type>,
                          "Scale tensors must be provided when UseABScale is enabled");

            static_assert(std::is_same_v<remove_cvref_t<decltype(a_scale_block_encode)>,
                                         remove_cvref_t<decltype(AScaleBlockTensor::get_tile_distribution()
                                                                     .get_static_tile_distribution_encoding())>>,
                          "A scale distribution is wrong!");

            static_assert(std::is_same_v<remove_cvref_t<decltype(b_scale_block_encode)>,
                                         remove_cvref_t<decltype(BScaleBlockTensor::get_tile_distribution()
                                                                     .get_static_tile_distribution_encoding())>>,
                          "B scale distribution is wrong!");
        }

        // constexpr auto I0 = number<0>{};

        constexpr bool kDirectInterleavedNativeMmac =
            !kHasScale && detail::use_direct_c_buffer_mmac<Policy>::value &&
            !TransposeC && KIterPerWarp == 1 && WarpGemm::kM == 16 &&
            WarpGemm::kN == 16 && WarpGemm::kK == 16;

        if constexpr(kDirectInterleavedNativeMmac)
        {
            // Triton's warpsPerCTA=[2,4] lowering keeps the native M16xN16
            // MMAC tile and interleaves the wave coordinate between eight M
            // repeats and four N repeats. The block-distributed thread
            // buffers already have exactly that [m-repeat,n-repeat] order, so
            // update their base MMAC vectors directly. This avoids creating
            // and copying 32 one-MMAC AWarp/BWarp/CWarp tensor slices while
            // the 128-FP32 accumulator is live.
            using WarpAttr = typename WarpGemm::WarpGemmAttribute;
            using Impl     = typename WarpAttr::Impl;
            static_assert(WarpAttr::kM == Impl::kM &&
                              WarpAttr::kN == Impl::kN &&
                              WarpAttr::kK == Impl::kK &&
                              MIterPerWarp == 8 && NIterPerWarp == 4,
                          "native interleaved direct MMAC requires Triton's "
                          "eight-M/four-N repeat geometry");

            auto& c_vecs = c_block_tensor.get_thread_buffer()
                               .template get_as<typename Impl::CVecType>();
            const auto& a_vecs = a_block_tensor.get_thread_buffer()
                                     .template get_as<typename Impl::AVecType>();
            const auto& b_vecs = b_block_tensor.get_thread_buffer()
                                     .template get_as<typename Impl::BVecType>();

            static_for<0, MIterPerWarp, 1>{}([&](auto iMRepeat) {
                static_for<0, NIterPerWarp, 1>{}([&](auto iNRepeat) {
                    constexpr auto c_index =
                        iMRepeat * number<NIterPerWarp>{} + iNRepeat;
                    Impl{}(c_vecs[c_index], a_vecs[iMRepeat], b_vecs[iNRepeat]);
                });
            });
        }
        else
        {
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

                    if constexpr(kHasScale)
                    {
                        using c_iter_idx = std::conditional_t<TransposeC,
                                                              sequence<nIter, mIter>,
                                                              sequence<mIter, nIter>>;
                        CWarpTensor c_warp_tensor;
                        c_warp_tensor.get_thread_buffer() =
                            c_block_tensor.get_y_sliced_thread_data(
                                merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // const auto a_scale_thread_buffer = a_scale_block_tensor->get_y_sliced_thread_data(
                        //     merge_sequences(sequence<mIter, kIter>{}, a_scale_warp_y_index_zeros),
                        //     merge_sequences(sequence<1, 1>{}, a_scale_warp_y_lengths));

                        // const auto b_scale_thread_buffer = b_scale_block_tensor->get_y_sliced_thread_data(
                        //     merge_sequences(sequence<nIter, kIter>{}, b_scale_warp_y_index_zeros),
                        //     merge_sequences(sequence<1, 1>{}, b_scale_warp_y_lengths));

                        // const auto a_scale_vec =
                        //     a_scale_tensor.template get_as<AScaleVec>()[I0];
                        // const auto b_scale_vec =
                        //     b_scale_tensor.template get_as<BScaleVec>()[I0];

                        using AScaleTensor = typename WarpGemm::AScaleWarpTensor;
                        using BScaleTensor = typename WarpGemm::BScaleWarpTensor;

                        AScaleTensor a_scale_tensor;
                        BScaleTensor b_scale_tensor;

                        a_scale_tensor.get_thread_buffer() = a_scale_block_tensor->get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, kIter>{}, a_scale_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_scale_warp_y_lengths));

                        b_scale_tensor.get_thread_buffer() = b_scale_block_tensor->get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_scale_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_scale_warp_y_lengths));

                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor, a_scale_tensor, b_scale_tensor);
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    }
                    else if constexpr(detail::use_direct_c_buffer_mmac<Policy>::value)
                    {
                        // Phase-39 gfx936 operand-layout path. Avoid constructing a
                        // 32-register wide CWarpTensor beside the long-lived 128-VGPR
                        // block accumulator. The embedded non-transposed C distribution
                        // is outer-[m,n], inner-[MRepeat,NRepeat], so its CVec storage can
                        // be updated in place by the base MMAC implementation.
                        static_assert(!TransposeC && MIterPerWarp == 1 &&
                                          NIterPerWarp == 1 &&
                                          WarpGemm::kM == 128 && WarpGemm::kN == 64 &&
                                          (WarpGemm::kK == 16 || WarpGemm::kK == 32),
                                      "direct C-buffer MMAC is specialized for the "
                                      "gfx936 M128xN64 RC operand probes");
                        using WarpAttr = typename WarpGemm::WarpGemmAttribute;
                        using Impl     = typename WarpAttr::Impl;
                        constexpr index_t kKRepeat = WarpAttr::kK / Impl::kK;
                        constexpr index_t kMRepeat = WarpAttr::kM / Impl::kM;
                        constexpr index_t kNRepeat = WarpAttr::kN / Impl::kN;
                        static_assert((kKRepeat == 1 || kKRepeat == 2) &&
                                          kMRepeat == 8 && kNRepeat == 4 &&
                                          WarpAttr::kM == kMRepeat * Impl::kM &&
                                          WarpAttr::kN == 4 * Impl::kN,
                                      "unexpected MMAC repeat shape");

                        auto& c_vecs = c_block_tensor.get_thread_buffer()
                                           .template get_as<typename Impl::CVecType>();
                        const auto& a_vecs = a_warp_tensor.get_thread_buffer()
                                                 .template get_as<typename Impl::AVecType>();
                        const auto& b_vecs = b_warp_tensor.get_thread_buffer()
                                                 .template get_as<typename Impl::BVecType>();

                        static_for<0, kMRepeat, 1>{}([&](auto iMRepeat) {
                            static_for<0, kNRepeat, 1>{}([&](auto iNRepeat) {
                                static_for<0, kKRepeat, 1>{}([&](auto iKRepeat) {
                                    constexpr auto c_index =
                                        iMRepeat * number<kNRepeat>{} + iNRepeat;
                                    Impl{}(c_vecs[c_index],
                                           a_vecs[iMRepeat * number<kKRepeat>{} + iKRepeat],
                                           b_vecs[iNRepeat * number<kKRepeat>{} + iKRepeat]);
                                });
                            });
                        });
                    }
                    else
                    {
                        using c_iter_idx = std::conditional_t<TransposeC,
                                                              sequence<nIter, mIter>,
                                                              sequence<mIter, nIter>>;
                        CWarpTensor c_warp_tensor;
                        c_warp_tensor.get_thread_buffer() =
                            c_block_tensor.get_y_sliced_thread_data(
                                merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    }
                });
            });
        });
        }
    }

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
    {
        ComputeGemm(c_block_tensor,
                    a_block_tensor,
                    b_block_tensor,
                    static_cast<const ck_tile::null_type*>(nullptr),
                    static_cast<const ck_tile::null_type*>(nullptr));
    }

    template <typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename AScaleBlockTensor,
              typename BScaleBlockTensor,
              bool Enabled = UseABScale>
    CK_TILE_DEVICE std::enable_if_t<Enabled, void>
    operator()(CBlockTensor& c_block_tensor,
               const ABlockTensor& a_block_tensor,
               const BBlockTensor& b_block_tensor,
               const AScaleBlockTensor& a_scale_block_tensor,
               const BScaleBlockTensor& b_scale_block_tensor) const
    {
        ComputeGemm(c_block_tensor,
                    a_block_tensor,
                    b_block_tensor,
                    &a_scale_block_tensor,
                    &b_scale_block_tensor);
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        using c_distr_ys_major = std::conditional_t<TransposeC, sequence<2, 1>, sequence<1, 2>>;
        if constexpr(UseDefaultScheduler)   //false
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<MWarp>,
                tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
                tuple<>,
                tuple<>,
                c_distr_ys_major,
                sequence<0, 0>>{};

            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
            constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
            auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
            return c_block_tensor;
        }
        else
        {
            constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 1>>,
                c_distr_ys_major,
                sequence<0, 0>>{};

            constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
            constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
            auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
            return c_block_tensor;
        }
    }

    // Convert the MMAC accumulator register layout into the logical output
    // layout without using LDS. The base MMAC instruction exposes C as
    // [N-interleave, N0, N1] register fragments, while raw global stores
    // require [N0, lane-N, N1]. CShuffle performs this conversion implicitly;
    // the Triton-layout direct-store probe needs it explicitly.
    template <typename CBlockTensor>
    CK_TILE_DEVICE static constexpr auto MakeCOutputBlockTile(const CBlockTensor& c_block_tensor)
    {
            // The selected gfx936 policy uses a TransC MMAC warp attribute
            // even though the block-level TransposeC template argument is
            // false.  Detecting this conversion through the block flag is
            // therefore incorrect: use the warp attribute's explicit
            // GetCWarpPermuteTensor path and retain the block flag only for
            // the outer M/N iteration order.
            using WarpAttribute = typename WarpGemm::WarpGemmAttribute;
            using c_distr_ys_major =
                std::conditional_t<TransposeC, sequence<2, 1>, sequence<1, 2>>;
            constexpr auto c_block_output_dstr_encode = []() {
                if constexpr(UseDefaultScheduler)
                {
                    constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                        sequence<MWarp>,
                        tuple<sequence<MIterPerWarp>, sequence<NIterPerWarp, NWarp>>,
                        tuple<>,
                        tuple<>,
                        c_distr_ys_major,
                        sequence<0, 0>>{};
                    return detail::make_embed_tile_distribution_encoding(
                        c_block_outer_dstr_encoding,
                        typename WarpAttribute::CWarpPermuteDstrEncoding{});
                }
                else
                {
                    constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
                        sequence<>,
                        tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
                        tuple<sequence<1, 2>>,
                        tuple<sequence<1, 1>>,
                        c_distr_ys_major,
                        sequence<0, 0>>{};
                    return detail::make_embed_tile_distribution_encoding(
                        c_block_outer_dstr_encoding,
                        typename WarpAttribute::CWarpPermuteDstrEncoding{});
                }
            }();
            constexpr auto c_block_output_dstr =
                make_static_tile_distribution(c_block_output_dstr_encode);
            auto c_block_output_tensor =
                make_static_distributed_tensor<CDataType>(c_block_output_dstr);

            using CWarpDstr   = typename WarpGemm::CWarpDstr;
            using CWarpTensor = typename WarpGemm::CWarpTensor;
            using CWarpPermuteDstr = remove_cvref_t<decltype(make_static_tile_distribution(
                typename WarpAttribute::CWarpPermuteDstrEncoding{}))>;

            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros =
                uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_permute_y_lengths =
                to_sequence(CWarpPermuteDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_permute_y_index_zeros =
                uniform_sequence_gen_t<CWarpPermuteDstr::NDimY, 0>{};

            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    using c_iter_idx = std::conditional_t<TransposeC,
                                                          sequence<nIter, mIter>,
                                                          sequence<mIter, nIter>>;
                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() =
                        c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(c_iter_idx{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    const auto c_warp_permute_tensor =
                        WarpAttribute::GetCWarpPermuteTensor(c_warp_tensor);
                    c_block_output_tensor.set_y_sliced_thread_data(
                        merge_sequences(c_iter_idx{}, c_warp_permute_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_permute_y_lengths),
                        c_warp_permute_tensor.get_thread_buffer());
                });
            });

            return c_block_output_tensor;
    }

    // C = A * B
    template <typename ABlockTensor, typename BBlockTensor>
    CK_TILE_DEVICE auto operator()(const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor) const
    {
        static_assert(!UseABScale,
                      "Scale-aware BlockGemm requires explicit scale tensors in the call site");
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_tensor, b_block_tensor);
        return c_block_tensor;
    }
};

} // namespace ck_tile
