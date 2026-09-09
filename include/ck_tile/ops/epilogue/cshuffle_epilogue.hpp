// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

template <typename AType,
          typename BType,
          typename CType,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t KPerXdl,
          bool isCTransposed,
          index_t MRepeat = MPerXdl / 16,
          index_t NRepeat = NPerXdl / 16,
          index_t MInterleave = 1,
          index_t NInterleave = 1,
          typename Enable = void>
struct CShuffleWarpGemmSelector
{
    using Type = WarpGemmMmacDispatcher<AType,
                                        BType,
                                        CType,
                                        MPerXdl,
                                        NPerXdl,
                                        KPerXdl,
                                        isCTransposed,
                                        MRepeat,
                                        NRepeat,
                                        MInterleave,
                                        NInterleave>;
};

// this epilogue aiming to store a matrix with different layout from the shared memory to the global
// memory.
template <typename ADataType_,
          typename BDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename DsLayout_,
          typename ELayout_,
          typename CDElementwise_,
          index_t kM_,
          index_t kN_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_,
          bool isCTransposed_,
          memory_operation_enum MemoryOperation_,
          index_t kNumWaveGroups_ = 1,
          bool FixedVectorSize_   = false,
          index_t VectorSizeC_    = 1,
          bool TiledMMAPermuteN_  = false,
          index_t WarpGemmMRepeat_ = MPerXdl_ / 16,
          index_t WarpGemmNRepeat_ = NPerXdl_ / 16,
          index_t WarpGemmMInterleave_ = 1,
          index_t WarpGemmNInterleave_ = 1>
struct CShuffleEpilogueProblem
{
    using ADataType                                        = remove_cvref_t<ADataType_>;
    using BDataType                                        = remove_cvref_t<BDataType_>;
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;        //half
    using DsDataType                                       = remove_cvref_t<DsDataType_>;
    using DsLayout                                         = remove_cvref_t<DsLayout_>;
    using ELayout                                          = remove_cvref_t<ELayout_>;
    using CDElementwise                                    = remove_cvref_t<CDElementwise_>;
    static constexpr index_t kBlockSize                    = MWave_ * NWave_ * get_warp_size();
    static constexpr index_t kMPerBlock                    = kM_;
    static constexpr index_t kNPerBlock                    = kN_;
    static constexpr index_t MWave                         = MWave_;        //2
    static constexpr index_t NWave                         = NWave_;        //2
    static constexpr index_t MPerXdl                       = MPerXdl_;      //32
    static constexpr index_t NPerXdl                       = NPerXdl_;      //32
    static constexpr index_t KPerXdl                       = KPerXdl_;      //16 for half
    static constexpr index_t isCTransposed                 = isCTransposed_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
    static constexpr bool FixedVectorSize                  = FixedVectorSize_;      //false
    static constexpr index_t VectorSizeC                   = VectorSizeC_;
    static constexpr bool TiledMMAPermuteN                 = TiledMMAPermuteN_;
    static constexpr index_t WarpGemmMRepeat               = WarpGemmMRepeat_;
    static constexpr index_t WarpGemmNRepeat               = WarpGemmNRepeat_;
    static constexpr index_t WarpGemmMInterleave           = WarpGemmMInterleave_;
    static constexpr index_t WarpGemmNInterleave           = WarpGemmNInterleave_;
    static constexpr index_t kNumWaveGroups                = kNumWaveGroups_;
    static constexpr index_t NumDTensor                    = DsDataType::size();

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
};

template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using ADataType   = remove_cvref_t<typename Problem::ADataType>;
    using BDataType   = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    using DsDataType  = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout    = remove_cvref_t<typename Problem::DsLayout>;
#if CK_TILE_ARCH_SUPPORTS_INT4
    using ATypeToUse =
        std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
#else
    using ATypeToUse = ADataType;
    using BTypeToUse = BDataType;
#endif
    using ELayout       = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise = remove_cvref_t<typename Problem::CDElementwise>;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;
    static constexpr index_t kBlockSize                    = Problem::kBlockSize;
    static constexpr index_t kMPerBlock                    = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock                    = Problem::kNPerBlock;
    static constexpr index_t MWave                         = Problem::MWave;        //2
    static constexpr index_t NWave                         = Problem::NWave;        //2
    static constexpr index_t MPerXdl                       = Problem::MPerXdl;      //32
    static constexpr index_t NPerXdl                       = Problem::NPerXdl;      //32
    static constexpr index_t KPerXdl                       = Problem::KPerXdl;      //16 for half
    static constexpr index_t isCTransposed                 = Problem::isCTransposed;
    static constexpr bool FixedVectorSize                  = Problem::FixedVectorSize;  //false
    static constexpr bool TiledMMAPermuteN                 = Problem::TiledMMAPermuteN;
    static constexpr index_t VectorSizeC                   = Problem::VectorSizeC;
    static constexpr index_t WarpGemmMRepeat               = Problem::WarpGemmMRepeat;
    static constexpr index_t WarpGemmNRepeat               = Problem::WarpGemmNRepeat;
    static constexpr index_t WarpGemmMInterleave           = Problem::WarpGemmMInterleave;
    static constexpr index_t WarpGemmNInterleave           = Problem::WarpGemmNInterleave;
    static constexpr index_t MPerIteration                 = MPerXdl * MWave;       //64
    static constexpr index_t NPerIteration                 = NPerXdl * NWave;       //64
    static constexpr index_t NumDTensor                    = Problem::NumDTensor;
    static constexpr index_t MRepeat                       = kMPerBlock / (MPerXdl * MWave);
    static constexpr index_t NRepeat                       = kNPerBlock / (NPerXdl * NWave);

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
    /**
     * @brief Get the vector store size for C tensor.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeC;
        }
        constexpr index_t max_vector_size = 16;
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));     // min(64, 16/4=4)=4
        }
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    /**
     * @brief Get the vector store size for Di tensor.
     *
     * @return The vector store size for Di tensor.
     */
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> index)
    {
        constexpr index_t max_vector_size = 16;
        using DiDataType = remove_cvref_t<std::tuple_element_t<index.value, DsDataType>>;
        using DiLayout   = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
        if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else
        {
            static_assert(false, "Unsupported DLayout!");
        }
        return max_vector_size / sizeof(DiDataType);
    }
    /**
     * @brief Shuffle tile configuration parameters
     *
     * @details These parameters control the number of XDL tiles processed per wave in each shuffle
     * iteration:
     * - NumMXdlPerWavePerShuffle: Number of XDL tiles in M dimension processed per wave
     * - NumNXdlPerWavePerShuffle: Number of XDL tiles in N dimension processed per wave
     */
    static constexpr auto shuffle_tile_tuple = [] {
#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_MERGE_M2)
        // After the GEMM K loop the 64-KiB operand allocation is dead.  Fold
        // two adjacent M warp tiles into one CShuffle pass so a 256x256 BF16
        // output needs two 64-KiB passes instead of four 32-KiB passes.
        // Keep this behind a tuning gate until resource use, correctness, and
        // gfx936 latency are validated.
        static_assert(kMPerBlock % (2 * MPerXdl * MWave) == 0);
        return std::make_tuple(2, 1);
#else
        constexpr index_t elem_per_thread = MPerXdl * NPerXdl / get_warp_size();    //32*32/64=16
        if constexpr(elem_per_thread >= GetVectorSizeC())   // 16 >= 8
        {
            return std::make_tuple(1, 1);
        }
        else
        {
            constexpr index_t num_xdl_shuffles = GetVectorSizeC() / elem_per_thread;
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                static_assert((kMPerBlock % (MPerXdl * MWave) == 0) &&
                                  (kMPerBlock % num_xdl_shuffles == 0),
                              "kMPerBlock must be divisible by MPerXdl*MWave and "
                              "num_xdl_shuffles for CShuffleEpilogue");
                return std::make_tuple(min(num_xdl_shuffles, kMPerBlock / (MPerXdl * MWave)), 1);
            }
            else
            {
                static_assert((kNPerBlock % (NPerXdl * NWave) == 0) &&
                                  (kNPerBlock % num_xdl_shuffles == 0),
                              "kNPerBlock must be divisible by NPerXdl*NWave and "
                              "num_xdl_shuffles for CShuffleEpilogue");
                return std::make_tuple(1, min(num_xdl_shuffles, kNPerBlock / (NPerXdl * NWave)));
            }
        }
#endif
    }();
    static constexpr index_t NumMXdlPerWavePerShuffle = std::get<0>(shuffle_tile_tuple);
    static constexpr index_t NumNXdlPerWavePerShuffle = std::get<1>(shuffle_tile_tuple);

    static constexpr auto MNPerIterationShuffle = [] {
        constexpr index_t m_val = MPerXdl * MWave * NumMXdlPerWavePerShuffle;   // 32 * 2 * 1 = 64
        constexpr index_t n_val = NPerXdl * NWave * NumNXdlPerWavePerShuffle;   // 32 * 2 * 1 = 64
        if constexpr(kMPerBlock % m_val != 0 || kNPerBlock % n_val != 0)
            return std::make_tuple(MPerXdl * MWave, NPerXdl * NWave);           // 64 * 2 = 128, 64 * 2 = 128
        else
            return std::make_tuple(m_val, n_val);
    }();
    static constexpr index_t MPerIterationShuffle = std::get<0>(MNPerIterationShuffle);     //64
    static constexpr index_t NPerIterationShuffle = std::get<1>(MNPerIterationShuffle);     //64

    using WG = typename CShuffleWarpGemmSelector<ATypeToUse,
                                                 BTypeToUse,
                                                 AccDataType,
                                                 MPerXdl,
                                                 NPerXdl,
                                                 KPerXdl,
                                                 isCTransposed,
                                                 WarpGemmMRepeat,
                                                 WarpGemmNRepeat,
                                                 WarpGemmMInterleave,
                                                 WarpGemmNInterleave>::Type;

    using CWarpDstr         = typename WG::CWarpDstr;
    using CWarpOutputDstr   = typename WG::CWarpOutputDstr;
    using CWarpTensor       = typename WG::CWarpTensor;
    using CWarpOutputTensor = typename WG::CWarpOutputTensor;
    // using CWarpDstrEncoding = typename WG::CWarpDstrEncoding;
    static constexpr bool UseMmacOutputLdsDistribution = Problem::TiledMMAPermuteN;
    static constexpr bool SkipMmacTensorRegShuffle = UseMmacOutputLdsDistribution;
    using LdsWarpDstr =
        std::conditional_t<UseMmacOutputLdsDistribution, CWarpOutputDstr, CWarpDstr>;
    using SFC               = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                                  sequence<0, 1>,
                                                  sequence<MPerIterationShuffle, NPerIterationShuffle>>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_VECTOR4) || \
    defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_VECTOR4_FORCE_LDS_STORE) || \
    defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE)
            // Keep each 4-element output vector physically contiguous while
            // XOR-swizzling its vector slot by the row. This preserves the
            // existing dwordx2 global writeback contract and distributes the
            // MMAC-output LDS traffic across all 32 banks.
            static_assert(NPerIterationShuffle % 4 == 0);
            constexpr auto packed_m_n0_n1 =
                make_naive_tensor_descriptor_packed(
                    make_tuple(number<MPerIterationShuffle>{},
                               number<NPerIterationShuffle / 4>{},
                               number<4>{}),
                    number<4>{});
            constexpr auto xor_m_n0 = transform_tensor_descriptor(
                packed_m_n0_n1,
                make_tuple(
                    make_xor_transform(
                        make_tuple(number<MPerIterationShuffle>{},
                                   number<NPerIterationShuffle / 4>{})),
                    make_pass_through_transform(number<4>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));
            return transform_tensor_descriptor(
                xor_m_n0,
                make_tuple(
                    make_pass_through_transform(
                        number<MPerIterationShuffle>{}),
                    make_merge_transform(
                        make_tuple(number<NPerIterationShuffle / 4>{},
                                   number<4>{}))),
                make_tuple(sequence<0>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
#else
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<NPerIterationShuffle>{}, number<1>{}));
#endif
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<1>{}, number<MPerIterationShuffle>{}));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeLdsDistributionEncode()
    {
        constexpr auto block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                                             sequence<NumNXdlPerWavePerShuffle, NWave>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto block_dstr_encoding = detail::make_embed_tile_distribution_encoding(
            block_outer_dstr_encoding, typename LdsWarpDstr::DstrEncode{});

        return block_dstr_encoding;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2)
        // Two independently swizzled slots let the next 32-KiB slab enter LDS
        // after the current slab is read, without keeping a doubled register
        // tile live.  The operand allocation is already 64 KiB for the tuned
        // grouped GEMM, so this does not increase kernel LDS residency.
        return 2 * MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
#else
        return MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
#endif
    }

    template <index_t iAccess, typename LdsTile, typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE void
    scale_tile(LdsTile& lds_tile, ScaleM& scale_m_window, ScaleN& scale_n_window)
    {
        // Load tiles
        const auto scale_m_tile = load_tile(scale_m_window);
        const auto scale_n_tile = load_tile(scale_n_window);

        // Compute element-wise product in-place i.e. lds_tile = lds_tile * scale_m * scale_n
        tile_elementwise_inout(
            element_wise::MultiDMultiply{}, lds_tile, lds_tile, scale_m_tile, scale_n_tile);

        // Move scale windows
        constexpr index_t num_access = SFC::get_num_of_access();
        if constexpr(iAccess != num_access - 1)
        {
            constexpr auto step = SFC::get_forward_step(number<iAccess>{});

            move_tile_window(scale_m_window, {step.at(number<0>{}), step.at(number<1>{})});
            move_tile_window(scale_n_window, {step.at(number<0>{}), step.at(number<1>{})});
        }
    }

    template <index_t iAccess, typename OAccTile, typename LdsTile>
    CK_TILE_DEVICE void slice_acc_tile(const OAccTile& o_acc_tile, LdsTile& lds_tile)
    {
        constexpr auto idx_y_start = SFC::get_index(number<iAccess>{});

        constexpr auto mIter = number<idx_y_start.at(number<0>{}) / (MPerIterationShuffle)>{};
        constexpr auto nIter = number<idx_y_start.at(number<1>{}) / (NPerIterationShuffle)>{};
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        if constexpr(UseMmacOutputLdsDistribution)
        {
            constexpr auto c_warp_output_y_lengths =
                to_sequence(CWarpOutputDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_output_y_index_zeros =
                uniform_sequence_gen_t<CWarpOutputDstr::NDimY, 0>{};

            static_ford<sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>>{}(
                [&](auto mn) {
                    constexpr auto iM = number<mn[number<0>{}]>{};
                    constexpr auto iN = number<mn[number<1>{}]>{};
                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter * NumMXdlPerWavePerShuffle + iM,
                                                 nIter * NumNXdlPerWavePerShuffle + iN>{},
                                        c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    CWarpOutputTensor c_warp_output_tensor;
                    constexpr index_t c_thread_buf_size = c_warp_tensor.get_thread_buffer_size();
                    if constexpr((std::is_same_v<ATypeToUse, half_t> ||
                                  std::is_same_v<ATypeToUse, bfloat16_t>) &&
                                 (std::is_same_v<BTypeToUse, half_t> ||
                                  std::is_same_v<BTypeToUse, bfloat16_t>) &&
                                 ((MPerXdl == 16 && NPerXdl == 32 && KPerXdl == 128) ||
                                  ((MPerXdl == 32 && NPerXdl == 32 && KPerXdl == 16) &&
                                   WarpGemmNInterleave == 1)))
                    {
                        static_for<0, c_thread_buf_size, 1>{}([&](auto i) {
                            c_warp_output_tensor.get_thread_buffer()[i] =
                                c_warp_tensor.get_thread_buffer()[i];
                        });
                    }
                    else
                    {
                        constexpr index_t Gemm0NInterleave = 2;
                        static_for<0, c_thread_buf_size, 1>{}([&](auto i) {
                            index_t reduce_index =
                                (i / (Gemm0NInterleave * 4)) * (Gemm0NInterleave * 4) +
                                (i % (Gemm0NInterleave * 4)) / Gemm0NInterleave +
                                i % Gemm0NInterleave * 4;
                            c_warp_output_tensor.get_thread_buffer()[i] =
                                c_warp_tensor.get_thread_buffer()[reduce_index];
                        });
                    }
                    lds_tile.set_y_sliced_thread_data(
                        merge_sequences(sequence<iM, iN>{}, c_warp_output_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_output_y_lengths),
                        c_warp_output_tensor.get_y_sliced_thread_data(
                            c_warp_output_y_index_zeros, c_warp_output_y_lengths));
                });
        }
        else
        {
            lds_tile.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<mIter * NumMXdlPerWavePerShuffle,
                                         nIter * NumNXdlPerWavePerShuffle>{},
                                c_warp_y_index_zeros),
                merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                                c_warp_y_lengths));
        }
    }

    template <typename LdsTile, typename InLdsWindow>
    CK_TILE_DEVICE void cast_lds_tile(LdsTile& lds_tile, InLdsWindow& in_lds_window)
    {
        const auto c_warptile_in_tensor_casted = cast_tile<ODataType>(lds_tile);

#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_PAIR_B64_STORE)
        // The two BF16x4 vectors in each pair land 32 bytes apart after the
        // row XOR (the ordering flips with the XOR bit). Root both writes at
        // the physical low address so the backend can select ds_write2_b64.
        in_lds_window.template store_force_vectorized_pair<6, 4>(
            c_warptile_in_tensor_casted);
#elif defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_XOR_VECTOR4_FORCE_LDS_STORE)
        // The XOR descriptor makes the generic safety analysis scalarize this
        // store. Y6 is the four-element contiguous N subspan of this MMAC
        // output distribution, so issue it explicitly as one BF16x4 vector.
        in_lds_window.template store_force_vectorized<6, 4>(
            c_warptile_in_tensor_casted);
#elif defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_BUILTIN_SWIZZLE) || \
    defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_SCALAR_GROUP16_SWIZZLE)
        // Apply the byte-offset swizzle at the vector transaction boundary.
        // Unlike expressing the XOR as a tensor transform, this keeps the
        // compiler's original 16-byte LDS write transaction intact.
        constexpr index_t lds_store_bytes =
            sizeof(typename remove_cvref_t<InLdsWindow>::load_store_traits::vector_t);
        if constexpr(
#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_SCALAR_GROUP16_SWIZZLE)
            lds_store_bytes == 2 ||
#endif
            lds_store_bytes == 4 || lds_store_bytes == 8 || lds_store_bytes == 16)
        {
            store_tile(in_lds_window, c_warptile_in_tensor_casted, bool_constant<true>{});
        }
        else
        {
            store_tile(in_lds_window, c_warptile_in_tensor_casted);
        }
#else
        store_tile(in_lds_window, c_warptile_in_tensor_casted);
#endif
    }

    template <typename DramWindows, typename COutTensor>
    CK_TILE_DEVICE void apply_d_tensors(DramWindows& d_dram_windows, COutTensor& c_out_tensor)
    {
        const auto ds_tensor = generate_tuple(
            [&](auto idx) { return load_tile(d_dram_windows[idx]); }, number<NumDTensor>{});

        const auto c_ds_tiles = concat_tuple_of_reference(
            tie(c_out_tensor, c_out_tensor),
            generate_tie([&](auto idx) -> const auto& { return ds_tensor[idx]; },
                         number<NumDTensor>{}));

        tile_elementwise_inout_unpack(typename Problem::CDElementwise{}, c_ds_tiles);
    }

    template <typename OutDramWindow, typename COutTensor>
    CK_TILE_DEVICE void store_to_dram(OutDramWindow& out_dram_window,
                                      const COutTensor& c_out_tensor)
    {
        if constexpr(MemoryOperation == memory_operation_enum::set)
        {
            store_tile(out_dram_window, c_out_tensor);
        }
        else
        {
            update_tile(out_dram_window, c_out_tensor);
        }
    }

    /**
     * @brief Move both the output and D tensors windows for the next access.
     */
    template <index_t iAccess, typename OutDramWindow, typename DDramWindows>
    CK_TILE_DEVICE void move_windows(OutDramWindow& out_dram_window, DDramWindows& d_dram_windows)
    {
        constexpr index_t num_access = SFC::get_num_of_access();
        if constexpr(iAccess != num_access - 1)
        {
            constexpr auto step = SFC::get_forward_step(number<iAccess>{});

            // move the output dram window
            move_tile_window(out_dram_window, {step.at(number<0>{}), step.at(number<1>{})});

            // move windows for each of the D matrices (inputs for element-wise)
            static_for<0, NumDTensor, 1>{}([&](auto idx) {
                move_tile_window(d_dram_windows[idx], {step.at(number<0>{}), step.at(number<1>{})});
            });
        }
    }

    // TODO: Check if there would be nicer ways to overload rather than with EmptyScale or nullptr_t
    struct EmptyScale
    {
    };

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM                         = EmptyScale,
              typename ScaleN                         = EmptyScale,
              int EnablePermuateN_                    = TiledMMAPermuteN,
              std::enable_if_t<EnablePermuateN_, int> = 0>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        return this->template operator()<ODramWindow, OAccTile, DsDramWindows, ScaleM, ScaleN, 0>(
            out_dram_window, o_acc_tile, ds_dram_windows, p_smem, scale_m, scale_n);

        // constexpr int kM0 = MWave;
        // constexpr int kM2 = 4;
        // constexpr int kM1 = MPerXdl / kM2;

        // constexpr int kN0 = NWave;
        // constexpr int kN1 = NPerXdl;
        // constexpr int kN2 = NRepeat;

        // using IntrThreadShuffleEncode =
        //     tile_distribution_encoding<sequence<>,
        //                                tuple<sequence<kM0, kM1, kM2>, sequence<kN0, kN1, kN2>>,
        //                                tuple<sequence<1, 2>, sequence<1, 2>>,
        //                                tuple<sequence<0, 0>, sequence<1, 1>>,
        //                                sequence<1, 2>,
        //                                sequence<2, 2>>;
        // constexpr auto dram_tile_distribution =
        //     make_static_tile_distribution(IntrThreadShuffleEncode{});

        // auto d_dram_windows = generate_tuple(
        //     [&](auto idx) {
        //         return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
        //     },
        //     number<NumDTensor>{});

        // constexpr auto c_warp_y_lengths =
        //     to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        // constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // auto shuffle_acc  = make_static_distributed_tensor<AccDataType>(dram_tile_distribution);
        // auto c_out_tensor = make_static_distributed_tensor<ODataType>(dram_tile_distribution);

        // // Optional scales (must share the same distribution to match per-thread indexing)
        // constexpr bool has_scales =
        //     !std::is_same<ScaleM, EmptyScale>::value && !std::is_same<ScaleN, EmptyScale>::value;

        // // Tiles to hold row/col scales when present
        // using SMType =
        //     std::conditional_t<has_scales, remove_cvref_t<typename ScaleM::DataType>, float>;
        // using SNType =
        //     std::conditional_t<has_scales, remove_cvref_t<typename ScaleN::DataType>, float>;

        // auto sm_tile = make_static_distributed_tensor<SMType>(dram_tile_distribution);
        // auto sn_tile = make_static_distributed_tensor<SNType>(dram_tile_distribution);

        // // Build windows only if scales are provided
        // auto scale_m_window = [&]() {
        //     if constexpr(has_scales)
        //     {
        //         return make_tile_window(scale_m, dram_tile_distribution);
        //     }
        //     else
        //     {
        //         return EmptyScale{};
        //     }
        // }();
        // auto scale_n_window = [&]() {
        //     if constexpr(has_scales)
        //     {
        //         return make_tile_window(scale_n, dram_tile_distribution);
        //     }
        //     else
        //     {
        //         return EmptyScale{};
        //     }
        // }();

        // static_for<0, MRepeat, 1>{}([&](auto mIter) {
        //     // Slice accumulators for this M repeat into the permuted layout
        //     shuffle_acc.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
        //         merge_sequences(sequence<mIter, 0>{}, c_warp_y_index_zeros),
        //         merge_sequences(sequence<1, NRepeat>{}, c_warp_y_lengths));

        //     // If scales provided, load them with identical distribution
        //     if constexpr(has_scales)
        //     {
        //         sm_tile = load_tile(scale_m_window); // row scales in permuted layout
        //         sn_tile = load_tile(scale_n_window); // col scales in permuted layout
        //     }

        //     // Pack 4 “rows per lane” as you already do
        //     static_for<0, NRepeat, 1>{}([&](auto n_idx) {
        //         // source indices in shuffle_acc: (n_idx * product(Y) + row)
        //         const index_t base = n_idx * c_warp_y_lengths.product();

        //         // local lambda to fuse scale (if present) and convert
        //         auto emit = [&](index_t out_idx, index_t src_row) {
        //             AccDataType v = shuffle_acc.get_thread_buffer()[base + src_row];

        //             if constexpr(has_scales)
        //             {
        //                 // same linear index mapping on the permuted distribution
        //                 const auto s_m = static_cast<float>(sm_tile.get_thread_buffer()[out_idx]);
        //                 const auto s_n = static_cast<float>(sn_tile.get_thread_buffer()[out_idx]);
        //                 v              = static_cast<AccDataType>(v * s_m * s_n);
        //             }

        //             c_out_tensor.get_thread_buffer()[out_idx] = type_convert<ODataType>(v);
        //         };

        //         // Your current packing pattern (rows 0..3, spaced by NRepeat)
        //         emit(n_idx + 0 * NRepeat, 0);
        //         emit(n_idx + 1 * NRepeat, 1);
        //         emit(n_idx + 2 * NRepeat, 2);
        //         emit(n_idx + 3 * NRepeat, 3);
        //     });

        //     // store/update
        //     if constexpr(MemoryOperation == memory_operation_enum::set)
        //     {
        //         store_tile(out_dram_window, c_out_tensor);
        //     }
        //     else
        //     {
        //         update_tile(out_dram_window, c_out_tensor);
        //     }

        //     // advance output (and any D-tensors) by one MPerXdl*MWave chunk
        //     move_tile_window(out_dram_window, {number<MPerXdl * MWave>{}, number<0>{}});
        //     static_for<0, NumDTensor, 1>{}([&](auto idx) {
        //         move_tile_window(d_dram_windows[idx], {number<MPerXdl * MWave>{}, number<0>{}});
        //     });
        // });
    }

    template <index_t thread_buf_size, typename ShlfTensor>
    CK_TILE_DEVICE void shfl_tensor_regs(ShlfTensor& tensor) {
        using DataType = typename ShlfTensor::DataType;

        static_assert(thread_buf_size % 4 == 0, "thread_buf_size should be multiple of 4");
        constexpr index_t loops = thread_buf_size / 4;
        DataType val_0[4], val_1[4];

        const int group = __lane_id() / 16;
        const int lane = __lane_id() % 16;

        static_for<0, loops, 1>{}([&](int lo) {
            int loop_off = lo * 4;

            for (int i = 0; i < 4; i++) {
                val_0[i] = tensor.get_thread_buffer()[loop_off + i];
            }

            static_for<0, 4, 1>{}([&](int g) {

                val_1[0] = warp_shuffle(val_0[g], lane);
                val_1[1] = warp_shuffle(val_0[g], 16 + lane);
                val_1[2] = warp_shuffle(val_0[g], 32 + lane);
                val_1[3] = warp_shuffle(val_0[g], 48 + lane);

                if (group == g) {
                    tensor.get_thread_buffer()[loop_off + 0] = val_1[0];
                    tensor.get_thread_buffer()[loop_off + 1] = val_1[1];
                    tensor.get_thread_buffer()[loop_off + 2] = val_1[2];
                    tensor.get_thread_buffer()[loop_off + 3] = val_1[3];
                }
            });

        });
    }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM                          = EmptyScale,
              typename ScaleN                          = EmptyScale,
              int EnablePermuateN_                     = TiledMMAPermuteN,
              std::enable_if_t<!EnablePermuateN_, int> = 0>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        constexpr auto LdsTileDistr = make_static_tile_distribution(MakeLdsDistributionEncode());

        auto lds_tile = make_static_distributed_tensor<AccDataType>(LdsTileDistr);

        constexpr auto lds_block_desc = MakeLdsBlockDescriptor<Problem>();
        auto o_lds_block              = make_tensor_view<address_space_enum::lds>(
            static_cast<ODataType*>(p_smem), lds_block_desc);

        auto in_lds_window = make_tile_window(
            o_lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0},
            LdsTileDistr);

        auto out_lds_window = make_tile_window(
            o_lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0});

        constexpr index_t num_access = SFC::get_num_of_access();    //block包含多少个 配置的block gemm

#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2)
        auto o_lds_block_alt = make_tensor_view<address_space_enum::lds>(
            static_cast<ODataType*>(p_smem) +
                MPerIterationShuffle * NPerIterationShuffle,
            lds_block_desc);
        auto in_lds_window_alt = make_tile_window(
            o_lds_block_alt,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0},
            LdsTileDistr);
        auto out_lds_window_alt = make_tile_window(
            o_lds_block_alt,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0});
#endif

        static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                      "Currently, the CShuffle Epilogue only supports the Row Major Output layout");

        using TileEncodingPattern =
            tile_distribution_encoding_pattern_2d<kBlockSize,
                                                  MPerIterationShuffle,
                                                  NPerIterationShuffle,
                                                  GetVectorSizeC(),
                                                  tile_distribution_pattern::thread_raked,
                                                  Problem::kNumWaveGroups>;
        constexpr auto dram_tile_distribution =
            TileEncodingPattern::make_2d_static_tile_distribution();

        auto d_dram_windows = generate_tuple(
            [&](auto idx) {
                return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
            },
            number<NumDTensor>{});

        constexpr bool has_scales =
            !std::is_same<ScaleM, EmptyScale>::value && !std::is_same<ScaleN, EmptyScale>::value;
        constexpr bool has_scalar_scales =
            has_scales && std::is_arithmetic_v<remove_cvref_t<ScaleM>> &&
            std::is_arithmetic_v<remove_cvref_t<ScaleN>>;
        constexpr bool has_scale_windows = has_scales && !has_scalar_scales;
        auto scale_m_window = [&]() {
            if constexpr(has_scale_windows)
            {
                return make_tile_window(scale_m, lds_tile.get_tile_distribution());
            }
            else
            {
                return EmptyScale{};
            }
        }();
        auto scale_n_window = [&]() {
            if constexpr(has_scale_windows)
            {
                return make_tile_window(scale_n, lds_tile.get_tile_distribution());
            }
            else
            {
                return EmptyScale{};
            }
        }();

#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2)
        if constexpr(num_access == 4)
        {
        auto write_lds_access = [&](auto i_access, auto& in_window) {
            constexpr index_t iAccess = decltype(i_access)::value;
            slice_acc_tile<iAccess>(o_acc_tile, lds_tile);

            constexpr index_t thread_buf_size = lds_tile.get_thread_buffer_size();
            if constexpr(!SkipMmacTensorRegShuffle)
            {
                shfl_tensor_regs<thread_buf_size>(lds_tile);
            }

            if constexpr(has_scale_windows)
            {
                scale_tile<iAccess>(lds_tile, scale_m_window, scale_n_window);
            }
            else if constexpr(has_scalar_scales)
            {
                tile_elementwise_inout(
                    [&](auto& x) { x = x * scale_m * scale_n; }, lds_tile);
            }
            cast_lds_tile(lds_tile, in_window);
        };
        auto load_lds_access = [&](auto& out_window) {
            auto c_out_lds_window =
                make_tile_window(out_window, dram_tile_distribution);
            return load_tile(c_out_lds_window);
        };

        // Five barriers replace the generic loop's eight:
        // finish GEMM, publish slab0, then each following barrier both
        // publishes the alternate slot and proves the prior slot is no longer
        // being read before it is reused.
        block_sync_lds();
        write_lds_access(number<0>{}, in_lds_window);
        block_sync_lds();

#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2_STORE_FIRST)
        {
            auto c_out_tensor = load_lds_access(out_lds_window);
            apply_d_tensors(d_dram_windows, c_out_tensor);
            store_to_dram(out_dram_window, c_out_tensor);
            move_windows<0>(out_dram_window, d_dram_windows);
        }
        write_lds_access(number<1>{}, in_lds_window_alt);
#else
        auto c_out_tensor_0 = load_lds_access(out_lds_window);
        write_lds_access(number<1>{}, in_lds_window_alt);
        apply_d_tensors(d_dram_windows, c_out_tensor_0);
        store_to_dram(out_dram_window, c_out_tensor_0);
        move_windows<0>(out_dram_window, d_dram_windows);
#endif
        block_sync_lds();

#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2_STORE_FIRST)
        {
            auto c_out_tensor = load_lds_access(out_lds_window_alt);
            apply_d_tensors(d_dram_windows, c_out_tensor);
            store_to_dram(out_dram_window, c_out_tensor);
            move_windows<1>(out_dram_window, d_dram_windows);
        }
        write_lds_access(number<2>{}, in_lds_window);
#else
        auto c_out_tensor_1 = load_lds_access(out_lds_window_alt);
        write_lds_access(number<2>{}, in_lds_window);
        apply_d_tensors(d_dram_windows, c_out_tensor_1);
        store_to_dram(out_dram_window, c_out_tensor_1);
        move_windows<1>(out_dram_window, d_dram_windows);
#endif
        block_sync_lds();

#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_PIPELINED_M2_STORE_FIRST)
        {
            auto c_out_tensor = load_lds_access(out_lds_window);
            apply_d_tensors(d_dram_windows, c_out_tensor);
            store_to_dram(out_dram_window, c_out_tensor);
            move_windows<2>(out_dram_window, d_dram_windows);
        }
        write_lds_access(number<3>{}, in_lds_window_alt);
#else
        auto c_out_tensor_2 = load_lds_access(out_lds_window);
        write_lds_access(number<3>{}, in_lds_window_alt);
        apply_d_tensors(d_dram_windows, c_out_tensor_2);
        store_to_dram(out_dram_window, c_out_tensor_2);
        move_windows<2>(out_dram_window, d_dram_windows);
#endif
        block_sync_lds();

        auto c_out_tensor_3 = load_lds_access(out_lds_window_alt);
        apply_d_tensors(d_dram_windows, c_out_tensor_3);
        store_to_dram(out_dram_window, c_out_tensor_3);
        }
        else
#endif
        {
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            block_sync_lds();
            slice_acc_tile<iAccess>(o_acc_tile, lds_tile);

            //for hg mmac && no Ninterleave, need to shuffle the tensor registers
            constexpr index_t thread_buf_size = lds_tile.get_thread_buffer_size();
            if constexpr(!SkipMmacTensorRegShuffle)
            {
                shfl_tensor_regs<thread_buf_size>(lds_tile);
            }

            if constexpr(has_scale_windows)
            {
                scale_tile<iAccess>(lds_tile, scale_m_window, scale_n_window);
            }
            else if constexpr(has_scalar_scales)
            {
                tile_elementwise_inout(
                    [&](auto& x) { x = x * scale_m * scale_n; }, lds_tile);
            }

            cast_lds_tile(lds_tile, in_lds_window);
            block_sync_lds();

            auto c_out_lds_window =
                make_tile_window(out_lds_window, dram_tile_distribution);
            auto c_out_tensor =
#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_BUILTIN_SWIZZLE) || \
    defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_SCALAR_GROUP16_SWIZZLE)
                [&]() {
                    constexpr index_t lds_load_bytes =
                        sizeof(typename remove_cvref_t<decltype(c_out_lds_window)>::
                                   load_store_traits::vector_t);
                    if constexpr(
#if defined(CK_TILE_GROUPED_GEMM_CSHUFFLE_SCALAR_GROUP16_SWIZZLE)
                        lds_load_bytes == 2 ||
#endif
                        lds_load_bytes == 4 || lds_load_bytes == 8 ||
                                 lds_load_bytes == 16)
                    {
                        return load_tile(c_out_lds_window,
                                         bool_constant<true>{},
                                         bool_constant<true>{});
                    }
                    else
                    {
                        return load_tile(c_out_lds_window);
                    }
                }();
#else
                load_tile(c_out_lds_window);
#endif

            apply_d_tensors(d_dram_windows, c_out_tensor);
            store_to_dram(out_dram_window, c_out_tensor);
            move_windows<iAccess>(out_dram_window, d_dram_windows);
        });
        }
    }
};

} // namespace ck_tile
