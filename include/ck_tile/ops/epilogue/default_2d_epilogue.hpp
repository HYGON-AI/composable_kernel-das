// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

// this epilogue just store out a M*N matrix, row major

template <typename AccDataType_,
          typename ODataType_,
          bool kPadM_,
          bool kPadN_,
          bool UseRawStore_                      = true,
          memory_operation_enum MemoryOperation_ = memory_operation_enum::set>
struct Default2DEpilogueProblem
{
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;
    static constexpr bool kPadM                            = kPadM_;
    static constexpr bool kPadN                            = kPadN_;
    static constexpr bool UseRawStore                      = UseRawStore_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
};

namespace legacy {
template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename CLayout_,
          bool kPadM_,
          bool kPadN_,
          index_t kMPerXdl_,
          index_t kNPerXdl_,
          index_t kKPerXdl_,
          bool isCTransposed_,
          bool UseRawStore_                      = true,
          memory_operation_enum MemoryOperation_ = memory_operation_enum::set>
struct DefaultGemm2DEpilogueProblem : public Default2DEpilogueProblem<AccDataType_,
                                                                      ODataType_,
                                                                      kPadM_,
                                                                      kPadN_,
                                                                      UseRawStore_,
                                                                      MemoryOperation_>
{
    using ADataType                        = remove_cvref_t<ADataType_>;
    using BDataType                        = remove_cvref_t<BDataType_>;
    using CLayout                          = remove_cvref_t<CLayout_>;
    static constexpr index_t kMPerXdl      = kMPerXdl_;
    static constexpr index_t kNPerXdl      = kNPerXdl_;
    static constexpr index_t kKPerXdl      = kKPerXdl_;
    static constexpr index_t isCTransposed = isCTransposed_;
};
} // namespace legacy

template <typename ADataType_,
          typename BDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename DsLayout_,
          typename CLayout_,
          typename CDElementwise_,
          index_t MPerBlock_,
          index_t NPerBlock_,
          bool PadM_,
          bool PadN_,
          index_t MPerWarp_,
          index_t NPerWarp_,
          index_t KPerWarp_,
          bool TransposeC_,
          bool UseRawStore_ = true>
struct DefaultGemm2DEpilogueProblem
    : legacy::DefaultGemm2DEpilogueProblem<ADataType_,
                                           BDataType_,
                                           AccDataType_,
                                           ODataType_,
                                           CLayout_,
                                           PadM_,
                                           PadN_,
                                           MPerWarp_,
                                           NPerWarp_,
                                           KPerWarp_,
                                           TransposeC_,
                                           UseRawStore_>
{
    using DsDataType = DsDataType_;
    using DsLayout = DsLayout_;
    using CDElementwise = CDElementwise_;
    static constexpr index_t kMPerBlock = MPerBlock_;
    static constexpr index_t kNPerBlock = NPerBlock_;
};

template <typename Problem_, typename Policy_ = void>
struct Default2DEpilogue
{
    using Problem                     = remove_cvref_t<Problem_>;
    using AccDataType                 = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                   = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM       = Problem::kPadM;
    static constexpr bool kPadN       = Problem::kPadN;
    static constexpr bool UseRawStore = Problem::UseRawStore;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?
    template <typename ODramWindowTmp, typename OAccTile>
    CK_TILE_DEVICE auto
    operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile, void* = nullptr)
    {

        // TODO: this is ugly
        if constexpr(UseRawStore && (kPadM || kPadN))
        {
            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            buffer_store_fence();
        }
        else
        {
            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
        }
    }
};

template <typename Problem, typename Policy = void>
struct DefaultGemm2DEpilogue : Default2DEpilogue<Problem, Policy>
{
    using Base = Default2DEpilogue<Problem, Policy>;
    using DsDataType = typename Problem::DsDataType;
    using DsLayout = typename Problem::DsLayout;
    static constexpr bool kUsesLogicalAccumulatorLayout = true;

    static_assert(DsDataType::size() == 0 && DsLayout::size() == 0);
    static_assert(std::is_same_v<typename Problem::CLayout, tensor_layout::gemm::RowMajor>);
    static_assert(std::is_same_v<typename Problem::CDElementwise, element_wise::PassThrough>);

    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC() { return 1; }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I>) { return 1; }

    template <typename OutputWindow, typename AccTile, typename DsWindows>
    CK_TILE_DEVICE void operator()(OutputWindow& output,
                                   const AccTile& acc,
                                   const DsWindows&,
                                   void* lds) const
    {
        Base{}(output, acc, lds);
    }
};
} // namespace ck_tile
