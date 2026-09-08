
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "moe_quant.hpp"
#include <iostream>

#pragma once

using S = ck_tile::stream_config;
using A = moe_quant_args;

template <typename InputType_,
          typename OutputType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_M_,
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_>
using trait_ = moe_quant_traits_<InputType_,
                                       OutputType_,
                                       Repeat_M_,
                                       Repeat_N_,
                                       ThreadPerBlock_M_,
                                       ThreadPerBlock_N_,
                                       Vector_M_,
                                       Vector_N_,
                                       kPadN_>;

template <typename Traits_>
float moe_quant_(const S& s, A a)
{
    using InputType  = typename Traits_::InputType;
    using OutputType = typename Traits_::OutputType;

    using PipelineProblem = ck_tile::quantPipelineProblem<
        typename MoequantTypeConfig<InputType, OutputType>::InputDataType,
        typename MoequantTypeConfig<InputType, OutputType>::ScaleDataType,
        typename MoequantTypeConfig<InputType, OutputType>::ComputeDataType,
        //typename MoequantTypeConfig<InputType, OutputType>::ScaleDataType,
        typename MoequantTypeConfig<InputType, OutputType>::OutDataType,
        typename Traits_::Shape,
        Traits_::kPadN>;

    using Pipeline = ck_tile::quantPipelineOnePass<PipelineProblem>;

    using Kernel = ck_tile::Moequant<Pipeline>;

    const dim3 grids                       = Kernel::GridSize(a);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}
