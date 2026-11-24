// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "gemm_utils.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include <stdexcept>

template <typename T,
          typename BLayout,
          Pipeline Kind,
          ck_tile::GemmPipelineScheduler Scheduler,
          bool Shuffle>
float invoke_universal_gemm(const ck_tile::UniversalGemmKernelArgs<>& args,
                            const ck_tile::stream_config& stream)
{
    using namespace ck_tile;
    using Partitioner = GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;
    constexpr bool pad = true;
    using HostTraits = TileGemmTraits<pad, pad, pad, Row, BLayout, Row>;
    using DeviceTraits =
        TileGemmUniversalTraits<pad, pad, pad, Kind == Pipeline::CompV4, Row, BLayout, Row>;
    using HostProblem = GemmPipelineProblem<T, T, float, GemmShape, HostTraits>;
    using Base = typename PipelineTypeTraits<Kind>::template Base<HostProblem>;
    using PassThrough = element_wise::PassThrough;
    using Default = DefaultGemm2DEpilogue<DefaultGemm2DEpilogueProblem<
        T, T, tuple<>, float, T, tuple<>, Row, PassThrough,
        GemmShape::kM, GemmShape::kN, pad, pad, 32, 32, 16, false>>;
    using CShuffle = CShuffleEpilogue<CShuffleEpilogueProblem<
        T, T, tuple<>, float, T, tuple<>, Row, PassThrough,
        GemmShape::kM, GemmShape::kN, 2, 2, 32, 32, 16, false>>;
    using Epilogue = std::conditional_t<Shuffle, CShuffle, Default>;

    const auto loops = Partitioner::GetLoopNum(args.K);
    const auto hot = Base::BlockHasHotloop(loops);
    const auto tail = Base::GetBlockLoopTailNum(loops);
    return Base::TailHandler(
        [&](auto hot_constant, auto tail_constant) {
            using Problem = UniversalGemmPipelineProblem<T,
                                                         T,
                                                         float,
                                                         GemmShape,
                                                         DeviceTraits,
                                                         Scheduler,
                                                         hot_constant.value,
                                                         tail_constant.value>;
            using Gemm = typename PipelineTypeTraits<Kind>::template Gemm<Problem>;
            using Kernel = GemmKernel<Partitioner, Gemm, Epilogue>;
            if(!Kernel::IsSupportedArgument(args))
                throw std::invalid_argument("unsupported shape, vector alignment or split_k");

            std::cout << Gemm::GetName() << " K tiles=" << loops << " hot=" << hot
                      << " tail=" << static_cast<int>(tail)
                      << " LDS=" << Kernel::GetSmemSize() * (Kind == Pipeline::CompV4 ? 2 : 1)
                      << '\n';
            return launch_kernel(stream,
                                 make_kernel<1>(Kernel{},
                                                Kernel::GridSize(args.M, args.N, args.k_batch),
                                                Kernel::BlockSize(),
                                                0,
                                                args));
        },
        hot,
        tail);
}
