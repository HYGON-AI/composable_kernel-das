// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include "universal_gemm_dispatch.hpp"
#include "universal_gemm_invoker.hpp"

template <typename T, typename BLayout, bool Shuffle>
float launch(const UniversalGemmOptions& options, const ck_tile::UniversalGemmKernelArgs<>& args)
{
    switch(options.pipeline)
    {
    case Pipeline::CompV3:
        return invoke_universal_gemm<T, BLayout, Pipeline::CompV3, Scheduler::Intrawave, Shuffle>(
            args, options.stream);
    case Pipeline::CompV4:
        return invoke_universal_gemm<T, BLayout, Pipeline::CompV4, Scheduler::Intrawave, Shuffle>(
            args, options.stream);
    case Pipeline::Mem:
        if(options.scheduler == Scheduler::Interwave)
            return invoke_universal_gemm<T, BLayout, Pipeline::Mem, Scheduler::Interwave, Shuffle>(
                args, options.stream);
        return invoke_universal_gemm<T, BLayout, Pipeline::Mem, Scheduler::Intrawave, Shuffle>(
            args, options.stream);
    default:
        throw std::invalid_argument("unknown pipeline");
    }
}
