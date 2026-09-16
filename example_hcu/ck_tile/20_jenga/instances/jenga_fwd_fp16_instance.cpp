// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "jenga_fwd.hpp"
#include "jenga_fwd_runner_impl.hpp"

int run_jenga_fwd_fp16(const ck_tile::ArgParser& arg_parser)
{
    return run_fwd<ck_tile::fp16_t, Jenga64Fp16PipelinePolicy>(arg_parser);
}
