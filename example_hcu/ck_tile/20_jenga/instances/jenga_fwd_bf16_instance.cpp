// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "jenga_fwd.hpp"
#include "jenga_fwd_runner_impl.hpp"

int run_jenga_fwd_bf16(const ck_tile::ArgParser& arg_parser)
{
    return run_fwd<ck_tile::bf16_t, Jenga64DefaultPipelinePolicy>(arg_parser);
}
