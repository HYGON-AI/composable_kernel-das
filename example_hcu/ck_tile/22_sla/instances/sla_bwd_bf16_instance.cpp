// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "sla_bwd.hpp"
#include "sla_bwd_runner_impl.hpp"

int run_sla_bwd_bf16(const ck_tile::ArgParser& arg_parser)
{
    return run_sla_bwd<ck_tile::bf16_t>(arg_parser);
}
