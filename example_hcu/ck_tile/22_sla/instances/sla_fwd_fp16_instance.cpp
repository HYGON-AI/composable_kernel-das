// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "sla_fwd.hpp"
#include "sla_fwd_runner_impl.hpp"

int run_sla_fwd_fp16(const ck_tile::ArgParser& arg_parser)
{
    return run_sla_fwd<ck_tile::fp16_t>(arg_parser);
}
