// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "sla_fwd_runner_impl.hpp"

using ck_tile::example::sla::is_sla_shape_indexable;

static_assert(is_sla_shape_indexable(8191, 1, 2048, 128));
static_assert(!is_sla_shape_indexable(8192, 1, 2048, 128));
static_assert(!is_sla_shape_indexable(128, 128, 2048, 128));
static_assert(!is_sla_shape_indexable(65536, 65536, 64, 128));
static_assert(!is_sla_shape_indexable(2147483647, 2147483647, 131072, 128));
static_assert(is_sla_shape_indexable(131071, 1, 64, 128));
static_assert(!is_sla_shape_indexable(131072, 1, 64, 128));
static_assert(!is_sla_shape_indexable(0, 1, 64, 128));
static_assert(!is_sla_shape_indexable(1, -1, 64, 128));

int main()
{
    // Size-only regression: 2^32 K-feature elements plus two [BH,128,128]
    // FP16 matrices and one [BH,128] FP32 sum. No host/device allocation.
    const auto bytes = sla::get_fused_linear_workspace_bytes<ck_tile::fp16_t>(
        128, 128, 2048, 128, sla::ActType::SOFTMAX);
    if(bytes != 9672065024ULL)
    {
        std::cerr << "incorrect large SLA workspace size: " << bytes << '\n';
        return 1;
    }
    std::cout << "SLA size and index-boundary checks passed\n";
    return 0;
}
