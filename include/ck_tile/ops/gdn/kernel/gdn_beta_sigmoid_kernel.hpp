// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include <hip/hip_runtime.h>
#include <cmath>

namespace ck_tile {

struct BetaSigmoidKernel
{
    CK_TILE_DEVICE void operator()(const float* input,
                                   float* output,
                                   size_t count) const
    {
        const size_t index =
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(index < count)
            output[index] = 1.0f / (1.0f + expf(-input[index]));
    }
};

} // namespace ck_tile
