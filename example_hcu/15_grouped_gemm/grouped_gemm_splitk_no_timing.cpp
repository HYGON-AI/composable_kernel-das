// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck/ck.hpp"
#undef CK_TIME_KERNEL
#define CK_TIME_KERNEL 0
#include "grouped_gemm_splitk_common.hpp"

int main()
{
    using namespace splitk_example;
    try
    {
        const bool direct =
            Run<TestOp<Ordinary<half_t>>, half_t>({16, 32}, {32, 64}, {256, 256}, 4, false, true);
        const bool staged =
            Run<TestOp<TwoStage<half_t>>, half_t>({16, 32}, {32, 64}, {256, 512}, 4, true, true);
        return !(direct && staged);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
