// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Adapted from AMD CK example/15_grouped_gemm/grouped_gemm_xdl_splitk_fp16.cpp.
#include "grouped_gemm_splitk_common.hpp"
int main(int argc, char** argv)
{
    using namespace splitk_example;
    try
    {
        const int batch = argc > 1 ? std::stoi(argv[1]) : 4;
        if(argc == 7)
        {
            const int groups = std::stoi(argv[3]);
            if(groups <= 0)
                throw std::invalid_argument("group count must be positive");
            return !Run<TestOp<Ordinary<half_t>>, half_t>(
                std::vector<index_t>(groups, std::stoi(argv[4])),
                std::vector<index_t>(groups, std::stoi(argv[5])),
                std::vector<index_t>(groups, std::stoi(argv[6])),
                batch,
                false,
                true);
        }
        return !Run<TestOp<Ordinary<half_t>>, half_t>(
            {64, 127, 255}, {128, 128, 128}, {256, 256, 256}, batch, false, argc > 2);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
