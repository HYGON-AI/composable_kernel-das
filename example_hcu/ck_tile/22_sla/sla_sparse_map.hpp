// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/kernel/sla_sparse_map_kernel.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>

namespace sla {

template <typename DataType, ck_tile::index_t PaddedCandidates>
static void launch_radix_topk(typename SlaSparseMapProblem<DataType>::Kargs kargs,
                              int BH,
                              int Lq_blocks,
                              int topk,
                              int64_t* lut_ptr,
                              int8_t* sparse_map_ptr,
                              hipStream_t stream)
{
    using Kernel = SlaSparseMapScoreRadixKernel<DataType, PaddedCandidates>;
    auto callable = ck_tile::make_kernel<Kernel::kBlockSize, 1>(
        Kernel{}, dim3(Lq_blocks, BH, 1), dim3(Kernel::kBlockSize, 1, 1), 0,
        kargs, lut_ptr, sparse_map_ptr, topk);
    callable(ck_tile::stream_config{stream, false});
}

template <typename DataType>
static void dispatch_radix_topk(typename SlaSparseMapProblem<DataType>::Kargs kargs,
                                int BH,
                                int Lq_blocks,
                                int Lk_blocks,
                                int topk,
                                int64_t* lut_ptr,
                                int8_t* sparse_map_ptr,
                                hipStream_t stream)
{
    if(Lk_blocks <= 64)
        launch_radix_topk<DataType, 64>(kargs, BH, Lq_blocks, topk, lut_ptr, sparse_map_ptr, stream);
    else if(Lk_blocks <= 128)
        launch_radix_topk<DataType, 128>(kargs, BH, Lq_blocks, topk, lut_ptr, sparse_map_ptr, stream);
    else if(Lk_blocks <= 256)
        launch_radix_topk<DataType, 256>(kargs, BH, Lq_blocks, topk, lut_ptr, sparse_map_ptr, stream);
    else if(Lk_blocks <= 512)
        launch_radix_topk<DataType, 512>(kargs, BH, Lq_blocks, topk, lut_ptr, sparse_map_ptr, stream);
    else if(Lk_blocks <= 1024)
        launch_radix_topk<DataType, 1024>(kargs, BH, Lq_blocks, topk, lut_ptr, sparse_map_ptr, stream);
    else
        launch_radix_topk<DataType, 2048>(kargs, BH, Lq_blocks, topk, lut_ptr, sparse_map_ptr, stream);
}

template <typename DataType>
inline void launch_sparse_map(
    const DataType* q_ptr,
    const DataType* k_ptr,
    DataType* pooled_q_ptr,
    DataType* pooled_k_ptr,
    int64_t* lut_ptr,
    int8_t* sparse_map_ptr,
    int B, int H, int Lq, int Lk, int D,
    int BLKQ, int BLKK, float topk_ratio,
    hipStream_t stream = nullptr)
{
    (void)D;
    const int Lq_blocks = (Lq + BLKQ - 1) / BLKQ;
    const int Lk_blocks = (Lk + BLKK - 1) / BLKK;
    const int topk = std::min(Lk_blocks, static_cast<int>(topk_ratio * Lk_blocks));
    const int BH = B * H;

    using Kargs = SlaSparseMapKargs<DataType>;
    Kargs kargs{
        q_ptr, k_ptr, nullptr, pooled_q_ptr, pooled_k_ptr,
        Lq, Lk, BLKQ, BLKK, Lq_blocks, Lk_blocks
    };

    using PoolQKernel = SlaSparseMapPoolQKernel<DataType>;
    auto pool_q_callable = ck_tile::make_kernel<PoolQKernel::kBlockSize, 1>(
        PoolQKernel{}, dim3(Lq_blocks, BH, 1),
        dim3(PoolQKernel::kBlockSize, 1, 1), 0, kargs);
    pool_q_callable(ck_tile::stream_config{stream, false});

    using PoolKKernel = SlaSparseMapPoolKKernel<DataType>;
    auto pool_k_callable = ck_tile::make_kernel<PoolKKernel::kBlockSize, 1>(
        PoolKKernel{}, dim3(Lk_blocks, BH, 1),
        dim3(PoolKKernel::kBlockSize, 1, 1), 0, kargs);
    pool_k_callable(ck_tile::stream_config{stream, false});

    if(topk == 0)
    {
        using ScatterKernel = SlaSparseMapScatterKernel<DataType>;
        auto scatter_callable = ck_tile::make_kernel<ScatterKernel::kBlockSize, 1>(
            ScatterKernel{}, dim3(Lq_blocks, BH, 1),
            dim3(ScatterKernel::kBlockSize, 1, 1), 0, kargs,
            static_cast<const int32_t*>(nullptr), lut_ptr, sparse_map_ptr, topk);
        scatter_callable(ck_tile::stream_config{stream, false});
    }
    else
    {
        dispatch_radix_topk<DataType>(
            kargs, BH, Lq_blocks, Lk_blocks, topk, lut_ptr, sparse_map_ptr, stream);
    }
}

template <typename DataType>
inline void launch_sparse_map(
    const DataType* q_ptr,
    const DataType* k_ptr,
    int8_t* sparse_map_ptr,
    int64_t* lut_ptr,
    int32_t* lut_size_ptr,
    int B, int H, int Lq, int Lk, int D,
    int BLKQ, int BLKK, int max_nnz,
    float topk_ratio,
    float sm_scale,
    hipStream_t stream = nullptr)
{
    (void)sm_scale;
    (void)max_nnz;
    (void)lut_size_ptr;
    const int Lq_blocks = (Lq + BLKQ - 1) / BLKQ;
    const int Lk_blocks = (Lk + BLKK - 1) / BLKK;
    ck_tile::DeviceMem pooled_q_buf(static_cast<size_t>(B * H * Lq_blocks * D) * sizeof(DataType));
    ck_tile::DeviceMem pooled_k_buf(static_cast<size_t>(B * H * Lk_blocks * D) * sizeof(DataType));
    launch_sparse_map<DataType>(
        q_ptr, k_ptr,
        static_cast<DataType*>(pooled_q_buf.GetDeviceBuffer()),
        static_cast<DataType*>(pooled_k_buf.GetDeviceBuffer()),
        lut_ptr, sparse_map_ptr,
        B, H, Lq, Lk, D, BLKQ, BLKK, topk_ratio, stream);
}

} // namespace sla

namespace ck_tile {
namespace example {
namespace sla {

using ::sla::launch_sparse_map;

} // namespace sla
} // namespace example
} // namespace ck_tile
