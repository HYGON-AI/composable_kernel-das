// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/sla/pipeline/sla_fused_linear_policy.hpp"

namespace sla {

template <ActType Type, typename Policy_ = SlaLinearDefaultPolicy>
struct SlaLinearFeaturePipeline
{
    using Policy = Policy_;
    using DataType = typename Policy::Problem::DataType;

    CK_TILE_DEVICE static float Apply(float x)
    {
        if constexpr(Type == ActType::RELU)
            return x > 0.0f ? x : 0.0f;
        else if constexpr(Type == ActType::ELU)
            return (x > 0.0f ? x : expf(x) - 1.0f) + 1.0f;
        else
            return x;
    }

    // BlockSoftmax2D uses the same max/sum reduction structure and requires
    // each row to stay within one wave. These 4/8-lane row groups satisfy
    // that constraint while matching the 64/32-row fused tiles.
    template <int Rows>
    CK_TILE_DEVICE static void LoadSoftmax(const DataType* __restrict__ src,
                                           int64_t base,
                                           int row0,
                                           int L,
                                           DataType* __restrict__ dst)
    {
        constexpr int D = Policy::kHeadDim;
        constexpr int GroupSize = Policy::kBlockSize / Rows;
        const int tid = threadIdx.x;
        const int row = tid / GroupSize;
        const int lane = tid % GroupSize;
        const int global_row = row0 + row;

        float row_max = -INFINITY;
        for(int d = lane; d < D; d += GroupSize)
        {
            const float x = global_row < L
                                ? ck_tile::type_convert<float>(src[base + global_row * D + d])
                                : -INFINITY;
            row_max = fmaxf(row_max, x);
        }
        for(int offset = GroupSize / 2; offset > 0; offset >>= 1)
            row_max = fmaxf(row_max, __shfl_down(row_max, offset, GroupSize));
        row_max = __shfl(row_max, 0, GroupSize);

        float row_sum = 0.0f;
        for(int d = lane; d < D; d += GroupSize)
        {
            const float x = global_row < L
                                ? ck_tile::type_convert<float>(src[base + global_row * D + d])
                                : -INFINITY;
            row_sum += expf(x - row_max);
        }
        for(int offset = GroupSize / 2; offset > 0; offset >>= 1)
            row_sum += __shfl_down(row_sum, offset, GroupSize);
        row_sum = __shfl(row_sum, 0, GroupSize);
        const float inv_sum = 1.0f / row_sum;

        for(int d = lane; d < D; d += GroupSize)
        {
            const float x = global_row < L
                                ? ck_tile::type_convert<float>(src[base + global_row * D + d])
                                : -INFINITY;
            dst[row * D + d] =
                ck_tile::type_convert<DataType>(expf(x - row_max) * inv_sum);
        }
    }
};



using LinearPolicy     = SlaLinearDefaultPolicy;
using LinearKBlockGemm = typename LinearPolicy::KBlockGemm;
using LinearQBlockGemm = typename LinearPolicy::QBlockGemm;

template <ActType Type, typename Policy = SlaLinearDefaultPolicy>
struct SlaLinearKFeatureComputePipeline
{
    using DataType = typename Policy::Problem::DataType;
    template <typename Kargs>
    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int D = Policy::kHeadDim;
        constexpr int Rows = Policy::kKRows;
        constexpr int DTile = 64;
        const int d0 = blockIdx.y * DTile;
        const int bh = blockIdx.z;
        const int tid = threadIdx.x;
        const int64_t base = static_cast<int64_t>(bh) * arg.L * D;
        __shared__ DataType stage[Rows][D];

        float total_sum = 0.0f;
        for(int row0 = blockIdx.x * Rows; row0 < arg.L; row0 += gridDim.x * Rows)
        {
            if constexpr(Type == ActType::SOFTMAX)
            {
                SlaLinearFeaturePipeline<Type, Policy>::template LoadSoftmax<Rows>(
                    arg.k, base, row0, arg.L, &stage[0][0]);
            }
            else
            {
                for(int i = tid; i < Rows * D; i += blockDim.x)
                {
                    const int r = i / D;
                    const int d = i % D;
                    const float x = row0 + r < arg.L
                                        ? ck_tile::type_convert<float>(arg.k[base + (row0 + r) * D + d])
                                        : 0.0f;
                    stage[r][d] = ck_tile::type_convert<DataType>(
                        SlaLinearFeaturePipeline<Type, Policy>::Apply(x));
                }
            }
            __syncthreads();

            if(tid < DTile)
            {
                const int d = d0 + tid;
                #pragma unroll
                for(int r = 0; r < Rows; ++r)
                {
                    if(row0 + r < arg.L)
                    {
                        const auto value = stage[r][d];
                        arg.k_feat[base + (row0 + r) * D + d] = value;
                        total_sum += ck_tile::type_convert<float>(value);
                    }
                }
            }
            __syncthreads();
        }
        if(tid < DTile) atomicAdd(arg.ksum + bh * D + d0 + tid, total_sum);

    }
};

template <int SplitL = LinearPolicy::kKvSplitL,
          typename Policy = SlaLinearDefaultPolicy>
struct SlaLinearKvSplitPipeline
{
    template <typename Kargs, typename AView, typename BView>
    CK_TILE_DEVICE void operator()(Kargs arg, const AView& a_view, const BView& b_view) const
    {
        constexpr int D = Policy::kHeadDim;
        constexpr int DTile = Policy::kKOutputTile;
        constexpr int KTile = Policy::kKTile;
        const int d_tile = blockIdx.x & 1;
        const int n_tile = blockIdx.x >> 1;
        const int split = blockIdx.y;
        const int bh = blockIdx.z;
        const int tiles = (arg.L + KTile - 1) / KTile;
        const int tile_begin = (tiles * split) / SplitL;
        const int tile_end = (tiles * (split + 1)) / SplitL;

        constexpr auto gemm = typename Policy::KBlockGemm{};
        auto acc = gemm.MakeCBlockTile();
        ck_tile::clear_tile(acc);
        for(int tile = tile_begin; tile < tile_end; ++tile)
        {
            const int l0 = tile * KTile;
            auto a = gemm.MakeABlockTile();
            auto a_win = ck_tile::make_tile_window(
                a_view, ck_tile::make_tuple(ck_tile::number<DTile>{}, ck_tile::number<KTile>{}),
                ck_tile::multi_index<2>{d_tile * DTile, l0}, a.get_tile_distribution());
            a = ck_tile::load_tile(a_win);

            auto b = gemm.MakeBBlockTile();
            auto b_win = ck_tile::make_tile_window(
                b_view, ck_tile::make_tuple(ck_tile::number<DTile>{}, ck_tile::number<KTile>{}),
                ck_tile::multi_index<2>{n_tile * DTile, l0}, b.get_tile_distribution());
            b = ck_tile::load_tile(b_win);
            gemm(acc, a, b);
        }

        auto out = gemm.MakeOuputLayout(acc);
        constexpr auto spans = decltype(out)::get_distributed_spans();
        ck_tile::sweep_tile_span(spans[ck_tile::number<0>{}], [&](auto i0) {
            ck_tile::sweep_tile_span(spans[ck_tile::number<1>{}], [&](auto i1) {
                constexpr auto idx = ck_tile::make_tuple(i0, i1);
                const auto xy = ck_tile::get_x_indices_from_distributed_indices(
                    out.get_tile_distribution(), idx);
                const int m = d_tile * DTile + xy.at(ck_tile::number<0>{});
                const int n = n_tile * DTile + xy.at(ck_tile::number<1>{});
                atomicAdd(arg.kv + (bh * D + m) * D + n, out[idx]);
            });
        });

    }
};

template <typename DataType>
struct SlaLinearKvGemmPipeline
{
    using GemmKernel = typename SlaLinearKvGemmPolicy<DataType>::Kernel;
    template <typename Kargs>
    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int D = SlaFusedLinearAttnProblem<DataType>::kHeadDim;
        const int bh = blockIdx.z;
        const int64_t input_offset = static_cast<int64_t>(bh) * arg.L * D;
        const int64_t output_offset = static_cast<int64_t>(bh) * D * D;
        typename GemmKernel::GemmCommonKargs gemm_args{arg.k_feat + input_offset,
                                                        arg.v + input_offset,
                                                        arg.kv + output_offset,
                                                        D,
                                                        D,
                                                        arg.L,
                                                        D,
                                                        D,
                                                        D};
        GemmKernel{}(gemm_args);

    }
};

template <typename DataType>
struct SlaLinearKvProjectionPipeline
{
    using GemmKernel = typename SlaLinearKvProjectionGemmPolicy<DataType>::Kernel;
    template <typename Kargs>
    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int D = SlaFusedLinearAttnProblem<DataType>::kHeadDim;
        const int bh = blockIdx.z;
        const int64_t offset = static_cast<int64_t>(bh) * D * D;
        typename GemmKernel::GemmCommonKargs gemm_args{arg.kv + offset,
                                                        arg.weight,
                                                        arg.kv_projected + offset,
                                                        D,
                                                        D,
                                                        D,
                                                        D,
                                                        D,
                                                        D};
        GemmKernel{}(gemm_args);

    }
};

template <ActType Type,
          int SplitL = LinearPolicy::kKvSplitL,
          typename Policy = SlaLinearDefaultPolicy>
struct SlaLinearKSplitPipeline
{
    using DataType = typename Policy::Problem::DataType;
    template <typename Kargs, typename VView>
    CK_TILE_DEVICE void operator()(Kargs arg, const VView& v_view) const
    {
    const auto* k = arg.k;
    const auto* v = arg.v;
    auto* partial_kv = arg.partial_kv;
    auto* partial_ksum = arg.partial_ksum;
    const int L = arg.L;
    constexpr int D = Policy::kHeadDim;
    constexpr int KTile = 32;
    constexpr int DTile = 64;
    const int d_tile = blockIdx.x & 1;
    const int n_tile = blockIdx.x >> 1;
    const int split = blockIdx.y;
    const int bh = blockIdx.z;
    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(bh) * L * D;

    __shared__ DataType k_stage[KTile][D];

    constexpr auto gemm = typename Policy::KBlockGemm{};
    auto acc = gemm.MakeCBlockTile();
    ck_tile::clear_tile(acc);
    float ksum_local = 0.0f;

    const int tiles = (L + KTile - 1) / KTile;
    const int tile_begin = (tiles * split) / SplitL;
    const int tile_end = (tiles * (split + 1)) / SplitL;

    for(int tile = tile_begin; tile < tile_end; ++tile)
    {
        const int l0 = tile * KTile;
        if constexpr(Type == ActType::SOFTMAX)
        {
            SlaLinearFeaturePipeline<Type, Policy>::template LoadSoftmax<KTile>(
                k, base, l0, L, &k_stage[0][0]);
        }
        else
        {
            for(int i = tid; i < KTile * D; i += blockDim.x)
            {
                const int r = i / D;
                const int d = i % D;
                const int l = l0 + r;
                const float x = l < L ? ck_tile::type_convert<float>(k[base + l * D + d]) : 0.0f;
                k_stage[r][d] = ck_tile::type_convert<DataType>(
                    SlaLinearFeaturePipeline<Type, Policy>::Apply(x));
            }
        }
        __syncthreads();

        if(n_tile == 0 && tid < DTile)
        {
            const int d = d_tile * DTile + tid;
            #pragma unroll
            for(int r = 0; r < KTile; ++r)
                if(l0 + r < L) ksum_local += ck_tile::type_convert<float>(k_stage[r][d]);
        }

        auto a_reg = gemm.MakeABlockTile();
        auto a_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::lds>(
            &k_stage[0][0], ck_tile::make_tuple(D, KTile), ck_tile::make_tuple(1, D),
            ck_tile::number<1>{}, ck_tile::number<1>{});
        auto a_win = ck_tile::make_tile_window(
            a_view, ck_tile::make_tuple(ck_tile::number<DTile>{}, ck_tile::number<KTile>{}),
            ck_tile::multi_index<2>{d_tile * DTile, 0}, a_reg.get_tile_distribution());
        a_reg = ck_tile::load_tile(a_win);

        auto b_reg = gemm.MakeBBlockTile();
        auto b_win = ck_tile::make_tile_window(
            v_view, ck_tile::make_tuple(ck_tile::number<DTile>{}, ck_tile::number<KTile>{}),
            ck_tile::multi_index<2>{n_tile * DTile, l0}, b_reg.get_tile_distribution());
        b_reg = ck_tile::load_tile(b_win);
        gemm(acc, a_reg, b_reg);
        __syncthreads();
    }

    if(n_tile == 0 && tid < DTile)
        partial_ksum[(static_cast<int64_t>(bh) * SplitL + split) * D +
                     d_tile * DTile + tid] = ksum_local;

    auto out = gemm.MakeOuputLayout(acc);
    constexpr auto spans = decltype(out)::get_distributed_spans();
    ck_tile::sweep_tile_span(spans[ck_tile::number<0>{}], [&](auto i0) {
        ck_tile::sweep_tile_span(spans[ck_tile::number<1>{}], [&](auto i1) {
            constexpr auto idx = ck_tile::make_tuple(i0, i1);
            const auto xy = ck_tile::get_x_indices_from_distributed_indices(out.get_tile_distribution(), idx);
            const int m = d_tile * DTile + xy.at(ck_tile::number<0>{});
            const int n = n_tile * DTile + xy.at(ck_tile::number<1>{});
            partial_kv[((static_cast<int64_t>(bh) * SplitL + split) * D + m) * D + n] =
                out[idx];
        });
    });

    }
};

template <typename DataType, int SplitL = LinearPolicy::kKvSplitL>
struct SlaLinearKSplitReducePipeline
{
    template <typename Kargs>
    CK_TILE_DEVICE void operator()(Kargs arg) const
    {
        constexpr int D = SlaFusedLinearAttnProblem<DataType>::kHeadDim;
        constexpr int matrix_values = D * D;
        constexpr int values_per_head = matrix_values + D;
        const int bh = blockIdx.y;
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= values_per_head) return;

        float value = 0.0f;
        if(index < matrix_values)
        {
            #pragma unroll
            for(int split = 0; split < SplitL; ++split)
                value += arg.partial_kv[
                    (static_cast<int64_t>(bh) * SplitL + split) * matrix_values + index];
            arg.kv[static_cast<int64_t>(bh) * matrix_values + index] =
                ck_tile::type_convert<DataType>(value);
        }
        else
        {
            const int d = index - matrix_values;
            #pragma unroll
            for(int split = 0; split < SplitL; ++split)
                value += arg.partial_ksum[
                    (static_cast<int64_t>(bh) * SplitL + split) * D + d];
            arg.ksum[static_cast<int64_t>(bh) * D + d] = value;
        }

    }
};

template <ActType Type, typename QPolicy = LinearPolicy>
struct SlaLinearQFusedPipeline
{
    using DataType = typename QPolicy::Problem::DataType;
    template <typename Kargs, typename KvView>
    CK_TILE_DEVICE void operator()(Kargs arg, const KvView& kv_view) const
    {
    const auto* q = arg.q;
    const auto* kv = arg.kv;
    const auto* ksum = arg.ksum;
    const auto* bias = arg.bias;
    const auto* os = arg.os;
    auto* output = arg.output;
    const int L = arg.L;
    constexpr int D = QPolicy::kHeadDim;
    constexpr int MTile = QPolicy::kQRows;
    constexpr int NTile = D;
    constexpr int KTile = 32;
    const int m0 = blockIdx.x * MTile;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t q_base = static_cast<int64_t>(bh) * L * D;

    __shared__ DataType stage[MTile][D];
    __shared__ float denom[MTile];
    __shared__ float ksum_stage[D];

    if(tid < D) ksum_stage[tid] = ksum[bh * D + tid];

    if constexpr(Type == ActType::SOFTMAX)
    {
        SlaLinearFeaturePipeline<Type, QPolicy>::template LoadSoftmax<MTile>(
            q, q_base, m0, L, &stage[0][0]);
    }
    else
    {
        for(int i = tid; i < MTile * D; i += blockDim.x)
        {
            const int r = i / D;
            const int d = i % D;
            const float x = m0 + r < L ? ck_tile::type_convert<float>(q[q_base + (m0 + r) * D + d]) : 0.0f;
            stage[r][d] = ck_tile::type_convert<DataType>(
                SlaLinearFeaturePipeline<Type, QPolicy>::Apply(x));
        }
    }
    __syncthreads();

    constexpr int DenomLanes = 4;
    if(tid < MTile * DenomLanes)
    {
        const int denom_row = tid / DenomLanes;
        const int denom_lane = tid % DenomLanes;
        float dsum = 0.0f;
        #pragma unroll
        for(int d = denom_lane; d < D; d += DenomLanes)
            dsum += ck_tile::type_convert<float>(stage[denom_row][d]) * ksum_stage[d];
        dsum += __shfl_down(dsum, 2, DenomLanes);
        dsum += __shfl_down(dsum, 1, DenomLanes);
        if(denom_lane == 0) denom[denom_row] = 1.0f / (dsum + 1.0e-5f);
    }
    __syncthreads();

    constexpr auto gemm = typename QPolicy::QBlockGemm{};
    auto num = gemm.MakeCBlockTile();
    ck_tile::clear_tile(num);

    for(int k0 = 0; k0 < D; k0 += KTile)
    {
        auto a = gemm.MakeABlockTile();
        auto a_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::lds>(
            &stage[0][0], ck_tile::make_tuple(MTile, D), ck_tile::make_tuple(D, 1),
            ck_tile::number<1>{}, ck_tile::number<1>{});
        auto a_win = ck_tile::make_tile_window(
            a_view, ck_tile::make_tuple(ck_tile::number<MTile>{}, ck_tile::number<KTile>{}),
            ck_tile::multi_index<2>{0, k0}, a.get_tile_distribution());
        a = ck_tile::load_tile(a_win);

        auto b = gemm.MakeBBlockTile();
        auto b_win = ck_tile::make_tile_window(
            kv_view, ck_tile::make_tuple(ck_tile::number<NTile>{}, ck_tile::number<KTile>{}),
            ck_tile::multi_index<2>{0, k0}, b.get_tile_distribution());
        b = ck_tile::load_tile(b_win);
        gemm(num, a, b);
    }

    auto ntile = gemm.MakeOuputLayout(num);
    constexpr auto nspans = decltype(ntile)::get_distributed_spans();

    if constexpr(Type != ActType::SOFTMAX)
    {
        ck_tile::sweep_tile_span(nspans[ck_tile::number<0>{}], [&](auto i0) {
            ck_tile::sweep_tile_span(nspans[ck_tile::number<1>{}], [&](auto i1) {
                constexpr auto idx = ck_tile::make_tuple(i0, i1);
                const auto xy = ck_tile::get_x_indices_from_distributed_indices(
                    ntile.get_tile_distribution(), idx);
                const int r = xy.at(ck_tile::number<0>{});
                const int n = xy.at(ck_tile::number<1>{});
                if(m0 + r < L)
                {
                    const auto normalized = ck_tile::type_convert<DataType>(
                        ntile[idx] * denom[r]);
                    float value = ck_tile::type_convert<float>(normalized);
                    if(bias) value += ck_tile::type_convert<float>(bias[n]);
                    if(os)
                        value += ck_tile::type_convert<float>(
                            os[q_base + (m0 + r) * D + n]);
                    output[q_base + (m0 + r) * D + n] =
                        ck_tile::type_convert<DataType>(value);
                }
            });
        });
        return;
    }

    auto store_norm = [&](auto& tile) {
        ck_tile::sweep_tile_span(nspans[ck_tile::number<0>{}], [&](auto i0) {
            ck_tile::sweep_tile_span(nspans[ck_tile::number<1>{}], [&](auto i1) {
                constexpr auto idx = ck_tile::make_tuple(i0, i1);
                const auto xy = ck_tile::get_x_indices_from_distributed_indices(tile.get_tile_distribution(), idx);
                const int r = xy.at(ck_tile::number<0>{});
                const int n = xy.at(ck_tile::number<1>{});
                stage[r][n] = ck_tile::type_convert<DataType>(tile[idx] * denom[r]);
            });
        });
    };
    store_norm(ntile);
    __syncthreads();

    if constexpr(Type == ActType::SOFTMAX)
    {
        for(int i = tid; i < MTile * D; i += blockDim.x)
        {
            const int r = i / D;
            const int n = i % D;
            if(m0 + r < L)
            {
                float value = ck_tile::type_convert<float>(stage[r][n]);
                if(bias) value += ck_tile::type_convert<float>(bias[n]);
                if(os)
                    value += ck_tile::type_convert<float>(
                        os[q_base + (m0 + r) * D + n]);
                output[q_base + (m0 + r) * D + n] =
                    ck_tile::type_convert<DataType>(value);
            }
        }
    }

    }
};

} // namespace sla
