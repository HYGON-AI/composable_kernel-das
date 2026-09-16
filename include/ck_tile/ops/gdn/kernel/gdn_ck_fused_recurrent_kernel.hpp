// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include <hip/hip_runtime.h>

enum class GdnTensorDtype : int
{
    Float16  = 0,
    BFloat16 = 1,
    Float32  = 2,
};

constexpr int gdn_tensor_dtype_code(GdnTensorDtype dtype)
{
    return static_cast<int>(dtype);
}

struct GdnFusedRecurrentKargs
{
    const void* q;
    const void* k;
    const void* v;
    const void* g;
    const void* gk;
    const void* gv;
    const void* beta;
    const void* a_log;
    const void* dt_bias;
    const float* initial_state;
    const int64_t* cu_seqlens;
    void* output;
    float* final_state;

    ck_tile::index_t batch;
    ck_tile::index_t time;
    ck_tile::index_t qk_heads;
    ck_tile::index_t value_heads;
    ck_tile::index_t key_dim;
    ck_tile::index_t value_dim;
    ck_tile::index_t sequences;

    float scale;
    int g_dtype;
    int gk_dtype;
    int gv_dtype;
    int beta_dtype;
    int a_log_dtype;
    int dt_bias_dtype;

    bool use_g;
    bool use_gk;
    bool use_gv;
    bool beta_headwise;
    bool use_initial_state;
    bool store_final_state;
    bool use_qk_l2norm;
    bool use_exp2;
    bool transpose_state;
    bool variable_length;
    bool gate_in_kernel;
    bool has_dt_bias;
};

#include "ck_tile/ops/gdn/pipeline/gdn_ck_fused_recurrent_pipeline.hpp"
#include "ck_tile/host.hpp"

template <typename Policy, bool ScalarGHeadwiseFastPath = false>
struct GdnFusedRecurrentKernel
{
    using Kargs = GdnFusedRecurrentKargs;
    using Pipeline = ck_tile::GdnFusedRecurrentPipeline<
        Policy, ScalarGHeadwiseFastPath>;

    CK_TILE_HOST static constexpr dim3 GridSize(const Kargs& args)
    {
        return dim3(
            static_cast<unsigned int>(
                (args.value_dim + Policy::kValueTile - 1) /
                Policy::kValueTile),
            static_cast<unsigned int>(
                args.sequences * args.value_heads),
            1);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize()
    {
        return dim3(Policy::kBlockSize, 1, 1);
    }

    CK_TILE_DEVICE void operator()(Kargs args) const
    {
        const ck_tile::index_t value_tile =
            static_cast<ck_tile::index_t>(blockIdx.x);
        const ck_tile::index_t sequence_head =
            static_cast<ck_tile::index_t>(blockIdx.y);
        const ck_tile::index_t sequence =
            sequence_head / args.value_heads;
        if(sequence >= args.sequences)
            return;

        ck_tile::long_index_t bos = 0;
        ck_tile::long_index_t eos = 0;
        if(args.variable_length)
        {
            bos = args.cu_seqlens[sequence];
            eos = args.cu_seqlens[sequence + 1];
        }
        else
        {
            bos = static_cast<ck_tile::long_index_t>(sequence) *
                  args.time;
            eos = bos + args.time;
        }

        Pipeline{}(
            args,
            sequence_head,
            value_tile * Policy::kValueTile,
            bos,
            eos);
    }
};
