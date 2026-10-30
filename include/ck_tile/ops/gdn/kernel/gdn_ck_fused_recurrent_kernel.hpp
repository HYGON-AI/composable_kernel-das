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
    // Paged state support: when `state_indices` is null the recurrent state is
    // dense -- slot `i` of the pool is sequence `i`. Otherwise the pool is
    // indexed by the block table and slot `state_indices[sequence]` is used;
    // `state_slot_stride` is then the element distance between two slots.
    const int64_t* state_indices;

    ck_tile::index_t batch;
    ck_tile::index_t time;
    ck_tile::index_t qk_heads;
    ck_tile::index_t value_heads;
    ck_tile::index_t key_dim;
    ck_tile::index_t value_dim;
    ck_tile::index_t sequences;
    // Element distance between two consecutive state-pool slots (dense: h_v *
    // key_dim * value_dim) and the slot this CTA resolved for its sequence.
    ck_tile::long_index_t state_slot_stride;
    ck_tile::long_index_t state_slot;

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

template <typename Policy, bool ScalarGHeadwiseFastPath = false, bool RawGateFastPath = false, bool BetaSigmoid = false>
struct GdnFusedRecurrentKernel
{
    using Kargs = GdnFusedRecurrentKargs;
    using Pipeline = ck_tile::GdnFusedRecurrentPipeline<
        Policy, ScalarGHeadwiseFastPath, BetaSigmoid, RawGateFastPath>;

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
        if constexpr(Policy::kStateVector > 1)
        {
            args.transpose_state = true;
            args.key_dim = 128;
            args.value_dim = 128;
        }
        if constexpr(RawGateFastPath)
        {
            constexpr bool bf = std::is_same_v<typename Policy::Problem::QKDataType, ck_tile::bf16_t>;
            args.use_qk_l2norm = Policy::kNormalize;
            args.g_dtype = args.dt_bias_dtype = bf ? 1 : 0;
            args.beta_dtype = args.a_log_dtype = 2;
            args.use_g = args.beta_headwise = args.use_initial_state = true;
            args.store_final_state = args.gate_in_kernel = args.has_dt_bias = true;
            args.use_gk = args.use_gv = args.use_exp2 = false;
        }
        const ck_tile::index_t value_tile =
            static_cast<ck_tile::index_t>(blockIdx.x);
        const ck_tile::index_t sequence_head =
            static_cast<ck_tile::index_t>(blockIdx.y);
        const ck_tile::index_t sequence =
            sequence_head / args.value_heads;
        if(sequence >= args.sequences)
            return;

        // Resolve this sequence's recurrent-state slot and hand it to the
        // pipeline, which only ever needs `state_slot` + `state_slot_stride` to
        // address the pool. Dense callers (state_indices == nullptr) keep the
        // historical "slot == sequence" mapping, so their numerics are
        // untouched. In pool mode a non-positive id is the framework's
        // NULL_BLOCK_ID: such a row has no state at all, so it runs from a zero
        // state and must not write a slot back.
        ck_tile::long_index_t state_slot =
            static_cast<ck_tile::long_index_t>(sequence);
        if(args.state_indices != nullptr)
        {
            const ck_tile::long_index_t state_index =
                static_cast<ck_tile::long_index_t>(args.state_indices[sequence]);
            if(state_index > 0)
            {
                state_slot = state_index;
            }
            else
            {
                args.use_initial_state = false;
                args.store_final_state = false;
            }
        }
        args.state_slot = state_slot;
        if(args.state_slot_stride <= 0)
        {
            args.state_slot_stride = static_cast<ck_tile::long_index_t>(args.value_heads) *
                                     args.key_dim * args.value_dim;
        }

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
