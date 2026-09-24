// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_ck_fused_recurrent_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Policy, bool ScalarGHeadwiseFastPath = false, bool BetaSigmoid = false, bool RawGateFastPath = false>
struct GdnFusedRecurrentPipeline
{
    using Problem = typename Policy::Problem;
    using QKDataType = typename Problem::QKDataType;
    using VDataType = typename Problem::VDataType;
    using AccDataType = typename Problem::AccDataType;

    static constexpr index_t kValueTile = Policy::kValueTile;
    static constexpr index_t kKeyTile = Policy::kKeyTile;

    CK_TILE_DEVICE static float load_as_float(const void* pointer,
                                              long_index_t offset,
                                              int dtype)
    {
        if(dtype == 0)
            return type_convert<float>(
                reinterpret_cast<const half_t*>(pointer)[offset]);
        if(dtype == 1)
            return type_convert<float>(
                reinterpret_cast<const bf16_t*>(pointer)[offset]);
        return reinterpret_cast<const float*>(pointer)[offset];
    }

    // Stable softplus for the common raw-gate path. The low branch avoids
    // cancellation in log(1 + exp(x)); eight Taylor terms at exp(x)<exp(-2)
    // have a relative truncation error below 2e-8. High/NaN/Inf follow the
    // original threshold contract. BF16 conversion still uses STANDARD RNE.
    CK_TILE_DEVICE static float raw_gate_softplus(float x)
    {
        if(x >= 20.0f || x != x) return x;
        const float y = __expf(x);
        if(x < -2.0f)
        {
            float p = -1.0f / 8.0f;
            p = fmaf(p,y,1.0f/7.0f);
            p = fmaf(p,y,-1.0f/6.0f);
            p = fmaf(p,y,1.0f/5.0f);
            p = fmaf(p,y,-1.0f/4.0f);
            p = fmaf(p,y,1.0f/3.0f);
            p = fmaf(p,y,-1.0f/2.0f);
            return y * fmaf(p,y,1.0f);
        }
        return __logf(1.0f + y);
    }

    CK_TILE_DEVICE static float gate_exp(float value, bool use_exp2)
    {
        return use_exp2 ? exp2f(value) : expf(value);
    }

    CK_TILE_DEVICE auto load_qk_tile(const QKDataType* pointer,
                                     long_index_t base,
                                     index_t key_dim) const
    {
        auto view = make_naive_tensor_view<address_space_enum::global>(
            pointer + base,
            make_tuple(number<1>{}, key_dim),
            make_tuple(number<0>{}, number<1>{}),
            number<Policy::kQKVector>{},
            number<1>{});
        auto window = make_tile_window(
            view,
            make_tuple(number<1>{}, number<kKeyTile>{}),
            multi_index<2>{0, 0},
            Policy::MakeQKDistribution());
        return cast_tile<AccDataType>(load_tile(window));
    }

    CK_TILE_DEVICE auto load_qk_aux_tile(const void* pointer,
                                         int dtype,
                                         long_index_t base,
                                         index_t key_dim) const
    {
        auto tile =
            make_static_distributed_tensor<AccDataType>(
                Policy::MakeQKDistribution());
        constexpr auto spans = decltype(tile)::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto dstr_idx = make_tuple(idx0, idx1);
                const auto x_idx = get_x_indices_from_distributed_indices(
                    tile.get_tile_distribution(), dstr_idx);
                const index_t key_index = x_idx.at(number<1>{});
                tile(dstr_idx) =
                    key_index < key_dim
                        ? load_as_float(pointer, base + key_index, dtype)
                        : 0.0f;
            });
        });
        return tile;
    }

    template <typename ValueDistribution>
    CK_TILE_DEVICE auto load_value_aux_tile(const void* pointer,
                                            int dtype,
                                            long_index_t base,
                                            index_t valid_values,
                                            const ValueDistribution& distribution,
                                            float invalid_value) const
    {
        auto tile =
            make_static_distributed_tensor<AccDataType>(distribution);
        constexpr auto spans = decltype(tile)::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            constexpr auto dstr_idx = make_tuple(idx0);
            const auto x_idx = get_x_indices_from_distributed_indices(
                tile.get_tile_distribution(), dstr_idx);
            const index_t value_index = x_idx.at(number<0>{});
            tile(dstr_idx) =
                value_index < valid_values
                    ? load_as_float(pointer, base + value_index, dtype)
                    : invalid_value;
        });
        return tile;
    }

    template <typename ValueDistribution>
    CK_TILE_DEVICE auto load_value_tile(const VDataType* pointer,
                                        long_index_t base,
                                        index_t valid_values,
                                        const ValueDistribution& distribution) const
    {
        auto view = make_naive_tensor_view<address_space_enum::global>(
            pointer + base,
            make_tuple(valid_values),
            make_tuple(number<1>{}),
            number<1>{},
            number<1>{});
        auto window = make_tile_window(
            view,
            make_tuple(number<kValueTile>{}),
            multi_index<1>{0},
            distribution);
        return cast_tile<AccDataType>(load_tile(window));
    }

    CK_TILE_DEVICE auto load_initial_state(const GdnFusedRecurrentKargs& args,
                                           index_t sequence_head,
                                           index_t value_begin,
                                           index_t valid_values) const
    {
        auto state =
            make_static_distributed_tensor<AccDataType>(
                Policy::MakeStateDistribution());
        if constexpr(RawGateFastPath && Policy::kValueTile == 16 && Policy::kBlockSize == 128)
        {
            // The two-wave raw-gate specialization owns complete V16/K128
            // tiles. Consecutive four-key vectors are aligned; each state
            // element has one owner, including in-place final-state updates.
            using Vec = float __attribute__((ext_vector_type(4)));
            constexpr int repeats = decltype(state)::get_thread_buffer_size() / 8;
            const int lane = get_lane_id();
            const int row = get_warp_id() * 4 + lane / 16;
            const long_index_t base = static_cast<long_index_t>(sequence_head) * 128 * 128;
            static_for<0, repeats, 1>{}([&](auto vr) {
                static_for<0, 2, 1>{}([&](auto kr) {
                    const auto* p = reinterpret_cast<const Vec*>(args.initial_state + base +
                        (value_begin + row + vr * (Policy::kBlockSize / 16)) * 128 +
                        kr * 64 + (lane % 16) * 4);
                    state.get_thread_buffer().template set_as<Vec>(number<vr * 2 + kr>{},
                        __builtin_nontemporal_load(p));
                });
            });
            return state;
        }

        tile_elementwise_inout(
            [](auto& value) { value = 0.0f; }, state);
        if(!args.use_initial_state)
            return state;

        const long_index_t head_base =
            static_cast<long_index_t>(sequence_head) *
            args.key_dim * args.value_dim;
        const index_t stride_v =
            args.transpose_state ? args.key_dim : 1;
        const index_t stride_k =
            args.transpose_state ? 1 : args.value_dim;
        const long_index_t value_offset =
            static_cast<long_index_t>(value_begin) * stride_v;
        auto view = make_naive_tensor_view<address_space_enum::global>(
            args.initial_state + head_base + value_offset,
            make_tuple(valid_values, args.key_dim),
            make_tuple(stride_v, stride_k),
            number<Policy::kStateVector>{},
            number<1>{});
        auto window = make_tile_window(
            view,
            make_tuple(number<kValueTile>{}, number<kKeyTile>{}),
            multi_index<2>{0, 0},
            Policy::MakeStateDistribution());
        load_tile(state, window);
        return state;
    }

    template <typename QKTile>
    CK_TILE_DEVICE static void l2_normalize(QKTile& tile)
    {
        if constexpr(Policy::kStateVector > 1)
        {
            float sum = 0.0f;
            static_for<0, QKTile::get_thread_buffer_size(), 1>{}([&](auto i) {
                const float x = tile.get_thread_buffer()[i];
                sum += x * x;
            });
            // Q/K are replicated across the four value rows and waves.
            // Only the 16 key lanes participate in the norm reduction.
            static_for<0, 4, 1>{}([&](auto i) { sum += __shfl_xor(sum, 1 << i, 16); });
            const float inv = 1.0f / sqrt(sum + 1.0e-6f);
            tile_elementwise_inout([&](auto& x) { x *= inv; }, tile);
        }
        else
        {
        const auto square = tile_elementwise_in(
            [](auto value) { return value * value; }, tile);
        const auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
        auto sum = block_tile_reduce<AccDataType>(
            square, sequence<1>{}, add, AccDataType{0});
        block_tile_reduce_xor_sync(sum, add);

        constexpr auto tile_spans = QKTile::get_distributed_spans();
        constexpr auto sum_spans = decltype(sum)::get_distributed_spans();
        sweep_tile_span(sum_spans[number<0>{}], [&](auto sum_idx0) {
            constexpr auto sum_idx = make_tuple(sum_idx0);
            const float inv_norm =
                1.0f / sqrt(sum(sum_idx) + 1.0e-6f);
            sweep_tile_span(tile_spans[number<0>{}], [&](auto tile_idx0) {
                sweep_tile_span(tile_spans[number<1>{}], [&](auto tile_idx1) {
                    constexpr auto tile_idx =
                        make_tuple(tile_idx0, tile_idx1);
                    tile(tile_idx) *= inv_norm;
                });
            });
        });
    }

    }

    template <typename StateTile, typename QKTile>
    CK_TILE_DEVICE static auto reduce_state_dot(const StateTile& state,
                                                const QKTile& vector)
    {
        auto product =
            make_static_distributed_tensor<AccDataType>(
                state.get_tile_distribution());
        constexpr auto state_spans = StateTile::get_distributed_spans();
        constexpr auto qk_spans = QKTile::get_distributed_spans();
        sweep_tile_span(qk_spans[number<0>{}], [&](auto qk_idx0) {
            sweep_tile_span(state_spans[number<0>{}], [&](auto value_idx) {
                sweep_tile_span(state_spans[number<1>{}], [&](auto key_idx) {
                    constexpr auto state_idx =
                        make_tuple(value_idx, key_idx);
                    constexpr auto qk_idx =
                        make_tuple(qk_idx0, key_idx);
                    product(state_idx) =
                        state[state_idx] * vector[qk_idx];
                });
            });
        });

        const auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
        auto result = block_tile_reduce<AccDataType>(
            product, sequence<1>{}, add, AccDataType{0});
        block_tile_reduce_xor_sync(result, add);
        return result;
    }

    template <typename StateTile,
              typename QKTile,
              typename ValueTile>
    CK_TILE_DEVICE static void rank_one_update(StateTile& state,
                                               const QKTile& key,
                                               const ValueTile& update)
    {
        constexpr auto state_spans = StateTile::get_distributed_spans();
        constexpr auto qk_spans = QKTile::get_distributed_spans();
        sweep_tile_span(qk_spans[number<0>{}], [&](auto qk_idx0) {
            sweep_tile_span(state_spans[number<0>{}], [&](auto value_idx) {
                constexpr auto update_idx = make_tuple(value_idx);
                const float value_update = update[update_idx];
                sweep_tile_span(state_spans[number<1>{}], [&](auto key_idx) {
                    constexpr auto state_idx =
                        make_tuple(value_idx, key_idx);
                    constexpr auto qk_idx =
                        make_tuple(qk_idx0, key_idx);
                    state(state_idx) += key[qk_idx] * value_update;
                });
            });
        });
    }

    template <typename StateTile,
              typename QKTile,
              typename ValueTile>
    CK_TILE_DEVICE static void apply_decay(StateTile& state,
                                           const QKTile& key_decay,
                                           const ValueTile& value_decay,
                                           float scalar_decay)
    {
        constexpr auto state_spans = StateTile::get_distributed_spans();
        constexpr auto qk_spans = QKTile::get_distributed_spans();
        sweep_tile_span(qk_spans[number<0>{}], [&](auto qk_idx0) {
            sweep_tile_span(state_spans[number<0>{}], [&](auto value_idx) {
                constexpr auto value_dstr_idx = make_tuple(value_idx);
                const float v_decay = value_decay[value_dstr_idx];
                sweep_tile_span(state_spans[number<1>{}], [&](auto key_idx) {
                    constexpr auto state_idx =
                        make_tuple(value_idx, key_idx);
                    constexpr auto qk_idx =
                        make_tuple(qk_idx0, key_idx);
                    state(state_idx) *=
                        scalar_decay * key_decay[qk_idx] * v_decay;
                });
            });
        });
    }

    template <typename ValueTile>
    CK_TILE_DEVICE void store_output_tile(const GdnFusedRecurrentKargs& args,
                                          const ValueTile& output,
                                          long_index_t base,
                                          index_t valid_values) const
    {
        auto typed_output = cast_tile<VDataType>(output);
        auto view = make_naive_tensor_view<address_space_enum::global>(
            reinterpret_cast<VDataType*>(args.output) + base,
            make_tuple(valid_values),
            make_tuple(number<1>{}),
            number<1>{},
            number<1>{});
        auto window = make_tile_window(
            view,
            make_tuple(number<kValueTile>{}),
            multi_index<1>{0},
            output.get_tile_distribution());
        store_tile(window, typed_output);
    }

    template <typename StateTile>
    CK_TILE_DEVICE void store_final_state(const GdnFusedRecurrentKargs& args,
                                          const StateTile& state,
                                          index_t sequence_head,
                                          index_t value_begin,
                                          index_t valid_values) const
    {

        if constexpr(RawGateFastPath && Policy::kValueTile == 16 && Policy::kBlockSize == 128)
        {
            using Vec = float __attribute__((ext_vector_type(4)));
            constexpr int repeats = StateTile::get_thread_buffer_size() / 8;
            const int lane = get_lane_id();
            const int row = get_warp_id() * 4 + lane / 16;
            const long_index_t base = static_cast<long_index_t>(sequence_head) * 128 * 128;
            static_for<0, repeats, 1>{}([&](auto vr) {
                static_for<0, 2, 1>{}([&](auto kr) {
                    auto* p = reinterpret_cast<Vec*>(args.final_state + base +
                        (value_begin + row + vr * (Policy::kBlockSize / 16)) * 128 +
                        kr * 64 + (lane % 16) * 4);
                    __builtin_nontemporal_store(
                        state.get_thread_buffer().template get_as<Vec>()[number<vr * 2 + kr>{}], p);
                });
            });
            return;
        }
        if(!args.store_final_state)
            return;
        const long_index_t head_base =
            static_cast<long_index_t>(sequence_head) *
            args.key_dim * args.value_dim;
        const index_t stride_v =
            args.transpose_state ? args.key_dim : 1;
        const index_t stride_k =
            args.transpose_state ? 1 : args.value_dim;
        const long_index_t value_offset =
            static_cast<long_index_t>(value_begin) * stride_v;
        auto view = make_naive_tensor_view<address_space_enum::global>(
            args.final_state + head_base + value_offset,
            make_tuple(valid_values, args.key_dim),
            make_tuple(stride_v, stride_k),
            number<Policy::kStateVector>{},
            number<1>{});
        auto window = make_tile_window(
            view,
            make_tuple(number<kValueTile>{}, number<kKeyTile>{}),
            multi_index<2>{0, 0},
            state.get_tile_distribution());
        store_tile(window, state);
    }

    CK_TILE_DEVICE void operator()(const GdnFusedRecurrentKargs& args,
                                   index_t sequence_head,
                                   index_t value_begin,
                                   long_index_t bos,
                                   long_index_t eos) const
    {
        const index_t value_head = sequence_head % args.value_heads;
        const index_t qk_head =
            value_head / (args.value_heads / args.qk_heads);
        const index_t valid_values =
            Policy::kStateVector > 1 ? kValueTile : min(args.value_dim - value_begin, kValueTile);

        auto state = load_initial_state(
            args, sequence_head, value_begin, valid_values);

        for(long_index_t token = bos; token < eos; ++token)
        {
            const long_index_t qk_base =
                (token * args.qk_heads + qk_head) * args.key_dim;
            auto query = load_qk_tile(
                reinterpret_cast<const QKDataType*>(args.q),
                qk_base,
                args.key_dim);
            auto key = load_qk_tile(
                reinterpret_cast<const QKDataType*>(args.k),
                qk_base,
                args.key_dim);
            if constexpr(!ScalarGHeadwiseFastPath)
            {
                if(args.use_qk_l2norm)
                {
                    l2_normalize(query);
                    l2_normalize(key);
                }
            }
            tile_elementwise_inout(
                [&](auto& value) { value *= args.scale; }, query);

            float scalar_decay = 1.0f;
            const long_index_t head_offset =
                token * args.value_heads + value_head;
            if constexpr(ScalarGHeadwiseFastPath)
            {
                // This specialization is selected only for the common GDN
                // inference path: FP32 scalar g, no gk/gv, and exp().  Keep
                // the hot token loop free of runtime dtype/feature branches.
                scalar_decay = expf(
                    reinterpret_cast<const float*>(args.g)[head_offset]);
            }
            else if(args.use_g)
            {
                float gate =
                    load_as_float(args.g, head_offset, args.g_dtype);
                if(args.gate_in_kernel)
                {
                    if(args.has_dt_bias)
                        gate += load_as_float(
                            args.dt_bias,
                            value_head,
                            args.dt_bias_dtype);
                    const float a = load_as_float(
                        args.a_log, value_head, args.a_log_dtype);
                    const float softplus =
                        RawGateFastPath ? raw_gate_softplus(gate) :
                        (gate < 20.0f ? log1pf(expf(gate)) : gate);
                    scalar_decay = RawGateFastPath ? __expf(-__expf(a) * softplus) : expf(-expf(a) * softplus);
                }
                else
                {
                    scalar_decay = gate_exp(gate, args.use_exp2);
                }
            }

            if constexpr(ScalarGHeadwiseFastPath)
            {
                tile_elementwise_inout(
                    [&](auto& value) { value *= scalar_decay; }, state);
            }
            else
            {
                auto key_decay =
                    make_static_distributed_tensor<AccDataType>(
                        Policy::MakeQKDistribution());
                tile_elementwise_inout(
                    [](auto& value) { value = 1.0f; }, key_decay);
                if(args.use_gk)
                {
                    key_decay = load_qk_aux_tile(
                        args.gk,
                        args.gk_dtype,
                        head_offset * args.key_dim,
                        args.key_dim);
                    tile_elementwise_inout(
                        [&](auto& value) {
                            value = gate_exp(value, args.use_exp2);
                        },
                        key_decay);
                }

                auto value_decay =
                    make_static_distributed_tensor<AccDataType>(
                        Policy::MakeValueDistribution());
                tile_elementwise_inout(
                    [](auto& value) { value = 1.0f; }, value_decay);
                if(args.use_gv)
                {
                    value_decay = load_value_aux_tile(
                        args.gv,
                        args.gv_dtype,
                        head_offset * args.value_dim + value_begin,
                        valid_values,
                        value_decay.get_tile_distribution(),
                        1.0f);
                    tile_elementwise_inout(
                        [&](auto& value) {
                            value = gate_exp(value, args.use_exp2);
                        },
                        value_decay);
                }
                apply_decay(
                    state, key_decay, value_decay, scalar_decay);
            }

            auto prediction = reduce_state_dot(state, key);
            const long_index_t value_base =
                head_offset * args.value_dim + value_begin;
            auto values = load_value_tile(
                reinterpret_cast<const VDataType*>(args.v),
                value_base,
                valid_values,
                prediction.get_tile_distribution());
            const float headwise_beta =
                ScalarGHeadwiseFastPath
                    ? type_convert<float>(
                          reinterpret_cast<const bf16_t*>(args.beta)[head_offset])
                    : 0.0f;
            auto update =
                make_static_distributed_tensor<AccDataType>(
                    prediction.get_tile_distribution());
            constexpr auto update_spans =
                decltype(update)::get_distributed_spans();
            sweep_tile_span(
                update_spans[number<0>{}], [&](auto value_idx) {
                    constexpr auto dstr_idx = make_tuple(value_idx);
                    const auto x_idx =
                        get_x_indices_from_distributed_indices(
                            update.get_tile_distribution(), dstr_idx);
                    const index_t local_value =
                        x_idx.at(number<0>{});
                    float beta = headwise_beta;
                    if constexpr(!ScalarGHeadwiseFastPath)
                    {
                        beta = 0.0f;
                        if(local_value < valid_values)
                        {
                            const long_index_t beta_offset =
                                args.beta_headwise
                                    ? head_offset
                                    : value_base + local_value;
                            beta = load_as_float(
                                args.beta, beta_offset, args.beta_dtype);
                        }
                    }
                    if constexpr(BetaSigmoid)
                        beta = 1.0f / (1.0f + (RawGateFastPath ? __expf(-beta) : expf(-beta)));
                    update(dstr_idx) =
                        beta * (values[dstr_idx] - prediction[dstr_idx]);
                });

            rank_one_update(state, key, update);
            auto output = reduce_state_dot(state, query);
            store_output_tile(
                args, output, value_base, valid_values);
        }

        store_final_state(
            args, state, sequence_head, value_begin, valid_values);
    }
};

} // namespace ck_tile
