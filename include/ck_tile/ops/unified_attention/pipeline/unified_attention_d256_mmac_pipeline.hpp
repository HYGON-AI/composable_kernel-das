// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/ops/unified_attention/block/unified_attention_d256_mmac_block_gemm.hpp"
#include "ck_tile/ops/reduce.hpp"
#include <cstdint>

inline constexpr ck_tile::index_t kUnifiedAttentionD256MmacVectorWidth = 8;

CK_TILE_DEVICE float unified_attention_d256_fast_exp2(float x)
{
    return __builtin_amdgcn_exp2f(x);
}

CK_TILE_DEVICE void unified_attention_d256_block_sync_lds_light()
{
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}

template <int M_, int K_>
CK_TILE_HOST_DEVICE constexpr auto make_phase1_copy_distribution()
{
    return ck_tile::tile_distribution_encoding_pattern_2d<
        256,
        M_,
        K_,
        kUnifiedAttentionD256MmacVectorWidth,
        ck_tile::tile_distribution_pattern::thread_raked,
        1>::make_2d_static_tile_distribution();
}

template <int N_, int K_>
CK_TILE_HOST_DEVICE constexpr auto make_phase2_v_lds_descriptor()
{
    constexpr ck_tile::index_t kKPack = kUnifiedAttentionD256MmacVectorWidth;
    constexpr ck_tile::index_t kPixelsPerRow = 64;
    constexpr ck_tile::index_t kNPerRow = kPixelsPerRow / kKPack;
    static_assert(N_ % kNPerRow == 0);
    static_assert(K_ % kKPack == 0);

    constexpr auto desc0 = ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(ck_tile::number<K_ / kKPack>{},
                            ck_tile::number<N_ / kNPerRow>{},
                            ck_tile::number<kNPerRow>{},
                            ck_tile::number<kKPack>{}),
        ck_tile::make_tuple(ck_tile::number<(N_ / kNPerRow) * (kPixelsPerRow + kKPack)>{},
                            ck_tile::number<kPixelsPerRow + kKPack>{},
                            ck_tile::number<kKPack>{},
                            ck_tile::number<1>{}),
        ck_tile::number<kKPack>{},
        ck_tile::number<1>{});

    return ck_tile::transform_tensor_descriptor(
        desc0,
        ck_tile::make_tuple(
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<N_ / kNPerRow>{},
                                    ck_tile::number<kNPerRow>{})),
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<K_ / kKPack>{},
                                    ck_tile::number<kKPack>{}))),
        ck_tile::make_tuple(ck_tile::sequence<1, 2>{}, ck_tile::sequence<0, 3>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
}

template <int N_, int K_>
CK_TILE_HOST_DEVICE constexpr auto make_phase2_v_shuffle_distribution()
{
    constexpr ck_tile::index_t kBlockSize = 256;
    constexpr ck_tile::index_t kVectorWidth =
        kUnifiedAttentionD256MmacVectorWidth;
    constexpr ck_tile::index_t kWaveSize = ck_tile::get_warp_size();
    constexpr ck_tile::index_t N1 = kVectorWidth;
    constexpr ck_tile::index_t N0 = N_ / N1;
    constexpr ck_tile::index_t total_pixels = N_ * K_ / kBlockSize;
    static_assert(total_pixels % N1 == 0);
    constexpr ck_tile::index_t K3 = total_pixels / N1;
    constexpr ck_tile::index_t kKPack = kVectorWidth;
    static_assert(kKPack % K3 == 0);
    constexpr ck_tile::index_t K2 = kKPack / K3;
    static_assert(kWaveSize % (K2 * N0) == 0);
    constexpr ck_tile::index_t K1 = kWaveSize / (K2 * N0);
    constexpr ck_tile::index_t K0 = kBlockSize / kWaveSize;

    return ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<1>,
            ck_tile::tuple<ck_tile::sequence<N0, N1>,
                           ck_tile::sequence<K0, K1, K2, K3>>,
            ck_tile::tuple<ck_tile::sequence<2>,
                           ck_tile::sequence<2, 1, 2>>,
            ck_tile::tuple<ck_tile::sequence<0>,
                           ck_tile::sequence<1, 0, 2>>,
            ck_tile::sequence<1, 2>,
            ck_tile::sequence<1, 3>>{});
}

template <int N_, int K_>
CK_TILE_HOST_DEVICE constexpr auto make_phase2_v_dram_distribution()
{
    constexpr ck_tile::index_t kBlockSize = 256;
    constexpr ck_tile::index_t kVectorWidth =
        kUnifiedAttentionD256MmacVectorWidth;
    constexpr ck_tile::index_t kWaveSize = ck_tile::get_warp_size();
    constexpr ck_tile::index_t N1 = kVectorWidth;
    constexpr ck_tile::index_t N0 = N_ / N1;
    constexpr ck_tile::index_t total_pixels = N_ * K_ / kBlockSize;
    static_assert(total_pixels % N1 == 0);
    constexpr ck_tile::index_t K3 = total_pixels / N1;
    constexpr ck_tile::index_t kKPack = kVectorWidth;
    static_assert(kKPack % K3 == 0);
    constexpr ck_tile::index_t K2 = kKPack / K3;
    static_assert(kWaveSize % (K2 * N0) == 0);
    constexpr ck_tile::index_t K1 = kWaveSize / (K2 * N0);
    constexpr ck_tile::index_t K0 = kBlockSize / kWaveSize;

    return ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<1>,
            ck_tile::tuple<ck_tile::sequence<N0, N1>,
                           ck_tile::sequence<K0, K1, K2, K3>>,
            ck_tile::tuple<ck_tile::sequence<2>,
                           ck_tile::sequence<2, 1, 2>>,
            ck_tile::tuple<ck_tile::sequence<0>,
                           ck_tile::sequence<1, 0, 2>>,
            ck_tile::sequence<2, 1>,
            ck_tile::sequence<3, 1>>{});
}


static constexpr ck_tile::index_t kAsyncKVector =
    kUnifiedAttentionD256MmacVectorWidth;
static constexpr ck_tile::index_t kAsyncKPack =
    kUnifiedAttentionD256MmacVectorWidth;
static constexpr ck_tile::index_t kAsyncKPad = 16;
static constexpr ck_tile::index_t kAsyncNumWarps = 4;
static constexpr ck_tile::index_t kAsyncWarpSize = ck_tile::get_warp_size();

enum class UnifiedAttentionD256AlibiMode
{
    Runtime,
    None,
    Linear,
    Sqrt,
};

template <int N_>
struct UnifiedAttentionD256AsyncKGeometry
{
    static_assert((N_ == 32 || N_ == 64), "async K supports 32/64 key rows");
    static constexpr ck_tile::index_t kRows = N_;
    static constexpr ck_tile::index_t kCols = 64;
    static constexpr ck_tile::index_t kLanesPerD = kCols / kAsyncKVector;
    static constexpr ck_tile::index_t kLaneGroups = kAsyncWarpSize / kLanesPerD;
    static constexpr ck_tile::index_t kNumIssues = kRows / (kLaneGroups * kAsyncNumWarps);
    static constexpr ck_tile::index_t kSmemElements =
        kNumIssues * kAsyncNumWarps * (kAsyncWarpSize * kAsyncKVector + kAsyncKPad);
    static constexpr ck_tile::index_t kDenseElements = kRows * kCols;
    static_assert(kSmemElements >= kDenseElements, "unexpected async K LDS geometry");
};

static_assert(UnifiedAttentionD256AsyncKGeometry<64>::kNumIssues == 2 &&
                  UnifiedAttentionD256AsyncKGeometry<64>::kDenseElements == 4096,
              "unexpected async K64 geometry");
static_assert(UnifiedAttentionD256AsyncKGeometry<32>::kNumIssues == 1 &&
                  UnifiedAttentionD256AsyncKGeometry<32>::kDenseElements == 2048,
              "unexpected async K32 geometry");

template <int N_, int K_>
CK_TILE_HOST_DEVICE constexpr auto make_phase1_k_async_dram_distribution()
{
    static_assert(K_ == 64, "async K copy currently supports K=64");
    using G = UnifiedAttentionD256AsyncKGeometry<N_>;
    return ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<1>,
            ck_tile::tuple<ck_tile::sequence<G::kNumIssues, G::kLaneGroups, kAsyncNumWarps>,
                           ck_tile::sequence<G::kLanesPerD, kAsyncKVector>>,
            ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
            ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<1, 0>>,
            ck_tile::sequence<1, 2>,
            ck_tile::sequence<0, 1>>{});
}

template <int N_, int Buf = 0>
CK_TILE_HOST_DEVICE constexpr auto make_phase1_k_async_lds_store_descriptor()
{
    using G = UnifiedAttentionD256AsyncKGeometry<N_>;
    constexpr auto desc0 = ck_tile::make_naive_tensor_descriptor_with_offset(
        ck_tile::make_tuple(ck_tile::number<G::kNumIssues>{},
                            ck_tile::number<G::kLaneGroups>{},
                            ck_tile::number<kAsyncNumWarps>{},
                            ck_tile::number<G::kLanesPerD>{},
                            ck_tile::number<kAsyncKVector>{}),
        ck_tile::make_tuple(ck_tile::number<kAsyncNumWarps *
                                            (kAsyncWarpSize * kAsyncKVector + kAsyncKPad)>{},
                            ck_tile::number<N_>{},
                            ck_tile::number<kAsyncWarpSize * kAsyncKVector + kAsyncKPad>{},
                            ck_tile::number<kAsyncKVector>{},
                            ck_tile::number<1>{}),
        ck_tile::number<Buf * G::kSmemElements>{},
        ck_tile::number<kAsyncKVector>{},
        ck_tile::number<1>{});

    return ck_tile::transform_tensor_descriptor(
        desc0,
        ck_tile::make_tuple(
            ck_tile::make_pass_through_transform(ck_tile::number<G::kNumIssues>{}),
            ck_tile::make_pass_through_transform(ck_tile::number<kAsyncNumWarps>{}),
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<G::kLaneGroups>{},
                                    ck_tile::number<G::kLanesPerD>{},
                                    ck_tile::number<kAsyncKVector>{}))),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<2>{},
                            ck_tile::sequence<1, 3, 4>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}, ck_tile::sequence<2>{}));
}

template <int N_>
CK_TILE_HOST_DEVICE constexpr auto make_phase1_k_async_lds_load_descriptor()
{
    using G = UnifiedAttentionD256AsyncKGeometry<N_>;
    constexpr auto desc0 = ck_tile::make_naive_tensor_descriptor(
        ck_tile::make_tuple(ck_tile::number<1>{},
                            ck_tile::number<G::kNumIssues>{},
                            ck_tile::number<kAsyncNumWarps>{},
                            ck_tile::number<G::kLaneGroups>{},
                            ck_tile::number<G::kCols / kAsyncKPack>{},
                            ck_tile::number<kAsyncKPack>{}),
        ck_tile::make_tuple(ck_tile::number<G::kSmemElements>{},
                            ck_tile::number<kAsyncNumWarps *
                                            (kAsyncWarpSize * kAsyncKVector + kAsyncKPad)>{},
                            ck_tile::number<kAsyncWarpSize * kAsyncKVector + kAsyncKPad>{},
                            ck_tile::number<G::kCols>{},
                            ck_tile::number<kAsyncKPack>{},
                            ck_tile::number<1>{}),
        ck_tile::number<kAsyncKPack>{},
        ck_tile::number<1>{});

    return ck_tile::transform_tensor_descriptor(
        desc0,
        ck_tile::make_tuple(
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<1>{},
                                    ck_tile::number<G::kNumIssues>{},
                                    ck_tile::number<G::kLaneGroups>{},
                                    ck_tile::number<kAsyncNumWarps>{})),
            ck_tile::make_merge_transform(
                ck_tile::make_tuple(ck_tile::number<G::kCols / kAsyncKPack>{},
                                    ck_tile::number<kAsyncKPack>{}))),
        ck_tile::make_tuple(ck_tile::sequence<0, 1, 3, 2>{}, ck_tile::sequence<4, 5>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
}


template <int N_TILE,
          bool HasPadding,
          bool HasText,
          bool HasSoftcap,
          bool HasQqBias,
          bool DirectCausal,
          bool DirectPage,
          bool StoreLse,
          UnifiedAttentionD256AlibiMode AlibiMode,
          typename PipelinePolicy>
struct UnifiedAttentionD256FusedQkSoftmaxPvPipeline
{
    static constexpr int QK_D_TILE = 64;
    static constexpr int KEY_TILE = PipelinePolicy::kUseSplitK32 ? 32 : 64;
    static constexpr int QK_D_CHUNKS = PipelinePolicy::template GetQkDChunks<N_TILE>();
    static constexpr int K_SMEM_BUFFERS = QK_D_CHUNKS > 1 ? 2 : 1;
    static constexpr int P_SMEM_STRIDE = (PipelinePolicy::kUseSplitK32 ? kBlockN : KEY_TILE) + 12;

    union SharedStorage {
        struct {
            uint16_t k_smem[K_SMEM_BUFFERS * UnifiedAttentionD256AsyncKGeometry<KEY_TILE>::kSmemElements];
        } qk;
        struct {
            uint16_t p_smem[kBlockM * P_SMEM_STRIDE];
            float row_smem[kBlockM];
        } pv;
    };

    CK_TILE_DEVICE void operator()(const uint16_t* __restrict__ q_bh,
                                   const uint16_t* __restrict__ k_bh,
                                   const uint16_t* __restrict__ v_bh,
                                   const int32_t* __restrict__ active_q,
                                   int active_count,
                                   int active_kb_begin,
                                   uint16_t* __restrict__ o_bh,
                                   float* __restrict__ lse,
                                   int bh,
                                   int qb,
                                   int start_m,
                                   int seqlen_val,
                                   int N_Q,
                                   int N_KV,
                                   int head_dim,
                                   int padded_dim,
                                   int64_t q_token_stride,
                                   int64_t k_token_stride,
                                   int64_t v_token_stride,
                                   int text_block_start,
                                   float text_amp,
                                   float qk_scale,
                                   float softcap,
                                   bool causal,
                                   int qhead,
                                   const float* __restrict__ sinks,
                                   const float* __restrict__ alibi,
                                   const float* __restrict__ qq_bias,
                                   int qq_bias_stride,
                                   const int32_t* __restrict__ mm_ranges,
                                   int max_mm_ranges,
                                   int sliding_window,
                                   bool alibi_sqrt,
                                   int64_t stride_om,
                                   int64_t stride_ok,
                                   int64_t stride_lz,
                                   int64_t stride_lm,
                                   SharedStorage& scratch) const
    {
    constexpr int QK_D_TILE = 64;
    constexpr int KEY_TILE = PipelinePolicy::kUseSplitK32 ? 32 : 64;
    using Scratch = SharedStorage;


    using Problem = typename PipelinePolicy::Problem;
    using DataType = typename Problem::QDataType;
    using QkBlockGemm = UnifiedAttentionD256QkBlockGemmN<Problem, KEY_TILE, QK_D_TILE, PipelinePolicy>;
    using QkARegBRegBlockGemm = Phase1ARegBRegBlockGemmN<KEY_TILE, QK_D_TILE, PipelinePolicy>;
    using PvARegBSmemBlockGemmK = Phase2PvARegBSmemBlockGemmK<N_TILE, KEY_TILE, PipelinePolicy>;
    using PvARegBRegBlockGemmK = Phase2PvARegBRegBlockGemmK<N_TILE, KEY_TILE, PipelinePolicy>;

    constexpr auto qk_block_gemm = QkBlockGemm{};
    constexpr auto qk_areg_breg_block_gemm = QkARegBRegBlockGemm{};
    constexpr auto pv_key_areg_bsmem_block_gemm = PvARegBSmemBlockGemmK{};
    constexpr auto pv_key_areg_breg_block_gemm = PvARegBRegBlockGemmK{};
    auto o_acc0 = pv_key_areg_breg_block_gemm.MakeCBlockTile();
    auto o_acc1 = pv_key_areg_breg_block_gemm.MakeCBlockTile();
    ck_tile::clear_tile(o_acc0);
    ck_tile::clear_tile(o_acc1);

    const auto f_max = [](auto e0, auto e1) { return ck_tile::max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };
    using QkComputeTile = decltype(ck_tile::cast_tile<float>(qk_block_gemm.MakeCBlockTile()));
    using MLTile = decltype(ck_tile::block_tile_reduce<float>(
        QkComputeTile{}, ck_tile::sequence<1>{}, f_max, 0.0f));
    auto m = MLTile{};
    auto l = MLTile{};
    ck_tile::set_tile(m, -ck_tile::numeric<float>::infinity());
    ck_tile::clear_tile(l);
    constexpr float log2e = ck_tile::log2e_v<float>;
    if(sinks != nullptr)
    {
        ck_tile::tile_elementwise_inout(
            [=](auto& x) { x = sinks[qhead] * log2e; }, m);
        ck_tile::tile_elementwise_inout([](auto& x) { x = 1.0f; }, l);
    }

    auto load_q_chunk = [&](int d_offset) {
        auto q_dram = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            reinterpret_cast<const DataType*>(q_bh) +
                static_cast<int64_t>(qb) * kBlockM * q_token_stride + d_offset,
            ck_tile::make_tuple(ck_tile::number<kBlockM>{}, ck_tile::number<QK_D_TILE>{}),
            ck_tile::make_tuple(q_token_stride, ck_tile::number<1>{}),
            ck_tile::number<kUnifiedAttentionD256MmacVectorWidth>{},
            ck_tile::number<1>{});
        auto q_areg = qk_areg_breg_block_gemm.MakeABlockTile();
        auto q_dram_win = ck_tile::make_tile_window(
            q_dram,
            ck_tile::make_tuple(ck_tile::number<kBlockM>{}, ck_tile::number<QK_D_TILE>{}),
            ck_tile::multi_index<2>{0, 0},
            q_areg.get_tile_distribution());
        return ck_tile::load_tile(q_dram_win);
    };

    auto q_reg_chunk0 = load_q_chunk(0);
    auto q_reg_chunk1 = q_reg_chunk0;
    auto q_reg_chunk2 = q_reg_chunk0;
    auto q_reg_chunk3 = q_reg_chunk0;
    if constexpr (QK_D_CHUNKS > 1) {
        q_reg_chunk1 = load_q_chunk(QK_D_TILE);
    }
    if constexpr (QK_D_CHUNKS > 2) {
        q_reg_chunk2 = load_q_chunk(2 * QK_D_TILE);
    }
    if constexpr (QK_D_CHUNKS > 3) {
        q_reg_chunk3 = load_q_chunk(3 * QK_D_TILE);
    }

    auto load_k_subtile = [&](int kb, int key_sub, int d_chunk, auto buf_num) {
        constexpr int buf = decltype(buf_num)::value;
        int start_n_sub = kb * kBlockN + key_sub * KEY_TILE;
        int d_base = d_chunk * QK_D_TILE;
        auto k_dram = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            reinterpret_cast<const DataType*>(k_bh) +
                static_cast<int64_t>(start_n_sub) * k_token_stride + d_base,
            ck_tile::make_tuple(ck_tile::number<KEY_TILE>{}, ck_tile::number<QK_D_TILE>{}),
            ck_tile::make_tuple(k_token_stride, ck_tile::number<1>{}),
            ck_tile::number<kUnifiedAttentionD256MmacVectorWidth>{},
            ck_tile::number<1>{});
        auto k_dram_win = ck_tile::make_tile_window(
            k_dram,
            ck_tile::make_tuple(ck_tile::number<KEY_TILE>{}, ck_tile::number<QK_D_TILE>{}),
            ck_tile::multi_index<2>{0, 0},
            make_phase1_k_async_dram_distribution<KEY_TILE, QK_D_TILE>());
        k_dram_win.init_raw();

        auto k_lds = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<DataType*>(scratch.qk.k_smem),
            make_phase1_k_async_lds_store_descriptor<KEY_TILE, buf>());
        auto k_lds_win = ck_tile::make_tile_window(
            k_lds,
            make_phase1_k_async_lds_store_descriptor<KEY_TILE, buf>().get_lengths(),
            ck_tile::multi_index<3>{0, 0, 0});
        ck_tile::async_load_tile_by_inline_asm(k_lds_win,
                                               k_dram_win,
                                               ck_tile::bool_constant<true>{});
    };

    auto load_v_subtile = [&](int start_n_sub, int d_base) {
        if constexpr (PipelinePolicy::kUseInterleavedV &&
                      N_TILE == 128 && KEY_TILE == 32) {
            return pv_key_areg_breg_block_gemm.LoadBBlockTileDword(
                reinterpret_cast<const DataType*>(v_bh),
                v_token_stride,
                d_base,
                start_n_sub);
        } else {
        auto v_dram = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            reinterpret_cast<const DataType*>(v_bh) +
                static_cast<int64_t>(start_n_sub) * v_token_stride + d_base,
            ck_tile::make_tuple(ck_tile::number<N_TILE>{}, ck_tile::number<KEY_TILE>{}),
            ck_tile::make_tuple(ck_tile::number<1>{}, v_token_stride),
            ck_tile::number<1>{},
            ck_tile::number<1>{});
        auto v_breg = pv_key_areg_breg_block_gemm.MakeBBlockTile();
        auto v_dram_win = ck_tile::make_tile_window(
            v_dram,
            ck_tile::make_tuple(ck_tile::number<N_TILE>{}, ck_tile::number<KEY_TILE>{}),
            ck_tile::multi_index<2>{0, 0},
            v_breg.get_tile_distribution());
        return ck_tile::load_tile(v_dram_win);
        }
    };

    for (int active_i = 0; active_i < active_count; ++active_i) {
        const int kb = DirectCausal ? active_kb_begin + active_i
                                    : active_q[active_i];
        const bool has_previous_active = active_i != 0;

        auto compute_s_subtile_impl = [&](auto key_sub_num,
                                          bool d0_preloaded,
                                          bool prefetch_next_d0,
                                          int prefetch_kb,
                                          int prefetch_key_sub) {
            constexpr int key_sub = decltype(key_sub_num)::value;
            const int start_n_sub = kb * kBlockN + key_sub * KEY_TILE;
            const bool tile_has_padding =
                (start_m + kBlockM > N_Q) || (start_m + kBlockM > seqlen_val) ||
                (start_n_sub + KEY_TILE > N_KV) || (start_n_sub + KEY_TILE > seqlen_val);
            auto qk_acc = qk_block_gemm.MakeCBlockTile();
            ck_tile::clear_tile(qk_acc);

            auto make_k_window = [&](auto buf_num) {
                constexpr int buf = decltype(buf_num)::value;
                return ck_tile::make_tile_window(
                    ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                        reinterpret_cast<DataType*>(scratch.qk.k_smem),
                        make_phase1_k_async_lds_load_descriptor<KEY_TILE>()),
                    ck_tile::make_tuple(ck_tile::number<KEY_TILE>{}, ck_tile::number<QK_D_TILE>{}),
                    ck_tile::multi_index<2>{buf * KEY_TILE, 0});
            };

            auto load_k_breg_fragment = [&](auto buf_num, auto k_iter_num) {
                constexpr int k_iter = decltype(k_iter_num)::value;
                constexpr int K_FRAGMENT = QK_D_TILE / 2;
                auto k_breg = qk_areg_breg_block_gemm.MakeKFragmentBlockTile();
                auto k_win = make_k_window(buf_num);
                auto k_reg_win = ck_tile::make_tile_window(
                    k_win.get_bottom_tensor_view(),
                    k_breg.get_lengths(),
                    k_win.get_window_origin() +
                        ck_tile::multi_index<2>{0, k_iter * K_FRAGMENT},
                    k_breg.get_tile_distribution());
                return ck_tile::load_tile(k_reg_win);
            };

            auto compute_k_fragment =
                [&](auto d_chunk_num, auto k_iter_num, const auto& k_reg) {
                    constexpr int d_chunk = decltype(d_chunk_num)::value;
                    const auto& q_reg = (d_chunk == 0)
                                            ? q_reg_chunk0
                                            : (d_chunk == 1)
                                                  ? q_reg_chunk1
                                                  : (d_chunk == 2) ? q_reg_chunk2 : q_reg_chunk3;
                    qk_areg_breg_block_gemm.RunKIter(
                        qk_acc, q_reg, k_reg, k_iter_num);
                };

            auto compute_d_chunk = [&](auto d_chunk_num, auto buf_num) {
                constexpr int d_chunk = decltype(d_chunk_num)::value;
                auto k_reg0 =
                    load_k_breg_fragment(buf_num, ck_tile::number<0>{});
                auto k_reg1 =
                    load_k_breg_fragment(buf_num, ck_tile::number<1>{});
                __builtin_amdgcn_sched_barrier(0);
                compute_k_fragment(
                    d_chunk_num, ck_tile::number<0>{}, k_reg0);
                compute_k_fragment(
                    d_chunk_num, ck_tile::number<1>{}, k_reg1);
            };

            if constexpr (QK_D_CHUNKS == 2) {
                if (!d0_preloaded) {
                    load_k_subtile(kb, key_sub, 0, ck_tile::number<0>{});
                    ck_tile::async_load_fence(0);
                    unified_attention_d256_block_sync_lds_light();
                }

                load_k_subtile(kb, key_sub, 1, ck_tile::number<1>{});
                compute_d_chunk(ck_tile::number<0>{}, ck_tile::number<0>{});
                if (prefetch_next_d0) {
                    unified_attention_d256_block_sync_lds_light();
                    load_k_subtile(prefetch_kb, prefetch_key_sub, 0, ck_tile::number<0>{});
                }
                ck_tile::async_load_fence(0);
                unified_attention_d256_block_sync_lds_light();
                compute_d_chunk(ck_tile::number<1>{}, ck_tile::number<1>{});
                unified_attention_d256_block_sync_lds_light();
            } else if constexpr (QK_D_CHUNKS == 4) {
                // D=256: four 64-dim K chunks over two LDS buffers.
                // Preload d0/d1, then overlap d{i+2} with d{i+1} compute.
                load_k_subtile(kb, key_sub, 0, ck_tile::number<0>{});
                load_k_subtile(kb, key_sub, 1, ck_tile::number<1>{});
                ck_tile::async_load_fence(0);
                unified_attention_d256_block_sync_lds_light();

                compute_d_chunk(ck_tile::number<0>{}, ck_tile::number<0>{});
                unified_attention_d256_block_sync_lds_light();

                load_k_subtile(kb, key_sub, 2, ck_tile::number<0>{});
                compute_d_chunk(ck_tile::number<1>{}, ck_tile::number<1>{});
                ck_tile::async_load_fence(0);
                unified_attention_d256_block_sync_lds_light();

                load_k_subtile(kb, key_sub, 3, ck_tile::number<1>{});
                compute_d_chunk(ck_tile::number<2>{}, ck_tile::number<0>{});
                ck_tile::async_load_fence(0);
                unified_attention_d256_block_sync_lds_light();

                compute_d_chunk(ck_tile::number<3>{}, ck_tile::number<1>{});
                unified_attention_d256_block_sync_lds_light();
            } else {
                ck_tile::static_for<0, QK_D_CHUNKS, 1>{}([&](auto d_chunk_num) {
                    constexpr int d_chunk = decltype(d_chunk_num)::value;
                    load_k_subtile(kb, key_sub, d_chunk, ck_tile::number<0>{});
                    ck_tile::async_load_fence(0);
                    unified_attention_d256_block_sync_lds_light();
                    compute_d_chunk(d_chunk_num, ck_tile::number<0>{});
                    unified_attention_d256_block_sync_lds_light();
                });
            }

            auto s = qk_block_gemm.MakeOuputLayout(qk_acc);

            // Block-level mm-range pruning: if no query row in this 64-row tile
            // can intersect any valid prefix range, the per-score range loop is
            // pure overhead and can be skipped.
            const int context = N_KV - N_Q;
            const int q_first = start_m;
            const int q_last = ck_tile::min(
                ck_tile::min(start_m + kBlockM - 1, N_Q - 1), seqlen_val - 1);
            bool q_block_in_range = false;
            if(mm_ranges != nullptr)
            {
                for(int i = 0; i < max_mm_ranges; ++i)
                {
                    const int begin = mm_ranges[i * 2];
                    const int end = mm_ranges[i * 2 + 1];
                    if(begin < end && (context + q_first) <= end &&
                       (context + q_last) >= begin)
                    {
                        q_block_in_range = true;
                        break;
                    }
                }
            }
            bool block_intersects_mm_range = false;
            if(mm_ranges != nullptr)
            {
                const int k_first = start_n_sub;
                const int k_last = start_n_sub + KEY_TILE - 1;
                for(int i = 0; i < max_mm_ranges; ++i)
                {
                    const int begin = mm_ranges[i * 2];
                    const int end = mm_ranges[i * 2 + 1];
                    if(begin < end && k_last >= begin && k_first <= end)
                    {
                        block_intersects_mm_range = true;
                        break;
                    }
                }
            }
            const int tile_first_q_abs = context + start_m;
            const bool fully_causal_visible =
                (!causal) ||
                ((start_n_sub + KEY_TILE - 1 <= tile_first_q_abs) &&
                 (sliding_window <= 0 ||
                  tile_first_q_abs + kBlockM - 1 - start_n_sub < sliding_window));
            const bool tile_needs_mask =
                tile_has_padding || (causal && !fully_causal_visible) ||
                (q_block_in_range && block_intersects_mm_range);

            constexpr auto s_spans = decltype(s)::get_distributed_spans();
            auto scale_s_tile = [&]() {
                ck_tile::sweep_tile_span(s_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    ck_tile::sweep_tile_span(s_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto tile_idx = ck_tile::get_x_indices_from_distributed_indices(
                            s.get_tile_distribution(), i_j_idx);
                        const int row = tile_idx.at(ck_tile::number<0>{});
                        const int col = tile_idx.at(ck_tile::number<1>{});
                        const int q_pos = start_m + row;
                        const int k_pos = start_n_sub + col;
                        s(i_j_idx) *= qk_scale;
                        if constexpr (HasSoftcap) {
                            const float softcap_log2 = softcap * log2e;
                            s(i_j_idx) = softcap_log2 * tanhf(s(i_j_idx) / softcap_log2);
                        }
                        if constexpr (HasQqBias) {
                            const int key_rel = k_pos - context;
                            if (key_rel >= 0 && key_rel < qq_bias_stride) {
                                s(i_j_idx) +=
                                    qq_bias[static_cast<int64_t>(q_pos) * qq_bias_stride + key_rel] *
                                    log2e;
                            }
                        }
                        if constexpr (HasText) {
                            if (kb >= text_block_start) {
                                s(i_j_idx) += text_amp;
                            }
                        }
                    });
                });
            };
            auto mask_and_scale_s_tile = [&]() {
                ck_tile::sweep_tile_span(s_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    ck_tile::sweep_tile_span(s_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto tile_idx = ck_tile::get_x_indices_from_distributed_indices(
                            s.get_tile_distribution(), i_j_idx);
                        const int row = tile_idx.at(ck_tile::number<0>{});
                        const int col = tile_idx.at(ck_tile::number<1>{});
                        const int q_pos = start_m + row;
                        const int k_pos = start_n_sub + col;
                        const int context = N_KV - N_Q;
                        const int q_abs = context + q_pos;
                        bool mm_allowed = false;
                        if (q_block_in_range && mm_ranges != nullptr) {
                            for (int i = 0; i < max_mm_ranges; ++i) {
                                const int begin = mm_ranges[i * 2];
                                const int end = mm_ranges[i * 2 + 1];
                                mm_allowed = mm_allowed ||
                                    (begin < end && q_abs >= begin && q_abs <= end &&
                                     k_pos >= begin && k_pos <= end);
                            }
                        }
                        const bool causal_allowed =
                            (!causal || k_pos <= q_abs) &&
                            (sliding_window <= 0 || q_abs - k_pos < sliding_window);
                        if (q_pos >= N_Q || q_pos >= seqlen_val ||
                            k_pos >= N_KV || k_pos >= seqlen_val ||
                            !(causal_allowed || mm_allowed)) {
                            s(i_j_idx) = -ck_tile::numeric<float>::infinity();
                        } else {
                            s(i_j_idx) *= qk_scale;
                            if constexpr (HasSoftcap) {
                                const float softcap_log2 = softcap * log2e;
                                s(i_j_idx) = softcap_log2 * tanhf(s(i_j_idx) / softcap_log2);
                            }
                            if constexpr (HasQqBias) {
                                const int key_rel = k_pos - context;
                                if (key_rel >= 0 && key_rel < qq_bias_stride) {
                                    s(i_j_idx) +=
                                        qq_bias[static_cast<int64_t>(q_pos) * qq_bias_stride + key_rel] *
                                        log2e;
                                }
                            }
                            if constexpr (HasText) {
                                if (kb >= text_block_start) {
                                    s(i_j_idx) += text_amp;
                                }
                            }
                        }
                    });
                });
            };
            if constexpr (HasPadding) {
                tile_needs_mask ? mask_and_scale_s_tile() : scale_s_tile();
            } else {
                scale_s_tile();
            }

            // ALiBi is uniform at the tile level. Keep its runtime mode
            // selection outside the statically-unrolled score-element sweep
            // so each register element contains arithmetic only.
            if constexpr(AlibiMode != UnifiedAttentionD256AlibiMode::None)
            {
                if(AlibiMode == UnifiedAttentionD256AlibiMode::Runtime &&
                   alibi == nullptr)
                    return s;

                const float alibi_scale = alibi[qhead] * log2e;
                if(AlibiMode == UnifiedAttentionD256AlibiMode::Sqrt ||
                   (AlibiMode == UnifiedAttentionD256AlibiMode::Runtime &&
                    alibi_sqrt))
                {
                    ck_tile::sweep_tile_span(
                        s_spans[ck_tile::number<0>{}], [&](auto idx0) {
                        ck_tile::sweep_tile_span(
                            s_spans[ck_tile::number<1>{}], [&](auto idx1) {
                            constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                            const auto tile_idx =
                                ck_tile::get_x_indices_from_distributed_indices(
                                    s.get_tile_distribution(), i_j_idx);
                            const int row = tile_idx.at(ck_tile::number<0>{});
                            const int col = tile_idx.at(ck_tile::number<1>{});
                            const int distance =
                                context + start_m + row - start_n_sub - col;
                            if(distance >= 0)
                                s(i_j_idx) -=
                                    alibi_scale * __builtin_amdgcn_sqrtf(static_cast<float>(distance));
                        });
                    });
                }
                else
                {
                    const float alibi_base =
                        alibi_scale * static_cast<float>(start_n_sub - context);
                    ck_tile::sweep_tile_span(
                        s_spans[ck_tile::number<0>{}], [&](auto idx0) {
                        ck_tile::sweep_tile_span(
                            s_spans[ck_tile::number<1>{}], [&](auto idx1) {
                            constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                            const auto tile_idx =
                                ck_tile::get_x_indices_from_distributed_indices(
                                    s.get_tile_distribution(), i_j_idx);
                            const int col = tile_idx.at(ck_tile::number<1>{});
                            s(i_j_idx) +=
                                alibi_base + alibi_scale * static_cast<float>(col);
                        });
                    });
                }
            }

            return s;
        };
        auto compute_s_subtile = [&](auto key_sub_num) {
            return compute_s_subtile_impl(key_sub_num, false, false, 0, 0);
        };

        auto make_p_compute = [&](auto& s_in, int start_n_sub) {
            const bool tile_has_padding =
                (start_m + kBlockM > N_Q) || (start_m + kBlockM > seqlen_val) ||
                (start_n_sub + KEY_TILE > N_KV) || (start_n_sub + KEY_TILE > seqlen_val);
            auto p_compute = ck_tile::make_static_distributed_tensor<float>(s_in.get_tile_distribution());
            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            auto fill_p_tile = [&]() {
                ck_tile::sweep_tile_span(p_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    const float row_m_val =
                        m[i_idx] == -ck_tile::numeric<float>::infinity() ? 0.0f : m[i_idx];
                    ck_tile::sweep_tile_span(p_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        p_compute(i_j_idx) = unified_attention_d256_fast_exp2(s_in[i_j_idx] - row_m_val);
                    });
                });
            };
            auto mask_and_fill_p_tile = [&]() {
                ck_tile::sweep_tile_span(p_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    const float row_m_val =
                        m[i_idx] == -ck_tile::numeric<float>::infinity() ? 0.0f : m[i_idx];
                    ck_tile::sweep_tile_span(p_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto tile_idx = ck_tile::get_x_indices_from_distributed_indices(
                            p_compute.get_tile_distribution(), i_j_idx);
                        const int row = tile_idx.at(ck_tile::number<0>{});
                        const int col = tile_idx.at(ck_tile::number<1>{});
                        const int q_pos = start_m + row;
                        const int k_pos = start_n_sub + col;
                        if (q_pos >= N_Q || q_pos >= seqlen_val || k_pos >= N_KV ||
                            k_pos >= seqlen_val ||
                            (causal && k_pos > N_KV - N_Q + q_pos)) {
                            p_compute(i_j_idx) = 0.0f;
                        } else {
                            p_compute(i_j_idx) = unified_attention_d256_fast_exp2(s_in[i_j_idx] - row_m_val);
                        }
                    });
                });
            };
            // QK already applied the complete unified predicate, including
            // (causal AND sliding-window) OR mm-prefix. Masked scores are -inf,
            // so a second causal-only predicate here would incorrectly erase
            // mm-prefix visibility.
            fill_p_tile();
            return p_compute;
        };

        auto make_p_value_and_rowsum = [&](auto& s_in, int start_n_sub, auto& p_value, auto& rowsum_p) {
            const bool tile_has_padding =
                (start_m + kBlockM > N_Q) || (start_m + kBlockM > seqlen_val) ||
                (start_n_sub + KEY_TILE > N_KV) || (start_n_sub + KEY_TILE > seqlen_val);
            ck_tile::clear_tile(rowsum_p);
            using PValueTile = ck_tile::remove_cvref_t<decltype(p_value)>;
            constexpr auto p_spans = PValueTile::get_distributed_spans();
            auto fill_p_value_tile = [&]() {
                ck_tile::sweep_tile_span(p_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    const float row_m_val =
                        m[i_idx] == -ck_tile::numeric<float>::infinity() ? 0.0f : m[i_idx];
                    ck_tile::sweep_tile_span(p_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        const float p_val = unified_attention_d256_fast_exp2(s_in[i_j_idx] - row_m_val);
                        p_value(i_j_idx) = ck_tile::type_convert<DataType>(p_val);
                        rowsum_p(i_idx) += p_val;
                    });
                });
            };
            auto mask_and_fill_p_value_tile = [&]() {
                ck_tile::sweep_tile_span(p_spans[ck_tile::number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    const float row_m_val =
                        m[i_idx] == -ck_tile::numeric<float>::infinity() ? 0.0f : m[i_idx];
                    ck_tile::sweep_tile_span(p_spans[ck_tile::number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto tile_idx = ck_tile::get_x_indices_from_distributed_indices(
                            p_value.get_tile_distribution(), i_j_idx);
                        const int row = tile_idx.at(ck_tile::number<0>{});
                        const int col = tile_idx.at(ck_tile::number<1>{});
                        const int q_pos = start_m + row;
                        const int k_pos = start_n_sub + col;
                        float p_val = 0.0f;
                        if (q_pos < N_Q && q_pos < seqlen_val && k_pos < N_KV &&
                            k_pos < seqlen_val &&
                            (!causal || k_pos <= N_KV - N_Q + q_pos)) {
                            p_val = unified_attention_d256_fast_exp2(s_in[i_j_idx] - row_m_val);
                        }
                        p_value(i_j_idx) = ck_tile::type_convert<DataType>(p_val);
                        rowsum_p(i_idx) += p_val;
                    });
                });
            };
            // QK already applied the complete unified predicate, including
            // (causal AND sliding-window) OR mm-prefix. Masked scores are -inf,
            // so a second causal-only predicate here would incorrectly erase
            // mm-prefix visibility.
            fill_p_value_tile();
        };

        auto update_l_and_row_alpha = [&](const auto& m_old, const auto& rowsum_p) {
            constexpr auto l_spans = decltype(l)::get_distributed_spans();
            ck_tile::sweep_tile_span(l_spans[ck_tile::number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = ck_tile::make_tuple(idx0);
                const auto tile_idx = ck_tile::get_x_indices_from_distributed_indices(
                    m.get_tile_distribution(), i_idx);
                const int row = tile_idx.at(ck_tile::number<0>{});
                const float alpha =
                    (m_old[i_idx] == -ck_tile::numeric<float>::infinity() ||
                     m[i_idx] == -ck_tile::numeric<float>::infinity())
                        ? 0.0f
                        : unified_attention_d256_fast_exp2(m_old[i_idx] - m[i_idx]);
                scratch.pv.row_smem[row] = alpha;
                l(i_idx) = l[i_idx] * alpha + rowsum_p[i_idx];
            });
        };

        auto scale_one_o = [&](auto& o_acc) {
            using OAcc = ck_tile::remove_cvref_t<decltype(o_acc)>;
            constexpr auto o_spans = OAcc::get_distributed_spans();
            ck_tile::sweep_tile_span(o_spans[ck_tile::number<0>{}], [&](auto idx0) {
                using Idx1ZeroSeq = typename ck_tile::uniform_sequence_gen<
                    decltype(o_spans[ck_tile::number<1>{}])::Impl::size(),
                    0>::type;
                constexpr auto idx1_zero =
                    ck_tile::detail::make_tile_distributed_index(Idx1ZeroSeq{});
                constexpr auto row_idx = ck_tile::make_tuple(idx0, idx1_zero);
                const auto row_tile_idx = ck_tile::get_x_indices_from_distributed_indices(
                    o_acc.get_tile_distribution(), row_idx);
                const int row = row_tile_idx.at(ck_tile::number<0>{});
                const float alpha = scratch.pv.row_smem[row];
                ck_tile::sweep_tile_span(o_spans[ck_tile::number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                    o_acc(i_j_idx) *= alpha;
                });
            });
        };
        auto scale_o_from_row_alpha = [&]() {
            scale_one_o(o_acc0);
            scale_one_o(o_acc1);
        };

        auto make_p_lds_view = [&](int slot) {
            return ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                reinterpret_cast<DataType*>(scratch.pv.p_smem + slot * KEY_TILE),
                ck_tile::make_naive_tensor_descriptor(
                    ck_tile::make_tuple(ck_tile::number<kBlockM>{}, ck_tile::number<KEY_TILE>{}),
                    ck_tile::make_tuple(ck_tile::number<P_SMEM_STRIDE>{}, ck_tile::number<1>{}),
                    ck_tile::number<KEY_TILE>{},
                    ck_tile::number<1>{}));
        };

        auto store_p_to_lds = [&](auto& p_compute, int slot) {
            auto p_value = ck_tile::cast_tile<DataType>(p_compute);
            auto p_lds = make_p_lds_view(slot);
            auto p_lds_store_win = ck_tile::make_tile_window(
                p_lds,
                ck_tile::make_tuple(ck_tile::number<kBlockM>{}, ck_tile::number<KEY_TILE>{}),
                ck_tile::multi_index<2>{0, 0},
                p_value.get_tile_distribution());
            ck_tile::store_tile(p_lds_store_win, p_value);
        };

        auto store_p_value_to_lds = [&](auto& p_value, int slot) {
            auto p_lds = make_p_lds_view(slot);
            auto p_lds_store_win = ck_tile::make_tile_window(
                p_lds,
                ck_tile::make_tuple(ck_tile::number<kBlockM>{}, ck_tile::number<KEY_TILE>{}),
                ck_tile::multi_index<2>{0, 0},
                p_value.get_tile_distribution());
            ck_tile::store_tile(p_lds_store_win, p_value);
        };

        auto load_p_after_ready = [&](int slot) {
            auto p_lds = make_p_lds_view(slot);
            auto p_lds_load_win = ck_tile::make_tile_window(
                p_lds,
                ck_tile::make_tuple(ck_tile::number<kBlockM>{}, ck_tile::number<KEY_TILE>{}),
                ck_tile::multi_index<2>{0, 0},
                pv_key_areg_breg_block_gemm.MakeABlockTile().get_tile_distribution());
            return ck_tile::load_tile(p_lds_load_win);
        };

        auto accumulate_pv_after_p_ready = [&](int start_n_sub, int slot,
                                                const auto& vt_frag0) {
            auto p_lds = make_p_lds_view(slot);
            auto p_lds_load_win = ck_tile::make_tile_window(
                p_lds,
                ck_tile::make_tuple(ck_tile::number<kBlockM>{}, ck_tile::number<KEY_TILE>{}),
                ck_tile::multi_index<2>{0, 0},
                pv_key_areg_breg_block_gemm.MakeABlockTile().get_tile_distribution());
            auto p_frag = ck_tile::load_tile(p_lds_load_win);
            pv_key_areg_breg_block_gemm(o_acc0, p_frag, vt_frag0);
            auto vt_frag1 = load_v_subtile(start_n_sub, N_TILE);
            pv_key_areg_breg_block_gemm(o_acc1, p_frag, vt_frag1);
        };

        if constexpr (PipelinePolicy::kUseSplitK32) {
            constexpr int start_n_sub0 = 0;
            constexpr int start_n_sub1 = KEY_TILE;
            auto s0 = compute_s_subtile_impl(ck_tile::number<0>{}, false, true, kb, 1);
            auto s1 = compute_s_subtile_impl(ck_tile::number<1>{}, true, false, 0, 0);

            auto m_local0 = ck_tile::block_tile_reduce<float>(
                s0, ck_tile::sequence<1>{}, f_max, -ck_tile::numeric<float>::infinity());
            {
                auto m_local1 = ck_tile::block_tile_reduce<float>(
                    s1, ck_tile::sequence<1>{}, f_max, -ck_tile::numeric<float>::infinity());
                ck_tile::tile_elementwise_inout(
                    [](auto& out, auto other) { out = ck_tile::max(out, other); },
                    m_local0, m_local1);
            }
            ck_tile::block_tile_reduce_xor_sync(m_local0, f_max);

            const auto m_old = m;
            ck_tile::tile_elementwise_inout(
                [](auto& out, auto old_v, auto local_v) { out = ck_tile::max(old_v, local_v); },
                m, m_old, m_local0);

            auto rowsum_p0 = MLTile{};
            {
                auto p_value_0 = ck_tile::make_static_distributed_tensor<DataType>(
                    s0.get_tile_distribution());
                make_p_value_and_rowsum(s0, kb * kBlockN + start_n_sub0, p_value_0, rowsum_p0);
                store_p_value_to_lds(p_value_0, 0);
            }
            {
                auto rowsum_p1 = MLTile{};
                auto p_value_1 = ck_tile::make_static_distributed_tensor<DataType>(
                    s1.get_tile_distribution());
                make_p_value_and_rowsum(s1, kb * kBlockN + start_n_sub1, p_value_1, rowsum_p1);
                store_p_value_to_lds(p_value_1, 1);
                ck_tile::tile_elementwise_inout(
                    [](auto& out, auto other) { out += other; },
                    rowsum_p0, rowsum_p1);
            }
            ck_tile::block_tile_reduce_xor_sync(rowsum_p0, f_sum);

            update_l_and_row_alpha(m_old, rowsum_p0);
            unified_attention_d256_block_sync_lds_light();
            if (has_previous_active) {
                scale_o_from_row_alpha();
            }

            ck_tile::static_for<0, 2, 1>{}([&](auto half_num) {
                constexpr int half = decltype(half_num)::value;
                constexpr int out_d_base = half * N_TILE;
                auto& o_acc = (half == 0) ? o_acc0 : o_acc1;

                auto vt_frag0 =
                    load_v_subtile(kb * kBlockN + start_n_sub0, out_d_base);
                auto vt_frag1 =
                    load_v_subtile(kb * kBlockN + start_n_sub1, out_d_base);
                __builtin_amdgcn_sched_barrier(0);
                unified_attention_d256_block_sync_lds_light();

                auto p_frag0 = load_p_after_ready(0);
                if constexpr (PipelinePolicy::kUseInterleavedV &&
                              N_TILE == 128 && KEY_TILE == 32) {
                    vt_frag0.template FenceKIter<0, 12>();
                    pv_key_areg_breg_block_gemm.RunKIter(
                        o_acc, p_frag0, vt_frag0, ck_tile::number<0>{});
                    vt_frag0.template FenceKIter<
                        1, kUnifiedAttentionD256MmacVectorWidth>();
                    pv_key_areg_breg_block_gemm.RunKIter(
                        o_acc, p_frag0, vt_frag0, ck_tile::number<1>{});
                } else {
                    pv_key_areg_breg_block_gemm(o_acc, p_frag0, vt_frag0);
                }

                auto p_frag1 = load_p_after_ready(1);
                if constexpr (PipelinePolicy::kUseInterleavedV &&
                              N_TILE == 128 && KEY_TILE == 32) {
                    vt_frag1.template FenceKIter<0, 4>();
                    pv_key_areg_breg_block_gemm.RunKIter(
                        o_acc, p_frag1, vt_frag1, ck_tile::number<0>{});
                    vt_frag1.template FenceKIter<1, 0>();
                    pv_key_areg_breg_block_gemm.RunKIter(
                        o_acc, p_frag1, vt_frag1, ck_tile::number<1>{});
                } else {
                    pv_key_areg_breg_block_gemm(o_acc, p_frag1, vt_frag1);
                }
                unified_attention_d256_block_sync_lds_light();
            });
            unified_attention_d256_block_sync_lds_light();
        } else {
            ck_tile::static_for<0, kBlockN / KEY_TILE, 1>{}([&](auto key_sub_num) {
                constexpr int key_sub = decltype(key_sub_num)::value;
                const int start_n_sub = kb * kBlockN + key_sub * KEY_TILE;
                auto s = compute_s_subtile(key_sub_num);
                auto m_local = ck_tile::block_tile_reduce<float>(
                    s, ck_tile::sequence<1>{}, f_max, -ck_tile::numeric<float>::infinity());
                ck_tile::block_tile_reduce_sync(m_local, f_max, ck_tile::bool_constant<false>{});

                const auto m_old = m;
                ck_tile::tile_elementwise_inout(
                    [](auto& out, auto old_v, auto local_v) { out = ck_tile::max(old_v, local_v); },
                    m, m_old, m_local);

                auto vt_frag0 = load_v_subtile(start_n_sub, 0);
                auto p_compute = make_p_compute(s, start_n_sub);
                auto rowsum_p = ck_tile::block_tile_reduce<float>(
                    p_compute, ck_tile::sequence<1>{}, f_sum, 0.0f);
                ck_tile::block_tile_reduce_sync(rowsum_p, f_sum, ck_tile::bool_constant<false>{});

                update_l_and_row_alpha(m_old, rowsum_p);
                store_p_to_lds(p_compute, 0);
                unified_attention_d256_block_sync_lds_light();
                if (has_previous_active || key_sub != 0) {
                    scale_o_from_row_alpha();
                }
                accumulate_pv_after_p_ready(start_n_sub, 0, vt_frag0);
                unified_attention_d256_block_sync_lds_light();
            });
        }
    }

    constexpr auto ml_spans = decltype(l)::get_distributed_spans();
    ck_tile::sweep_tile_span(ml_spans[ck_tile::number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = ck_tile::make_tuple(idx0);
        const auto tile_idx = ck_tile::get_x_indices_from_distributed_indices(
            l.get_tile_distribution(), i_idx);
        const int row = tile_idx.at(ck_tile::number<0>{});
        scratch.pv.row_smem[row] = l[i_idx] > 1e-10f ? (1.0f / l[i_idx]) : 0.0f;
    });
    unified_attention_d256_block_sync_lds_light();

    if constexpr(StoreLse)
    {
        constexpr auto l_spans = decltype(l)::get_distributed_spans();
        ck_tile::sweep_tile_span(l_spans[ck_tile::number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = ck_tile::make_tuple(idx0);
            const auto tile_idx = ck_tile::get_x_indices_from_distributed_indices(
                l.get_tile_distribution(), i_idx);
            const int row = tile_idx.at(ck_tile::number<0>{});
            const int q_pos = start_m + row;
            if ((!HasPadding) || (q_pos < N_Q && q_pos < seqlen_val)) {
                float l_i = l[i_idx];
                float m_i = m[i_idx];
                lse[static_cast<int64_t>(bh) * stride_lz +
                    static_cast<int64_t>(q_pos) * stride_lm] =
                    (l_i > 1e-10f) ? (m_i + log2f(l_i)) : -INFINITY;
            }
        });
    }

    auto store_one_o = [&](auto& o_acc, int out_d_base) {
        auto o_out = pv_key_areg_breg_block_gemm.MakeOuputLayout(o_acc);
        constexpr auto o_store_spans = decltype(o_out)::get_distributed_spans();
        ck_tile::sweep_tile_span(o_store_spans[ck_tile::number<0>{}], [&](auto idx0) {
            ck_tile::sweep_tile_span(o_store_spans[ck_tile::number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                const auto tile_idx = ck_tile::get_x_indices_from_distributed_indices(
                    o_out.get_tile_distribution(), i_j_idx);
                const int row = tile_idx.at(ck_tile::number<0>{});
                const int col = tile_idx.at(ck_tile::number<1>{});
                const int q_pos = start_m + row;
                const int physical_col = [&]() {
                    if constexpr (PipelinePolicy::kUseInterleavedV &&
                                  N_TILE == 128 && KEY_TILE == 32) {
                        constexpr int kColumnsPerIteration = 64;
                        constexpr int kColumnsPerWave      = 16;
                        constexpr int kInterleaveStride    = 2;
                        const int n_iter = col / kColumnsPerIteration;
                        const int n_wave =
                            (col % kColumnsPerIteration) / kColumnsPerWave;
                        const int n_lane = col % kColumnsPerWave;
                        return n_wave * (kColumnsPerWave * kInterleaveStride) +
                               n_lane * kInterleaveStride + n_iter;
                    } else {
                        return col;
                    }
                }();
                const int d_pos = out_d_base + physical_col;
                if (((!HasPadding) || (q_pos < N_Q && q_pos < seqlen_val)) && d_pos < head_dim) {
                    const float inv_l = scratch.pv.row_smem[row];
                    const auto o_val = ck_tile::type_convert<DataType>(o_out[i_j_idx] * inv_l);
                    o_bh[static_cast<int64_t>(q_pos) * stride_om +
                         static_cast<int64_t>(d_pos) * stride_ok] =
                        ck_tile::bit_cast<uint16_t>(o_val);
                }
            });
        });
    };
    store_one_o(o_acc0, 0);
    store_one_o(o_acc1, N_TILE);
    }
};
