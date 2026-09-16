// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_attention_bias_enum.hpp>
#include <ck_tile/ops/fmha/block/block_rotary_embedding.hpp>
#include <ck_tile/ops/fmha/block/variants.hpp>
#include <ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs_default_policy.hpp>
#include <ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp>
#include <ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_problem.hpp>
#include <ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac_impl.hpp>
#include <ck_tile/ops/reduce.hpp>

namespace ua {

struct CommonArgs
{
    const void* q;
    const void* k;
    const void* v;
    void* out;
    const int32_t* cu_q;
    const int32_t* seq_lens;
    const int32_t* block_table;
    const float* sinks;
    const float* alibi;
    const void* qq_bias;
    const int32_t* mm_ranges;
    int64_t q_s0, q_s1;
    int64_t k_s0, k_s1, k_s2, k_s3;
    int64_t v_s0, v_s1, v_s2, v_s3;
    int64_t o_s0, o_s1;
    int64_t bt_s0;
    int64_t bias_s0;
    int num_tokens;
    int num_seqs;
    int num_q_heads;
    int num_kv_heads;
    int head_size;
    int block_size;
    int max_mm_ranges;
    float scale;
    float softcap;
    int sliding_window;
    bool use_sinks;
    bool use_alibi;
    bool use_qq_bias;
    bool use_mm_prefix;
    bool alibi_sqrt;
};

// Compact argument pack for the common 3D decode specialization. q/k/v are
// required to be contiguous by the C++ binding, so their strides are derived
// in the kernel. Keeping optional-feature pointers and eleven 64-bit tensor
// strides out of the kernarg sharply reduces scalar register pressure.
struct SimpleDecodeArgs
{
    const void* q;
    const void* k;
    const void* v;
    const int32_t* cu_q;
    const int32_t* seq_lens;
    const int32_t* block_table;
    const float* sinks;
    const float* alibi;
    int64_t bt_s0;
    int num_q_heads;
    int num_kv_heads;
    int numq_per_kv;
    int head_size;
    float scale;
    int sliding_window;
    bool use_sinks;
};

CK_TILE_DEVICE bool map_query_block(const CommonArgs& a,
                                    int global_block,
                                    int block_m,
                                    int& seq,
                                    int& local_block)
{
    int remaining = global_block;
    for(int s = 0; s < a.num_seqs; ++s)
    {
        const int qlen   = a.cu_q[s + 1] - a.cu_q[s];
        const int blocks = (qlen + block_m - 1) / block_m;
        if(remaining < blocks)
        {
            seq         = s;
            local_block = remaining;
            return true;
        }
        remaining -= blocks;
    }
    return false;
}

CK_TILE_DEVICE bool mm_allowed(const CommonArgs& a, int seq, int qpos, int kpos)
{
    if(!a.use_mm_prefix)
        return false;
    for(int i = 0; i < a.max_mm_ranges; ++i)
    {
        const int64_t off = (static_cast<int64_t>(seq) * a.max_mm_ranges + i) * 2;
        const int begin   = a.mm_ranges[off];
        const int end     = a.mm_ranges[off + 1];
        if(begin < end && qpos >= begin && qpos <= end && kpos >= begin && kpos <= end)
            return true;
    }
    return false;
}

struct UnifiedAttentionMaskTag
{
    static constexpr bool IsMasking = true;
};

struct UnifiedAttentionTraits
{
    static constexpr bool kPadSeqLenQ       = true;
    static constexpr bool kPadSeqLenK       = true;
    static constexpr bool kPadHeadDimQ      = true;
    static constexpr bool kPadHeadDimV      = true;
    static constexpr bool kStoreLSE         = true;
    static constexpr bool kDoFp8StaticQuant = false;
    static constexpr bool kIsPagedKV        = true;
    static constexpr bool kHasUnevenSplits  = true;
    // The unified pipeline handles these runtime features itself; the shared
    // FMHA problem supplies only the tile and policy types.
    static constexpr bool kHasLogitsSoftCap          = false;
    static constexpr bool kMergeNumHeadGroupsSeqLenQ = false;
    static constexpr bool kHasSink                   = false;
    static constexpr int kBlockPerCu        = 1;
    static constexpr auto BiasEnum = ck_tile::BlockAttentionBiasEnum::ALIBI;
};

template <typename DataType>
struct UnifiedAttentionMmacImpl;

template <>
struct UnifiedAttentionMmacImpl<ck_tile::half_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <>
struct UnifiedAttentionMmacImpl<ck_tile::bf16_t>
{
    using Type = ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <typename DataType>
struct UnifiedAttentionWarpGemm
{
    using WarpGemmAttribute = ck_tile::WarpGemmAttributeMmacIterateK<
        typename UnifiedAttentionMmacImpl<DataType>::Type,
        1,
        1,
        1,
        1,
        1>;

    static constexpr ck_tile::index_t kM = WarpGemmAttribute::kM;
    static constexpr ck_tile::index_t kN = WarpGemmAttribute::kN;
    static constexpr ck_tile::index_t kK = WarpGemmAttribute::kK;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;
    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;
    using CWarpOutputDstrEncoding =
        typename WarpGemmAttribute::CWarpOutputDstrEncoding;
    using AWarpDstr = ck_tile::remove_cvref_t<decltype(
        ck_tile::make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = ck_tile::remove_cvref_t<decltype(
        ck_tile::make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = ck_tile::remove_cvref_t<decltype(
        ck_tile::make_static_tile_distribution(CWarpDstrEncoding{}))>;
    using CWarpOutputDstr = ck_tile::remove_cvref_t<decltype(
        ck_tile::make_static_tile_distribution(CWarpOutputDstrEncoding{}))>;
    using AWarpTensor = ck_tile::static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = ck_tile::static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = ck_tile::static_distributed_tensor<CDataType, CWarpDstr>;
    using CWarpOutputTensor =
        ck_tile::static_distributed_tensor<CDataType, CWarpOutputDstr>;

    CK_TILE_DEVICE void operator()(CWarpTensor& c,
                                   const AWarpTensor& a,
                                   const BWarpTensor& b) const
    {
        WarpGemmAttribute{}(
            c.get_thread_buffer().template get_as<
                typename WarpGemmAttribute::Impl::CVecType>(),
            a.get_thread_buffer().template get_as<
                typename WarpGemmAttribute::Impl::AVecType>(),
            b.get_thread_buffer().template get_as<
                typename WarpGemmAttribute::Impl::BVecType>());
    }

    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c) const
    {
        return WarpGemmAttribute{}.MakeCOutputLayout(c);
    }
};

// CK Tile's A-register/B-LDS block GEMM exposes the native MMAC accumulator
// layout. Unified attention also needs the exact A tile layout (for Q and P)
// and the logical C layout (for mask/softmax/output), so expose both here.
template <typename Problem_, typename Policy_>
struct UnifiedAttentionBlockGemmARegBSmemCReg
    : ck_tile::BlockGemmARegBSmemCRegV2<Problem_, Policy_>
{
    using Base           = ck_tile::BlockGemmARegBSmemCRegV2<Problem_, Policy_>;
    using Problem        = ck_tile::remove_cvref_t<Problem_>;
    using Policy         = ck_tile::remove_cvref_t<Policy_>;
    using ADataType      = ck_tile::remove_cvref_t<typename Problem::ADataType>;
    using CDataType      = ck_tile::remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = ck_tile::remove_cvref_t<typename Problem::BlockGemmShape>;

    using Base::MakeCBlockTile;
    using Base::operator();

    CK_TILE_HOST_DEVICE static constexpr auto MakeABlockTile()
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = ck_tile::remove_cvref_t<decltype(config.template at<0>())>;
        constexpr ck_tile::index_t MWarp = config.template at<1>();
        constexpr ck_tile::index_t NWarp = config.template at<2>();
        constexpr ck_tile::index_t MIterPerWarp =
            BlockGemmShape::kM / (MWarp * WG::kM);
        constexpr ck_tile::index_t KIterPerWarp = BlockGemmShape::kK / WG::kK;

        constexpr auto outer = ck_tile::tile_distribution_encoding<
            ck_tile::sequence<NWarp>,
            ck_tile::tuple<ck_tile::sequence<MIterPerWarp, MWarp>,
                           ck_tile::sequence<KIterPerWarp>>,
            ck_tile::tuple<ck_tile::sequence<1, 0>>,
            ck_tile::tuple<ck_tile::sequence<1, 0>>,
            ck_tile::sequence<1, 2>,
            ck_tile::sequence<0, 0>>{};
        constexpr auto encoding = ck_tile::detail::make_embed_tile_distribution_encoding(
            outer, typename WG::AWarpDstrEncoding{});
        constexpr auto distribution = ck_tile::make_static_tile_distribution(encoding);
        return ck_tile::make_static_distributed_tensor<ADataType>(distribution);
    }

    template <typename CBlockTensor>
    CK_TILE_DEVICE auto MakeCOutputLayout(const CBlockTensor& c) const
    {
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = ck_tile::remove_cvref_t<decltype(config.template at<0>())>;
        constexpr ck_tile::index_t MWarp = config.template at<1>();
        constexpr ck_tile::index_t NWarp = config.template at<2>();
        constexpr ck_tile::index_t MIterPerWarp =
            BlockGemmShape::kM / (MWarp * WG::kM);
        constexpr ck_tile::index_t NIterPerWarp =
            BlockGemmShape::kN / (NWarp * WG::kN);

        constexpr auto outer = ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<MIterPerWarp, MWarp>,
                           ck_tile::sequence<NIterPerWarp, NWarp>>,
            ck_tile::tuple<ck_tile::sequence<1, 2>>,
            ck_tile::tuple<ck_tile::sequence<1, 1>>,
            ck_tile::sequence<1, 2>,
            ck_tile::sequence<0, 0>>{};
        constexpr auto output_encoding =
            ck_tile::detail::make_embed_tile_distribution_encoding(
                outer, typename WG::CWarpOutputDstrEncoding{});
        constexpr auto output_distribution =
            ck_tile::make_static_tile_distribution(output_encoding);
        auto output =
            ck_tile::make_static_distributed_tensor<CDataType>(output_distribution);

        using CWarpDstr         = typename WG::CWarpDstr;
        using CWarpOutputDstr   = typename WG::CWarpOutputDstr;
        using CWarpTensor       = typename WG::CWarpTensor;
        using CWarpOutputTensor = typename WG::CWarpOutputTensor;
        constexpr auto c_lengths = ck_tile::to_sequence(
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto out_lengths = ck_tile::to_sequence(
            CWarpOutputDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_zeros =
            ck_tile::uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        constexpr auto out_zeros =
            ck_tile::uniform_sequence_gen_t<CWarpOutputDstr::NDimY, 0>{};

        ck_tile::static_for<0, MIterPerWarp, 1>{}([&](auto m_iter) {
            ck_tile::static_for<0, NIterPerWarp, 1>{}([&](auto n_iter) {
                CWarpTensor c_warp;
                c_warp.get_thread_buffer() = c.get_y_sliced_thread_data(
                    ck_tile::merge_sequences(
                        ck_tile::sequence<m_iter, n_iter>{}, c_zeros),
                    ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_lengths));
                const CWarpOutputTensor out_warp = WG{}.MakeCOutputLayout(c_warp);
                output.set_y_sliced_thread_data(
                    ck_tile::merge_sequences(
                        ck_tile::sequence<m_iter, n_iter>{}, out_zeros),
                    ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, out_lengths),
                    out_warp.get_thread_buffer());
            });
        });
        return output;
    }
};

struct UnifiedAttentionPolicy
    : ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVSDefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem = ck_tile::BlockGemmProblem<
            typename Problem::QDataType,
            typename Problem::KDataType,
            typename Problem::SaccDataType,
            Problem::kNumGemm0Warps * ck_tile::get_warp_size(),
            ck_tile::TileGemmShape<
                ck_tile::sequence<Problem::BlockFmhaShape::kM0,
                                  Problem::BlockFmhaShape::kN0,
                                  Problem::BlockFmhaShape::kK0>,
                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                typename Problem::BlockFmhaShape::Gemm0WarpTile>>;
        using WarpGemm = UnifiedAttentionWarpGemm<typename Problem::QDataType>;
        using BlockPolicy = ck_tile::BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::QDataType,
            typename Problem::KDataType,
            typename Problem::SaccDataType,
            typename Problem::BlockFmhaShape::Gemm0BlockWarps,
            WarpGemm>;
        return UnifiedAttentionBlockGemmARegBSmemCReg<GemmProblem, BlockPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        using GemmProblem = ck_tile::BlockGemmProblem<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            Problem::kNumGemm1Warps * ck_tile::get_warp_size(),
            ck_tile::TileGemmShape<
                ck_tile::sequence<Problem::BlockFmhaShape::kM0,
                                  Problem::BlockFmhaShape::kN1,
                                  Problem::BlockFmhaShape::kK1>,
                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                typename Problem::BlockFmhaShape::Gemm1WarpTile>>;
        using WarpGemm = UnifiedAttentionWarpGemm<typename Problem::PDataType>;
        using BlockPolicy = ck_tile::BlockGemmARegBSmemCRegV2CustomPolicy<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            typename Problem::BlockFmhaShape::Gemm1BlockWarps,
            WarpGemm>;
        return UnifiedAttentionBlockGemmARegBSmemCReg<GemmProblem, BlockPolicy>{};
    }
};

template <typename DataType_,
          ck_tile::index_t HeadDim_,
          ck_tile::index_t M_ = 32,
          bool GroupQHeads_   = false,
          ck_tile::index_t PagedBlockSize_ = 0,
          bool SimpleDecode_ = false,
          bool FastAlibi_ = false>
struct UnifiedAttentionProblem
{
    using DataType = DataType_;

    static constexpr ck_tile::index_t kM           = M_;
    static constexpr ck_tile::index_t kN =
        GroupQHeads_ && HeadDim_ > 32 ? 64 : 32;
    static constexpr ck_tile::index_t kK0          = 16;
    static constexpr ck_tile::index_t kK1          = 32;
    static constexpr ck_tile::index_t kHeadDim     = HeadDim_;
    static constexpr ck_tile::index_t kWarpTileM   = 16;
    static constexpr ck_tile::index_t kWarpTileN   = 16;
    static constexpr ck_tile::index_t kWarpTileK   = 16;
    static constexpr bool kGroupQHeads              = GroupQHeads_;
    // A non-zero value specializes paged-KV division/modulo and lets each
    // loader thread reuse one page-table result for all elements it owns.
    // Zero keeps the fully generic runtime-stride path used by 2D/fallbacks.
    static constexpr ck_tile::index_t kPagedBlockSize = PagedBlockSize_;
    // The common decode path has no ALiBi, QQ bias, softcap, or MM-prefix.
    // Compiling those branches out materially reduces scalar live ranges.
    static constexpr bool kSimpleDecode = SimpleDecode_;
    // Linear ALiBi keeps the compact decode arguments and the same paged
    // Q/K/V pipeline; only the score epilogue differs from plain decode.
    static constexpr bool kFastAlibi = FastAlibi_;

    static_assert(kHeadDim == 32 || kHeadDim == 192 || kHeadDim == 256);
    static_assert(kHeadDim % kK0 == 0);
    static_assert(kM == 16 || kM == 32);
    static_assert(!kGroupQHeads || kM == 16);
    static_assert(kPagedBlockSize == 0 || kPagedBlockSize == 16 ||
                  kPagedBlockSize == 32);

    // Decode puts the Q heads sharing one KV head in M=16 and distributes the
    // sequence/output N dimension over four waves for non-small head sizes.
    // Small heads keep two waves because their PV N dimension is only 32.
    using Gemm0BlockWarps = std::conditional_t<
        kGroupQHeads,
        std::conditional_t<(kHeadDim > 32),
                           ck_tile::sequence<1, 4, 1>,
                           ck_tile::sequence<1, 2, 1>>,
        ck_tile::sequence<2, 1, 1>>;
    using Gemm1BlockWarps = Gemm0BlockWarps;

    using BlockFmhaShape = ck_tile::TileFmhaShape<
        ck_tile::sequence<kM, kN, kK0, kHeadDim, kK1, kHeadDim>,
        Gemm0BlockWarps,
        ck_tile::sequence<kWarpTileM, kWarpTileN, kWarpTileK>,
        Gemm1BlockWarps,
        ck_tile::sequence<kWarpTileM, kWarpTileN, kWarpTileK>,
        true>;

    using FmhaProblem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
        DataType,
        DataType,
        DataType,
        float,
        float,
        float,
        float,
        DataType,
        float,
        DataType,
        BlockFmhaShape,
        false,
        ck_tile::StandardAttention,
        UnifiedAttentionMaskTag,
        UnifiedAttentionTraits>;

    using Policy = UnifiedAttentionPolicy;

    static constexpr ck_tile::index_t kBlockSize = FmhaProblem::kBlockSize;
    // Avoid the 32-FP16 (=16-bank) row stride used by the logical V tile.
    // Two padding elements rotate consecutive D rows by 17 banks instead.
    static constexpr ck_tile::index_t kVSmemStride =
        kPagedBlockSize > 0 ? kK1 + 2 : kK1;
    static constexpr ck_tile::index_t kKSmemRows =
        kSimpleDecode ? (kHeadDim <= 192 ? kHeadDim + 4 : 128 + 4) : kK0;
    static constexpr ck_tile::index_t kKSmemElements = kN * kKSmemRows;
    static constexpr ck_tile::index_t kSmemElements =
        ck_tile::max(kHeadDim * kVSmemStride,
                     ck_tile::max(kM * kN, kKSmemElements));
    static constexpr ck_tile::index_t kDataSmemSize =
        kSmemElements * sizeof(DataType);
    static constexpr ck_tile::index_t kNumWarps =
        kBlockSize / ck_tile::get_warp_size();
    static constexpr ck_tile::index_t kReductionScratchSize =
        2 * kM * kNumWarps * sizeof(float);
    static constexpr ck_tile::index_t kSmemSize =
        kDataSmemSize + kReductionScratchSize;

    CK_TILE_HOST_DEVICE static constexpr auto MakeKLoadDistribution()
    {
        return ck_tile::tile_distribution_encoding_pattern_2d<
            kBlockSize,
            kN,
            kK0,
            4,
            ck_tile::tile_distribution_pattern::thread_raked,
            1>::make_2d_static_tile_distribution();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeVLoadDistribution()
    {
        return ck_tile::tile_distribution_encoding_pattern_2d<
            kBlockSize,
            kHeadDim,
            kK1,
            4,
            ck_tile::tile_distribution_pattern::thread_raked,
            1>::make_2d_static_tile_distribution();
    }

    // Global V is laid out [page, token, kv_head, dim], hence dim is the
    // contiguous axis. Load [K1, HeadDim] vectors and transpose them while
    // writing LDS for the P@V block GEMM, whose B tile is [HeadDim, K1].
    CK_TILE_HOST_DEVICE static constexpr auto MakeVContiguousLoadDistribution()
    {
        return ck_tile::tile_distribution_encoding_pattern_2d<
            kBlockSize,
            kK1,
            kHeadDim,
            4,
            ck_tile::tile_distribution_pattern::thread_raked,
            1>::make_2d_static_tile_distribution();
    }

};

} // namespace ua
