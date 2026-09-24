// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gdn/pipeline/gdn_fwd_output_problem.hpp"
#include <ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1.hpp>

namespace gdn {

// The current CK checkout exposes a generic WarpGemmImpl and a separate MMAC
// WarpGemmImpl with the same class name, so including both headers is invalid.
// Keep the public BlockGemm implementations and provide only the MMAC-specific
// register-buffer invocation they require under a GDN-local type name.
template <typename WarpGemmAttribute_>
struct GdnOutputMmacWarpGemm
{
    using WarpGemmAttribute = ck_tile::remove_cvref_t<WarpGemmAttribute_>;
    static constexpr ck_tile::index_t kM = WarpGemmAttribute::kM;
    static constexpr ck_tile::index_t kN = WarpGemmAttribute::kN;
    static constexpr ck_tile::index_t kK = WarpGemmAttribute::kK;
    static constexpr ck_tile::index_t kKPerThread = WarpGemmAttribute::kKPerThread;

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
    using AWarpTensor = ck_tile::static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = ck_tile::static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = ck_tile::static_distributed_tensor<CDataType, CWarpDstr>;

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
};

template <typename DataType>
struct GdnOutputMmacImpl;

template <>
struct GdnOutputMmacImpl<ck_tile::half_t>
{
    using type = ck_tile::WarpGemmAttributeMmacImplF16F16F32M16N16K16;
};

template <>
struct GdnOutputMmacImpl<ck_tile::bf16_t>
{
    using type = ck_tile::WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;
};

template <typename DataType, ck_tile::index_t K, ck_tile::index_t NPerWarp>
using GdnOutputWarpGemm = GdnOutputMmacWarpGemm<ck_tile::WarpGemmAttributeMmacIterateK<
    typename GdnOutputMmacImpl<DataType>::type,
    1,
    NPerWarp / 16,
    1,
    1,
    K / 16>>;

template <typename DataType, ck_tile::index_t K, ck_tile::index_t NPerWarp>
using GdnOutputPreshuffleWarpGemm =
    GdnOutputMmacWarpGemm<ck_tile::WarpGemmAttributeMmacIterateKShuffle<
        typename GdnOutputMmacImpl<DataType>::type,
        1,
        NPerWarp / 16,
        1,
        1,
        K / 16>>;

template <typename Problem>
struct GdnOutputFwdPolicy
{
    using DataType = typename Problem::DataType;

    template <ck_tile::index_t Rows,
              ck_tile::index_t Cols,
              ck_tile::index_t Vector = 8,
              ck_tile::index_t NumWarps = 4>
    struct GlobalCopyGeometry
    {
        static constexpr ck_tile::index_t kWarpSize = 64;
        static constexpr ck_tile::index_t kLanesPerRow = Cols / Vector;
        static constexpr ck_tile::index_t kRowGroups = kWarpSize / kLanesPerRow;
        static constexpr ck_tile::index_t kNumIssues =
            Rows / (kRowGroups * NumWarps);

        static_assert(Cols % Vector == 0,
                      "global copy expects vector-aligned columns");
        static_assert(Rows % (kRowGroups * NumWarps) == 0,
                      "global copy rows do not fit the wave geometry");
    };

    // Logical [Rows, Cols] DRAM tile. The vector dimension is kept as a
    // compile-time Y dimension, so the pipeline never derives token/value
    // coordinates from tid at runtime.
    template <ck_tile::index_t Rows,
              ck_tile::index_t Cols,
              ck_tile::index_t Vector = 8,
              ck_tile::index_t NumWarps = 4>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalCopyDistribution()
    {
        using G = GlobalCopyGeometry<Rows, Cols, Vector, NumWarps>;
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<1>,
                ck_tile::tuple<
                    ck_tile::sequence<G::kNumIssues, G::kRowGroups, NumWarps>,
                    ck_tile::sequence<G::kLanesPerRow, Vector>>,
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<1, 0>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 1>>{});
    }

    // Epilogue copies assign exactly one vector to every CTA lane.  The vector
    // width is derived from the logical tile at compile time (vec4 for 32x32,
    // vec8 for 32x64 or 64x32) instead of being recomputed from tid in the
    // pipeline.
    template <ck_tile::index_t Rows, ck_tile::index_t Cols>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOutputCopyDistribution()
    {
        constexpr ck_tile::index_t Vector =
            Rows * Cols / Problem::Config::kBlockSize;
        static_assert(Vector == 4 || Vector == 8,
                      "output copy expects vec4 or vec8 per lane");
        return MakeGlobalCopyDistribution<Rows, Cols, Vector>();
    }

    template <ck_tile::index_t Rows,
              ck_tile::index_t Cols,
              ck_tile::index_t Padding>
    CK_TILE_HOST_DEVICE static constexpr auto MakePaddedRowMajorLdsDescriptor()
    {
        return ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<Rows>{}, ck_tile::number<Cols>{}),
            ck_tile::make_tuple(ck_tile::number<Cols + Padding>{},
                                ck_tile::number<1>{}));
    }

    // The same padded row-major storage exposed as logical [Cols, Rows].
    // QH/PV consume H/V through this transposed LDS view.
    template <ck_tile::index_t Rows,
              ck_tile::index_t Cols,
              ck_tile::index_t Padding>
    CK_TILE_HOST_DEVICE static constexpr auto MakePaddedTransposeLdsDescriptor()
    {
        return ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<Cols>{}, ck_tile::number<Rows>{}),
            ck_tile::make_tuple(ck_tile::number<1>{},
                                ck_tile::number<Cols + Padding>{}));
    }

    template <ck_tile::index_t M,
              ck_tile::index_t N,
              ck_tile::index_t K,
              bool IsQK,
              bool PreshuffleK = false>
    struct BlockGemmPolicy
    {
        template <typename GemmProblem>
        CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
        {
            constexpr ck_tile::index_t NPerWarp = IsQK ? 32 : (N >= 64 ? 32 : 16);
            if constexpr(PreshuffleK)
                return ck_tile::make_tuple(
                    GdnOutputPreshuffleWarpGemm<DataType, K, NPerWarp>{},
                    ck_tile::number<(M >= 64 ? 4 : 2)>{},
                    ck_tile::number<2>{});
            else
                return ck_tile::make_tuple(GdnOutputWarpGemm<DataType, K, NPerWarp>{},
                                           ck_tile::number<(M >= 64 ? 4 : 2)>{},
                                           ck_tile::number<2>{});
        }

        template <typename GemmProblem>
        CK_TILE_HOST_DEVICE static constexpr auto MakeARegDistribution()
        {
            constexpr auto config = GetWarpGemmMWarpNWarp<GemmProblem>();
            using WG = ck_tile::remove_cvref_t<decltype(config.template at<0>())>;
            constexpr ck_tile::index_t MWarp = config.template at<1>();
            constexpr ck_tile::index_t NWarp = config.template at<2>();
            constexpr ck_tile::index_t MIter = M / (MWarp * WG::kM);
            constexpr ck_tile::index_t KIter = K / WG::kK;
            constexpr auto outer = ck_tile::tile_distribution_encoding<
                ck_tile::sequence<NWarp>,
                ck_tile::tuple<ck_tile::sequence<MIter, MWarp>,
                               ck_tile::sequence<KIter>>,
                ck_tile::tuple<ck_tile::sequence<1, 0>>,
                ck_tile::tuple<ck_tile::sequence<1, 0>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 0>>{};
            constexpr auto encoding =
                ck_tile::detail::make_embed_tile_distribution_encoding(
                    outer, typename WG::AWarpDstrEncoding{});
            return ck_tile::make_static_tile_distribution(encoding);
        }

        template <typename GemmProblem>
        CK_TILE_HOST_DEVICE static constexpr auto MakeBRegDistribution()
        {
            constexpr auto config = GetWarpGemmMWarpNWarp<GemmProblem>();
            using WG = ck_tile::remove_cvref_t<decltype(config.template at<0>())>;
            constexpr ck_tile::index_t MWarp = config.template at<1>();
            constexpr ck_tile::index_t NWarp = config.template at<2>();
            constexpr ck_tile::index_t NIter = N / (NWarp * WG::kN);
            constexpr ck_tile::index_t KIter = K / WG::kK;
            constexpr auto outer = ck_tile::tile_distribution_encoding<
                ck_tile::sequence<MWarp>,
                ck_tile::tuple<ck_tile::sequence<NIter, NWarp>,
                               ck_tile::sequence<KIter>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 0>>{};
            constexpr auto encoding =
                ck_tile::detail::make_embed_tile_distribution_encoding(
                    outer, typename WG::BWarpDstrEncoding{});
            return ck_tile::make_static_tile_distribution(encoding);
        }

        template <typename GemmProblem>
        CK_TILE_HOST_DEVICE static constexpr auto MakeCRegDistribution()
        {
            constexpr auto config = GetWarpGemmMWarpNWarp<GemmProblem>();
            using WG = ck_tile::remove_cvref_t<decltype(config.template at<0>())>;
            constexpr ck_tile::index_t MWarp = config.template at<1>();
            constexpr ck_tile::index_t NWarp = config.template at<2>();
            constexpr ck_tile::index_t MIter = M / (MWarp * WG::kM);
            constexpr ck_tile::index_t NIter = N / (NWarp * WG::kN);
            constexpr auto outer = ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<MIter, MWarp>,
                               ck_tile::sequence<NIter, NWarp>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 0>>{};
            constexpr auto encoding =
                ck_tile::detail::make_embed_tile_distribution_encoding(
                    outer, typename WG::CWarpDstrEncoding{});
            return ck_tile::make_static_tile_distribution(encoding);
        }

        template <typename GemmProblem>
        CK_TILE_HOST_DEVICE static constexpr auto MakeCOutputRegDistribution()
        {
            constexpr auto config = GetWarpGemmMWarpNWarp<GemmProblem>();
            using WG = ck_tile::remove_cvref_t<decltype(config.template at<0>())>;
            using WarpAttribute = typename WG::WarpGemmAttribute;
            constexpr ck_tile::index_t MWarp = config.template at<1>();
            constexpr ck_tile::index_t NWarp = config.template at<2>();
            constexpr ck_tile::index_t MIter = M / (MWarp * WG::kM);
            constexpr ck_tile::index_t NIter = N / (NWarp * WG::kN);
            constexpr auto outer = ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<MIter, MWarp>,
                               ck_tile::sequence<NIter, NWarp>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 0>>{};
            constexpr auto encoding =
                ck_tile::detail::make_embed_tile_distribution_encoding(
                    outer, typename WarpAttribute::CWarpOutputDstrEncoding{});
            return ck_tile::make_static_tile_distribution(encoding);
        }
    };

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using BRegBlockGemmPolicy = BlockGemmPolicy<M, N, K, false, false>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using QKBlockGemmPolicy = BlockGemmPolicy<M, N, K, true, false>;

    // Stage-specific GEMM types. QK keeps K in LDS and lets the public
    // AReg/BSmem implementation form the MMAC B fragments. QH and PV keep
    // their B operands in registers because H/V are reused by both row tiles,
    // while PV keeps score in AReg across all value tiles.
    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using QKProblem = typename Problem::template GemmProblem<M, N, K>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using QKPolicy = QKBlockGemmPolicy<M, N, K>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using QKBlockGemm = ck_tile::BlockGemmARegBSmemCRegV1<
        QKProblem<M, N, K>, QKPolicy<M, N, K>>;

    // Two-row pipeline variant: K is physically [token, head_dim], exactly
    // the logical [N,K] QK-B matrix.  Load it once into the native MMAC BReg
    // distribution and reuse it for both row tiles, avoiding the 16-KiB K LDS
    // allocation used by the generic/short-sequence path.
    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using QKBlockGemmBReg = ck_tile::BlockGemmARegBRegCRegV1<
        QKProblem<M, N, K>, QKPolicy<M, N, K>>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using QHProblem = typename Problem::template GemmProblem<M, N, K>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using QHPolicy = BlockGemmPolicy<M, N, K, false, Problem::kPreshuffledH>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using QHBlockGemm = ck_tile::BlockGemmARegBRegCRegV1<
        QHProblem<M, N, K>, QHPolicy<M, N, K>>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using PVProblem = typename Problem::template GemmProblem<M, N, K>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using PVPolicy = BRegBlockGemmPolicy<M, N, K>;

    template <ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K>
    using PVBlockGemm = ck_tile::BlockGemmARegBRegCRegV1<
        PVProblem<M, N, K>, PVPolicy<M, N, K>>;

    template <typename GemmProblem, typename GemmPolicy>
    CK_TILE_DEVICE static constexpr auto MakeARegTile()
    {
        return ck_tile::make_static_distributed_tensor<
            typename GemmProblem::ADataType>(
            GemmPolicy::template MakeARegDistribution<GemmProblem>());
    }

    template <typename GemmProblem, typename GemmPolicy>
    CK_TILE_DEVICE static constexpr auto MakeBRegTile()
    {
        return ck_tile::make_static_distributed_tensor<
            typename GemmProblem::BDataType>(
            GemmPolicy::template MakeBRegDistribution<GemmProblem>());
    }

    // Native gfx936 preshuffled-H load.  Physical H is
    // [N/16, K/8, N_lane=16, K_vec=8].  The N-lane dimension makes the
    // wave's 16-byte transactions adjacent, while each lane receives the
    // exact K-vector order consumed by IterateKShuffle; no LDS or lane
    // shuffle is required.
    template <typename GemmProblem, typename GemmPolicy>
    CK_TILE_DEVICE static auto LoadPreshuffledH(const DataType* h_head_base,
                                                ck_tile::index_t value_begin)
    {
        using Vec8 = ck_tile::ext_vector_t<DataType, 8>;
        auto tile = MakeBRegTile<GemmProblem, GemmPolicy>();
        auto& vec = tile.get_thread_buffer().template get_as<Vec8>();
        const ck_tile::index_t wave = ck_tile::get_thread_id() / 64;
        const ck_tile::index_t lane = ck_tile::get_lane_id();
        const ck_tile::index_t n_local = (wave & 1) * 16 + (lane & 15);
        const ck_tile::index_t n = value_begin + n_local;
        const ck_tile::index_t k_lane = (lane >> 4) & 3;
        ck_tile::static_for<0, 4, 1>{}([&](auto k_pair) {
            const ck_tile::index_t k_base = k_pair * 32 + k_lane * 8;
            const ck_tile::index_t offset =
                (((n / 16) * 16 + k_base / 8) * 16 + n % 16) * 8;
            vec(k_pair) = *reinterpret_cast<const Vec8*>(h_head_base + offset);
        });
        return tile;
    }

    // Convert the hardware MMAC accumulator distribution to the logical
    // [M, N] register distribution consumed by gating and the epilogue.
    template <typename GemmProblem, typename GemmPolicy, typename CBlockTensor>
    CK_TILE_DEVICE static auto MakeCOutputLayout(const CBlockTensor& input)
    {
        using BlockGemmShape = typename GemmProblem::BlockGemmShape;
        using CDataType      = typename GemmProblem::CDataType;
        constexpr auto config =
            GemmPolicy::template GetWarpGemmMWarpNWarp<GemmProblem>();
        using WG = ck_tile::remove_cvref_t<decltype(config.template at<0>())>;
        using WarpAttribute = typename WG::WarpGemmAttribute;
        constexpr ck_tile::index_t MWarp = config.template at<1>();
        constexpr ck_tile::index_t NWarp = config.template at<2>();
        constexpr ck_tile::index_t MIter =
            BlockGemmShape::kM / (MWarp * WG::kM);
        constexpr ck_tile::index_t NIter =
            BlockGemmShape::kN / (NWarp * WG::kN);

        using CWarpDstr = typename WG::CWarpDstr;
        using CWarpOutputDstr = ck_tile::remove_cvref_t<decltype(
            ck_tile::make_static_tile_distribution(
                typename WarpAttribute::CWarpOutputDstrEncoding{}))>;
        using CWarpTensor = typename WG::CWarpTensor;
        using CWarpOutputTensor =
            ck_tile::static_distributed_tensor<CDataType, CWarpOutputDstr>;

        auto output = ck_tile::make_static_distributed_tensor<CDataType>(
            GemmPolicy::template MakeCOutputRegDistribution<GemmProblem>());
        constexpr auto in_lengths = ck_tile::to_sequence(
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto out_lengths = ck_tile::to_sequence(
            CWarpOutputDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto in_zeros =
            ck_tile::uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        constexpr auto out_zeros =
            ck_tile::uniform_sequence_gen_t<CWarpOutputDstr::NDimY, 0>{};

        ck_tile::static_for<0, MIter, 1>{}([&](auto m) {
            ck_tile::static_for<0, NIter, 1>{}([&](auto n) {
                CWarpTensor warp_input;
                warp_input.get_thread_buffer() = input.get_y_sliced_thread_data(
                    ck_tile::merge_sequences(ck_tile::sequence<m, n>{}, in_zeros),
                    ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, in_lengths));
                CWarpOutputTensor warp_output =
                    WarpAttribute{}.MakeCOutputLayout(warp_input);
                output.set_y_sliced_thread_data(
                    ck_tile::merge_sequences(ck_tile::sequence<m, n>{}, out_zeros),
                    ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, out_lengths),
                    warp_output.get_thread_buffer());
            });
        });
        return output;
    }
};

} // namespace gdn
