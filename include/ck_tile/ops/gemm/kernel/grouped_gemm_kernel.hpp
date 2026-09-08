// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/host/stream_utils.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
// #include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"
#include "ck_tile/host.hpp"

#include <hip/hip_runtime.h>
namespace ck_tile {

template <typename T, typename = void>
struct is_grouped_gemm_mls_pipeline : std::false_type
{
};

template <typename T>
struct is_grouped_gemm_mls_pipeline<T, std::void_t<decltype(T::IsMlsPipeline)>>
    : std::bool_constant<T::IsMlsPipeline>
{
};

/// @brief The Grouped GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref GroupedGemmKernel "GroupedGemmKernel" when creating kernel
///      arguments object. It contain all necessary information required to build proper kernel
///      argument and launch kernel on GPU. This structure defines the GEMM problem configuration by
///      stating all required information like M,N,K sizes and respective strides.
template <index_t NumDTensor = 0>
struct GroupedGemmHostArgsImpl
{
    CK_TILE_HOST GroupedGemmHostArgsImpl(const void* a_ptr_,
                                         const void* b_ptr_,
                                         void* e_ptr_,
                                         index_t k_batch_,
                                         index_t M_,
                                         index_t N_,
                                         index_t K_,
                                         index_t stride_A_,
                                         index_t stride_B_,
                                         index_t stride_E_)
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr{},
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_Ds{},
          stride_E(stride_E_),
          k_batch(k_batch_)
    {
    }

    CK_TILE_HOST GroupedGemmHostArgsImpl(const void* a_ptr_,
                                         const void* b_ptr_,
                                         const std::array<const void*, NumDTensor>& ds_ptr_,
                                         void* e_ptr_,
                                         index_t k_batch_,
                                         index_t M_,
                                         index_t N_,
                                         index_t K_,
                                         index_t stride_A_,
                                         index_t stride_B_,
                                         const std::array<index_t, NumDTensor>& stride_Ds_,
                                         index_t stride_E_)
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };

    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    std::array<index_t, NumDTensor> stride_Ds;

    union
    {
        index_t stride_E;
        index_t stride_C;
    };

    index_t k_batch;
};

using GroupedGemmHostArgs = GroupedGemmHostArgsImpl<0>;

template <index_t NumDTensor = 0>
struct GemmTransKernelArgImpl
{
    UniversalGemmKernelArgs<1, 1, NumDTensor> group_karg;
    ck_tile::index_t block_start;
    ck_tile::index_t block_end;

    GemmTransKernelArgImpl() = delete;
    GemmTransKernelArgImpl(UniversalGemmKernelArgs<1, 1, NumDTensor>&& karg,
                           index_t bl_start,
                           index_t bl_end)
        : group_karg{karg},
          block_start{bl_start},
          block_end{bl_end}
    {
    }

    GemmTransKernelArgImpl(UniversalGemmKernelArgs<1, 1, NumDTensor>&& karg)
        : group_karg{karg},
          block_start{0},
          block_end{0}
    {
    }
};

using GemmTransKernelArg = GemmTransKernelArgImpl<0>;

enum class GroupedGemmMFilter
{
    All,
    Aligned,
    Unaligned
};

template <typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_,
          GroupedGemmMFilter MFilter_ = GroupedGemmMFilter::All,
          index_t MFilterAlign_       = 0>
struct GroupedGemmKernel
{
    /// @brief Inject the UniversalGemmKernel base class to support execution of all necessary
    /// functions.
    using Base = UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    //// @brief Specify the layout configurations for A, B, C/E
    using ALayout = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout = remove_cvref_t<typename GemmPipeline::CLayout>;

    /// @brief Specify the data type configurations for A, B, C/E
    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    /// @brief ALayout and ADataType are expected to be scalars, not a tuple.
    static_assert(
        !is_detected<is_tuple, ALayout>::value && !is_detected<is_tuple, ADataType>::value,
        "ALayout and ADataType must be scalars. Multiple parameters are not currently supported.");

    /// @brief  BLayout and BDataType are expected to be scalars, not a tuple.
    static_assert(
        !is_detected<is_tuple, BLayout>::value && !is_detected<is_tuple, BDataType>::value,
        "BLayout and BDataType must be scalars. Multiple parameters are not currently supported.");

    /// @brief  C/ELayout and C/EDataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, CLayout>::value &&
                      !is_detected<is_tuple, CDataType>::value,
                  "C/ELayout and C/EDataType must be scalars.");

    using OffsetTile1DPartitioner = OffsettedTile1DPartitioner<TilePartitioner>;
    using Kernel = GroupedGemmKernel<TilePartitioner,
                                     GemmPipeline,
                                     EpiloguePipeline,
                                     MFilter_,
                                     MFilterAlign_>;

    static constexpr index_t kBlockSize       = GemmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = GemmPipeline::UsePersistentKernel;
    static constexpr index_t NumDTensor       = EpiloguePipeline::NumDTensor;
    using HostArgs                            = GroupedGemmHostArgsImpl<NumDTensor>;
    using KernelArgs                          = UniversalGemmKernelArgs<1, 1, NumDTensor>;
    using TransKernelArg                      = GemmTransKernelArgImpl<NumDTensor>;

    // Hybrid M-only launches may pair a large aligned tile with a smaller padded
    // tail. Filter both kernels against the aligned tile so groups such as
    // M=1920 (128-mod-256) are not dropped by both sides.
    CK_TILE_HOST_DEVICE static constexpr index_t FilterMPerBlock()
    {
        return MFilterAlign_ != 0 ? MFilterAlign_ : TilePartitioner::MPerBlock;
    }

    CK_TILE_HOST_DEVICE static constexpr bool ShouldProcessM(index_t m)
    {
        const index_t align = FilterMPerBlock();
        if constexpr(MFilter_ == GroupedGemmMFilter::Aligned)
        {
            return m % align == 0;
        }
        else if constexpr(MFilter_ == GroupedGemmMFilter::Unaligned)
        {
            return m % align != 0;
        }
        else
        {
            return true;
        }
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;

        return concat('_', "gemm_grouped", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK),
                      (UsePersistentKernel ? "Persistent" : "NonPersistent"),
                      (MFilter_ == GroupedGemmMFilter::Aligned
                           ? "MAligned"
                           : (MFilter_ == GroupedGemmMFilter::Unaligned ? "MUnaligned" : "MAll")),
                      FilterMPerBlock());
        // clang-format on
    }

    CK_TILE_HOST static auto
    GetWorkSpaceSize(const std::vector<HostArgs>& gemm_descs) -> std::size_t
    {
        return gemm_descs.size() * sizeof(TransKernelArg);
    }

    CK_TILE_HOST static auto GetWorkSpaceSize(index_t group_count) -> std::size_t
    {
        return group_count * sizeof(TransKernelArg);
    }

    CK_TILE_HOST static constexpr auto BlockSize() -> dim3
    {
        if constexpr (is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     *       `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        using ConstantPointer = const void CK_TILE_CONSTANT_ADDRESS_SPACE*;
        const auto kernel     = kentry<kBlockSize, 1, Kernel, ConstantPointer, index_t>;
        int device;
        HIP_CHECK_ERROR(hipGetDevice(&device));

        // Framework integrations call this path for every grouped-GEMM
        // invocation. Occupancy is invariant for one kernel on one device, but
        // querying it repeatedly is measurable at short kernel latencies.
        // Thread-local storage avoids cross-thread contention. Remember only
        // the most recently used device: framework workers normally stay on
        // one device, while a device switch still refreshes the value.
        static thread_local int cached_device    = -1;
        static thread_local int cached_occupancy = 0;
        if(device != cached_device)
        {
            HIP_CHECK_ERROR(
                hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    &cached_occupancy, kernel, kBlockSize, 0));
            cached_device = device;
        }
        const int grid_size = get_available_compute_units(s) * cached_occupancy;

        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto GridSize(const std::vector<HostArgs>& gemm_descs)
    {
        index_t grid_size = 0;
        for(const auto& it_desc : gemm_descs)
        {
            const auto local_grid_size = TilePartitioner::GridSize(it_desc.M, it_desc.N);
            grid_size += local_grid_size * it_desc.k_batch;
        }
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto
    MakeKargs(const std::vector<HostArgs>& gemm_descs) -> std::vector<TransKernelArg>
    {
        std::vector<TransKernelArg> gemm_kernel_args_;
        index_t group_count = ck_tile::type_convert<ck_tile::index_t>(gemm_descs.size());
        index_t grid_size   = 0;
        gemm_kernel_args_.reserve(group_count);

        for(std::size_t i = 0; i < gemm_descs.size(); ++i)
        {
            const index_t M = gemm_descs[i].M;
            const index_t N = gemm_descs[i].N;
            const index_t K = gemm_descs[i].K;

            if(M == 0 || N == 0 || K == 0)
            {
                continue;
            }

            const index_t stride_a = gemm_descs[i].stride_A;
            const index_t stride_b = gemm_descs[i].stride_B;
            const index_t stride_e = gemm_descs[i].stride_E;

            const index_t grid_size_grp = TilePartitioner::GridSize(M, N) * gemm_descs[i].k_batch;

            const index_t block_start = grid_size;
            const index_t block_end   = grid_size + grid_size_grp;

            grid_size += grid_size_grp;

            auto karg =
                KernelArgs{{type_convert<const ADataType*>(gemm_descs[i].a_ptr)},
                           {type_convert<const BDataType*>(gemm_descs[i].b_ptr)},
                           gemm_descs[i].ds_ptr,
                           type_convert<CDataType*>(gemm_descs[i].e_ptr),
                           M,
                           N,
                           K,
                           {stride_a},
                           {stride_b},
                           gemm_descs[i].stride_Ds,
                           stride_e,
                           gemm_descs[i].k_batch};

            gemm_kernel_args_.emplace_back(std::move(karg), block_start, block_end);
        }

        return gemm_kernel_args_;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const std::vector<TransKernelArg>& kargs)
    {
        if constexpr(is_grouped_gemm_mls_pipeline<GemmPipeline>::value)
        {
            if(get_hcu_target_enum() != hcu_target_enum::gfx938)
            {
                return false;
            }
        }

        for(const auto& karg : kargs)
        {
            if(!ShouldProcessM(karg.group_karg.M))
            {
                continue;
            }
            if(!Base::IsSupportedArgument(karg.group_karg))
            {
                return false;
            }
            if constexpr(is_grouped_gemm_mls_pipeline<GemmPipeline>::value)
            {
                const auto& group_karg = karg.group_karg;
                constexpr index_t k_warp_tile =
                    TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
                const index_t k_grain = group_karg.k_batch * k_warp_tile;
                const index_t k_read =
                    integer_divide_ceil(group_karg.K, k_grain) * k_warp_tile;
                index_t required_k_span = 0;

                for(index_t split = 0; split < group_karg.k_batch; ++split)
                {
                    const index_t split_k = split + 1 < group_karg.k_batch
                                                ? k_read
                                                : group_karg.K -
                                                      k_read * (group_karg.k_batch - 1);
                    if(split_k <= 0)
                    {
                        return false;
                    }
                    const index_t num_loop =
                        integer_divide_ceil(split_k, GemmPipeline::KPerBlock);
                    if(!GemmPipeline::IsSupported(num_loop))
                    {
                        return false;
                    }
                    required_k_span =
                        max(required_k_span,
                            split * k_read + num_loop * GemmPipeline::KPerBlock);
                }

                // MLS issues a fixed-width physical matrix load for each K tile. The tail
                // mask zero-fills invalid K values, but a compact leading dimension can
                // still cross row boundaries and fault before masking. Require physical
                // row padding through the last split; aligned K naturally satisfies this.
                const bool a_stride_safe =
                    std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>
                        ? group_karg.stride_As[0] >= required_k_span
                        : group_karg.stride_As[0] >= group_karg.M;
                const bool b_stride_safe =
                    std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>
                        ? group_karg.stride_Bs[0] >= required_k_span
                        : group_karg.stride_Bs[0] >= group_karg.N;
                if(!a_stride_safe || !b_stride_safe)
                {
                    return false;
                }
            }
        }
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSize() -> index_t
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX)
    CK_TILE_DEVICE __attribute__((noinline))
    static void FusedPackGradX(const ADataType* a_ptr,
                              CDataType* c_ptr,
                              void* p_smem,
                              const KernelArgs& kargs,
                              const index_t group_id,
                              const index_t group_count,
                              const index_t physical_block_id,
                              const index_t physical_grid_size)
    {
        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor> &&
                          std::is_same_v<BLayout, tensor_layout::gemm::RowMajor> &&
                          std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>,
                      "fused grad-X packing is specialized for transposed grad-W C,R,C");
        static_assert(std::is_same_v<ADataType, CDataType> &&
                          sizeof(ADataType) == 2,
                      "fused grad-X packing requires matching 16-bit input/output types");
        static_assert(TilePartitioner::MPerBlock == 256 && kBlockSize == 512,
                      "fused grad-X packing requires the selected 256-M/512-thread kernel");

        // The regular grad-W problem views physical row-major grad_out[K,M]
        // as column-major A[M,K]. The appended output is packed row-major
        // [M,K], which is the C,R,C grad-X operand expected by the fast lane.
        //
        // Run once per physical persistent workgroup after that workgroup has
        // completed its final GEMM tile. Keeping this out of the logical-tile
        // loop avoids interleaving LDS barriers with every GEMM iteration and
        // the noinline boundary prevents the transpose coordinates from
        // extending the GEMM hot loop's live ranges.
        if(kargs.k_batch != 1 || kargs.K != kargs.N)
        {
            return;
        }

        block_sync_lds();

        constexpr index_t TileDim   = 32;
        constexpr index_t TilePitch = 33;
        constexpr index_t TileElems = TileDim * TilePitch;
        auto* tile = reinterpret_cast<ADataType*>(p_smem);

        const index_t tid      = get_thread_id();
        const index_t tile_sel = tid >> 8;
        const index_t local    = tid & 255;
        const index_t tx       = local & 31;
        const index_t ty       = local >> 5;

        const index_t M = kargs.M;
        const index_t K = kargs.K;
        const index_t m_tiles =
            (M + TilePartitioner::MPerBlock - 1) / TilePartitioner::MPerBlock;
        const index_t k_tiles =
            (K + TilePartitioner::NPerBlock - 1) / TilePartitioner::NPerBlock;
        const index_t tasks_per_group = m_tiles * k_tiles;
        const index_t total_tasks     = group_count * tasks_per_group;

        const ADataType* global_a_base =
            a_ptr - group_id * K * kargs.stride_As[0];
        CDataType* global_c_base = c_ptr - group_id * M * kargs.N;
        auto* packed_base = reinterpret_cast<ADataType*>(
            global_c_base + group_count * M * kargs.N);

        for(index_t task = physical_block_id; task < total_tasks;
            task += physical_grid_size)
        {
            const index_t pack_group = task / tasks_per_group;
            const index_t group_task = task - pack_group * tasks_per_group;
            const index_t block_idx_m = group_task / k_tiles;
            const index_t block_idx_k = group_task - block_idx_m * k_tiles;
            const index_t m_block =
                block_idx_m * TilePartitioner::MPerBlock;
            const index_t k_block =
                block_idx_k * TilePartitioner::NPerBlock;
            const index_t k_block_end =
                min(k_block + TilePartitioner::NPerBlock, K);
            const auto* a_group =
                global_a_base + pack_group * K * kargs.stride_As[0];
            auto* packed_group = packed_base + pack_group * M * K;

            for(index_t m_sub = 0; m_sub < TilePartitioner::MPerBlock;
                m_sub += TileDim)
            {
                const index_t m0 = m_block + m_sub;
                for(index_t k_pair = k_block; k_pair < k_block_end;
                    k_pair += 2 * TileDim)
                {
                    const index_t k0 = k_pair + tile_sel * TileDim;
                    const index_t tile_base = tile_sel * TileElems;

#pragma unroll
                    for(index_t j = 0; j < TileDim; j += 8)
                    {
                        tile[tile_base + (ty + j) * TilePitch + tx] =
                            a_group[(k0 + ty + j) * kargs.stride_As[0] +
                                    m0 + tx];
                    }
                    block_sync_lds();

#pragma unroll
                    for(index_t j = 0; j < TileDim; j += 8)
                    {
                        packed_group[(m0 + ty + j) * K + k0 + tx] =
                            tile[tile_base + tx * TilePitch + ty + j];
                    }
                    block_sync_lds();
                }
            }
        }
    }
#endif

#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DSREADM)
    CK_TILE_DEVICE static void
    RepairPackedAPenultimateStage(const ADataType* a_ptr,
                                  ADataType* packed_a_ptr,
                                  const index_t source_stride,
                                  const index_t packed_stride,
                                  const index_t m_block)
    {
        static_assert(sizeof(ADataType) == 2 && kBlockSize == 512 &&
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
                          TilePartitioner::MPerBlock == 128,
#else
                          TilePartitioner::MPerBlock == 256,
#endif
                      "packed-A tail repair requires the selected BF16 tile");
        const index_t tid       = get_thread_id();
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
        if(tid >= 256)
        {
            return;
        }
#endif
        const index_t wave      = tid >> 6;
        const index_t lane      = tid & 63;
        const index_t k         = packed_stride - 32 + (lane & 15);
        const index_t m_base    = m_block + wave * 32 + (lane >> 4) * 8;

#pragma unroll
        for(index_t j = 0; j < 8; ++j)
        {
            const index_t m = m_base + j;
            packed_a_ptr[m * packed_stride + k] =
                a_ptr[k * source_stride + m];
        }
    }
#endif

    CK_TILE_DEVICE void Run(const KernelArgs& kargs,
                            const tuple<index_t, index_t>& block_idx_2d,
                            const index_t block_idx_z,
                            const index_t group_id,
                            const index_t group_count,
                            const bool run_fused_tail,
                            const index_t physical_block_id,
                            const index_t physical_grid_size) const
    {

        const auto [iM, iN] = block_idx_2d;

        const index_t i_m = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        //考虑k_batch及block gemm的k切分向上取整后的k维度，如果k_batch=1，且K正好是KPerBlock的整数倍，则splitted_k=K
        const typename Base::SplitKBatchOffset splitk_batch_offset(kargs, block_idx_z);

        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) +
                                 splitk_batch_offset.as_k_split_offset[0];
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) +
                                 splitk_batch_offset.bs_k_split_offset[0];
        CDataType* c_ptr = static_cast<CDataType*>(kargs.e_ptr);

        // allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        if constexpr(is_grouped_gemm_mls_pipeline<GemmPipeline>::value)
        {
            RunGemmMls(a_ptr,
                       b_ptr,
                       c_ptr,
                       smem_ptr_0,
                       kargs,
                       splitk_batch_offset,
                       i_m,
                       i_n);
        }
        else
        {
            static_assert(
                GemmPipeline::DoubleSmemBuffer || !GemmPipeline::Preshuffle,
                "SingleSmemBuffer and Preshuffle cannot both be enabled simultaneously!");

//         // TO DO:
//         // Can we simplify this branching logic?
            if constexpr(GemmPipeline::DoubleSmemBuffer == true)
            {
                __shared__ char smem_ptr_1[GetSmemSize()];
                if constexpr(UsePersistentKernel || GemmPipeline::Preshuffle)
                {
                    RunGemmWithPipelineSelection2LDS(a_ptr,
                                                     b_ptr,
                                                     c_ptr,
                                                     smem_ptr_0,
                                                     smem_ptr_1,
                                                     kargs,
                                                     splitk_batch_offset,
                                                     i_m,
                                                     i_n);
                }
                else
                {

                    Base::RunGemm2LDS({a_ptr},
                                      {b_ptr},
                                      kargs.ds_ptr,
                                      c_ptr,
                                      smem_ptr_0,
                                      smem_ptr_1,
                                      kargs,
                                      splitk_batch_offset,
                                      i_m,
                                      i_n);
                }
            }
            else // SingleSmemBuffer
            {
                if constexpr(UsePersistentKernel)
                {
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DSREADM)
                    ADataType* packed_a_ptr = nullptr;
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_FROM_B_TRANSPOSED_C)
                    if(kargs.k_batch == 1 && kargs.K == kargs.M)
#else
                    if(kargs.k_batch == 1 && kargs.K == kargs.N)
#endif
                    {
                        CDataType* global_c_base =
                            c_ptr - group_id * kargs.M * kargs.N;
                        auto* packed_base = reinterpret_cast<ADataType*>(
                            global_c_base +
                            group_count * kargs.M * kargs.N);
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_FROM_B_TRANSPOSED_C)
                        packed_a_ptr =
                            packed_base + group_id * kargs.N * kargs.K;
#else
                        packed_a_ptr =
                            packed_base + group_id * kargs.M * kargs.K;
#endif
                    }
                    RunGemmWithPipelineSelection(a_ptr,
                                                 b_ptr,
                                                 c_ptr,
                                                 smem_ptr_0,
                                                 kargs,
                                                 splitk_batch_offset,
                                                 i_m,
                                                 i_n,
                                                 packed_a_ptr,
                                                 kargs.K,
#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_FROM_B_TRANSPOSED_C)
                                                 iM);
                    if(packed_a_ptr != nullptr && iM == 15)
                    {
                        RepairPackedAPenultimateStage(b_ptr,
                                                      packed_a_ptr,
                                                      kargs.stride_Bs[0],
                                                      kargs.K,
                                                      i_n);
                    }
#else
                                                 iN);
                    if(packed_a_ptr != nullptr && iN == 15)
                    {
                        RepairPackedAPenultimateStage(a_ptr,
                                                      packed_a_ptr,
                                                      kargs.stride_As[0],
                                                      kargs.K,
                                                      i_m);
                    }
#endif
#else
                    RunGemmWithPipelineSelection(
                        a_ptr, b_ptr, c_ptr, smem_ptr_0, kargs, splitk_batch_offset, i_m, i_n);
#endif
                }
                else // Non-persistent kernel
                {
                    Base::RunGemm({a_ptr},
                                  {b_ptr},
                                  kargs.ds_ptr,
                                  c_ptr,
                                  smem_ptr_0,
                                  kargs,
                                  splitk_batch_offset,
                                  i_m,
                                  i_n);
                }
            }
        }

#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX)
        if(run_fused_tail)
        {
            FusedPackGradX(a_ptr,
                           c_ptr,
                           smem_ptr_0,
                           kargs,
                           group_id,
                           group_count,
                           physical_block_id,
                           physical_grid_size);
        }
#else
        (void)group_id;
        (void)group_count;
        (void)run_fused_tail;
        (void)physical_block_id;
        (void)physical_grid_size;
#endif
    }

    CK_TILE_DEVICE static void
    RunGemmMls(const ADataType* a_ptr,
               const BDataType* b_ptr,
               CDataType* c_ptr,
               void* p_smem,
               const KernelArgs& kargs,
               const typename Base::SplitKBatchOffset& splitk_batch_offset,
               const index_t block_idx_m,
               const index_t block_idx_n)
    {
        static_assert(is_grouped_gemm_mls_pipeline<GemmPipeline>::value);

        auto a_desc = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_As[0], 1),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(1, kargs.stride_As[0]),
                    number<1>{},
                    number<1>{});
            }
        }();
        auto b_desc = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(kargs.N, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_Bs[0], 1),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(kargs.N, splitk_batch_offset.splitted_k),
                    make_tuple(1, kargs.stride_Bs[0]),
                    number<1>{},
                    number<1>{});
            }
        }();
        auto a_view = make_hcu_tensor_view<address_space_enum::global>(a_ptr, a_desc);
        auto b_view = make_hcu_tensor_view<address_space_enum::global>(b_ptr, b_desc);

        const auto& gemm_tensor_views_tuple =
            Base::template MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                {a_ptr}, {b_ptr}, kargs.ds_ptr, c_ptr, kargs, splitk_batch_offset);
        const auto& gemm_pad_views = Base::MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows =
            Base::MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);
        const auto& d_block_window = gemm_tile_windows.at(Base::I2);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));
        const index_t k_remainder = __builtin_amdgcn_readfirstlane(
            num_loop * GemmPipeline::KPerBlock - splitk_batch_offset.splitted_k);
        const bool has_hot_loop = GemmPipeline::BlockHasHotloop(num_loop);

        const auto& c_block_tile = GemmPipeline{}(a_view,
                                                  b_view,
                                                  block_idx_m,
                                                  block_idx_n,
                                                  kargs.stride_As[0],
                                                  kargs.stride_Bs[0],
                                                  k_remainder,
                                                  num_loop,
                                                  has_hot_loop,
                                                  p_smem);

        auto& c_block_window = gemm_tile_windows.at(Base::I3);
        EpiloguePipeline{}.template
        operator()<decltype(c_block_window), decltype(c_block_tile), decltype(d_block_window)>(
            c_block_window, c_block_tile, d_block_window, p_smem);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @note The GEMM pipeline is selected in-kernel based on the number of K-loops
     *       and the tail-number. This is needed for the persistent tile-loop when
     *       we didn't have access to the K dimension on the host.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k
     * batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void
    RunGemmWithPipelineSelection(const ADataType* a_ptr,
                                 const BDataType* b_ptr,
                                 CDataType* c_ptr,
                                 void* smem_ptr_0,
                                 const KernelArgs& kargs,
                                 const typename Base::SplitKBatchOffset& splitk_batch_offset,
                                 const index_t block_idx_m,
                                 const index_t block_idx_n,
                                 [[maybe_unused]] ADataType* packed_a_ptr = nullptr,
                                 [[maybe_unused]] const index_t packed_a_stride = 0,
                                 [[maybe_unused]] const index_t packed_a_stage_group = 0)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            Base::template MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                {a_ptr}, {b_ptr}, kargs.ds_ptr, c_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = Base::MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows =
            Base::MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);
        const auto& a_block_window = gemm_tile_windows.at(Base::I0);
        const auto& b_block_window = gemm_tile_windows.at(Base::I1);
        const auto& d_block_window = gemm_tile_windows.at(Base::I2);

        // Get hot-loop and tail configuration
        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));
        [[maybe_unused]] const bool has_hot_loop =
            GemmPipeline::BlockHasHotloop(num_loop);
        [[maybe_unused]] const TailNumber tail_num =
            GemmPipeline::GetBlockLoopTailNum(num_loop);

        // Run GEMM pipeline
        const auto& c_block_tile =
#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE)
            GemmPipeline{}.RunWithEpiloguePreStore(
                a_block_window[Base::I0],
                b_block_window[Base::I0],
                num_loop,
                smem_ptr_0,
                [&](const auto& partial_c_block_tile) {
                    EpiloguePipeline::PreStoreRound0(
                        partial_c_block_tile, smem_ptr_0);
                });
#elif defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_DSREADM)
            [&]() {
                if(packed_a_ptr != nullptr)
                {
                    return GemmPipeline{}.template operator()(
                        a_block_window[Base::I0],
                        b_block_window[Base::I0],
                        num_loop,
                        has_hot_loop,
                        tail_num,
                        smem_ptr_0,
                        packed_a_ptr,
                        packed_a_stride,
                        packed_a_stage_group);
                }
                return GemmPipeline{}.template operator()(
                    a_block_window[Base::I0],
                    b_block_window[Base::I0],
                    num_loop,
                    has_hot_loop,
                    tail_num,
                    smem_ptr_0);
            }();
#else
            GemmPipeline{}.template operator()(a_block_window[Base::I0],
                                               b_block_window[Base::I0],
                                               num_loop,
                                               has_hot_loop,
                                               tail_num,
                                               smem_ptr_0);
#endif
        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(Base::I3);
        EpiloguePipeline{}.template
        operator()<decltype(c_block_window), decltype(c_block_tile), decltype(d_block_window)>(
            c_block_window, c_block_tile, d_block_window, smem_ptr_0);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @note The GEMM pipeline is selected in-kernel based on the number of K-loops
     *       and the tail-number. This is needed for the persistent tile-loop when
     *       we didn't have access to the K dimension on the host.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param smem_ptr_1 The second start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k
     * batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void
    RunGemmWithPipelineSelection2LDS(const ADataType* a_ptr,
                                     const BDataType* b_ptr,
                                     CDataType* c_ptr,
                                     void* __restrict__ smem_ptr_0,
                                     void* __restrict__ smem_ptr_1,
                                     const KernelArgs& kargs,
                                     const typename Base::SplitKBatchOffset& splitk_batch_offset,
                                     const index_t block_idx_m,
                                     const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            Base::template MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                {a_ptr}, {b_ptr}, kargs.ds_ptr, c_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = Base::MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows =
            Base::MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);
        const auto& a_block_window = gemm_tile_windows.at(Base::I0);
        const auto& b_block_window = gemm_tile_windows.at(Base::I1);
        const auto& d_block_window = gemm_tile_windows.at(Base::I2);

        // Get hot-loop and tail configuration
        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));
        const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        // Run GEMM pipeline with compile-time branching
        const auto& c_block_tile = [&]() {
            if constexpr(GemmPipeline::Preshuffle)
            {
                static_assert(false, "RunGemmWithPipelineSelection2LDS Not check!");

                // Preshuffle version - without has_hot_loop parameter
                return GemmPipeline{}.template operator()(a_block_window[Base::I0],
                                                          b_block_window[Base::I0],
                                                          num_loop,
                                                          tail_num,
                                                          smem_ptr_0,
                                                          smem_ptr_1);
            }
            else
            {
                // Regular version - with has_hot_loop parameter
                const bool has_hot_loop = GemmPipeline::BlockHasHotloop(num_loop);
                return GemmPipeline{}.template operator()(a_block_window[Base::I0],
                                                          b_block_window[Base::I0],
                                                          num_loop,
                                                          has_hot_loop,
                                                          tail_num,
                                                          smem_ptr_0,
                                                          smem_ptr_1);
            }
        }();

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(Base::I3);
        EpiloguePipeline{}.template
        operator()<decltype(c_block_window), decltype(c_block_tile), decltype(d_block_window)>(
            c_block_window, c_block_tile, d_block_window, smem_ptr_0);
    }

    CK_TILE_DEVICE index_t FindGroupId(const TransKernelArg* gemm_desc_ptr,
                                       index_t block_id,
                                       index_t group_count) const
    {
        index_t left     = 0;
        index_t right    = group_count;
        index_t group_id = index_t((left + right) >> 1);

        while((!(block_id >= gemm_desc_ptr[group_id].block_start &&
                 block_id < gemm_desc_ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].block_start)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) >> 1);
        }

        return group_id;
    }

    // For non-persistent kernels
    template <bool U = UsePersistentKernel, typename = std::enable_if_t<!U>>
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   index_t group_count) const
    {
        const index_t block_id   = ck_tile::get_block_1d_id();

        // gemm_desc_ptr 参数中存储了所有group的描述信息，及每个group对应的block_start id和block_end id，
        // group从0开始，block id 从0开始依次递增，见MakeKargs()。因此可以通过block id定位到对应的group id
        const auto gemm_desc_ptr = reinterpret_cast<const TransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        const index_t group_id = FindGroupId(gemm_desc_ptr, block_id, group_count);
        const auto& kargs      = gemm_desc_ptr[group_id];

        // Device-prepared grouped descriptors may reserve a uniform block range
        // for an empty group. Skip it before deriving a zero-sized tile grid.
        if(kargs.group_karg.M == 0 || kargs.group_karg.N == 0 || kargs.group_karg.K == 0)
        {
            return;
        }

        if(!ShouldProcessM(kargs.group_karg.M))
        {
            return;
        }

        const auto grid_size_2d = TilePartitioner::GridSize(kargs.group_karg.M, kargs.group_karg.N);

        //通过1为block id序号，计算出二维的M, N tile索引
        const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
            0,
            kargs.group_karg.M,
            kargs.group_karg.N,
            (block_id - kargs.block_start) % grid_size_2d);
        Run(kargs.group_karg,
            block_idx_2d,
            (block_id - kargs.block_start) / grid_size_2d,
            group_id,
            group_count,
            false,
            0,
            1);
    }

    // For persistent kernels
    template <bool U   = UsePersistentKernel,
              typename = std::enable_if_t<U>,
              typename = void> // extra template parameter to avoid redefinition
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   const index_t group_count) const
    {
        const index_t grid_size  = ck_tile::get_grid_size();
        const auto gemm_desc_ptr = reinterpret_cast<const TransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));
        index_t block_id      = ck_tile::get_block_1d_id(); // initial block_id
        index_t cum_grid_size = 0;
        for(index_t group_id = 0; group_id < group_count; ++group_id)
        {
            const auto& kargs      = gemm_desc_ptr[group_id].group_karg;
            const auto& k_batch    = kargs.k_batch;
            const auto block_start = cum_grid_size;
            cum_grid_size += TilePartitioner::GridSize(kargs.M, kargs.N) * k_batch;
            while(block_id < cum_grid_size)
            {
                if(ShouldProcessM(kargs.M))
                {
                    const auto grid_size_2d = TilePartitioner::GridSize(kargs.M, kargs.N);
                    const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
                        0, kargs.M, kargs.N, (block_id - block_start) % grid_size_2d);
                    Run(kargs,
                        block_idx_2d,
                        (block_id - block_start) / grid_size_2d,
                        group_id,
                        group_count,
                        group_id + 1 == group_count &&
                            block_id + grid_size >= cum_grid_size,
                        ck_tile::get_block_1d_id(),
                        grid_size);
                }
                block_id = block_id + grid_size; // advance to next block
                // NOTE: this check is redundant but helps the compiler avoid spilling some VGPR
                if(block_id >= cum_grid_size)
                {
                    break; // exit the loop if all blocks are processed
                }
            }
        }
    }
};

} // namespace ck_tile
