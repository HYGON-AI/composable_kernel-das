// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_bw_family_selected.hpp"
#include "ck_tile/host/device_prop.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

extern "C" int ck_tile_hcu_grouped_gemm_bw_family_fp16_device_args(
    void*, int, std::uint32_t, hipStream_t);
extern "C" int ck_tile_hcu_grouped_gemm_bw_family_fp16_nn_device_args(
    void*, int, std::uint32_t, hipStream_t);
extern "C" int ck_tile_hcu_grouped_gemm_bw_family_bf16_device_args(
    void*, int, std::uint32_t, hipStream_t);
extern "C" int ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_device_args(
    void*, int, std::uint32_t, hipStream_t);

#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_BW_TN)
extern "C" int ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_device_args(
    void*, int, std::uint32_t, hipStream_t);
extern "C" int
ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_logical_k_tail_device_args(
    void*, int, std::uint32_t, hipStream_t);
#endif

extern "C" int ck_tile_hcu_grouped_gemm_registry_fp16_device_args(
    std::uint32_t, std::uint32_t, void*, int, std::uint32_t, hipStream_t);
extern "C" int ck_tile_hcu_grouped_gemm_registry_bf16_device_args(
    std::uint32_t, std::uint32_t, void*, int, std::uint32_t, hipStream_t);

#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED)
extern "C" int
ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_rr_fixed_srd_lds8_boundary_lds0_vmem0_device_args(
    void*, int, std::uint32_t, hipStream_t);
extern "C" int
ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_rr_fast_vmem3_early_final_barrier_ck_device_args(
    void*, int, std::uint32_t, hipStream_t);
#endif

namespace {

constexpr std::uint32_t ArchGfx936 =
    CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1;
constexpr std::uint32_t ArchGfx938 =
    CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1;
constexpr std::uint32_t ArchGfx946 =
    CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX946_V1;
constexpr std::uint32_t DTypeFp16 = CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_FP16_V1;
constexpr std::uint32_t DTypeBf16 = CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1;
constexpr std::uint32_t LayoutNt  = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1;
constexpr std::uint32_t LayoutNn  = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1;
constexpr std::uint32_t LayoutTn  = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_TN_V1;

using InstanceId = ck_tile::GroupedGemmInstanceId;
static_assert(static_cast<std::uint32_t>(InstanceId::none) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_NONE_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::bw_family_selected) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_SELECTED_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::bw_family_bf16_nn_fixed_srd_lds8) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BF16_NN_FIXED_SRD_LDS8_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::bw_family_bf16_nn_fast_vmem3) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BF16_NN_FAST_VMEM3_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::bw_family_blas_transposed) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BLAS_TRANSPOSED_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::bw_family_logical_k_tail) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_LOGICAL_K_TAIL_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::gfx936_v3_256_m_only_padding) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_256_M_ONLY_PADDING_V1);
static_assert(
    static_cast<std::uint32_t>(InstanceId::gfx936_v3_dsreadm_backward_m_only_padding) ==
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_DSREADM_BACKWARD_M_ONLY_PADDING_V1);
static_assert(
    static_cast<std::uint32_t>(InstanceId::gfx936_v3_default_backward_m_only_padding) ==
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_DEFAULT_BACKWARD_M_ONLY_PADDING_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::gfx938_mls_large_256) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX938_MLS_LARGE_256_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::gfx938_mls_small_128) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX938_MLS_SMALL_128_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::v4_128_m_only_padding) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_128_M_ONLY_PADDING_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::v4_128_full_padding) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_128_FULL_PADDING_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::v4_64_nonpadding) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_NONPADDING_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::v4_64_full_padding) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_FULL_PADDING_V1);
static_assert(
    static_cast<std::uint32_t>(
        InstanceId::v6_m32_nonpadding_mle8_family_b_gfx936) ==
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V6_M32_NONPADDING_MLE8_FAMILY_B_GFX936_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::v3_128_m4_full_padding) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V3_128_M4_FULL_PADDING_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::v3_128_m4_full_padding_gfx936) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V3_128_M4_FULL_PADDING_GFX936_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::v4_64_m01_4_gfx936) ==
              CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_M01_4_GFX936_V1);
static_assert(static_cast<std::int32_t>(ck_tile::GroupedGemmCandidateFamily::bw_family) ==
              CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_BW_V1);
static_assert(static_cast<std::int32_t>(ck_tile::GroupedGemmCandidateFamily::gfx936_v3) ==
              CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX936_V3_V1);
static_assert(static_cast<std::int32_t>(ck_tile::GroupedGemmCandidateFamily::gfx938_mls) ==
              CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX938_MLS_V1);
static_assert(static_cast<std::int32_t>(ck_tile::GroupedGemmCandidateFamily::v4) ==
              CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1);

constexpr std::uint32_t KnownProblemFlags =
    CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1 |
    CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1;
constexpr std::uint32_t KnownPolicyFlags =
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BW_FAMILY_V1 |
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_V3_V1 |
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_DSREADM_V1 |
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX938_MLS_V1 |
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_BF16_NN_PROMOTED_V1 |
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BF16_TN_LOGICAL_K_TAIL_V1;

bool all_zero(const std::uint64_t* values, std::size_t count)
{
    for(std::size_t i = 0; i < count; ++i)
    {
        if(values[i] != 0)
            return false;
    }
    return true;
}

template <typename T>
int validate_versioned_header(const T* value)
{
    if(value == nullptr)
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1;
    if(value->struct_size < sizeof(T))
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_STRUCT_SIZE_V1;
    if(value->abi_version != CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION)
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_VERSION_V1;
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
}

template <std::size_t N>
void copy_name(char (&destination)[N], const std::string& source)
{
    const auto count = std::min(source.size(), N - 1);
    std::memcpy(destination, source.data(), count);
    destination[count] = '\0';
}

template <std::size_t N>
void copy_name(char (&destination)[N], std::string_view source)
{
    const auto count = std::min(source.size(), N - 1);
    std::memcpy(destination, source.data(), count);
    destination[count] = '\0';
}

int detect_arch(ck_tile::GroupedGemmArch& arch)
{
    const auto target = ck_tile::get_hcu_target_enum();
    if(target == ck_tile::hcu_target_enum::gfx936)
    {
        arch = ck_tile::GroupedGemmArch::gfx936;
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
    }
    if(target == ck_tile::hcu_target_enum::gfx938)
    {
        arch = ck_tile::GroupedGemmArch::gfx938;
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
    }
    if(target == ck_tile::hcu_target_enum::gfx946)
    {
        arch = ck_tile::GroupedGemmArch::gfx946;
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
    }
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_ARCH_V1;
}

int convert_dimension(const ck_tile_hcu_grouped_gemm_dimension_v1& source,
                      ck_tile::GroupedGemmDimensionSummary& destination)
{
    if(source.reserved != 0)
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1;
    if(source.kind == CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1)
    {
        destination = ck_tile::GroupedGemmDimensionSummary::Common(source.extent);
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
    }
    if(source.kind ==
       CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1)
    {
        destination =
            ck_tile::GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(source.extent);
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
    }
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1;
}

int convert_problem(const ck_tile_hcu_grouped_gemm_problem_v1* source,
                    ck_tile::GroupedGemmProblem& problem,
                    ck_tile::GroupedGemmSelectorPolicy& policy)
{
    int status = validate_versioned_header(source);
    if(status != 0)
        return status;
    if(source->reserved0 != 0 || !all_zero(source->reserved, 4) ||
       (source->problem_flags & ~KnownProblemFlags) != 0 ||
       (source->policy_disable_flags & ~KnownPolicyFlags) != 0)
    {
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1;
    }

    status = detect_arch(problem.arch);
    if(status != 0)
        return status;

    if(source->data_type == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1)
        problem.data_type = ck_tile::GroupedGemmDataType::fp16;
    else if(source->data_type == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1)
        problem.data_type = ck_tile::GroupedGemmDataType::bf16;
    else
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;

    if(source->layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1)
        problem.layout = ck_tile::GroupedGemmLayout::nt;
    else if(source->layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1)
        problem.layout = ck_tile::GroupedGemmLayout::nn;
    else if(source->layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1)
        problem.layout = ck_tile::GroupedGemmLayout::tn;
    else
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;

    problem.group_count = source->group_count;
    problem.may_have_empty_groups =
        (source->problem_flags &
         CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1) != 0;
    problem.group_lengths_are_ragged =
        (source->problem_flags &
         CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1) != 0;
    if((status = convert_dimension(source->m, problem.m)) != 0 ||
       (status = convert_dimension(source->n, problem.n)) != 0 ||
       (status = convert_dimension(source->k, problem.k)) != 0)
    {
        return status;
    }

    const auto disabled = source->policy_disable_flags;
    policy.enable_bw_family =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BW_FAMILY_V1) == 0;
    policy.enable_gfx936_v3 =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_V3_V1) == 0;
    policy.enable_gfx936_dsreadm =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_DSREADM_V1) == 0;
    policy.enable_gfx938_mls =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX938_MLS_V1) == 0;
    policy.enable_gfx936_bf16_nn_promoted =
        (disabled &
         CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_BF16_NN_PROMOTED_V1) == 0;
    policy.enable_bf16_tn_logical_k_tail =
        (disabled &
         CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BF16_TN_LOGICAL_K_TAIL_V1) == 0;

    // CK-owned rollback switches. The selector itself is a pure function of
    // Problem+Policy; env handling stays in this adapter so same-process A/B
    // does not require the consumer to translate those names into flags.
    if(!ck_tile_hcu_grouped_gemm_bf16_nn_fixed_srd_lds8_enabled())
        policy.enable_gfx936_bf16_nn_promoted = false;
    if(!ck_tile_hcu_grouped_gemm_bf16_tn_logical_k_tail_enabled())
        policy.enable_bf16_tn_logical_k_tail = false;

#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED)
    // Do not name InstanceId 1001/1002 when this TU cannot dispatch them.
    policy.enable_gfx936_bf16_nn_promoted = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX938_DISABLE)
    policy.enable_p1_family_a_v3_128_gfx938 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX936_DISABLE)
    policy.enable_p1_family_a_v3_128_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_B_V6_M32_MLE8_GFX936_DISABLE)
    policy.enable_p1_family_b_v6_m32_mle8_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_V4_64_M01_4_GFX936_DISABLE)
    policy.enable_p2_bf16_g8_v4_64_m01_4_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX936_FC2_NN_FAST_VMEM3_DISABLE)
    policy.enable_p2_bf16_g8_gfx936_fc2_nn_fast_vmem3 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX938_V5_FC1_ROUTE_DISABLE)
    policy.enable_p2_bf16_g8_gfx938_v5_fc1_route = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P3_FP16_TN_M1024_K4096_ROUTE_DISABLE)
    policy.enable_p3_fp16_tn_m1024_k4096_route = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
    // These architecture-specialized code objects are retained for non-RDC
    // builds, but real gfx936/gfx938 validation shows they are not correct
    // after device linking. Keep the selected InstanceId honest by applying
    // a build-capability policy before selection instead of dispatching a
    // different kernel behind the specialized ID.
    policy.enable_bw_selected      = false;
    policy.enable_gfx938_mls_large = false;
#endif

    return ck_tile::is_valid_grouped_gemm_problem(problem)
               ? CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1
               : CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
}

int write_selection(const ck_tile::GroupedGemmProblem& problem,
                    const ck_tile::GroupedGemmSelection& selected,
                    ck_tile_hcu_grouped_gemm_selection_v1* output)
{
    int status = validate_versioned_header(output);
    if(status != 0)
        return status;
    const std::uint32_t output_size = output->struct_size;
    *output                         = {};
    output->struct_size             = output_size;
    output->abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    output->instance_id = static_cast<std::uint32_t>(selected.instance_id);
    output->reason      = static_cast<std::int32_t>(selected.reason);
    copy_name(output->instance_name,
              ck_tile::stable_grouped_gemm_instance_id(problem, selected));
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
}


int dispatch_registry_candidate(const ck_tile::GroupedGemmProblem& problem,
                                ck_tile::GroupedGemmInstanceId instance_id,
                                const ck_tile_hcu_grouped_gemm_launch_v1& launch)
{
    const auto stream = reinterpret_cast<hipStream_t>(launch.stream);
    void* device_args = const_cast<void*>(launch.device_args);
    const int groups  = problem.group_count;
    const auto dtype  = problem.data_type;
    const auto layout = problem.layout;

    switch(instance_id)
    {
    case ck_tile::GroupedGemmInstanceId::none: break;
    case ck_tile::GroupedGemmInstanceId::bw_family_selected:
        if(dtype == ck_tile::GroupedGemmDataType::fp16)
        {
            if(layout == ck_tile::GroupedGemmLayout::nt)
                return ck_tile_hcu_grouped_gemm_bw_family_fp16_device_args(
                    device_args, groups, launch.num_cu, stream);
            if(layout == ck_tile::GroupedGemmLayout::nn)
                return ck_tile_hcu_grouped_gemm_bw_family_fp16_nn_device_args(
                    device_args, groups, launch.num_cu, stream);
        }
        if(dtype == ck_tile::GroupedGemmDataType::bf16)
        {
            if(layout == ck_tile::GroupedGemmLayout::nt)
                return ck_tile_hcu_grouped_gemm_bw_family_bf16_device_args(
                    device_args, groups, launch.num_cu, stream);
            if(layout == ck_tile::GroupedGemmLayout::nn)
                return ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_device_args(
                    device_args, groups, launch.num_cu, stream);
        }
        break;
    case ck_tile::GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8:
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED)
        if(dtype == ck_tile::GroupedGemmDataType::bf16 &&
           layout == ck_tile::GroupedGemmLayout::nn)
            return ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_rr_fixed_srd_lds8_boundary_lds0_vmem0_device_args(
                device_args, groups, launch.num_cu, stream);
#endif
        break;
    case ck_tile::GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3:
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED)
        if(dtype == ck_tile::GroupedGemmDataType::bf16 &&
           layout == ck_tile::GroupedGemmLayout::nn)
            return ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_rr_fast_vmem3_early_final_barrier_ck_device_args(
                device_args, groups, launch.num_cu, stream);
#endif
        break;
    case ck_tile::GroupedGemmInstanceId::bw_family_blas_transposed:
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_BW_TN)
        if(dtype == ck_tile::GroupedGemmDataType::bf16 &&
           layout == ck_tile::GroupedGemmLayout::tn)
            return ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_device_args(
                device_args, groups, launch.num_cu, stream);
#endif
        break;
    case ck_tile::GroupedGemmInstanceId::bw_family_logical_k_tail:
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_BW_TN)
        if(dtype == ck_tile::GroupedGemmDataType::bf16 &&
           layout == ck_tile::GroupedGemmLayout::tn)
            return ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_logical_k_tail_device_args(
                device_args, groups, launch.num_cu, stream);
#endif
        break;
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_256_m_only_padding:
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding:
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding:
    case ck_tile::GroupedGemmInstanceId::gfx938_mls_large_256:
    case ck_tile::GroupedGemmInstanceId::gfx938_mls_small_128:
    case ck_tile::GroupedGemmInstanceId::v4_128_m_only_padding:
    case ck_tile::GroupedGemmInstanceId::v4_128_full_padding:
    case ck_tile::GroupedGemmInstanceId::v4_64_nonpadding:
    case ck_tile::GroupedGemmInstanceId::v4_64_full_padding:
    case ck_tile::GroupedGemmInstanceId::v6_m32_nonpadding_mle8_family_b_gfx936:
    case ck_tile::GroupedGemmInstanceId::v3_128_m4_full_padding:
    case ck_tile::GroupedGemmInstanceId::v3_128_m4_full_padding_gfx936:
    case ck_tile::GroupedGemmInstanceId::v4_64_m01_4_gfx936:
        if(dtype == ck_tile::GroupedGemmDataType::fp16)
            return ck_tile_hcu_grouped_gemm_registry_fp16_device_args(
                static_cast<std::uint32_t>(instance_id),
                static_cast<std::uint32_t>(layout),
                device_args,
                groups,
                launch.num_cu,
                stream);
        if(dtype == ck_tile::GroupedGemmDataType::bf16)
            return ck_tile_hcu_grouped_gemm_registry_bf16_device_args(
                static_cast<std::uint32_t>(instance_id),
                static_cast<std::uint32_t>(layout),
                device_args,
                groups,
                launch.num_cu,
                stream);
        break;
    }
    return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
}

void candidate_masks(ck_tile::GroupedGemmInstanceId id,
                     std::uint32_t& architecture_mask,
                     std::uint32_t& data_type_mask,
                     std::uint32_t& layout_mask,
                     std::uint32_t& contract_flags)
{
    architecture_mask = ArchGfx936 | ArchGfx938 | ArchGfx946;
    data_type_mask     = DTypeFp16 | DTypeBf16;
    layout_mask        = LayoutNt | LayoutNn | LayoutTn;
    contract_flags     = 0;
    switch(id)
    {
    case ck_tile::GroupedGemmInstanceId::none: break;
    case ck_tile::GroupedGemmInstanceId::bw_family_selected:
        architecture_mask = ArchGfx936;
        layout_mask        = LayoutNt | LayoutNn;
        break;
    case ck_tile::GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8:
    case ck_tile::GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3:
        architecture_mask = ArchGfx936;
        data_type_mask     = DTypeBf16;
        layout_mask        = LayoutNn;
        break;
    case ck_tile::GroupedGemmInstanceId::bw_family_blas_transposed:
    case ck_tile::GroupedGemmInstanceId::bw_family_logical_k_tail:
        architecture_mask = ArchGfx936 | ArchGfx938;
        data_type_mask = DTypeBf16;
        layout_mask    = LayoutTn;
        contract_flags = CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_CONTRACT_C_TRANSPOSED_VIEW_V1;
        break;
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_256_m_only_padding:
        architecture_mask = ArchGfx936;
        layout_mask        = LayoutNt;
        break;
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding:
        architecture_mask = ArchGfx936;
        layout_mask        = LayoutNn | LayoutTn;
        break;
    case ck_tile::GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding:
        architecture_mask = ArchGfx936;
        layout_mask        = LayoutNn;
        break;
    case ck_tile::GroupedGemmInstanceId::gfx938_mls_large_256:
        architecture_mask = ArchGfx938;
        break;
    case ck_tile::GroupedGemmInstanceId::gfx938_mls_small_128:
        architecture_mask = ArchGfx938;
        layout_mask        = LayoutNt;
        break;
    case ck_tile::GroupedGemmInstanceId::v4_128_m_only_padding:
    case ck_tile::GroupedGemmInstanceId::v4_64_nonpadding:
    case ck_tile::GroupedGemmInstanceId::v4_64_full_padding: break;
    case ck_tile::GroupedGemmInstanceId::v3_128_m4_full_padding:
        architecture_mask = ArchGfx938;
        data_type_mask     = DTypeFp16;
        layout_mask        = LayoutNn;
        break;
    case ck_tile::GroupedGemmInstanceId::v3_128_m4_full_padding_gfx936:
        architecture_mask = ArchGfx936;
        data_type_mask     = DTypeFp16;
        layout_mask        = LayoutNn;
        break;
    case ck_tile::GroupedGemmInstanceId::v4_64_m01_4_gfx936:
        architecture_mask = ArchGfx936;
        data_type_mask     = DTypeBf16;
        layout_mask        = LayoutNt | LayoutNn;
        break;
    case ck_tile::GroupedGemmInstanceId::v4_128_full_padding:
        data_type_mask = DTypeBf16;
        break;
    case ck_tile::GroupedGemmInstanceId::v6_m32_nonpadding_mle8_family_b_gfx936:
        architecture_mask = ArchGfx936;
        data_type_mask     = DTypeFp16;
        layout_mask        = LayoutNn;
        break;
    }
}

} // namespace

extern "C" int ck_tile_hcu_grouped_gemm_select_device_args_v1(
    const ck_tile_hcu_grouped_gemm_problem_v1* source,
    ck_tile_hcu_grouped_gemm_selection_v1* output)
{
    try
    {
        ck_tile::GroupedGemmProblem problem;
        ck_tile::GroupedGemmSelectorPolicy policy;
        int status = convert_problem(source, problem, policy);
        if(status != 0)
            return status;
        const auto selected = ck_tile::select_grouped_gemm_candidate(problem, policy);
        if(!selected.supported())
            return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
        return write_selection(problem, selected, output);
    }
    catch(...)
    {
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INTERNAL_ERROR_V1;
    }
}

extern "C" int ck_tile_hcu_grouped_gemm_run_device_args_v1(
    const ck_tile_hcu_grouped_gemm_problem_v1* source,
    const ck_tile_hcu_grouped_gemm_launch_v1* launch,
    ck_tile_hcu_grouped_gemm_selection_v1* output)
{
    try
    {
        int status = validate_versioned_header(launch);
        if(status != 0)
            return status;
        if(launch->device_args == nullptr || !all_zero(launch->reserved, 4))
            return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1;

        ck_tile::GroupedGemmProblem problem;
        ck_tile::GroupedGemmSelectorPolicy policy;
        status = convert_problem(source, problem, policy);
        if(status != 0)
            return status;
        const auto selected = ck_tile::select_grouped_gemm_candidate(problem, policy);
        if(!selected.supported())
            return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
        if(launch->expected_instance_id != 0 &&
           launch->expected_instance_id != static_cast<std::uint32_t>(selected.instance_id))
        {
            return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INSTANCE_MISMATCH_V1;
        }
        if(output != nullptr)
        {
            status = write_selection(problem, selected, output);
            if(status != 0)
                return status;
        }
        return dispatch_registry_candidate(problem, selected.instance_id, *launch);
    }
    catch(...)
    {
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INTERNAL_ERROR_V1;
    }
}

extern "C" const char* ck_tile_hcu_grouped_gemm_instance_name_v1(
    std::uint32_t instance_id)
{
    const auto* candidate = ck_tile::find_grouped_gemm_candidate(
        static_cast<ck_tile::GroupedGemmInstanceId>(instance_id));
    return candidate == nullptr ? nullptr : candidate->canonical_suffix.data();
}

extern "C" int ck_tile_hcu_grouped_gemm_query_instance_v1(
    std::uint32_t instance_id,
    ck_tile_hcu_grouped_gemm_candidate_info_v1* output)
{
    try
    {
        int status = validate_versioned_header(output);
        if(status != 0)
            return status;
        const auto id = static_cast<ck_tile::GroupedGemmInstanceId>(instance_id);
        const auto* candidate = ck_tile::find_grouped_gemm_candidate(id);
        if(candidate == nullptr)
            return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;

        std::uint32_t architecture_mask = 0;
        std::uint32_t data_type_mask     = 0;
        std::uint32_t layout_mask        = 0;
        std::uint32_t contract_flags     = 0;
        candidate_masks(
            id, architecture_mask, data_type_mask, layout_mask, contract_flags);
        const std::uint32_t output_size = output->struct_size;
        *output                         = {};
        output->struct_size             = output_size;
        output->abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
        output->instance_id = instance_id;
        output->family      = static_cast<std::int32_t>(candidate->family);
        output->architecture_mask = architecture_mask;
        output->data_type_mask     = data_type_mask;
        output->layout_mask        = layout_mask;
        output->contract_flags     = contract_flags;
        copy_name(output->canonical_name, candidate->canonical_suffix);
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
    }
    catch(...)
    {
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INTERNAL_ERROR_V1;
    }
}
