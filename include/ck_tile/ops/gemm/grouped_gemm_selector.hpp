// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace ck_tile {

enum class GroupedGemmArch
{
    unsupported,
    gfx936,
    gfx938,
    gfx946,
};

enum class GroupedGemmDataType
{
    unsupported,
    fp16,
    bf16,
};

enum class GroupedGemmLayout
{
    unsupported,
    nt,
    nn,
    tn,
};

// A dimension is either common to every group or represented by device-side
// per-group lengths backed by one physical allocation. The selector may use
// the capacity as a conservative host-visible summary, but must not treat it
// as an individual group's logical length.
enum class GroupedGemmDimensionKind
{
    common,
    device_lengths_with_capacity,
};

struct GroupedGemmDimensionSummary
{
    GroupedGemmDimensionKind kind = GroupedGemmDimensionKind::common;
    std::int64_t extent           = 0;

    static constexpr GroupedGemmDimensionSummary Common(std::int64_t value)
    {
        return {GroupedGemmDimensionKind::common, value};
    }

    static constexpr GroupedGemmDimensionSummary DeviceLengthsWithCapacity(
        std::int64_t capacity)
    {
        return {GroupedGemmDimensionKind::device_lengths_with_capacity, capacity};
    }
};

struct GroupedGemmProblem
{
    GroupedGemmArch arch           = GroupedGemmArch::unsupported;
    GroupedGemmDataType data_type  = GroupedGemmDataType::unsupported;
    GroupedGemmLayout layout       = GroupedGemmLayout::unsupported;
    std::int32_t group_count       = 0;
    GroupedGemmDimensionSummary m  = {};
    GroupedGemmDimensionSummary n  = {};
    GroupedGemmDimensionSummary k  = {};
    bool may_have_empty_groups     = false;
    bool group_lengths_are_ragged  = false;
};

// Policy is an explicit input so selection is deterministic and testable.
// Environment-variable handling and hardware detection belong to API adapters.
struct GroupedGemmSelectorPolicy
{
    bool enable_bw_family                 = true;
    bool enable_gfx936_v3                 = true;
    bool enable_gfx936_dsreadm            = true;
    bool enable_gfx938_mls                = true;
    bool enable_gfx936_bf16_nn_promoted   = true;
    bool enable_bf16_tn_logical_k_tail    = true;
    bool enable_bw_selected               = true;
    bool enable_gfx938_mls_large          = true;
    // P1 production defaults after architecture-local correctness and full G392 replay.
    bool enable_p1_family_a_v3_128_gfx938 = true;
    bool enable_p1_family_a_v3_128_gfx936 = true;
    bool enable_p1_family_b_v6_m32_mle8_gfx936 = true;
    // Production default after independent gfx936 focused/full316 validation. Callers can
    // still set this field false to roll the narrow fixed-M route back to V4 64x128 M01=1.
    bool enable_p2_bf16_g8_v4_64_m01_4_gfx936 = true;
    // Production default after independent gfx936 correctness/focused validation. Callers can
    // still set this field false to roll the exact FC2 NN route back to bw_family_selected.
    bool enable_p2_bf16_g8_gfx936_fc2_nn_fast_vmem3 = true;
    // Production default after dual-session focused/full316 validation. Callers can
    // still set this field false to roll the narrow gfx938 FC1 route back to MLS.
    bool enable_p2_bf16_g8_gfx938_v5_fc1_route = true;
    // Production default after independent gfx936/gfx938 correctness and focused fallback
    // comparisons. Callers can still set this field false for an explicit V4 rollback.
    bool enable_p3_fp16_tn_m1024_k4096_route = true;
};

enum class GroupedGemmInstanceId
{
    // Values are stable identity codes. Add new instances to a family range;
    // never renumber an existing entry.
    none                                            = 0,
    bw_family_selected                              = 1000,
    bw_family_bf16_nn_fixed_srd_lds8               = 1001,
    bw_family_bf16_nn_fast_vmem3                    = 1002,
    bw_family_blas_transposed                       = 1003,
    bw_family_logical_k_tail                        = 1004,
    gfx936_v3_256_m_only_padding                    = 2000,
    gfx936_v3_dsreadm_backward_m_only_padding       = 2001,
    gfx936_v3_default_backward_m_only_padding       = 2002,
    gfx938_mls_large_256                            = 3000,
    gfx938_mls_small_128                            = 3001,
    v4_128_m_only_padding                          = 4000,
    v4_128_full_padding                            = 4001,
    v4_64_nonpadding                               = 4002,
    v4_64_full_padding                             = 4003,
    v3_128_m4_full_padding                          = 4009,
    v3_128_m4_full_padding_gfx936                   = 4010,
    v4_64_m01_4_gfx936                              = 4011,
    v6_m32_nonpadding_mle8_family_b_gfx936          = 4104,
};

enum class GroupedGemmCandidateFamily
{
    bw_family,
    gfx936_v3,
    gfx938_mls,
    v4,
};

struct GroupedGemmCandidateIdentity
{
    GroupedGemmInstanceId id;
    std::string_view canonical_suffix;
    GroupedGemmCandidateFamily family;
};

inline constexpr std::array<GroupedGemmCandidateIdentity, 18>
    grouped_gemm_candidate_registry{{
        {GroupedGemmInstanceId::bw_family_selected,
         "bw_family.selected",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8,
         "bw_family.rr_fixed_srd_lds8_m1920_2304_n7168_k4096",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3,
         "bw_family.rr_fast_vmem3_m1920_2304_n2048_k7168",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::bw_family_blas_transposed,
         "bw_family.blas_transposed",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::bw_family_logical_k_tail,
         "bw_family.logical_k_tail",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::gfx936_v3_256_m_only_padding,
         "v3.256x256x64.m_only_padding",
         GroupedGemmCandidateFamily::gfx936_v3},
        {GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding,
         "v3.dsreadm_stage_backward_hybrid.m_only_padding",
         GroupedGemmCandidateFamily::gfx936_v3},
        {GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding,
         "v3.default_backward_hybrid.m_only_padding",
         GroupedGemmCandidateFamily::gfx936_v3},
        {GroupedGemmInstanceId::gfx938_mls_large_256,
         "mls.large.256x256x32",
         GroupedGemmCandidateFamily::gfx938_mls},
        {GroupedGemmInstanceId::gfx938_mls_small_128,
         "mls.small.128x128x32",
         GroupedGemmCandidateFamily::gfx938_mls},
        {GroupedGemmInstanceId::v4_128_m_only_padding,
         "v4.128x128x64.m_only_padding",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v4_128_full_padding,
         "v4.128x128x64.full_padding",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v4_64_nonpadding,
         "v4.64x128x64.nonpadding",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v4_64_full_padding,
         "v4.64x128x64.full_padding",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v3_128_m4_full_padding,
         "v3.128x128x64.m4.full_padding.single_lds",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v3_128_m4_full_padding_gfx936,
         "v3.128x128x64.m4.full_padding.single_lds.gfx936",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v4_64_m01_4_gfx936,
         "v4.64x128x64.nonpadding.m01_4.gfx936",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v6_m32_nonpadding_mle8_family_b_gfx936,
         "v6.32x128x64.nonpadding.selector_m_le_8.family_b.gfx936",
         GroupedGemmCandidateFamily::v4},
    }};

constexpr const GroupedGemmCandidateIdentity*
find_grouped_gemm_candidate(GroupedGemmInstanceId id)
{
    for(const auto& candidate : grouped_gemm_candidate_registry)
    {
        if(candidate.id == id)
        {
            return &candidate;
        }
    }
    return nullptr;
}

enum class GroupedGemmSelectionReason
{
    invalid_contract,
    bw_family,
    gfx936_v3,
    gfx938_mls,
    v4_aligned,
    v4_bf16_padding,
    p1_family_a_v3_128_gfx938,
    p1_family_a_v3_128_gfx936,
    p1_family_b_v6_m32_mle8_gfx936,
    p2_bf16_g8_v4_64_m01_4_gfx936,
    p2_bf16_g8_gfx936_fc2_nn_fast_vmem3,
    p2_bf16_g8_gfx938_v5_fc1_route,
    p3_fp16_tn_m1024_k4096_route,
    v4_n_aligned,
    v4_fallback,
};

struct GroupedGemmSelection
{
    GroupedGemmInstanceId instance_id = GroupedGemmInstanceId::none;
    GroupedGemmSelectionReason reason  = GroupedGemmSelectionReason::invalid_contract;

    constexpr bool supported() const { return instance_id != GroupedGemmInstanceId::none; }
};

constexpr bool is_valid_grouped_gemm_problem(const GroupedGemmProblem& problem)
{
    if((problem.arch != GroupedGemmArch::gfx936 && problem.arch != GroupedGemmArch::gfx938 &&
        problem.arch != GroupedGemmArch::gfx946) ||
       (problem.data_type != GroupedGemmDataType::fp16 &&
        problem.data_type != GroupedGemmDataType::bf16) ||
       (problem.layout != GroupedGemmLayout::nt && problem.layout != GroupedGemmLayout::nn &&
        problem.layout != GroupedGemmLayout::tn) ||
       problem.group_count <= 0 || problem.m.extent <= 0 || problem.n.extent <= 0 ||
       problem.k.extent <= 0 || problem.n.kind != GroupedGemmDimensionKind::common)
    {
        return false;
    }

    const int variable_dimensions =
        (problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity ? 1 : 0) +
        (problem.k.kind == GroupedGemmDimensionKind::device_lengths_with_capacity ? 1 : 0);
    return variable_dimensions <= 1;
}

constexpr std::int64_t grouped_gemm_selector_m_extent(const GroupedGemmProblem& problem)
{
    return problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity
               ? problem.m.extent / problem.group_count
               : problem.m.extent;
}

constexpr std::int64_t grouped_gemm_selector_k_extent(const GroupedGemmProblem& problem)
{
    return problem.k.kind == GroupedGemmDimensionKind::device_lengths_with_capacity
               ? problem.k.extent / problem.group_count
               : problem.k.extent;
}

constexpr GroupedGemmSelection select_grouped_gemm_candidate(
    const GroupedGemmProblem& problem,
    const GroupedGemmSelectorPolicy& policy = {})
{
    if(!is_valid_grouped_gemm_problem(problem))
    {
        return {};
    }

    const auto effective_m = grouped_gemm_selector_m_extent(problem);
    const auto effective_k = grouped_gemm_selector_k_extent(problem);
    const auto m           = problem.m.extent;
    const auto n           = problem.n.extent;
    const auto k           = problem.k.extent;
    const bool is_nt       = problem.layout == GroupedGemmLayout::nt;
    const bool is_nn       = problem.layout == GroupedGemmLayout::nn;
    const bool is_tn       = problem.layout == GroupedGemmLayout::tn;
    const bool is_bf16     = problem.data_type == GroupedGemmDataType::bf16;
    const bool is_bf16_tn  = is_bf16 && is_tn;
    const bool has_variable_k =
        problem.k.kind == GroupedGemmDimensionKind::device_lengths_with_capacity;
    const bool has_integral_device_m_capacity =
        problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity &&
        problem.m.extent % problem.group_count == 0;
    const bool logical_k_tail =
        is_bf16_tn && has_variable_k && policy.enable_bf16_tn_logical_k_tail;
    const std::int64_t large_tile_m_threshold =
        problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity ? 1920 : 2048;

    if(policy.enable_p3_fp16_tn_m1024_k4096_route && !is_bf16 && is_tn &&
       problem.m.kind == GroupedGemmDimensionKind::common && m == 1024 && n == 4096 &&
       has_variable_k && k % problem.group_count == 0 && effective_k == 4096 &&
       !problem.may_have_empty_groups && !problem.group_lengths_are_ragged)
    {
        if(problem.arch == GroupedGemmArch::gfx936)
        {
            return {GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding,
                    GroupedGemmSelectionReason::p3_fp16_tn_m1024_k4096_route};
        }
        if(problem.arch == GroupedGemmArch::gfx938)
        {
            return {GroupedGemmInstanceId::gfx938_mls_large_256,
                    GroupedGemmSelectionReason::p3_fp16_tn_m1024_k4096_route};
        }
    }

    const bool bw_arch_layout =
        problem.arch == GroupedGemmArch::gfx936 ||
        (problem.arch == GroupedGemmArch::gfx938 && is_bf16_tn);
    const bool selected_layout = is_nt || is_nn || is_bf16_tn;
    if(policy.enable_bw_family && bw_arch_layout && selected_layout &&
       effective_m >= large_tile_m_threshold && n >= 2048 && k >= 2048 && n % 256 == 0 &&
       (k % 64 == 0 || logical_k_tail))
    {
        if(problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
           policy.enable_gfx936_bf16_nn_promoted && problem.group_count == 16 &&
           problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity &&
           effective_m >= 1920 && effective_m <= 4096)
        {
            if(n == 7168 && k == 4096)
            {
                return {GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8,
                        GroupedGemmSelectionReason::bw_family};
            }
            if(n == 2048 && k == 7168)
            {
                return {GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3,
                        GroupedGemmSelectionReason::bw_family};
            }
        }

        if(policy.enable_p2_bf16_g8_gfx936_fc2_nn_fast_vmem3 &&
           problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
           problem.group_count == 8 &&
           (problem.m.kind == GroupedGemmDimensionKind::common ||
            has_integral_device_m_capacity) &&
           effective_m >= 2048 && effective_m <= 2221 && n == 2048 && k == 7168)
        {
            // ID1002 consumes the device-built per-group descriptors, as does the
            // existing G16 route above. Primus reports the G8 M summary as total
            // packed capacity, so use its integral per-group extent here; the
            // ragged/empty flags remain valid and do not imply a common length.
            return {GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3,
                    GroupedGemmSelectionReason::p2_bf16_g8_gfx936_fc2_nn_fast_vmem3};
        }

        if(is_bf16_tn)
        {
            return {logical_k_tail ? GroupedGemmInstanceId::bw_family_logical_k_tail
                                   : GroupedGemmInstanceId::bw_family_blas_transposed,
                    GroupedGemmSelectionReason::bw_family};
        }
        if(policy.enable_bw_selected)
        {
            return {GroupedGemmInstanceId::bw_family_selected,
                    GroupedGemmSelectionReason::bw_family};
        }
    }

    if(policy.enable_gfx936_v3 && problem.arch == GroupedGemmArch::gfx936 &&
       effective_m >= large_tile_m_threshold && n >= 2048 && k >= 2048 && n % 256 == 0 &&
       k % 64 == 0)
    {
        if(is_nt)
        {
            return {GroupedGemmInstanceId::gfx936_v3_256_m_only_padding,
                    GroupedGemmSelectionReason::gfx936_v3};
        }
        if((is_nn || is_tn) && policy.enable_gfx936_dsreadm)
        {
            return {GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding,
                    GroupedGemmSelectionReason::gfx936_v3};
        }
        if(is_nn)
        {
            return {GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding,
                    GroupedGemmSelectionReason::gfx936_v3};
        }
    }

    if(policy.enable_p2_bf16_g8_gfx938_v5_fc1_route && is_bf16 && is_nt &&
       problem.arch == GroupedGemmArch::gfx938 && problem.group_count == 8 &&
       problem.m.kind == GroupedGemmDimensionKind::common && m >= 1918 && m <= 1921 &&
       n == 4096 && k == 7168)
    {
        return {m % 128 == 0 ? GroupedGemmInstanceId::v4_128_m_only_padding
                            : GroupedGemmInstanceId::v4_64_nonpadding,
                GroupedGemmSelectionReason::p2_bf16_g8_gfx938_v5_fc1_route};
    }

    if(policy.enable_gfx938_mls && problem.arch == GroupedGemmArch::gfx938 &&
       effective_m >= 512 && n >= 512 && k >= 512)
    {
        if(effective_m >= large_tile_m_threshold && n >= 2048 &&
           policy.enable_gfx938_mls_large)
        {
            return {GroupedGemmInstanceId::gfx938_mls_large_256,
                    GroupedGemmSelectionReason::gfx938_mls};
        }
        if(is_nt)
        {
            return {GroupedGemmInstanceId::gfx938_mls_small_128,
                    GroupedGemmSelectionReason::gfx938_mls};
        }
    }

    if(effective_m >= 512 && n >= 512 && k >= 512 && m % 128 == 0 && n % 128 == 0 &&
       k % 64 == 0)
    {
        return {GroupedGemmInstanceId::v4_128_m_only_padding,
                GroupedGemmSelectionReason::v4_aligned};
    }
    if(is_bf16 && m >= 512 && n >= 512 && k >= 512 && m % 128 == 0 && n % 128 == 0)
    {
        return {GroupedGemmInstanceId::v4_128_full_padding,
                GroupedGemmSelectionReason::v4_bf16_padding};
    }
    if(policy.enable_p2_bf16_g8_v4_64_m01_4_gfx936 && is_bf16 &&
       problem.group_count == 8 && problem.m.kind == GroupedGemmDimensionKind::common &&
       m >= 1918 && m <= 1921 && m % 128 != 0 && n >= 2048 && k >= 2048 &&
       n % 128 == 0 && k % 64 == 0 && problem.arch == GroupedGemmArch::gfx936 &&
       (is_nt || is_nn))
    {
        return {GroupedGemmInstanceId::v4_64_m01_4_gfx936,
                GroupedGemmSelectionReason::p2_bf16_g8_v4_64_m01_4_gfx936};
    }

    // Earlier P1 V4 M4/K32/M-threshold/128x64/grid2 experiments did not beat
    // these architecture-local V3 single-LDS routes and were retired.
    if(policy.enable_p1_family_a_v3_128_gfx938 &&
       problem.arch == GroupedGemmArch::gfx938 && !is_bf16 && is_nn && effective_m >= 33 &&
       n >= 1024 && k >= 1024 && n % 128 != 0 && k % 64 == 0)
    {
        return {GroupedGemmInstanceId::v3_128_m4_full_padding,
                GroupedGemmSelectionReason::p1_family_a_v3_128_gfx938};
    }
    if(policy.enable_p1_family_a_v3_128_gfx936 &&
       problem.arch == GroupedGemmArch::gfx936 && !is_bf16 && is_nn && effective_m >= 33 &&
       n >= 1024 && k >= 1024 && n % 128 != 0 && k % 64 == 0)
    {
        return {GroupedGemmInstanceId::v3_128_m4_full_padding_gfx936,
                GroupedGemmSelectionReason::p1_family_a_v3_128_gfx936};
    }
    // The wider M<=10 Family-B experiment had regressions; retain the proven M<=8 gate.
    if(policy.enable_p1_family_b_v6_m32_mle8_gfx936 &&
       problem.arch == GroupedGemmArch::gfx936 && !is_bf16 && is_nn &&
       effective_m <= 8 && n == 3072 && k == 1232)
    {
        return {GroupedGemmInstanceId::v6_m32_nonpadding_mle8_family_b_gfx936,
                GroupedGemmSelectionReason::p1_family_b_v6_m32_mle8_gfx936};
    }
    if(n % 128 == 0)
    {
        return {GroupedGemmInstanceId::v4_64_nonpadding,
                GroupedGemmSelectionReason::v4_n_aligned};
    }
    return {GroupedGemmInstanceId::v4_64_full_padding,
            GroupedGemmSelectionReason::v4_fallback};
}

constexpr std::string_view grouped_gemm_arch_name(GroupedGemmArch arch)
{
    return arch == GroupedGemmArch::gfx936 ? "gfx936"
           : arch == GroupedGemmArch::gfx938 ? "gfx938"
           : arch == GroupedGemmArch::gfx946 ? "gfx946"
                                              : "unsupported";
}

constexpr std::string_view grouped_gemm_data_type_name(GroupedGemmDataType data_type)
{
    return data_type == GroupedGemmDataType::fp16 ? "fp16"
           : data_type == GroupedGemmDataType::bf16 ? "bf16"
                                                     : "unsupported";
}

constexpr std::string_view grouped_gemm_layout_name(GroupedGemmLayout layout)
{
    return layout == GroupedGemmLayout::nt ? "NT"
           : layout == GroupedGemmLayout::nn ? "NN"
           : layout == GroupedGemmLayout::tn ? "TN"
                                              : "unsupported";
}

inline std::string stable_grouped_gemm_instance_id(const GroupedGemmProblem& problem,
                                                   const GroupedGemmSelection& selection)
{
    const auto* candidate = find_grouped_gemm_candidate(selection.instance_id);
    if(candidate == nullptr)
    {
        return {};
    }
    std::string id{"ck.gg."};
    id.append(grouped_gemm_arch_name(problem.arch));
    id.push_back('.');
    id.append(grouped_gemm_data_type_name(problem.data_type));
    id.push_back('.');
    id.append(grouped_gemm_layout_name(problem.layout));
    id.push_back('.');
    id.append(candidate->canonical_suffix);
    return id;
}

} // namespace ck_tile
