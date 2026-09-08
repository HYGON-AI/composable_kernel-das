// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ck_tile;

struct TestCase
{
    const char* name;
    GroupedGemmProblem problem;
    GroupedGemmSelectorPolicy policy;
    GroupedGemmInstanceId expected;
};

GroupedGemmProblem make_problem(GroupedGemmArch arch,
                                GroupedGemmDataType data_type,
                                GroupedGemmLayout layout,
                                std::int32_t groups,
                                std::int64_t m,
                                std::int64_t n,
                                std::int64_t k)
{
    const bool variable_k = layout == GroupedGemmLayout::tn;
    return {arch,
            data_type,
            layout,
            groups,
            variable_k ? GroupedGemmDimensionSummary::Common(m)
                       : GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(m),
            GroupedGemmDimensionSummary::Common(n),
            variable_k ? GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(k)
                       : GroupedGemmDimensionSummary::Common(k),
            false,
            true};
}

GroupedGemmProblem make_fixed_m_problem(GroupedGemmArch arch,
                                        GroupedGemmDataType data_type,
                                        GroupedGemmLayout layout,
                                        std::int32_t groups,
                                        std::int64_t m,
                                        std::int64_t n,
                                        std::int64_t k)
{
    auto problem = make_problem(arch, data_type, layout, groups, m, n, k);
    problem.m    = GroupedGemmDimensionSummary::Common(m);
    return problem;
}

GroupedGemmProblem make_homogeneous_tn_problem(GroupedGemmArch arch,
                                               std::int32_t groups,
                                               std::int64_t m,
                                               std::int64_t n,
                                               std::int64_t k_per_group)
{
    auto problem = make_problem(arch,
                                GroupedGemmDataType::fp16,
                                GroupedGemmLayout::tn,
                                groups,
                                m,
                                n,
                                groups * k_per_group);
    problem.group_lengths_are_ragged = false;
    return problem;
}

std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    for(std::string field; std::getline(stream, field, ',');)
    {
        fields.push_back(field);
    }
    return fields;
}

GroupedGemmArch parse_arch(const std::string& value)
{
    if(value == "gfx936")
        return GroupedGemmArch::gfx936;
    if(value == "gfx938")
        return GroupedGemmArch::gfx938;
    if(value == "gfx946")
        return GroupedGemmArch::gfx946;
    return GroupedGemmArch::unsupported;
}

GroupedGemmDataType parse_data_type(const std::string& value)
{
    if(value == "fp16")
        return GroupedGemmDataType::fp16;
    if(value == "bf16")
        return GroupedGemmDataType::bf16;
    return GroupedGemmDataType::unsupported;
}

GroupedGemmLayout parse_layout(const std::string& value)
{
    if(value == "NT")
        return GroupedGemmLayout::nt;
    if(value == "NN")
        return GroupedGemmLayout::nn;
    if(value == "TN")
        return GroupedGemmLayout::tn;
    return GroupedGemmLayout::unsupported;
}

int run_table_tests()
{
    const GroupedGemmSelectorPolicy defaults{};
    GroupedGemmSelectorPolicy no_bw = defaults;
    no_bw.enable_bw_family          = false;
    GroupedGemmSelectorPolicy no_dsreadm = no_bw;
    no_dsreadm.enable_gfx936_dsreadm    = false;
    GroupedGemmSelectorPolicy no_mls = defaults;
    no_mls.enable_gfx938_mls        = false;
    GroupedGemmSelectorPolicy no_tail = defaults;
    no_tail.enable_bf16_tn_logical_k_tail = false;
    GroupedGemmSelectorPolicy no_bw_selected = defaults;
    no_bw_selected.enable_bw_selected        = false;
    GroupedGemmSelectorPolicy no_mls_large = defaults;
    no_mls_large.enable_gfx938_mls_large   = false;
    GroupedGemmSelectorPolicy p1_family_a_gfx938_disabled = defaults;
    p1_family_a_gfx938_disabled.enable_p1_family_a_v3_128_gfx938 = false;
    GroupedGemmSelectorPolicy p1_family_a_gfx936_disabled = defaults;
    p1_family_a_gfx936_disabled.enable_p1_family_a_v3_128_gfx936 = false;
    GroupedGemmSelectorPolicy p1_family_b_gfx936_disabled = defaults;
    p1_family_b_gfx936_disabled.enable_p1_family_b_v6_m32_mle8_gfx936 = false;
    GroupedGemmSelectorPolicy p2_bf16_g8_v4_64_m01_4_gfx936_disabled = defaults;
    p2_bf16_g8_v4_64_m01_4_gfx936_disabled
        .enable_p2_bf16_g8_v4_64_m01_4_gfx936 = false;
    GroupedGemmSelectorPolicy p2_bf16_g8_gfx936_fc2_nn_fast_vmem3_disabled = defaults;
    p2_bf16_g8_gfx936_fc2_nn_fast_vmem3_disabled
        .enable_p2_bf16_g8_gfx936_fc2_nn_fast_vmem3 = false;
    GroupedGemmSelectorPolicy p2_bf16_g8_gfx938_v5_fc1_route_disabled = defaults;
    p2_bf16_g8_gfx938_v5_fc1_route_disabled
        .enable_p2_bf16_g8_gfx938_v5_fc1_route = false;
    GroupedGemmSelectorPolicy p3_fp16_tn_m1024_k4096_route_disabled = defaults;
    p3_fp16_tn_m1024_k4096_route_disabled
        .enable_p3_fp16_tn_m1024_k4096_route = false;

    auto p3_ragged = make_homogeneous_tn_problem(
        GroupedGemmArch::gfx936, 3, 1024, 4096, 4096);
    p3_ragged.group_lengths_are_ragged = true;
    auto p3_empty = make_homogeneous_tn_problem(
        GroupedGemmArch::gfx938, 16, 1024, 4096, 4096);
    p3_empty.may_have_empty_groups = true;

    auto empty_ragged = make_problem(GroupedGemmArch::gfx936,
                                     GroupedGemmDataType::fp16,
                                     GroupedGemmLayout::nt,
                                     5,
                                     5 * 1920,
                                     2048,
                                     2048);
    empty_ragged.may_have_empty_groups    = true;
    empty_ragged.group_lengths_are_ragged = true;

    auto invalid_two_variable = make_problem(GroupedGemmArch::gfx936,
                                             GroupedGemmDataType::fp16,
                                             GroupedGemmLayout::nt,
                                             4,
                                             8192,
                                             2048,
                                             2048);
    invalid_two_variable.k = GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(8192);

    const std::vector<TestCase> cases{
        {"gfx946 FP16 NT small aligned N uses generic V4 nonpadding",
         make_problem(GroupedGemmArch::gfx946, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 64, 128, 128),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"gfx946 FP16 NN padding uses generic V4 full padding",
         make_problem(GroupedGemmArch::gfx946, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 1, 65, 129, 129),
         defaults,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"gfx946 FP16 TN variable K uses generic V4 nonpadding",
         make_problem(GroupedGemmArch::gfx946, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::tn, 1, 64, 128, 128),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P3 gfx936 FP16 TN homogeneous route selects V3 dsreadm by default",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx936, 1, 1024, 4096, 4096),
         defaults,
         GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding},
        {"P3 gfx938 FP16 TN homogeneous route selects MLS large by default",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx938, 16, 1024, 4096, 4096),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_large_256},
        {"P3 gfx936 FP16 TN homogeneous explicit rollback stays V4",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx936, 1, 1024, 4096, 4096),
         p3_fp16_tn_m1024_k4096_route_disabled,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"P3 gfx938 FP16 TN homogeneous explicit rollback stays V4",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx938, 16, 1024, 4096, 4096),
         p3_fp16_tn_m1024_k4096_route_disabled,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"P3 gfx936 FP16 TN homogeneous G3 route selects V3 dsreadm",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx936, 3, 1024, 4096, 4096),
         defaults,
         GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding},
        {"P3 gfx936 FP16 TN homogeneous G16 route selects V3 dsreadm",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx936, 16, 1024, 4096, 4096),
         defaults,
         GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding},
        {"P3 gfx938 FP16 TN homogeneous G3 route selects MLS large",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx938, 3, 1024, 4096, 4096),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_large_256},
        {"P3 gfx938 FP16 TN homogeneous G16 route selects MLS large",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx938, 16, 1024, 4096, 4096),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_large_256},
        {"P3 route leaves N3072 near miss on V4",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx936, 8, 1024, 3072, 4096),
         defaults,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"P3 route leaves K2048 near miss on V4",
         make_homogeneous_tn_problem(GroupedGemmArch::gfx938, 8, 1024, 4096, 2048),
         defaults,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"P3 route rejects ragged K lengths",
         p3_ragged,
         defaults,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"P3 route rejects possibly empty groups",
         p3_empty,
         defaults,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"P2 gfx938 V5 FC1 route is default for misaligned NT",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1919, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 gfx938 V5 FC1 route explicit rollback stays MLS",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1919, 4096, 7168),
         p2_bf16_g8_gfx938_v5_fc1_route_disabled,
         GroupedGemmInstanceId::gfx938_mls_small_128},
        {"P2 gfx938 V5 FC1 route misaligned NT",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1919, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 gfx938 V5 FC1 route aligned NT",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1920, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"P2 gfx938 V5 FC1 route leaves FC2 NT on MLS",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1919, 7168, 2048),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_small_128},
        {"P2 gfx938 V5 FC1 route leaves NN unchanged",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 1919, 7168, 4096),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 gfx936 fixed-M NT route is default",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1919, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_m01_4_gfx936},
        {"P2 gfx936 fixed-M NT explicit rollback",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1919, 4096, 7168),
         p2_bf16_g8_v4_64_m01_4_gfx936_disabled,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 gfx936 fixed-M NT non-aligned route",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1919, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_m01_4_gfx936},
        {"P2 gfx936 fixed-M NN non-aligned route",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 1921, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_m01_4_gfx936},
        {"P2 gfx936 FC2 NN fast-vmem3 production default",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 2052, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3},
        {"P2 gfx936 FC2 NN fast-vmem3 rollback",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 2052, 2048, 7168),
         p2_bf16_g8_gfx936_fc2_nn_fast_vmem3_disabled,
         GroupedGemmInstanceId::bw_family_selected},
        {"P2 gfx936 FC2 NN fast-vmem3 route lower boundary",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 2048, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3},
        {"P2 gfx936 FC2 NN fast-vmem3 route upper boundary",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 2221, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3},
        {"P2 gfx936 FC2 NN fast-vmem3 route below range",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 2047, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 gfx936 FC2 NN fast-vmem3 route above range",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 2222, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_selected},
        {"P2 gfx936 FC2 NN fast-vmem3 route leaves gfx938 unchanged",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 2052, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_large_256},
        {"P2 gfx936 FC2 NN fast-vmem3 route rejects G16 common-M",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 16, 2052, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_selected},
        {"P2 gfx936 FC2 NN fast-vmem3 route rejects NT",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 2052, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_selected},
        {"P2 gfx936 FC2 NN fast-vmem3 route accepts integral device-length M",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 8, 8 * 2052, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3},
        {"P2 gfx936 FC2 NN fast-vmem3 device-length rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 8, 8 * 2052, 2048, 7168),
         p2_bf16_g8_gfx936_fc2_nn_fast_vmem3_disabled,
         GroupedGemmInstanceId::bw_family_selected},
        {"P2 gfx936 FC2 NN fast-vmem3 accepts possibly empty device groups",
         [] {
             auto problem = make_problem(GroupedGemmArch::gfx936,
                                         GroupedGemmDataType::bf16,
                                         GroupedGemmLayout::nn,
                                         8,
                                         8 * 2052,
                                         2048,
                                         7168);
             problem.may_have_empty_groups = true;
             return problem;
         }(),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3},
        {"P2 gfx936 FC2 NN fast-vmem3 rejects fractional device capacity",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 8, 8 * 2052 + 1, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_selected},
        {"P2 fixed-M aligned anchor stays ID4000",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1920, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"P2 gfx938 fixed-M NN non-aligned fallback",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nn, 8, 1919, 7168, 4096),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 gfx936 route retains gfx938 V5 default",
         make_fixed_m_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1919, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 FP16 near miss",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                              GroupedGemmLayout::nt, 8, 1919, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 G7 near miss",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 7, 1919, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 M1917 near miss",
         make_fixed_m_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                              GroupedGemmLayout::nt, 8, 1917, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"P2 variable-capacity near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nt, 8, 8 * 1919, 4096, 7168),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"gfx936 regular threshold below",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 1919, 2048, 2048),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"gfx936 regular threshold at",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 1920, 2048, 2048),
         defaults,
         GroupedGemmInstanceId::bw_family_selected},
        {"gfx936 TN threshold below",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::tn, 4, 2047, 2048, 8192),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"gfx936 TN logical K tail",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::tn, 4, 2048, 2048, 8191),
         defaults,
         GroupedGemmInstanceId::bw_family_logical_k_tail},
        {"gfx936 TN aligned rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::tn, 4, 2048, 2048, 8192),
         no_tail,
         GroupedGemmInstanceId::bw_family_blas_transposed},
        {"gfx936 BF16 NN first promoted family",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 16, 16 * 2048, 7168, 4096),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8},
        {"gfx936 BF16 NN second promoted family",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 16, 16 * 2048, 2048, 7168),
         defaults,
         GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3},
        {"gfx936 V3 NT rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 4, 4 * 2048, 2048, 2048),
         no_bw,
         GroupedGemmInstanceId::gfx936_v3_256_m_only_padding},
        {"gfx936 BW selected capability rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 4, 4 * 2048, 2048, 2048),
         no_bw_selected,
         GroupedGemmInstanceId::gfx936_v3_256_m_only_padding},
        {"gfx936 DSReadM NN rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 4, 4 * 2048, 2048, 2048),
         no_bw,
         GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding},
        {"gfx936 default backward when DSReadM disabled",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 4, 4 * 2048, 2048, 2048),
         no_dsreadm,
         GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding},
        {"gfx938 MLS large",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 4, 4 * 2048, 2048, 2048),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_large_256},
        {"gfx938 MLS small NT",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 512, 512, 512),
         defaults,
         GroupedGemmInstanceId::gfx938_mls_small_128},
        {"gfx938 MLS large capability rollback",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 1, 2048, 2048, 2048),
         no_mls_large,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"gfx938 MLS disabled aligned V4",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 512, 512, 512),
         no_mls,
         GroupedGemmInstanceId::v4_128_m_only_padding},
        {"BF16 K tail V4 padding",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nt, 1, 512, 512, 513),
         defaults,
         GroupedGemmInstanceId::v4_128_full_padding},
        {"FP16 N aligned fallback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 511, 512, 511),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"full fallback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 511, 513, 511),
         defaults,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 Family A gfx936 below M boundary keeps fallback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 32, 1232, 1536),
         defaults,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 Family A gfx936 is default at M boundary",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         defaults,
         GroupedGemmInstanceId::v3_128_m4_full_padding_gfx936},
        {"P1 Family A gfx936 explicit rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_gfx936_disabled,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 Family A gfx938 is default at M boundary",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         defaults,
         GroupedGemmInstanceId::v3_128_m4_full_padding},
        {"P1 Family A gfx938 explicit rollback",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 33, 1232, 1536),
         p1_family_a_gfx938_disabled,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"P1 Family A does not replace BF16 full padding",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 8, 8 * 128, 1232, 1536),
         defaults,
         GroupedGemmInstanceId::v4_64_full_padding},
        {"Family-B gfx936 selector-effective M8 is default",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 8, 3072, 1232),
         defaults,
         GroupedGemmInstanceId::v6_m32_nonpadding_mle8_family_b_gfx936},
        {"Family-B gfx936 selector-effective M8 explicit rollback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 8, 3072, 1232),
         p1_family_b_gfx936_disabled,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B gfx936 selector-effective M9 production fallback",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 9, 3072, 1232),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B gfx938 remains on fallback",
         make_problem(GroupedGemmArch::gfx938, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 392, 392 * 8, 3072, 1232),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B BF16 near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::bf16,
                      GroupedGemmLayout::nn, 1, 255, 3072, 1232),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B NT near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 1, 255, 3072, 1232),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B N near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 1, 255, 2944, 1232),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"Family-B K near miss",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nn, 1, 255, 3072, 1233),
         defaults,
         GroupedGemmInstanceId::v4_64_nonpadding},
        {"empty and ragged metadata stays selector-neutral",
         empty_ragged,
         defaults,
         GroupedGemmInstanceId::bw_family_selected},
        {"invalid zero groups",
         make_problem(GroupedGemmArch::gfx936, GroupedGemmDataType::fp16,
                      GroupedGemmLayout::nt, 0, 2048, 2048, 2048),
         defaults,
         GroupedGemmInstanceId::none},
        {"invalid two variable dimensions",
         invalid_two_variable,
         defaults,
         GroupedGemmInstanceId::none},
    };

    int failures = 0;
    for(const auto& test : cases)
    {
        const auto actual = select_grouped_gemm_candidate(test.problem, test.policy);
        if(actual.instance_id != test.expected)
        {
            std::cerr << "TABLE_MISMATCH name=" << test.name
                      << " expected=" << static_cast<int>(test.expected)
                      << " actual=" << static_cast<int>(actual.instance_id) << '\n';
            ++failures;
        }
    }

    std::set<int> ids;
    std::set<std::string_view> suffixes;
    for(const auto& candidate : grouped_gemm_candidate_registry)
    {
        ids.insert(static_cast<int>(candidate.id));
        suffixes.insert(candidate.canonical_suffix);
    }
    if(ids.size() != grouped_gemm_candidate_registry.size() ||
       suffixes.size() != grouped_gemm_candidate_registry.size())
    {
        std::cerr << "REGISTRY_DUPLICATE\n";
        ++failures;
    }

    if(failures == 0)
    {
        std::cout << "SELECTOR_TABLE_PASS cases=" << cases.size()
                  << " registry=" << grouped_gemm_candidate_registry.size() << '\n';
    }
    return failures;
}

int compare_golden(const std::string& golden_path, const std::string& diff_path)
{
    std::ifstream input(golden_path);
    if(!input)
    {
        throw std::runtime_error("cannot open golden CSV: " + golden_path);
    }
    std::ofstream diff;
    if(!diff_path.empty())
    {
        diff.open(diff_path);
        if(!diff)
            throw std::runtime_error("cannot open diff CSV: " + diff_path);
        diff << "arch,dtype,layout,group_num,m_arg,n_arg,k_arg,expected,actual\n";
    }

    std::string line;
    std::getline(input, line);
    const auto header = split_csv_line(line);
    const std::vector<std::string> required{"arch", "dtype", "layout", "group_num",
                                            "m_arg", "n_arg", "k_arg", "instance_id"};
    std::vector<std::size_t> columns;
    for(const auto& name : required)
    {
        const auto it = std::find(header.begin(), header.end(), name);
        if(it == header.end())
            throw std::runtime_error("missing golden column: " + name);
        columns.push_back(static_cast<std::size_t>(it - header.begin()));
    }

    int rows = 0;
    int mismatches = 0;
    while(std::getline(input, line))
    {
        if(line.empty())
            continue;
        const auto fields = split_csv_line(line);
        for(auto column : columns)
            if(column >= fields.size())
                throw std::runtime_error("short golden row");

        const auto arch = parse_arch(fields[columns[0]]);
        const auto data_type = parse_data_type(fields[columns[1]]);
        const auto layout = parse_layout(fields[columns[2]]);
        const auto groups = static_cast<std::int32_t>(std::stol(fields[columns[3]]));
        const auto m = std::stoll(fields[columns[4]]);
        const auto n = std::stoll(fields[columns[5]]);
        const auto k = std::stoll(fields[columns[6]]);
        const auto expected = fields[columns[7]];
        const auto problem = make_problem(arch, data_type, layout, groups, m, n, k);
        const auto selection = select_grouped_gemm_candidate(problem);
        const auto actual = stable_grouped_gemm_instance_id(problem, selection);
        ++rows;
        if(actual != expected)
        {
            ++mismatches;
            if(diff)
            {
                diff << fields[columns[0]] << ',' << fields[columns[1]] << ','
                     << fields[columns[2]] << ',' << groups << ',' << m << ',' << n << ',' << k
                     << ',' << expected << ',' << actual << '\n';
            }
        }
    }

    if(mismatches == 0)
    {
        std::cout << "SELECTOR_GOLDEN_PASS rows=" << rows << '\n';
    }
    else
    {
        std::cerr << "SELECTOR_GOLDEN_MISMATCH rows=" << rows
                  << " mismatches=" << mismatches << '\n';
    }
    return mismatches;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        int failures = run_table_tests();
        std::string golden_path;
        std::string diff_path;
        for(int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if(arg == "--golden" && i + 1 < argc)
                golden_path = argv[++i];
            else if(arg == "--diff" && i + 1 < argc)
                diff_path = argv[++i];
            else
                throw std::runtime_error("usage: grouped_gemm_selector_test [--golden CSV] [--diff CSV]");
        }
        if(!golden_path.empty())
            failures += compare_golden(golden_path, diff_path);
        return failures == 0 ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << "SELECTOR_TEST_ERROR " << error.what() << '\n';
        return 2;
    }
}
