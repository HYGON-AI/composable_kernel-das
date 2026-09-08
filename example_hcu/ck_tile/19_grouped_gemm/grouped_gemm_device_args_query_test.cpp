// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <string>

namespace {

ck_tile_hcu_grouped_gemm_problem_v1 make_problem(int data_type,
                                                  int layout,
                                                  int groups,
                                                  std::int64_t m,
                                                  std::int64_t n,
                                                  std::int64_t k)
{
    ck_tile_hcu_grouped_gemm_problem_v1 problem{};
    problem.struct_size = sizeof(problem);
    problem.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    problem.data_type   = data_type;
    problem.layout      = layout;
    problem.group_count = groups;
    problem.problem_flags = CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1;
    problem.m = {layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1
                     ? CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1
                     : CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1,
                 0,
                 m};
    problem.n = {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, n};
    problem.k = {layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1
                     ? CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1
                     : CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1,
                 0,
                 k};
    return problem;
}

ck_tile_hcu_grouped_gemm_problem_v1 make_fixed_m_problem(int data_type,
                                                          int layout,
                                                          int groups,
                                                          std::int64_t m,
                                                          std::int64_t n,
                                                          std::int64_t k)
{
    auto problem          = make_problem(data_type, layout, groups, m, n, k);
    problem.problem_flags = 0;
    problem.m = {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, m};
    return problem;
}

ck_tile_hcu_grouped_gemm_problem_v1 make_homogeneous_tn_problem(int groups,
                                                                 std::int64_t m,
                                                                 std::int64_t n,
                                                                 std::int64_t k_per_group)
{
    auto problem = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1,
                                groups,
                                m,
                                n,
                                groups * k_per_group);
    problem.problem_flags = 0;
    return problem;
}

ck_tile_hcu_grouped_gemm_selection_v1 make_selection()
{
    ck_tile_hcu_grouped_gemm_selection_v1 selection{};
    selection.struct_size = sizeof(selection);
    selection.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    return selection;
}

bool expect_selection(ck_tile_hcu_grouped_gemm_problem_v1 problem,
                      std::uint32_t expected,
                      const char* name)
{
    auto selection = make_selection();
    const int status = ck_tile_hcu_grouped_gemm_select_device_args_v1(
        &problem, &selection);
    if(status != 0 || selection.instance_id != expected || selection.instance_name[0] == '\0')
    {
        std::cerr << "QUERY_CASE name=" << name << " status=" << status
                  << " expected=" << expected << " actual=" << selection.instance_id
                  << " result=FAIL\n";
        return false;
    }
    std::cout << "QUERY_CASE name=" << name << " instance=" << selection.instance_name
              << " result=PASS\n";
    return true;
}

} // namespace

int main()
{
    struct CandidateExpected
    {
        std::uint32_t id;
        std::int32_t family;
        std::uint32_t architecture_mask;
        std::uint32_t data_type_mask;
        std::uint32_t layout_mask;
        std::uint32_t contract_flags = 0;
    };
    constexpr std::uint32_t both_arch =
        CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1 |
        CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1;
    constexpr std::uint32_t all_arch =
        both_arch | CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX946_V1;
    constexpr std::uint32_t both_dtype =
        CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_FP16_V1 |
        CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1;
    constexpr std::uint32_t all_layout = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1 |
                                         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1 |
                                         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_TN_V1;
    constexpr std::array<CandidateExpected, 18> expected{{
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_SELECTED_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_BW_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         both_dtype,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1 |
             CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BF16_NN_FIXED_SRD_LDS8_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_BW_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BF16_NN_FAST_VMEM3_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_BW_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BLAS_TRANSPOSED_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_BW_V1,
         both_arch,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_TN_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_CONTRACT_C_TRANSPOSED_VIEW_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_LOGICAL_K_TAIL_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_BW_V1,
         both_arch,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_TN_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_CONTRACT_C_TRANSPOSED_VIEW_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_256_M_ONLY_PADDING_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX936_V3_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         both_dtype,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_DSREADM_BACKWARD_M_ONLY_PADDING_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX936_V3_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         both_dtype,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1 |
             CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_TN_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_DEFAULT_BACKWARD_M_ONLY_PADDING_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX936_V3_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         both_dtype,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX938_MLS_LARGE_256_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX938_MLS_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1,
         both_dtype,
         all_layout},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX938_MLS_SMALL_128_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX938_MLS_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1,
         both_dtype,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_128_M_ONLY_PADDING_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1,
         all_arch,
         both_dtype,
         all_layout},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_128_FULL_PADDING_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1,
         all_arch,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1,
         all_layout},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_NONPADDING_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1,
         all_arch,
         both_dtype,
         all_layout},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_FULL_PADDING_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1,
         all_arch,
         both_dtype,
         all_layout},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V3_128_M4_FULL_PADDING_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_FP16_V1,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V3_128_M4_FULL_PADDING_GFX936_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_FP16_V1,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_M01_4_GFX936_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1 |
             CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1},
        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V6_M32_NONPADDING_MLE8_FAMILY_B_GFX936_V1,
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1,
         CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1,
         CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_FP16_V1,
         CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1},
    }};
    std::set<std::string> names;
    bool passed = true;
    for(const auto& item : expected)
    {
        ck_tile_hcu_grouped_gemm_candidate_info_v1 info{};
        info.struct_size = sizeof(info);
        info.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
        const int status = ck_tile_hcu_grouped_gemm_query_instance_v1(item.id, &info);
        const char* direct_name = ck_tile_hcu_grouped_gemm_instance_name_v1(item.id);
        if(status != 0 || info.instance_id != item.id || info.family != item.family ||
           info.architecture_mask != item.architecture_mask ||
           info.data_type_mask != item.data_type_mask || info.layout_mask != item.layout_mask ||
           info.contract_flags != item.contract_flags ||
           info.canonical_name[0] == '\0' || direct_name == nullptr ||
           std::strcmp(info.canonical_name, direct_name) != 0)
        {
            std::cerr << "REGISTRY_CASE id=" << item.id << " status=" << status
                      << " result=FAIL\n";
            passed = false;
        }
        names.insert(info.canonical_name);
    }
    passed = passed && names.size() == expected.size() &&
             ck_tile_hcu_grouped_gemm_instance_name_v1(999999) == nullptr;

    const auto target = ck_tile::get_hcu_target_enum();
    if(target == ck_tile::hcu_target_enum::gfx936)
    {
        auto selected = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                     CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1,
                                     1,
                                     2048,
                                     2048,
                                     2048);
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
        passed &= expect_selection(selected, 2000, "gfx936_default_rdc_fallback");
#else
        passed &= expect_selection(selected, 1000, "gfx936_default");
#endif
        selected.policy_disable_flags =
            CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BW_FAMILY_V1;
        passed &= expect_selection(selected, 2000, "gfx936_v3_rollback");

        auto p1_family_a = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                        CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1,
                                        392,
                                        392 * 33,
                                        1232,
                                        1536);
#if !defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX936_DISABLE)
        passed &= expect_selection(p1_family_a, 4010, "gfx936_p1_family_a_default");
#else
        passed &= expect_selection(p1_family_a, 4003, "gfx936_p1_family_a_rollback");
#endif

        auto p1_family_b = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                        CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1,
                                        392,
                                        392 * 8,
                                        3072,
                                        1232);
#if !defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_B_V6_M32_MLE8_GFX936_DISABLE)
        passed &= expect_selection(p1_family_b, 4104, "gfx936_p1_family_b_default");
#else
        passed &= expect_selection(p1_family_b, 4002, "gfx936_p1_family_b_rollback");
#endif

        auto p2_fixed = make_fixed_m_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1,
                                             CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1,
                                             8,
                                             1919,
                                             4096,
                                             7168);
#if !defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_V4_64_M01_4_GFX936_DISABLE)
        passed &= expect_selection(p2_fixed, 4011, "gfx936_p2_fixed_m_default");
#else
        passed &= expect_selection(p2_fixed, 4002, "gfx936_p2_fixed_m_rollback");
#endif

        auto p2_fc2_nn = make_fixed_m_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1,
                                              CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1,
                                              8,
                                              2052,
                                              2048,
                                              7168);
#if !defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX936_FC2_NN_FAST_VMEM3_DISABLE)
        passed &= expect_selection(p2_fc2_nn, 1002, "gfx936_p2_fc2_nn_fast_vmem3_default");
#else
        passed &= expect_selection(p2_fc2_nn, 1000, "gfx936_p2_fc2_nn_rollback");
#endif

        auto p3_homogeneous = make_homogeneous_tn_problem(3, 1024, 4096, 4096);
#if !defined(CK_TILE_GROUPED_GEMM_P3_FP16_TN_M1024_K4096_ROUTE_DISABLE)
        passed &= expect_selection(p3_homogeneous, 2001, "gfx936_p3_homogeneous_default");
#else
        passed &= expect_selection(p3_homogeneous, 4000, "gfx936_p3_homogeneous_rollback");
#endif
        auto p3_ragged = p3_homogeneous;
        p3_ragged.problem_flags = CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1;
        passed &= expect_selection(p3_ragged, 4000, "gfx936_p3_ragged_rejected");
        auto p3_empty = p3_homogeneous;
        p3_empty.problem_flags = CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1;
        passed &= expect_selection(p3_empty, 4000, "gfx936_p3_empty_rejected");
    }
    else if(target == ck_tile::hcu_target_enum::gfx938)
    {
        auto selected = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                     CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1,
                                     1,
                                     512,
                                     512,
                                     512);
        passed &= expect_selection(selected, 3001, "gfx938_default");
        auto large = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1,
                                  CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1,
                                  4,
                                  8192,
                                  2048,
                                  2048);
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
        passed &= expect_selection(large, 4000, "gfx938_large_rdc_fallback");
#else
        passed &= expect_selection(large, 3000, "gfx938_large");
#endif
        selected.policy_disable_flags =
            CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX938_MLS_V1;
        passed &= expect_selection(selected, 4000, "gfx938_v4_rollback");

        auto p1_family_a = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                        CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1,
                                        392,
                                        392 * 33,
                                        1232,
                                        1536);
#if !defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX938_DISABLE)
        passed &= expect_selection(p1_family_a, 4009, "gfx938_p1_family_a_default");
#else
        passed &= expect_selection(p1_family_a, 4003, "gfx938_p1_family_a_rollback");
#endif

        auto p2_fc2_nn = make_fixed_m_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1,
                                              CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1,
                                              8,
                                              2052,
                                              2048,
                                              7168);
        passed &= expect_selection(p2_fc2_nn, 3000, "gfx938_p2_gfx936_fc2_nn_isolated");

        auto p3_homogeneous = make_homogeneous_tn_problem(16, 1024, 4096, 4096);
#if !defined(CK_TILE_GROUPED_GEMM_P3_FP16_TN_M1024_K4096_ROUTE_DISABLE)
        passed &= expect_selection(p3_homogeneous, 3000, "gfx938_p3_homogeneous_default");
#else
        passed &= expect_selection(p3_homogeneous, 4000, "gfx938_p3_homogeneous_rollback");
#endif
        auto p3_ragged = p3_homogeneous;
        p3_ragged.problem_flags = CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1;
        passed &= expect_selection(p3_ragged, 4000, "gfx938_p3_ragged_rejected");
        auto p3_empty = p3_homogeneous;
        p3_empty.problem_flags = CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1;
        passed &= expect_selection(p3_empty, 4000, "gfx938_p3_empty_rejected");
    }
    else if(target == ck_tile::hcu_target_enum::gfx946)
    {
        auto aligned = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1,
                                    1,
                                    64,
                                    128,
                                    128);
        passed &= expect_selection(aligned, 4002, "gfx946_generic_aligned_n");
        auto padded = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                   CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1,
                                   1,
                                   65,
                                   129,
                                   129);
        passed &= expect_selection(padded, 4003, "gfx946_generic_padding");
    }
    else
    {
        std::cerr << "QUERY_CASE result=FAIL reason=unsupported_test_arch\n";
        return 1;
    }

    auto invalid = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1,
                                1,
                                64,
                                128,
                                128);
    invalid.n.kind =
        CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1;
    auto invalid_selection = make_selection();
    passed &= ck_tile_hcu_grouped_gemm_select_device_args_v1(
                  &invalid, &invalid_selection) ==
              CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;

    invalid = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                           CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1,
                           1,
                           64,
                           128,
                           128);
    invalid.abi_version = 99;
    passed &= ck_tile_hcu_grouped_gemm_select_device_args_v1(
                  &invalid, &invalid_selection) ==
              CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_VERSION_V1;

    invalid = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                           CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1,
                           1,
                           64,
                           128,
                           128);
    invalid.struct_size = sizeof(invalid) - 1;
    passed &= ck_tile_hcu_grouped_gemm_select_device_args_v1(
                  &invalid, &invalid_selection) ==
              CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_STRUCT_SIZE_V1;

    auto run_problem = make_problem(CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1,
                                    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1,
                                    1,
                                    target == ck_tile::hcu_target_enum::gfx936 ? 2048 : 512,
                                    target == ck_tile::hcu_target_enum::gfx936 ? 2048 : 512,
                                    target == ck_tile::hcu_target_enum::gfx936 ? 2048 : 512);
    ck_tile_hcu_grouped_gemm_launch_v1 launch{};
    launch.struct_size = sizeof(launch);
    launch.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    passed &= ck_tile_hcu_grouped_gemm_run_device_args_v1(
                  &run_problem, &launch, nullptr) ==
              CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1;
    launch.device_args = reinterpret_cast<const void*>(1);
    launch.expected_instance_id = 999999;
    passed &= ck_tile_hcu_grouped_gemm_run_device_args_v1(
                  &run_problem, &launch, nullptr) ==
              CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INSTANCE_MISMATCH_V1;

    std::cout << "DEVICE_ARGS_QUERY_" << (passed ? "PASS" : "FAIL")
              << " registry=" << expected.size() << "\n";
    return passed ? 0 : 1;
}
