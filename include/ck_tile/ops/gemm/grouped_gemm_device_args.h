// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#ifndef CK_TILE_OPS_GEMM_GROUPED_GEMM_DEVICE_ARGS_H
#define CK_TILE_OPS_GEMM_GROUPED_GEMM_DEVICE_ARGS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION 1u
#define CK_TILE_HCU_GROUPED_GEMM_INSTANCE_NAME_CAPACITY 160u

typedef enum ck_tile_hcu_grouped_gemm_device_args_status_v1
{
    CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1           = 0,
    CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1  = -1,
    CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_VERSION_V1       = -2,
    CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_STRUCT_SIZE_V1   = -3,
    CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_ARCH_V1  = -4,
    CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1       = -5,
    CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INSTANCE_MISMATCH_V1 = -6,
    CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INTERNAL_ERROR_V1    = -100
} ck_tile_hcu_grouped_gemm_device_args_status_v1;

typedef enum ck_tile_hcu_grouped_gemm_data_type_v1
{
    CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_INVALID_V1 = 0,
    CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1    = 1,
    CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1    = 2
} ck_tile_hcu_grouped_gemm_data_type_v1;

typedef enum ck_tile_hcu_grouped_gemm_layout_v1
{
    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_INVALID_V1 = 0,
    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1      = 1,
    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1      = 2,
    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1      = 3
} ck_tile_hcu_grouped_gemm_layout_v1;

// Stable IDs name precompiled CK candidates. Consumers may select among
// already-linked entries, but cannot instantiate C++ templates at runtime.
typedef enum ck_tile_hcu_grouped_gemm_instance_id_v1
{
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_NONE_V1 = 0,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_SELECTED_V1 = 1000,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BF16_NN_FIXED_SRD_LDS8_V1 = 1001,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BF16_NN_FAST_VMEM3_V1 = 1002,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_BLAS_TRANSPOSED_V1 = 1003,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_BW_FAMILY_LOGICAL_K_TAIL_V1 = 1004,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_256_M_ONLY_PADDING_V1 = 2000,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_DSREADM_BACKWARD_M_ONLY_PADDING_V1 = 2001,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX936_V3_DEFAULT_BACKWARD_M_ONLY_PADDING_V1 = 2002,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX938_MLS_LARGE_256_V1 = 3000,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_GFX938_MLS_SMALL_128_V1 = 3001,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_128_M_ONLY_PADDING_V1 = 4000,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_128_FULL_PADDING_V1 = 4001,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_NONPADDING_V1 = 4002,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_FULL_PADDING_V1 = 4003,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V3_128_M4_FULL_PADDING_V1 = 4009,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V3_128_M4_FULL_PADDING_GFX936_V1 = 4010,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V4_64_M01_4_GFX936_V1 = 4011,
    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V6_M32_NONPADDING_MLE8_FAMILY_B_GFX936_V1 = 4104
} ck_tile_hcu_grouped_gemm_instance_id_v1;

typedef enum ck_tile_hcu_grouped_gemm_candidate_family_v1
{
    CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_BW_V1 = 0,
    CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX936_V3_V1 = 1,
    CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_GFX938_MLS_V1 = 2,
    CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_FAMILY_V4_V1 = 3
} ck_tile_hcu_grouped_gemm_candidate_family_v1;

typedef enum ck_tile_hcu_grouped_gemm_architecture_mask_v1
{
    CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1 = 1u << 0,
    CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1 = 1u << 1,
    CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX946_V1 = 1u << 2
} ck_tile_hcu_grouped_gemm_architecture_mask_v1;

typedef enum ck_tile_hcu_grouped_gemm_data_type_mask_v1
{
    CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_FP16_V1 = 1u << 0,
    CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_BF16_V1 = 1u << 1
} ck_tile_hcu_grouped_gemm_data_type_mask_v1;

typedef enum ck_tile_hcu_grouped_gemm_layout_mask_v1
{
    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1 = 1u << 0,
    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NN_V1 = 1u << 1,
    CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_TN_V1 = 1u << 2
} ck_tile_hcu_grouped_gemm_layout_mask_v1;

typedef enum ck_tile_hcu_grouped_gemm_candidate_contract_flags_v1
{
    CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_CONTRACT_C_TRANSPOSED_VIEW_V1 = 1u << 0
} ck_tile_hcu_grouped_gemm_candidate_contract_flags_v1;

typedef enum ck_tile_hcu_grouped_gemm_dimension_kind_v1
{
    CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1 = 0,
    CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1 = 1
} ck_tile_hcu_grouped_gemm_dimension_kind_v1;

typedef enum ck_tile_hcu_grouped_gemm_problem_flags_v1
{
    CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1 = 1u << 0,
    // Set only when the caller knows that host-visible per-group logical lengths differ.
    // A DEVICE_LENGTHS_WITH_CAPACITY dimension may still describe homogeneous groups.
    CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1  = 1u << 1
} ck_tile_hcu_grouped_gemm_problem_flags_v1;

typedef enum ck_tile_hcu_grouped_gemm_selection_reason_v1
{
    CK_TILE_HCU_GROUPED_GEMM_SELECTION_INVALID_CONTRACT_V1 = 0,
    CK_TILE_HCU_GROUPED_GEMM_SELECTION_BW_FAMILY_V1        = 1,
    CK_TILE_HCU_GROUPED_GEMM_SELECTION_GFX936_V3_V1        = 2,
    CK_TILE_HCU_GROUPED_GEMM_SELECTION_GFX938_MLS_V1       = 3,
    CK_TILE_HCU_GROUPED_GEMM_SELECTION_V4_ALIGNED_V1       = 4,
    CK_TILE_HCU_GROUPED_GEMM_SELECTION_V4_BF16_PADDING_V1  = 5,
    CK_TILE_HCU_GROUPED_GEMM_SELECTION_V4_N_ALIGNED_V1     = 6,
    CK_TILE_HCU_GROUPED_GEMM_SELECTION_V4_FALLBACK_V1      = 7
} ck_tile_hcu_grouped_gemm_selection_reason_v1;

// These opt-outs expose CK's rollback policy without letting a consumer name
// or instantiate a C++ template. Zero selects the CK production policy.
typedef enum ck_tile_hcu_grouped_gemm_policy_disable_flags_v1
{
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BW_FAMILY_V1 = 1u << 0,
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_V3_V1  = 1u << 1,
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_DSREADM_V1 = 1u << 2,
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX938_MLS_V1 = 1u << 3,
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_BF16_NN_PROMOTED_V1 = 1u << 4,
    CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BF16_TN_LOGICAL_K_TAIL_V1 = 1u << 5
} ck_tile_hcu_grouped_gemm_policy_disable_flags_v1;

typedef struct ck_tile_hcu_grouped_gemm_dimension_v1
{
    uint32_t kind;
    uint32_t reserved;
    int64_t extent;
} ck_tile_hcu_grouped_gemm_dimension_v1;

// `extent` is a logical common length or a host-visible backing capacity. The
// API never copies or reads the device-resident per-group lengths.
typedef struct ck_tile_hcu_grouped_gemm_problem_v1
{
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t data_type;
    int32_t layout;
    int32_t group_count;
    uint32_t problem_flags;
    uint32_t policy_disable_flags;
    uint32_t reserved0;
    ck_tile_hcu_grouped_gemm_dimension_v1 m;
    ck_tile_hcu_grouped_gemm_dimension_v1 n;
    ck_tile_hcu_grouped_gemm_dimension_v1 k;
    uint64_t reserved[4];
} ck_tile_hcu_grouped_gemm_problem_v1;

typedef struct ck_tile_hcu_grouped_gemm_selection_v1
{
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t instance_id;
    int32_t reason;
    char instance_name[CK_TILE_HCU_GROUPED_GEMM_INSTANCE_NAME_CAPACITY];
    uint64_t reserved[4];
} ck_tile_hcu_grouped_gemm_selection_v1;

typedef struct ck_tile_hcu_grouped_gemm_launch_v1
{
    uint32_t struct_size;
    uint32_t abi_version;
    const void* device_args;
    uint32_t num_cu;
    uint32_t expected_instance_id;
    void* stream;
    uint64_t reserved[4];
} ck_tile_hcu_grouped_gemm_launch_v1;

typedef struct ck_tile_hcu_grouped_gemm_candidate_info_v1
{
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t instance_id;
    int32_t family;
    uint32_t architecture_mask;
    uint32_t data_type_mask;
    uint32_t layout_mask;
    uint32_t contract_flags;
    char canonical_name[CK_TILE_HCU_GROUPED_GEMM_INSTANCE_NAME_CAPACITY];
    uint64_t reserved[4];
} ck_tile_hcu_grouped_gemm_candidate_info_v1;

int ck_tile_hcu_grouped_gemm_select_device_args_v1(
    const ck_tile_hcu_grouped_gemm_problem_v1* problem,
    ck_tile_hcu_grouped_gemm_selection_v1* selection);

int ck_tile_hcu_grouped_gemm_run_device_args_v1(
    const ck_tile_hcu_grouped_gemm_problem_v1* problem,
    const ck_tile_hcu_grouped_gemm_launch_v1* launch,
    ck_tile_hcu_grouped_gemm_selection_v1* selection);

const char* ck_tile_hcu_grouped_gemm_instance_name_v1(uint32_t instance_id);

int ck_tile_hcu_grouped_gemm_query_instance_v1(
    uint32_t instance_id,
    ck_tile_hcu_grouped_gemm_candidate_info_v1* candidate);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CK_TILE_OPS_GEMM_GROUPED_GEMM_DEVICE_ARGS_H
