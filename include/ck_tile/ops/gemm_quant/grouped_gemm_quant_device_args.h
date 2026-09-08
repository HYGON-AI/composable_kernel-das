// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#ifndef CK_TILE_OPS_GEMM_QUANT_GROUPED_GEMM_QUANT_DEVICE_ARGS_H
#define CK_TILE_OPS_GEMM_QUANT_GROUPED_GEMM_QUANT_DEVICE_ARGS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION 1u
#define CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_NAME_CAPACITY 160u

typedef enum ck_tile_hcu_quant_grouped_gemm_architecture_mask_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1 = 1u << 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_ARCHITECTURE_MASK_GFX946_V1 = 1u << 1
} ck_tile_hcu_quant_grouped_gemm_architecture_mask_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_quant_mode_mask_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_MASK_TENSORWISE_V1 = 1u << 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_MASK_ROWWISE_V1    = 1u << 1,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_MASK_BLOCKWISE_V1  = 1u << 2
} ck_tile_hcu_quant_grouped_gemm_quant_mode_mask_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_status_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1           = 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_INVALID_ARGUMENT_V1  = -1,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_BAD_VERSION_V1       = -2,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_BAD_STRUCT_SIZE_V1   = -3,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_ARCH_V1  = -4,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_V1       = -5,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_MISMATCH_V1 = -6,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_BAD_DEVICE_ARGS_V1   = -7,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_INTERNAL_ERROR_V1    = -100
} ck_tile_hcu_quant_grouped_gemm_status_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_data_type_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_INVALID_V1  = 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E4M3_V1 = 1,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E5M2_V1 = 2,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP16_V1     = 3,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_BF16_V1     = 4
} ck_tile_hcu_quant_grouped_gemm_data_type_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_layout_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_INVALID_V1 = 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_NN_V1      = 1,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_NT_V1      = 2,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_TN_V1      = 3
} ck_tile_hcu_quant_grouped_gemm_layout_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_mode_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_INVALID_V1     = 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_TENSORWISE_V1  = 1,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_ROWWISE_V1     = 2,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_BLOCKWISE_V1   = 3
} ck_tile_hcu_quant_grouped_gemm_mode_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_instance_id_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_NONE_V1 = 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_PERSISTENT_128X128X32_V1 = 5000,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_PERSISTENT_128X128X128_V1 = 5001
} ck_tile_hcu_quant_grouped_gemm_instance_id_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_dimension_kind_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DIMENSION_COMMON_V1 = 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1 = 1
} ck_tile_hcu_quant_grouped_gemm_dimension_kind_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_problem_flags_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1 = 1u << 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1  = 1u << 1
} ck_tile_hcu_quant_grouped_gemm_problem_flags_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_device_args_kind_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_KIND_INVALID_V1 = 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_KIND_QUANT_GEMM_TRANS_V1 = 1
} ck_tile_hcu_quant_grouped_gemm_device_args_kind_v1;

typedef enum ck_tile_hcu_quant_grouped_gemm_selection_reason_v1
{
    CK_TILE_HCU_QUANT_GROUPED_GEMM_SELECTION_INVALID_CONTRACT_V1 = 0,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_SELECTION_TENSOR_OR_ROW_V1    = 1,
    CK_TILE_HCU_QUANT_GROUPED_GEMM_SELECTION_BLOCKWISE_V1        = 2
} ck_tile_hcu_quant_grouped_gemm_selection_reason_v1;

typedef struct ck_tile_hcu_quant_grouped_gemm_dimension_v1
{
    uint32_t kind;
    uint32_t reserved;
    int64_t extent;
} ck_tile_hcu_quant_grouped_gemm_dimension_v1;

typedef struct ck_tile_hcu_quant_grouped_gemm_problem_v1
{
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t a_data_type;
    int32_t b_data_type;
    int32_t c_data_type;
    int32_t layout;
    int32_t quant_mode;
    int32_t group_count;
    uint32_t problem_flags;
    int32_t k_batch;
    ck_tile_hcu_quant_grouped_gemm_dimension_v1 m;
    ck_tile_hcu_quant_grouped_gemm_dimension_v1 n;
    ck_tile_hcu_quant_grouped_gemm_dimension_v1 k;
    int64_t qk_a;
    int64_t qk_b;
    int64_t stride_aq;
    int64_t stride_bq;
    uint64_t reserved[4];
} ck_tile_hcu_quant_grouped_gemm_problem_v1;

typedef struct ck_tile_hcu_quant_grouped_gemm_selection_v1
{
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t instance_id;
    int32_t reason;
    char instance_name[CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_NAME_CAPACITY];
    uint64_t reserved[4];
} ck_tile_hcu_quant_grouped_gemm_selection_v1;

typedef struct ck_tile_hcu_quant_grouped_gemm_launch_v1
{
    uint32_t struct_size;
    uint32_t abi_version;
    const void* device_args;
    uint32_t device_args_kind;
    uint32_t device_args_stride_bytes;
    uint32_t num_cu;
    uint32_t expected_instance_id;
    void* stream;
    uint64_t reserved[4];
} ck_tile_hcu_quant_grouped_gemm_launch_v1;

typedef struct ck_tile_hcu_quant_grouped_gemm_candidate_info_v1
{
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t instance_id;
    uint32_t architecture_mask;
    uint32_t quant_mode_mask;
    char canonical_name[CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_NAME_CAPACITY];
    uint64_t reserved[4];
} ck_tile_hcu_quant_grouped_gemm_candidate_info_v1;

int ck_tile_hcu_quant_grouped_gemm_select_device_args_v1(
    const ck_tile_hcu_quant_grouped_gemm_problem_v1* problem,
    ck_tile_hcu_quant_grouped_gemm_selection_v1* selection);

int ck_tile_hcu_quant_grouped_gemm_run_device_args_v1(
    const ck_tile_hcu_quant_grouped_gemm_problem_v1* problem,
    const ck_tile_hcu_quant_grouped_gemm_launch_v1* launch,
    ck_tile_hcu_quant_grouped_gemm_selection_v1* selection);

size_t ck_tile_hcu_quant_grouped_gemm_device_args_size_v1(uint32_t device_args_kind);

const char* ck_tile_hcu_quant_grouped_gemm_instance_name_v1(uint32_t instance_id);

int ck_tile_hcu_quant_grouped_gemm_query_instance_v1(
    uint32_t instance_id,
    ck_tile_hcu_quant_grouped_gemm_candidate_info_v1* candidate);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CK_TILE_OPS_GEMM_QUANT_GROUPED_GEMM_QUANT_DEVICE_ARGS_H
