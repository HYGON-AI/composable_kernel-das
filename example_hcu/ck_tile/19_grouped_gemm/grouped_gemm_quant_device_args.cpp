// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm_quant/grouped_gemm_quant_device_args.h"
#include "ck_tile/ops/gemm_quant/grouped_gemm_quant_selector.hpp"
#include "ck_tile/host/device_prop.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

extern "C" std::size_t ck_tile_hcu_quant_grouped_gemm_provider_device_args_size_v1();
extern "C" int ck_tile_hcu_quant_grouped_gemm_provider_run_v1(std::uint32_t,
                                                               std::uint32_t,
                                                               std::uint32_t,
                                                               std::uint32_t,
                                                               std::uint32_t,
                                                               void*,
                                                               int,
                                                               std::uint32_t,
                                                               hipStream_t);

namespace {

using InstanceId = ck_tile::QuantGroupedGemmInstanceId;
static_assert(static_cast<std::uint32_t>(InstanceId::none) ==
              CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_NONE_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::persistent_128x128x32) ==
              CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_PERSISTENT_128X128X32_V1);
static_assert(static_cast<std::uint32_t>(InstanceId::persistent_128x128x128) ==
              CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_PERSISTENT_128X128X128_V1);

constexpr std::uint32_t KnownProblemFlags =
    CK_TILE_HCU_QUANT_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1 |
    CK_TILE_HCU_QUANT_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1;

bool dispatch_trace_enabled()
{
    const char* value = std::getenv("CK_TILE_HCU_QUANT_GROUPED_GEMM_DISPATCH_TRACE");
    return value != nullptr && value[0] != '0';
}

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
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_INVALID_ARGUMENT_V1;
    if(value->struct_size < sizeof(T))
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_BAD_STRUCT_SIZE_V1;
    if(value->abi_version != CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION)
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_BAD_VERSION_V1;
    return CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1;
}

template <std::size_t N>
void copy_name(char (&destination)[N], std::string_view source)
{
    const auto count = std::min(source.size(), N - 1);
    std::memcpy(destination, source.data(), count);
    destination[count] = '\0';
}

template <std::size_t N>
void copy_name(char (&destination)[N], const std::string& source)
{
    copy_name(destination, std::string_view{source});
}

int convert_dimension(const ck_tile_hcu_quant_grouped_gemm_dimension_v1& source,
                      ck_tile::QuantGroupedGemmDimensionSummary& destination)
{
    if(source.reserved != 0 || source.extent <= 0)
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_INVALID_ARGUMENT_V1;
    if(source.kind == CK_TILE_HCU_QUANT_GROUPED_GEMM_DIMENSION_COMMON_V1)
    {
        destination = ck_tile::QuantGroupedGemmDimensionSummary::Common(source.extent);
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1;
    }
    if(source.kind ==
       CK_TILE_HCU_QUANT_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1)
    {
        destination =
            ck_tile::QuantGroupedGemmDimensionSummary::DeviceLengthsWithCapacity(source.extent);
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1;
    }
    return CK_TILE_HCU_QUANT_GROUPED_GEMM_INVALID_ARGUMENT_V1;
}

int convert_problem(const ck_tile_hcu_quant_grouped_gemm_problem_v1* source,
                    ck_tile::QuantGroupedGemmProblem& problem)
{
    int status = validate_versioned_header(source);
    if(status != 0)
        return status;
    if(!all_zero(source->reserved, 4) ||
       (source->problem_flags & ~KnownProblemFlags) != 0)
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_INVALID_ARGUMENT_V1;

    const auto target = ck_tile::get_hcu_target_enum();
    if(target != ck_tile::hcu_target_enum::gfx938 &&
       target != ck_tile::hcu_target_enum::gfx946)
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_ARCH_V1;
    problem.arch = target == ck_tile::hcu_target_enum::gfx938
                       ? ck_tile::QuantGroupedGemmArch::gfx938
                       : ck_tile::QuantGroupedGemmArch::gfx946;

    const auto convert_input = [](std::int32_t type) {
        return type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E4M3_V1
                   ? ck_tile::QuantGroupedGemmDataType::fp8_e4m3
               : type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP8_E5M2_V1
                   ? ck_tile::QuantGroupedGemmDataType::fp8_e5m2
                   : ck_tile::QuantGroupedGemmDataType::unsupported;
    };
    problem.a_data_type = convert_input(source->a_data_type);
    problem.b_data_type = convert_input(source->b_data_type);
    problem.c_data_type =
        source->c_data_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_FP16_V1
            ? ck_tile::QuantGroupedGemmDataType::fp16
        : source->c_data_type == CK_TILE_HCU_QUANT_GROUPED_GEMM_DATA_TYPE_BF16_V1
            ? ck_tile::QuantGroupedGemmDataType::bf16
            : ck_tile::QuantGroupedGemmDataType::unsupported;
    problem.layout = source->layout == CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_NN_V1
                         ? ck_tile::QuantGroupedGemmLayout::nn
                     : source->layout == CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_NT_V1
                         ? ck_tile::QuantGroupedGemmLayout::nt
                     : source->layout == CK_TILE_HCU_QUANT_GROUPED_GEMM_LAYOUT_TN_V1
                         ? ck_tile::QuantGroupedGemmLayout::tn
                         : ck_tile::QuantGroupedGemmLayout::unsupported;
    problem.quant_mode =
        source->quant_mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_TENSORWISE_V1
            ? ck_tile::QuantGroupedGemmMode::tensorwise
        : source->quant_mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_ROWWISE_V1
            ? ck_tile::QuantGroupedGemmMode::rowwise
        : source->quant_mode == CK_TILE_HCU_QUANT_GROUPED_GEMM_MODE_BLOCKWISE_V1
            ? ck_tile::QuantGroupedGemmMode::blockwise
            : ck_tile::QuantGroupedGemmMode::unsupported;
    problem.group_count = source->group_count;
    problem.k_batch     = source->k_batch;
    problem.qk_a        = source->qk_a;
    problem.qk_b        = source->qk_b;
    problem.stride_aq   = source->stride_aq;
    problem.stride_bq   = source->stride_bq;
    problem.may_have_empty_groups =
        (source->problem_flags &
         CK_TILE_HCU_QUANT_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1) != 0;
    problem.group_lengths_are_ragged =
        (source->problem_flags &
         CK_TILE_HCU_QUANT_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1) != 0;
    if((status = convert_dimension(source->m, problem.m)) != 0 ||
       (status = convert_dimension(source->n, problem.n)) != 0 ||
       (status = convert_dimension(source->k, problem.k)) != 0)
        return status;

    return ck_tile::is_valid_quant_grouped_gemm_problem(problem)
               ? CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1
               : CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_V1;
}

int write_selection(const ck_tile::QuantGroupedGemmProblem& problem,
                    const ck_tile::QuantGroupedGemmSelection& selected,
                    ck_tile_hcu_quant_grouped_gemm_selection_v1* output)
{
    const int status = validate_versioned_header(output);
    if(status != 0)
        return status;
    const std::uint32_t output_size = output->struct_size;
    *output                         = {};
    output->struct_size             = output_size;
    output->abi_version = CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    output->instance_id = static_cast<std::uint32_t>(selected.instance_id);
    output->reason      = static_cast<std::int32_t>(selected.reason);
    copy_name(output->instance_name,
              ck_tile::stable_quant_grouped_gemm_instance_id(problem, selected));
    return CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1;
}

} // namespace

extern "C" int ck_tile_hcu_quant_grouped_gemm_select_device_args_v1(
    const ck_tile_hcu_quant_grouped_gemm_problem_v1* source,
    ck_tile_hcu_quant_grouped_gemm_selection_v1* output)
{
    try
    {
        ck_tile::QuantGroupedGemmProblem problem;
        const int status = convert_problem(source, problem);
        if(status != 0)
            return status;
        const auto selected = ck_tile::select_quant_grouped_gemm_candidate(problem);
        if(!selected.supported())
            return CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_V1;
        return write_selection(problem, selected, output);
    }
    catch(...)
    {
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_INTERNAL_ERROR_V1;
    }
}

extern "C" int ck_tile_hcu_quant_grouped_gemm_run_device_args_v1(
    const ck_tile_hcu_quant_grouped_gemm_problem_v1* source,
    const ck_tile_hcu_quant_grouped_gemm_launch_v1* launch,
    ck_tile_hcu_quant_grouped_gemm_selection_v1* output)
{
    try
    {
        int status = validate_versioned_header(launch);
        if(status != 0)
            return status;
        if(launch->device_args == nullptr || !all_zero(launch->reserved, 4) ||
           launch->device_args_kind !=
               CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_KIND_QUANT_GEMM_TRANS_V1 ||
           launch->device_args_stride_bytes !=
               ck_tile_hcu_quant_grouped_gemm_provider_device_args_size_v1())
            return CK_TILE_HCU_QUANT_GROUPED_GEMM_BAD_DEVICE_ARGS_V1;

        ck_tile::QuantGroupedGemmProblem problem;
        status = convert_problem(source, problem);
        if(status != 0)
            return status;
        const auto selected = ck_tile::select_quant_grouped_gemm_candidate(problem);
        if(!selected.supported())
            return CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_V1;
        if(launch->expected_instance_id != 0 &&
           launch->expected_instance_id != static_cast<std::uint32_t>(selected.instance_id))
            return CK_TILE_HCU_QUANT_GROUPED_GEMM_INSTANCE_MISMATCH_V1;
        if(output != nullptr && (status = write_selection(problem, selected, output)) != 0)
            return status;

        status = ck_tile_hcu_quant_grouped_gemm_provider_run_v1(
            static_cast<std::uint32_t>(source->a_data_type),
            static_cast<std::uint32_t>(source->b_data_type),
            static_cast<std::uint32_t>(source->c_data_type),
            static_cast<std::uint32_t>(source->layout),
            static_cast<std::uint32_t>(source->quant_mode),
            const_cast<void*>(launch->device_args),
            problem.group_count,
            launch->num_cu,
            reinterpret_cast<hipStream_t>(launch->stream));
        if(dispatch_trace_enabled() || status != CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1)
        {
            std::fprintf(stderr,
                         "CK_QUANT_GROUPED_GEMM_DISPATCH status=%d instance_id=%u "
                         "instance_name=%s a_dtype=%d b_dtype=%d c_dtype=%d layout=%d "
                         "quant_mode=%d group_count=%d\n",
                         status,
                         static_cast<std::uint32_t>(selected.instance_id),
                         ck_tile::stable_quant_grouped_gemm_instance_id(problem, selected).c_str(),
                         source->a_data_type,
                         source->b_data_type,
                         source->c_data_type,
                         source->layout,
                         source->quant_mode,
                         source->group_count);
        }
        return status;
    }
    catch(...)
    {
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_INTERNAL_ERROR_V1;
    }
}

extern "C" std::size_t
ck_tile_hcu_quant_grouped_gemm_device_args_size_v1(std::uint32_t device_args_kind)
{
    return device_args_kind ==
                   CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_KIND_QUANT_GEMM_TRANS_V1
               ? ck_tile_hcu_quant_grouped_gemm_provider_device_args_size_v1()
               : 0;
}

extern "C" const char*
ck_tile_hcu_quant_grouped_gemm_instance_name_v1(std::uint32_t instance_id)
{
    const auto* candidate = ck_tile::find_quant_grouped_gemm_candidate(
        static_cast<ck_tile::QuantGroupedGemmInstanceId>(instance_id));
    return candidate == nullptr ? nullptr : candidate->canonical_suffix.data();
}

extern "C" int ck_tile_hcu_quant_grouped_gemm_query_instance_v1(
    std::uint32_t instance_id,
    ck_tile_hcu_quant_grouped_gemm_candidate_info_v1* output)
{
    try
    {
        const int status = validate_versioned_header(output);
        if(status != 0)
            return status;
        const auto id = static_cast<ck_tile::QuantGroupedGemmInstanceId>(instance_id);
        const auto* candidate = ck_tile::find_quant_grouped_gemm_candidate(id);
        if(candidate == nullptr)
            return CK_TILE_HCU_QUANT_GROUPED_GEMM_UNSUPPORTED_V1;
        const std::uint32_t output_size = output->struct_size;
        *output                         = {};
        output->struct_size             = output_size;
        output->abi_version = CK_TILE_HCU_QUANT_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
        output->instance_id = instance_id;
        output->architecture_mask =
            CK_TILE_HCU_QUANT_GROUPED_GEMM_ARCHITECTURE_MASK_GFX938_V1 |
            CK_TILE_HCU_QUANT_GROUPED_GEMM_ARCHITECTURE_MASK_GFX946_V1;
        output->quant_mode_mask = candidate->quant_mode_mask;
        copy_name(output->canonical_name, candidate->canonical_suffix);
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_SUCCESS_V1;
    }
    catch(...)
    {
        return CK_TILE_HCU_QUANT_GROUPED_GEMM_INTERNAL_ERROR_V1;
    }
}
