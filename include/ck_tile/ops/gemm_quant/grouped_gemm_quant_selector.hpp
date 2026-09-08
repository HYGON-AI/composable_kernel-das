// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace ck_tile {

enum class QuantGroupedGemmArch
{
    unsupported,
    gfx938,
    gfx946,
};

enum class QuantGroupedGemmDataType
{
    unsupported,
    fp8_e4m3,
    fp8_e5m2,
    fp16,
    bf16,
};

enum class QuantGroupedGemmLayout
{
    unsupported,
    nn,
    nt,
    tn,
};

enum class QuantGroupedGemmMode
{
    unsupported,
    tensorwise,
    rowwise,
    blockwise,
};

enum class QuantGroupedGemmDimensionKind
{
    common,
    device_lengths_with_capacity,
};

struct QuantGroupedGemmDimensionSummary
{
    QuantGroupedGemmDimensionKind kind = QuantGroupedGemmDimensionKind::common;
    std::int64_t extent                = 0;

    static constexpr QuantGroupedGemmDimensionSummary Common(std::int64_t value)
    {
        return {QuantGroupedGemmDimensionKind::common, value};
    }

    static constexpr QuantGroupedGemmDimensionSummary DeviceLengthsWithCapacity(
        std::int64_t capacity)
    {
        return {QuantGroupedGemmDimensionKind::device_lengths_with_capacity, capacity};
    }
};

struct QuantGroupedGemmProblem
{
    QuantGroupedGemmArch arch                 = QuantGroupedGemmArch::unsupported;
    QuantGroupedGemmDataType a_data_type      = QuantGroupedGemmDataType::unsupported;
    QuantGroupedGemmDataType b_data_type      = QuantGroupedGemmDataType::unsupported;
    QuantGroupedGemmDataType c_data_type      = QuantGroupedGemmDataType::unsupported;
    QuantGroupedGemmLayout layout             = QuantGroupedGemmLayout::unsupported;
    QuantGroupedGemmMode quant_mode           = QuantGroupedGemmMode::unsupported;
    std::int32_t group_count                  = 0;
    QuantGroupedGemmDimensionSummary m        = {};
    QuantGroupedGemmDimensionSummary n        = {};
    QuantGroupedGemmDimensionSummary k        = {};
    std::int64_t qk_a                         = 0;
    std::int64_t qk_b                         = 0;
    std::int64_t stride_aq                    = 0;
    std::int64_t stride_bq                    = 0;
    std::int32_t k_batch                      = 1;
    bool may_have_empty_groups                = false;
    bool group_lengths_are_ragged             = false;
};

enum class QuantGroupedGemmInstanceId
{
    none                         = 0,
    persistent_128x128x32        = 5000,
    persistent_128x128x128       = 5001,
};

enum class QuantGroupedGemmSelectionReason
{
    invalid_contract,
    tensor_or_row,
    blockwise,
};

struct QuantGroupedGemmCandidateIdentity
{
    QuantGroupedGemmInstanceId id;
    std::string_view canonical_suffix;
    std::uint32_t quant_mode_mask;
};

inline constexpr std::array<QuantGroupedGemmCandidateIdentity, 2>
    quant_grouped_gemm_candidate_registry{{
        {QuantGroupedGemmInstanceId::persistent_128x128x32,
         "persistent.128x128x32",
         (1u << 0) | (1u << 1)},
        {QuantGroupedGemmInstanceId::persistent_128x128x128,
         "persistent.128x128x128",
         1u << 2},
    }};

constexpr const QuantGroupedGemmCandidateIdentity*
find_quant_grouped_gemm_candidate(QuantGroupedGemmInstanceId id)
{
    for(const auto& candidate : quant_grouped_gemm_candidate_registry)
    {
        if(candidate.id == id)
            return &candidate;
    }
    return nullptr;
}

struct QuantGroupedGemmSelection
{
    QuantGroupedGemmInstanceId instance_id = QuantGroupedGemmInstanceId::none;
    QuantGroupedGemmSelectionReason reason  = QuantGroupedGemmSelectionReason::invalid_contract;

    constexpr bool supported() const
    {
        return instance_id != QuantGroupedGemmInstanceId::none;
    }
};

constexpr bool is_quant_grouped_gemm_input_type(QuantGroupedGemmDataType type)
{
    return type == QuantGroupedGemmDataType::fp8_e4m3 ||
           type == QuantGroupedGemmDataType::fp8_e5m2;
}

constexpr bool is_quant_grouped_gemm_output_type(QuantGroupedGemmDataType type)
{
    return type == QuantGroupedGemmDataType::fp16 || type == QuantGroupedGemmDataType::bf16;
}

constexpr bool is_valid_quant_grouped_gemm_problem(const QuantGroupedGemmProblem& problem)
{
    if((problem.arch != QuantGroupedGemmArch::gfx938 &&
        problem.arch != QuantGroupedGemmArch::gfx946) ||
       !is_quant_grouped_gemm_input_type(problem.a_data_type) ||
       !is_quant_grouped_gemm_input_type(problem.b_data_type) ||
       !is_quant_grouped_gemm_output_type(problem.c_data_type) ||
       (problem.layout != QuantGroupedGemmLayout::nn &&
        problem.layout != QuantGroupedGemmLayout::nt &&
        problem.layout != QuantGroupedGemmLayout::tn) ||
       (problem.quant_mode != QuantGroupedGemmMode::tensorwise &&
        problem.quant_mode != QuantGroupedGemmMode::rowwise &&
        problem.quant_mode != QuantGroupedGemmMode::blockwise) ||
       problem.group_count <= 0 || problem.k_batch != 1 || problem.m.extent <= 0 ||
       problem.n.extent <= 0 || problem.k.extent <= 0 ||
       problem.n.kind != QuantGroupedGemmDimensionKind::common || problem.qk_a <= 0 ||
       problem.qk_b <= 0 || problem.stride_aq <= 0 || problem.stride_bq <= 0)
    {
        return false;
    }

    const int variable_dimensions =
        (problem.m.kind == QuantGroupedGemmDimensionKind::device_lengths_with_capacity ? 1 : 0) +
        (problem.k.kind == QuantGroupedGemmDimensionKind::device_lengths_with_capacity ? 1 : 0);
    if(variable_dimensions > 1)
        return false;

    if(problem.quant_mode == QuantGroupedGemmMode::tensorwise ||
       problem.quant_mode == QuantGroupedGemmMode::rowwise)
    {
        return problem.qk_a == 1 && problem.qk_b == 1 && problem.stride_aq == 1 &&
               problem.stride_bq == 1;
    }

    const std::int64_t quant_k = (problem.k.extent + 127) / 128;
    return problem.qk_a == quant_k && problem.qk_b == quant_k &&
           problem.stride_aq == quant_k && problem.stride_bq == quant_k;
}

constexpr QuantGroupedGemmSelection
select_quant_grouped_gemm_candidate(const QuantGroupedGemmProblem& problem)
{
    if(!is_valid_quant_grouped_gemm_problem(problem))
        return {};
    if(problem.quant_mode == QuantGroupedGemmMode::blockwise)
    {
        return {QuantGroupedGemmInstanceId::persistent_128x128x128,
                QuantGroupedGemmSelectionReason::blockwise};
    }
    return {QuantGroupedGemmInstanceId::persistent_128x128x32,
            QuantGroupedGemmSelectionReason::tensor_or_row};
}

constexpr std::string_view quant_grouped_gemm_arch_name(QuantGroupedGemmArch arch)
{
    return arch == QuantGroupedGemmArch::gfx938 ? "gfx938"
           : arch == QuantGroupedGemmArch::gfx946 ? "gfx946"
                                                   : "unsupported";
}

constexpr std::string_view quant_grouped_gemm_data_type_name(QuantGroupedGemmDataType type)
{
    return type == QuantGroupedGemmDataType::fp8_e4m3 ? "fp8_e4m3"
           : type == QuantGroupedGemmDataType::fp8_e5m2 ? "fp8_e5m2"
           : type == QuantGroupedGemmDataType::fp16 ? "fp16"
           : type == QuantGroupedGemmDataType::bf16 ? "bf16"
                                                     : "unsupported";
}

constexpr std::string_view quant_grouped_gemm_layout_name(QuantGroupedGemmLayout layout)
{
    return layout == QuantGroupedGemmLayout::nn ? "NN"
           : layout == QuantGroupedGemmLayout::nt ? "NT"
           : layout == QuantGroupedGemmLayout::tn ? "TN"
                                                   : "unsupported";
}

constexpr std::string_view quant_grouped_gemm_mode_name(QuantGroupedGemmMode mode)
{
    return mode == QuantGroupedGemmMode::tensorwise ? "tensorwise"
           : mode == QuantGroupedGemmMode::rowwise ? "rowwise"
           : mode == QuantGroupedGemmMode::blockwise ? "blockwise"
                                                      : "unsupported";
}

inline std::string stable_quant_grouped_gemm_instance_id(
    const QuantGroupedGemmProblem& problem,
    const QuantGroupedGemmSelection& selection)
{
    const auto* candidate = find_quant_grouped_gemm_candidate(selection.instance_id);
    if(candidate == nullptr)
        return {};

    std::string id{"ck.qgg."};
    id.append(quant_grouped_gemm_arch_name(problem.arch));
    id.push_back('.');
    id.append(quant_grouped_gemm_data_type_name(problem.a_data_type));
    id.push_back('x');
    id.append(quant_grouped_gemm_data_type_name(problem.b_data_type));
    id.append("_to_");
    id.append(quant_grouped_gemm_data_type_name(problem.c_data_type));
    id.push_back('.');
    id.append(quant_grouped_gemm_layout_name(problem.layout));
    id.push_back('.');
    id.append(quant_grouped_gemm_mode_name(problem.quant_mode));
    id.push_back('.');
    id.append(candidate->canonical_suffix);
    return id;
}

} // namespace ck_tile
