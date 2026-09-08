// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm_quant/grouped_gemm_quant_selector.hpp"

#include <array>
#include <cstdint>
#include <iostream>

namespace {

using Arch   = ck_tile::QuantGroupedGemmArch;
using DType  = ck_tile::QuantGroupedGemmDataType;
using Layout = ck_tile::QuantGroupedGemmLayout;
using Mode   = ck_tile::QuantGroupedGemmMode;
using Id     = ck_tile::QuantGroupedGemmInstanceId;
using Dim    = ck_tile::QuantGroupedGemmDimensionSummary;

struct Case
{
    const char* name;
    int groups;
    std::int64_t m;
    std::int64_t n;
    std::int64_t k;
    DType a;
    DType b;
    DType c;
    Layout layout;
    Mode mode;
    bool variable_m;
    bool variable_k;
    Id expected;
};

constexpr std::array<Case, 15> cases{{
    {"report_tensor_b32", 32, 128, 3072, 5120, DType::fp8_e4m3, DType::fp8_e4m3,
     DType::bf16, Layout::tn, Mode::tensorwise, false, true, Id::persistent_128x128x32},
    {"report_tensor_b8_hybrid", 8, 1024, 3072, 5120, DType::fp8_e4m3,
     DType::fp8_e5m2, DType::fp16, Layout::tn, Mode::tensorwise, false, true,
     Id::persistent_128x128x32},
    {"report_row_b16_nt", 16, 256, 4096, 7168, DType::fp8_e4m3, DType::fp8_e4m3,
     DType::bf16, Layout::nt, Mode::rowwise, true, false, Id::persistent_128x128x32},
    {"report_row_b16_tn", 16, 2048, 7168, 2048, DType::fp8_e5m2, DType::fp8_e5m2,
     DType::bf16, Layout::tn, Mode::rowwise, false, true, Id::persistent_128x128x32},
    {"report_qt_b32", 32, 128, 3072, 5120, DType::fp8_e4m3, DType::fp8_e4m3,
     DType::bf16, Layout::tn, Mode::rowwise, false, true, Id::persistent_128x128x32},
    {"report_qt_b16", 16, 128, 1408, 2048, DType::fp8_e4m3, DType::fp8_e4m3,
     DType::bf16, Layout::tn, Mode::rowwise, false, true, Id::persistent_128x128x32},
    {"report_qt_b8", 8, 128, 4096, 7168, DType::fp8_e5m2, DType::fp8_e5m2,
     DType::bf16, Layout::tn, Mode::rowwise, false, true, Id::persistent_128x128x32},
    {"control_tensor_b16", 16, 1024, 3072, 5120, DType::fp8_e4m3,
     DType::fp8_e5m2, DType::fp16, Layout::tn, Mode::tensorwise, false, true,
     Id::persistent_128x128x32},
    {"control_row_b32", 32, 256, 4096, 7168, DType::fp8_e4m3, DType::fp8_e4m3,
     DType::bf16, Layout::nt, Mode::rowwise, true, false, Id::persistent_128x128x32},
    {"control_qt_b32", 32, 128, 1408, 2048, DType::fp8_e4m3, DType::fp8_e4m3,
     DType::bf16, Layout::tn, Mode::rowwise, false, true, Id::persistent_128x128x32},
    {"blockwise_nn", 4, 128, 256, 512, DType::fp8_e4m3, DType::fp8_e4m3,
     DType::fp16, Layout::nn, Mode::blockwise, true, false,
     Id::persistent_128x128x128},
    {"blockwise_nt", 4, 128, 256, 512, DType::fp8_e5m2, DType::fp8_e4m3,
     DType::bf16, Layout::nt, Mode::blockwise, true, false,
     Id::persistent_128x128x128},
    {"tensor_nn", 3, 129, 257, 513, DType::fp8_e4m3, DType::fp8_e5m2,
     DType::fp16, Layout::nn, Mode::tensorwise, true, false,
     Id::persistent_128x128x32},
    {"row_nt", 5, 255, 511, 1025, DType::fp8_e5m2, DType::fp8_e4m3,
     DType::bf16, Layout::nt, Mode::rowwise, true, false,
     Id::persistent_128x128x32},
    {"blockwise_tn", 8, 128, 4096, 7168, DType::fp8_e4m3, DType::fp8_e5m2,
     DType::bf16, Layout::tn, Mode::blockwise, false, true,
     Id::persistent_128x128x128},
}};

ck_tile::QuantGroupedGemmProblem make_problem(const Case& spec)
{
    ck_tile::QuantGroupedGemmProblem problem{};
    problem.arch            = Arch::gfx938;
    problem.a_data_type     = spec.a;
    problem.b_data_type     = spec.b;
    problem.c_data_type     = spec.c;
    problem.layout          = spec.layout;
    problem.quant_mode      = spec.mode;
    problem.group_count     = spec.groups;
    problem.m               = spec.variable_m ? Dim::DeviceLengthsWithCapacity(spec.m * spec.groups)
                                                : Dim::Common(spec.m);
    problem.n               = Dim::Common(spec.n);
    problem.k               = spec.variable_k ? Dim::DeviceLengthsWithCapacity(spec.k)
                                                : Dim::Common(spec.k);
    problem.qk_a            = spec.mode == Mode::blockwise ? (spec.k + 127) / 128 : 1;
    problem.qk_b            = problem.qk_a;
    problem.stride_aq       = problem.qk_a;
    problem.stride_bq       = problem.qk_b;
    problem.k_batch         = 1;
    problem.may_have_empty_groups = spec.variable_m || spec.variable_k;
    problem.group_lengths_are_ragged = spec.variable_m || spec.variable_k;
    return problem;
}

} // namespace

int main()
{
    int failures = 0;
    for(const auto& spec : cases)
    {
        const auto problem   = make_problem(spec);
        const auto selection = ck_tile::select_quant_grouped_gemm_candidate(problem);
        const auto stable    = ck_tile::stable_quant_grouped_gemm_instance_id(problem, selection);
        const bool pass      = selection.instance_id == spec.expected && !stable.empty();
        std::cout << spec.name << " instance=" << static_cast<int>(selection.instance_id)
                  << " stable=" << stable << " pass=" << pass << '\n';
        failures += pass ? 0 : 1;
    }

    auto gfx946_problem      = make_problem(cases.front());
    gfx946_problem.arch      = Arch::gfx946;
    const auto gfx946_select = ck_tile::select_quant_grouped_gemm_candidate(gfx946_problem);
    const auto gfx946_stable =
        ck_tile::stable_quant_grouped_gemm_instance_id(gfx946_problem, gfx946_select);
    const bool gfx946_pass =
        gfx946_select.instance_id == Id::persistent_128x128x32 &&
        gfx946_stable.rfind("ck.qgg.gfx946.", 0) == 0;
    std::cout << "gfx946_control instance=" << static_cast<int>(gfx946_select.instance_id)
              << " stable=" << gfx946_stable << " pass=" << gfx946_pass << '\n';
    failures += gfx946_pass ? 0 : 1;

    auto invalid        = make_problem(cases.front());
    invalid.group_count = 0;
    failures += ck_tile::select_quant_grouped_gemm_candidate(invalid).supported() ? 1 : 0;
    invalid                   = make_problem(cases.front());
    invalid.arch              = Arch::unsupported;
    failures += ck_tile::select_quant_grouped_gemm_candidate(invalid).supported() ? 1 : 0;
    invalid                   = make_problem(cases.front());
    invalid.c_data_type       = DType::fp8_e4m3;
    failures += ck_tile::select_quant_grouped_gemm_candidate(invalid).supported() ? 1 : 0;
    invalid                   = make_problem(cases.front());
    invalid.k_batch           = 2;
    failures += ck_tile::select_quant_grouped_gemm_candidate(invalid).supported() ? 1 : 0;
    invalid                   = make_problem(cases.front());
    invalid.m.kind            = ck_tile::QuantGroupedGemmDimensionKind::device_lengths_with_capacity;
    invalid.k.kind            = ck_tile::QuantGroupedGemmDimensionKind::device_lengths_with_capacity;
    failures += ck_tile::select_quant_grouped_gemm_candidate(invalid).supported() ? 1 : 0;
    invalid             = make_problem(cases[10]);
    invalid.qk_a        = 1;
    invalid.stride_aq   = 1;
    failures += ck_tile::select_quant_grouped_gemm_candidate(invalid).supported() ? 1 : 0;

    std::cout << "quant grouped GEMM selector failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
