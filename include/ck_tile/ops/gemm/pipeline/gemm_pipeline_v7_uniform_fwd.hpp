// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Shared declarations for the v7/v8 uniform pipeline
// (gemm_pipeline_v7_uniform.hpp). The grouped-GEMM kernel header includes
// THIS header instead of the pipeline body: the tile parameters must be a
// COMPLETE type for every TU that may instantiate RunGemmV7Uniform (the HIP
// device pass instantiates the body eagerly), while the pipeline itself is
// still completed only inside the v7/v8 translation units. The struct is a
// trivially-copyable parameter block; parsing it costs the other TUs nothing.
#pragma once

namespace ck_tile {

enum class V7UniformKind
{
    NN,   // A = [M, K], B = [K, N]
    TN,   // A = [K, M], B = [K, N]
};

// Per-tile parameters, as the kernel path derives them from the device kargs.
// Raw pointers and explicit origins: the pipeline builds its own per-lane
// addresses from them (no tensor views, no windows), which is what keeps the
// register budget at the reference kernel's level.
struct V7UniformTileParams
{
    const void* a_ptr  = nullptr;
    const void* b_ptr  = nullptr;
    void* d_ptr        = nullptr;
    index_t m          = 0;   // this group's row count (M)
    index_t n          = 0;   // common N
    index_t k          = 0;   // common K
    index_t a_stride   = 0;   // row pitch of the TRUE A under the active layout
    index_t b_stride   = 0;   // row pitch of the TRUE B (n)
    index_t d_stride   = 0;   // row pitch of D (n)
    index_t m_start    = 0;   // this tile's global M origin (from the group base)
    index_t n_start    = 0;   // this tile's global N origin
};

template <typename Problem, V7UniformKind kKind>
struct GemmPipelineV7Uniform;

}   // namespace ck_tile
