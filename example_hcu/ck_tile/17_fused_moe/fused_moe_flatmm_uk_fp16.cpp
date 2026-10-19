// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Ltd.
// Modified by Hygon Information Technology Co., Ltd.
//
// Example/validation harness for the ported FusedMoeGemmPipeline_FlatmmUk
// (HCU MMAC engine). Runs a 2-GEMM fused MoE (gate+up, silu*mul bridge,
// down-proj with topk-weighted fp32 atomic accumulation) and checks against
// a fp32 CPU reference.
//
// Weight layout (natural, no host pre-shuffle), same as fork's
// FusedMoeGemmKernel: W1/G[e] = [2*interm, hidden] row-major (gate|up),
// W2/D[e] = [hidden, interm] row-major. sorted_token_ids holds raw token
// ids (default, enc=1); padding slots use the num_tokens sentinel, weight 0.
// enc=2 packs ids the way the repo sorting kernel does under
// CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID (token | (topk_idx << 24)).
//
// args: tokens experts topk hidden interm [v=1] [warmup=5] [repeat=20]
//       [init=0|1] [seed] [enc=1|0|2] [shuf=0]
//
// hidden is auto-padded up to the gmem load-vector alignment (16B = 8 fp16
// here): the engine's A/G/U loads mask out-of-range accesses per whole
// vector, so an unpadded hidden would read across row ends. The pad region
// is zero on both the device buffers and the reference, and only logical
// columns are validated. intermediate_size has no such constraint (the D
// loads are guarded element-wise).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_problem.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_traits.hpp"
#include "ck_tile/ops/fused_moe/kernel/fused_moegemm_shape.hpp"
#include "ck_tile/ops/fused_moe/kernel/fused_moegemm_tile_partitioner.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_uk.hpp"
#include "ck_tile/ops/fused_moe/kernel/fused_moegemm_uk_kernel.hpp"

// ---------------------------------------------------------------- utils
static uint64_t lcg_state;
static void lcg_seed(uint64_t s) { lcg_state = s ? s : 1; }
static float lcg_float() // [-0.5, 0.5)
{
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<int32_t>(lcg_state >> 41) & 0x7fffff) /
               static_cast<float>(0x800000) -
           0.5f;
}

// Host fp16 conversions are done with these inlined integer-only helpers
// instead of _Float16 casts: the DTK aicc -c (object) offload-bundle path
// miscompiles the cast call sites in this TU (garbage arguments reach
// __truncsfhf2/__extendhfsf2: device buffers get wrong bits and the CPU
// reference silently reads zeros, so validation degrades to a vacuous PASS;
// see the "sanity:" line below). The bit math is immune, matches the device
// v_cvt rounding (round-to-nearest-even), and mirrors the software fallback
// in ck_tile/core/numeric/half.hpp. Device code is unaffected.
static inline uint16_t f32_to_f16_bits(float x)
{
    uint32_t f;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    const int e         = static_cast<int>((f >> 23) & 0xffu);
    uint32_t m          = f & 0x7fffffu;
    if(e == 255)
        return static_cast<uint16_t>(sign | 0x7c00u | (m ? 1u : 0u)); // inf/nan
    int ee = e - 127 + 15;
    if(ee >= 31)
        return static_cast<uint16_t>(sign | 0x7c00u); // overflow -> inf
    if(ee <= 0)
    {
        if(ee < -10)
            return static_cast<uint16_t>(sign); // underflow -> zero
        m |= 0x800000u;
        const int shift     = 14 - ee;
        const uint32_t half = 1u << (shift - 1);
        const uint32_t rem  = m & ((1u << shift) - 1);
        m >>= shift;
        if(rem > half || (rem == half && (m & 1u)))
            m++; // RNE
        if(m & 0x400u)
            return static_cast<uint16_t>(sign | (1u << 10)); // carried to min normal
        return static_cast<uint16_t>(sign | m);
    }
    const uint32_t half = 1u << 12;
    const uint32_t rem  = m & 0x1fffu;
    m >>= 13;
    if(rem > half || (rem == half && (m & 1u)))
    {
        m++;
        if(m == 0x400u)
        {
            m = 0;
            ee++;
            if(ee >= 31)
                return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(ee) << 10) | m);
}

static inline float f16_bits_to_f32(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const int e         = (h >> 10) & 0x1f;
    uint32_t m          = h & 0x3ffu;
    uint32_t f;
    if(e == 0)
    {
        if(m == 0)
            f = sign;
        else
        {
            int ee = -14;
            while(!(m & 0x400u))
            {
                m <<= 1;
                ee--;
            }
            m &= 0x3ffu;
            f = sign | (static_cast<uint32_t>(ee + 127) << 23) | (m << 13);
        }
    }
    else if(e == 31)
        f = sign | 0x7f800000u | (m << 13);
    else
        f = sign | (static_cast<uint32_t>(e - 15 + 127) << 23) | (m << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

static inline ck_tile::fp16_t h_fp16(float x)
{
    const uint16_t bits = f32_to_f16_bits(x);
    ck_tile::fp16_t h;
    std::memcpy(&h, &bits, 2);
    return h;
}

static inline float h_f32(ck_tile::fp16_t h)
{
    uint16_t bits;
    std::memcpy(&bits, &h, 2);
    return f16_bits_to_f32(bits);
}

#define HIP_CHECK(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        hipError_t e = (cmd);                                                            \
        if(e != hipSuccess)                                                              \
        {                                                                                \
            printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
            exit(1);                                                                     \
        }                                                                                \
    } while(0)

// ---------------------------------------------------------------- types
template <ck_tile::index_t... Is>
using S = ck_tile::sequence<Is...>;

// Block: gemm0 32x256x32 (warp 32x64x32 x 4), gemm1 32x64x256 (warp 32x16x256 x 4)
// config from the fork's fused_moe setup (flatmm_ex engine): O=fp32 + OAtomic=2,
// Gemm0NInterleave=4, swizzled=false, PadHiddenSize=true
using FmoeShape = ck_tile::FusedMoeGemmShape<S<32, 256, 32>,
                                             S<1, 4, 1>,
                                             S<32, 64, 32>,
                                             S<32, 64, 256>,
                                             S<1, 4, 1>,
                                             S<32, 16, 256>>;

using FmoeTraits = ck_tile::FusedMoeGemmTraits<false,               // gate+up
                                               false,               // smooth quant
                                               2,                   // fp32 atomic
                                               S<2, 4, 4, 4>,       // gmem load vec
                                               S<4>,                // bridge smem store vec
                                               S<8, 8, 8, 4>,       // smem load vec
                                               1,                   // o gmem store vec
                                               4,                   // gemm0 interleave
                                               1,                   // gemm1 interleave
                                               ck_tile::FusedMoeGemmWeightPermuteEnum::no_permute,
                                               false,  // swizzled
                                               true,   // pad hidden size
                                               false>; // pad intermediate size

using FmoeProblem = ck_tile::FusedMoeGemmPipelineProblem<ck_tile::fp16_t, // A
                                                         ck_tile::fp16_t, // G
                                                         ck_tile::fp16_t, // D
                                                         float,           // acc
                                                         float,           // O (fp32 atomic)
                                                         float,
                                                         float,
                                                         float,
                                                         float,
                                                         float, // topk weight
                                                         int32_t,
                                                         ck_tile::element_wise::Silu,
                                                         FmoeShape,
                                                         FmoeTraits>;

using FmoePipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<FmoeProblem>;
using FmoePartitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<FmoeShape>;
using FmoeKernel      = ck_tile::FusedMoeGemmUKKernel<FmoePartitioner, FmoePipeline, void>;

constexpr int BlockM0 = 32;

// ---------------------------------------------------------------- host sort
struct MoeSortResult
{
    std::vector<int32_t> sorted_token_ids; // base-offset encoding
    std::vector<float> sorted_weight;
    std::vector<int32_t> sorted_expert_ids; // per BlockM0 tile
    int32_t num_sorted_tiles;               // in tiles of BlockM0
    int padded_rows = 0;
};

static MoeSortResult host_sort(const std::vector<std::vector<int>>& topk_ids,
                               const std::vector<std::vector<float>>& topk_w,
                               int tokens,
                               int experts,
                               int topk,
                               int enc_mode)
{
    std::vector<std::vector<int>> rows_of_expert(experts);
    std::vector<std::vector<float>> w_of_expert(experts);
    std::vector<std::vector<int>> kidx_of_expert(experts); // topk slot, for enc=2
    for(int t = 0; t < tokens; t++)
        for(int k = 0; k < topk; k++)
        {
            int e = topk_ids[t][k];
            rows_of_expert[e].push_back(t);
            w_of_expert[e].push_back(topk_w[t][k]);
            kidx_of_expert[e].push_back(k);
        }

    MoeSortResult r;
    for(int e = 0; e < experts; e++)
    {
        int n = static_cast<int>(rows_of_expert[e].size());
        if(n == 0)
            continue;
        int padded = (n + BlockM0 - 1) / BlockM0 * BlockM0;
        for(int i = 0; i < padded; i++)
        {
            if(i < n)
            {
                int tok = rows_of_expert[e][i];
                int kid = kidx_of_expert[e][i];
                // enc=0: base-offset ids; enc=2: repo mock id (token | k<<24),
                // what moe_sorting emits when CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID=1
                // (the repo default); enc=1: raw token ids.
                if(enc_mode == 0)
                    r.sorted_token_ids.push_back(tok - (r.padded_rows + i));
                else if(enc_mode == 2)
                    r.sorted_token_ids.push_back((tok & 0x00ffffff) | ((kid & 0xff) << 24));
                else
                    r.sorted_token_ids.push_back(tok);
                r.sorted_weight.push_back(w_of_expert[e][i]);
            }
            else
            {
                // padding: invalid token sentinel + weight 0 -> zero contribution
                if(enc_mode == 0)
                    r.sorted_token_ids.push_back(-(r.padded_rows + i));
                else if(enc_mode == 2)
                    r.sorted_token_ids.push_back((tokens & 0x00ffffff) | ((topk & 0xff) << 24));
                else
                    r.sorted_token_ids.push_back(tokens);
                r.sorted_weight.push_back(0.0f);
            }
        }
        for(int i = 0; i < padded / BlockM0; i++) r.sorted_expert_ids.push_back(e);
        r.padded_rows += padded;
    }
    r.num_sorted_tiles = r.padded_rows / BlockM0;
    return r;
}

// ---------------------------------------------------------------- weight shuffle
// fork fused_moe contract (mfma_type=1): [b, n, k] -> [b, n/16, k/32, 4, 16, 8]
// (i.e. per (n16, k32) block: [k%32/8 : 4][n%16 : 16][k%8 : 8] contiguous)
template <typename T>
static std::vector<T> shuffle_moe_weight(const std::vector<T>& w, int b_, int n_, int k_)
{
    std::vector<T> out(w.size());
    for(int b = 0; b < b_; b++)
        for(int n = 0; n < n_; n++)
            for(int k = 0; k < k_; k++)
            {
                size_t src = (static_cast<size_t>(b) * n_ + n) * k_ + k;
                size_t dst =
                    (((static_cast<size_t>(b) * (n_ / 16) + n / 16) * (k_ / 32) + k / 32) * 4 +
                     (k % 32) / 8) * 128 +
                    (n % 16) * 8 + (k % 8);
                out[dst] = w[src];
            }
    return out;
}

// ---------------------------------------------------------------- main
int main(int argc, char* argv[])
{
    int tokens    = argc > 1 ? atoi(argv[1]) : 256;
    int experts   = argc > 2 ? atoi(argv[2]) : 8;
    int topk      = argc > 3 ? atoi(argv[3]) : 2;
    int hidden    = argc > 4 ? atoi(argv[4]) : 512;
    int interm    = argc > 5 ? atoi(argv[5]) : 512;
    int do_valid  = argc > 6 ? atoi(argv[6]) : 1;
    int warmup    = argc > 7 ? atoi(argv[7]) : 5;
    int repeat    = argc > 8 ? atoi(argv[8]) : 20;
    int init_mode = argc > 9 ? atoi(argv[9]) : 0;
    uint32_t seed = argc > 10 ? static_cast<uint32_t>(atoi(argv[10])) : 11939;
    int enc_mode  = argc > 11 ? atoi(argv[11]) : 1; // 1: raw token ids (default), 0: base-offset ids

    printf("fused_moe flatmm_uk(HCU): t:%d e:%d k:%d h:%d i:%d v:%d init:%d\n",
           tokens,
           experts,
           topk,
           hidden,
           interm,
           do_valid,
           init_mode);
    printf("kernel: %s\n", FmoeKernel::GetName().c_str());

    // Load-vector alignment contract: the engine's A/G/U global loads
    // vectorize along hidden (Policy::GetAlignment_{A,G}() elements;
    // 4/8 for this fp16 config), and ck_tile masks out-of-range accesses
    // per WHOLE vector - a hidden not divisible by the vector length lets
    // the first vector's tail read past the row end into the next row's
    // data (verified: hidden=1/2/3/6 give silent wrong results, multiples
    // of 4 are bit-clean). Same contract as ck_tile GEMM's K alignment:
    // the caller pads. Everything device-visible below uses hidden_p as
    // the hidden stride with zeros in [hidden, hidden_p); the reference
    // still models and validates only the logical columns.
    constexpr int align_a = FmoePipeline::Engine::Policy::template GetAlignment_A<FmoeProblem>();
    constexpr int align_g = FmoePipeline::Engine::Policy::template GetAlignment_G<FmoeProblem>();
    const int align_h     = align_a > align_g ? align_a : align_g;
    const int hidden_p    = (hidden + align_h - 1) / align_h * align_h;
    if(hidden_p != hidden)
        printf("note: hidden %d padded to %d (load-vector alignment: A=%d G=%d)\n",
               hidden,
               hidden_p,
               align_a,
               align_g);

    // ---- host data (natural layouts, no shuffle)
    lcg_seed(seed);
    std::vector<ck_tile::fp16_t> a(static_cast<size_t>(tokens) * hidden);
    if(init_mode == 0)
        for(auto& v : a) v = h_fp16(lcg_float() * 2.0f);
    else if(init_mode == 2)
        for(auto& v : a) v = h_fp16(1.0f);
    else
        for(size_t i = 0; i < a.size(); i++)
            a[i] = h_fp16(0.01f * static_cast<float>(i % 97));

    std::vector<float> w1(static_cast<size_t>(experts) * 2 * interm * hidden);
    std::vector<float> w2(static_cast<size_t>(experts) * hidden * interm);
    if(init_mode == 0)
    {
        for(auto& v : w1) v = lcg_float();
        for(auto& v : w2) v = lcg_float();
    }
    else if(init_mode == 2)
    {
        for(auto& v : w1) v = 1.0f;
        for(auto& v : w2) v = 1.0f;
    }
    else if(init_mode == 3)
    {
        // coordinate-decodable D: value*100 = n*8 + k%8 + 1
        for(auto& v : w1) v = 0.01f;
        for(int e = 0; e < experts; e++)
            for(int n = 0; n < hidden; n++)
                for(int k = 0; k < interm; k++)
                    w2[(static_cast<size_t>(e) * hidden + n) * interm + k] =
                        0.01f * (n * 8 + k % 8 + 1);
    }
    else
    {
        for(size_t i = 0; i < w1.size(); i++)
            w1[i] = 0.001f * static_cast<float>((i * 7919) % 113) - 0.05f;
        for(size_t i = 0; i < w2.size(); i++)
            w2[i] = 0.001f * static_cast<float>((i * 104729) % 127) - 0.06f;
    }
    if(init_mode == 4)
    {
        // full-chain decode: a=1, gate row n = 0.001*(n+1), up = 1 ->
        //   y[n] = silu(0.001*(n+1)*hidden) * hidden
        // D selects interm column n == j for output column j ->
        //   o[t][j] = silu(0.001*(j+1)*hidden) * hidden   (strictly increasing in j)
        for(auto& v : a) v = h_fp16(1.0f);
        for(int e = 0; e < experts; e++)
            for(int n = 0; n < 2 * interm; n++)
                for(int kk = 0; kk < hidden; kk++)
                {
                    float g = (n < interm) ? 0.001f * (n + 1) : 1.0f;
                    w1[(static_cast<size_t>(e) * 2 * interm + n) * hidden + kk] = g;
                }
        for(int e = 0; e < experts; e++)
            for(int j = 0; j < hidden; j++)
                for(int n = 0; n < interm; n++)
                    w2[(static_cast<size_t>(e) * hidden + j) * interm + n] =
                        (n == j) ? 1.0f : 0.0f;
    }
    if(init_mode == 5)
    {
        // position-encoded D: value = 1000*row + col, so any in-kernel read of
        // the D tile decodes to the exact (hidden row, interm col) it touched
        for(int e = 0; e < experts; e++)
            for(int j = 0; j < hidden; j++)
                for(int n = 0; n < interm; n++)
                    w2[(static_cast<size_t>(e) * hidden + j) * interm + n] =
                        1000.0f * j + n;
    }
    if(init_mode == 6)
    {
        // separable decode: a[t] = 0.01*(t+1), gate row n = 0.001*(n+1), up = 1,
        // D selects interm column n == j  =>  o[t][j] ~ (t+1)^2 * (j+1):
        // a wrong A-row gather permutes the (t+1)^2 factor, a wrong column
        // mapping permutes the (j+1) factor
        for(int t = 0; t < tokens; t++)
            for(int kk = 0; kk < hidden; kk++)
                a[static_cast<size_t>(t) * hidden + kk] = h_fp16(0.01f * (t + 1));
        for(int e = 0; e < experts; e++)
            for(int n = 0; n < 2 * interm; n++)
                for(int kk = 0; kk < hidden; kk++)
                {
                    float g = (n < interm) ? 0.001f * (n + 1) : 1.0f;
                    w1[(static_cast<size_t>(e) * 2 * interm + n) * hidden + kk] = g;
                }
        for(int e = 0; e < experts; e++)
            for(int j = 0; j < hidden; j++)
                for(int n = 0; n < interm; n++)
                    w2[(static_cast<size_t>(e) * hidden + j) * interm + n] =
                        (n == j) ? 1.0f : 0.0f;
    }

    // topk assignment
    std::vector<std::vector<int>> topk_ids(tokens, std::vector<int>(topk));
    std::vector<std::vector<float>> topk_w(tokens, std::vector<float>(topk));
    for(int t = 0; t < tokens; t++)
    {
        std::vector<bool> used(experts, false);
        for(int k = 0; k < topk; k++)
        {
            int e = static_cast<int>(std::abs(static_cast<int>(lcg_float() * 1000))) % experts;
            while(used[e]) e = (e + 1) % experts;
            used[e]        = true;
            topk_ids[t][k] = e;
            topk_w[t][k]   = 0.5f + std::fabs(lcg_float());
        }
    }

    MoeSortResult sorted = host_sort(topk_ids, topk_w, tokens, experts, topk, enc_mode);
    printf("sorted: %d tiles (%d padded rows), experts used: %zu\n",
           sorted.num_sorted_tiles,
           sorted.padded_rows,
           sorted.sorted_expert_ids.size());
    if(do_valid >= 2)
    {
        printf("sorted_ids:");
        for(int i = 0; i < sorted.padded_rows && i < 64; i++)
            printf(" %d", sorted.sorted_token_ids[i]);
        printf("\nsorted_w:");
        for(int i = 0; i < sorted.padded_rows && i < 64; i++)
            printf(" %.2f", sorted.sorted_weight[i]);
        printf("\ntopk_map:");
        for(int t = 0; t < tokens && t < 32; t++)
            printf(" %d", topk_ids[t][0]);
        printf("\n");
    }

    // output uses the padded hidden stride too (kernel writes zeros into the
    // pad columns: D rows [hidden, hidden_p) are zero-padded below)
    std::vector<float> o(static_cast<size_t>(tokens) * hidden_p, 0.0f);

    // fp16 device copies of weights, pre-shuffled to the fork's fused_moe
    // mfma_type=1 layout (engine contract, see main.cpp of fork's fused_moe)
    int shuf_mode = argc > 12 ? atoi(argv[12]) : 0; // 0:none(natural, default) 1:W1 2:W2 3:both
    std::vector<ck_tile::fp16_t> w1_h(w1.size()), w2_h(w2.size());
    for(size_t i = 0; i < w1.size(); i++) w1_h[i] = h_fp16(w1[i]);
    for(size_t i = 0; i < w2.size(); i++) w2_h[i] = h_fp16(w2[i]);
    // reference snapshots BEFORE shuffling (logical layout)
    std::vector<float> g1f(w1_h.size()), g2f(w2_h.size());
    for(size_t i = 0; i < w1_h.size(); i++) g1f[i] = h_f32(w1_h[i]);
    for(size_t i = 0; i < w2_h.size(); i++) g2f[i] = h_f32(w2_h[i]);
    {
        // toolchain probe: the data path above no longer uses native _Float16
        // casts (see h_fp16 note); this one surviving cast only serves as a
        // canary — native_roundtrip must read 0.770000, anything else means
        // the DTK -c host miscompile is back and the run is not trustworthy.
        ck_tile::fp16_t ph = h_fp16(0.77f);
        uint16_t pbits;
        std::memcpy(&pbits, &ph, 2);
        printf("fp16 sanity: bits(0.77f)=0x%04x roundtrip=%.6f native_roundtrip=%.6f\n",
               pbits,
               h_f32(ph),
               static_cast<float>(static_cast<ck_tile::fp16_t>(0.77f)));
    }
    if(shuf_mode & 1)
        w1_h = shuffle_moe_weight(w1_h, experts, 2 * interm, hidden);
    if(shuf_mode & 2)
        w2_h = shuffle_moe_weight(w2_h, experts, hidden, interm);

    // padded upload staging: rows of A/W1 get hidden_p columns (zero tail),
    // W2 gets hidden_p rows (zero tail). Pad values are zero so they add
    // nothing to either gemm, and the zero W2 rows make the kernel write
    // exact zeros into the pad columns of O.
    std::vector<ck_tile::fp16_t> a_p(static_cast<size_t>(tokens) * hidden_p,
                                     h_fp16(0.0f));
    std::vector<ck_tile::fp16_t> w1_p(static_cast<size_t>(experts) * 2 * interm * hidden_p,
                                      h_fp16(0.0f));
    std::vector<ck_tile::fp16_t> w2_p(static_cast<size_t>(experts) * hidden_p * interm,
                                      h_fp16(0.0f));
    for(int t = 0; t < tokens; t++)
        for(int c = 0; c < hidden; c++)
            a_p[static_cast<size_t>(t) * hidden_p + c] = a[static_cast<size_t>(t) * hidden + c];
    for(size_t i = 0; i < w1_h.size(); i++)
        w1_p[(i / hidden) * static_cast<size_t>(hidden_p) + i % hidden] = w1_h[i];
    for(int e = 0; e < experts; e++)
        for(int r = 0; r < hidden; r++)
            for(int c = 0; c < interm; c++)
                w2_p[(static_cast<size_t>(e) * hidden_p + r) * interm + c] =
                    w2_h[(static_cast<size_t>(e) * hidden + r) * interm + c];

    // ---- device buffers
    void *d_a, *d_g, *d_d, *d_o, *d_sid, *d_sw, *d_seid, *d_nst;
    HIP_CHECK(hipMalloc(&d_a, a_p.size() * 2));
    HIP_CHECK(hipMalloc(&d_g, w1_p.size() * 2));
    HIP_CHECK(hipMalloc(&d_d, w2_p.size() * 2));
    HIP_CHECK(hipMalloc(&d_o, o.size() * 4));
    HIP_CHECK(hipMalloc(&d_sid, sorted.sorted_token_ids.size() * 4));
    HIP_CHECK(hipMalloc(&d_sw, sorted.sorted_weight.size() * 4));
    HIP_CHECK(hipMalloc(&d_seid, sorted.sorted_expert_ids.size() * 4));
    HIP_CHECK(hipMalloc(&d_nst, 4));

    HIP_CHECK(hipMemcpy(d_a, a_p.data(), a_p.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_g, w1_p.data(), w1_p.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d, w2_p.data(), w2_p.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sid, sorted.sorted_token_ids.data(),
                        sorted.sorted_token_ids.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sw, sorted.sorted_weight.data(),
                        sorted.sorted_weight.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seid, sorted.sorted_expert_ids.data(),
                        sorted.sorted_expert_ids.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_nst, &sorted.padded_rows, 4, hipMemcpyHostToDevice));

    // TEMP-DEBUG probe (remove after diagnosis)
    printf("DBG: hidden_p=%d a_p=%zu w1_p=%zu w2_p=%zu o=%zu\n",
           hidden_p, a_p.size(), w1_p.size(), w2_p.size(), o.size());
    printf("DBG: a_p0=%.4f w1_p0=%.4f w2_p0=%.4f a_plast=%.4f\n",
           h_f32(a_p[0]), h_f32(w1_p[0]), h_f32(w2_p[0]), h_f32(a_p.back()));
    {
        std::vector<ck_tile::fp16_t> rt(8);
        HIP_CHECK(hipMemcpy(rt.data(), d_g, 16, hipMemcpyDeviceToHost));
        printf("DBG: d_g[0..3]=%.4f %.4f %.4f %.4f\n",
               h_f32(rt[0]), h_f32(rt[1]), h_f32(rt[2]), h_f32(rt[3]));
        HIP_CHECK(hipMemcpy(rt.data(), d_a, 16, hipMemcpyDeviceToHost));
        printf("DBG: d_a[0..3]=%.4f %.4f %.4f %.4f\n",
               h_f32(rt[0]), h_f32(rt[1]), h_f32(rt[2]), h_f32(rt[3]));
        HIP_CHECK(hipMemcpy(rt.data(), d_d, 16, hipMemcpyDeviceToHost));
        printf("DBG: d_d[0..3]=%.4f %.4f %.4f %.4f\n",
               h_f32(rt[0]), h_f32(rt[1]), h_f32(rt[2]), h_f32(rt[3]));
    }

    // ---- kernel args (field order must match Kargs exactly)
    FmoeKernel::Hargs hargs{};
    hargs.a_ptr                     = d_a;
    hargs.g_ptr                     = d_g;
    hargs.d_ptr                     = d_d;
    hargs.o_ptr                     = d_o;
    hargs.sorted_token_ids_ptr      = d_sid;
    hargs.sorted_weight_ptr         = d_sw;
    hargs.sorted_expert_ids_ptr     = d_seid;
    hargs.num_sorted_tiles_ptr      = d_nst;
    hargs.hidden_size               = hidden_p; // padded: see load-vector note above
    hargs.intermediate_size         = interm;
    hargs.num_tokens                = tokens;
    hargs.num_experts               = experts;
    hargs.topk                      = topk;
    hargs.stride_token              = hidden_p;

    auto kargs = FmoeKernel::MakeKargs(hargs);
    auto grids = FmoeKernel::GridSize(hargs);
    auto blk   = FmoeKernel::BlockSize();

    if(do_valid >= 2)
    {
        // debug: dump the y-dim structure of the C-block and O distributions
        auto c_block = FmoePipeline::Engine::Policy::template MakeCBlockTile_Gemm1<FmoeProblem>();
        auto o_dist  = FmoePipeline::Engine::Policy::template MakeGlobalTileDistribution_O<FmoeProblem>();
        auto c_lens = to_sequence(
            c_block.get_tile_distribution().get_ys_to_d_descriptor().get_lengths());
        auto o_lens = to_sequence(o_dist.get_ys_to_d_descriptor().get_lengths());
        printf("C block y-lens:");
        ck_tile::static_for<0, c_lens.size(), 1>{}(
            [&](auto j) { printf(" %d", static_cast<int>(c_lens.at(j))); });
        printf("\nO dist  y-lens:");
        ck_tile::static_for<0, o_lens.size(), 1>{}(
            [&](auto j) { printf(" %d", static_cast<int>(o_lens.at(j))); });
        printf("\n");
    }

    printf("grid: %ux%ux%u block: %u smem: %d bytes\n",
           grids.x,
           grids.y,
           grids.z,
           blk.x,
           FmoePipeline::GetSmemSizeBytes());

    ck_tile::stream_config sc;
    auto launch = [&]() {
        float t = ck_tile::launch_kernel(
            sc,
            ck_tile::make_kernel<FmoeKernel::BlockShape::BlockSize, 1>(
                FmoeKernel{}, grids, blk, 0, kargs));
        (void)t;
    };

    for(int i = 0; i < warmup; i++)
    {
        HIP_CHECK(hipMemset(d_o, 0, o.size() * 4));
        launch();
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventRecord(e0));
    for(int i = 0; i < repeat; i++)
    {
        HIP_CHECK(hipMemset(d_o, 0, o.size() * 4));
        launch();
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    ms /= repeat;
    double flops = static_cast<double>(tokens) * topk * (6.0 * hidden * interm);
    printf("time: %.4f ms  perf: %.2f TFlops\n", ms, flops / (ms * 1e-3) / 1e12);

    // ---- validation
    int ret = 0;
    if(do_valid)
    {
        std::vector<float> o_h(o.size());
        HIP_CHECK(hipMemcpy(o_h.data(), d_o, o.size() * 4, hipMemcpyDeviceToHost));
        printf("DBG: o_h[0..3]=%.6f %.6f %.6f %.6f\n", o_h[0], o_h[1], o_h[2], o_h[3]);

        // compact the padded output back to logical columns; the pad columns
        // are exact zeros on both sides, and everything below (reference,
        // counters, dumps) works on the compact view
        std::vector<float> o_c(static_cast<size_t>(tokens) * hidden);
        for(int t = 0; t < tokens; t++)
            for(int c = 0; c < hidden; c++)
                o_c[static_cast<size_t>(t) * hidden + c] =
                    o_h[static_cast<size_t>(t) * hidden_p + c];

        // fp32 reference
        std::vector<float> ref(static_cast<size_t>(tokens) * hidden, 0.0f);
        std::vector<float> y_row(interm);
        std::vector<float> af(static_cast<size_t>(tokens) * hidden);
        for(size_t i = 0; i < af.size(); i++) af[i] = h_f32(a[i]);
        // g1f/g2f already snapshotted from unshuffled fp16 weights
        for(int t = 0; t < tokens; t++)
            for(int k = 0; k < topk; k++)
            {
                int e   = topk_ids[t][k];
                float w = topk_w[t][k];
                for(int n = 0; n < interm; n++)
                {
                    float gacc = 0, uacc = 0;
                    size_t gr  = (static_cast<size_t>(e) * 2 * interm + n) * hidden;
                    size_t ur  = (static_cast<size_t>(e) * 2 * interm + interm + n) * hidden;
                    for(int kk = 0; kk < hidden; kk++)
                    {
                        gacc += af[static_cast<size_t>(t) * hidden + kk] * g1f[gr + kk];
                        uacc += af[static_cast<size_t>(t) * hidden + kk] * g1f[ur + kk];
                    }
                    // mirror the engine's fp16 rounding chain: acc->fp16 before
                    // activation, activation out ->fp16, product ->fp16
                    float g0 = h_f32(h_fp16(gacc));
                    float u0 = h_f32(h_fp16(uacc));
                    float sg = g0 / (1.0f + std::exp(-g0));
                    float sgh = h_f32(h_fp16(sg));
                    y_row[n] = h_f32(h_fp16(sgh * u0));
                }
                for(int nn = 0; nn < hidden; nn++)
                {
                    float acc = 0;
                    size_t dr = (static_cast<size_t>(e) * hidden + nn) * interm;
                    for(int kk = 0; kk < interm; kk++) acc += y_row[kk] * g2f[dr + kk];
                    ref[static_cast<size_t>(t) * hidden + nn] += w * acc;
                }
            }

        double max_abs = 0, max_rel = 0, sum_sq = 0, sum_ref = 0;
        long bad = 0, nonfinite_o = 0, nonzero_ref = 0, nonzero_o = 0;
        size_t n = ref.size();
        for(size_t i = 0; i < n; i++)
        {
            double d     = std::fabs(static_cast<double>(o_c[i]) - ref[i]);
            double denom = std::fabs(ref[i]);
            if(denom < 1e-2)
                denom = 1e-2;
            double r = d / denom;
            if(d > max_abs)
                max_abs = d;
            if(r > max_rel)
                max_rel = r;
            sum_sq += d * d;
            sum_ref += ref[i] * ref[i];
            if(d > 3e-2 + 3e-2 * std::fabs(ref[i]))
                bad++;
            // degeneracy counters: NaN comparisons above are all false, so a
            // NaN output or an all-zero reference would otherwise pass vacuously
            if(std::isnan(o_c[i]))
                nonfinite_o++;
            if(ref[i] != 0.0f)
                nonzero_ref++;
            if(o_c[i] != 0.0f)
                nonzero_o++;
        }
        printf("verify: max_abs=%.5f max_rel=%.5f rmse=%.6f ref_rms=%.6f bad=%ld/%zu\n",
               max_abs,
               max_rel,
               std::sqrt(sum_sq / n),
               std::sqrt(sum_ref / n),
               bad,
               n);
        if(do_valid >= 2)
        {
            // dump first rows to inspect error structure (index-guarded:
            // degenerate shapes can be smaller than the fixed window)
            for(int t = 0; t < 2; t++)
            {
                printf("row %d o  :", t);
                for(int c = 0; c < 24; c++)
                    if(static_cast<size_t>(t) * hidden + c < o_c.size())
                        printf(" %12.4f", o_c[static_cast<size_t>(t) * hidden + c]);
                printf("\nrow %d ref:", t);
                for(int c = 0; c < 24; c++)
                    if(static_cast<size_t>(t) * hidden + c < o_c.size())
                        printf(" %12.4f", ref[static_cast<size_t>(t) * hidden + c]);
                printf("\n");
            }
            // ratio stats over non-tiny refs
            double rsum = 0;
            long rcnt = 0;
            for(size_t i = 0; i < n; i++)
                if(std::fabs(ref[i]) > 1e-1 && std::fabs(o_c[i]) > 1e-3)
                {
                    rsum += o_c[i] / ref[i];
                    rcnt++;
                }
            if(rcnt > 0)
                printf("mean o/ref ratio over %ld elems: %.5f\n", rcnt, rsum / rcnt);
            else
                printf("no ratio samples (output near zero?)\n");
            // per-row bad counts
            for(int t = 0; t < tokens && t < 40; t++)
            {
                long rb = 0;
                for(int c = 0; c < hidden; c++)
                {
                    size_t i = static_cast<size_t>(t) * hidden + c;
                    if(std::fabs(o_c[i] - ref[i]) > 3e-2 + 3e-2 * std::fabs(ref[i]))
                        rb++;
                }
                if(rb > 0)
                    printf("row %d: %d/%d bad, o[0]=%.4f ref[0]=%.4f\n",
                           t,
                           static_cast<int>(rb),
                           hidden,
                           o_c[static_cast<size_t>(t) * hidden],
                           ref[static_cast<size_t>(t) * hidden]);
            }
            // column pattern of row 0: compact bad ranges
            printf("row0 bad ranges:");
            int c0 = -1;
            for(int c = 0; c <= hidden; c++)
            {
                size_t i = c;
                bool badc = c < hidden &&
                            std::fabs(o_c[i] - ref[i]) > 3e-2 + 3e-2 * std::fabs(ref[i]);
                if(badc && c0 < 0)
                    c0 = c;
                else if(!badc && c0 >= 0)
                {
                    printf(" [%d,%d)", c0, c);
                    c0 = -1;
                }
            }
            printf("  o[48]=%.3f o[80]=%.3f o[112]=%.3f\n",
                   o_c.size() > 48 ? o_c[48] : 0.0f,
                   o_c.size() > 80 ? o_c[80] : 0.0f,
                   o_c.size() > 112 ? o_c[112] : 0.0f);
        }
        printf("sanity: nan_o=%ld nonzero_ref=%ld nonzero_o=%ld\n", nonfinite_o, nonzero_ref, nonzero_o);
        if(nonfinite_o > 0 || nonzero_ref == 0 || (nonzero_ref > 0 && nonzero_o == 0))
        {
            printf("FAIL (degenerate output/reference; the tolerance check above is vacuous)\n");
            ret = 1;
            return ret;
        }
        if(bad == 0)
        {
            printf("PASS\n");
        }
        else
        {
            printf("FAIL (%ld elems exceed tol)\n", bad);
            ret = 1;
        }
    }

    static_cast<void>(hipFree(d_a));
    static_cast<void>(hipFree(d_g));
    static_cast<void>(hipFree(d_d));
    static_cast<void>(hipFree(d_o));
    static_cast<void>(hipFree(d_sid));
    static_cast<void>(hipFree(d_sw));
    static_cast<void>(hipFree(d_seid));
    static_cast<void>(hipFree(d_nst));
    return ret;
}
