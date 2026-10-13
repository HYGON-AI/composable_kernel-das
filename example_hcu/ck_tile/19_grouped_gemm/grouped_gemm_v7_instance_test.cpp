// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technologies Co., Ltd.
//
// Correctness + performance driver for the v7 BF16 NN / TN grouped-GEMM
// instances (InstanceId 5000 / 5001).
//
// WHAT THIS DRIVES
// The instances themselves, through the versioned device-args ABI Primus uses:
//   ck_tile_hcu_grouped_gemm_select_device_args_v1  -> must return the v7 id
//   ck_tile_hcu_grouped_gemm_run_device_args_v1     -> launches the kernel
// Nothing is called behind the ABI's back: the kargs array is built here exactly
// as the ABI's builders build it (one ck_tile::GemmTransKernelArg per group,
// uploaded once, consumed by the device-side tile walk).
//
// The v7 routes are OPT-IN, so the driver sets the same switch production
// would set:
//   CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN / _TN
// and --ref=1 re-runs the same kargs through the DEFAULT instance for the same
// problem (switch disabled) to give a same-process, same-descriptor A/B.
//
// WHAT IT CHECKS
//  * --check=1 (default): an independent CPU reference (fp32 accumulate over
//    bf16 inputs, which are exact in fp32 -- only the summation order differs)
//    and, with --ref=1, the default CK instance as a triangle.
//  * always: selection identity, launch status, and the achieved time/TFLOPS.
//
// Layout contract (the one the v7 NN / TN entries consume):
//   NN : A [M_g, K] row-major (stride_a = K)
//        B [K, N]  row-major (stride_b = N)
//   TN : A [K, M_g] row-major panel, per-group a_ptr, stride_a = the panel pitch
//        B [K, N]  row-major (stride_b = N)
//   D    [M_g, N] row-major (stride_e = N)
// TN's A panel load covers a whole 256-wide column tile, so the A allocation
// carries slack past the last group (kAtailPad) -- the caller-side pad
// obligation the v7 TN entry's header documents.
//
// Usage:
//   grouped_gemm_v7_instance_test [--layout=NN|TN] [--m=512] [--n=512]
//                                   [--k=256] [--groups=2] [--iters=20]
//                                   [--warmup=3] [--check=1] [--ref=1]
//                                   [--hipblaslt-ref=0] [--hipblaslt-perf=0]
//   --hipblaslt-ref=1: verify against one hipBLASLt GEMM per group, compared
//       on the device (no CPU pass over the output)
//   --hipblaslt-perf=1: time hipBLASLt (plans built once, bare Matmul in the
//       loop) and print blas_ms / blas_tf next to the v7 numbers

#include <hip/hip_runtime.h>

#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"

#if defined(CK_TILE_V7_TEST_HAS_HIPBLASLT)
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bf16 = ck_tile::bf16_t;

constexpr std::uint32_t kV7NnInstance = 5000;
constexpr std::uint32_t kV8TnInstance = 5001;

// Slack the TN A panel needs past the last group's columns (see the header).
constexpr std::size_t kAtailPad = 1024;

struct Options
{
    std::string layout = "NN";
    int m              = 512;
    int n              = 512;
    int k              = 256;
    int groups         = 2;
    int iters          = 20;
    int warmup         = 3;
    bool check         = true;
    bool ref           = true;
    // The independent fp32 CPU reference the checked runs compare against (bf16
    // inputs are exact in fp32 -- only the summation order differs). --check=0
    // skips it for perf runs on production shapes.
    bool cpu_ref       = true;
    // Ragged group lengths: the shapes the v7 harness benchmarks
    // (--grouped-length=8100,8100,8091,8091). Empty => `groups` copies of `m`.
    std::vector<int> group_lengths;
    bool dump          = false;
    // Grid override. 0 => the device's CU count (the production value).
    int num_cu         = 0;
    // Device-side functional reference: run the same kargs through hipBLASLt
    // (one plain GEMM per group) and compare on the GPU -- no CPU pass over the
    // output, so production shapes check in seconds. Built only when the
    // hipBLASLt install is present (see CMakeLists CK_TILE_V7_TEST_HAS_HIPBLASLT).
    bool hipblaslt_ref = false;
    // Performance reference: time hipBLASLt (plans built once, bare Matmul in
    // the loop) with the same warmup/iters as the v7 timing, and print
    // blas_ms / blas_tf next to v7_ms / v7_tf.
    bool hipblaslt_perf = false;
};

std::size_t align_to(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// Every hip* call below whose result carries [[nodiscard]] goes through this, so
// the driver both stays -Wunused-result clean and reports the first failure
// instead of running on with garbage.
void hip_check(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "HIP error %d (%s) at %s\n", static_cast<int>(status),
                     hipGetErrorName(status), what);
        std::exit(1);
    }
}

// Small multiples of 0.125: exact in bf16 (8 mantissa bits), so the only
// difference between the CPU reference and the kernel is the summation order.
// Both indices contribute, so no lane pattern of the kernels cancels a term.
float a_value(int row, int kk)
{
    return static_cast<float>((row % 7) - 3) * 0.5f +
           static_cast<float>(kk % 3) * 0.25f;
}
float b_value(int kk, int col)
{
    return static_cast<float>((col % 5) - 2) * 0.25f +
           static_cast<float>(kk % 4) * 0.125f;
}

struct Case
{
    // Per-group row counts (ragged) plus their sum. `m` stays the first group's
    // length for the single-group paths; every buffer is sized from group_m.
    std::vector<int> group_m;
    int total_m = 0;
    bool tn  = false;
    int m    = 0;
    int n    = 0;
    int k    = 0;
    int g    = 0;
    std::vector<int> strides_a;
    std::vector<int> strides_b;
    std::vector<int> a_elements;
    std::vector<int> b_elements;
    std::vector<std::vector<Bf16>> host_a;
    std::vector<std::vector<Bf16>> host_b;
    std::vector<std::vector<float>> host_d_ref;
};

int parse_int_arg(const char* text, const char* name)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if(end == text || value < 0)
    {
        std::fprintf(stderr, "bad --%s=%s\n", name, text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

void parse_args(int argc, char* argv[], Options& options)
{
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto value_of   = [&](const char* name) -> const char* {
            const std::size_t at = arg.find('=');
            if(at == std::string::npos)
            {
                std::fprintf(stderr, "%s expects --%s=VALUE\n", arg.c_str(), name);
                std::exit(2);
            }
            return argv[i] + at + 1;
        };
        if(arg.rfind("--layout=", 0) == 0)
        {
            options.layout = value_of("layout");
        }
        else if(arg.rfind("--m=", 0) == 0)
        {
            options.m = parse_int_arg(value_of("m"), "m");
        }
        else if(arg.rfind("--n=", 0) == 0)
        {
            options.n = parse_int_arg(value_of("n"), "n");
        }
        else if(arg.rfind("--k=", 0) == 0)
        {
            options.k = parse_int_arg(value_of("k"), "k");
        }
        else if(arg.rfind("--groups=", 0) == 0)
        {
            options.groups = parse_int_arg(value_of("groups"), "groups");
        }
        else if(arg.rfind("--iters=", 0) == 0)
        {
            options.iters = parse_int_arg(value_of("iters"), "iters");
        }
        else if(arg.rfind("--warmup=", 0) == 0)
        {
            options.warmup = parse_int_arg(value_of("warmup"), "warmup");
        }
        else if(arg.rfind("--check=", 0) == 0)
        {
            options.check = parse_int_arg(value_of("check"), "check") != 0;
        }
        else if(arg.rfind("--ref=", 0) == 0)
        {
            options.ref = parse_int_arg(value_of("ref"), "ref") != 0;
        }
        else if(arg.rfind("--group-lengths=", 0) == 0)
        {
            options.group_lengths.clear();
            const std::string list = value_of("group-lengths");
            std::size_t start      = 0;
            while(start <= list.size())
            {
                const std::size_t comma = list.find(',', start);
                const std::string item  = list.substr(start, comma - start);
                if(!item.empty())
                {
                    options.group_lengths.push_back(parse_int_arg(item.c_str(), "group-lengths"));
                }
                if(comma == std::string::npos) break;
                start = comma + 1;
            }
        }
        else if(arg.rfind("--cpu-ref=", 0) == 0)
        {
            options.cpu_ref = parse_int_arg(value_of("cpu-ref"), "cpu-ref") != 0;
        }
        else if(arg.rfind("--dump=", 0) == 0)
        {
            options.dump = parse_int_arg(value_of("dump"), "dump") != 0;
        }
        else if(arg.rfind("--num-cu=", 0) == 0)
        {
            options.num_cu = parse_int_arg(value_of("num-cu"), "num-cu");
        }
        else if(arg.rfind("--hipblaslt-ref=", 0) == 0)
        {
            options.hipblaslt_ref = parse_int_arg(value_of("hipblaslt-ref"), "hipblaslt-ref") != 0;
        }
        else if(arg.rfind("--hipblaslt-perf=", 0) == 0)
        {
            options.hipblaslt_perf = parse_int_arg(value_of("hipblaslt-perf"), "hipblaslt-perf") != 0;
        }
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            std::exit(2);
        }
    }
    if(options.layout != "NN" && options.layout != "TN")
    {
        std::fprintf(stderr, "--layout must be NN or TN\n");
        std::exit(2);
    }
}

void build_case(const Options& options, Case& c)
{
    c.tn = options.layout == "TN";
    c.n  = options.n;
    c.k  = options.k;
    c.group_m = options.group_lengths.empty()
                    ? std::vector<int>(static_cast<std::size_t>(options.groups), options.m)
                    : options.group_lengths;
    c.g  = static_cast<int>(c.group_m.size());
    c.m  = c.group_m.front();
    c.total_m = 0;
    for(int g = 0; g < c.g; ++g) c.total_m += c.group_m[g];
    // TN's A panel is [K, sum(M_g)]: it is addressed across the whole group set,
    // so its pitch is the summed M (padded up to a whole tile, which is also the
    // 256-element slack the panel fetch needs past the last group's columns).
    const int a_pitch_tn = (c.total_m + 255) / 256 * 256;

    c.strides_a.resize(c.g);
    c.strides_b.resize(c.g);
    c.a_elements.resize(c.g);
    c.b_elements.resize(c.g);
    c.host_a.resize(c.g);
    c.host_b.resize(c.g);
    c.host_d_ref.resize(c.g);

    for(int g = 0; g < c.g; ++g)
    {
        // NN: A row-major [M_g, K]. TN: A is the [K, M_g] panel of the
        // column-major [M_g, K] view, so its pitch is the M extent (the same
        // choice the device-args smoke makes for its TN cases).
        const int stride_a = c.tn ? a_pitch_tn : c.k;
        const int stride_b = c.n;
        c.strides_a[g]     = stride_a;
        c.strides_b[g]     = stride_b;
        c.a_elements[g] =
            (c.tn ? c.k : c.group_m[g]) * stride_a + (c.tn ? static_cast<int>(kAtailPad) : 0);
        c.b_elements[g] = c.k * stride_b;

        c.host_a[g].assign(static_cast<std::size_t>(c.a_elements[g]),
                           ck_tile::type_convert<Bf16>(0.0f));
        c.host_b[g].assign(static_cast<std::size_t>(c.b_elements[g]),
                           ck_tile::type_convert<Bf16>(0.0f));
        c.host_d_ref[g].assign(static_cast<std::size_t>(c.group_m[g]) * c.n, 0.0f);

        for(int row = 0; row < c.group_m[g]; ++row)
        {
            for(int kk = 0; kk < c.k; ++kk)
            {
                const std::size_t at =
                    c.tn ? static_cast<std::size_t>(kk) * stride_a + row
                         : static_cast<std::size_t>(row) * stride_a + kk;
                c.host_a[g][at] = ck_tile::type_convert<Bf16>(a_value(row, kk));
            }
        }
        for(int kk = 0; kk < c.k; ++kk)
        {
            for(int col = 0; col < c.n; ++col)
            {
                c.host_b[g][static_cast<std::size_t>(kk) * stride_b + col] =
                    ck_tile::type_convert<Bf16>(b_value(kk, col));
            }
        }
    }
}

void cpu_reference(Case& c)
{
    for(int g = 0; g < c.g; ++g)
    {
        const int stride_a = c.strides_a[g];
        const int stride_b = c.strides_b[g];
        for(int col = 0; col < c.n; ++col)
        {
            for(int row = 0; row < c.m; ++row)
            {
                float acc = 0.0f;
                for(int kk = 0; kk < c.k; ++kk)
                {
                    const std::size_t at =
                        c.tn ? static_cast<std::size_t>(kk) * stride_a + row
                             : static_cast<std::size_t>(row) * stride_a + kk;
                    const float a = ck_tile::type_convert<float>(c.host_a[g][at]);
                    const float b = ck_tile::type_convert<float>(
                        c.host_b[g][static_cast<std::size_t>(kk) * stride_b + col]);
                    acc += a * b;
                }
                c.host_d_ref[g][static_cast<std::size_t>(row) * c.n + col] = acc;
            }
        }
    }
}

#if defined(CK_TILE_V7_TEST_HAS_HIPBLASLT)
// ---------------------------------------------------------------------------
// Device-side functional reference: one plain hipBLASLt GEMM per group, then a
// device-side element compare -- the output never touches the CPU, so
// production shapes verify in seconds.
//
// The col-major formulation (same construction bf16_gemm_hipblaslt.cpp uses):
//   D_row[M_g, N] = A . B   <=>   C_col[N, M_g] = B_view[N, K] . A_view[K, M_g]
// with B_view the [N, K] col-major view of the [K, N] row-major B (ld = N),
// and A_view the operand buffer seen as
//   NN: [K, M_g] col-major, ld = K        (the [M_g, K] row-major A)  -> opB = N
//   TN: [M_g, K] col-major, ld = pitch    (the [K, pitch] A panel)   -> opB = T
// The panel leading dimension is the one thing the bf16_gemm_hipblaslt wrapper
// cannot express for the grouped TN operand (it hardcodes ldb = k); here it is
// passed explicitly.
// ---------------------------------------------------------------------------
void hipblaslt_check_status(hipblasStatus_t status, const char* what)
{
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "hipBLASLt error %d at %s\n",
                     static_cast<int>(status),
                     what);
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Tuned-solution table -- same file and semantics as bf16_gemm_hipblaslt.cpp
// (BF16GEMM_HIPBLASLT_SOLUTION_FILE selects it, BF16GEMM_NO_TUNING=1 disables;
// lines "m,n,k,transA,transB,solution_index", '*' = wildcard, '#' comments).
// The keys live in THIS test's hipblasLt parameter space (m = caller N,
// n = M_g, k = K; transA is always N here, transB = T for the TN panel form),
// which is the same space the wrapper tunes in. Resolution: best-scoring
// entry -> hipblaslt_ext getAlgosFromIndex + matmulIsAlgoSupported -> else the
// stock heuristic, warning once per rejected key.
// ---------------------------------------------------------------------------
struct TunedSolution
{
    long long m = -1, n = -1, k = -1;   // -1 = wildcard
    char ta = '?', tb = '?';
    int index = -1;
};

static long long tuned_num(const std::string& s)
{
    return (s == "*") ? -1 : std::strtoll(s.c_str(), nullptr, 10);
}

static bool tuned_hit(long long key, long long want)
{
    return key < 0 || key == want;
}

static bool tuned_hit(char key, char want)
{
    return key == '*' || key == '?' || key == want;
}

static const std::vector<TunedSolution>& tuned_table()
{
    static const std::vector<TunedSolution> table = [] {
        std::vector<TunedSolution> t;
        const char* disable = std::getenv("BF16GEMM_NO_TUNING");
        if(disable != nullptr && std::atoi(disable) != 0)
        {
            return t;
        }
        const char* path = std::getenv("BF16GEMM_HIPBLASLT_SOLUTION_FILE");
        if(path == nullptr || *path == '\0')
        {
            return t;
        }
        FILE* f = std::fopen(path, "r");
        if(f == nullptr)
        {
            std::fprintf(stderr,
                         "[hipblasLt] WARNING: cannot open '%s', using the stock heuristic\n",
                         path);
            return t;
        }
        char line[512];
        while(std::fgets(line, sizeof(line), f) != nullptr)
        {
            char* p = line;
            while(*p == ' ' || *p == '\t')
            {
                ++p;
            }
            if(*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
            {
                continue;
            }
            std::string fields[6];
            int nf = 0;
            for(char* tok = std::strtok(p, ", \t\r\n"); tok != nullptr && nf < 6;
                tok = std::strtok(nullptr, ", \t\r\n"))
            {
                fields[nf++] = tok;
            }
            if(nf < 6)
            {
                std::fprintf(stderr, "[hipblasLt] WARNING: malformed line ignored: %s", line);
                continue;
            }
            TunedSolution e;
            e.m  = tuned_num(fields[0]);
            e.n  = tuned_num(fields[1]);
            e.k  = tuned_num(fields[2]);
            e.ta = static_cast<char>(std::toupper(static_cast<unsigned char>(fields[3][0])));
            e.tb = static_cast<char>(std::toupper(static_cast<unsigned char>(fields[4][0])));
            e.index = static_cast<int>(std::strtol(fields[5].c_str(), nullptr, 10));
            t.push_back(e);
        }
        std::fclose(f);
        std::fprintf(stderr,
                     "[hipblasLt] tuned-solution table: %zu entries from %s\n",
                     t.size(),
                     path);
        return t;
    }();
    return table;
}

static bool tuned_pick(long long m, long long n, long long k, char ta, char tb, int& index)
{
    int best_score = 6;
    bool found     = false;
    for(const TunedSolution& e : tuned_table())
    {
        if(!(tuned_hit(e.m, m) && tuned_hit(e.n, n) && tuned_hit(e.k, k) &&
             tuned_hit(e.ta, ta) && tuned_hit(e.tb, tb)))
        {
            continue;
        }
        const int score = (e.m < 0 ? 1 : 0) + (e.n < 0 ? 1 : 0) + (e.k < 0 ? 1 : 0) +
                          ((e.ta == '*' || e.ta == '?') ? 1 : 0) +
                          ((e.tb == '*' || e.tb == '?') ? 1 : 0);
        if(score < best_score)
        {
            best_score = score;
            index      = e.index;
            found      = true;
            if(score == 0)
            {
                break;   // exact key: cannot do better
            }
        }
    }
    return found;
}

// D = 1.0 * op(B) . op(A) + 0.0, all bf16 in / bf16 out, fp32 compute.
// Descriptor + algorithm selection is split from the execution: the perf
// reference builds each group's plan once and times bare hipblasLtMatmul
// calls, exactly what a production consumer would do.
struct HipblasltPlan
{
    hipblasLtMatmulDesc_t desc       = nullptr;
    hipblasLtMatrixLayout_t a_layout = nullptr;   // the B operand, col-major [N, K], ld = N
    hipblasLtMatrixLayout_t b_layout = nullptr;   // the A operand
    hipblasLtMatrixLayout_t c_layout = nullptr;   // D, col-major [N, M_g], ld = N
    hipblasLtMatmulAlgo_t algo{};
};

bool hipblaslt_make_plan(HipblasltPlan& plan,
                         hipblasLtHandle_t handle,
                         void* workspace,
                         std::size_t workspace_size,
                         int m_g,
                         int n,
                         int k,
                         int ld_a,
                         bool tn)
{
    (void)workspace;   // the workspace pointer only matters at Matmul time
    hipblasLtMatmulPreference_t pref = nullptr;

    hipblasOperation_t op_a = HIPBLAS_OP_N;
    hipblasOperation_t op_b = tn ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    hipblaslt_check_status(
        hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
        "hipblasLtMatmulDescCreate");
    hipblaslt_check_status(
        hipblasLtMatmulDescSetAttribute(plan.desc,
                                        HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &op_a,
                                        sizeof(op_a)),
        "SetAttribute(TRANSA)");
    hipblaslt_check_status(
        hipblasLtMatmulDescSetAttribute(plan.desc,
                                        HIPBLASLT_MATMUL_DESC_TRANSB,
                                        &op_b,
                                        sizeof(op_b)),
        "SetAttribute(TRANSB)");
    hipblaslt_check_status(
        hipblasLtMatrixLayoutCreate(&plan.a_layout,
                                    HIP_R_16BF,
                                    static_cast<std::size_t>(n),
                                    static_cast<std::size_t>(k),
                                    static_cast<std::size_t>(n)),
        "LayoutCreate(B view)");
    // opT stores the [K, M_g] logical matrix through the [M_g, K] panel: ld is
    // the panel pitch, not K.
    const std::size_t b_ld = tn ? static_cast<std::size_t>(ld_a)
                                : static_cast<std::size_t>(k);
    hipblaslt_check_status(
        hipblasLtMatrixLayoutCreate(&plan.b_layout,
                                    HIP_R_16BF,
                                    static_cast<std::size_t>(k),
                                    static_cast<std::size_t>(m_g),
                                    b_ld),
        "LayoutCreate(A view)");
    hipblaslt_check_status(
        hipblasLtMatrixLayoutCreate(&plan.c_layout,
                                    HIP_R_16BF,
                                    static_cast<std::size_t>(n),
                                    static_cast<std::size_t>(m_g),
                                    static_cast<std::size_t>(n)),
        "LayoutCreate(D)");

    hipblaslt_check_status(
        hipblasLtMatmulPreferenceCreate(&pref),
        "hipblasLtMatmulPreferenceCreate");
    hipblaslt_check_status(
        hipblasLtMatmulPreferenceSetAttribute(
            pref,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)),
        "PreferenceSetAttribute(WORKSPACE)");

    // 1) tuned table (best-scoring entry, resolved through the ext API);
    // 2) stock heuristic. A table entry the library rejects only warns.
    const char ta_ch = (op_a == HIPBLAS_OP_N) ? 'N' : 'T';
    const char tb_ch = (op_b == HIPBLAS_OP_N) ? 'N' : 'T';
    int tuned_index  = -1;
    if(tuned_pick(n, m_g, k, ta_ch, tb_ch, tuned_index))
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> resolved(1);
        std::vector<int> algo_index(1, tuned_index);
        const float probe_alpha = 1.0f;
        const float probe_beta  = 0.0f;
        std::size_t required    = 0;
        if(hipblaslt_ext::getAlgosFromIndex(handle, algo_index, resolved) !=
               HIPBLAS_STATUS_SUCCESS ||
           resolved[0].state != HIPBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "[hipblasLt] WARNING: solution index %d does not exist; "
                         "stock heuristic for m=%d n=%d k=%d trans=%c%c\n",
                         tuned_index,
                         n,
                         m_g,
                         k,
                         ta_ch,
                         tb_ch);
        }
        else if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                     plan.desc,
                                                     &probe_alpha,
                                                     plan.a_layout,
                                                     plan.b_layout,
                                                     &probe_beta,
                                                     plan.c_layout,
                                                     plan.c_layout,
                                                     resolved[0].algo,
                                                     required) !=
                    HIPBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr,
                         "[hipblasLt] WARNING: solution index %d unsupported here; "
                         "stock heuristic for m=%d n=%d k=%d trans=%c%c\n",
                         tuned_index,
                         n,
                         m_g,
                         k,
                         ta_ch,
                         tb_ch);
        }
        else if(required > workspace_size)
        {
            std::fprintf(stderr,
                         "[hipblasLt] WARNING: solution index %d needs %zu bytes "
                         "workspace (have %zu); stock heuristic\n",
                         tuned_index,
                         required,
                         workspace_size);
        }
        else
        {
            plan.algo = resolved[0].algo;
            std::fprintf(stderr,
                         "[hipblasLt] tuned: m=%d n=%d k=%d trans=%c%c -> index=%d\n",
                         n,
                         m_g,
                         k,
                         ta_ch,
                         tb_ch,
                         tuned_index);
            hipblasLtMatmulPreferenceDestroy(pref);
            return true;
        }
    }

    hipblasLtMatmulHeuristicResult_t algo{};
    int algo_count = 0;
    hipblaslt_check_status(
        hipblasLtMatmulAlgoGetHeuristic(handle,
                                        plan.desc,
                                        plan.a_layout,
                                        plan.b_layout,
                                        plan.c_layout,
                                        plan.c_layout,
                                        pref,
                                        1,
                                        &algo,
                                        &algo_count),
        "hipblasLtMatmulAlgoGetHeuristic");
    const bool ok = algo_count > 0;
    if(ok)
    {
        plan.algo = algo.algo;
    }
    hipblasLtMatmulPreferenceDestroy(pref);
    return ok;
}

void hipblaslt_run_plan(const HipblasltPlan& plan,
                        hipblasLtHandle_t handle,
                        void* workspace,
                        std::size_t workspace_size,
                        hipStream_t stream,
                        const void* b_dev,
                        const void* a_dev,
                        void* d_dev)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    hipblaslt_check_status(
        hipblasLtMatmul(handle,
                        plan.desc,
                        &alpha,
                        b_dev,
                        plan.a_layout,
                        a_dev,
                        plan.b_layout,
                        &beta,
                        d_dev,
                        plan.c_layout,
                        d_dev,
                        plan.c_layout,
                        &plan.algo,
                        workspace,
                        workspace_size,
                        stream),
        "hipblasLtMatmul");
}

void hipblaslt_destroy_plan(HipblasltPlan& plan)
{
    hipblasLtMatrixLayoutDestroy(plan.c_layout);
    hipblasLtMatrixLayoutDestroy(plan.b_layout);
    hipblasLtMatrixLayoutDestroy(plan.a_layout);
    hipblasLtMatmulDescDestroy(plan.desc);
    plan = HipblasltPlan{};
}

// Element compare on the device: bad = |got - ref| > atol + rtol*|ref|,
// max_err tracked via the non-negative-float bit pattern (uint order == float
// order for non-negative values).
__global__ void blas_compare_kernel(const Bf16* __restrict__ got,
                                    const Bf16* __restrict__ ref,
                                    long long count,
                                    float atol,
                                    float rtol,
                                    unsigned long long* bad,
                                    unsigned int* max_bits)
{
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= count)
    {
        return;
    }
    const float g = ck_tile::type_convert<float>(got[i]);
    const float r = ck_tile::type_convert<float>(ref[i]);
    const float d = fabsf(g - r);
    if(d > atol + rtol * fabsf(r))
    {
        atomicAdd(bad, 1ull);
    }
    atomicMax(max_bits, __float_as_uint(d));
}

struct BlasCheckResult
{
    long long bad     = 0;
    double max_err    = 0.0;
};

BlasCheckResult blas_compare(hipStream_t stream,
                             const void* got,
                             const void* ref,
                             long long count)
{
    unsigned long long* bad_dev  = nullptr;
    unsigned int* max_bits_dev   = nullptr;
    hip_check(hipMalloc(&bad_dev, sizeof(*bad_dev)), "hipMalloc(bad)");
    hip_check(hipMalloc(&max_bits_dev, sizeof(*max_bits_dev)), "hipMalloc(max)");
    hip_check(hipMemset(bad_dev, 0, sizeof(*bad_dev)), "hipMemset(bad)");
    hip_check(hipMemset(max_bits_dev, 0, sizeof(*max_bits_dev)), "hipMemset(max)");

    const int block = 256;
    const long long grid = (count + block - 1) / block;
    blas_compare_kernel<<<static_cast<unsigned>(grid), block, 0, stream>>>(
        static_cast<const Bf16*>(got),
        static_cast<const Bf16*>(ref),
        count,
        1.0f,
        0.01f,
        bad_dev,
        max_bits_dev);
    hip_check(hipGetLastError(), "blas_compare_kernel launch");

    unsigned long long bad = 0;
    unsigned int max_bits  = 0;
    hip_check(hipMemcpyAsync(&bad, bad_dev, sizeof(bad), hipMemcpyDeviceToHost, stream),
              "hipMemcpyAsync(bad)");
    hip_check(hipMemcpyAsync(&max_bits, max_bits_dev, sizeof(max_bits), hipMemcpyDeviceToHost, stream),
              "hipMemcpyAsync(max)");
    hip_check(hipStreamSynchronize(stream), "blas_compare sync");
    hip_check(hipFree(bad_dev), "hipFree(bad)");
    hip_check(hipFree(max_bits_dev), "hipFree(max)");

    BlasCheckResult result;
    result.bad = static_cast<long long>(bad);
    // The bit pattern came back over memcpy: rebuild the non-negative float.
    static_assert(sizeof(max_bits) == sizeof(float), "bit transfer size");
    std::memcpy(&result.max_err, &max_bits, sizeof(result.max_err));
    return result;
}
#endif   // CK_TILE_V7_TEST_HAS_HIPBLASLT

struct DeviceCase
{
    std::vector<const void*> a_ptrs;
    std::vector<const void*> b_ptrs;
    std::vector<void*> c_ptrs;
    std::vector<void*> a_blocks;
    std::vector<void*> b_blocks;
    std::vector<void*> c_blocks;
    void* kargs          = nullptr;
    std::size_t kargs_sz = 0;
    bool allocated       = false;

    ~DeviceCase() { release(); }

    void release()
    {
        if(!allocated) return;
        for(auto* p : a_blocks) hip_check(hipFree(p), "hipFree");
        for(auto* p : b_blocks) hip_check(hipFree(p), "hipFree");
        for(auto* p : c_blocks) hip_check(hipFree(p), "hipFree");
        if(kargs != nullptr) hip_check(hipFree(kargs), "hipFree");
        a_blocks.clear(); b_blocks.clear(); c_blocks.clear();
        kargs     = nullptr;
        allocated = false;
    }

    void allocate(const Case& c, const Options& options)
    {
        a_ptrs.resize(c.g);
        b_ptrs.resize(c.g);
        c_ptrs.resize(c.g);
        a_blocks.resize(c.g);
        b_blocks.resize(c.g);
        c_blocks.resize(c.g);

        for(int g = 0; g < c.g; ++g)
        {
            const std::size_t a_bytes =
                static_cast<std::size_t>(c.a_elements[g]) * sizeof(Bf16);
            const std::size_t b_bytes =
                static_cast<std::size_t>(c.b_elements[g]) * sizeof(Bf16);
            const std::size_t c_bytes =
                static_cast<std::size_t>(c.m) * c.n * sizeof(Bf16);
            hip_check(hipMalloc(&a_blocks[g], align_to(a_bytes, 256)), "hipMalloc(A)");
            hip_check(hipMalloc(&b_blocks[g], align_to(b_bytes, 256)), "hipMalloc(B)");
            hip_check(hipMalloc(&c_blocks[g], align_to(c_bytes, 256)), "hipMalloc(C)");
            hip_check(hipMemcpy(a_blocks[g],
                                c.host_a[g].data(),
                                a_bytes,
                                hipMemcpyHostToDevice),
                      "hipMemcpy(A)");
            hip_check(hipMemcpy(b_blocks[g],
                                c.host_b[g].data(),
                                b_bytes,
                                hipMemcpyHostToDevice),
                      "hipMemcpy(B)");
            hip_check(hipMemset(c_blocks[g], 0, c_bytes), "hipMemset(C)");
            a_ptrs[g] = a_blocks[g];
            b_ptrs[g] = b_blocks[g];
            c_ptrs[g] = c_blocks[g];
        }

        // block_start/block_end must be filled: the framework's non-persistent
        // kernel path maps a 1D block id to (group, m tile, n tile) through them,
        // which is how the v7 uniform instances resolve their tile.
        const ck_tile::index_t n_tiles = (c.n + 255) / 256;
        ck_tile::index_t block_cursor = 0;
        std::vector<ck_tile::GemmTransKernelArg> host_kargs;
        host_kargs.reserve(c.g);
        for(int g = 0; g < c.g; ++g)
        {
            host_kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{a_ptrs[g]},
                                                                       {b_ptrs[g]},
                                                                       {},
                                                                       c_ptrs[g],
                                                                       c.group_m[g],
                                                                       c.n,
                                                                       c.k,
                                                                       {c.strides_a[g]},
                                                                       {c.strides_b[g]},
                                                                       {},
                                                                       c.n,
                                                                       1},
                                                          block_cursor,
                                                          block_cursor +
                                                              ((c.group_m[g] + 255) / 256) * n_tiles);
            block_cursor += ((c.group_m[g] + 255) / 256) * n_tiles;
        }
        (void)options;
        kargs_sz = host_kargs.size() * sizeof(ck_tile::GemmTransKernelArg);
        hip_check(hipMalloc(&kargs, kargs_sz), "hipMalloc(kargs)");
        hip_check(hipMemcpy(kargs, host_kargs.data(), kargs_sz, hipMemcpyHostToDevice),
                  "hipMemcpy(kargs)");
        allocated = true;
    }
};

// --dump=1: the first few host operands, the CPU reference and the device
// output. Values, not aggregates: an aggregate cannot tell a broken mapping
// from a broken reference.
void dump_case(const Case& c, const std::vector<void*>& c_blocks)
{
    std::printf("dump   host_a[0][0..7]:");
    for(int i = 0; i < 8; ++i)
        std::printf(" %g", static_cast<double>(ck_tile::type_convert<float>(c.host_a[0][i])));
    std::printf("\ndump   host_b[0][0..7]:");
    for(int i = 0; i < 8; ++i)
        std::printf(" %g", static_cast<double>(ck_tile::type_convert<float>(c.host_b[0][i])));
    std::printf("\ndump   host_d[0][0..7]:");
    for(int i = 0; i < 8; ++i)
        std::printf(" %g", static_cast<double>(c.host_d_ref[0][i]));
    std::vector<unsigned short> raw(16);
    hip_check(hipMemcpy(raw.data(),
                        c_blocks[0],
                        raw.size() * sizeof(unsigned short),
                        hipMemcpyDeviceToHost),
              "hipMemcpy(dump)");
    std::printf("\ndump   device[0][0..7]:");
    for(int i = 0; i < 8; ++i)
    {
        const Bf16 value = *reinterpret_cast<Bf16*>(&raw[i]);
        std::printf(" %g(0x%04x)",
                    static_cast<double>(ck_tile::type_convert<float>(value)),
                    raw[i]);
    }
    std::printf("\n");
}

void build_problem(const Case& c, ck_tile_hcu_grouped_gemm_problem_v1& problem)
{
    problem                     = {};
    problem.struct_size         = sizeof(problem);
    problem.abi_version         = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    problem.data_type           = CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1;
    problem.layout              = c.tn ? CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1
                                       : CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1;
    problem.group_count         = c.g;
    problem.problem_flags       = 0;
    problem.policy_disable_flags = 0;
    // Ragged per-group M: declared the way the ABI has it -- a device-lengths
    // dimension with a host-visible capacity (the summed rows, plus one tile of
    // rounding per group). A common M stays expressible too.
    const bool common_m = (c.total_m == c.m * c.g);
    problem.m           = {common_m ? CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1
                                    : CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1,
                           0,
                           common_m ? static_cast<std::int64_t>(c.m)
                                    : static_cast<std::int64_t>(c.total_m) + 256 * c.g};
    problem.n = {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, c.n};
    problem.k = {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, c.k};
}

int select_instance(const ck_tile_hcu_grouped_gemm_problem_v1& problem,
                    std::uint32_t& instance_id,
                    std::string& name)
{
    ck_tile_hcu_grouped_gemm_selection_v1 selection{};
    selection.struct_size = sizeof(selection);
    selection.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    const int status      = ck_tile_hcu_grouped_gemm_select_device_args_v1(&problem, &selection);
    if(status != 0) return status;
    instance_id = selection.instance_id;
    const char* instance_name =
        ck_tile_hcu_grouped_gemm_instance_name_v1(selection.instance_id);
    name = instance_name == nullptr ? std::string("?") : std::string(instance_name);
    return 0;
}

int run_instance(const ck_tile_hcu_grouped_gemm_problem_v1& problem,
                 void* kargs,
                 std::uint32_t instance_id,
                 std::uint32_t num_cu,
                 hipStream_t stream)
{
    ck_tile_hcu_grouped_gemm_launch_v1 launch{};
    launch.struct_size          = sizeof(launch);
    launch.abi_version          = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    launch.device_args          = kargs;
    launch.num_cu               = num_cu;
    launch.stream               = reinterpret_cast<void*>(stream);
    launch.expected_instance_id = instance_id;
    ck_tile_hcu_grouped_gemm_selection_v1 launched{};
    launched.struct_size = sizeof(launched);
    launched.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    return ck_tile_hcu_grouped_gemm_run_device_args_v1(&problem, &launch, &launched);
}

double time_instance(const ck_tile_hcu_grouped_gemm_problem_v1& problem,
                     void* kargs,
                     std::uint32_t instance_id,
                     std::uint32_t num_cu,
                     hipStream_t stream,
                     int warmup,
                     int iters)
{
    for(int i = 0; i < warmup; ++i)
    {
        run_instance(problem, kargs, instance_id, num_cu, stream);
    }
    hipEvent_t start{};
    hipEvent_t stop{};
    hip_check(hipEventCreate(&start), "hipEventCreate");
    hip_check(hipEventCreate(&stop), "hipEventCreate");
    hip_check(hipEventRecord(start, stream), "hipEventRecord");
    for(int i = 0; i < iters; ++i)
    {
        run_instance(problem, kargs, instance_id, num_cu, stream);
    }
    hip_check(hipEventRecord(stop, stream), "hipEventRecord");
    hip_check(hipEventSynchronize(stop), "hipEventSynchronize");
    float ms = 0.0f;
    hip_check(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
    hip_check(hipEventDestroy(start), "hipEventDestroy");
    hip_check(hipEventDestroy(stop), "hipEventDestroy");
    return static_cast<double>(ms) / iters;
}

struct CheckResult
{
    double max_err = 0.0;
    std::size_t bad = 0;
    std::size_t total = 0;
    // First few mismatches, as (group, row, col, got, expected): the pattern is
    // what identifies a broken tile/row/column mapping, and aggregate counts do
    // not carry it.
    struct Sample
    {
        int g;
        int row;
        int col;
        float got;
        float ref;
    };
    std::vector<Sample> samples;
};

CheckResult check_against_reference(const Case& c,
                                    const std::vector<void*>& blocks,
                                    int stride_c)
{
    CheckResult result;
    for(int g = 0; g < c.g; ++g)
    {
        std::vector<Bf16> host_d(static_cast<std::size_t>(c.m) * stride_c);
        hip_check(hipMemcpy(host_d.data(),
                            blocks[g],
                            host_d.size() * sizeof(Bf16),
                            hipMemcpyDeviceToHost),
                  "hipMemcpy(D)");
        for(int row = 0; row < c.m; ++row)
        {
            for(int col = 0; col < c.n; ++col)
            {
                const std::size_t at = static_cast<std::size_t>(row) * stride_c + col;
                const double got     = static_cast<double>(ck_tile::type_convert<float>(host_d[at]));
                const double ref =
                    static_cast<double>(c.host_d_ref[g][static_cast<std::size_t>(row) * c.n + col]);
                const double err = std::fabs(got - ref);
                if(err > result.max_err) result.max_err = err;
                ++result.total;
                if(err > 0.5)
                {
                    ++result.bad;
                    if(result.samples.size() < 8)
                    {
                        result.samples.push_back({g,
                                                  row,
                                                  col,
                                                  static_cast<float>(got),
                                                  static_cast<float>(ref)});
                    }
                }
            }
        }
    }
    return result;
}

void print_check(const char* tag, const CheckResult& checked)
{
    std::printf("check  %s max_err=%.6g bad=%zu/%zu\n",
                tag,
                checked.max_err,
                checked.bad,
                checked.total);
    for(const auto& s : checked.samples)
    {
        std::printf("       sample g=%d row=%d col=%d got=%.6g ref=%.6g\n",
                    s.g,
                    s.row,
                    s.col,
                    static_cast<double>(s.got),
                    static_cast<double>(s.ref));
    }
}

}   // namespace

int main(int argc, char* argv[])
{
    Options options;
    parse_args(argc, argv, options);

    Case c;
    build_case(options, c);
    if(options.check && options.cpu_ref)
    {
        cpu_reference(c);
    }

    DeviceCase device;
    device.allocate(c, options);

    ck_tile_hcu_grouped_gemm_problem_v1 problem{};
    build_problem(c, problem);

    const std::uint32_t num_cu = options.num_cu > 0
                                     ? static_cast<std::uint32_t>(options.num_cu)
                                     : static_cast<std::uint32_t>(ck_tile::get_num_cu());
    hipStream_t stream         = nullptr;
    std::printf("grid   num_cu=%u (device=%d)\n",
                num_cu,
                static_cast<int>(ck_tile::get_num_cu()));

    bool ok = true;

    const char* enable_switch = c.tn ? "CK_TILE_GROUPED_GEMM_ENABLE_BF16_V8_TN"
                                     : "CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN";
    const char* disable_switch = c.tn ? "CK_TILE_GROUPED_GEMM_DISABLE_BF16_V8_TN"
                                      : "CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN";

    // --- v7 instance -----------------------------------------------------
    setenv(enable_switch, "1", 1);
    unsetenv(disable_switch);

    std::uint32_t v7_id = 0;
    std::string v7_name;
    if(select_instance(problem, v7_id, v7_name) != 0)
    {
        std::printf("layout=%s result=FAIL reason=select_status\n", options.layout.c_str());
        return 1;
    }
    const std::uint32_t expected = c.tn ? kV8TnInstance : kV7NnInstance;
    std::printf("select layout=%s instance_id=%u expected=%u name=%s\n",
                options.layout.c_str(),
                v7_id,
                expected,
                v7_name.c_str());
    ok = ok && v7_id == expected;

    const int run_status =
        run_instance(problem, device.kargs, v7_id, num_cu, stream);
    if(run_status != 0)
    {
        std::printf("layout=%s result=FAIL reason=run_status=%d\n", options.layout.c_str(), run_status);
        return 1;
    }
    const double v7_ms =
        time_instance(problem, device.kargs, v7_id, num_cu, stream, options.warmup, options.iters);

    if(options.dump)
    {
        dump_case(c, device.c_ptrs);
    }

    if(options.check)
    {
        if(options.cpu_ref)
        {
            const CheckResult checked = check_against_reference(c, device.c_ptrs, c.n);
            print_check("port_vs_cpu", checked);
            ok = ok && checked.bad == 0;
        }
    }

    // --- hipBLASLt functional reference (device-side, no CPU pass) ----------
    if(options.hipblaslt_ref)
    {
#ifdef CK_TILE_V7_TEST_HAS_HIPBLASLT
        hipblasLtHandle_t handle = nullptr;
        hipblaslt_check_status(hipblasLtCreate(&handle), "hipblasLtCreate");
        void* workspace                   = nullptr;
        const std::size_t workspace_size  = 256ull << 20;
        hip_check(hipMalloc(&workspace, workspace_size), "hipMalloc(blas workspace)");

        std::vector<void*> ref_blocks(static_cast<std::size_t>(c.g), nullptr);
        std::vector<HipblasltPlan> plans(static_cast<std::size_t>(c.g));
        for(int g = 0; g < c.g; ++g)
        {
            const std::size_t bytes =
                static_cast<std::size_t>(c.group_m[g]) * c.n * sizeof(Bf16);
            hip_check(hipMalloc(&ref_blocks[g], bytes), "hipMalloc(blas ref)");
            hip_check(hipMemset(ref_blocks[g], 0, bytes), "hipMemset(blas ref)");
            if(!hipblaslt_make_plan(plans[g],
                                    handle,
                                    workspace,
                                    workspace_size,
                                    c.group_m[g],
                                    c.n,
                                    c.k,
                                    c.strides_a[g],
                                    c.tn))
            {
                std::fprintf(stderr, "hipBLASLt: no algorithm for group %d\n", g);
                std::exit(1);
            }
            // One plain GEMM per group on the same stream the instance used.
            hipblaslt_run_plan(plans[g],
                               handle,
                               workspace,
                               workspace_size,
                               stream,
                               device.b_blocks[g],
                               device.a_blocks[g],
                               ref_blocks[g]);
        }
        hip_check(hipStreamSynchronize(stream), "hipblaslt ref sync");

        long long blas_bad  = 0;
        double blas_max_err = 0.0;
        for(int g = 0; g < c.g; ++g)
        {
            const BlasCheckResult r =
                blas_compare(stream,
                             device.c_ptrs[g],
                             ref_blocks[g],
                             static_cast<long long>(c.group_m[g]) * c.n);
            blas_bad += r.bad;
            blas_max_err = blas_max_err > r.max_err ? blas_max_err : r.max_err;
            hip_check(hipFree(ref_blocks[g]), "hipFree(blas ref)");
        }
        std::printf("blas_check layout=%s bad=%lld max_err=%.4f tolerance=1.0+0.01*|ref|\n",
                    options.layout.c_str(),
                    blas_bad,
                    blas_max_err);
        ok = ok && blas_bad == 0;
        for(auto& plan : plans)
        {
            hipblaslt_destroy_plan(plan);
        }
        hip_check(hipFree(workspace), "hipFree(blas workspace)");
        hipblasLtDestroy(handle);
#else
        std::printf("blas_check result=FAIL reason=built_without_hipblaslt\n");
        ok = false;
#endif
    }

    // --- hipBLASLt performance reference --------------------------------------
    // Plans are built once per group; the timed loop issues bare
    // hipblasLtMatmul calls for all groups per iteration -- the launch pattern a
    // production consumer would use against the same buffers.
    if(options.hipblaslt_perf)
    {
#ifdef CK_TILE_V7_TEST_HAS_HIPBLASLT
        hipblasLtHandle_t handle = nullptr;
        hipblaslt_check_status(hipblasLtCreate(&handle), "hipblasLtCreate");
        void* workspace                  = nullptr;
        const std::size_t workspace_size = 256ull << 20;
        hip_check(hipMalloc(&workspace, workspace_size), "hipMalloc(blas workspace)");

        std::vector<HipblasltPlan> plans(static_cast<std::size_t>(c.g));
        bool plans_ok = true;
        for(int g = 0; g < c.g && plans_ok; ++g)
        {
            plans_ok = hipblaslt_make_plan(plans[g],
                                           handle,
                                           workspace,
                                           workspace_size,
                                           c.group_m[g],
                                           c.n,
                                           c.k,
                                           c.strides_a[g],
                                           c.tn);
        }
        if(!plans_ok)
        {
            std::printf("blas_perf layout=%s result=FAIL reason=no_algo\n",
                        options.layout.c_str());
            ok = false;
        }
        else
        {
            for(int i = 0; i < options.warmup; ++i)
            {
                for(int g = 0; g < c.g; ++g)
                {
                    hipblaslt_run_plan(plans[g],
                                       handle,
                                       workspace,
                                       workspace_size,
                                       stream,
                                       device.b_blocks[g],
                                       device.a_blocks[g],
                                       device.c_ptrs[g]);
                }
            }
            hipEvent_t start{};
            hipEvent_t stop{};
            hip_check(hipEventCreate(&start), "hipEventCreate");
            hip_check(hipEventCreate(&stop), "hipEventCreate");
            hip_check(hipEventRecord(start, stream), "hipEventRecord");
            for(int i = 0; i < options.iters; ++i)
            {
                for(int g = 0; g < c.g; ++g)
                {
                    hipblaslt_run_plan(plans[g],
                                       handle,
                                       workspace,
                                       workspace_size,
                                       stream,
                                       device.b_blocks[g],
                                       device.a_blocks[g],
                                       device.c_ptrs[g]);
                }
            }
            hip_check(hipEventRecord(stop, stream), "hipEventRecord");
            hip_check(hipEventSynchronize(stop), "hipEventSynchronize");
            float elapsed_ms = 0.0f;
            hip_check(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
            const double blas_ms =
                static_cast<double>(elapsed_ms) / static_cast<double>(options.iters);
            const double blas_tf =
                2.0 * static_cast<double>(c.total_m) * c.n * c.k /
                (blas_ms * 1.0e-3) / 1.0e12;
            std::printf("blas_perf layout=%s m=%d n=%d k=%d groups=%d "
                        "blas_ms=%.4f blas_tf=%.1f\n",
                        options.layout.c_str(),
                        c.m,
                        c.n,
                        c.k,
                        c.g,
                        blas_ms,
                        blas_tf);
            hip_check(hipEventDestroy(start), "hipEventDestroy");
            hip_check(hipEventDestroy(stop), "hipEventDestroy");
        }
        for(auto& plan : plans)
        {
            hipblaslt_destroy_plan(plan);
        }
        hip_check(hipFree(workspace), "hipFree(blas workspace)");
        hipblasLtDestroy(handle);
#else
        std::printf("blas_perf result=FAIL reason=built_without_hipblaslt\n");
        ok = false;
#endif
    }

    const double flops = 2.0 * static_cast<double>(c.total_m) * c.n * c.k;
    std::printf("perf   layout=%s m=%d n=%d k=%d groups=%d v7_ms=%.4f v7_tf=%.1f\n",
                options.layout.c_str(),
                c.m,
                c.n,
                c.k,
                c.g,
                v7_ms,
                flops / (v7_ms * 1.0e-3) / 1.0e12);

    // --- default CK instance for the same kargs (same-process A/B) ----------
    if(options.ref)
    {
        setenv(disable_switch, "1", 1);
        std::uint32_t ref_id = 0;
        std::string ref_name;
        if(select_instance(problem, ref_id, ref_name) != 0)
        {
            std::printf("ref    layout=%s result=SKIP reason=select_status\n", options.layout.c_str());
        }
        else if(ref_id == v7_id)
        {
            std::printf("ref    layout=%s result=SKIP reason=same_instance\n", options.layout.c_str());
        }
        else
        {
            // Contract check: the TN candidates in the registry consume the
            // TRANSPOSED view of the problem (C^T = B^T A), i.e. a different
            // kargs meaning than the direct view this driver builds for the v7
            // entry. Running them on these kargs would compare two different
            // problems, so skip instead of printing a meaningless number.
            ck_tile_hcu_grouped_gemm_candidate_info_v1 ref_info{};
            ref_info.struct_size = sizeof(ref_info);
            ref_info.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
            const bool ref_transposed_view =
                ck_tile_hcu_grouped_gemm_query_instance_v1(ref_id, &ref_info) == 0 &&
                (ref_info.contract_flags &
                 CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_CONTRACT_C_TRANSPOSED_VIEW_V1) != 0;

            const int ref_status = ref_transposed_view
                                       ? 0
                                       : run_instance(problem, device.kargs, ref_id, num_cu, stream);
            if(ref_transposed_view)
            {
                std::printf("ref    layout=%s instance_id=%u name=%s result=SKIP "
                            "reason=transposed_view_contract\n",
                            options.layout.c_str(),
                            ref_id,
                            ref_name.c_str());
            }
            else if(ref_status != 0)
            {
                std::printf("ref    layout=%s instance_id=%u result=SKIP reason=run_status=%d\n",
                            options.layout.c_str(),
                            ref_id,
                            ref_status);
            }
            else
            {
                const double ref_ms = time_instance(
                    problem, device.kargs, ref_id, num_cu, stream, options.warmup, options.iters);
                std::printf("ref    layout=%s instance_id=%u name=%s ref_ms=%.4f ref_tf=%.1f\n",
                            options.layout.c_str(),
                            ref_id,
                            ref_name.c_str(),
                            ref_ms,
                            flops / (ref_ms * 1.0e-3) / 1.0e12);
            }
        }
        unsetenv(disable_switch);
    }

    std::printf("layout=%s result=%s\n", options.layout.c_str(), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
