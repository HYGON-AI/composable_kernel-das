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

#include <hip/hip_runtime.h>

#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"

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

    const double flops = 2.0 * static_cast<double>(c.m) * c.n * c.k * c.g;
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
