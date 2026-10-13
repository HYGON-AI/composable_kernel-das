// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technologies Co., Ltd.
//
// Correctness + performance driver for the WASP (warp-specialized
// producer / consumer) NT grouped GEMM pipeline.
//
// grouped_gemm_wasp_geometry_smoke.cpp guards the one structural assumption of
// the design (a 12 wave workgroup carrying a 1 x 8 consumer warp grid). THIS
// file answers the two questions that guard cannot: does it compute the right
// numbers, and how fast is it.
//
// Design notes
// ------------
//  * kargs come from the FRAMEWORK's own host-side builder,
//    `GroupedGemmKernel<...>::MakeKargs(std::vector<GroupedGemmHostArgs>)`, so
//    the descriptor the kernel sees is field-for-field what the device-args ABI
//    would produce. Nothing is hand-filled, so a harness bug cannot hide behind
//    a hand-copied layout.
//  * A, B and C are laid out the way the ABI lays them out for NT:
//        A  [sum(M_g), K]  row major, stride_A = K   (group g at prefix_M[g]*K)
//        B  [G, N, K]      row major, stride_B = K   (group g at g*N*K)
//        C  [sum(M_g), N]  row major, stride_E = N   (group g at prefix_M[g]*N)
//    Each group's a_ptr/e_ptr points INTO the shared buffer, which is exactly
//    what lets the kernel recover the group table from the kargs alone.
//  * The oracle is an independent CPU reference (fp32 accumulate), not another
//    GPU kernel, so the check still means something when every GPU path is
//    wrong. bf16 products are exact in fp32, so only the summation ORDER
//    differs between the reference and the kernel: the tolerance can be tight.
//  * `--ref=1` (default) additionally runs the same kargs through an existing CK
//    configuration. That gives a triangle of evidence: if WASP and the CK
//    reference both disagree with the CPU reference, the harness is wrong; if
//    only WASP disagrees, it is the kernel.
//
// Current port limitations this driver enforces
// --------------------------------------------
// The port does not yet mask A rows past M, B columns past N, or a K tail, so
// only aligned shapes are legal: per-group M % 256 == 0, N % 256 == 0,
// K % 64 == 0. Unaligned shapes are reported as SKIPPED (not FAIL) unless
// --allow-unaligned is given, which is what you want when debugging the masking
// itself.
//
// Usage:
//   grouped_gemm_wasp_test [--m=...] [--n=...] [--k=...] [--groups=...]
//                          [--iters=N] [--warmup=N] [--ref=0|1]
//                          [--allow-unaligned] [--quiet]

#include "instances/grouped_gemm_impl.hpp"

#include <hip/hip_runtime.h>

#if defined(CK_TILE_GROUPED_GEMM_WASP_HAS_HIPBLASLT)
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Bf16 = ck_tile::bf16_t;
using Row  = ck_tile::tensor_layout::gemm::RowMajor;
using Col  = ck_tile::tensor_layout::gemm::ColumnMajor;

using WaspConfig = GemmConfigWasp256<Bf16>;

// ---------------------------------------------------------------------------
// The type chain `grouped_gemm_tileloop` assembles internally (see
// instances/grouped_gemm_impl.hpp:920-1033), rebuilt here so that this driver
// can reach the framework's host-side kargs builder. Only the kargs builder is
// used from it; the launches below go through grouped_gemm_tileloop, which
// assembles its own kernel per configuration.
// ---------------------------------------------------------------------------
using WaspShape = ck_tile::TileGemmShape<
    ck_tile::sequence<WaspConfig::M_Tile, WaspConfig::N_Tile, WaspConfig::K_Tile>,
    ck_tile::sequence<WaspConfig::M_Warp, WaspConfig::N_Warp, WaspConfig::K_Warp>,
    ck_tile::sequence<WaspConfig::M_Warp_Tile,
                      WaspConfig::N_Warp_Tile,
                      WaspConfig::K_Warp_Tile>>;

using WaspUniversalTraits = ck_tile::PersistentTileGemmUniversalTraits<
    WaspConfig::kPadM,
    WaspConfig::kPadN,
    WaspConfig::kPadK,
    WaspConfig::DoubleSmemBuffer,
    Row,
    Col,
    Row,
    WaspConfig::TransposeC,
    false,
    false,
    true,
    WaspConfig::SupportsFastRowMajorB,
    WaspConfig::SupportsFastColumnMajorA>;

using WaspProblem = ck_tile::UniversalGemmPipelineProblem<Bf16,
                                                          Bf16,
                                                          float,
                                                          WaspShape,
                                                          WaspUniversalTraits,
                                                          WaspConfig::Scheduler,
                                                          true,
                                                          ck_tile::TailNumber::Full,
                                                          ck_tile::null_type,
                                                          ck_tile::null_type,
                                                          false,
                                                          2,
                                                          2>;

using WaspPipeline =
    typename PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_WASP>::template GemmPipeline<WaspProblem>;

using WaspEpilogue = GroupedGemmWaspStoreEpilogue<float, Bf16, WaspPipeline>;

using WaspPartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
    WaspShape,
    WaspConfig::TileParitionerGroupNum,
    WaspConfig::TileParitionerM01>;

using WaspKernel = ck_tile::GroupedGemmKernel<WaspPartitioner, WaspPipeline, WaspEpilogue>;

// Independent CK reference for the same 256x256x64 NT tile.
using RefConfig = GemmConfigComputeV3TritonOperand256ExactTraversal<Bf16>;

// Only used to compare the two partitioners' grid derivation on the host.
using RefShape = ck_tile::TileGemmShape<
    ck_tile::sequence<RefConfig::M_Tile, RefConfig::N_Tile, RefConfig::K_Tile>,
    ck_tile::sequence<RefConfig::M_Warp, RefConfig::N_Warp, RefConfig::K_Warp>,
    ck_tile::sequence<RefConfig::M_Warp_Tile,
                      RefConfig::N_Warp_Tile,
                      RefConfig::K_Warp_Tile>>;

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

struct Options
{
    // The option vocabulary mirrors the standalone reference harness
    // (hipc_kernels/main.cpp) so that ONE shape script can drive both binaries
    // and the two columns are directly comparable:
    //   --m / --n / --k / --grouped-length / --layout / --output-dtype /
    //   --bench / --persistent-kernel / --warmup-iters / --bench-iters
    // The defaults match that harness too, so "run both with no arguments" means
    // the same problem; the small correctness shapes are always passed
    // explicitly by the script.
    int  m      = 4096;   // rows of the single group, when --grouped-length is absent
    int  n      = 4096;
    int  k      = 1024;
    int  groups = 1;      // shorthand for N equal groups of --m rows (extension)
    // One M per group, in group order. Empty => a single group of `m` rows.
    std::vector<int> group_lens;
    int  warmup            = 5;      // harness default
    int  iters             = 20;     // harness default
    bool bench             = false;  // harness semantics: timing only, no verdict
    bool ref               = true;
    bool allow_unaligned   = false;
    bool quiet             = false;
    bool  hipblaslt_ref     = false;
    bool  hipblaslt_perf    = false;
    bool  persistent_kernel = true;   // accepted for CLI parity; see main()
    std::string layout    = "NT";
    std::string out_dtype = "bf16";
    // bf16 operands: the products are exact in fp32, so the CPU oracle and the
    // kernel differ only by summation order. In practice the observed noise
    // floor of this comparison is ~one bf16 ULP (2^-8), so the default is two
    // orders of magnitude above it -- tight enough to catch a wrong mapping,
    // loose enough not to flag rounding.
    float rel_tol = 1.0e-2F;
    float abs_tol = 1.0e-2F;
    // Input pattern. These exist to make the OUTPUT self-describing, which is
    // what turns "26% of elements are wrong" into an attributable diagnosis:
    //   random : the real workload
    //   ones   : every output must be exactly K. Any deviation is a value/K
    //            problem (a permutation of correctly computed values would still
    //            read K), so this separates "wrong value" from "wrong location".
    //   bindex : A[m,k] = (k == 0), B[n,k] = n. Then C[m,n] = n exactly, i.e.
    //            every output element PRINTS the B row it was computed from, so
    //            a wrong or shifted B stripe is visible directly in the values.
    std::string pattern = "random";
    bool debug_ctas     = false;
    // When >= 0, print row `dump_row` of group 0 across n in steps of 8. With
    // --pattern=bindex the printed numbers ARE the B row each output column was
    // computed from, so a shifted/stale consumer stripe shows up as a step at
    // the exact column where the mapping breaks.
    int dump_row = -1;
    // Ablation mask for the timing decomposition (see gemm_pipeline_ag_bg_cr_wasp.hpp):
    // 1 = no B loads, 2 = no MMACs, 4 = every A read from row 0, 8 = no barriers.
    // Needs CK_TILE_GROUPED_GEMM_WASP_ABLATION. Results become garbage by design.
    int ablate = 0;
};

// Round-to-nearest-even fp32 -> bf16, returned as the exact float value the
// bf16 can hold. Doing it here (instead of relying on a conversion) means the
// CPU reference and the device buffer are guaranteed to see identical inputs.
float round_to_bf16(float x)
{
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    const std::uint32_t lsb = (u >> 16) & 1u;
    if((u & 0x7FFFFFFFu) > 0x7F800000u)
    {
        return x;   // NaN / inf: leave alone, the test data never contains one
    }
    u += 0x7FFFu + lsb;
    u &= 0xFFFF0000u;
    float r = 0.0F;
    std::memcpy(&r, &u, sizeof(r));
    return r;
}

// Deterministic, portable LCG. Values in [-0.5, 0.5).
float next_value(std::uint32_t& state)
{
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>(state >> 8) / static_cast<float>(1u << 24)) - 0.5F;
}

template <typename T>
class DevBuf
{
    public:
    DevBuf() = default;
    DevBuf(const DevBuf&)            = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf()
    {
        if(ptr_ != nullptr)
        {
            (void)hipFree(ptr_);
        }
    }

    bool Alloc(std::size_t count)
    {
        return hipMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)) == hipSuccess;
    }
    T*                         Data() { return ptr_; }
    const T*                   Data() const { return ptr_; }
    bool Upload(const std::vector<T>& src, std::size_t count)
    {
        return hipMemcpy(ptr_, src.data(), count * sizeof(T), hipMemcpyHostToDevice) == hipSuccess;
    }
    bool Download(std::vector<T>& dst, std::size_t count) const
    {
        return hipMemcpy(dst.data(), ptr_, count * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess;
    }

    private:
    T* ptr_ = nullptr;
};

template <typename T>
bool UploadRaw(T* dst, const void* src, std::size_t bytes)
{
    return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) == hipSuccess;
}

int parse_int(const char* text, int fallback)
{
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if(end == text || v < 0 || v > 100000000L)
    {
        return fallback;
    }
    return static_cast<int>(v);
}

float parse_float(const char* text, float fallback)
{
    char*       end = nullptr;
    const float v   = std::strtof(text, &end);
    if(end == text || !(v >= 0.0F))
    {
        return fallback;
    }
    return v;
}

// ---------------------------------------------------------------------------
// The problem shape, resolved from Options. Per-group M is a first-class thing
// here (not an equal split) because every real NT shape in the reference
// harness is ragged: 8152,8152,8257,8257 / 1923,2047,2179,2043 / 5120,1536,... .
// `prefix` carries the row offsets that A and C use, so group g's rows live at
// [prefix[g], prefix[g+1]) of the concatenated operand -- the same convention
// the reference harness and the device-args ABI use.
// ---------------------------------------------------------------------------
struct ShapeInfo
{
    std::vector<int>  lens;
    std::vector<long> prefix;   // size == lens.size() + 1
    long              sum_m = 0;
    int               n     = 0;
    int               k     = 0;

    int groups() const { return static_cast<int>(lens.size()); }

    // Group owning a GLOBAL output row. Ragged groups mean this cannot be a
    // division by a per-group M.
    int group_of(long row) const
    {
        for(int g = static_cast<int>(lens.size()) - 1; g >= 0; --g)
        {
            if(row >= prefix[static_cast<std::size_t>(g)])
            {
                return g;
            }
        }
        return 0;
    }
    int row_in_group(long row) const
    {
        const int g = group_of(row);
        return static_cast<int>(row - prefix[static_cast<std::size_t>(g)]);
    }
};

ShapeInfo make_shape(const Options& opt)
{
    ShapeInfo sh;
    if(!opt.group_lens.empty())
    {
        sh.lens = opt.group_lens;
    }
    else
    {
        sh.lens.assign(static_cast<std::size_t>(opt.groups > 0 ? opt.groups : 1), opt.m);
    }
    sh.prefix.resize(sh.lens.size() + 1);
    long off = 0;
    for(std::size_t g = 0; g < sh.lens.size(); ++g)
    {
        sh.prefix[g] = off;
        off += sh.lens[g];
    }
    sh.prefix[sh.lens.size()] = off;
    sh.sum_m                  = off;
    sh.n                      = opt.n;
    sh.k                      = opt.k;
    return sh;
}

// "512,512" / "8152,8152,8257,8257". Returns false on a malformed list.
bool parse_group_lens(const char* text, std::vector<int>& out)
{
    out.clear();
    const std::string s(text);
    std::size_t       begin = 0;
    for(;;)
    {
        const std::size_t comma = s.find(',', begin);
        const std::string tok =
            s.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if(tok.empty())
        {
            return false;
        }
        char*      end = nullptr;
        const long v   = std::strtol(tok.c_str(), &end, 10);
        if(end == tok.c_str() || *end != '\0' || v <= 0 || v > 100000000L)
        {
            return false;
        }
        out.push_back(static_cast<int>(v));
        if(comma == std::string::npos)
        {
            break;
        }
        begin = comma + 1;
    }
    return !out.empty();
}

void print_usage()
{
    std::printf(
        "usage: grouped_gemm_wasp_test [--m=N] [--n=N] [--k=N] [--grouped-length=a,b,c]\n"
        "                              [--layout=NT] [--output-dtype=bf16]\n"
        "                              [--bench=0|1] [--persistent-kernel=1]\n"
        "                              [--warmup-iters=N] [--bench-iters=N]\n"
        "                              [--ref=0|1] [--rel-tol=F] [--abs-tol=F]\n"
        "                              [--hipblaslt-ref=0|1] [--hipblaslt-perf=0|1]\n"
        "                              [--allow-unaligned] [--quiet]\n"
        "  extras not present in the reference harness:\n"
        "                              [--groups=N]   equal groups of --m rows\n"
        "                              [--warmup=N] [--iters=N]  aliases\n"
        "                              [--pattern=random|ones|bindex|aindex]\n"
        "                              [--dump-row=N] [--debug-ctas]\n");
}

// ---------------------------------------------------------------------------
// One problem instance
// ---------------------------------------------------------------------------

struct CompareStats
{
    double      max_abs = 0.0;
    double      max_rel = 0.0;
    std::size_t bad     = 0;
};

// One failing output element, with its grouped coordinates. The PATTERN of these
// is what localises a broken kernel: a wrong consumer stripe shows up as a whole
// n range, a wrong MMAC fragment map as a stride-16 row pattern, a missing k tail
// as a uniform small error.
struct Mismatch
{
    int   g   = 0;
    int   m   = 0;
    int   n   = 0;
    float ref = 0.0F;
    float got = 0.0F;
};

struct CaseResult
{
    bool skipped  = false;
    bool passed   = false;
    double ms     = 0.0;
    double tflops = 0.0;
    bool cpu_ran  = false;
    double wasp_max_abs = 0.0;
    double wasp_max_rel = 0.0;
    std::size_t wasp_bad = 0;
    bool ref_ran = false;
    double ref_max_abs = 0.0;
    double ref_max_rel = 0.0;
    std::size_t ref_bad = 0;
    // WASP vs the CK reference configuration, run on the same kargs. Once a shape
    // outgrows the CPU oracle's budget this is the ONLY oracle available, so it
    // is what gives a large shape a verdict at all.
    bool        wasp_vs_ref_ran     = false;
    double      wasp_vs_ref_max_abs = 0.0;
    std::size_t wasp_vs_ref_bad     = 0;
    bool        wasp_vs_hipblaslt_ran = false;
    double      wasp_vs_hipblaslt_max_abs = 0.0;
    std::size_t wasp_vs_hipblaslt_bad = 0;
    double      hipblaslt_ms = 0.0;
    std::vector<Mismatch> wasp_examples;
    std::vector<Mismatch> ref_examples;
    // Per-32-column-stripe and per-(row%16) failure counts. These two
    // histograms localise almost every class of bug in this kernel: a bad
    // consumer stripe shows up in the first, a bad MMAC fragment mapping or a
    // bad LDS row stride in the second.
    std::size_t bad_per_stripe[8] = {};
    std::size_t bad_per_row16[16] = {};
    // Per 16-row block, i.e. per m-iteration of a consumer's 256-row stripe:
    // isolates a bad MMAC fragment map from a bad store.
    std::size_t bad_per_miter[16] = {};
    // Outputs still equal to the sentinel, i.e. NEVER WRITTEN by either kernel.
    // Distinguishes "the epilogue stored the wrong value" from "the epilogue
    // never reached this output".
    std::size_t wasp_not_written = 0;
    // First element of the second half of the output, used only with
    // --debug-ctas: the epilogue writes a 777 probe there.
    float probe_cell = 0.0F;
    std::vector<float> dump_ref;
    std::vector<float> dump_got;
};

CompareStats compare(const std::vector<float>& ref,
                     const std::vector<float>& got,
                     const ShapeInfo&          sh,
                     float                     rel_tol,
                     float                     abs_tol,
                     std::vector<Mismatch>*    examples,
                     std::size_t               max_examples)
{
    CompareStats s{};
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double r    = ref[i];
        const double g    = got[i];
        const double diff = std::fabs(r - g);
        if(diff > s.max_abs)
        {
            s.max_abs = diff;
        }
        const double rel = diff / (std::fabs(r) + 1e-6);
        if(rel > s.max_rel)
        {
            s.max_rel = rel;
        }
        if(diff > static_cast<double>(rel_tol) * std::fabs(r) + static_cast<double>(abs_tol))
        {
            ++s.bad;
            if(examples != nullptr && examples->size() < max_examples)
            {
                const long row = static_cast<long>(i) / sh.n;
                const int  col = static_cast<int>(i) % sh.n;
                examples->push_back(Mismatch{sh.group_of(row),
                                             sh.row_in_group(row),
                                             col,
                                             ref[i],
                                             got[i]});
            }
        }
    }
    return s;
}

#if defined(CK_TILE_GROUPED_GEMM_WASP_HAS_HIPBLASLT)
struct TunedSolution
{
    long long m = -1;
    long long n = -1;
    long long k = -1;
    char ta = '?';
    char tb = '?';
    int index = -1;
};

const std::vector<TunedSolution>& tuned_solutions()
{
    static const std::vector<TunedSolution> table = [] {
        std::vector<TunedSolution> entries;
        const char* disable = std::getenv("BF16GEMM_NO_TUNING");
        if(disable != nullptr && std::atoi(disable) != 0)
        {
            return entries;
        }

        const char* path = std::getenv("BF16GEMM_HIPBLASLT_SOLUTION_FILE");
        if(path == nullptr || *path == '\0')
        {
            path = "/root/workspace/bf16_gemm/tune/hipblaslt_solutions.conf";
        }
        FILE* file = std::fopen(path, "r");
        if(file == nullptr)
        {
            std::fprintf(stderr,
                         "[hipBLASLt] WARNING: cannot open tune file '%s'; using heuristic\n",
                         path);
            return entries;
        }

        char line[512];
        while(std::fgets(line, sizeof(line), file) != nullptr)
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
            int field_count = 0;
            for(char* token = std::strtok(p, ", \t\r\n");
                token != nullptr && field_count < 6;
                token = std::strtok(nullptr, ", \t\r\n"))
            {
                fields[field_count++] = token;
            }
            if(field_count < 6 || fields[3].empty() || fields[4].empty())
            {
                std::fprintf(stderr, "[hipBLASLt] WARNING: malformed tune entry ignored\n");
                continue;
            }

            auto parse_dimension = [](const std::string& value, long long& output) {
                if(value == "*")
                {
                    output = -1;
                    return true;
                }
                char* end = nullptr;
                output = std::strtoll(value.c_str(), &end, 10);
                return end != value.c_str() && *end == '\0';
            };
            TunedSolution entry;
            if(!parse_dimension(fields[0], entry.m) ||
               !parse_dimension(fields[1], entry.n) ||
               !parse_dimension(fields[2], entry.k))
            {
                std::fprintf(stderr, "[hipBLASLt] WARNING: malformed tune dimensions ignored\n");
                continue;
            }
            entry.ta = static_cast<char>(
                std::toupper(static_cast<unsigned char>(fields[3][0])));
            entry.tb = static_cast<char>(
                std::toupper(static_cast<unsigned char>(fields[4][0])));
            char* end = nullptr;
            entry.index = static_cast<int>(std::strtol(fields[5].c_str(), &end, 10));
            if(end == fields[5].c_str() || *end != '\0' || entry.index < 0)
            {
                std::fprintf(stderr, "[hipBLASLt] WARNING: malformed tune index ignored\n");
                continue;
            }
            entries.push_back(entry);
        }
        std::fclose(file);
        std::fprintf(stderr,
                     "[hipBLASLt] loaded %zu tuned solutions from %s\n",
                     entries.size(),
                     path);
        return entries;
    }();
    return table;
}

bool tuned_solution_matches(const TunedSolution& entry,
                            long long m,
                            long long n,
                            long long k,
                            char ta,
                            char tb)
{
    return (entry.m < 0 || entry.m == m) && (entry.n < 0 || entry.n == n) &&
           (entry.k < 0 || entry.k == k) &&
           (entry.ta == '*' || entry.ta == '?' || entry.ta == ta) &&
           (entry.tb == '*' || entry.tb == '?' || entry.tb == tb);
}

bool pick_tuned_solution(long long m,
                         long long n,
                         long long k,
                         char ta,
                         char tb,
                         int& index)
{
    int best_score = 6;
    bool found = false;
    for(const TunedSolution& entry : tuned_solutions())
    {
        if(!tuned_solution_matches(entry, m, n, k, ta, tb))
        {
            continue;
        }
        const int score = (entry.m < 0 ? 1 : 0) + (entry.n < 0 ? 1 : 0) +
                          (entry.k < 0 ? 1 : 0) +
                          ((entry.ta == '*' || entry.ta == '?') ? 1 : 0) +
                          ((entry.tb == '*' || entry.tb == '?') ? 1 : 0);
        if(score < best_score)
        {
            best_score = score;
            index = entry.index;
            found = true;
            if(score == 0)
            {
                break;
            }
        }
    }
    return found;
}
#endif

CaseResult run_case(const Options& opt, const ShapeInfo& sh, int num_cu, hipStream_t stream)
{
    CaseResult res{};

    const int    groups = sh.groups();
    const int    n      = sh.n;
    const int    k      = sh.k;
    const long   sum_m  = sh.sum_m;
    const double macs   = static_cast<double>(sum_m) * n * k;

    // STAGE 4 (M tail) is in place: a ragged per-group M is masked -- the
    // producer zero-fills the A rows past M and the epilogue drops the output
    // rows past M. N and K tails are NOT masked yet, so N must still be a whole
    // number of N-tiles and K a whole number of K-tiles. The real NT reference
    // shapes (N=4096/2048, K=4096/2048) all satisfy that; only their M is
    // ragged (8152, 8257, 1923, ...), which is now legal.
    bool aligned =
        (n % WaspConfig::N_Tile == 0) && (k % WaspConfig::K_Tile == 0);
    if(!aligned && !opt.allow_unaligned)
    {
        res.skipped = true;
        return res;
    }

    // ---- host data ----
    std::vector<float> a_host(static_cast<std::size_t>(sum_m) * k);
    std::vector<float> b_host(static_cast<std::size_t>(groups) * n * k);
    std::vector<float> c_ref(static_cast<std::size_t>(sum_m) * n, 0.0F);
    // CK-reference output, hoisted to function scope so the diagnostics below
    // can use it as the oracle once a shape outgrows the CPU reference.
    std::vector<float> ref_out;

    if(opt.pattern == "ones")
    {
        for(auto& v : a_host)
        {
            v = 1.0F;
        }
        for(auto& v : b_host)
        {
            v = 1.0F;
        }
    }
    else if(opt.pattern == "bindex")
    {
        for(std::size_t i = 0; i < a_host.size(); ++i)
        {
            a_host[i] = (i % static_cast<std::size_t>(k) == 0) ? 1.0F : 0.0F;
        }
        for(int g = 0; g < groups; ++g)
        {
            for(int j = 0; j < n; ++j)
            {
                for(int kk = 0; kk < k; ++kk)
                {
                    b_host[(static_cast<std::size_t>(g) * n + j) * k + kk] =
                        static_cast<float>(j);
                }
            }
        }
    }
    else if(opt.pattern == "aindex")
    {
        // Mirror of bindex for the A side: A[row][k] = row (bf16-exact up to
        // 255) and B[col][k] = (k == 0), so C[row][col] = A[row][0] = row.
        // Every output element therefore PRINTS the A row it was actually
        // computed from -- a wrong row shows up as a wrong (but valid) number,
        // an out-of-bounds read as garbage.
        // The printed value is the row WITHIN the group, which bf16 represents
        // exactly only up to 256 -- larger groups round, and the readout is then
        // approximate. Fine for the debugger it is.
        for(int g = 0; g < groups; ++g)
        {
            const long row0 = sh.prefix[static_cast<std::size_t>(g)];
            const int  m_g  = sh.lens[static_cast<std::size_t>(g)];
            for(int i = 0; i < m_g; ++i)
            {
                for(int kk = 0; kk < k; ++kk)
                {
                    a_host[(static_cast<std::size_t>(row0) + static_cast<std::size_t>(i)) * k +
                           kk] = static_cast<float>(i);
                }
            }
        }
        for(auto& v : b_host)
        {
            v = 0.0F;
        }
        for(int g = 0; g < groups; ++g)
        {
            for(int j = 0; j < n; ++j)
            {
                b_host[(static_cast<std::size_t>(g) * n + j) * k] = 1.0F;
            }
        }
    }
    else
    {
        std::uint32_t state = 12345u;
        for(auto& v : a_host)
        {
            v = round_to_bf16(next_value(state));
        }
        for(auto& v : b_host)
        {
            v = round_to_bf16(next_value(state));
        }
    }

    // ---- device buffers ----
    DevBuf<Bf16> a_dev;
    DevBuf<Bf16> b_dev;
    DevBuf<Bf16> c_dev;
    if(!a_dev.Alloc(a_host.size()) || !b_dev.Alloc(b_host.size()) || !c_dev.Alloc(c_ref.size()))
    {
        std::printf("  ERROR: device allocation failed\n");
        return res;
    }

    // The host arrays hold exact bf16 values, so this conversion is exact.
    std::vector<Bf16> a_bf16(a_host.size());
    std::vector<Bf16> b_bf16(b_host.size());
    for(std::size_t i = 0; i < a_host.size(); ++i)
    {
        a_bf16[i] = ck_tile::type_convert<Bf16>(a_host[i]);
    }
    for(std::size_t i = 0; i < b_host.size(); ++i)
    {
        b_bf16[i] = ck_tile::type_convert<Bf16>(b_host[i]);
    }
    if(!a_dev.Upload(a_bf16, a_bf16.size()) || !b_dev.Upload(b_bf16, b_bf16.size()))
    {
        std::printf("  ERROR: device upload failed\n");
        return res;
    }

    // Poison the output so that "never written" is distinguishable from "written
    // with a wrong value". 1024 is exactly representable in bf16, and no GEMM in
    // this test can produce it from unit-magnitude inputs.
    constexpr float kSentinel = 1024.0F;
    {
        std::vector<Bf16> poison(c_ref.size(), ck_tile::type_convert<Bf16>(kSentinel));
        if(!c_dev.Upload(poison, poison.size()))
        {
            std::printf("  ERROR: output poison upload failed\n");
            return res;
        }
    }

    // ---- kargs via the framework builder ----
    std::vector<ck_tile::GroupedGemmHostArgs> hdescs;
    hdescs.reserve(static_cast<std::size_t>(groups));
    for(int g = 0; g < groups; ++g)
    {
        const long  row0 = sh.prefix[static_cast<std::size_t>(g)];
        const void* a_g  = a_dev.Data() + static_cast<std::size_t>(row0) * k;
        const void* b_g  = b_dev.Data() + static_cast<std::size_t>(g) * n * k;
        void*       c_g  = c_dev.Data() + static_cast<std::size_t>(row0) * n;
        hdescs.emplace_back(a_g,
                            b_g,
                            c_g,
                            1,         // k_batch
                            sh.lens[static_cast<std::size_t>(g)],   // M of this group
                            n,
                            k,
                            k,          // stride_A: A is [sumM, K] row major
                            k,          // stride_B: B is [N, K] with K contiguous (NT)
                            n);         // stride_E
    }
    const auto kargs = WaspKernel::MakeKargs(hdescs);
    if(kargs.size() != static_cast<std::size_t>(groups))
    {
        std::printf("  ERROR: MakeKargs produced %zu kargs for %d groups\n",
                    kargs.size(),
                    groups);
        return res;
    }
    if(!WaspKernel::IsSupportedArgument(kargs))
    {
        std::printf("  ERROR: Wasp requires gfx936 BF16 NT with aligned N/K\n");
        return res;
    }
    const ck_tile::index_t flat_grid_size = WaspKernel::GridSize(hdescs).x;

    DevBuf<ck_tile::GemmTransKernelArg> kargs_dev;
    if(!kargs_dev.Alloc(kargs.size()) ||
       !UploadRaw(kargs_dev.Data(), kargs.data(), kargs.size() * sizeof(kargs[0])))
    {
        std::printf("  ERROR: kargs upload failed\n");
        return res;
    }

    // ---- warmup + timed launches ----
    const ck_tile::stream_config sc{stream};
    for(int i = 0; i < opt.warmup; ++i)
    {
        (void)grouped_gemm_tileloop<WaspConfig,
                                    Row,
                                    Col,
                                    Row,
                                    Bf16,
                                    Bf16,
                                    float,
                                    Bf16>(sc,
                                         groups,
                                         kargs_dev.Data(),
                                         {},
                                         false,
                                         0,
                                         flat_grid_size);
    }
    (void)hipDeviceSynchronize();

    hipEvent_t ev0 = nullptr;
    hipEvent_t ev1 = nullptr;
    if(hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess)
    {
        std::printf("  ERROR: hipEventCreate failed\n");
        return res;
    }
    (void)hipEventRecord(ev0, stream);
    for(int i = 0; i < opt.iters; ++i)
    {
        (void)grouped_gemm_tileloop<WaspConfig,
                                    Row,
                                    Col,
                                    Row,
                                    Bf16,
                                    Bf16,
                                    float,
                                    Bf16>(sc,
                                         groups,
                                         kargs_dev.Data(),
                                         {},
                                         false,
                                         0,
                                         flat_grid_size);
    }
    (void)hipEventRecord(ev1, stream);
    const hipError_t sync_err = hipEventSynchronize(ev1);
    float            elapsed_ms = 0.0F;
    (void)hipEventElapsedTime(&elapsed_ms, ev0, ev1);
    {
        // Every HIP call in this driver was previously unchecked, which hides a
        // device-side fault behind a plausible-looking timing number.
        const hipError_t last = hipGetLastError();
        if(sync_err != hipSuccess)
        {
            std::printf("  ERROR: hipEventSynchronize: %s\n", hipGetErrorString(sync_err));
        }
        if(last != hipSuccess)
        {
            std::printf("  ERROR: hipGetLastError after launches: %s\n",
                        hipGetErrorString(last));
        }
    }
    (void)hipEventDestroy(ev0);
    (void)hipEventDestroy(ev1);

    res.ms     = static_cast<double>(elapsed_ms) / opt.iters;
    res.tflops = res.ms > 0.0 ? (2.0 * macs / (res.ms * 1.0e-3)) / 1.0e12 : 0.0;

    // ---- read back and compare ----
    std::vector<Bf16> c_host(c_ref.size());
    if(!c_dev.Download(c_host, c_host.size()))
    {
        std::printf("  ERROR: device download failed\n");
        return res;
    }
    std::vector<float> wasp_out(c_ref.size());
    for(std::size_t i = 0; i < c_ref.size(); ++i)
    {
        wasp_out[i] = ck_tile::type_convert<float>(c_host[i]);
    }

    // ---- CPU reference ----
    const bool run_cpu = macs <= 2.0e9;
    if(run_cpu)
    {
        for(int g = 0; g < groups; ++g)
        {
            const long row0 = sh.prefix[static_cast<std::size_t>(g)];
            const int  m_g  = sh.lens[static_cast<std::size_t>(g)];
            for(int i = 0; i < m_g; ++i)
            {
                const float* a_row = &a_host[static_cast<std::size_t>(row0 + i) * k];
                const float* b_mat = &b_host[static_cast<std::size_t>(g) * n * k];
                for(int j = 0; j < n; ++j)
                {
                    const float* b_row = b_mat + static_cast<std::size_t>(j) * k;
                    double       acc   = 0.0;
                    for(int kk = 0; kk < k; ++kk)
                    {
                        acc += static_cast<double>(a_row[kk]) * static_cast<double>(b_row[kk]);
                    }
                    c_ref[static_cast<std::size_t>(row0 + i) * n + j] = static_cast<float>(acc);
                }
            }
        }
        const CompareStats ws = compare(c_ref,
                                        wasp_out,
                                        sh,
                                        opt.rel_tol,
                                        opt.abs_tol,
                                        &res.wasp_examples,
                                        8);
        res.cpu_ran           = true;
        res.wasp_max_abs      = ws.max_abs;
        res.wasp_max_rel      = ws.max_rel;
        res.wasp_bad          = ws.bad;
    }

    // ---- optional CK reference on the same kargs ----
    if(opt.ref)
    {
        DevBuf<Bf16> c_ref_dev;
        if(c_ref_dev.Alloc(c_ref.size()))
        {
            std::vector<ck_tile::GroupedGemmHostArgs> ref_descs;
            ref_descs.reserve(static_cast<std::size_t>(groups));
            for(int g = 0; g < groups; ++g)
            {
                const long  row0 = sh.prefix[static_cast<std::size_t>(g)];
                const void* a_g  = a_dev.Data() + static_cast<std::size_t>(row0) * k;
                const void* b_g  = b_dev.Data() + static_cast<std::size_t>(g) * n * k;
                void*       c_g  = c_ref_dev.Data() + static_cast<std::size_t>(row0) * n;
                ref_descs.emplace_back(
                    a_g, b_g, c_g, 1, sh.lens[static_cast<std::size_t>(g)], n, k, k, k, n);
            }
            const auto ref_kargs = WaspKernel::MakeKargs(ref_descs);
            DevBuf<ck_tile::GemmTransKernelArg> ref_kargs_dev;
            if(ref_kargs_dev.Alloc(ref_kargs.size()) &&
               UploadRaw(ref_kargs_dev.Data(),
                         ref_kargs.data(),
                         ref_kargs.size() * sizeof(ref_kargs[0])))
            {
                (void)grouped_gemm_tileloop<RefConfig,
                                            Row,
                                            Col,
                                            Row,
                                            Bf16,
                                            Bf16,
                                            float,
                                            Bf16>(sc,
                                                 groups,
                                                 ref_kargs_dev.Data(),
                                                 {},
                                                 false,
                                                 static_cast<std::uint32_t>(num_cu));
                std::vector<Bf16> ref_host(c_ref.size());
                if(c_ref_dev.Download(ref_host, ref_host.size()))
                {
                    ref_out.assign(c_ref.size(), 0.0F);
                    for(std::size_t i = 0; i < ref_out.size(); ++i)
                    {
                        ref_out[i] = ck_tile::type_convert<float>(ref_host[i]);
                    }
                    res.ref_ran = true;
                    if(run_cpu)
                    {
                        const CompareStats rs = compare(c_ref,
                                                        ref_out,
                                                        sh,
                                                        opt.rel_tol,
                                                        opt.abs_tol,
                                                        &res.ref_examples,
                                                        4);
                        res.ref_max_abs       = rs.max_abs;
                        res.ref_max_rel       = rs.max_rel;
                        res.ref_bad           = rs.bad;
                    }
                    {
                        const CompareStats wr = compare(wasp_out,
                                                        ref_out,
                                                        sh,
                                                        opt.rel_tol,
                                                        opt.abs_tol,
                                                        nullptr,
                                                        0);
                        res.wasp_vs_ref_ran     = true;
                        res.wasp_vs_ref_max_abs = wr.max_abs;
                        res.wasp_vs_ref_bad     = wr.bad;
                    }
                }
            }
        }
    }

    if(opt.hipblaslt_ref || opt.hipblaslt_perf)
    {
#if defined(CK_TILE_GROUPED_GEMM_WASP_HAS_HIPBLASLT)
        hipblasLtHandle_t handle = nullptr;
        hipblasLtMatmulDesc_t desc = nullptr;
        hipblasLtMatrixLayout_t b_layout = nullptr;
        hipblasLtMatrixLayout_t a_layout = nullptr;
        hipblasLtMatrixLayout_t d_layout = nullptr;
        hipblasLtMatmulPreference_t preference = nullptr;
        hipblasLtMatmulAlgo_t algo{};
        constexpr std::size_t workspace_size = 256ull << 20;
        void* workspace = nullptr;
        DevBuf<Bf16> blas_out;

        auto blas_check = [](hipblasStatus_t status, const char* operation) {
            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                std::fprintf(stderr,
                             "hipBLASLt error %d at %s\n",
                             static_cast<int>(status),
                             operation);
                std::exit(1);
            }
        };
        auto hip_runtime_check = [](hipError_t status, const char* operation) {
            if(status != hipSuccess)
            {
                std::fprintf(stderr,
                             "HIP error %d (%s) at %s\n",
                             static_cast<int>(status),
                             hipGetErrorName(status),
                             operation);
                std::exit(1);
            }
        };
        blas_check(hipblasLtCreate(&handle), "hipblasLtCreate");
        blas_check(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                   "hipblasLtMatmulDescCreate");
        const hipblasOperation_t op_a = HIPBLAS_OP_T;
        const hipblasOperation_t op_b = HIPBLAS_OP_N;
        blas_check(hipblasLtMatmulDescSetAttribute(
                       desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)),
                   "hipblasLtMatmulDescSetAttribute(TRANSA)");
        blas_check(hipblasLtMatmulDescSetAttribute(
                       desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)),
                   "hipblasLtMatmulDescSetAttribute(TRANSB)");
        // NT input storage is B[N,K] and A[M,K], both row-major. Viewed as
        // column-major matrices, the product is B^T * A^T = C^T.
        blas_check(hipblasLtMatrixLayoutCreate(
                       &b_layout, HIP_R_16BF, k, n, k),
                   "hipblasLtMatrixLayoutCreate(B)");
        blas_check(hipblasLtMatrixLayoutCreate(
                       &a_layout, HIP_R_16BF, k, sh.lens[0], k),
                   "hipblasLtMatrixLayoutCreate(A)");
        blas_check(hipblasLtMatrixLayoutCreate(
                       &d_layout, HIP_R_16BF, n, sh.lens[0], n),
                   "hipblasLtMatrixLayoutCreate(D)");
        blas_check(hipblasLtMatmulPreferenceCreate(&preference),
                   "hipblasLtMatmulPreferenceCreate");
        blas_check(hipblasLtMatmulPreferenceSetAttribute(
                       preference,
                       HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &workspace_size,
                       sizeof(workspace_size)),
                   "hipblasLtMatmulPreferenceSetAttribute");
        bool selected_tuned = false;
        int tuned_index = -1;
        if(pick_tuned_solution(n, sh.lens[0], k, 'T', 'N', tuned_index))
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> resolved(1);
            std::vector<int> indices{tuned_index};
            if(hipblaslt_ext::getAlgosFromIndex(handle, indices, resolved) ==
                   HIPBLAS_STATUS_SUCCESS &&
               resolved[0].state == HIPBLAS_STATUS_SUCCESS)
            {
                const float probe_alpha = 1.0F;
                const float probe_beta = 0.0F;
                std::size_t required_workspace = 0;
                const hipblasStatus_t supported =
                    hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                         desc,
                                                         &probe_alpha,
                                                         b_layout,
                                                         a_layout,
                                                         &probe_beta,
                                                         d_layout,
                                                         d_layout,
                                                         resolved[0].algo,
                                                         required_workspace);
                if(supported == HIPBLAS_STATUS_SUCCESS &&
                   required_workspace <= workspace_size)
                {
                    algo = resolved[0].algo;
                    selected_tuned = true;
                    std::fprintf(stderr,
                                 "[hipBLASLt] tuned NT shape m=%d n=%d k=%d "
                                 "transA=T transB=N -> solution_index=%d "
                                 "(workspace=%zu B)\n",
                                 n,
                                 sh.lens[0],
                                 k,
                                 tuned_index,
                                 required_workspace);
                }
                else
                {
                    std::fprintf(stderr,
                                 "[hipBLASLt] WARNING: tuned solution %d is unsupported "
                                 "or needs %zu bytes workspace (available %zu); "
                                 "using heuristic\n",
                                 tuned_index,
                                 required_workspace,
                                 workspace_size);
                }
            }
            else
            {
                std::fprintf(stderr,
                             "[hipBLASLt] WARNING: tuned solution %d is unavailable; "
                             "using heuristic\n",
                             tuned_index);
            }
        }
        if(!selected_tuned)
        {
            hipblasLtMatmulHeuristicResult_t heuristic{};
            int algo_count = 0;
            blas_check(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                       desc,
                                                       b_layout,
                                                       a_layout,
                                                       d_layout,
                                                       d_layout,
                                                       preference,
                                                       1,
                                                       &heuristic,
                                                       &algo_count),
                       "hipblasLtMatmulAlgoGetHeuristic");
            if(algo_count == 0)
            {
                std::fprintf(stderr, "hipBLASLt did not return an algorithm for NT shape\n");
                std::exit(1);
            }
            algo = heuristic.algo;
            std::fprintf(stderr, "[hipBLASLt] using heuristic algorithm\n");
        }
        blas_check(hipblasLtMatmulPreferenceDestroy(preference),
                   "hipblasLtMatmulPreferenceDestroy");
        if(hipMalloc(&workspace, workspace_size) != hipSuccess ||
           !blas_out.Alloc(c_ref.size()))
        {
            std::fprintf(stderr, "hipBLASLt workspace/output allocation failed\n");
            std::exit(1);
        }

        const float alpha = 1.0F;
        const float beta = 0.0F;
        auto run_blas = [&](void* output) {
            blas_check(hipblasLtMatmul(handle,
                                       desc,
                                       &alpha,
                                       b_dev.Data(),
                                       b_layout,
                                       a_dev.Data(),
                                       a_layout,
                                       &beta,
                                       output,
                                       d_layout,
                                       output,
                                       d_layout,
                                       &algo,
                                       workspace,
                                       workspace_size,
                                       stream),
                       "hipblasLtMatmul");
        };

        if(opt.hipblaslt_ref)
        {
            run_blas(blas_out.Data());
            if(hipStreamSynchronize(stream) != hipSuccess)
            {
                std::fprintf(stderr, "hipBLASLt reference synchronization failed\n");
                std::exit(1);
            }
            std::vector<Bf16> blas_host(c_ref.size());
            std::vector<float> blas_values(c_ref.size());
            if(!blas_out.Download(blas_host, blas_host.size()))
            {
                std::fprintf(stderr, "hipBLASLt output download failed\n");
                std::exit(1);
            }
            for(std::size_t i = 0; i < blas_values.size(); ++i)
            {
                blas_values[i] = ck_tile::type_convert<float>(blas_host[i]);
            }
            const CompareStats compared = compare(wasp_out,
                                                  blas_values,
                                                  sh,
                                                  opt.rel_tol,
                                                  opt.abs_tol,
                                                  nullptr,
                                                  0);
            res.wasp_vs_hipblaslt_ran = true;
            res.wasp_vs_hipblaslt_max_abs = compared.max_abs;
            res.wasp_vs_hipblaslt_bad = compared.bad;
        }
        if(opt.hipblaslt_perf)
        {
            for(int i = 0; i < opt.warmup; ++i)
            {
                run_blas(blas_out.Data());
            }
            hipEvent_t start = nullptr;
            hipEvent_t stop = nullptr;
            if(hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess)
            {
                std::fprintf(stderr, "hipBLASLt timing event creation failed\n");
                std::exit(1);
            }
            hip_runtime_check(hipEventRecord(start, stream), "hipEventRecord(start)");
            for(int i = 0; i < opt.iters; ++i)
            {
                run_blas(blas_out.Data());
            }
            hip_runtime_check(hipEventRecord(stop, stream), "hipEventRecord(stop)");
            hip_runtime_check(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
            float elapsed = 0.0F;
            hip_runtime_check(hipEventElapsedTime(&elapsed, start, stop),
                              "hipEventElapsedTime");
            res.hipblaslt_ms = static_cast<double>(elapsed) / opt.iters;
            hip_runtime_check(hipEventDestroy(start), "hipEventDestroy(start)");
            hip_runtime_check(hipEventDestroy(stop), "hipEventDestroy(stop)");
        }

        hip_runtime_check(hipFree(workspace), "hipFree(hipBLASLt workspace)");
        hipblasLtMatrixLayoutDestroy(d_layout);
        hipblasLtMatrixLayoutDestroy(a_layout);
        hipblasLtMatrixLayoutDestroy(b_layout);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
#else
        std::fprintf(stderr,
                     "hipBLASLt requested but this target was built without hipBLASLt\n");
        std::exit(2);
#endif
    }

    // -----------------------------------------------------------------------
    // Diagnostics against whichever oracle actually ran. The CPU reference is
    // preferred (it is independent of every GPU path), but a real benchmark shape
    // costs far more MACs than it can afford -- so the CK reference is the
    // fallback, otherwise the histograms below would be silently absent on
    // exactly the shapes that matter.
    // -----------------------------------------------------------------------
    const std::vector<float>* oracle = nullptr;
    if(res.cpu_ran)
    {
        oracle = &c_ref;
    }
    else if(res.wasp_vs_ref_ran)
    {
        oracle = &ref_out;
    }
    if(oracle != nullptr)
    {
        const std::size_t len = oracle->size();
        for(std::size_t i = 0; i < len; ++i)
        {
            const double r = (*oracle)[i];
            const double g = wasp_out[i];
            if(std::fabs(g - static_cast<double>(kSentinel)) < 1.0)
            {
                ++res.wasp_not_written;
            }
            if(std::fabs(r - g) > static_cast<double>(opt.rel_tol) * std::fabs(r) +
                                      static_cast<double>(opt.abs_tol))
            {
                const int  col    = static_cast<int>(i % static_cast<std::size_t>(n));
                const long grow   = static_cast<long>(i / static_cast<std::size_t>(n));
                const int  row    = sh.row_in_group(grow);   // ragged: not a modulo
                const int  stripe = col / 32;
                if(stripe >= 0 && stripe < 8)
                {
                    ++res.bad_per_stripe[stripe];
                }
                ++res.bad_per_row16[row % 16];
                // Bucket per m-tile within the group, so the array stays in range
                // for any M.
                ++res.bad_per_miter[(row % 256) / 16];
                // On a real shape there can be millions of mismatches; keeping
                // eight coordinates is enough to see the pattern.
                if(res.wasp_examples.size() < 8)
                {
                    res.wasp_examples.push_back(
                        Mismatch{sh.group_of(grow), row, col, (*oracle)[i], wasp_out[i]});
                }
            }
        }
        // First element of the second half of the whole output. With a ragged
        // group list this straddles groups, so it is a coarse "did the second
        // half get written at all" probe, nothing finer.
        const std::size_t half_idx =
            static_cast<std::size_t>(sum_m) * static_cast<std::size_t>(n) / 2;
        if(len > half_idx)
        {
            res.probe_cell = wasp_out[half_idx];
        }
        if(opt.dump_row >= 0 && static_cast<long>(opt.dump_row) < sum_m)
        {
            const std::size_t base = static_cast<std::size_t>(opt.dump_row) * n;
            for(int j = 0; j < n; j += 8)
            {
                res.dump_ref.push_back((*oracle)[base + j]);
                res.dump_got.push_back(wasp_out[base + j]);
            }
        }
    }

    // Verdict: prefer the CPU oracle, fall back to the CK reference. If neither
    // ran there is nothing to assert, and claiming a pass would be a lie -- main
    // reports that state explicitly instead.
    if(res.cpu_ran)
    {
        res.passed = (res.wasp_bad == 0);
    }
    else if(res.wasp_vs_ref_ran)
    {
        res.passed = (res.wasp_vs_ref_bad == 0);
    }
    else
    {
        res.passed = true;
    }
    if(res.wasp_vs_hipblaslt_ran)
    {
        res.passed = res.passed && res.wasp_vs_hipblaslt_bad == 0;
    }
    return res;
}

} // namespace

int main(int argc, char* argv[])
{
    Options opt{};
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--help")
        {
            print_usage();
            return 0;
        }
        else if(arg == "--allow-unaligned")
        {
            opt.allow_unaligned = true;
        }
        else if(arg == "--quiet")
        {
            opt.quiet = true;
        }
        else if(arg == "--debug-ctas")
        {
            opt.debug_ctas = true;
        }
        else if(arg.rfind("--m=", 0) == 0)
        {
            opt.m = parse_int(arg.c_str() + 4, opt.m);
        }
        else if(arg.rfind("--n=", 0) == 0)
        {
            opt.n = parse_int(arg.c_str() + 4, opt.n);
        }
        else if(arg.rfind("--k=", 0) == 0)
        {
            opt.k = parse_int(arg.c_str() + 4, opt.k);
        }
        else if(arg.rfind("--groups=", 0) == 0)
        {
            opt.groups = parse_int(arg.c_str() + 9, opt.groups);
        }
        else if(arg.rfind("--warmup=", 0) == 0)
        {
            opt.warmup = parse_int(arg.c_str() + 9, opt.warmup);
        }
        else if(arg.rfind("--iters=", 0) == 0)
        {
            opt.iters = parse_int(arg.c_str() + 8, opt.iters);
        }
        else if(arg.rfind("--ref=", 0) == 0)
        {
            opt.ref = parse_int(arg.c_str() + 6, 1) != 0;
        }
        else if(arg.rfind("--hipblaslt-ref=", 0) == 0)
        {
            opt.hipblaslt_ref = parse_int(arg.c_str() + 16, 0) != 0;
        }
        else if(arg.rfind("--hipblaslt-perf=", 0) == 0)
        {
            opt.hipblaslt_perf = parse_int(arg.c_str() + 17, 0) != 0;
        }
        else if(arg.rfind("--rel-tol=", 0) == 0)
        {
            opt.rel_tol = parse_float(arg.c_str() + 10, opt.rel_tol);
        }
        else if(arg.rfind("--abs-tol=", 0) == 0)
        {
            opt.abs_tol = parse_float(arg.c_str() + 10, opt.abs_tol);
        }
        else if(arg.rfind("--pattern=", 0) == 0)
        {
            opt.pattern = arg.substr(10);
        }
        else if(arg.rfind("--dump-row=", 0) == 0)
        {
            opt.dump_row = parse_int(arg.c_str() + 11, opt.dump_row);
        }
        else if(arg.rfind("--ablate=", 0) == 0)
        {
            opt.ablate = parse_int(arg.c_str() + 9, opt.ablate);
        }
        // ---- reference-harness vocabulary ----
        else if(arg.rfind("--grouped-length=", 0) == 0)
        {
            if(!parse_group_lens(arg.c_str() + 17, opt.group_lens))
            {
                std::printf("bad --grouped-length: %s\n", arg.c_str());
                return 2;
            }
        }
        else if(arg.rfind("--layout=", 0) == 0)
        {
            opt.layout = arg.substr(9);
        }
        else if(arg.rfind("--output-dtype=", 0) == 0)
        {
            opt.out_dtype = arg.substr(15);
        }
        else if(arg.rfind("--bench=", 0) == 0)
        {
            opt.bench = parse_int(arg.c_str() + 8, 1) != 0;
        }
        else if(arg == "--bench")
        {
            opt.bench = true;
        }
        else if(arg.rfind("--persistent-kernel=", 0) == 0)
        {
            opt.persistent_kernel = parse_int(arg.c_str() + 20, 1) != 0;
        }
        else if(arg.rfind("--warmup-iters=", 0) == 0)
        {
            opt.warmup = parse_int(arg.c_str() + 15, opt.warmup);
        }
        else if(arg.rfind("--bench-iters=", 0) == 0)
        {
            opt.iters = parse_int(arg.c_str() + 14, opt.iters);
        }
        else
        {
            std::printf("unknown argument: %s\n", arg.c_str());
            print_usage();
            return 2;
        }
    }
    if(opt.iters < 1)
    {
        opt.iters = 1;
    }

    // ---- Options this port cannot honour yet are refused LOUDLY. Accepting them
    //      and quietly doing something else is how a benchmark stops meaning
    //      anything. ----
    if(opt.layout != "NT")
    {
        std::printf("wasp test: layout=%s is not ported (NT only so far)\n",
                    opt.layout.c_str());
        return 2;
    }
    if(opt.out_dtype != "bf16")
    {
        std::printf("wasp test: output-dtype=%s is not ported (bf16 only so far)\n",
                    opt.out_dtype.c_str());
        return 2;
    }
    if((opt.hipblaslt_ref || opt.hipblaslt_perf) &&
       (opt.groups != 1 || !opt.group_lens.empty()))
    {
        std::printf("wasp test: hipBLASLt comparison/timing currently requires one group\n");
        return 2;
    }
    if(!opt.persistent_kernel)
    {
        // The CK grouped kernel walks a grid-stride tile list inside the device
        // kernel; there is no per-group launch path to select.
        std::printf("wasp test: --persistent-kernel=0 is not representable here (the CK "
                    "grouped kernel is always persistent); ignoring\n");
    }
    const ShapeInfo sh = make_shape(opt);

    hipDeviceProp_t prop{};
    int             dev = 0;
    if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess)
    {
        std::printf("wasp test: no HIP device -> SKIP\n");
        return 0;
    }
    const int         num_cu = prop.multiProcessorCount;
    hipStream_t       stream = nullptr;

    std::printf("wasp test: device CU=%d, tile=%dx%dx%d, block=%d threads, LDS=%d B\n",
                num_cu,
                static_cast<int>(WaspConfig::M_Tile),
                static_cast<int>(WaspConfig::N_Tile),
                static_cast<int>(WaspConfig::K_Tile),
                static_cast<int>(WaspPipeline::BlockSize),
                static_cast<int>(WaspPipeline::kWaspLdsBytes));
    // The whole design assumes 12 waves of 64 lanes. If the framework ever
    // launches this kernel with wave32, BlockSize() halves and the role split
    // silently collapses, so print what the framework actually resolved.
    std::printf("wasp test: is_wave32=%d launched_block_x=%d (pipeline BlockSize=%d)\n",
                static_cast<int>(ck_tile::is_wave32()),
                static_cast<int>(WaspKernel::BlockSize().x),
                static_cast<int>(WaspPipeline::BlockSize));
    // The persistent tile loop derives its total work from the PARTITIONER's
    // GridSize, not from the host. Print it, and the reference config's, so a
    // "tile never processed" symptom can be attributed to the partitioner or
    // ruled out immediately.
    {
        using RefPartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            RefShape,
            RefConfig::TileParitionerGroupNum,
            RefConfig::TileParitionerM01>;
        // Per group: the grid is derived from each group's own M, so this is
        // printed for group 0 only as a sanity check.
        std::printf("wasp test: partitioner GridSize(m=%d n=%d) group0: wasp=%d "
                    "(MPerBlock=%d NPerBlock=%d M01=%d GroupNum=%d) ref=%d (M01=%d)\n",
                    sh.lens[0],
                    sh.n,
                    static_cast<int>(WaspPartitioner::GridSize(sh.lens[0], sh.n)),
                    static_cast<int>(WaspPartitioner::MPerBlock),
                    static_cast<int>(WaspPartitioner::NPerBlock),
                    static_cast<int>(WaspConfig::TileParitionerM01),
                    static_cast<int>(WaspConfig::TileParitionerGroupNum),
                    static_cast<int>(RefPartitioner::GridSize(sh.lens[0], sh.n)),
                    static_cast<int>(RefConfig::TileParitionerM01));
    }
    std::printf("wasp test: layout=%s groups=%d sum(M)=%ld M=[",
                opt.layout.c_str(),
                sh.groups(),
                sh.sum_m);
    for(int g = 0; g < sh.groups(); ++g)
    {
        std::printf("%s%d", g ? "," : "", sh.lens[static_cast<std::size_t>(g)]);
    }
    std::printf("] N=%d K=%d persistent=1 (warmup=%d iters=%d ck_ref=%s)\n",
                sh.n,
                sh.k,
                opt.warmup,
                opt.iters,
                opt.ref ? "on" : "off");

#if defined(CK_TILE_GROUPED_GEMM_WASP_ABLATION)
    if(opt.ablate != 0)
    {
        const int mask = opt.ablate;
        if(hipMemcpyToSymbol(ck_tile::wasp_detail::g_wasp_ablate,
                             &mask,
                             sizeof(mask),
                             0,
                             hipMemcpyHostToDevice) != hipSuccess)
        {
            std::printf("wasp test: ERROR: could not set the ablation mask\n");
            return 2;
        }
        std::printf("wasp test: ABLATION mask=%d (1=no B loads, 2=no MMACs, "
                    "4=A read row0, 8=no barriers) -- results are garbage by design\n",
                    mask);
    }
#endif

    const CaseResult r = run_case(opt, sh, num_cu, stream);

    if(r.skipped)
    {
        std::printf("wasp test: SKIPPED (shape outside the ported alignment contract)\n");
        return 0;
    }

#if defined(CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT)
    {
        std::vector<uint64_t> cycles(1024 * 16 * 9);
        if(hipMemcpyFromSymbol(cycles.data(),
                               g_wasp_epi_cycles,
                               cycles.size() * sizeof(uint64_t),
                               0,
                               hipMemcpyDeviceToHost) != hipSuccess)
        {
            std::fprintf(stderr, "failed to read Wasp epilogue cycle counters\n");
            return 2;
        }
        constexpr const char* stage_names[] = {
            "bf16-convert",
            "half0-lds-issue",
            "half0-lds-drain",
            "half0-lds-read+global-issue",
            "inter-half-lds-drain",
            "half1-lds-issue",
            "half1-lds-drain",
            "half1-lds-read+global-issue",
            "global-store-drain"};
        for(std::size_t stage = 0; stage < 9; ++stage)
        {
            uint64_t min_cycles = UINT64_MAX;
            uint64_t max_cycles = 0;
            uint64_t sum_cycles = 0;
            std::size_t sample_count = 0;
            for(std::size_t i = stage; i < cycles.size(); i += 9)
            {
                const uint64_t value = cycles[i];
                if(value != 0)
                {
                    min_cycles = value < min_cycles ? value : min_cycles;
                    max_cycles = value > max_cycles ? value : max_cycles;
                    sum_cycles += value;
                    ++sample_count;
                }
            }
            if(sample_count != 0)
            {
                std::printf("wasp epilogue %s cycles: min=%llu avg=%.1f max=%llu samples=%zu\n",
                            stage_names[stage],
                            static_cast<unsigned long long>(min_cycles),
                            static_cast<double>(sum_cycles) / static_cast<double>(sample_count),
                            static_cast<unsigned long long>(max_cycles),
                            sample_count);
            }
        }
    }
#endif

#if defined(CK_TILE_GROUPED_GEMM_WASP_VMEM_WAIT_COUNT)
    {
        std::vector<ck_tile::wasp_detail::WaspVmemSample> samples(1024 * 16);
        if(hipMemcpyFromSymbol(samples.data(),
                               ck_tile::wasp_detail::g_wasp_vmem_samples,
                               samples.size() * sizeof(samples[0]),
                               0,
                               hipMemcpyDeviceToHost) != hipSuccess)
        {
            std::fprintf(stderr, "failed to read Wasp VMEM wait samples\n");
            return 2;
        }
        std::printf("wasp VMEM wait samples (stage %u; cycles, aggregated across CTAs):\n",
                    static_cast<unsigned>(ck_tile::wasp_detail::kVmemSampleStage));
        for(int wave = 0; wave < 12; ++wave)
        {
            uint64_t issue_min = UINT64_MAX;
            uint64_t issue_max = 0;
            uint64_t issue_sum = 0;
            uint64_t wait_min  = UINT64_MAX;
            uint64_t wait_max  = 0;
            uint64_t wait_sum  = 0;
            uint64_t vmcnt4_min = UINT64_MAX;
            uint64_t vmcnt4_max = 0;
            uint64_t vmcnt4_sum = 0;
            uint64_t vmcnt2_min = UINT64_MAX;
            uint64_t vmcnt2_max = 0;
            uint64_t vmcnt2_sum = 0;
            std::size_t count  = 0;
            for(int cta = 0; cta < 1024; ++cta)
            {
                const auto& sample = samples[cta * 16 + wave];
                if(sample.valid == 0)
                {
                    continue;
                }
                const uint64_t issue_cycles = sample.issue_to_wait_start;
                const uint64_t wait_cycles  = sample.wait_cycles;
                issue_min = std::min(issue_min, issue_cycles);
                issue_max = std::max(issue_max, issue_cycles);
                issue_sum += issue_cycles;
                wait_min = std::min(wait_min, wait_cycles);
                wait_max = std::max(wait_max, wait_cycles);
                wait_sum += wait_cycles;
                vmcnt4_min = std::min(vmcnt4_min, sample.vmcnt4_cycles);
                vmcnt4_max = std::max(vmcnt4_max, sample.vmcnt4_cycles);
                vmcnt4_sum += sample.vmcnt4_cycles;
                vmcnt2_min = std::min(vmcnt2_min, sample.vmcnt2_cycles);
                vmcnt2_max = std::max(vmcnt2_max, sample.vmcnt2_cycles);
                vmcnt2_sum += sample.vmcnt2_cycles;
                ++count;
            }
            if(count != 0)
            {
                if(wave < 4)
                {
                    std::printf(
                        "  producer wave %d: issue-to-wait-start min/avg/max=%llu/%.1f/%llu, "
                        "vmcnt<=4 elapsed min/avg/max=%llu/%.1f/%llu, "
                        "vmcnt<=2 elapsed min/avg/max=%llu/%.1f/%llu, "
                        "vmcnt(0)-wait min/avg/max=%llu/%.1f/%llu, CTAs=%zu\n",
                        wave,
                        static_cast<unsigned long long>(issue_min),
                        static_cast<double>(issue_sum) / static_cast<double>(count),
                        static_cast<unsigned long long>(issue_max),
                        static_cast<unsigned long long>(vmcnt4_min),
                        static_cast<double>(vmcnt4_sum) / static_cast<double>(count),
                        static_cast<unsigned long long>(vmcnt4_max),
                        static_cast<unsigned long long>(vmcnt2_min),
                        static_cast<double>(vmcnt2_sum) / static_cast<double>(count),
                        static_cast<unsigned long long>(vmcnt2_max),
                        static_cast<unsigned long long>(wait_min),
                        static_cast<double>(wait_sum) / static_cast<double>(count),
                        static_cast<unsigned long long>(wait_max),
                        count);
                }
                else
                {
                    std::printf("  consumer wave %d: issue-to-wait-start min/avg/max=%llu/%.1f/%llu, "
                                "vmcnt-wait min/avg/max=%llu/%.1f/%llu, CTAs=%zu\n",
                                wave - 4,
                                static_cast<unsigned long long>(issue_min),
                                static_cast<double>(issue_sum) / static_cast<double>(count),
                                static_cast<unsigned long long>(issue_max),
                                static_cast<unsigned long long>(wait_min),
                                static_cast<double>(wait_sum) / static_cast<double>(count),
                                static_cast<unsigned long long>(wait_max),
                                count);
                }
            }
        }

        const int n_tiles =
            (sh.n + static_cast<int>(WaspPartitioner::NPerBlock) - 1) /
            static_cast<int>(WaspPartitioner::NPerBlock);
        const int m_tiles =
            (sh.lens[0] + static_cast<int>(WaspPartitioner::MPerBlock) - 1) /
            static_cast<int>(WaspPartitioner::MPerBlock);
#if defined(CK_TILE_GROUPED_GEMM_WASP_REVERSE_CTA_ORDER)
        const int tile_count = m_tiles * n_tiles;
#endif
        std::printf("producer A wait by M tile (N-tile-fastest tile order%s):\n",
#if defined(CK_TILE_GROUPED_GEMM_WASP_REVERSE_CTA_ORDER)
                    ", reversed"
#else
                    ""
#endif
        );
        for(int m_tile = 0; m_tile < m_tiles; ++m_tile)
        {
            std::vector<uint64_t> waits;
            uint64_t              vmcnt4_sum = 0;
            uint64_t              vmcnt2_sum = 0;
            uint64_t              wait_sum   = 0;
            for(int n_tile = 0; n_tile < n_tiles; ++n_tile)
            {
                const int tile_id = m_tile * n_tiles + n_tile;
#if defined(CK_TILE_GROUPED_GEMM_WASP_REVERSE_CTA_ORDER)
                const int cta = tile_count - 1 - tile_id;
#else
                const int cta = tile_id;
#endif
                for(int wave = 0; wave < 4; ++wave)
                {
                    const auto& sample = samples[cta * 16 + wave];
                    if(sample.valid != 0)
                    {
                        waits.push_back(sample.wait_cycles);
                        wait_sum += sample.wait_cycles;
                        vmcnt4_sum += sample.vmcnt4_cycles;
                        vmcnt2_sum += sample.vmcnt2_cycles;
                    }
                }
            }
            if(waits.empty())
            {
                continue;
            }
            std::sort(waits.begin(), waits.end());
            const std::size_t p95_index = (waits.size() - 1) * 95 / 100;
            const int row_start = m_tile * static_cast<int>(WaspPartitioner::MPerBlock);
            const int row_end = std::min(
                sh.lens[0], row_start + static_cast<int>(WaspPartitioner::MPerBlock));
            const double sample_count = static_cast<double>(waits.size());
            std::printf("  M tile %d rows [%d,%d): vmcnt(0) avg/p95/max=%0.1f/%llu/%llu, "
                        "vmcnt<=4/<=2 avg=%0.1f/%0.1f, samples=%zu\n",
                        m_tile,
                        row_start,
                        row_end,
                        static_cast<double>(wait_sum) / sample_count,
                        static_cast<unsigned long long>(waits[p95_index]),
                        static_cast<unsigned long long>(waits.back()),
                        static_cast<double>(vmcnt4_sum) / sample_count,
                        static_cast<double>(vmcnt2_sum) / sample_count,
                        waits.size());
        }
    }
#endif

    if(!opt.quiet)
    {
        std::printf("wasp test: %.3f ms  %.2f TFLOPS\n", r.ms, r.tflops);
        if(r.hipblaslt_ms > 0.0)
        {
            const double macs = static_cast<double>(sh.sum_m) * sh.n * sh.k;
            const double blas_tflops = 2.0 * macs / (r.hipblaslt_ms * 1.0e-3) / 1.0e12;
            std::printf("hipblaslt: %.3f ms  %.2f TFLOPS\n", r.hipblaslt_ms, blas_tflops);
        }
    }
    if(opt.bench)
    {
        // Same semantics as the reference harness: --bench is timing only.
        return 0;
    }
    if(r.cpu_ran)
    {
        std::printf("wasp test: wasp   vs CPU  max_abs=%.6g max_rel=%.3g bad=%zu\n",
                    r.wasp_max_abs,
                    r.wasp_max_rel,
                    r.wasp_bad);
    }
    else
    {
        std::printf("wasp test: wasp   vs CPU  (skipped: shape too large for the naive "
                    "oracle; the ck_ref comparison below is the oracle instead)\n");
    }
    if(r.ref_ran)
    {
        if(r.cpu_ran)
        {
            std::printf("wasp test: ck_ref vs CPU  max_abs=%.6g max_rel=%.3g bad=%zu\n",
                        r.ref_max_abs,
                        r.ref_max_rel,
                        r.ref_bad);
        }
        else
        {
            // Printing the zero-initialised fields here would read as a pass; the
            // comparison simply never happened.
            std::printf("wasp test: ck_ref vs CPU  (not compared: the CPU oracle was "
                        "skipped)\n");
        }
    }
    if(r.wasp_vs_ref_ran)
    {
        std::printf("wasp test: wasp   vs ck_ref  max_abs=%.6g bad=%zu\n",
                    r.wasp_vs_ref_max_abs,
                    r.wasp_vs_ref_bad);
    }
    if(r.wasp_vs_hipblaslt_ran)
    {
        std::printf("wasp test: wasp   vs hipBLASLt max_abs=%.6g bad=%zu\n",
                    r.wasp_vs_hipblaslt_max_abs,
                    r.wasp_vs_hipblaslt_bad);
    }
    // The histograms are oracle independent: print them whenever they were
    // filled, not only when the CPU oracle happened to run -- otherwise they go
    // missing on exactly the real shapes this is meant to debug.
    if(!r.wasp_examples.empty())
    {
        std::printf("wasp test: first failing outputs (g, m, n): ref -> got\n");
        for(const auto& e : r.wasp_examples)
        {
            std::printf("wasp test:   (%d, %d, %d): %.6g -> %.6g\n",
                        e.g,
                        e.m,
                        e.n,
                        static_cast<double>(e.ref),
                        static_cast<double>(e.got));
        }
        std::printf("wasp test: bad per 32-col stripe:");
        for(int s = 0; s < 8; ++s)
        {
            std::printf(" %zu", r.bad_per_stripe[s]);
        }
        std::printf("\nwasp test: bad per (row%%16):");
        for(int m = 0; m < 16; ++m)
        {
            std::printf(" %zu", r.bad_per_row16[m]);
        }
        std::printf("\nwasp test: bad per (row/16):");
        for(int m = 0; m < 16; ++m)
        {
            std::printf(" %zu", r.bad_per_miter[m]);
        }
        std::printf("\nwasp test: outputs never written by the kernel (sentinel left): %zu\n",
                    r.wasp_not_written);
    }
    if(!r.cpu_ran && !r.wasp_vs_ref_ran)
    {
        std::printf("wasp test: no oracle ran (--ref=0 and the shape is too large for the "
                    "CPU reference); nothing was verified\n");
        return 1;
    }

#if defined(CK_TILE_GROUPED_GEMM_WASP_DEBUG_RECORD)
    if(opt.debug_ctas)
    {
        std::vector<int> dbg(128 * 10);
        std::vector<int> waves(128 * 16);
        const bool ok_a =
            hipMemcpyFromSymbol(dbg.data(), g_wasp_dbg, dbg.size() * sizeof(int), 0,
                                hipMemcpyDeviceToHost) == hipSuccess;
        const bool ok_b =
            hipMemcpyFromSymbol(waves.data(), g_wasp_dbg_waves, waves.size() * sizeof(int), 0,
                                hipMemcpyDeviceToHost) == hipSuccess;
        if(ok_a && ok_b)
        {
            int active = 0;
            for(int c = 0; c < 128; ++c)
            {
                if(dbg[static_cast<std::size_t>(c) * 10] > 0)
                {
                    ++active;
                }
            }
            std::printf("wasp test: CTAs that reached the epilogue: %d\n", active);
            std::printf("wasp test: probe cell (first element of the second half) = %g "
                        "(777 = the store reached it, 1024 = still the sentinel)\n",
                        static_cast<double>(r.probe_cell));
            for(int c = 0; c < 4; ++c)
            {
                const std::size_t b = static_cast<std::size_t>(c) * 10;
                if(dbg[b] == 0)
                {
                    continue;
                }
                unsigned mask = 0;
                for(int w = 0; w < 16; ++w)
                {
                    if(waves[static_cast<std::size_t>(c) * 16 + static_cast<std::size_t>(w)] != 0)
                    {
                        mask |= (1U << static_cast<unsigned>(w));
                    }
                }
                std::printf(
                    "wasp test:   cta %d: waves=0x%03x block_org=(%d,%d) M=%d N=%d ldd=%d "
                    "calc_off=%d d_ptr-p_data_=%d elems\n",
                    c,
                    mask,
                    dbg[b + 1],
                    dbg[b + 8],
                    dbg[b + 2],
                    dbg[b + 3],
                    dbg[b + 4],
                    dbg[b + 5],
                    dbg[b + 7] - dbg[b + 6]);
            }
        }
    }
#endif
    if(opt.dump_row >= 0 && !r.dump_ref.empty())
    {
        std::printf("wasp test: row %d dump, every 8th column\n", opt.dump_row);
        for(std::size_t i = 0; i < r.dump_ref.size(); ++i)
        {
            std::printf("wasp test:   n=%-4zu ref=%-14.6g got=%-14.6g\n",
                        i * 8,
                        static_cast<double>(r.dump_ref[i]),
                        static_cast<double>(r.dump_got[i]));
        }
    }

#if defined(CK_TILE_GROUPED_GEMM_WASP_DUMP_A_STAGE)
    // A staging verification. The stage is expected to hold the 256x64 A tile in
    // the documented layout; only the VALUES are checked here (with --pattern=ones
    // every entry must be bf16 1.0 == 0x3F80), which is exactly the property the
    // consumer depends on. A wrong layout would leave far more than the ~5 % the
    // runtime shows.
    {
        std::vector<unsigned short> stage(2 * 256 * 64);
        if(hipMemcpyFromSymbol(stage.data(),
                               ck_tile::wasp_detail::g_wasp_a_dump,
                               stage.size() * sizeof(unsigned short),
                               0,
                               hipMemcpyDeviceToHost) == hipSuccess)
        {
            for(int s = 0; s < 2; ++s)
            {
                std::size_t good   = 0;
                std::size_t per_m[16] = {};
                for(int rr = 0; rr < 256; ++rr)
                {
                    for(int cc = 0; cc < 64; ++cc)
                    {
                        if(stage[static_cast<std::size_t>(s) * 256 * 64 +
                                 static_cast<std::size_t>(rr) * 64 + cc] == 0x3F80U)
                        {
                            ++good;
                        }
                        else
                        {
                            ++per_m[rr / 16];
                        }
                    }
                }
                std::printf("wasp test: A stage snapshot slot %d: %zu / 16384 entries are bf16 1.0\n",
                            s,
                            good);
                std::printf("wasp test:   not-1.0 per 16-row m-iteration:");
                for(int m = 0; m < 16; ++m)
                {
                    std::printf(" %zu", per_m[m]);
                }
                std::printf("\n");
            }
        }
    }
#endif

    if(!r.passed)
    {
        std::printf("wasp test: FAIL\n");
        return 1;
    }
    std::printf("wasp test: PASS\n");
    return 0;
}
