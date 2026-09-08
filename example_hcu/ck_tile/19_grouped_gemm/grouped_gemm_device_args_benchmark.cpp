// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

/*
 * grouped GEMM device-args 参数化示例使用说明
 * ============================================
 *
 * 一、用途与边界
 * --------------
 * 本程序用于在同一个进程、同一块 HCU、同一批输入数据上比较：
 *
 *   1. CK versioned device-args C ABI 选择并运行的预编译 grouped GEMM；
 *   2. 传统 hipBLAS 或 hipBLASLt 逐组 GEMM 基线。
 *
 * CK 路径遵循 Problem -> pure selector -> stable InstanceId -> candidate registry ->
 * 预编译 kernel 的生产调用流程。运行时参数只选择已经编译进 ELF 的 C++ template
 * 实例，不会在运行时实例化或编译新的 C++ template。实际 HCU 架构由 CK 内部检测。
 *
 * 不指定 --group-lengths 时，所有 group 使用相同 M/N/K。指定后，NT/NN 把该列表
 * 解释为每组逻辑 M，TN 把它解释为每组逻辑 K；--variable-capacity 单独描述一块连续
 * backing allocation 的提交容量。这样可以测试 ragged、empty、compact/padded 和
 * logical-K tail，同时 CK 接口中不出现任何上层模型或训练阶段名称。
 *
 * 二、构建方法
 * ------------
 * 本 target 支持 gfx936、gfx938 与 gfx946。在对应架构的机器上配置独立 build
 * tree；下面以 gfx936 为例，其他架构只需修改 GPU_TARGETS：
 *
 *   resource_dir=$(/opt/dtk/bin/aicc -print-resource-dir)
 *   cmake -S . -B build_device_args_example_bw \
 *       -DCMAKE_PREFIX_PATH=/opt/dtk \
 *       -DCMAKE_CXX_COMPILER=/opt/dtk/bin/aicc \
 *       -DCMAKE_CXX_FLAGS="-std=c++17 -O3 -ftemplate-backtrace-limit=0 -fPIE -Wno-gnu-line-marker" \
 *       -DCMAKE_BUILD_TYPE=Release \
 *       -DBUILD_DEV=ON \
 *       -DBUILD_EXAMPLE=ON \
 *       -DUSE_BITINT_EXTENSION=ON \
 *       -DGPU_TARGETS=gfx936 \
 *       -DHIP_CLANG_INCLUDE_PATH="${resource_dir}/include" \
 *       -DCK_GROUPED_GEMM_HIPBLASLT_ROOT=/wksp/hipblaslt-install
 *   cmake --build build_device_args_example_bw \
 *       --target tile_example_grouped_gemm_device_args_benchmark -j 16
 *
 * 必须在首次配置这个 build tree 时选择 aicc。若 CMakeCache.txt 已把
 * CMAKE_CXX_COMPILER 缓存为 /usr/bin/c++，请保留旧目录作为失败证据并新建 build
 * tree，不要在同一个 cache 上继续构建；否则 HIP kernel 源文件无法正确编译。
 * BUILD_EXAMPLE 必须为 ON；CK 顶层默认值是 OFF。若未打开，grouped GEMM example
 * 的 CMakeLists.txt 不会被处理，表现为本 target 不存在且 hipBLASLt 变量未使用。
 *
 * CK_GROUPED_GEMM_HIPBLASLT_ROOT 只决定本 example 编译时使用哪套 hipBLASLt
 * 头文件。它不会把该安装路径写入 CK selector、registry 或 grouped GEMM 公共 ABI。
 *
 * 三、基本运行方法
 * --------------
 *
 *   ./build_device_args_example_bw/bin/\
 *       tile_example_grouped_gemm_device_args_benchmark \
 *       --m=2048 --n=2048 --k=2048 --groups=4 \
 *       --dtype=bf16 --layout=TN --backend=both \
 *       --blas-root=/wksp/hipblaslt-install
 *
 * ragged variable-M（NT/NN）示例：
 *
 *   --m=2120 --n=4096 --k=7168 --layout=NT \
 *   --group-lengths=2102,2120,0,2092 --variable-capacity=16384 \
 *   --allow-empty-groups=1 --blas-streams=4
 *
 * ragged variable-K（TN）示例：
 *
 *   --m=4096 --n=7168 --k=2120 --layout=TN \
 *   --group-lengths=2102,2120,2108,2092 --variable-capacity=16384
 *
 * 也可以直接指定一个运行时 ELF：
 *
 *   --blas-lib=/wksp/hipblaslt-install/lib/libhipblaslt.so
 *
 * 或沿用 Primus 风格的环境变量：
 *
 *   export HIPBLASLT_ROOT=/wksp/hipblaslt-install
 *   # 也支持 PRIMUS_TURBO_HIPBLASLT_HOME
 *
 * BLAS 选择优先级为：
 *
 *   --blas-lib
 *   -> --blas-root
 *   -> CK_GROUPED_GEMM_BLAS_LIBRARY
 *   -> PRIMUS_TURBO_HIPBLASLT_HOME
 *   -> HIPBLASLT_ROOT
 *   -> 系统 libhipblas.so
 *
 * 当指定 libhipblaslt.so 的绝对路径时，动态加载器不会自动搜索它的同级目录。
 * 本程序会先从同一 lib 目录以 RTLD_GLOBAL 加载 liborigami.so.1，再加载
 * libhipblaslt.so，因此通常不需要额外修改 LD_LIBRARY_PATH。
 *
 * 四、参数含义
 * ------------
 *
 *   --m/--n/--k       每组 GEMM 的逻辑 M、N、K。
 *   --groups          group 数量；指定 group-lengths 时可省略，否则必须与列表长度一致。
 *   --group-lengths   逗号分隔的 per-group logical length；NT/NN 对应 M，TN 对应 K。
 *   --variable-capacity
 *                     变量维 backing capacity，必须不小于 lengths 合计；省略时取合计。
 *                     它是通用容量，不代表任一 group 的逻辑长度。
 *   --allow-empty-groups
 *                     允许列表中包含 0，并向 CK 声明 empty-group contract。
 *   --dtype           fp16 或 bf16。
 *   --layout          NT、NN 或 TN；C 始终是 RowMajor：
 *                       NT = A RowMajor，B ColumnMajor；
 *                       NN = A RowMajor，B RowMajor；
 *                       TN = A ColumnMajor，B RowMajor。
 *   --backend         ck、blas 或 both。
 *   --blas-root       hipBLASLt 安装根目录，预期包含 include/ 与 lib/。
 *   --blas-lib        精确的 libhipblaslt.so 或 libhipblas.so 路径。
 *   --time            1 时执行 warmup/GPU event 采样；0 时只做 correctness。
 *   --warmup          正式采样前的预热次数，性能对比建议固定为 10。
 *   --samples         GPU event 样本数，性能对比建议固定为 30。
 *   --correctness-repeats
 *                     correctness 重复次数，性能对比建议固定为 3。
 *   --validate        1 时先验证精度，0 时跳过；正式结果不得关闭。
 *   --num-cu          传给 CK launch 的 CU 限制，0 表示使用 CK 默认值。
 *   --include-args-build
 *                     1 时 CK 每个计时样本包含一次 GPU device-args 构造
 *                     kernel，更接近 Primus 调用；0 时只测最终 CK grouped
 *                     GEMM launch。比较前必须固定该值并在结果中记录。
 *   --blas-streams    1 为单流逐组；4 为四流 round-robin。后者用于贴近采用四流
 *                     grouped BLAS 的 consumer，但仍不能替代 consumer 自身验收。
 *
 * 五、精度与性能输出
 * ------------------
 * 程序坚持 correctness first：CK 和 BLAS 分别与 CPU 抽样 reference 比较；
 * backend=both 时还会对 CK/BLAS 的全部输出元素逐一比较。任一 failed 非零，
 * 程序以失败结束，不应继续采用对应性能结果。
 * `--time=0` 强制 warmup/samples 为 0，在 correctness 后直接返回；适合指令
 * 模拟器和常规 correctness smoke，不创建 GPU event timer。
 *
 * 性能使用 HIP GPU event，输出 median/mean/min/max/CV/TFLOPS。both 模式按
 * 样本奇偶交替 CK 与 BLAS 的先后顺序，以减弱温度和频率漂移。正式比较建议使用
 * 10 warmup、30 samples、3 次 correctness，并要求 CK 与 BLAS 的 CV 都不超过
 * 10%。`RATIO ck_over_blas` 是 CK median / BLAS median，小于 1 表示 CK 更快。
 *
 * 六、单流逐组与四流轮转
 * ----------------------
 * `--blas-streams=1` 的 hipBLAS/hipBLASLt 基线是 sequential-per-group：
 * 所有 group 都提交到调用方传入的同一 HIP stream。同一 stream 严格有序，因而
 * G0、G1、...、G(G-1) 不发生组间并发。这种模式实现简单，适合观察单个 BLAS
 * GEMM 的算法、数值和稳定性，也不会额外创建 stream/event。
 *
 * `--blas-streams=4` 的 hipBLASLt 路径采用 four-stream round-robin：它创建 4 个
 * hipBLASLt handle、4 条 non-blocking stream、4 份 workspace 和配套 event。
 * 第 i 组发送到 stream[i % 4]。四条流先等待调用方 stream 的入口 event；全部
 * group 提交后，各计算流记录完成 event，调用方 stream 再等待这些 event。因此
 * 最多四组可以并发，G>4 时每条流内部仍保持顺序。
 *
 * 两种模式的含义可以概括为：
 *
 *   单流：       caller stream: G0 -> G1 -> G2 -> G3
 *   四流：       stream 0:     G0 -> G4 -> ...
 *               stream 1:     G1 -> G5 -> ...
 *               stream 2:     G2 -> G6 -> ...
 *               stream 3:     G3 -> G7 -> ...
 *               caller stream 在入口和出口通过 event 与四条计算流汇合。
 *
 * 多流可能通过组间并发降低总延迟，但也会增加 workspace、handle/event 管理和
 * 计算资源竞争。程序会明确报告 stream 数量、workspace 和 mode；前后比较必须固定
 * `--blas-streams`。即使使用四流，本程序仍不覆盖 Torch 参数转换、后端管理和
 * variable-K 产品后处理，最终产品级结论仍须运行 consumer 原生 benchmark。
 *
 * 七、后续优化记录建议
 * --------------------
 * 每次优化至少记录：CK commit、InstanceId/instance_name、架构、dtype/layout、
 * M/N/K/G、BLAS 与 origami SHA256、include_args_build、stream 模式、warmup、
 * samples、三次精度结果、median、CV、CK/BLAS ratio，以及 kernel spill/资源。
 * 只有同设备、同 BLAS、同调度模式、同参数协议下的数据才适合作为前后回归依据。
 *
 * 八、CSV 批量执行
 * ----------------
 * 同目录的 `run_grouped_gemm_device_args_cases.py` 可把通用 CSV 行转换成本程序参数，
 * 并同时保存原始日志、JSONL 和汇总 CSV。列表在 CSV 中用分号分隔，避免与 CSV
 * 本身的逗号冲突；runner 调用本程序时会转换回逗号：
 *
 *   runner=example_hcu/ck_tile/19_grouped_gemm/run_grouped_gemm_device_args_cases.py
 *   binary=build_device_args_example_bw/bin/tile_example_grouped_gemm_device_args_benchmark
 *   cases=example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args_cases_example.csv
 *   python3 "$runner" --binary="$binary" --cases="$cases" \
 *     --output-dir=hygon_tmp/grouped_gemm_results \
 *     --blas-root=/wksp/hipblaslt-install --max-cv-pct=10
 *
 * CSV 的核心字段是 dtype/layout/m/n/k/group_lengths/variable_capacity；产品 PRD
 * 可以在仓库外生成自己的 case CSV，但 CK 示例及其接口保持 consumer-independent。
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include "grouped_gemm.hpp"

namespace {

struct Options
{
    int m                   = 1024;
    int n                   = 1024;
    int k                   = 1024;
    int group_count         = 4;
    int warmup              = 10;
    int samples             = 30;
    int correctness_repeats = 3;
    std::uint32_t num_cu    = 0;
    std::string dtype       = "fp16";
    std::string layout      = "NT";
    std::string backend     = "both";
    std::string blas_library;
    std::string blas_root;
    std::vector<std::int64_t> group_lengths;
    std::int64_t variable_capacity = 0;
    int blas_streams                = 1;
    bool validate           = true;
    bool time               = true;
    bool include_args_build = true;
    bool allow_empty_groups = false;
    bool group_count_explicit = false;
};

[[noreturn]] void print_usage_and_exit(const char* program, int code)
{
    std::ostream& os = code == 0 ? std::cout : std::cerr;
    os << "用法: " << program << " [选项]\n"
       << "  --m=<int> --n=<int> --k=<int> --groups=<int>\n"
       << "  --group-lengths=2102,2120,... --variable-capacity=65536\n"
       << "  --dtype=fp16|bf16 --layout=NT|NN|TN\n"
       << "  --backend=ck|blas|both\n"
       << "  --blas-root=/path/to/hipblaslt-install\n"
       << "  --blas-lib=/path/to/libhipblaslt.so|libhipblas.so\n"
       << "  --time=1 --warmup=10 --samples=30 --correctness-repeats=3\n"
       << "  --num-cu=0 --validate=0|1 --include-args-build=0|1\n"
       << "  --allow-empty-groups=0|1 --blas-streams=1|4\n"
       << "\n"
       << "说明: 显式 --blas-lib 优先；其次为 --blas-root、"
          "CK_GROUPED_GEMM_BLAS_LIBRARY、PRIMUS_TURBO_HIPBLASLT_HOME、"
          "HIPBLASLT_ROOT，最后回退 libhipblas.so。\n"
       << "      hipBLASLt 安装根目录应包含 include/ 和 lib/；运行时会先从"
          "同一 lib 目录加载 liborigami.so.1。\n"
       << "      CK 性能默认包含一次 GPU device-args 构造，以贴近 Primus 调用；"
          "设置 --include-args-build=0 可只测最终 grouped GEMM launch。\n";
    std::exit(code);
}

std::pair<std::string, std::string> split_option(const std::string& argument)
{
    std::size_t begin = 0;
    while(begin < argument.size() && argument[begin] == '-')
        ++begin;
    const auto equal = argument.find('=', begin);
    if(equal == std::string::npos)
        return {argument.substr(begin), {}};
    return {argument.substr(begin, equal - begin), argument.substr(equal + 1)};
}

int parse_positive_or_zero(const std::string& name, const std::string& value)
{
    if(value.empty())
        throw std::runtime_error("选项 --" + name + " 缺少值");
    std::size_t consumed = 0;
    const long parsed    = std::stol(value, &consumed);
    if(consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max())
        throw std::runtime_error("选项 --" + name + " 的值无效: " + value);
    return static_cast<int>(parsed);
}

std::vector<std::int64_t> parse_nonnegative_list(const std::string& name,
                                                 const std::string& value)
{
    if(value.empty())
        throw std::runtime_error("选项 --" + name + " 缺少值");
    std::vector<std::int64_t> result;
    std::stringstream stream(value);
    std::string item;
    while(std::getline(stream, item, ','))
    {
        if(item.empty())
            throw std::runtime_error("选项 --" + name + " 包含空元素");
        std::size_t consumed = 0;
        const long long parsed = std::stoll(item, &consumed);
        if(consumed != item.size() || parsed < 0)
            throw std::runtime_error("选项 --" + name + " 的元素无效: " + item);
        result.push_back(static_cast<std::int64_t>(parsed));
    }
    if(result.empty())
        throw std::runtime_error("选项 --" + name + " 不能为空");
    return result;
}

Options parse_options(int argc, char* argv[])
{
    Options options;
    for(int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if(argument == "-h" || argument == "--help")
            print_usage_and_exit(argv[0], 0);
        const auto [name, value] = split_option(argument);
        if(name == "m")
            options.m = parse_positive_or_zero(name, value);
        else if(name == "n")
            options.n = parse_positive_or_zero(name, value);
        else if(name == "k")
            options.k = parse_positive_or_zero(name, value);
        else if(name == "groups" || name == "group-count" || name == "group_count")
        {
            options.group_count = parse_positive_or_zero(name, value);
            options.group_count_explicit = true;
        }
        else if(name == "warmup")
            options.warmup = parse_positive_or_zero(name, value);
        else if(name == "samples" || name == "repeat")
            options.samples = parse_positive_or_zero(name, value);
        else if(name == "time")
            options.time = parse_positive_or_zero(name, value) != 0;
        else if(name == "correctness-repeats")
            options.correctness_repeats = parse_positive_or_zero(name, value);
        else if(name == "num-cu")
            options.num_cu = static_cast<std::uint32_t>(parse_positive_or_zero(name, value));
        else if(name == "dtype")
            options.dtype = value;
        else if(name == "layout")
            options.layout = value;
        else if(name == "backend")
            options.backend = value;
        else if(name == "blas-lib")
            options.blas_library = value;
        else if(name == "blas-root")
            options.blas_root = value;
        else if(name == "group-lengths" || name == "group-lens")
            options.group_lengths = parse_nonnegative_list(name, value);
        else if(name == "variable-capacity" || name == "capacity")
            options.variable_capacity = static_cast<std::int64_t>(parse_positive_or_zero(name, value));
        else if(name == "blas-streams")
            options.blas_streams = parse_positive_or_zero(name, value);
        else if(name == "validate")
            options.validate = parse_positive_or_zero(name, value) != 0;
        else if(name == "include-args-build")
            options.include_args_build = parse_positive_or_zero(name, value) != 0;
        else if(name == "allow-empty-groups")
            options.allow_empty_groups = parse_positive_or_zero(name, value) != 0;
        else
            throw std::runtime_error("未知选项: " + argument);
    }

    const auto lower = [](unsigned char value) { return static_cast<char>(std::tolower(value)); };
    const auto upper = [](unsigned char value) { return static_cast<char>(std::toupper(value)); };
    std::transform(options.dtype.begin(), options.dtype.end(), options.dtype.begin(), lower);
    std::transform(options.layout.begin(), options.layout.end(), options.layout.begin(), upper);
    std::transform(options.backend.begin(), options.backend.end(), options.backend.begin(), lower);
    if(!options.time)
    {
        options.warmup  = 0;
        options.samples = 0;
    }
    if(!options.group_lengths.empty())
    {
        if(options.group_count_explicit &&
           options.group_count != static_cast<int>(options.group_lengths.size()))
            throw std::runtime_error("--groups 必须与 --group-lengths 元素数量一致");
        options.group_count = static_cast<int>(options.group_lengths.size());
        const std::int64_t effective =
            std::accumulate(options.group_lengths.begin(), options.group_lengths.end(), std::int64_t{0});
        if(effective <= 0)
            throw std::runtime_error("--group-lengths 至少包含一个正数");
        if(options.variable_capacity == 0)
            options.variable_capacity = effective;
        if(options.variable_capacity < effective)
            throw std::runtime_error("--variable-capacity 小于 group lengths 合计");
        if(!options.allow_empty_groups &&
           std::find(options.group_lengths.begin(), options.group_lengths.end(), 0) !=
               options.group_lengths.end())
            throw std::runtime_error("group length 为 0 时必须指定 --allow-empty-groups=1");
    }
    else if(options.variable_capacity != 0)
        throw std::runtime_error("--variable-capacity 必须与 --group-lengths 同时使用");

    if(options.m <= 0 || options.n <= 0 || options.k <= 0 || options.group_count <= 0)
        throw std::runtime_error("M/N/K/group count 必须为正数");
    if((options.time && options.samples <= 0) || options.correctness_repeats <= 0)
        throw std::runtime_error("计时开启时 samples、所有模式下 correctness-repeats 必须为正数");
    if(options.dtype != "fp16" && options.dtype != "bf16")
        throw std::runtime_error("dtype 只支持 fp16 或 bf16");
    if(options.layout != "NT" && options.layout != "NN" && options.layout != "TN")
        throw std::runtime_error("layout 只支持 NT、NN 或 TN");
    if(options.backend != "ck" && options.backend != "blas" && options.backend != "both")
        throw std::runtime_error("backend 只支持 ck、blas 或 both");
    if(options.blas_streams != 1 && options.blas_streams != 4)
        throw std::runtime_error("--blas-streams 只支持 1 或 4");
    if(!options.blas_library.empty() && !options.blas_root.empty())
        throw std::runtime_error("--blas-lib 与 --blas-root 不能同时指定");
    return options;
}

void hip_check(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

void hipblas_check(hipblasStatus_t status, const char* operation)
{
    if(status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed, status=" +
                                 std::to_string(static_cast<int>(status)));
}

int selector_layout(const std::string& layout)
{
    if(layout == "NT")
        return CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1;
    if(layout == "NN")
        return CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1;
    return CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1;
}

bool a_is_column_major(const std::string& layout) { return layout == "TN"; }
bool b_is_column_major(const std::string& layout) { return layout == "NT"; }

struct BlasLibraryRequest
{
    std::string path;
    bool use_hipblaslt = false;
};

std::string append_path(std::string root, const std::string& suffix)
{
    while(!root.empty() && root.back() == '/')
        root.pop_back();
    return root + suffix;
}

bool names_hipblaslt(const std::string& path)
{
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return lower.find("hipblaslt") != std::string::npos;
}

BlasLibraryRequest resolve_blas_library(const Options& options)
{
    if(!options.blas_library.empty())
        return {options.blas_library, names_hipblaslt(options.blas_library)};
    if(!options.blas_root.empty())
        return {append_path(options.blas_root, "/lib/libhipblaslt.so"), true};
    if(const char* library = std::getenv("CK_GROUPED_GEMM_BLAS_LIBRARY"))
        return {library, names_hipblaslt(library)};
    if(const char* root = std::getenv("PRIMUS_TURBO_HIPBLASLT_HOME"))
        return {append_path(root, "/lib/libhipblaslt.so"), true};
    if(const char* root = std::getenv("HIPBLASLT_ROOT"))
        return {append_path(root, "/lib/libhipblaslt.so"), true};
    return {"libhipblas.so", false};
}

ck_tile_hcu_grouped_gemm_dimension_v1 common_dimension(std::int64_t extent)
{
    return {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, extent};
}

ck_tile_hcu_grouped_gemm_dimension_v1 device_dimension(std::int64_t capacity)
{
    return {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1,
            0,
            capacity};
}

struct GroupPlan
{
    bool variable_m = false;
    bool variable_k = false;
    std::vector<std::int64_t> lengths;
    std::vector<std::int64_t> offsets;
    std::int64_t effective_length = 0;
    std::int64_t capacity         = 0;
    std::size_t a_elements        = 0;
    std::size_t b_elements        = 0;
    std::size_t c_elements        = 0;
    double operations             = 0;

    std::int64_t group_m(const Options& options, int group) const
    {
        return variable_m ? lengths.at(group) : options.m;
    }

    std::int64_t group_k(const Options& options, int group) const
    {
        return variable_k ? lengths.at(group) : options.k;
    }

    std::size_t a_offset(const Options& options, int group) const
    {
        return static_cast<std::size_t>(offsets.at(group)) *
               (options.layout == "TN" ? options.m : options.k);
    }

    std::size_t b_offset(const Options& options, int group) const
    {
        return options.layout == "TN"
                   ? static_cast<std::size_t>(offsets.at(group)) * options.n
                   : static_cast<std::size_t>(group) * options.k * options.n;
    }

    std::size_t c_offset(const Options& options, int group) const
    {
        return options.layout == "TN"
                   ? static_cast<std::size_t>(group) * options.m * options.n
                   : static_cast<std::size_t>(offsets.at(group)) * options.n;
    }
};

GroupPlan make_group_plan(const Options& options)
{
    GroupPlan plan;
    plan.variable_m = !options.group_lengths.empty() && options.layout != "TN";
    plan.variable_k = !options.group_lengths.empty() && options.layout == "TN";
    const std::int64_t common_length = options.layout == "TN" ? options.k : options.m;
    plan.lengths = options.group_lengths.empty()
                       ? std::vector<std::int64_t>(options.group_count, common_length)
                       : options.group_lengths;
    plan.offsets.resize(options.group_count);
    for(int group = 0; group < options.group_count; ++group)
    {
        plan.offsets[group] = plan.effective_length;
        plan.effective_length += plan.lengths[group];
    }
    plan.capacity = options.group_lengths.empty() ? plan.effective_length
                                                  : options.variable_capacity;

    if(options.layout == "TN")
    {
        plan.a_elements = static_cast<std::size_t>(plan.capacity) * options.m;
        plan.b_elements = static_cast<std::size_t>(plan.capacity) * options.n;
        plan.c_elements = static_cast<std::size_t>(options.group_count) * options.m * options.n;
    }
    else
    {
        plan.a_elements = static_cast<std::size_t>(plan.capacity) * options.k;
        plan.b_elements =
            static_cast<std::size_t>(options.group_count) * options.k * options.n;
        plan.c_elements = static_cast<std::size_t>(plan.capacity) * options.n;
    }
    for(int group = 0; group < options.group_count; ++group)
        plan.operations += 2.0 * plan.group_m(options, group) * options.n *
                           plan.group_k(options, group);
    return plan;
}

template <typename T>
T make_a_value(int group, int row, int reduction)
{
    const float value = static_cast<float>((group * 7 + row * 3 + reduction * 5) % 11 - 5) /
                        16.0f;
    return ck_tile::type_convert<T>(value);
}

template <typename T>
T make_b_value(int group, int reduction, int column)
{
    const float value =
        static_cast<float>((group * 3 + reduction * 2 + column * 7) % 13 - 6) / 16.0f;
    return ck_tile::type_convert<T>(value);
}

template <typename T>
void initialize_inputs(const Options& options,
                       const GroupPlan& plan,
                       std::vector<T>& a,
                       std::vector<T>& b)
{
    const T a_tail = ck_tile::type_convert<T>(63.0f);
    const T b_tail = ck_tile::type_convert<T>(-31.0f);
    a.assign(plan.a_elements, a_tail);
    b.assign(plan.b_elements, b_tail);
    const bool a_col = a_is_column_major(options.layout);
    const bool b_col = b_is_column_major(options.layout);

    for(int group = 0; group < options.group_count; ++group)
    {
        const auto group_m = static_cast<int>(plan.group_m(options, group));
        const auto group_k = static_cast<int>(plan.group_k(options, group));
        const std::size_t a_base = plan.a_offset(options, group);
        const std::size_t b_base = plan.b_offset(options, group);
        for(int row = 0; row < group_m; ++row)
            for(int reduction = 0; reduction < group_k; ++reduction)
            {
                const std::size_t offset = a_col
                                               ? static_cast<std::size_t>(reduction) * group_m + row
                                               : static_cast<std::size_t>(row) * group_k + reduction;
                a[a_base + offset] = make_a_value<T>(group, row, reduction);
            }
        for(int reduction = 0; reduction < group_k; ++reduction)
            for(int column = 0; column < options.n; ++column)
            {
                const std::size_t offset = b_col
                                               ? static_cast<std::size_t>(column) * group_k + reduction
                                               : static_cast<std::size_t>(reduction) * options.n + column;
                b[b_base + offset] = make_b_value<T>(group, reduction, column);
            }
    }
}

// 关键流程 4：和 Primus 一样，在 GPU 上构造 GemmTransKernelArg 数组。
// selector 若声明 C_TRANSPOSED_VIEW contract，则将 TN 问题改写成 C^T=B^T*A，
// 仅交换 descriptor 解释，不移动 A/B/C 的物理数据。
template <typename T>
__global__ void build_device_args(ck_tile::GemmTransKernelArg* args,
                                  const T* a,
                                  const T* b,
                                  T* c,
                                  const std::int64_t* group_lengths,
                                  const std::int64_t* group_offsets,
                                  int group_count,
                                  int common_m,
                                  int n,
                                  int common_k,
                                  int layout,
                                  bool transposed_output_view)
{
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if(group >= group_count)
        return;

    const bool variable_k = layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1;
    const std::int64_t length = group_lengths[group];
    const std::int64_t offset = group_offsets[group];
    const int m = variable_k ? (length > 0 ? common_m : 0) : static_cast<int>(length);
    const int k = variable_k ? static_cast<int>(length) : common_k;
    const std::size_t a_offset = static_cast<std::size_t>(offset) *
                                 (variable_k ? common_m : common_k);
    const std::size_t b_offset = variable_k
                                     ? static_cast<std::size_t>(offset) * n
                                     : static_cast<std::size_t>(group) * common_k * n;
    const std::size_t c_offset = variable_k
                                     ? static_cast<std::size_t>(group) * common_m * n
                                     : static_cast<std::size_t>(offset) * n;
    auto& karg = args[group].group_karg;

    if(transposed_output_view)
    {
        const_cast<std::array<const void*, 1>&>(karg.as_ptr)[0] = b + b_offset;
        const_cast<std::array<const void*, 1>&>(karg.bs_ptr)[0] = a + a_offset;
        karg.e_ptr                                                = c + c_offset;
        karg.M                                                    = n;
        karg.N                                                    = m;
        karg.K                                                    = k;
        karg.stride_As[0]                                         = n;
        karg.stride_Bs[0]                                         = common_m;
        karg.stride_E                                             = n;
        karg.k_batch                                              = 1;
        return;
    }

    const bool a_col = layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1;
    const bool b_col = layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1;
    const_cast<std::array<const void*, 1>&>(karg.as_ptr)[0] = a + a_offset;
    const_cast<std::array<const void*, 1>&>(karg.bs_ptr)[0] = b + b_offset;
    karg.e_ptr                                                = c + c_offset;
    karg.M                                                    = m;
    karg.N                                                    = n;
    karg.K                                                    = k;
    karg.stride_As[0]                                         = a_col ? common_m : common_k;
    karg.stride_Bs[0]                                         = b_col ? k : n;
    karg.stride_E                                             = n;
    karg.k_batch                                              = 1;
}

class DynamicHipBlas
{
    using CreateFn = hipblasStatus_t (*)(hipblasHandle_t*);
    using DestroyFn = hipblasStatus_t (*)(hipblasHandle_t);
    using SetStreamFn = hipblasStatus_t (*)(hipblasHandle_t, hipStream_t);
    using GemmExFn = hipblasStatus_t (*)(hipblasHandle_t,
                                         hipblasOperation_t,
                                         hipblasOperation_t,
                                         int,
                                         int,
                                         int,
                                         const void*,
                                         const void*,
                                         hipblasDatatype_t,
                                         int,
                                         const void*,
                                         hipblasDatatype_t,
                                         int,
                                         const void*,
                                         void*,
                                         hipblasDatatype_t,
                                         int,
                                         hipblasDatatype_t,
                                         hipblasGemmAlgo_t);

    void* library_          = nullptr;
    hipblasHandle_t handle_ = nullptr;
    DestroyFn destroy_      = nullptr;
    SetStreamFn set_stream_ = nullptr;
    GemmExFn gemm_ex_       = nullptr;
    std::string loaded_name_;

    template <typename Fn>
    Fn load_symbol(const char* name)
    {
        dlerror();
        void* symbol      = dlsym(library_, name);
        const char* error = dlerror();
        if(error != nullptr || symbol == nullptr)
            throw std::runtime_error(std::string("dlsym(") + name + ") failed: " +
                                     (error == nullptr ? "unknown error" : error));
        return reinterpret_cast<Fn>(symbol);
    }

public:
    explicit DynamicHipBlas(const std::string& requested_path)
    {
        std::vector<std::string> candidates;
        if(!requested_path.empty())
            candidates.push_back(requested_path);
        else if(const char* environment = std::getenv("CK_GROUPED_GEMM_BLAS_LIBRARY"))
            candidates.emplace_back(environment);
        else
            candidates = {"libhipblas.so", "libhipblas.so.2"};

        std::string errors;
        for(const auto& candidate : candidates)
        {
            library_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if(library_ != nullptr)
            {
                loaded_name_ = candidate;
                break;
            }
            const char* error = dlerror();
            errors += candidate + ": " + (error == nullptr ? "unknown error" : error) + "; ";
        }
        if(library_ == nullptr)
            throw std::runtime_error("无法加载 hipBLAS: " + errors);

        const auto create = load_symbol<CreateFn>("hipblasCreate");
        destroy_          = load_symbol<DestroyFn>("hipblasDestroy");
        set_stream_       = load_symbol<SetStreamFn>("hipblasSetStream");
        gemm_ex_          = load_symbol<GemmExFn>("hipblasGemmEx");
        hipblas_check(create(&handle_), "hipblasCreate");
    }

    ~DynamicHipBlas()
    {
        if(handle_ != nullptr && destroy_ != nullptr)
            destroy_(handle_);
        if(library_ != nullptr)
            dlclose(library_);
    }

    const std::string& loaded_name() const { return loaded_name_; }

    template <typename T>
    void grouped_gemm(const Options& options,
                      const GroupPlan& plan,
                      const T* a,
                      const T* b,
                      T* c,
                      hipStream_t stream)
    {
        hipblas_check(set_stream_(handle_, stream), "hipblasSetStream");
        const hipblasDatatype_t data_type =
            std::is_same_v<T, ck_tile::half_t> ? HIPBLAS_R_16F : HIPBLAS_R_16B;
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        const bool a_col  = a_is_column_major(options.layout);
        const bool b_col  = b_is_column_major(options.layout);

        // hipBLAS 使用 column-major。这里直接计算 C^T=B^T*A^T，使输出字节仍是
        // 调用方要求的 row-major C[M,N]，因此不需要额外 transpose kernel。
        const hipblasOperation_t trans_b_logical =
            b_col ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const hipblasOperation_t trans_a_logical =
            a_col ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        for(int group = 0; group < options.group_count; ++group)
        {
            const int group_m = static_cast<int>(plan.group_m(options, group));
            const int group_k = static_cast<int>(plan.group_k(options, group));
            if(group_m == 0 || group_k == 0)
                continue;
            const int lda = b_col ? group_k : options.n;
            const int ldb = a_col ? group_m : group_k;
            hipblas_check(gemm_ex_(handle_,
                                   trans_b_logical,
                                   trans_a_logical,
                                   options.n,
                                   group_m,
                                   group_k,
                                   &alpha,
                                   b + plan.b_offset(options, group),
                                   data_type,
                                   lda,
                                   a + plan.a_offset(options, group),
                                   data_type,
                                   ldb,
                                   &beta,
                                   c + plan.c_offset(options, group),
                                   data_type,
                                   options.n,
                                   HIPBLAS_R_32F,
                                   HIPBLAS_GEMM_DEFAULT),
                          "hipblasGemmEx");
        }
    }
};

class DynamicHipBlasLt
{
    using CreateFn                    = decltype(&hipblasLtCreate);
    using DestroyFn                   = decltype(&hipblasLtDestroy);
    using MatrixLayoutCreateFn        = decltype(&hipblasLtMatrixLayoutCreate);
    using MatrixLayoutDestroyFn       = decltype(&hipblasLtMatrixLayoutDestroy);
    using MatmulDescCreateFn          = decltype(&hipblasLtMatmulDescCreate);
    using MatmulDescDestroyFn         = decltype(&hipblasLtMatmulDescDestroy);
    using MatmulDescSetAttributeFn    = decltype(&hipblasLtMatmulDescSetAttribute);
    using PreferenceCreateFn          = decltype(&hipblasLtMatmulPreferenceCreate);
    using PreferenceDestroyFn         = decltype(&hipblasLtMatmulPreferenceDestroy);
    using PreferenceSetAttributeFn    = decltype(&hipblasLtMatmulPreferenceSetAttribute);
    using MatmulAlgoGetHeuristicFn    = decltype(&hipblasLtMatmulAlgoGetHeuristic);
    using MatmulFn                    = decltype(&hipblasLtMatmul);

    static constexpr std::size_t workspace_size_ = 32 * 1024 * 1024;
    static constexpr int max_stream_count_        = 4;

    void* origami_library_                = nullptr;
    void* library_                        = nullptr;
    std::array<void*, max_stream_count_> workspaces_{};
    std::array<hipblasLtHandle_t, max_stream_count_> handles_{};
    std::array<hipStream_t, max_stream_count_> compute_streams_{};
    std::array<hipEvent_t, max_stream_count_> complete_events_{};
    hipEvent_t sync_event_ = nullptr;
    int stream_count_      = 1;
    DestroyFn destroy_                    = nullptr;
    MatrixLayoutCreateFn layout_create_   = nullptr;
    MatrixLayoutDestroyFn layout_destroy_ = nullptr;
    MatmulDescCreateFn desc_create_       = nullptr;
    MatmulDescDestroyFn desc_destroy_     = nullptr;
    MatmulDescSetAttributeFn desc_set_    = nullptr;
    PreferenceCreateFn pref_create_       = nullptr;
    PreferenceDestroyFn pref_destroy_     = nullptr;
    PreferenceSetAttributeFn pref_set_    = nullptr;
    MatmulAlgoGetHeuristicFn heuristic_   = nullptr;
    MatmulFn matmul_                      = nullptr;
    std::string loaded_name_;
    std::string dependency_name_;

    template <typename Fn>
    Fn load_symbol(const char* name)
    {
        dlerror();
        void* symbol      = dlsym(library_, name);
        const char* error = dlerror();
        if(error != nullptr || symbol == nullptr)
            throw std::runtime_error(std::string("dlsym(") + name + ") failed: " +
                                     (error == nullptr ? "unknown error" : error));
        return reinterpret_cast<Fn>(symbol);
    }

    void preload_origami_sibling(const std::string& library_path)
    {
        const auto slash = library_path.find_last_of('/');
        if(slash == std::string::npos)
            return;
        const std::string directory = library_path.substr(0, slash);
        for(const char* soname : {"liborigami.so.1", "liborigami.so"})
        {
            const std::string candidate = append_path(directory, "/" + std::string(soname));
            origami_library_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if(origami_library_ != nullptr)
            {
                dependency_name_ = candidate;
                return;
            }
        }
    }

public:
    explicit DynamicHipBlasLt(const std::string& requested_path, int stream_count)
        : stream_count_(stream_count), loaded_name_(requested_path)
    {
        // 关键流程：只给 libhipblaslt.so 的绝对路径并不会让动态加载器自动搜索其
        // 同级目录中的 DT_NEEDED。Primus 通过安装根目录/rpath 解决；本示例为保持
        // 运行时可选 ELF，先以 RTLD_GLOBAL 预加载同目录的 liborigami.so.1。
        preload_origami_sibling(requested_path);
        library_ = dlopen(requested_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if(library_ == nullptr)
        {
            const char* error = dlerror();
            throw std::runtime_error("无法加载 hipBLASLt: " + requested_path + ": " +
                                     (error == nullptr ? "unknown error" : error));
        }

        const auto create = load_symbol<CreateFn>("hipblasLtCreate");
        destroy_          = load_symbol<DestroyFn>("hipblasLtDestroy");
        layout_create_    = load_symbol<MatrixLayoutCreateFn>("hipblasLtMatrixLayoutCreate");
        layout_destroy_   = load_symbol<MatrixLayoutDestroyFn>("hipblasLtMatrixLayoutDestroy");
        desc_create_      = load_symbol<MatmulDescCreateFn>("hipblasLtMatmulDescCreate");
        desc_destroy_     = load_symbol<MatmulDescDestroyFn>("hipblasLtMatmulDescDestroy");
        desc_set_ = load_symbol<MatmulDescSetAttributeFn>("hipblasLtMatmulDescSetAttribute");
        pref_create_ = load_symbol<PreferenceCreateFn>("hipblasLtMatmulPreferenceCreate");
        pref_destroy_ = load_symbol<PreferenceDestroyFn>("hipblasLtMatmulPreferenceDestroy");
        pref_set_ =
            load_symbol<PreferenceSetAttributeFn>("hipblasLtMatmulPreferenceSetAttribute");
        heuristic_ = load_symbol<MatmulAlgoGetHeuristicFn>("hipblasLtMatmulAlgoGetHeuristic");
        matmul_     = load_symbol<MatmulFn>("hipblasLtMatmul");
        if(stream_count_ > 1)
            hip_check(hipEventCreateWithFlags(&sync_event_, hipEventDisableTiming),
                      "hipEventCreate(sync)");
        for(int stream_index = 0; stream_index < stream_count_; ++stream_index)
        {
            hipblas_check(create(&handles_[stream_index]), "hipblasLtCreate");
            hip_check(hipMalloc(&workspaces_[stream_index], workspace_size_),
                      "hipBLASLt workspace hipMalloc");
            if(stream_count_ > 1)
            {
                hip_check(hipStreamCreateWithPriority(&compute_streams_[stream_index],
                                                      hipStreamNonBlocking,
                                                      -1),
                          "hipStreamCreateWithPriority");
                hip_check(hipEventCreateWithFlags(&complete_events_[stream_index],
                                                  hipEventDisableTiming),
                          "hipEventCreate(complete)");
            }
        }
    }

    ~DynamicHipBlasLt()
    {
        if(sync_event_ != nullptr)
            (void)hipEventDestroy(sync_event_);
        for(int stream_index = 0; stream_index < stream_count_; ++stream_index)
        {
            if(complete_events_[stream_index] != nullptr)
                (void)hipEventDestroy(complete_events_[stream_index]);
            if(compute_streams_[stream_index] != nullptr)
                (void)hipStreamDestroy(compute_streams_[stream_index]);
            if(workspaces_[stream_index] != nullptr)
                (void)hipFree(workspaces_[stream_index]);
            if(handles_[stream_index] != nullptr && destroy_ != nullptr)
                destroy_(handles_[stream_index]);
        }
        if(library_ != nullptr)
            dlclose(library_);
        if(origami_library_ != nullptr)
            dlclose(origami_library_);
    }

    const std::string& loaded_name() const { return loaded_name_; }
    const std::string& dependency_name() const { return dependency_name_; }
    int stream_count() const { return stream_count_; }

    template <typename T>
    void grouped_gemm(const Options& options,
                      const GroupPlan& plan,
                      const T* a,
                      const T* b,
                      T* c,
                      hipStream_t caller_stream)
    {
        const hipDataType data_type =
            std::is_same_v<T, ck_tile::half_t> ? HIP_R_16F : HIP_R_16BF;
        const bool a_col = a_is_column_major(options.layout);
        const bool b_col = b_is_column_major(options.layout);

        // 与 Primus hipblaslt_grouped_gemm.cu 相同，把 row-major C[M,N] 解释成
        // column-major C^T[N,M]，按 B^T * A^T 调用 hipBLASLt，不增加转置 kernel。
        const hipblasOperation_t trans_b_logical = b_col ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const hipblasOperation_t trans_a_logical = a_col ? HIPBLAS_OP_T : HIPBLAS_OP_N;

        auto launch_group = [&](int group, int stream_index, hipStream_t stream) {
            const std::int64_t group_m = plan.group_m(options, group);
            const std::int64_t group_k = plan.group_k(options, group);
            const std::int64_t b_rows  = b_col ? group_k : options.n;
            const std::int64_t b_cols  = b_col ? options.n : group_k;
            const std::int64_t b_ld    = b_col ? group_k : options.n;
            const std::int64_t a_rows  = a_col ? group_m : group_k;
            const std::int64_t a_cols  = a_col ? group_k : group_m;
            const std::int64_t a_ld    = a_col ? group_m : group_k;
            auto handle                = handles_[stream_index];
            auto workspace             = workspaces_[stream_index];

            hipblasLtMatrixLayout_t b_desc = nullptr;
            hipblasLtMatrixLayout_t a_desc = nullptr;
            hipblasLtMatrixLayout_t c_desc = nullptr;
            hipblasLtMatmulDesc_t operation_desc = nullptr;
            hipblasLtMatmulPreference_t preference = nullptr;
            hipblasLtMatmulHeuristicResult_t algorithm{};
            int returned_algorithm_count = 0;
            const std::int64_t workspace_size = static_cast<std::int64_t>(workspace_size_);
            const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

            hipblas_check(layout_create_(&b_desc, data_type, b_rows, b_cols, b_ld),
                          "hipblasLtMatrixLayoutCreate(B)");
            hipblas_check(layout_create_(&a_desc, data_type, a_rows, a_cols, a_ld),
                          "hipblasLtMatrixLayoutCreate(A)");
            hipblas_check(layout_create_(&c_desc, data_type, options.n, group_m, options.n),
                           "hipblasLtMatrixLayoutCreate(C)");
            hipblas_check(desc_create_(&operation_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                          "hipblasLtMatmulDescCreate");
            hipblas_check(desc_set_(operation_desc,
                                    HIPBLASLT_MATMUL_DESC_TRANSA,
                                    &trans_b_logical,
                                    sizeof(trans_b_logical)),
                          "hipblasLtMatmulDescSetAttribute(TRANSA)");
            hipblas_check(desc_set_(operation_desc,
                                    HIPBLASLT_MATMUL_DESC_TRANSB,
                                    &trans_a_logical,
                                    sizeof(trans_a_logical)),
                          "hipblasLtMatmulDescSetAttribute(TRANSB)");
            hipblas_check(desc_set_(operation_desc,
                                    HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                    &epilogue,
                                    sizeof(epilogue)),
                          "hipblasLtMatmulDescSetAttribute(EPILOGUE)");
            hipblas_check(pref_create_(&preference), "hipblasLtMatmulPreferenceCreate");
            hipblas_check(pref_set_(preference,
                                    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                    &workspace_size,
                                    sizeof(workspace_size)),
                          "hipblasLtMatmulPreferenceSetAttribute");
            hipblas_check(heuristic_(handle,
                                     operation_desc,
                                     b_desc,
                                     a_desc,
                                     c_desc,
                                     c_desc,
                                     preference,
                                     1,
                                     &algorithm,
                                     &returned_algorithm_count),
                          "hipblasLtMatmulAlgoGetHeuristic");
            if(returned_algorithm_count <= 0)
                throw std::runtime_error("hipBLASLt 未找到当前 GEMM 的可用算法");

            const float alpha = 1.0f;
            const float beta  = 0.0f;
            hipblas_check(matmul_(handle,
                                  operation_desc,
                                  &alpha,
                                  b + plan.b_offset(options, group),
                                  b_desc,
                                  a + plan.a_offset(options, group),
                                  a_desc,
                                  &beta,
                                  c + plan.c_offset(options, group),
                                  c_desc,
                                  c + plan.c_offset(options, group),
                                  c_desc,
                                  &algorithm.algo,
                                  workspace,
                                  workspace_size_,
                                  stream),
                          "hipblasLtMatmul");

            hipblas_check(pref_destroy_(preference), "hipblasLtMatmulPreferenceDestroy");
            hipblas_check(desc_destroy_(operation_desc), "hipblasLtMatmulDescDestroy");
            hipblas_check(layout_destroy_(c_desc), "hipblasLtMatrixLayoutDestroy(C)");
            hipblas_check(layout_destroy_(a_desc), "hipblasLtMatrixLayoutDestroy(A)");
            hipblas_check(layout_destroy_(b_desc), "hipblasLtMatrixLayoutDestroy(B)");
        };

        if(stream_count_ == 1)
        {
            for(int group = 0; group < options.group_count; ++group)
            {
                if(plan.group_m(options, group) == 0 || plan.group_k(options, group) == 0)
                    continue;
                launch_group(group, 0, caller_stream);
            }
            return;
        }

        int nonempty_groups = 0;
        for(int group = 0; group < options.group_count; ++group)
            nonempty_groups +=
                plan.group_m(options, group) > 0 && plan.group_k(options, group) > 0 ? 1 : 0;
        if(nonempty_groups == 0)
            return;
        const int used_streams = std::min(stream_count_, nonempty_groups);
        hip_check(hipEventRecord(sync_event_, caller_stream), "hipEventRecord(BLAS entry)");
        for(int stream_index = 0; stream_index < used_streams; ++stream_index)
            hip_check(hipStreamWaitEvent(compute_streams_[stream_index], sync_event_, 0),
                      "hipStreamWaitEvent(BLAS entry)");

        int dispatch_index = 0;
        for(int group = 0; group < options.group_count; ++group)
        {
            if(plan.group_m(options, group) == 0 || plan.group_k(options, group) == 0)
                continue;
            const int stream_index = dispatch_index++ % stream_count_;
            launch_group(group, stream_index, compute_streams_[stream_index]);
        }
        for(int stream_index = 0; stream_index < used_streams; ++stream_index)
            hip_check(hipEventRecord(complete_events_[stream_index], compute_streams_[stream_index]),
                      "hipEventRecord(BLAS complete)");
        for(int stream_index = 0; stream_index < used_streams; ++stream_index)
            hip_check(hipStreamWaitEvent(caller_stream, complete_events_[stream_index], 0),
                      "hipStreamWaitEvent(BLAS complete)");
    }
};

struct AccuracyResult
{
    std::size_t checked = 0;
    std::size_t failed  = 0;
    double max_abs      = 0;
    double max_rel      = 0;
};

template <typename T>
float as_float(T value)
{
    return ck_tile::type_convert<float>(value);
}

template <typename T>
AccuracyResult compare_values(const std::vector<T>& actual,
                              const std::vector<T>& expected,
                              float atol,
                              float rtol)
{
    AccuracyResult result;
    if(actual.size() != expected.size())
    {
        result.failed = 1;
        return result;
    }
    result.checked = actual.size();
    for(std::size_t i = 0; i < actual.size(); ++i)
    {
        const double a       = as_float(actual[i]);
        const double e       = as_float(expected[i]);
        const double abs_err = std::abs(a - e);
        const double rel_err = abs_err / std::max(std::abs(e), 1.0e-6);
        result.max_abs       = std::max(result.max_abs, abs_err);
        result.max_rel       = std::max(result.max_rel, rel_err);
        if(abs_err > atol + rtol * std::abs(e))
            ++result.failed;
    }
    return result;
}

template <typename T>
AccuracyResult check_cpu_samples(const Options& options,
                                 const GroupPlan& plan,
                                 const std::vector<T>& a,
                                 const std::vector<T>& b,
                                 const std::vector<T>& c,
                                 float atol,
                                 float rtol)
{
    AccuracyResult result;
    const bool a_col = a_is_column_major(options.layout);
    const bool b_col = b_is_column_major(options.layout);
    for(int group = 0; group < options.group_count; ++group)
    {
        const int group_m = static_cast<int>(plan.group_m(options, group));
        const int group_k = static_cast<int>(plan.group_k(options, group));
        if(group_m == 0 || group_k == 0)
            continue;
        const std::size_t a_base = plan.a_offset(options, group);
        const std::size_t b_base = plan.b_offset(options, group);
        const std::size_t c_base = plan.c_offset(options, group);
        const std::size_t c_group_elements = static_cast<std::size_t>(group_m) * options.n;
        const std::size_t checks_per_group = std::min<std::size_t>(257, c_group_elements);
        for(std::size_t check = 0; check < checks_per_group; ++check)
        {
            const std::size_t linear = checks_per_group == 1
                                           ? 0
                                           : check * (c_group_elements - 1) /
                                                 (checks_per_group - 1);
            const int row    = static_cast<int>(linear / options.n);
            const int column = static_cast<int>(linear % options.n);
            float reference  = 0;
            for(int reduction = 0; reduction < group_k; ++reduction)
            {
                const std::size_t a_offset =
                    a_col ? static_cast<std::size_t>(reduction) * group_m + row
                          : static_cast<std::size_t>(row) * group_k + reduction;
                const std::size_t b_offset =
                    b_col ? static_cast<std::size_t>(column) * group_k + reduction
                          : static_cast<std::size_t>(reduction) * options.n + column;
                reference += as_float(a[a_base + a_offset]) * as_float(b[b_base + b_offset]);
            }
            const double actual   = as_float(c[c_base + linear]);
            const double expected = as_float(ck_tile::type_convert<T>(reference));
            const double abs_err  = std::abs(actual - expected);
            const double rel_err  = abs_err / std::max(std::abs(expected), 1.0e-6);
            ++result.checked;
            result.max_abs = std::max(result.max_abs, abs_err);
            result.max_rel = std::max(result.max_rel, rel_err);
            if(abs_err > atol + rtol * std::abs(expected))
                ++result.failed;
        }
    }
    return result;
}

template <typename T>
AccuracyResult check_output_guards(const Options& options,
                                   const GroupPlan& plan,
                                   const std::vector<T>& c,
                                   T sentinel)
{
    AccuracyResult result;
    const float expected = as_float(sentinel);
    auto check_range = [&](std::size_t begin, std::size_t end) {
        for(std::size_t index = begin; index < end; ++index)
        {
            const double actual = as_float(c[index]);
            const double error  = std::abs(actual - expected);
            ++result.checked;
            result.max_abs = std::max(result.max_abs, error);
            if(error > 0)
                ++result.failed;
        }
    };

    if(plan.variable_m && plan.capacity > plan.effective_length)
        check_range(static_cast<std::size_t>(plan.effective_length) * options.n,
                    static_cast<std::size_t>(plan.capacity) * options.n);
    if(plan.variable_k)
    {
        const std::size_t group_elements = static_cast<std::size_t>(options.m) * options.n;
        for(int group = 0; group < options.group_count; ++group)
            if(plan.lengths[group] == 0)
                check_range(static_cast<std::size_t>(group) * group_elements,
                            static_cast<std::size_t>(group + 1) * group_elements);
    }
    return result;
}

struct TimingStats
{
    double median = 0;
    double mean   = 0;
    double min    = 0;
    double max    = 0;
    double cv     = 0;
};

TimingStats summarize(std::vector<float> values)
{
    if(values.empty())
        return {};
    TimingStats result;
    result.min  = *std::min_element(values.begin(), values.end());
    result.max  = *std::max_element(values.begin(), values.end());
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0;
    for(float value : values)
        variance += (value - result.mean) * (value - result.mean);
    variance /= values.size();
    result.cv = result.mean <= 0 ? 0 : std::sqrt(variance) / result.mean;
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    result.median = values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0
                                           : values[middle];
    return result;
}

class EventTimer
{
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_  = nullptr;

public:
    EventTimer()
    {
        hip_check(hipEventCreate(&start_), "hipEventCreate(start)");
        hip_check(hipEventCreate(&stop_), "hipEventCreate(stop)");
    }
    ~EventTimer()
    {
        if(start_ != nullptr)
            static_cast<void>(hipEventDestroy(start_));
        if(stop_ != nullptr)
            static_cast<void>(hipEventDestroy(stop_));
    }

    float measure(hipStream_t stream, const std::function<void()>& launch)
    {
        hip_check(hipEventRecord(start_, stream), "hipEventRecord(start)");
        launch();
        hip_check(hipEventRecord(stop_, stream), "hipEventRecord(stop)");
        hip_check(hipEventSynchronize(stop_), "hipEventSynchronize(stop)");
        float milliseconds = 0;
        hip_check(hipEventElapsedTime(&milliseconds, start_, stop_), "hipEventElapsedTime");
        return milliseconds;
    }
};

void print_timing(const char* backend,
                  const GroupPlan& plan,
                  const TimingStats& timing)
{
    // 某些外部 BLAS 实现会修改进程级 iostream 浮点格式；结构化记录必须显式恢复，
    // 否则 median/CV/ratio 可能被静默压缩为两位小数。
    std::cout << std::defaultfloat << std::setprecision(8);
    const double tflops = timing.median <= 0 ? 0 : plan.operations / timing.median / 1.0e9;
    std::cout << "PERF backend=" << backend << " median_ms=" << timing.median
              << " mean_ms=" << timing.mean << " min_ms=" << timing.min
              << " max_ms=" << timing.max << " cv=" << timing.cv
              << " tflops=" << tflops << std::endl;
}

template <typename T>
int run(const Options& options)
{
    const bool run_ck_backend   = options.backend == "ck" || options.backend == "both";
    const bool run_blas_backend = options.backend == "blas" || options.backend == "both";
    const GroupPlan plan         = make_group_plan(options);
    const T output_sentinel      = ck_tile::type_convert<T>(-7.0f);

    std::vector<T> host_a;
    std::vector<T> host_b;
    initialize_inputs(options, plan, host_a, host_b);
    std::vector<T> host_ck(plan.c_elements, output_sentinel);
    std::vector<T> host_blas(plan.c_elements, output_sentinel);
    const std::vector<T> initial_output(plan.c_elements, output_sentinel);

    ck_tile::DeviceMem device_a(plan.a_elements * sizeof(T));
    ck_tile::DeviceMem device_b(plan.b_elements * sizeof(T));
    ck_tile::DeviceMem device_ck(plan.c_elements * sizeof(T));
    ck_tile::DeviceMem device_blas(plan.c_elements * sizeof(T));
    ck_tile::DeviceMem device_args(static_cast<std::size_t>(options.group_count) *
                                   sizeof(ck_tile::GemmTransKernelArg));
    ck_tile::DeviceMem device_group_lengths(static_cast<std::size_t>(options.group_count) *
                                            sizeof(std::int64_t));
    ck_tile::DeviceMem device_group_offsets(static_cast<std::size_t>(options.group_count) *
                                            sizeof(std::int64_t));
    device_a.ToDevice(host_a.data());
    device_b.ToDevice(host_b.data());
    device_ck.ToDevice(initial_output.data());
    device_blas.ToDevice(initial_output.data());
    device_group_lengths.ToDevice(plan.lengths.data());
    device_group_offsets.ToDevice(plan.offsets.data());
    hipStream_t stream = nullptr;

    // 关键流程 1：把用户参数转换成 consumer-independent Problem。架构不由调用者
    // 传入，而是在 CK C ABI 内通过 get_hcu_target_enum() 检测。
    ck_tile_hcu_grouped_gemm_problem_v1 problem{};
    problem.struct_size   = sizeof(problem);
    problem.abi_version   = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    problem.data_type     = std::is_same_v<T, ck_tile::half_t>
                                ? CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1
                                : CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1;
    problem.layout        = selector_layout(options.layout);
    problem.group_count   = options.group_count;
    if(!options.group_lengths.empty())
    {
        const bool group_lengths_are_ragged =
            std::adjacent_find(options.group_lengths.begin(),
                               options.group_lengths.end(),
                               [](std::int64_t lhs, std::int64_t rhs) { return lhs != rhs; }) !=
            options.group_lengths.end();
        problem.problem_flags =
            (group_lengths_are_ragged
                 ? CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1
                 : 0u) |
            (options.allow_empty_groups
                 ? CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1
                 : 0u);
    }
    problem.m             = plan.variable_m ? device_dimension(plan.capacity)
                                            : common_dimension(options.m);
    problem.n             = common_dimension(options.n);
    problem.k             = plan.variable_k ? device_dimension(plan.capacity)
                                            : common_dimension(options.k);

    // 关键流程 2：pure selector 只返回已经预编译的 stable InstanceId；这里不会
    // 根据 M/N/K 在运行时实例化任何 C++ template。
    ck_tile_hcu_grouped_gemm_selection_v1 selection{};
    selection.struct_size = sizeof(selection);
    selection.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    const int select_status =
        ck_tile_hcu_grouped_gemm_select_device_args_v1(&problem, &selection);
    if(select_status != CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1)
        throw std::runtime_error("CK selector failed, status=" + std::to_string(select_status));

    // 关键流程 3：读取候选 contract。尤其是 BF16 TN 的转置输出视图必须在构造
    // device args 前确定，不能让 consumer 猜测具体 TileConfig。
    ck_tile_hcu_grouped_gemm_candidate_info_v1 candidate{};
    candidate.struct_size = sizeof(candidate);
    candidate.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    const int query_status =
        ck_tile_hcu_grouped_gemm_query_instance_v1(selection.instance_id, &candidate);
    if(query_status != CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1)
        throw std::runtime_error("CK candidate query failed, status=" +
                                 std::to_string(query_status));
    const bool transposed_output_view =
        (candidate.contract_flags &
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_CONTRACT_C_TRANSPOSED_VIEW_V1) != 0;

    auto build_args = [&] {
        constexpr int threads = 256;
        const int grids       = (options.group_count + threads - 1) / threads;
        build_device_args<T><<<grids, threads, 0, stream>>>(
            static_cast<ck_tile::GemmTransKernelArg*>(device_args.GetDeviceBuffer()),
            static_cast<const T*>(device_a.GetDeviceBuffer()),
            static_cast<const T*>(device_b.GetDeviceBuffer()),
            static_cast<T*>(device_ck.GetDeviceBuffer()),
            static_cast<const std::int64_t*>(device_group_lengths.GetDeviceBuffer()),
            static_cast<const std::int64_t*>(device_group_offsets.GetDeviceBuffer()),
            options.group_count,
            options.m,
            options.n,
            options.k,
            problem.layout,
            transposed_output_view);
        hip_check(hipGetLastError(), "build_device_args launch");
    };

    ck_tile_hcu_grouped_gemm_launch_v1 ck_launch{};
    ck_launch.struct_size          = sizeof(ck_launch);
    ck_launch.abi_version          = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    ck_launch.device_args          = device_args.GetDeviceBuffer();
    ck_launch.num_cu               = options.num_cu;
    ck_launch.expected_instance_id = selection.instance_id;
    ck_launch.stream               = stream;

    // 关键流程 5：run ABI 会再次执行同一个 pure selector，并校验 expected ID。
    // 随后 registry 按 ID/dtype/layout 调用链接进 ELF 的预编译 kernel specialization。
    auto launch_ck = [&] {
        if(options.include_args_build)
            build_args();
        const int status =
            ck_tile_hcu_grouped_gemm_run_device_args_v1(&problem, &ck_launch, nullptr);
        if(status != CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1)
            throw std::runtime_error("CK device-args launch failed, status=" +
                                     std::to_string(status));
    };
    build_args();

    std::unique_ptr<DynamicHipBlas> hipblas;
    std::unique_ptr<DynamicHipBlasLt> hipblaslt;
    BlasLibraryRequest blas_request;
    if(run_blas_backend)
    {
        blas_request = resolve_blas_library(options);
        if(blas_request.use_hipblaslt)
            hipblaslt =
                std::make_unique<DynamicHipBlasLt>(blas_request.path, options.blas_streams);
        else
        {
            if(options.blas_streams != 1)
                throw std::runtime_error("传统 hipBLAS 只支持 --blas-streams=1");
            hipblas = std::make_unique<DynamicHipBlas>(blas_request.path);
        }
    }
    auto launch_blas = [&] {
        if(hipblaslt)
            hipblaslt->grouped_gemm(options,
                                    plan,
                                    static_cast<const T*>(device_a.GetDeviceBuffer()),
                                    static_cast<const T*>(device_b.GetDeviceBuffer()),
                                    static_cast<T*>(device_blas.GetDeviceBuffer()),
                                    stream);
        else
            hipblas->grouped_gemm(options,
                                  plan,
                                  static_cast<const T*>(device_a.GetDeviceBuffer()),
                                  static_cast<const T*>(device_b.GetDeviceBuffer()),
                                  static_cast<T*>(device_blas.GetDeviceBuffer()),
                                  stream);
    };

    std::cout << "CONFIG arch=" << ck_tile::get_device_name()
              << " dtype=" << options.dtype << " layout=" << options.layout
               << " groups=" << options.group_count << " m=" << options.m
               << " n=" << options.n << " k=" << options.k
               << " variable_axis="
               << (plan.variable_m ? "M" : (plan.variable_k ? "K" : "none"))
               << " effective_length=" << plan.effective_length
               << " variable_capacity=" << plan.capacity
               << " timing=" << (options.time ? "on" : "off")
               << " warmup=" << options.warmup << " samples=" << options.samples
               << " include_args_build=" << options.include_args_build
               << " num_cu=" << options.num_cu
               << " blas_streams=" << options.blas_streams << std::endl;
    std::cout << "GROUP_LENGTHS values=";
    for(int group = 0; group < options.group_count; ++group)
        std::cout << (group == 0 ? "" : ",") << plan.lengths[group];
    std::cout << std::endl;
    std::cout << "SELECT instance_id=" << selection.instance_id
              << " instance_name=" << selection.instance_name
              << " contract_flags=" << candidate.contract_flags << std::endl;
    if(hipblaslt)
    {
        std::cout << "BLAS library=" << hipblaslt->loaded_name()
                  << " mode="
                  << (hipblaslt->stream_count() == 1 ? "hipblaslt_sequential_per_group"
                                                     : "hipblaslt_round_robin")
                  << " streams=" << hipblaslt->stream_count()
                  << " workspace_bytes="
                  << static_cast<std::size_t>(hipblaslt->stream_count()) * 32 * 1024 * 1024
                  << std::endl;
        if(!hipblaslt->dependency_name().empty())
            std::cout << "BLAS dependency=" << hipblaslt->dependency_name()
                      << " load_mode=RTLD_GLOBAL" << std::endl;
    }
    else if(hipblas)
        std::cout << "BLAS library=" << hipblas->loaded_name()
                  << " mode=hipblas_sequential_per_group" << std::endl;

    // 关键流程 6：先做 correctness，再允许性能计时。CK 与 BLAS 都和 CPU 抽样
    // reference 比较；both 模式还会逐元素比较 CK/BLAS 输出。
    if(options.validate)
    {
        const float atol = std::is_same_v<T, ck_tile::half_t> ? 0.08f : 0.35f;
        const float rtol = std::is_same_v<T, ck_tile::half_t> ? 0.015f : 0.03f;
        for(int repeat = 0; repeat < options.correctness_repeats; ++repeat)
        {
            if(run_ck_backend)
            {
                device_ck.ToDevice(initial_output.data());
                launch_ck();
                hip_check(hipStreamSynchronize(stream), "CK correctness synchronize");
                device_ck.FromDevice(host_ck.data());
                const auto cpu =
                    check_cpu_samples(options, plan, host_a, host_b, host_ck, atol, rtol);
                std::cout << std::defaultfloat << std::setprecision(8);
                std::cout << "ACCURACY backend=ck repeat=" << repeat
                          << " checked=" << cpu.checked << " failed=" << cpu.failed
                          << " max_abs=" << cpu.max_abs << " max_rel=" << cpu.max_rel
                          << std::endl;
                if(cpu.failed != 0)
                    return 2;
                const auto guard =
                    check_output_guards(options, plan, host_ck, output_sentinel);
                std::cout << "ACCURACY backend=ck_guard repeat=" << repeat
                          << " checked=" << guard.checked << " failed=" << guard.failed
                          << " max_abs=" << guard.max_abs << std::endl;
                if(guard.failed != 0)
                    return 2;
            }
            if(run_blas_backend)
            {
                device_blas.ToDevice(initial_output.data());
                launch_blas();
                hip_check(hipStreamSynchronize(stream), "BLAS correctness synchronize");
                device_blas.FromDevice(host_blas.data());
                const auto cpu =
                    check_cpu_samples(options, plan, host_a, host_b, host_blas, atol, rtol);
                std::cout << std::defaultfloat << std::setprecision(8);
                std::cout << "ACCURACY backend=blas repeat=" << repeat
                          << " checked=" << cpu.checked << " failed=" << cpu.failed
                          << " max_abs=" << cpu.max_abs << " max_rel=" << cpu.max_rel
                          << std::endl;
                if(cpu.failed != 0)
                    return 3;
                const auto guard =
                    check_output_guards(options, plan, host_blas, output_sentinel);
                std::cout << "ACCURACY backend=blas_guard repeat=" << repeat
                          << " checked=" << guard.checked << " failed=" << guard.failed
                          << " max_abs=" << guard.max_abs << std::endl;
                if(guard.failed != 0)
                    return 3;
            }
            if(run_ck_backend && run_blas_backend)
            {
                const auto parity = compare_values(host_ck, host_blas, atol, rtol);
                std::cout << "ACCURACY backend=ck_vs_blas repeat=" << repeat
                          << " checked=" << parity.checked << " failed=" << parity.failed
                          << " max_abs=" << parity.max_abs << " max_rel=" << parity.max_rel
                          << std::endl;
                if(parity.failed != 0)
                    return 4;
            }
        }
    }

    if(!options.time)
    {
        std::cout << "TIMING status=off warmup=0 samples=0" << std::endl;
        std::cout << "SUMMARY result=PASS" << std::endl;
        return 0;
    }

    // 关键流程 7：使用 GPU event；both 模式按样本奇偶交替 CK/BLAS 顺序，减少
    // 温度和频率漂移。每个样本独立同步 stop event，最终报告 median 和 CV。
    for(int i = 0; i < options.warmup; ++i)
    {
        if(run_ck_backend)
            launch_ck();
        if(run_blas_backend)
            launch_blas();
    }
    hip_check(hipStreamSynchronize(stream), "warmup synchronize");

    EventTimer timer;
    std::vector<float> ck_samples;
    std::vector<float> blas_samples;
    ck_samples.reserve(options.samples);
    blas_samples.reserve(options.samples);
    for(int sample = 0; sample < options.samples; ++sample)
    {
        const bool ck_first = sample % 2 == 0;
        if(run_ck_backend && (!run_blas_backend || ck_first))
            ck_samples.push_back(timer.measure(stream, launch_ck));
        if(run_blas_backend)
            blas_samples.push_back(timer.measure(stream, launch_blas));
        if(run_ck_backend && run_blas_backend && !ck_first)
            ck_samples.push_back(timer.measure(stream, launch_ck));
    }

    TimingStats ck_timing;
    TimingStats blas_timing;
    if(run_ck_backend)
    {
        ck_timing = summarize(ck_samples);
        print_timing("ck_device_args", plan, ck_timing);
    }
    if(run_blas_backend)
    {
        blas_timing = summarize(blas_samples);
        print_timing(blas_request.use_hipblaslt ? "hipblaslt" : "hipblas",
                     plan,
                     blas_timing);
    }
    if(run_ck_backend && run_blas_backend)
        std::cout << "RATIO ck_over_blas="
                  << (blas_timing.median <= 0 ? 0 : ck_timing.median / blas_timing.median)
                  << std::endl;

    std::cout << "SUMMARY result=PASS" << std::endl;
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const Options options = parse_options(argc, argv);
        if(options.dtype == "fp16")
            return run<ck_tile::half_t>(options);
        return run<ck_tile::bf16_t>(options);
    }
    catch(const std::exception& error)
    {
        std::cerr << "SUMMARY result=FAIL reason=\"" << error.what() << "\"" << std::endl;
        return 1;
    }
}
