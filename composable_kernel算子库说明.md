# Composable Kernel算子库说明

# 1. Composable Kernel 简介

Composable Kernel（CK）是面向 HIP/HCU/GPU 的高性能算子模板库，核心目标是用 tile-based programming model 和 tensor coordinate transformation 组织矩阵乘、卷积、归一化、注意力、MoE 等深度学习算子。当前 `composable_kernel` 工程是在源基础上适配海光 HCU 的源码树，包含传统 `ck` device operator、`ck_tile` tile-programming operator、示例程序、profiler、Python 侧 ck4inductor 适配等模块。

本文档只整理HCU适配 `example_hcu` 中示例调用到的接口。

## 1.1 编译使用

### 1.1.1 环境依赖

常用环境如下：

- DTK 或 ROCm HIP 工具链。HCU 环境优先使用 `/opt/dtk/bin/aicc`，其次使用 `/opt/dtk/llvm/bin/hipcc`。
- CMake 3.14 及以上。
- Python3。`ck_tile` 的 FMHA、LayerNorm、RMSNorm 等示例包含代码生成脚本，配置或构建时可能调用 Python。
- GPU target 按机器选择。NMZ/gfx938 常用 `GPU_TARGETS="gfx938"`；双架构包可用 `GPU_TARGETS="gfx936;gfx938"`。

### 1.1.2 开发构建

开发构建脚本为 `script/cmake-ck-dev.sh`，默认开启 `BUILD_EXAMPLE=ON`，并配置目标机器架构，如NMZ使用 `GPU_TARGETS="gfx938"`：

```bash
mkdir -p build_dev
cd build_dev
../script/cmake-ck-dev.sh ..
```

等价的关键 CMake 选项包括：

```bash
cmake \
  -D CMAKE_PREFIX_PATH=/opt/dtk \
  -D CMAKE_CXX_COMPILER=/opt/dtk/bin/aicc \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_DEV=ON \
  -D BUILD_EXAMPLE=ON \
  -D GPU_TARGETS="gfx938" \
  -D USE_BITINT_EXTENSION=ON \
  ..
```

全部编译：
```bash
make -j32
```

常用的单独编译目标：

```bash
# 编译全部通过 add_example_executable 接入的 HCU 示例
cmake --build build_dev --target examples -j 16

# 编译 ckProfiler
cmake --build build_dev --target ckProfiler -j 16

# 单独编译某个 example_hcu target
cmake --build build_dev --target tile_example_gemm_basic -j 16
cmake --build build_dev --target example_hcu_gemm_xdl_v3 -j 16
```

### 1.1.3 Release 构建和安装

Release 配置脚本为 `script/cmake-ck-release.sh`，默认 `BUILD_DEV=OFF`，`GPU_TARGETS="gfx936;gfx938"`，并把包安装前缀设置到当前工具链路径：

```bash
mkdir -p build_release
cd build_release
../script/cmake-ck-release.sh ..
cmake --build . --target ckProfiler -j 16
cmake --build . --target package -j 16
```

工程顶层输出目录约定：

| 输出 | 默认路径 |
| --- | --- |
| 可执行文件 | `build*/bin` |
| 静态库/动态库 | `build*/lib` |
| 生成头文件 | `build*/include` |
| profiler | `build*/bin/ckProfiler` |

### 1.1.4 运行示例

示例可执行文件通常位于构建目录的 `bin` 下：

```bash
cd build_dev/bin
./tile_example_gemm_basic -m=1024 -n=2048 -k=64 -prec=fp16
./example_hcu_gemm_xdl_bf16_v3 1 1 0 256 256 128 128 128 256 1
./tile_example_moe_sorting -t=32 -e=8 -k=2
```

部分传统 CK 示例使用位置参数，常见前三个参数为：

| 参数 | 含义 |
| --- | --- |
| `arg1` | 是否校验，`0` 不校验，`1` 校验 |
| `arg2` | 初始化方式，`0` 不初始化，`1` 整数随机，`2` 小数随机 |
| `arg3` | 是否计时 kernel，`0` 不计时，`1` 计时 |

`ck_tile` 示例多数使用 `-key=value` 风格，可通过 `-?` 查看参数。

## 1.2 主要模块

| 模块 | 目录 | 说明 |
| --- | --- | --- |
| CK 核心模板 | `include/ck` | 传统 CK device operator、tensor descriptor、block/thread transfer、数据类型、utility 等。 |
| CK Tile 核心 | `include/ck_tile` | tile-programming API、host utilities、GEMM/FMHA/conv/norm/MoE 等 tile 算子组件。 |
| 实例库 | `library` | device operation instance、profiler/library 公共头文件和实例注册。 |
| HCU 示例 | `example_hcu` | 本文统计的示例入口，包含 legacy CK 示例和 `ck_tile` 示例。 |
| Profiler | `profiler` | `ckProfiler` 命令行性能测试入口，按 profiler target 链接相关实例。 |
| 构建脚本 | `script` | `cmake-ck-dev.sh`、`cmake-ck-release.sh`、profiling 脚本等。 |
| Python 适配 | `python/ck4inductor` | 面向 TorchInductor/CK4Inductor 的 Python 侧包装和模板解析。 |
| 文档 | `docs` | Sphinx 文档源文件和 CK 架构说明。 |

# 2. 算子简介

下表只列入`example_hcu` 示例接口：

| 序号 | 算子/接口 | 示例 target | 来源路径 |
| --- | --- | --- | --- |
| 1 | GEMM MMAC | `example_gemm_mmac_nt_fp16` | `example_hcu/05_gemm` |
| 2 | GEMM XDL v3 | `example_hcu_gemm_xdl_bf16_v3`、`example_hcu_gemm_multi_d_xdl_bf16_v3` | `example_hcu/25_gemm_xdl_v3` |
| 3 | Batched GEMM XDL v3 | `example_hcu_batched_gemm_xdl_bf16_v3` | `example_hcu/24_batched_gemm` |
| 4 | CK Tile GEMM | `tile_example_gemm_basic` | `example_hcu/ck_tile/03_gemm` |
| 5 | Grouped GEMM | `tile_example_grouped_gemm` 及 fast 变体 | `example_hcu/ck_tile/19_grouped_gemm` |
| 6 | FMHA | `tile_example_fmha_fwd`、`tile_example_fmha_bwd` | `example_hcu/ck_tile/01_fmha` |
| 7 | MoE Sorting | `tile_example_moe_sorting` | `example_hcu/ck_tile/13_moe_sorting` |
| 8 | Fused MoE | `tile_example_fused_moe` | `example_hcu/ck_tile/17_fused_moe` |
| 9 | MoE Quant | `tile_example_moe_quant` | `example_hcu/ck_tile/18_moe_quant` |
| 10 | Grouped Conv Bwd Data | `example_grouped_conv_bwd_data_mmac_nhwc_v2_fp16_cshuffle` | `example_hcu/02_grouped_conv_bwd_data_mmac` |
| 11 | Tensor Layout Transform | `example_tensor_layout_transform_ngchw_ngchwc32_fp16` | `example_hcu/04_tensor_layout_transform` |
| 12 | BatchNorm Forward/Backward | `example_batchnorm_forward_training`、`example_batchnorm_backward` | `example_hcu/08_batchnorm` |
| 13 | LayerNorm2D | `tile_example_layernorm2d_fwd` | `example_hcu/ck_tile/02_layernorm2d` |
| 14 | RMSNorm2D | `tile_example_rmsnorm2d_fwd` | `example_hcu/ck_tile/10_rmsnorm2d` |
| 15 | Add + RMSNorm2D + RdQuant | `tile_example_add_rmsnorm2d_rdquant_fwd` | `example_hcu/ck_tile/11_add_rmsnorm2d_rdquant` |
| 16 | CK Tile Conv Fwd | `tile_example_conv_fwd` | `example_hcu/ck_tile/14_conv` |
| 17 | CK Tile Conv3D Fwd | `tile_example_conv3d_fwd` | `example_hcu/ck_tile/16_conv3d_fwd` |

# 3. 主要算子接口

本节面向 CK 算子调用者，说明 `example_hcu` 中示例实际调用到的 C/C++ 接口。这里描述客户端集成时需要包含的头文件、需要填写的参数对象、返回值语义以及调用示例。

接口大体分为两类：

- `ck_tile` wrapper/C ABI：示例目录直接提供 `float xxx(traits, args, stream_config)` 或 `extern "C"` 函数，适合上层框架再封装成共享库接口。
- legacy CK DeviceOp：示例直接实例化 `Device*` 模板类型，通过 `MakeArgument`、`IsSupportedArgument`、`MakeInvoker`、`Run` 完成调用。

公共约定：

- 所有指针参数均为 device pointer，host 侧只负责填充 descriptor/args。
- `ck_tile::stream_config` 负责传入 HIP stream、计时开关、日志级别、warmup/repeat 等运行配置。
- `float` 返回值通常为平均 kernel 时间，返回负值表示没有匹配到可用 kernel 或参数不支持。
- `traits` 负责选择数据类型、mask、量化、布局等编译期/派发条件，`args` 负责传入运行期 shape、stride 和数据指针。

## 3.1 CK Tile GEMM: `gemm_calc`

来源路径：`example_hcu/ck_tile/03_gemm/gemm_basic.hpp`、`example_hcu/ck_tile/03_gemm/gemm_basic.cpp`。

功能：执行基础 GEMM：

```text
C[M, N] = A[M, K] * B[K, N]
```

当前示例头文件中的 `Types` 固定为 `GemmBasicTypeConfig<ck_tile::half_t, ck_tile::half_t>`，即 FP16 输入、FP16 输出、FP32 accumulator。

接口原型：

```cpp
#include "gemm_basic.hpp"

struct gemm_basic_args
{
    const void* p_a;
    const void* p_b;
    void* p_c;
    ck_tile::index_t kbatch;
    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    ck_tile::index_t stride_C;
};

float gemm_calc(gemm_basic_args args, const ck_tile::stream_config& s);
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `p_a` | A 矩阵 device pointer。 |
| `p_b` | B 矩阵 device pointer。 |
| `p_c` | C 输出矩阵 device pointer。 |
| `kbatch` | Split-K batch 数，基础示例通常填 `1`。 |
| `M/N/K` | GEMM 维度。 |
| `stride_A/stride_B/stride_C` | 对应矩阵的行/列 stride，需与示例实例采用的布局一致。 |
| `stream_config` | HIP stream 和计时配置。 |

调用示例：

```cpp
hipStream_t stream = nullptr;

gemm_basic_args args{
    a_dev,
    b_dev,
    c_dev,
    1,
    M,
    N,
    K,
    stride_a,
    stride_b,
    stride_c,
};

float time_ms = gemm_calc(args, ck_tile::stream_config{stream, true});
```

## 3.2 Grouped GEMM C ABI: `ck_tile_hcu_grouped_gemm_run`

声明路径：`example_hcu/ck_tile/19_grouped_gemm/grouped_gemm.hpp`。实现路径：
`example_hcu/ck_tile/19_grouped_gemm/grouped_gemm.cpp` 与 `instances/`。aiter
等现有消费者通过把该 example 目录加入 include path 来复用接口，本次不新增
另一个 C ABI 头文件。

功能：一次调用提交多组 GEMM descriptor，每组可以有不同的 `M/N/K/stride`。该接口是 `extern "C"` ABI，适合被 Python extension、推理框架 runtime 或其他 C/C++ 客户端动态加载。

接口原型：

```cpp
#include "grouped_gemm.hpp"

extern "C" {

enum ck_tile_hcu_grouped_gemm_dtype
{
    CK_TILE_HCU_GROUPED_GEMM_FP16 = 0,
    CK_TILE_HCU_GROUPED_GEMM_FP8  = 1,
    CK_TILE_HCU_GROUPED_GEMM_INT8 = 2,
    CK_TILE_HCU_GROUPED_GEMM_BF8  = 3,
    CK_TILE_HCU_GROUPED_GEMM_BF16 = 4,
    CK_TILE_HCU_GROUPED_GEMM_INT4 = 5
};

struct ck_tile_hcu_grouped_gemm_desc
{
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    int k_batch;
    int M;
    int N;
    int K;
    int stride_A;
    int stride_B;
    int stride_C;
    int num_d_tensors;
    const void* const* d_ptrs;
    const int* stride_Ds;
};

std::size_t ck_tile_hcu_grouped_gemm_workspace_size(int group_count,
                                                    int num_d_tensors = 0);

int ck_tile_hcu_grouped_gemm_run(const ck_tile_hcu_grouped_gemm_desc* descs,
                                 int group_count,
                                 int dtype,
                                 char a_layout,
                                 char b_layout,
                                 void* workspace,
                                 hipStream_t stream);

}
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `descs` | `group_count` 个 GEMM descriptor 数组，host 内存。 |
| `group_count` | GEMM 组数。 |
| `dtype` | `ck_tile_hcu_grouped_gemm_dtype` 枚举值。 |
| `a_layout/b_layout` | `R` 表示 row-major，`C` 表示 column-major。 |
| `workspace` | device workspace，大小由 `ck_tile_hcu_grouped_gemm_workspace_size()` 返回。 |
| `num_d_tensors` | epilogue D tensor 数量，当前 C ABI 支持 `0/1/2`。 |
| `d_ptrs/stride_Ds` | D tensor device pointer 数组和 stride 数组；无 D tensor 时为 `nullptr`。 |

返回值：

| 返回值 | 含义 |
| --- | --- |
| `0` | 成功提交 kernel。 |
| `-1/-2/-3/-4/-6` | descriptor、workspace、布局、shape 或 D tensor 数量不支持。 |
| `-5` | dtype 当前构建未启用或不支持。 |
| `-100` | C ABI dispatch 捕获到异常。 |

调用示例：

```cpp
std::vector<ck_tile_hcu_grouped_gemm_desc> descs(group_count);
descs[0] = {
    a0_dev,
    b0_dev,
    c0_dev,
    1,
    M0,
    N0,
    K0,
    stride_a0,
    stride_b0,
    stride_c0,
    0,
    nullptr,
    nullptr,
};

std::size_t ws_bytes = ck_tile_hcu_grouped_gemm_workspace_size(group_count);
void* ws_dev = nullptr;
hipMalloc(&ws_dev, ws_bytes);

int rc = ck_tile_hcu_grouped_gemm_run(descs.data(),
                                      group_count,
                                      CK_TILE_HCU_GROUPED_GEMM_FP16,
                                      'R',
                                      'R',
                                      ws_dev,
                                      stream);
```

上例的 `'R','R'` 是 NN layout。variable-K 不需要另一个 API：为每个
descriptor 分别填写 `K`、`stride_A` 和 `stride_B` 即可。例如三组 NN 可用
`K={128,192,256}`、`stride_A=K`、`stride_B=N`；当前 FP16/BF16 CompV4
contract 要求 `k_batch=1`、每组 `K>=128` 且按 64 对齐。可直接运行
`tile_example_grouped_gemm_c_api_smoke` 验证该框架入口。

## 3.3 FMHA: `fmha_fwd` / `fmha_bwd`

来源路径：`example_hcu/ck_tile/01_fmha/fmha_fwd.hpp`、`example_hcu/ck_tile/01_fmha/fmha_bwd.hpp`。

功能：fused multi-head attention forward/backward。Forward 支持 batch/group mode、MQA/GQA、causal/window mask、bias/alibi、LSE、dropout、paged KV、split KV、append KV、batch prefill 等派生接口。

主要接口：

```cpp
#include "fmha_fwd.hpp"
#include "fmha_bwd.hpp"

float fmha_fwd(fmha_fwd_traits traits,
               fmha_fwd_args args,
               const ck_tile::stream_config& config);

template <int Version = 2>
float fmha_bwd(const fmha_bwd_traits& traits,
               fmha_bwd_args args,
               const ck_tile::stream_config& config);

float fmha_fwd_pagedkv(fmha_fwd_pagedkv_traits& traits,
                       fmha_fwd_pagedkv_args& args,
                       const ck_tile::stream_config& config);

float fmha_fwd_splitkv(fmha_fwd_splitkv_traits traits,
                       fmha_fwd_splitkv_args args,
                       const ck_tile::stream_config& config);

float fmha_fwd_appendkv(fmha_fwd_appendkv_traits traits,
                        fmha_fwd_appendkv_args args,
                        const ck_tile::stream_config& config);

float fmha_batch_prefill(fmha_batch_prefill_traits traits,
                         fmha_batch_prefill_args args,
                         const ck_tile::stream_config& config);
```

`fmha_fwd_traits` 关键字段：

| 字段 | 说明 |
| --- | --- |
| `hdim_q/hdim_v` | Q/K 和 V head dim，用于 kernel 派发。 |
| `data_type` | `fp16/bf16/fp8/...` 等输入数据类型字符串。 |
| `is_group_mode` | 是否 varlen/group mode。 |
| `is_v_rowmajor` | V layout 是否 row-major。 |
| `has_logits_soft_cap` | 是否启用 logits soft cap。 |
| `mask_type` | `mask_enum::no_mask/mask_top_left/mask_bottom_right/window_generic`。 |
| `bias_type` | `bias_enum::no_bias/elementwise_bias/alibi`。 |
| `has_lse` | 是否输出 LSE。 |
| `has_dropout` | 是否启用 dropout。 |
| `qscale_type` | `quant_scale_enum::no_scale/pertensor/blockscale/kv_blockscale/mx`。 |
| `skip_min_seqlen_q` | 是否跳过 min seqlen q 检查。 |
| `has_sink` | 是否启用 sink score。 |

`fmha_fwd_args` 关键字段：

| 字段 | 说明 |
| --- | --- |
| `q_ptr/k_ptr/v_ptr/o_ptr` | Q/K/V 输入和 O 输出 device pointer。 |
| `bias_ptr` | elementwise bias 或 alibi slope pointer，无 bias 时为 `nullptr`。 |
| `q_descale_ptr/k_descale_ptr/v_descale_ptr` | FP8/MX 等量化路径的 descale pointer。 |
| `rand_val_ptr/lse_ptr` | dropout 随机值和 LSE 输出 pointer，可按 trait 置空。 |
| `seqstart_* / seqlen_* / cu_seqlen_*` | group/varlen 或 padding 模式下的序列长度元数据。 |
| `seqlen_q/seqlen_k/batch/max_seqlen_q` | batch mode 或 group mode 的序列维度。 |
| `hdim_q/hdim_v/nhead_q/nhead_k` | head 维度和 Q/K head 数，支持 GQA/MQA。 |
| `scale_s/logits_soft_cap` | attention scale 和 logits cap。 |
| `stride_* / nhead_stride_* / batch_stride_*` | 各 tensor 在 S/H/B 维度上的 stride。 |
| `window_size_left/window_size_right/mask_type` | sliding window/causal mask 运行参数。 |
| `p_drop/drop_seed_offset` | dropout 概率和 seed/offset。 |

调用示例：

```cpp
fmha_fwd_traits traits{
    128,
    128,
    "fp16",
    false,
    true,
    false,
    mask_enum::mask_bottom_right,
    bias_enum::no_bias,
    true,
    false,
    quant_scale_enum::no_scale,
};

fmha_fwd_args args{};
args.q_ptr = q_dev;
args.k_ptr = k_dev;
args.v_ptr = v_dev;
args.o_ptr = o_dev;
args.lse_ptr = lse_dev;
args.seqlen_q = seqlen_q;
args.seqlen_k = seqlen_k;
args.batch = batch;
args.max_seqlen_q = seqlen_q;
args.hdim_q = 128;
args.hdim_v = 128;
args.nhead_q = nhead_q;
args.nhead_k = nhead_k;
args.scale_s = 1.0f / std::sqrt(128.0f);
args.stride_q = 128;
args.stride_k = 128;
args.stride_v = 128;
args.stride_o = 128;
args.nhead_stride_q = seqlen_q * 128;
args.nhead_stride_k = seqlen_k * 128;
args.nhead_stride_v = seqlen_k * 128;
args.nhead_stride_o = seqlen_q * 128;
args.batch_stride_q = nhead_q * seqlen_q * 128;
args.batch_stride_k = nhead_k * seqlen_k * 128;
args.batch_stride_v = nhead_k * seqlen_k * 128;
args.batch_stride_o = nhead_q * seqlen_q * 128;
args.window_size_left = -1;
args.window_size_right = 0;
args.mask_type = static_cast<int>(traits.mask_type);
args.p_drop = 0.0f;

float time_ms = fmha_fwd(traits, args, ck_tile::stream_config{stream, true});
```

Backward 调用方式相同：填充 `fmha_bwd_traits` 和 `fmha_bwd_args`，其中 `q/k/v/o/lse/do` 为输入，`dq/dk/dv/dbias/d_sink` 为输出，`workspace_ptr` 需要由客户端按 launcher 或封装层要求分配。

## 3.4 MoE Sorting: `moe_sorting`

来源路径：`example_hcu/ck_tile/13_moe_sorting/moe_sorting_api.hpp`、`example_hcu/ck_tile/13_moe_sorting/moe_sorting_api.cpp`。

功能：将 `[tokens, topk]` 的专家路由结果按 expert 分桶、排序并 padding 到 `unit_size` 对齐，为 Fused MoE/GEMM 的 grouped token 输入做准备。

接口原型：

```cpp
#include "moe_sorting_api.hpp"

struct moe_sorting_trait
{
    std::string index_type;
    std::string weight_type;
    bool local_expert_masking;
};

struct moe_sorting_args : public ck_tile::MoeSortingHostArgs
{
};

int moe_sorting_get_workspace_size(int tokens, int num_experts, int topk);

float moe_sorting(moe_sorting_trait t,
                  moe_sorting_args a,
                  ck_tile::stream_config s);

float moe_sorting_mp(moe_sorting_trait t,
                     moe_sorting_args a,
                     ck_tile::stream_config s);
```

`moe_sorting_trait`：

| 字段 | 说明 |
| --- | --- |
| `index_type` | 当前示例支持 `int32`。 |
| `weight_type` | 当前示例支持 `fp32`。 |
| `local_expert_masking` | 是否根据 `p_local_expert_mask` 屏蔽非本地 expert。 |

`moe_sorting_args` 继承自 `ck_tile::MoeSortingHostArgs`，关键字段如下：

| 字段 | 说明 |
| --- | --- |
| `p_topk_ids` | 输入 expert id，shape `[tokens, topk]`。 |
| `p_weights` | 输入 top-k 权重，shape `[tokens, topk]`。 |
| `p_local_expert_mask` | expert mask，shape `[num_experts]`，未启用 mask 时为 `nullptr`。 |
| `p_sorted_token_ids` | 输出排序后的 token id，长度按 padded token 数分配。 |
| `p_sorted_weights` | 输出排序后的权重。 |
| `p_sorted_expert_ids` | 输出每个 token block 对应的 expert id。 |
| `p_tokens_positions_per_expert` | 输出 expert token 位置统计，shape `[num_experts * 2]`。 |
| `p_total_tokens_post_pad` | 输出 padding 后 token 总数。 |
| `p_moe_buf` | 可选，需要顺带清零的 MoE buffer；不用时为 `nullptr`。 |
| `p_ws` | workspace pointer；`moe_sorting_get_workspace_size()` 为 0 时可为 `nullptr`。 |
| `tokens` | token 数。 |
| `unit_size` | sorted token block size，也是 padding 对齐粒度。 |
| `num_experts` | expert 数。 |
| `topk` | 每个 token 选择的 expert 数。 |
| `moe_buf_bytes` | `p_moe_buf` 字节数。 |

workspace 约束：调用前先通过 `moe_sorting_get_workspace_size(tokens, num_experts, topk)` 查询；返回非 0 时，客户端需要分配 device buffer 并清零后传给 `p_ws`。

调用示例：

```cpp
int ws_bytes = moe_sorting_get_workspace_size(tokens, num_experts, topk);
void* ws_dev = nullptr;
if(ws_bytes != 0)
{
    hipMalloc(&ws_dev, ws_bytes);
    hipMemsetAsync(ws_dev, 0, ws_bytes, stream);
}

moe_sorting_trait trait{"int32", "fp32", false};
moe_sorting_args args{
    topk_ids_dev,
    topk_weights_dev,
    nullptr,
    sorted_token_ids_dev,
    sorted_weights_dev,
    sorted_expert_ids_dev,
    tokens_positions_per_expert_dev,
    total_tokens_post_pad_dev,
    nullptr,
    ws_dev,
    tokens,
    unit_size,
    num_experts,
    topk,
    0,
};

float time_ms = moe_sorting(trait, args, ck_tile::stream_config{stream, true});
```

## 3.5 Fused MoE: `fused_moe`

来源路径：`example_hcu/ck_tile/17_fused_moe/fused_moe.hpp`、`example_hcu/ck_tile/17_fused_moe/fused_moesorting.hpp`、`example_hcu/ck_tile/17_fused_moe/fused_moegemm.hpp`。

功能：完成 MoE sorting、expert gate/up GEMM、activation、down GEMM、top-k 权重乘法和 token 输出累加。客户端可调用完整 `fused_moe()`，也可拆分调用 `fused_moesorting()` 和 `fused_moegemm()`。

主要接口：

```cpp
#include "fused_moe.hpp"

float fused_moe(fused_moe_traits traits,
                fused_moe_args args,
                const ck_tile::stream_config& config);

void fused_moe_get_solutions(fused_moe_traits traits,
                             fused_moe_args args,
                             int* sol_data,
                             int* sol_size);

int fused_moe_get_workspace_size(int tokens, int num_experts, int topk);
```

`fused_moe_traits`：

| 字段 | 说明 |
| --- | --- |
| `prec_i/prec_w/prec_o` | input、weight、output 数据类型。 |
| `prec_st/prec_sw/prec_sq/prec_kw/prec_zp` | token scale、weight scale、smooth quant scale、topk weight、zero-point 数据类型。 |
| `block_m` | sorted token block size。 |
| `activation` | `0` gelu，`1` silu。 |
| `gate_only` | `0` gate+up，`1` only gate。 |
| `fused_quant` | `0` no quant，`1` smooth dynamic quant，`2` int8_w8a16，`3` int4_w4a16，`4` int8_w8a8 block，`5` int4_w4a8 block，`10` dynamic quant。 |
| `solution_id` | kernel solution 选择，`0` 表示默认。 |
| `use_wt_shuffle` | 是否使用预 shuffle 权重布局。 |
| `local_expert_masking` | 是否启用本地 expert mask。 |

`fused_moe_args` 关键字段：

| 字段 | 说明 |
| --- | --- |
| `a_ptr/o_ptr` | 输入 token `[num_tokens, hidden_size]` 和输出 token。 |
| `g_ptr/d_ptr` | gate/up 权重和 down 权重，按 expert 组织。 |
| `a_scale_ptr/g_scale_ptr/d_scale_ptr/y_smooth_scale_ptr` | 量化 scale pointer，未启用对应量化时可为 `nullptr`。 |
| `g_zp_ptr/d_zp_ptr` | 量化 zero-point pointer。 |
| `local_expert_mask_ptr` | expert mask pointer。 |
| `ws_ptr` | MoE sorting workspace，大小由 `fused_moe_get_workspace_size()` 返回。 |
| `topk_ids_ptr/topk_weight_ptr` | 路由输入，shape `[num_tokens, topk]`。 |
| `sorted_token_ids_ptr/sorted_weight_ptr/sorted_expert_ids_ptr` | sorting 输出 buffer。 |
| `tokens_positions_per_expert_ptr/num_sorted_tiles_ptr` | sorting/GEMM 元数据输出。 |
| `block_m/hidden_size/intermediate_size/num_tokens/num_experts/topk` | MoE 维度参数。 |
| `stride_token` | 输入/输出 token row stride，需大于等于 `hidden_size`。 |
| `block_shape_n/block_shape_k` | block quant 路径使用的 N/K block size。 |

调用示例：

```cpp
int ws_bytes = fused_moe_get_workspace_size(num_tokens, num_experts, topk);
void* ws_dev = nullptr;
if(ws_bytes != 0)
{
    hipMalloc(&ws_dev, ws_bytes);
    hipMemsetAsync(ws_dev, 0, ws_bytes, stream);
}

fused_moe_traits traits{
    "bf16",
    "bf16",
    "fp32",
    "fp32",
    "fp32",
    "fp32",
    "fp32",
    "int32",
    block_m,
    1,
    0,
    0,
    0,
    false,
    false,
};

fused_moe_args args{};
args.a_ptr = input_dev;
args.g_ptr = gate_up_weight_dev;
args.d_ptr = down_weight_dev;
args.o_ptr = output_dev;
args.ws_ptr = ws_dev;
args.topk_ids_ptr = topk_ids_dev;
args.topk_weight_ptr = topk_weight_dev;
args.sorted_token_ids_ptr = sorted_token_ids_dev;
args.sorted_weight_ptr = sorted_weight_dev;
args.sorted_expert_ids_ptr = sorted_expert_ids_dev;
args.tokens_positions_per_expert_ptr = tokens_positions_per_expert_dev;
args.num_sorted_tiles_ptr = num_sorted_tiles_dev;
args.block_m = block_m;
args.hidden_size = hidden_size;
args.intermediate_size = intermediate_size;
args.num_tokens = num_tokens;
args.num_experts = num_experts;
args.topk = topk;
args.stride_token = hidden_size;
args.block_shape_n = 0;
args.block_shape_k = 0;

float time_ms = fused_moe(traits, args, ck_tile::stream_config{stream, true});
```

## 3.6 MoE Quant: `moe_quant`

来源路径：`example_hcu/ck_tile/18_moe_quant/moe_quant.hpp`、`include/ck_tile/ops/moe_quant/kernel/moe_quant_kernel.hpp`。

功能：对 MoE token 输入做 rowwise dynamic quantization，输出量化后的 token tensor 和每个 token 的 scale。

接口原型：

```cpp
#include "moe_quant.hpp"

struct moe_quant_args : public ck_tile::MoequantHostArgs
{
};

struct moe_quant_traits
{
    std::string in_type;
    std::string out_type;
};

float moe_quant(moe_quant_traits traits,
                moe_quant_args args,
                const ck_tile::stream_config& config);
```

`moe_quant_args` 字段：

| 字段 | 说明 |
| --- | --- |
| `p_x` | 输入 `[tokens, hidden_size]`，fp16/bf16。 |
| `p_yscale` | 输出 `[tokens, 1]` rowwise scale。 |
| `p_qy` | 输出 `[tokens, hidden_size]` 量化结果。 |
| `tokens` | token 数。 |
| `hidden_size` | hidden size。 |
| `x_stride` | 输入 row stride。 |
| `y_stride` | 输出 row stride。 |

调用示例：

```cpp
moe_quant_traits traits{"fp16", "int8"};
moe_quant_args args{x_dev, y_scale_dev, qy_dev, tokens, hidden_size, x_stride, y_stride};

float time_ms = moe_quant(traits, args, ck_tile::stream_config{stream, true});
```

## 3.7 LayerNorm2D: `layernorm2d_fwd`

来源路径：`example_hcu/ck_tile/02_layernorm2d/layernorm2d_fwd.hpp`、`include/ck_tile/ops/layernorm2d/kernel/layernorm2d_fwd_kernel.hpp`。

功能：执行二维 LayerNorm forward，并可融合 input bias、residual add、保存 mean/invStd、rowwise dynamic quantization 或 smooth dynamic quantization。

接口原型：

```cpp
#include "layernorm2d_fwd.hpp"

struct layernorm2d_fwd_args : public ck_tile::Layernorm2dFwdHostArgs
{
};

struct layernorm2d_fwd_traits
{
    std::string prec_i;
    std::string prec_o;
    std::string prec_sm;
    std::string prec_sy;
    bool save_mean_var;
    int xbias;
    int fused_add;
    int fused_quant;
};

float layernorm2d_fwd(layernorm2d_fwd_traits traits,
                      layernorm2d_fwd_args args,
                      const ck_tile::stream_config& config);
```

`layernorm2d_fwd_args` 字段：

| 字段 | 说明 |
| --- | --- |
| `p_x` | 输入 `[m, n]`，fp16/bf16。 |
| `p_x_residual` | residual 输入，未启用 fused add 时为 `nullptr`。 |
| `p_sm_scale` | smooth quant scale，未启用时为 `nullptr`。 |
| `p_x_bias` | 输入 bias，shape `[1, n]`，`xbias=0` 时可为 `nullptr`。 |
| `p_gamma` | gamma，shape `[1, n]`。 |
| `p_beta` | beta，shape `[1, n]`。 |
| `p_y` | 输出 `[m, n]`。 |
| `p_y_residual` | residual 输出，`fused_add=1` 时使用。 |
| `p_y_scale` | rowwise quant scale 输出，未启用 quant 时为 `nullptr`。 |
| `p_mean` | mean 输出，`save_mean_var=false` 时为 `nullptr`。 |
| `p_invStd` | inv-std 输出，`save_mean_var=false` 时为 `nullptr`。 |
| `epsilon` | LayerNorm epsilon。 |
| `m/n` | 输入二维形状。 |
| `x_stride/xr_stride/y_stride/yr_stride` | 各 tensor row stride。 |

`layernorm2d_fwd_traits` 说明：

| 字段 | 说明 |
| --- | --- |
| `prec_i/prec_o` | 输入/输出精度，如 `fp16/bf16/int8`。 |
| `prec_sm/prec_sy` | smooth scale 和输出 scale 精度，默认常用 `fp32`。 |
| `save_mean_var` | 是否保存 mean 和 invStd，训练场景可启用。 |
| `xbias` | `0` 不加 bias，`1` 在 fused add 前加 bias。 |
| `fused_add` | `0` no add，`1` pre-add 并保存 residual，`2` pre-add only。 |
| `fused_quant` | `0` no quant，`1` smooth dynamic quant，`2` dynamic quant。 |

调用示例：

```cpp
layernorm2d_fwd_traits traits{
    "fp16",
    "fp16",
    "fp32",
    "fp32",
    false,
    0,
    0,
    0,
};

layernorm2d_fwd_args args{};
args.p_x = x_dev;
args.p_x_residual = nullptr;
args.p_sm_scale = nullptr;
args.p_x_bias = nullptr;
args.p_gamma = gamma_dev;
args.p_beta = beta_dev;
args.p_y = y_dev;
args.p_y_residual = nullptr;
args.p_y_scale = nullptr;
args.p_mean = nullptr;
args.p_invStd = nullptr;
args.epsilon = 1.0e-5f;
args.m = m;
args.n = n;
args.x_stride = n;
args.xr_stride = n;
args.y_stride = n;
args.yr_stride = n;

float time_ms = layernorm2d_fwd(traits, args, ck_tile::stream_config{stream, true});
```

## 3.8 RMSNorm2D: `rmsnorm2d_fwd`

来源路径：`example_hcu/ck_tile/10_rmsnorm2d/rmsnorm2d_fwd.hpp`、`include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_kernel.hpp`。

功能：执行二维 RMSNorm forward，并可融合 residual add、保存 inv-rms、保存未量化输出、rowwise dynamic quantization 或 smooth dynamic quantization。

接口原型：

```cpp
#include "rmsnorm2d_fwd.hpp"

struct rmsnorm2d_fwd_args : public ck_tile::Rmsnorm2dFwdHostArgs
{
};

struct rmsnorm2d_fwd_traits
{
    std::string prec_i;
    std::string prec_o;
    std::string prec_sm;
    std::string prec_sy;
    bool save_rms;
    bool save_unquant;
    int fused_add;
    int fused_quant;
};

float rmsnorm2d_fwd(rmsnorm2d_fwd_traits traits,
                    rmsnorm2d_fwd_args args,
                    const ck_tile::stream_config& config);
```

`rmsnorm2d_fwd_args` 字段：

| 字段 | 说明 |
| --- | --- |
| `p_x` | 输入 `[m, n]`。 |
| `p_x_residual` | residual 输入，未启用 fused add 时为 `nullptr`。 |
| `p_sm_scale` | smooth quant scale，未启用时为 `nullptr`。 |
| `p_gamma` | RMSNorm gamma，shape `[1, n]`。 |
| `p_y` | 输出 `[m, n]`。 |
| `p_y_residual` | residual 输出，未启用时为 `nullptr`。 |
| `p_y_scale` | rowwise quant scale 输出，未启用时为 `nullptr`。 |
| `p_invRms` | inv-rms 输出，`save_rms=false` 时为 `nullptr`。 |
| `p_y_unquant` | 量化前输出，`save_unquant=false` 时为 `nullptr`。 |
| `epsilon` | RMSNorm epsilon。 |
| `m/n` | 输入二维形状。 |
| `x_stride/xr_stride/y_stride/yr_stride` | 各 tensor row stride。 |

调用示例：

```cpp
rmsnorm2d_fwd_traits traits{
    "fp16",
    "fp16",
    "fp32",
    "fp32",
    false,
    false,
    0,
    0,
};

rmsnorm2d_fwd_args args{};
args.p_x = x_dev;
args.p_gamma = gamma_dev;
args.p_y = y_dev;
args.epsilon = 1.0e-5f;
args.m = m;
args.n = n;
args.x_stride = n;
args.y_stride = n;

float time_ms = rmsnorm2d_fwd(traits, args, ck_tile::stream_config{stream, true});
```

## 3.9 Add + RMSNorm2D + RdQuant: `add_rmsnorm2d_rdquant_fwd`

来源路径：`example_hcu/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`、`include/ck_tile/ops/add_rmsnorm2d_rdquant/kernel/add_rmsnorm2d_rdquant_fwd_kernel.hpp`。

功能：融合 `x = a + b`、RMSNorm2D 和 rowwise dynamic quantization，输出可选中间 `x`、rowwise scale 和量化结果。

接口原型：

```cpp
#include "add_rmsnorm2d_rdquant_fwd.hpp"

struct add_rmsnorm2d_rdquant_fwd_args
    : public ck_tile::AddRmsnorm2dRdquantFwdHostArgs
{
};

struct add_rmsnorm2d_rdquant_fwd_traits
{
    std::string input_data_type;
    std::string quantized_data_type;
    bool save_x;
};

float add_rmsnorm2d_rdquant_fwd(add_rmsnorm2d_rdquant_fwd_traits traits,
                                add_rmsnorm2d_rdquant_fwd_args args,
                                const ck_tile::stream_config& config);
```

参数说明：

| 字段 | 说明 |
| --- | --- |
| `p_a/p_b` | 输入 `[m, n]`，fp16/bf16。 |
| `p_gamma` | RMSNorm gamma，shape `[1, n]`。 |
| `p_x` | 输出 `p_a + p_b`，`save_x=false` 时可为空。 |
| `p_yscale` | rowwise quant scale 输出 `[m, 1]`。 |
| `p_qy` | 量化输出 `[m, n]`，int8/fp8。 |
| `epsilon` | RMSNorm epsilon。 |
| `m/n/stride` | 输入输出形状和 row stride。 |

调用示例：

```cpp
add_rmsnorm2d_rdquant_fwd_traits traits{"fp16", "int8", true};

add_rmsnorm2d_rdquant_fwd_args args{
    a_dev,
    b_dev,
    gamma_dev,
    x_dev,
    y_scale_dev,
    qy_dev,
    1.0e-5f,
    m,
    n,
    n,
};

float time_ms =
    add_rmsnorm2d_rdquant_fwd(traits, args, ck_tile::stream_config{stream, true});
```

## 3.10 CK Tile Conv Fwd: `ConvIgemmFwdKernel`

来源路径：`example_hcu/ck_tile/14_conv/conv_fwd.cpp`。

功能：使用 CK Tile IGEMM pipeline 执行 FP16 forward convolution。当前示例固定：

- A/B/C 数据类型：`fp16/fp16/fp16`，accumulator 为 `float`。
- 布局：输入 `NHWGC`，权重 `GKYXC`，输出 `NHWGK`。
- 默认 2D convolution，默认 shape 为 `G=1, N=64, K=64, C=64, Y=X=3, Hi=Wi=4, pad=1`。

核心调用模式：

```cpp
using Kernel = ck_tile::ConvIgemmFwdKernel<TilePartitioner, Pipeline, Epilogue, ConvSpec>;

auto host_args = Kernel::MakeHostArgs(in_dev,
                                      wei_dev,
                                      out_dev,
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      c_g_n_k_wos_lengths,
                                      c_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads);

auto kernel_args = Kernel::MakeKernelArgs(host_args);

float time_ms = ck_tile::launch_kernel(
    ck_tile::stream_config{stream, true},
    ck_tile::make_kernel<BlockSize, MinBlockPerCU>(
        Kernel{}, kernel_args.tile_partitioner.CalculateGridSize(), BlockSize, 0, kernel_args));
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `in_dev/wei_dev/out_dev` | 输入、权重、输出 device pointer。 |
| `a_g_n_c_wis_lengths/strides` | 输入 tensor 的 `G,N,C,spatial...` 维度和 stride。 |
| `b_g_k_c_xs_lengths/strides` | 权重 tensor 的 `G,K,C,filter_spatial...` 维度和 stride。 |
| `c_g_n_k_wos_lengths/strides` | 输出 tensor 的 `G,N,K,output_spatial...` 维度和 stride。 |
| `conv_filter_strides` | 卷积 stride。 |
| `conv_filter_dilations` | 卷积 dilation。 |
| `input_left_pads/input_right_pads` | 输入 padding。 |

校验说明：该示例为 FP16 输出，GPU kernel 与 host reference 的舍入顺序不同，可能出现 1-2 ULP 量级误差。默认校验容差使用 `rtol=1e-3, atol=1e-3`；若需要更严格或更宽松的验证，可在示例命令中传入 `-rtol` 和 `-atol`。

## 3.11 CK Tile Conv3D Fwd: `Conv3dFwdWaspKernelV1`

来源路径：`example_hcu/ck_tile/16_conv3d_fwd/conv3d_fwd.cpp`。

功能：使用 CK Tile WASP IGEMM pipeline 执行 BF16 3D forward convolution。当前示例固定：

- A/B/C 数据类型：`bf16/bf16/bf16`，accumulator 为 `float`。
- 布局：输入 `NDHWGC`，权重 `GKZYXC`，输出 `NDHWGK`。
- 卷积规格：`Filter1x1x1Stride1Pad0`，默认 shape 为 `G=1, N=16, K=256, C=256, Z=Y=X=1, Di=Hi=Wi=14`。

核心调用模式：

```cpp
using Kernel = ck_tile::Conv3dFwdWaspKernelV1<TilePartitioner, Pipeline, Epilogue, ConvSpec>;

auto host_args = Kernel::MakeHostArgs(in_dev,
                                      wei_dev,
                                      out_dev,
                                      in_g_n_c_wis_lengths,
                                      in_g_n_c_wis_strides,
                                      wei_g_k_c_xs_lengths,
                                      wei_g_k_c_xs_strides,
                                      out_g_n_k_wos_lengths,
                                      out_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads);

auto kernel_args = Kernel::MakeKernelArgs(host_args);

if(!Kernel::IsSupportedArgument(host_args, kernel_args))
{
    throw std::runtime_error("conv3d argument is not supported");
}

float time_ms = ck_tile::launch_kernel(
    ck_tile::stream_config{stream, true},
    ck_tile::make_kernel<TileShape::BlockSize, MinBlockPerCU>(
        Kernel{},
        conv_param.G_ * kernel_args.tile_partitioner.CalculateGridSize(),
        TileShape::BlockSize,
        0,
        kernel_args,
        ck_tile::bool_constant<true>{}));
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `in_dev/wei_dev/out_dev` | 输入、权重、输出 device pointer。 |
| `in_g_n_c_wis_lengths/strides` | 输入 tensor 的 `G,N,C,D,H,W` 维度和 stride。 |
| `wei_g_k_c_xs_lengths/strides` | 权重 tensor 的 `G,K,C,Z,Y,X` 维度和 stride。 |
| `out_g_n_k_wos_lengths/strides` | 输出 tensor 的 `G,N,K,Do,Ho,Wo` 维度和 stride。 |
| `conv_filter_strides` | 3D 卷积 stride，顺序为 `D,H,W`。 |
| `conv_filter_dilations` | 3D 卷积 dilation，顺序为 `D,H,W`。 |
| `input_left_pads/input_right_pads` | 3D 输入 padding，顺序为 `D,H,W`。 |

返回值说明：`run_conv3d_fwd` 返回 `ck_tile::check_err` 的布尔结果；示例 `main` 将 `true` 映射为进程返回码 `0`，将 `false` 映射为 `1`。

## 3.12 legacy CK DeviceOp 调用模式

以下 `example_hcu` 示例没有额外 wrapper 函数，而是直接实例化 CK device operator。客户端若要复用这些算子，通常有两种方式：直接在业务代码中复用示例里的 `Device*Instance` typedef，或把该 typedef 包装成自己的稳定 C/C++ ABI。

覆盖示例：

| 算子 | 示例路径 | 典型 DeviceOp |
| --- | --- | --- |
| GEMM MMAC | `example_hcu/05_gemm` | `DeviceGemm...` MMAC instance。 |
| GEMM XDL v3 | `example_hcu/25_gemm_xdl_v3` | `DeviceGemm_Xdl_CShuffleV3`、`DeviceGemmMultiD_Xdl_CShuffle_V3`。 |
| Batched GEMM XDL v3 | `example_hcu/24_batched_gemm` | `DeviceBatchedGemmMultiD_Xdl_CShuffle_V3`。 |
| Grouped Conv Bwd Data | `example_hcu/02_grouped_conv_bwd_data_mmac` | grouped convolution backward-data device op。 |
| Tensor Layout Transform | `example_hcu/04_tensor_layout_transform` | `DeviceTensorLayoutTransform...`。 |
| BatchNorm Forward/Backward | `example_hcu/08_batchnorm` | `DeviceBatchNormForward...`、`DeviceBatchNormBackward...`。 |

代表性 `MakeArgument` 参数形状：

GEMM MMAC：

```cpp
auto argument = gemm.MakeArgument(a_dev,
                                  b_dev,
                                  c_dev,
                                  M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  StrideC,
                                  AElementOp{},
                                  BElementOp{},
                                  CElementOp{});
```

GEMM XDL v3：

```cpp
auto argument = gemm.MakeArgument(a_dev,
                                  b_dev,
                                  c_dev,
                                  M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  StrideC,
                                  KBatch,
                                  AElementOp{},
                                  BElementOp{},
                                  CElementOp{});
```

GEMM Multi-D / Batched GEMM XDL v3：

```cpp
auto argument = gemm.MakeArgument(a_dev,
                                  b_dev,
                                  d_ptrs,
                                  e_dev,
                                  M,
                                  N,
                                  K,
                                  batch_count,
                                  stride_A,
                                  stride_B,
                                  stride_Ds,
                                  stride_E,
                                  batch_stride_A,
                                  batch_stride_B,
                                  batch_stride_Ds,
                                  batch_stride_E,
                                  AElementOp{},
                                  BElementOp{},
                                  CDEElementOp{});
```

其中普通 multi-D GEMM 没有 `batch_count/batch_stride_*` 参数；示例中 `d_ptrs` 和 `stride_Ds` 可为空 `{}`，表示只输出 E/C。

Grouped Conv Bwd Data：

```cpp
auto argument = conv.MakeArgument(out_dev,
                                  wei_dev,
                                  in_dev,
                                  out_lengths,
                                  out_strides,
                                  wei_lengths,
                                  wei_strides,
                                  in_lengths,
                                  in_strides,
                                  conv_filter_strides,
                                  conv_filter_dilations,
                                  input_left_pads,
                                  input_right_pads,
                                  out_element_op,
                                  wei_element_op,
                                  in_element_op);
```

Tensor Layout Transform：

```cpp
auto argument_ptr = trans.MakeArgumentPointer(src_lengths,
                                              dst_lengths,
                                              src_dev,
                                              dst_dev);
```

BatchNorm Forward Training：

```cpp
auto argument_ptr = batchnorm_fwd.MakeArgumentPointer(
    inout_lengths,
    x_strides,
    y_strides,
    reduce_dims,
    scale_bias_mean_var_lengths,
    scale_strides,
    bias_strides,
    mean_var_strides,
    x_dev,
    scale_dev,
    bias_dev,
    epsilon,
    PassThroughOp{},
    y_dev,
    save_mean_dev,
    save_inv_var_dev,
    average_factor,
    running_mean_dev,
    running_var_dev);
```

BatchNorm Backward：

```cpp
auto argument_ptr = batchnorm_bwd.MakeArgumentPointer(
    inout_lengths,
    x_strides,
    dy_strides,
    dx_strides,
    reduce_dims,
    scale_bias_mean_var_lengths,
    scale_strides,
    saved_mean_strides,
    saved_inv_var_strides,
    x_dev,
    dy_dev,
    scale_dev,
    saved_mean_dev,
    saved_inv_var_dev,
    epsilon,
    PassThroughOp{},
    dx_dev,
    dscale_dev,
    dbias_dev);
```

BatchNorm 这类接口还需要按返回的 workspace size 分配 device workspace，并通过 `SetWorkSpacePointer()` 写回 argument 后再调用 invoker。

通用调用骨架：

```cpp
using DeviceOp = DeviceOpInstanceFromExample;

DeviceOp op;

auto argument = op.MakeArgument(/* device pointers, lengths, strides, attributes */);

if(!op.IsSupportedArgument(argument))
{
    // shape/layout/dtype 不被当前实例支持，客户端应选择其他实例或返回错误。
    return;
}

auto invoker = op.MakeInvoker();
float time_ms = invoker.Run(argument,
                            StreamConfig{stream, /* time_kernel = */ true});
```

对于 `MakeArgumentPointer()` / `MakeInvokerPointer()` 风格的示例，调用模式等价：

```cpp
auto argument_ptr = op.MakeArgumentPointer(/* args */);
if(!op.IsSupportedArgument(argument_ptr.get()))
{
    return;
}

auto invoker_ptr = op.MakeInvokerPointer();
float time_ms = invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, true});
```

集成建议：

- 客户端应把示例中可用的 `DeviceOp` typedef 固化到自己的 wrapper 中，避免在业务层直接暴露长模板参数。
- `IsSupportedArgument` 是必要的参数合法性检查，不能省略。
- 同一个算子族通常需要多份实例覆盖不同 dtype、layout、shape 或 pipeline，业务层应在 wrapper 中做实例选择。

# 4. example示例说明

## 4.1 GEMM MMAC

### 功能描述

GEMM MMAC 示例使用 legacy CK device GEMM 路径，通过 HCU MMAC 完成矩阵乘：

```text
C = A * B
```

示例入口为 `example_gemm_mmac_nt_fp16`，默认形状为 `M=128, N=512, K=256`。

### 参数说明

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `arg1` | int | 是否校验，`0` 不校验，`1` 校验。 |
| `arg2` | int | 初始化方式，`0` 不初始化，`1` 整数随机，`2` 小数随机。 |
| `arg3` | int | 是否计时 kernel，`0` 不计时，`1` 计时。 |
| `arg4` | int | `M`。 |
| `arg5` | int | `N`。 |
| `arg6` | int | `K`。 |
| `arg7` | int | `StrideA`。 |
| `arg8` | int | `StrideB`。 |
| `arg9` | int | `StrideC`。 |

### 调用示例

```bash
./example_gemm_mmac_nt_fp16
./example_gemm_mmac_nt_fp16 1 2 1 128 512 256 128 512 512
```

## 4.2 GEMM XDL v3

### 功能描述

GEMM XDL v3 示例用于验证 upstream CK 移植过来的 XDL CShuffle v3 device 路径：

- `example_hcu_gemm_xdl_bf16_v3`：`DeviceGemm_Xdl_CShuffleV3`
- `example_hcu_gemm_multi_d_xdl_bf16_v3`：`DeviceGemmMultiD_Xdl_CShuffle_V3`

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `arg1` | 是否校验结果，`0` 否，`1` 是。 |
| `arg2` | 初始化方式，`0` 不初始化，`1` 整数随机值，`2` 小数随机值。 |
| `arg3` | 是否计时 kernel，`0` 否，`1` 是。 |
| `arg4` 到 `arg10` | `M, N, K, StrideA, StrideB, StrideC/StrideE, KBatch`。 |

### 调用示例

```bash
./example_hcu_gemm_xdl_bf16_v3
./example_hcu_gemm_xdl_bf16_v3 1 1 0 256 256 128 128 128 256 1

./example_hcu_gemm_multi_d_xdl_bf16_v3
./example_hcu_gemm_multi_d_xdl_bf16_v3 1 1 0 3840 4096 4096 4096 4096 4096 1
```

### 约束与限制

v3 pipeline 对 K 循环深度有要求。若 `K` 太小，`IsSupportedArgument` 或 gridwise 合法性检查可能拒绝参数。可设置 `CK_LOGGING=1` 查看拒绝原因。

## 4.3 Batched GEMM XDL v3

### 功能描述

Batched GEMM XDL v3 示例使用 `DeviceBatchedGemmMultiD_Xdl_CShuffle_V3`，执行 BF16 batched GEMM：

```text
E[b] = A[b] * B[b]
```

默认随机生成 `M/N/K`，`batch_count=2`。

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `arg1` | 是否校验结果，`0` 否，`1` 是。 |
| `arg2` | 初始化方式，`0` 不初始化，`1` 整数随机值，`2` 小数随机值。 |
| `arg3` | 是否计时 kernel，`0` 否，`1` 是。 |
| `arg4` 到 `arg7` | 可选，`M, N, K, Batch`。 |

### 调用示例

```bash
./example_hcu_batched_gemm_xdl_bf16_v3
./example_hcu_batched_gemm_xdl_bf16_v3 1 1 1 256 256 128 2
```

## 4.4 CK Tile GEMM

### 功能描述

CK Tile GEMM 示例使用 `ck_tile` tile-programming API 实现 GEMM，目前示例主入口为 basic pipeline。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-b` | `1` | batch size。 |
| `-m` | `1024` | M 维度。 |
| `-n` | `2048` | N 维度。 |
| `-k` | `64` | K 维度。 |
| `-stride_a` | `0` | A stride，`0` 时由示例推导。 |
| `-stride_b` | `0` | B stride。 |
| `-stride_c` | `0` | C stride。 |
| `-v` | `2` | 校验方式，`0` 不校验，`1` CPU 校验，`2` GPU 校验。 |
| `-e` | `1e-5` | 绝对误差阈值。 |
| `-prec` | `fp16` | 数据类型，支持 `fp16/bf16/fp8/bf8`。 |
| `-warmup` | `10` | 预热次数。 |
| `-repeat` | `100` | 计时重复次数。 |
| `-timer` | `gpu` | 计时器类型，`gpu/cpu`。 |

### 调用示例

```bash
./tile_example_gemm_basic -m=1024 -n=2048 -k=64 -prec=fp16
./tile_example_gemm_basic -b=4 -m=512 -n=512 -k=128 -v=1
```

## 4.5 Grouped GEMM

### 功能描述

Grouped GEMM 用一组 GEMM descriptor 描述多个不同 shape 的 GEMM，在一次 grouped 调度中执行。示例支持普通 GEMM、bias epilogue、multiple-D epilogue、SplitK 和多精度实例。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-Ms` | 空 | 每组 GEMM 的 M 列表。 |
| `-Ns` | 空 | 每组 GEMM 的 N 列表。 |
| `-Ks` | 空 | 每组 GEMM 的 K 列表。 |
| `-stride_As` | 空 | 每组 A stride。 |
| `-stride_Bs` | 空 | 每组 B stride。 |
| `-stride_Cs` | 空 | 每组 C stride。 |
| `-a_layout` | `R` | A 布局，`R` row-major，`C` column-major。 |
| `-b_layout` | `C` | B 布局。 |
| `-c_layout` | `R` | C 布局。 |
| `-validate` | `1` | CPU 校验开关。 |
| `-prec` | 由 fast target 决定 | `fp16/bf16/fp8/bf8/int8/int4`。 |
| `-config` | 空 | 可选配置选择，如 `int8_32x32`、`bf8_128x64`。 |
| `-group_count` | `8` | group 数量。 |
| `-kbatch` | `1` | SplitK kbatch。 |
| `-bias` | `0` | `1` 表示 `C=A*B+D`。 |
| `-multiple_d` | `0` | `1` 表示带两个 D tensor。 |
| `-multiple_d_op` | `add` | `add` 或 `multiply`。 |
| `-warmup` | `10` | 预热次数。 |
| `-repeat` | `100` | 重复次数。 |
| `-json` | `0` | 是否输出 JSON。 |

### 调用示例

```bash
./tile_example_grouped_gemm -prec=fp16 -group_count=8 -validate=1
./tile_example_grouped_gemm_fp16_fast -Ms=256,512 -Ns=128,256 -Ks=64,64 -validate=1
./tile_example_grouped_gemm_int4_fast -prec=int4 -group_count=8 -int4_const=1
```

### 约束与限制

`int4` 相关参数包含调试初始化项，如 `-int4_const`、`-int4_const_a`、`-int4_const_b`、`-int4_b_one_k`，主要用于验证 packed int4 路径。

## 4.6 FMHA

### 功能描述

FMHA 示例实现 fused multi-head attention，包含 forward 和 backward。Forward 支持 batch/group 模式、MQA/GQA、mask、bias、alibi、V layout、LSE、padding/variable length 等特性。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-v` | `1` | 是否 CPU 校验。 |
| `-mode` | `0` | `0` batch，`1` group/varlen。 |
| `-b` | `2` | batch size。 |
| `-h` | `8` | Q head 数。 |
| `-h_k` | `-1` | K/V head 数，`-1` 表示等于 `-h`。 |
| `-s` | `3328` | Q 序列长度。group 模式可传逗号分隔列表。 |
| `-s_k` | `-1` | K/V 序列长度，`-1` 表示等于 `-s`。 |
| `-d` | `128` | Q/K head dim。 |
| `-d_v` | `-1` | V head dim，`-1` 表示等于 `-d`。 |
| `-scale_s` | `0` | attention scale，`0` 表示 `1/sqrt(d)`。 |
| `-iperm` | `1` | 输入是否按 `b*h*s*d` 组织。 |
| `-operm` | `1` | 输出是否 permute。 |
| `-bias` | `n` | bias 类型，支持 none、elementwise、alibi。 |
| `-prec` | `fp16` | `fp32/fp16/bf16/fp8/fp8bf16/fp8fp32/mxfp8/mxfp4`。 |
| `-mask` | `0` | no mask、top-left causal、bottom-right causal、sliding window 等。 |
| `-vlayout` | `r` | V 矩阵 row-major 或 col-major。 |
| `-lse` | `0` | 是否保存 log-sum-exp。 |
| `-num_splits` | `1` | split-K/V 数量，`0` 使用启发式。 |
| `-warmup` | `5` | 预热次数。 |
| `-repeat` | `20` | 重复次数。 |

### 调用示例

```bash
./tile_example_fmha_fwd -b=1 -h=16 -s=16384 -d=128
./tile_example_fmha_fwd -mode=1 -b=2 -h=8 -s=1024,2048 -s_k=1024,2048 -d=128
./tile_example_fmha_bwd -b=1 -h=8 -s=2048 -d=128
```

## 4.7 MoE Sorting

### 功能描述

MoE Sorting 是 Fused MoE 前处理算子，用于把 `tokens * topk` 的专家路由信息重排为按 expert 聚合的 token 列表，为后续 grouped/fused MoE GEMM 提供输入。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-v` | `1` | 是否 CPU 校验。 |
| `-pr_i` | `int32` | index 数据类型。 |
| `-pr_w` | `fp32` | 输出 weight 数据类型。 |
| `-t` | `32` | token 数。 |
| `-e` | `8` | expert 数。 |
| `-k` | `2` | top-k。 |
| `-st_i` | `-1` | input row stride，`-1` 表示按 expert 数推导。 |
| `-seed` | `-1` | 随机种子。 |
| `-kname` | `0` | 是否打印 kernel 名称。 |

### 调用示例

```bash
./tile_example_moe_sorting -t=32 -e=8 -k=2
./tile_example_moe_sorting -t=1024 -e=64 -k=8 -v=1
```

## 4.8 Fused MoE

### 功能描述

Fused MoE 示例将 MoE sorting、两阶段 expert GEMM、activation、top-k 权重乘法和输出累加融合到更少 kernel 中，目标是减少 workspace 和 kernel 启动开销。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-t` | `32` | 输入 token 数。 |
| `-e` | `32` | expert 数。 |
| `-k` | `5` | top-k。 |
| `-h` | `7168` | hidden size。 |
| `-i` | `8192` | FFN intermediate size。 |
| `-stride` | `-1` | 输入 row stride，`-1` 表示等于 hidden size。 |
| `-bm` | `16` | sorted token block size。 |
| `-tp` | `1` | tensor parallel size。 |
| `-v` | `1` | CPU 校验开关。 |
| `-kname` | `1` | 是否打印 kernel 名。 |
| `-prec_i` | `fp16` | input precision。 |
| `-prec_w` | `fp16` | weight precision。 |
| `-prec_o` | `fp32` | output precision。 |
| `-fquant` | `0` | `0` no quant，`1` smooth dynamic quant，`2` dynamic quant。 |
| `-gate_only` | `0` | `0` gate+up，`1` only gate。 |
| `-api` | `0` | `0` fused-moe，`1` moe-gemm。 |
| `-act` | `1` | `0` gelu，`1` silu。 |
| `-balance` | `0` | 是否让 topk ids 更均衡，便于测试。 |
| `-init` | `1` | 初始化方式。 |
| `-seed` | `11939` | 随机种子。 |
| `-warmup` | `5` | 预热次数。 |
| `-repeat` | `20` | 重复次数。 |

### 调用示例

```bash
./tile_example_fused_moe -t=32 -e=32 -k=5 -h=7168 -i=8192 -prec_i=bf16 -prec_w=bf16
./tile_example_fused_moe -api=1 -t=64 -e=32 -k=2 -h=4096 -i=11008
```

### 约束与限制

当前 README 提醒：gate+up 的 fp16 场景容易出现 accumulator overflow 导致 INF，建议 gate+up 使用 BF16。

## 4.9 MoE Quant

### 功能描述

MoE Quant 示例对 MoE 输入按 token/hidden 维度进行量化，当前示例路径主要验证 `fp16/bf16` 输入到 `int8` 输出。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-t` | `128` | token 数。 |
| `-h` | `8192` | hidden size。 |
| `-v` | `1` | CPU 校验开关。 |
| `-kname` | `1` | 是否打印 kernel 名。 |
| `-prec_i` | `fp16` | 输入精度，`fp16/bf16`。 |
| `-prec_o` | `int8` | 输出精度，示例支持 `int8`。 |
| `-warmup` | `5` | 预热次数。 |
| `-repeat` | `20` | 重复次数。 |

### 调用示例

```bash
./tile_example_moe_quant -prec_i=fp16 -prec_o=int8 -t=17 -h=16
./tile_example_moe_quant -prec_i=bf16 -prec_o=int8 -t=128 -h=8192
```

## 4.10 Grouped Conv Bwd Data

### 功能描述

Grouped Conv Bwd Data 示例验证 grouped convolution backward-data 的 MMAC HCU 路径。已验证代表入口为：

```text
example_grouped_conv_bwd_data_mmac_nhwc_v2_fp16_cshuffle
```

### 参数说明

该类示例使用传统 CK convolution 参数格式：

| 参数 | 说明 |
| --- | --- |
| `arg1` | 是否校验，`0` 否，`1` 是。 |
| `arg2` | 初始化方式，`0` 不初始化，`1` 整数随机，`2` 小数随机。 |
| `arg3` | 是否计时 kernel，`0` 否，`1` 是。 |
| `arg4` | spatial 维数，示例为 `2`。 |
| 后续参数 | `G, N, K, C, filter spatial, input/output spatial, stride, dilation, left padding, right padding`。 |

### 调用示例

```bash
./example_grouped_conv_bwd_data_mmac_nhwc_v2_fp16_cshuffle
```

## 4.11 Tensor Layout Transform

### 功能描述

Tensor Layout Transform 示例执行卷积张量布局转换，已验证入口为：

```text
NGCHW fp16 -> NGCHWc32 fp16
```

对应 target：

```text
example_tensor_layout_transform_ngchw_ngchwc32_fp16
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `arg1` | 是否校验，`0` 否，`1` 是。 |
| `arg2` | 初始化方式，`0` 不初始化，`1` 整数随机，`2` 小数随机。 |
| `arg3` | 是否计时 kernel，`0` 否，`1` 是。 |
| 后续参数 | 可选 convolution 参数，格式同 CK convolution parser。 |

### 调用示例

```bash
./example_tensor_layout_transform_ngchw_ngchwc32_fp16
./example_tensor_layout_transform_ngchw_ngchwc32_fp16 1 2 1
```

## 4.12 BatchNorm Forward/Backward

### 功能描述

BatchNorm 示例包含 NCHW forward training 和 backward，用于验证 legacy CK BatchNorm device operator。

### 参数说明

Forward training：

| 参数 | 说明 |
| --- | --- |
| `-D` 或 `--inOutLengths` | 输入输出形状，NCHW 4 维逗号分隔。 |
| `-v` 或 `--verify` | 是否与 host reference 比较。 |
| `Arg1` | data type，`0` 表示 fp16。 |
| `Arg2` | 是否更新 running mean/variance。 |
| `Arg3` | 是否保存 mean/invVariance。 |
| `Arg4` | 初始化方式。 |
| `Arg5` | 是否计时 kernel。 |
| `Arg6` | 是否使用 multi-block welford。 |

Backward：

| 参数 | 说明 |
| --- | --- |
| `-D` 或 `--inOutLengths` | 输入输出形状，NCHW 4 维逗号分隔。 |
| `-v` 或 `--verify` | 是否与 host reference 比较。 |
| `Arg1` | data type，`0` 表示 fp16。 |
| `Arg2` | 是否使用 saved mean/invVariance。 |
| `Arg3` | `dy` 和 `bnScale` 初始化方式。 |
| `Arg4` | 是否计时 kernel。 |
| `Arg5` | 是否使用 multi-block welford，当前 single-block 不支持。 |

### 调用示例

```bash
./example_batchnorm_forward_training -D 128,16,3,1024 -v 1 0 1 1 3 0 1
./example_batchnorm_backward -D 128,16,3,1024 -v 1 0 1 3 0 1
./example_batchnorm_backward -D 128,16,3,1024 -v 1 0 0 3 0 1
```

## 4.13 LayerNorm2D

### 功能描述

LayerNorm2D 示例实现二维 LayerNorm forward，输入形状按 `m x n` 组织。默认 FP16 路径会调用生成的 `layernorm2d_fwd_fp16_...` kernel。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-m` | `3328` | M 维度。 |
| `-n` | `4096` | N 维度。 |
| `-x_stride` | `-1` | x row stride，`-1` 表示等于 `n`。 |
| `-xr_stride` | `-1` | residual x row stride，`-1` 表示等于 `n`。 |
| `-y_stride` | `-1` | y row stride，`-1` 表示等于 `n`。 |
| `-yr_stride` | `-1` | residual y row stride，`-1` 表示等于 `n`。 |
| `-e` | `1e-5` | epsilon。 |
| `-save_mv` | `0` | 是否保存 mean/invStd，训练场景可设为 `1`。 |
| `-v` | `1` | CPU 校验开关。 |
| `-kname` | `1` | 是否打印 kernel 名称。 |
| `-prec_i` | `fp16` | 输入精度。 |
| `-prec_o` | `auto` | 输出精度，`auto` 表示与输入一致。 |
| `-prec_sm` | `auto` | smooth quant scale 精度，`auto` 表示 `fp32`。 |
| `-prec_sy` | `auto` | 输出 scale 精度，`auto` 表示 `fp32`。 |
| `-xbias` | `0` | 是否在 fused add 前加 input bias。 |
| `-fadd` | `0` | fused add，`0` no add，`1` preadd+store，`2` preadd only。 |
| `-fquant` | `0` | fused quant，`0` no quant，`1` smooth dynamic quant，`2` dynamic quant。 |
| `-warmup` | `5` | 预热次数。 |
| `-repeat` | `20` | 计时重复次数。 |

### 调用示例

```bash
./tile_example_layernorm2d_fwd
./tile_example_layernorm2d_fwd -prec_i=fp16 -prec_o=fp16 -m=3328 -n=4096
```


## 4.14 RMSNorm2D

### 功能描述

RMSNorm2D 示例实现二维 RMSNorm forward，输入形状按 `m x n` 组织。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-m` | `3328` | M 维度。 |
| `-n` | `4096` | N 维度。 |
| `-stride` | `-1` | row stride，`-1` 表示等于 `n`。 |
| `-e` | `1e-5` | epsilon。 |
| `-v` | `1` | CPU 校验开关。 |
| `-prec` | `fp16` | 精度。 |
| `-warmup` | `0` | 预热次数。 |
| `-repeat` | `1` | 重复次数。 |

### 调用示例

```bash
./tile_example_rmsnorm2d_fwd -prec=fp16 -m=17 -n=16
```

## 4.15 Add + RMSNorm2D + RdQuant

### 功能描述

该示例融合 Add、RMSNorm2D 和 rowwise dynamic quantization，输出量化结果和缩放因子。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-m` | `3328` | M 维度。 |
| `-n` | `4096` | N 维度。 |
| `-stride` | `-1` | row stride，`-1` 表示等于 `n`。 |
| `-e` | `1e-5` | epsilon。 |
| `-v` | `1` | CPU 校验开关。 |
| `-prec` | `fp16` | 输入精度。 |
| `-warmup` | `0` | 预热次数。 |
| `-repeat` | `1` | 重复次数。 |

### 调用示例

```bash
./tile_example_add_rmsnorm2d_rdquant_fwd -prec=fp16 -m=17 -n=16
```

## 4.16 CK Tile Conv Fwd

### 功能描述

`tile_example_conv_fwd` 验证 `example_hcu/ck_tile/14_conv/conv_fwd.cpp` 中的 FP16 CK Tile forward convolution 路径，默认使用 `NHWGC/GKYXC/NHWGK` 布局。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-m` | `2` | spatial 维度，目前示例支持 2D。 |
| `-g` | `1` | group 数。 |
| `-n` | `64` | batch size。 |
| `-k` | `64` | 输出通道数。 |
| `-c` | `64` | 输入通道数。 |
| `-r/-s` | `3/3` | filter 高/宽。 |
| `-h/-w` | `4/4` | 输入高/宽。 |
| `-u/-v` | `1/1` | stride 高/宽。 |
| `-l/-j` | `1/1` | dilation 高/宽。 |
| `-p/-q` | `1/1` | padding 高/宽。 |
| `-rtol/-atol` | `1e-3/1e-3` | FP16 校验相对/绝对容差。 |

### 调用示例

```bash
./tile_example_conv_fwd
./tile_example_conv_fwd -rtol=1e-3 -atol=1e-3
```

### 约束与限制

该示例输出为 FP16，默认 shape 下可能出现约 `4.88e-4` 的最大绝对误差和少量逐点不一致。这属于 FP16 1-2 ULP 量级误差，使用 `1e-3` 容差校验更符合该示例的数值特性。

## 4.17 CK Tile Conv3D Fwd

### 功能描述

`tile_example_conv3d_fwd` 验证 `example_hcu/ck_tile/16_conv3d_fwd/conv3d_fwd.cpp` 中的 BF16 3D forward convolution 路径，默认使用 `NDHWGC/GKZYXC/NDHWGK` 布局和 `1x1x1 stride1 pad0` 卷积配置。

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-m` | `3` | spatial 维度，目前示例支持 3D。 |
| `-g` | `1` | group 数。 |
| `-n` | `16` | batch size。 |
| `-k` | `256` | 输出通道数。 |
| `-c` | `256` | 输入通道数。 |
| `-z/-r/-s` | `1/1/1` | filter 深/高/宽。 |
| `-d/-h/-w` | `14/14/14` | 输入深/高/宽。 |
| `-o/-u/-v` | `1/1/1` | stride 深/高/宽。 |
| `-t/-l/-j` | `1/1/1` | dilation 深/高/宽。 |
| `-p/-q` | `0/0` | 左/右 padding，示例会分别扩展到 D/H/W 三个空间维度。 |

### 调用示例

```bash
./tile_example_conv3d_fwd
```

### 返回值说明

校验通过返回 `0`，校验失败返回 `1`。


## 4.18 暂未处理的示例接口

以下接口来自 `example_hcu`，暂未提供。

| 示例 | 当前状态 |
| --- | --- |
| `example_grouped_conv_fwd_mmac_nhwc_v2_fp16_cshuffle` | 构建通过，默认运行被 `IsSupportedArgument` 拒绝。 |
| `example_grouped_conv_bwd_weight_mmac_nhwc_fp16` | 构建通过，默认运行校验失败，`max err` 约 `963.3891`。 |
| `example_gemm_bias_relu_mmac_fp16` | 构建通过，默认运行校验失败，`max err` 约 `21.51562`。 |
| `example_grouped_conv_fwd_bias_relu_mmac_nhwc_fp16_cshuffle` | 当前远端 build 中无 target，源码受 `CK_BUILD_UNSTABLE_HCU_EXAMPLES` 控制。 |
| `example_grouped_conv_fwd_bias_add_relu_mmac_nhwc_fp16_cshuffle` | 当前远端 build 中无 target，源码受 `CK_BUILD_UNSTABLE_HCU_EXAMPLES` 控制。 |
| `tile_example_fused_conv` | 当前远端 build 中无 target，源码受 `CK_BUILD_UNSTABLE_HCU_EXAMPLES` 控制。 |
