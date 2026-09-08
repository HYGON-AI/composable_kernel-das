# HCU ck_tile 分组 GEMM

本目录是 HCU 侧 ck_tile 分组 GEMM 路径，和 legacy CK
`DeviceGroupedGemm_*` library/factory 路径不同，也和
`example_hcu/ck_tile/17_fused_moe` 下的 MoE 专用 group GEMM 不同。

## 关注的目标

当前 FP16/BF16 最小 smoke 使用已有 fast-build target：

```bash
cmake --build . --target tile_example_grouped_gemm_fp16_fast -j 16
cmake --build . --target tile_example_grouped_gemm_bf16_fast -j 16
```

显式 smoke target 会构建上述两个 target，并运行 FP16/BF16 的 NT/NN/TN、
fixed-K/variable-K 以及非对齐 M/N/K correctness 检查；最后还会运行一个绕过
example CLI、直接链接现有 C ABI 的 layout/padding consumer：

```bash
cmake --build . --target tile_example_grouped_gemm_fp16_bf16_fast_smoke -j 16
```

等价的直接运行命令：

```bash
./bin/tile_example_grouped_gemm_fp16_fast -prec=fp16 -a_layout=R -b_layout=C -group_count=2 -Ms=128,256 -Ns=128,128 -Ks=128,128 -stride_As=128,128 -stride_Bs=128,128 -stride_Cs=128,128 -validate=1 -warmup=1 -repeat=1
./bin/tile_example_grouped_gemm_fp16_fast -prec=fp16 -a_layout=R -b_layout=R -group_count=2 -Ms=128,256 -Ns=128,128 -Ks=128,128 -stride_As=128,128 -stride_Bs=128,128 -stride_Cs=128,128 -validate=1 -warmup=1 -repeat=1
./bin/tile_example_grouped_gemm_bf16_fast -prec=bf16 -a_layout=R -b_layout=C -group_count=3 -Ms=128,256,192 -Ns=128,128,256 -Ks=128,192,256 -stride_As=128,192,256 -stride_Bs=128,192,256 -stride_Cs=128,128,256 -validate=1 -warmup=1 -repeat=1
./bin/tile_example_grouped_gemm_bf16_fast -prec=bf16 -a_layout=R -b_layout=R -group_count=3 -Ms=128,256,192 -Ns=128,128,256 -Ks=128,192,256 -stride_As=128,192,256 -stride_Bs=128,128,256 -stride_Cs=128,128,256 -validate=1 -warmup=1 -repeat=1
```

也可以通过 CTest 运行这组 17 项 smoke：

```bash
ctest -R 'tile_example_grouped_gemm_.*_fast_smoke' --output-on-failure
```

## FP8/BF8 Quant Grouped GEMM

gfx938（以及预留的 gfx946 host gate）提供三种 quant grouped GEMM 示例：

```bash
cmake --build . --target tile_example_grouped_gemm_quant_smoke -j 16
```

也可以分别构建、运行或交给 CTest：

```bash
cmake --build . --target tile_example_grouped_gemm_quant_tensor -j 16
cmake --build . --target tile_example_grouped_gemm_quant_rowcol -j 16
cmake --build . --target tile_example_grouped_gemm_quant_blockwise -j 16

./bin/tile_example_grouped_gemm_quant_tensor
./bin/tile_example_grouped_gemm_quant_rowcol
./bin/tile_example_grouped_gemm_quant_blockwise

ctest -R 'tile_example_grouped_gemm_quant_.*_smoke' --output-on-failure
```

每个程序均构造三个独立 group，并用逐 group CPU reference 验证 25 项用例；
三种 quant mode 合计 75 项。覆盖范围为：

- 输入：FP8×FP8、BF8×BF8、FP8×BF8、BF8×FP8；
- 输出：FP16、BF16；
- 前向固定 K：NN（Row/Row/Row）和 NT（Row/Column/Row）；
- 反向 variable-K：TN（Column/Row/Row），同一次 launch 中每组 K 不同；
- M/N/K 非 block tile 整数倍的 padding shape。

四种输入组合与两种输出在 NN、NT、TN 三种 layout 上均实际实例化并运行，
不是只验证一个 layout 后依赖模板推断。

三种 scale 合同为：

- TensorQuant：每个 group 的 A、B 各一个 scale；
- RowColQuant：AQ shape 为 `[M,1]`，BQ shape 为 `[1,N]`；
- ABQuantGrouped：AQ shape 为 `[M,ceil(K/128)]`；NN/NT 的 BQ 每 128 列、
  每 128 个 K 共享，TN 使用 dense-BQ，N 方向不共享 scale。

padding 是逻辑 tensor descriptor 的 tile right-pad，不要求额外分配物理 padded
显存；但 RowMajor C 等连续维仍必须满足对应向量 load/store 的对齐要求。因此
示例中的非 tile 对齐 N 选用 `192/120/136`，而不是宣称支持任意整数 N。

### 为什么示例不再按 FP8/BF8 拆分源文件

上游 `example/ck_tile/17_grouped_gemm` 中的
`quant_grouped_gemm_fp8_*.cpp` 和 `quant_grouped_gemm_bf8_*.cpp` 是同一个
`tile_example_quant_grouped_gemm` 可执行文件的显式模板实例化单元。例如
tensorwise 两个文件的核心区别只是分别实例化 `ck_tile::fp8_t` 和
`ck_tile::bf8_t`，不是两套不同的 kernel API。

HCU 示例按 quant mode 分为 tensorwise、rowwise 和 blockwise 三个可执行文件，
每个文件内部用独立的 A/B/C 模板参数实例化
FP8×FP8、BF8×BF8、FP8×BF8、BF8×FP8，以及 FP16/BF16 输出。这样能直接表达
混合输入类型；如果仍只按“FP8 文件/BF8 文件”拆分，FP8×BF8 和 BF8×FP8
还需要额外的组合文件。代价是当前三个示例源文件的模板实例较多，编译时间和
编译内存会高于细粒度显式实例化。后续若建设生产 instance library，建议再按
quant mode、A/B/C dtype、layout 和 tile config 拆分显式实例化目标。

### 第三方框架接入

第三方不应调用这三个 example 的 `main()`，而应在自己的 C++/HIP runner 中
实例化并启动 `ck_tile::QuantGroupedGemmKernel`。需要包含：

```cpp
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
```

`Kernel` 的模板组成和参考文件如下：

| quant mode | Problem / Pipeline | `QuantType` | 完整类型参考 |
| --- | --- | --- | --- |
| tensorwise | `GemmRowColTensorQuantPipelineProblem` / `GemmPipelineAgBgCrCompV3` | `TensorQuant` | `quant_grouped_gemm_tensor.cpp` |
| rowwise | `GemmRowColTensorQuantPipelineProblem` / `GemmPipelineAgBgCrCompV3` | `RowColQuant` | `quant_grouped_gemm_rowcol.cpp` |
| blockwise | `GemmABQuantPipelineProblem` / `ABQuantGemmPipelineAgBgCrCompV3` | `ABQuantGrouped` | `quant_grouped_gemm_blockwise.cpp` |

三个模式最终都使用：

```cpp
using Kernel =
    ck_tile::QuantGroupedGemmKernel<TilePartitioner, Pipeline, Epilogue, QuantMode>;
```

调用方为每个 group 构造一个 `ck_tile::QuantGroupedGemmHostArgs`。其构造参数
顺序为：

```cpp
ck_tile::QuantGroupedGemmHostArgs{
    a_ptr, b_ptr, c_ptr, aq_ptr, bq_ptr,
    k_batch,
    M, N, K,
    QK_A, QK_B,
    stride_A, stride_B, stride_C,
    stride_AQ, stride_BQ};
```

其中 stride 均以元素为单位。当前验证路径使用 `k_batch=1`。scale descriptor
约定为：

| quant mode | `QK_A/QK_B` | AQ | BQ |
| --- | --- | --- | --- |
| tensorwise | `1/1` | 单个 FP32 scale，`stride_AQ=1` | 单个 FP32 scale，`stride_BQ=1` |
| rowwise | `1/1` | `[M,1]`，`stride_AQ=1` | `[1,N]`，`stride_BQ=1` |
| blockwise | `ceil(K/128)` | RowMajor `[M,QK_A]`，`stride_AQ>=QK_A` | ColumnMajor `[QK_B,BQN]`，`stride_BQ>=QK_B` |

blockwise 的 NN/NT 使用 `BQN=ceil(N/128)`；TN 使用 dense-BQ，
`BQN=N`。A、B、AQ、BQ、C 指针可以在不同 group 间指向不同分配，且每个
descriptor 都有自己的 M/N/K，因此同一次 launch 支持 variable-K。

已经在 host 端得到各 group descriptor 时，可直接使用当前 example 验证过的
non-persistent launch 流程：

```cpp
std::vector<ck_tile::QuantGroupedGemmHostArgs> groups = /* one entry per group */;

auto translated_args = Kernel::MakeKargs(groups);
if(translated_args.size() != groups.size() ||
   !Kernel::IsSupportedArgument(translated_args))
{
    // 拒绝当前 Kernel 配置不支持的 shape、stride 或对齐。
}

void* workspace = nullptr;
const auto workspace_bytes = Kernel::GetWorkSpaceSize(groups);
hipMalloc(&workspace, workspace_bytes);
hipMemcpyAsync(workspace,
               translated_args.data(),
               workspace_bytes,
               hipMemcpyHostToDevice,
               stream);

ck_tile::launch_kernel(
    ck_tile::stream_config{stream},
    ck_tile::make_kernel<Kernel::kBlockSize, 1>(
        Kernel{},
        Kernel::GridSize(groups),
        Kernel::BlockSize(),
        0,
        ck_tile::cast_pointer_to_constant_address_space(workspace),
        ck_tile::type_convert<ck_tile::index_t>(translated_args.size())));
```

输入、scale、输出和 workspace 必须至少存活到该 stream 上 kernel 执行完成。
实际框架代码还应检查 HIP API 返回值，并缓存 workspace，避免每次调用
`hipMalloc`。空 M/N/K group 会被 `MakeKargs` 跳过；框架若允许空 group，
不能再用上面严格的 `translated_args.size() == groups.size()` 检查。

Primus-Turbo 这类高频调用方也可以在 GPU 上直接生成
`ck_tile::QuantGemmTransKernelArg` 数组，避免每次把 host descriptor 翻译后再
拷贝。此时：

- workspace 大小为 `Kernel::GetWorkSpaceSize(group_count)`；
- 每项必须完整填写 `group_karg`，并为 non-persistent 查找填写
  `[block_start, block_end)`；
- non-persistent kernel 使用所有 group 的实际 tile 总数作为 grid；
- persistent kernel 可按 `Kernel::MaxOccupancyGridSize(stream_config)` 启动，
  具体选择必须与 `Kernel::UsePersistentKernel` 一致。

Primus-Turbo 中可参考
`csrc/kernels/grouped_gemm/ck/ck_grouped_gemm_kernel_template.h` 的
`CKQuantGroupedGemmRunner` 和
`csrc/kernels/grouped_gemm/ck_grouped_gemm.cu` 的 quant args setup。CK 侧只提供
kernel、descriptor 和 launch building blocks；dtype/layout/tile dispatch、
workspace 缓存及框架注册仍由第三方负责。

gfx938 编译时，第三方 target 除指定 `--offload-arch=gfx938` 外，还需要在
host 编译阶段定义 `CK_TILE_HCU_QUANT_GEMM_GFX938=1`；本目录 CMake 已自动处理。
gfx946 对应宏为 `CK_TILE_HCU_QUANT_GEMM_GFX946=1`，目前只是代码预留，尚未在
当前 nmz-2 工具链上编译验证。

当前没有为 quant grouped GEMM 新增稳定 C ABI。下面的
`ck_tile_hcu_grouped_gemm_run` 只覆盖非量化 FP16/BF16 grouped GEMM，不能用它
调用 tensorwise、rowwise 或 blockwise quant kernel。需要纯 C/PyTorch ABI 的
第三方应在自己的扩展层封装上述 runner，或者后续在 CK 中单独设计并验证 quant
C ABI。

## 现有 C ABI（框架入口）

现有框架（包括 aiter）继续从本 example 目录包含原有 header：

```cpp
#include "grouped_gemm.hpp"
```

`grouped_gemm.hpp` 保留 dtype 枚举、descriptor、workspace 与 run 声明；
`CK_GROUPED_GEMM_ABI_DEFINED` guard 用于和 aiter 的纯 C ABI 声明共存。
本次不新增另一个 C ABI 接口头文件。

FP16/BF16 实现可通过以下 target 构建为位置无关的静态库：

```bash
cmake --build . --target ck_tile_hcu_grouped_gemm -j 32
```

独立 C ABI correctness consumer 的构建与运行命令为：

```bash
cmake --build . --target tile_example_grouped_gemm_c_api_smoke -j 32
./bin/tile_example_grouped_gemm_c_api_smoke
```

该程序不调用 `run_grouped_gemm_example`，而是分别构造对齐 NN variable-K 和
非对齐 TN descriptor，直接调用 `ck_tile_hcu_grouped_gemm_run` 并验证 FP16/BF16
输出。因此 layout 与 padding 能力不依赖 example CLI；fast CLI 是否开放
RowMajor B/ColumnMajor A 分别由 `GemmConfig::SupportsFastRowMajorB` 和
`GemmConfig::SupportsFastColumnMajorA` 声明。

gfx936 的 BF16 TN（A 列主序、B 行主序）大 shape 还提供一个 CK-owned
logical-K-tail 选择项。它把每组逻辑 K 直接传给 tag93607 的 peeled final-K64
pipeline；调用方不需要把 compact A/B 扩容到 K64，也不需要把各组容量补到
65536。只有下面的全部条件成立时，公共 C ABI 才选择该路径：

- 当前设备为 gfx936，dtype 为 BF16，layout 为 TN；
- group count 为 16，`k_batch=1`，没有 D tensor；
- A/B/C 使用标准 TN/row-major output stride；
- 每组 `M/N >= 2048` 且均为 256 的倍数，`K >= 128`。

其他 dtype、layout、shape、group count 和 GPU 架构继续使用原 generic
padding/non-padding portfolio。需要做同一 ELF 的回滚消融时，设置：

```bash
export CK_TILE_GROUPED_GEMM_DISABLE_GFX936_BF16_TN_LOGICAL_K_TAIL=1
```

生产 selector exact-60 case 和专用 logical-tail guard/hash/rollback smoke 可用以下
命令复现：

```bash
cmake --build . --target \
  tile_example_grouped_gemm_c_api_selected_smoke \
  tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke -j 16

./bin/tile_example_grouped_gemm_c_api_selected_smoke
./bin/tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke
```

其中 `tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke` 只在
`GPU_TARGETS` 包含 `gfx936` 时入图；其他架构没有该 gfx936 host-descriptor
selected path，应使用前一个通用 production-selector smoke 验收。

第二个 smoke 使用 16 个不同逻辑 K（总和 32757），以两个不同的 A/B
post-allocation guard 运行默认路径，再在同一进程设置 rollback 开关运行 generic
padding。验收要求两个默认输出 hash 相同、rollback 输出 hash 相同、抽样 FP32
reference 和 output guard 全部通过。

ck4inductor 的 `ck_tile_grouped_gemm` 模块枚举 FP16/BF16 × NT/NN/TN 六个
consumer contract，并提供 variable-K、padding、最小 K、现有 ABI 符号和 CMake
target 信息。它是 ROCm/HCU Torch backend 的接入材料；未修改的 PyTorch
TorchInductor 仍需要 grouped lowering/template 才会自动把 `aten._grouped_mm`
调度到该 C ABI。

## 当前边界

- FP16/BF16 fast target 覆盖 NT（A 行主序、B 列主序）、NN（A/B 行主序）和
  TN（A 列主序、B 行主序），C 均为行主序。其他 fast dtype 保持原有限制。
- variable-K 表示同一次 grouped launch 中每个 descriptor 的 K 可以不同；
  它不等同于 split-K，且当前所有 group 仍使用同一个 `kbatch`。
- FP16/BF16 对齐 workload 中，只有所有 group 均满足 `M/N/K>=512`、M/N 对齐
  128、且每个 split 的 K 对齐 64 时，才切换到 `128x128x64` square 配置。
  gfx936/gfx938 使用 4×2 waves、512 threads 的 spill-reduction 版本；其他架构
  继续使用原 2×2 waves、256 threads 的 square/square-wide 版本；
  其他对齐 shape 保持原 `64x128x64` `GemmConfigComputeV4`。任一 group 存在
  M/N/K 尾块时，example 与 C ABI 仍自动改用原 `kPadM/N/K=true` 配置。
- padding 是 tensor descriptor 的逻辑 right-pad，不要求调用方申请物理 padded
  A/B/C 显存；调用方仍按逻辑 M/N/K 和 stride 分配。workspace 仍只存每组
  `TransKernelArg`，大小不随 padded M/N/K 增长。
- 全 padding 路径为边界安全会降低相应向量化粒度，性能可能不同；后续性能
  调优应分别比较对齐 fast 路径与非对齐 padding 路径。
- 现有 FP16/BF16 fast target 使用 CompV4 persistent double-buffer 路径；
  当前要求 `num_loop = K / K_Tile >= 2`。
- FP16/BF16 的 `K_Tile=64`，因此当前 fast example 的 K 维 baseline
  应满足 `K>=128`。
- `num_loop < 2` 的小 K shape 已在 host-side launch/support check 中显式
  拒绝；若要真正支持，需要补充 small-K fallback。
- 本目标不注册 standard grouped GEMM factory/profiler instance，也不覆盖
  grouped GEMM MultiABD。

## `num_loop >= 2` 限制

`tile_example_grouped_gemm_fp16_fast` 对应的配置是
`GemmConfigComputeV4<fp16>`：`M_Tile=64`、`N_Tile=128`、`K_Tile=64`、
`DoubleSmemBuffer=true`、`Persistent=true`。BF16 fast target 同样使用
`K_Tile=64`。

CompV4 pipeline 的 `PrefetchStages=2`，tail 选择逻辑会把奇数 loop 映射到
`TailNumber::Three`。当前 fast example 以至少 2 个 K tile 作为支持前提：

```text
num_loop = K / K_Tile >= 2
```

对 FP16/BF16 fast target 来说，这等价于：

```text
K >= 2 * 64 = 128
```

因此 README 中的 correctness smoke 只选用 `K>=128` 的 shape。当前 fast path
会在 host 侧拒绝 `num_loop < 2` 的 group；后续如果要把 `K<128` 纳入支持范围，
应先增加专门的小 K 路径。
