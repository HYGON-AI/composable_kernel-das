# HCU DeviceGemmWmma_CShuffle 示例与验证

本示例调用公开接口
`ck::tensor_operation::device::DeviceGemmWmma_CShuffle`，头文件为
`ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp`。
在 gfx936/gfx938 上，实际计算使用 HCU 已适配的 XDL/MMAC CShuffle 内核。
接口名称中的 WMMA 不表示执行 AMD WMMA 指令。

示例和测试均位于本目录，可独立在 CK 中编译运行，不依赖 Torch 或临时验证目录。

| 文件 | 用途 |
|---|---|
| `gemm_wmma_fp16.cpp` | FP16 示例入口 |
| `gemm_wmma_bf16.cpp` | BF16 示例入口 |
| `gemm_wmma_hcu_example.hpp` | 实例配置、CPU 参考计算、数值与接口契约测试 |
| `CMakeLists.txt` | 构建目标和 CTest 注册 |

## 1. 编译

以下命令在已配置 DTK 的 Linux 环境中执行，当前目录应为包含本次修改的 CK
源码根目录。需要将 `aicc`、CMake 和主机 `g++` 加入 `PATH`。
gfx936 与 gfx938 使用各自的构建目录，避免混用架构。

```bash
set -euo pipefail

# 在 gfx936 机器上改为 gfx936；以下以 gfx938 为例。
arch=gfx938
build_dir="build_wmma_${arch}"
aicc_path="$(command -v aicc)"
# 按 aicc 位于工具链 bin 目录的常规安装布局推导工具链前缀。
dtk_root="$(dirname "$(dirname "$aicc_path")")"

cmake -S . -B "$build_dir" \
  -DCMAKE_CXX_COMPILER="$aicc_path" -DCMAKE_PREFIX_PATH="$dtk_root" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_DEV=ON -DBUILD_EXAMPLE=ON \
  -DGPU_TARGETS="$arch" \
  -DHIP_CLANG_INCLUDE_PATH="$(aicc -print-resource-dir)/include" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-L$(dirname "$(g++ -print-file-name=libstdc++.so)") -lstdc++"

# 同时编译 FP16 和 BF16 示例。
cmake --build "$build_dir" --target example_gemm_wmma -j2
```

可执行文件为：

```text
build_wmma_<架构>/bin/example_gemm_wmma_fp16
build_wmma_<架构>/bin/example_gemm_wmma_bf16
```

也可分别构建 `example_gemm_wmma_fp16`、`example_gemm_wmma_bf16` 目标。
`HIP_CLANG_INCLUDE_PATH` 指向 aicc 自身的编译器资源目录，避免混用其他 Clang
版本的头文件及运行库配置。

## 2. 如何测试验证

以下命令沿用上面的 `build_dir` 变量。若重新打开 shell，请先按实际架构设置
`build_dir=build_wmma_gfx936` 或 `build_dir=build_wmma_gfx938`。

### 2.1 选择设备并运行默认用例

先在运行环境中查看 GPU 使用率，选择空闲设备。下面的设备编号 `0` 需要按实际情况调整。

```bash
rocm-smi --showuse --showmemuse
export HIP_VISIBLE_DEVICES=0

"$build_dir/bin/example_gemm_wmma_fp16"
"$build_dir/bin/example_gemm_wmma_bf16"
```

不传参数时，执行 `M=64, N=128, K=64` 的数值验证，不计时。
程序开头会打印 `device=gfx936` 或 `device=gfx938`，以及包含
`HCU_XDL_Fallback` 的实例信息。确认架构与编译配置一致，并检查结果为 `PASS`。

### 2.2 运行完整数值与接口回归

```bash
set -euo pipefail

"$build_dir/bin/example_gemm_wmma_fp16" --test \
  | tee "$build_dir/wmma_fp16_test.log"
"$build_dir/bin/example_gemm_wmma_bf16" --test \
  | tee "$build_dir/wmma_bf16_test.log"
```

每种 dtype 应完成 **28 个数值用例和一组支持/拒绝契约检查**。通过标准是：

- 各数值用例输出 `PASS`。
- 输出 `PASS support/rejection contracts`。
- 最后输出 `ALL_PASS`，进程退出码为 `0`。
- 不出现 `FAIL`、`FAILED` 或 `ERROR`。

输出结尾示例：

```text
PASS RowMajor/ColumnMajor M=17 N=23 K=65 strides=68,70,30 api=typed max_error=0
PASS support/rejection contracts
ALL_PASS
```

`set -o pipefail` 保证测试失败时，使用 `tee` 保存日志也不会掩盖失败退出码。
数值或运行失败返回 `1`；参数个数错误返回 `2`。

### 2.3 使用 CTest 验证构建集成

```bash
ctest --test-dir "$build_dir" \
  -R '^example_gemm_wmma_(fp16|bf16)$' --output-on-failure
```

预期结果：

```text
100% tests passed, 0 tests failed out of 2
```

这两项 CTest 分别调用 FP16/BF16 示例的 `--test`，与上一节是同一测试集，
不能重复计为额外的数值覆盖。需要查看全部成功用例输出时，可增加 `-V`。

### 2.4 指定尺寸、stride 与计时

单用例命令行格式如下，FP16/BF16 的参数相同：

```text
example_gemm_wmma_<dtype> [verify init_method time [M N K StrideA StrideB StrideC]]
```

| 参数 | 含义 |
|---|---|
| `verify` | `1`：与 CPU 参考结果比较；`0`：仅运行，不做数值验证 |
| `init_method` | 保留 CK 常用命令行参数位置；当前始终使用确定性小数输入，该值不改变初始化方式 |
| `time` | `1`：启用 CK kernel 计时；`0`：关闭计时 |
| `M N K` | GEMM 的逻辑尺寸，均须大于 0 |
| `StrideA` | A 为行主序 `[M,K]`，要求 `StrideA >= K` |
| `StrideB` | B 为列主序 `[K,N]`，要求 `StrideB >= K` |
| `StrideC` | C 为行主序 `[M,N]`，要求 `StrideC >= N` |

stride 的单位是元素个数。单用例 CLI 固定使用上述 A/B/C 布局；其他 A/B 布局由
`--test` 覆盖。可使用以下命令分别验证尾块和开启计时的调用：

```bash
# 非整块 M/N/K，加上非紧凑 stride；开启数值验证，关闭计时。
"$build_dir/bin/example_gemm_wmma_fp16" 1 2 0 65 129 33 36 38 136
"$build_dir/bin/example_gemm_wmma_bf16" 1 2 0 65 129 33 36 38 136

# 数值验证与计时同时开启，输出应包含 PASS 和 time_ms。
"$build_dir/bin/example_gemm_wmma_fp16" 1 2 1 64 128 64 64 64 128
"$build_dir/bin/example_gemm_wmma_bf16" 1 2 1 64 128 64 64 64 128
```

`verify=0` 时输出 `RUN`，不能将其视为数值验证通过。计时使用非默认 HIP stream；
本示例的计时调用用于检查计时功能，不构成性能优化或与其他实现的性能对比结论。

## 3. 回归覆盖与数值判定

每种 dtype 的 28 个数值用例组成如下：

| 类别 | 覆盖内容 | 用例数 |
|---|---|---:|
| 四种 A/B 布局 | Row/Row、Row/Column、Column/Row、Column/Column；每种布局运行下列六组尺寸 | 24 |
| Default | 尺寸满足调用者声明的 M/N/K block 整数倍 | 1 |
| MNPadding | M/N 尾块，K 满足 block 整数倍 | 1 |
| 同 dtype CShuffle | 使用 Torch 调用中的 CShuffle dtype 配置 | 1 |
| 有状态 elementwise functor | 分别传递 A/B/C 的缩放系数，检查状态是否保留 | 1 |

六组尺寸按 `(M,N,K)` 排列：

```text
(64,128,64), (128,256,192), (65,129,33),
(7,9,5), (1,1,1), (32,48,513)
```

这些用例同时检查非紧凑 stride、非零指针偏移、类型化接口与 `DeviceGemm` 虚接口、
同一非默认 stream 上的异步 H2D/kernel/D2H，以及 K 主循环和仅尾块两种内核路径。

CPU 参考计算使用 FP32 累加，并模拟输入 elementwise、CShuffle 类型转换和输出
elementwise。每个逻辑输出元素须为有限值，且满足：

```text
abs(GPU结果 - 参考结果) <= tol * (1 + abs(参考结果))
FP16: tol = 0.001
BF16: tol = 0.008
```

输出前后保护区及每行 padding 区域按字节检查，必须保持初始值。
`max_error` 表示当前用例的最大绝对误差；通过标准采用上述容差，并不强制误差必须为 0。

支持/拒绝契约检查还覆盖零或负尺寸、空指针、过小 stride、禁止 padding 时的尾块、
C 列主序、空的虚接口参数，以及来自其他模板实例的 argument。
对不支持的参数，支持性检查应返回 `false`；调用 Invoker 时应抛出异常，不能直接启动内核。

## 4. 接口语义与支持范围

- 输入和输出类型必须相同，支持 FP16 或 BF16，使用 FP32 累加。A/B 支持行主序和
  列主序，C 仅支持行主序。示例已覆盖 FP32 和输入同 dtype 的 CShuffle 存储。
- 公开模板顺序与上游示例一致：`GemmSpec, NumPrefetch, BlockSize, MPerBlock,
  NPerBlock, KPerBlock, K1, ...`。`KPerBlock` 表示完整 K tile。
  旧 HCU 调用若将 `NumPrefetch` 放在尾部，需要移动该参数，并将旧 `K0PerBlock`
  替换为 `K0PerBlock * K1`。
- 支持 `MakeArgument`、`MakeInvoker`、`IsSupportedArgument`、`MakeArgumentPointer`
  和 `MakeInvokerPointer`，保留调用者传入的 A/B/C elementwise functor 状态和 HIP stream。
- HCU 内部固定采用 256 线程、`64x64x32` block tile、`16x16` XDL/MMAC、标量全局
  访存和 v1 intrawave pipeline。WMMA 搬运、wave 和 pipeline 调优参数用于源码接口
  兼容，不代表采用相同的 AMD kernel 配置或达到相同的性能。
- `GemmSpec` 按调用者声明的 M/N/K block 大小决定哪些维度允许尾块；内部使用带
  有效性掩码的 padding，适配固定的 HCU tile。
- 计算语义为 `C = C_op(A_op(A) * B_op(B))`。接口没有 alpha/beta、SplitK 或辅助 D
  张量参数，不负责 Torch 的框架调度。零 M/N/K 会被拒绝，不作为空操作处理。
- 当前验证范围是 gfx936/gfx938，不包含 INT8、FP32 输入、混合 A/B dtype、FP32
  输出或其他 GPU 架构。

## 5. 已完成的实机验证

2026-11-04 已在以下两种架构上完成实机验证，使用 aicc / HIP 6.2.0 / Clang 18：

| 架构 | FP16 | BF16 | CTest |
|---|---|---|---|
| gfx936 | 28/28 通过，契约检查通过 | 28/28 通过，契约检查通过 | 2/2 通过 |
| gfx938 | 28/28 通过，契约检查通过 | 28/28 通过，契约检查通过 | 2/2 通过 |

上述确定性数值用例的最大误差均为 0，两种 dtype 的开启计时调用也均通过。
此外完成了 Torch 使用的 CK include 顺序及调用形式的编译检查；未修改或完整构建
Torch，也未将该检查视为 Torch 端到端集成验证。

重新验证时，应分别在两个架构上构建并执行本说明第 2 节，保留各自日志。
若出现不支持参数的错误，先检查实际 GPU 架构、编译目标、M/N/K 和 stride；
若数值或保护区检查失败，保留失败行中的布局、尺寸、stride、API 类型及最大误差。
