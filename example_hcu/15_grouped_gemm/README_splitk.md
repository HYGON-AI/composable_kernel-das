# HCU grouped GEMM XDL Split-K

本目录提供以下两个 `ck::tensor_operation::device` 接口的独立 C++ 测试例子：

| 接口 | 例子源文件 | 本例计算 |
| --- | --- | --- |
| `DeviceGroupedGemmXdlSplitKCShuffle` | [grouped_gemm_xdl_splitk_fp16.cpp](grouped_gemm_xdl_splitk_fp16.cpp) | 每组 `E = A × B`，FP16 输入、FP16 输出 |
| `DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage` | [grouped_gemm_multiple_d_splitk_xdl_fp16.cpp](grouped_gemm_multiple_d_splitk_xdl_fp16.cpp) | 每组 `E = A × B + D0 + D1`，FP16 输入及 D、FP32 输出 |

两个例子均在 CK 内生成输入、运行 GPU 算子并逐元素对比 CPU 参考结果，不需要
PyTorch 或 Primus-Turbo。以下流程只依赖本仓库源码和正常安装的 HCU 开发环境，
从全新构建目录开始，不需要历史二进制、特定服务器、容器或仓库外的测试脚本。

## 1. 从源码配置构建目录（两个接口共用）

### 前置条件

- Linux、Bash、CMake（仓库最低要求 3.14）、Make，以及系统 C/C++ 编译器和标准库开发文件。
- `python3` 可从 PATH 调用，供 CMake 配置阶段的 CK Tile 实例生成脚本使用。
  这些脚本随仓库提供；本节构建不需要安装 PyTorch。
- 完整的 HCU DTK 开发环境，包括支持目标架构的 HIP 编译器、HIP 头文件/运行库、
  ROCM CMake 包；运行时需要可访问的 gfx936 或 gfx938 GPU 和匹配的驱动。
- 使用包含本目录例子及对应实现的 Hygon CK 源码。下面的命令在 **Linux Bash** 中执行。

本流程使用仓库内的 [cmake-ck-dev.sh](../../script/cmake-ck-dev.sh)。该脚本按顺序
选择 `/opt/dtk/bin/aicc`、`/opt/dtk/llvm/bin/hipcc`，最后回退到
`/opt/rocm/bin/hipcc`。选中的编译器必须支持实际使用的 HCU 架构；这些是脚本的
工具链安装路径约定，源码目录可以位于任意位置。非标准工具链安装路径需相应调整
脚本中的编译器和 CMake prefix 配置。

先进入本仓库的 `composable_kernel` 根目录，再执行以下完整配置命令：

```bash
set -e

CK_ROOT="$(pwd -P)"
test -f "$CK_ROOT/example_hcu/15_grouped_gemm/grouped_gemm_xdl_splitk_fp16.cpp"

# 按实际运行 GPU 选择一个架构；gfx938 机器改成 ARCH=gfx938。
ARCH=gfx936

# 每次生成新的空目录，不复用以前的 CMakeCache 或可执行文件。
BUILD="$(mktemp -d "$CK_ROOT/build-grouped-splitk-${ARCH}.XXXXXX")"
(
  cd "$BUILD"
  CK_GPU_TARGETS="$ARCH" CK_VERBOSE_MAKEFILE=OFF \
    bash "$CK_ROOT/script/cmake-ck-dev.sh" "$CK_ROOT"
)
printf 'Build directory: %s\n' "$BUILD"
```

脚本会启用 `BUILD_EXAMPLE=ON`、Release 和开发检查，并生成本节后续需要的目标。
**必须在新建的专用构建目录内调用该脚本**：它会清理当前目录中的 `CMakeCache.txt`、
`*.cmake` 和 `CMakeFiles`。上面的子 Shell 已保证这一点。

后面的命令继续在同一个 Bash 会话中执行，使用刚定义的 `$BUILD`。
可以只编译、运行其中一个接口；无需先编译整个 CK，也无需先运行另一个接口。
验证另一架构时，在对应 GPU 上设置相应的 `ARCH`，从本节重新配置并编译。

## 2. 接口一：DeviceGroupedGemmXdlSplitKCShuffle

实现头文件：[device_grouped_gemm_xdl_splitk_cshuffle.hpp](../../include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp)。

### 本例测试内容

- 每组计算 `E[M,N] = A[M,K] × B[K,N]`，没有 D 张量。
- A 为 RowMajor，B 为 ColumnMajor，E 为 RowMajor；A/B/E 为 FP16，GEMM 累加类型为 FP32。
- 默认 3 组：`M = [64,127,255]`，各组 `N = 128`、`K = 256`。
  A/B stride 为 256，E stride 为 128；包含 M 方向 tile 尾部。
- `k_batch=1` 不切分 K；`k_batch>1` 使用 Split-K，各分片结果累加到 E。

### 编译与运行

完成第 1 节配置后：

```bash
cmake --build "$BUILD" -j 4 --target example_hcu_grouped_gemm_xdl_splitk_fp16

# 默认 3 组，分别验证不切分 K 和 Split-K=4；均执行正确性检查。
"$BUILD/bin/example_hcu_grouped_gemm_xdl_splitk_fp16" 1
"$BUILD/bin/example_hcu_grouped_gemm_xdl_splitk_fp16" 4

# 相同的默认 3 组，另外开启 GPU 计时。
"$BUILD/bin/example_hcu_grouped_gemm_xdl_splitk_fp16" 4 timing

# 自定义统一形状：16 组，每组 M=32、N=128、K=1024，Split-K=4，并计时。
"$BUILD/bin/example_hcu_grouped_gemm_xdl_splitk_fp16" 4 timing 16 32 128 1024
```

支持两种调用形式（方括号表示可选参数，不要输入方括号）：

```text
example_hcu_grouped_gemm_xdl_splitk_fp16 [k_batch [timing]]
example_hcu_grouped_gemm_xdl_splitk_fp16 k_batch timing group_count M N K
```

- 不带参数等价于 `4`：默认 3 组、`k_batch=4`、不计时。
- `k_batch` 必须为正整数，并满足所选形状和实例的支持条件。
- `timing` 是位置占位参数：程序只检查第二个参数是否存在，不解析其文本或布尔值。
  因此 `4 0` **也会开启计时**；关闭计时应省略第二个参数。
- 自定义形状必须按第二种形式给齐 6 个参数，`group_count` 必须大于 0，
  所有组使用相同 M/N/K，且始终开启计时。其他参数个数组合不是自定义形状用法。
- 本例固定 FP16 和上述布局；不要在命令后追加 dtype 或布局参数。
  其他类型/布局由第 4 节契约测试覆盖。

### 成功判据

例如运行 `4` 后，应先打印 `CASE dtype=fp16 layout=RC D=0 ...`，
再打印 `PASS groups=3 k_batch=4 padded=0 timing=0 ...`，进程退出码为 0。
`4 timing` 对应 `timing=1`；自定义 16 组命令对应 `groups=16`。
输出中的误差、耗时和 workspace 大小由实际运行计算，不要求与其他机器相同。

## 3. 接口二：DeviceGroupedGemmMultipleDSplitKXdlCShuffleTwoStage

实现头文件：[device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp](../../include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp)。

### 本例测试内容

- 每组先将 `A × B` 累加到 FP32 workspace，再由 epilogue 计算 `E = workspace + D0 + D1`。
  D0、D1 在所有 K 分片完成之后各加一次，不能随 `k_batch` 重复添加。
- A 为 RowMajor，B 为 ColumnMajor，D0/D1/E 均为 RowMajor。
  **A/B/D0/D1 为 FP16，GEMM 累加、workspace 和 E 为 FP32**；
  可执行文件名中的 `fp16` 指输入类型，不代表 E 是 FP16。
- 固定 7 组：`M = [64,127,255,129,260,190,77]`，各组 `N = 252`、`K = 4608`。
  包含 M/N 方向 tile 尾部。
- `padded` 模式测试 A/B/D/E 使用各自非紧凑 stride 的情况。stride 单位为元素数：

| 模式 | A stride | B stride | D0 stride | D1 stride | E stride |
| --- | ---: | ---: | ---: | ---: | ---: |
| 不传 `padded` | 4608 | 4608 | 252 | 252 | 252 |
| 传入 `padded` | 4616 | 4616 | 256 | 260 | 264 |

### 编译与运行

完成第 1 节配置后，可独立执行：

```bash
cmake --build "$BUILD" -j 4 --target example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16

# 默认 7 组，分别验证不切分 K 和 Split-K=4。
"$BUILD/bin/example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16" 1
"$BUILD/bin/example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16" 4

# Split-K=4，开启带 padding 的 stride，检查输出及 padding/guard。
"$BUILD/bin/example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16" 4 padded

# 在上述 padded 用例基础上开启 GPU 计时。
"$BUILD/bin/example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16" 4 padded timing

# 默认形状下的较大 K 切分数用例。
"$BUILD/bin/example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16" 128
```

调用形式：

```text
example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16 [k_batch [padded [timing]]]
```

- 不带参数等价于 `4`：默认 7 组、`k_batch=4`、紧凑 stride、不计时。
- `k_batch` 必须为正整数，并满足实例支持条件。
- 第二个参数只按是否存在决定启用 padded stride，第三个参数只按是否存在决定计时。
  因此 `4 timing` 实际是 **开启 padded、不计时**；`4 0 0` 会同时开启两项。
  请直接使用上面的 `4 padded` 或 `4 padded timing`。
- 当前例子没有“紧凑 stride 且计时”的命令行组合，也没有自定义 group_count/M/N/K
  的参数。修改形状需修改本例源码后重编译；不要套用接口一的 6 参数形式。
- 本例固定两个 D 和 AddAdd。接口的零 D/PassThrough 路径由第 4 节契约测试覆盖。

### 成功判据

运行 `4 padded timing` 后，应先打印 `CASE dtype=fp16 layout=RC D=2 ...`，
再打印 `PASS groups=7 k_batch=4 padded=1 timing=1 ...`，进程退出码为 0。
其他命令中的 `padded`、`timing` 字段应与所选模式一致。

## 4. 两接口完整契约回归

下面从第 1 节新建的构建目录中编译全部 4 个相关目标，再执行本组 CTest。
即使没有执行第 2、3 节的编译命令，也可以直接执行本节：

```bash
for target in \
  example_hcu_grouped_gemm_xdl_splitk_fp16 \
  example_hcu_grouped_gemm_multiple_d_splitk_xdl_fp16 \
  example_hcu_grouped_gemm_splitk_contract \
  example_hcu_grouped_gemm_splitk_no_timing
do
  cmake --build "$BUILD" -j 4 --target "$target"
done

(
  cd "$BUILD"
  ctest --output-on-failure -L grouped_gemm_xdl_1103
)

# 可选：单独查看某种类型的详细输出。
"$BUILD/bin/example_hcu_grouped_gemm_splitk_contract" fp16
"$BUILD/bin/example_hcu_grouped_gemm_splitk_contract" bf16
"$BUILD/bin/example_hcu_grouped_gemm_splitk_contract" fp32
```

当前 [CMakeLists.txt](CMakeLists.txt) 为该标签注册 **11 项测试**，全部通过时应显示
`100% tests passed, 0 tests failed out of 11`。只编译某一个例子后不要直接运行整个标签，
否则其他测试的可执行文件尚未生成。

契约测试覆盖 FP16/BF16/FP32、A/B 的 RR/RC/CR/CC 布局、零组、零 M/N、对齐尾部、
两条 stream、重复运行、修改 k_batch、buffer 重绑定、非法参数和
`CK_TIME_KERNEL=0`。输入包括坐标模式及固定种子 1103 的随机数据。
对于有意构造的不支持参数，输出 `EXPECTED_UNSUPPORTED` 是预期结果；
测试会核对拒绝行为，不能将这类输出解释为所有参数都支持。

两个例子和契约测试共用 [grouped_gemm_splitk_common.hpp](grouped_gemm_splitk_common.hpp)：
逐元素检查输出、输出 stride padding 和 guard。正确性失败或非预期的不支持会返回非零退出码。
`PASS` 行的 `error` 为最大绝对误差，`workspace` 单位为字节，`ms` 单位为毫秒；
不开启计时时 `ms=0` 不代表实际执行耗时为零。开启计时仍会做 CPU 正确性检查，
程序整体墙钟时间不能当作 GPU 算子耗时。

## 5. 实例与能力边界

- 本例使用原生 HCU 16×16 MMAC：block256、M64/N128/K32、AK1=BK1=8、
  MPerXDL=NPerXDL=16、MXdlPerWave=2、NXdlPerWave=4、prefetch1/default scheduler。
  上游 32×32 XDL 实例参数不能直接照搬到本例。
- A/B 同类型且 elementwise operation 为 PassThrough；E 为 RowMajor。
  接口一没有 D，输出操作为 PassThrough；接口二覆盖零 D/PassThrough
  和两个 RowMajor D/AddAdd。
- vector8 实例要求 A/B 的连续维度、stride 和地址满足 vector8 对齐；
  接口一 E 和中间输出 workspace 使用 vector4。四种 A/B 布局的契约测试使用
  scalar input transfer；M/N tile 尾部由 padding 处理。
- 接口一 `k_batch=1` 使用 Set，支持 padded E stride；`k_batch>1` 要求 E 紧凑，
  且输出类型为 FP16 或 FP32。BF16 atomic 输出会被拒绝；BF16 Split-K 或
  Split-K 加 padded E 应使用接口二。
- 接口二使用 FP32 累加 workspace，再为每个有效 group 启动一个 epilogue。
  D/E 使用各自 stride，D 只在 K 分片归约完成后添加；`k_batch=1` 使用 Set，
  无需上游的 atomic-add-and-clear 序列。
- 所有非空 group 对所选 k_batch 必须具有相同的 main-loop 分类，否则 launch 前拒绝。
  例如该 tile 下 `K=[128,256]`、`k_batch=4` 分类不同，会被拒绝；`k_batch=1` 可支持。
- 非正 k_batch、负维度、非空组的 K=0 和溢出参数会被拒绝。
  M=0 或 N=0 的组跳过；没有有效 group 时不启动 kernel。
- 其他 pipeline、scheduler、输出布局和硬件架构不在本次适配的验证范围。
  这些测试验证正确性，不表示相对其他 grouped GEMM 后端具有普遍性能优势。

## 6. Buffer、stream 与计时语义

接口一的 `GetDeviceKernelArgSize` 和 `GetWorkSpaceSize` 描述同一类 device 参数缓冲区；
`SetDeviceKernelArgs`、`SetWorkSpacePointer` 均可绑定，最后一次设置生效。
这兼容只设置 workspace 的旧调用方式及依次调用两个 setter 的调用方式。

接口二需要**彼此独立**的 device 参数缓冲区和 FP32 workspace。
分别查询大小、分别绑定，所有操作使用调用方提供的 stream。
workspace 大小为 `sum(MPadded * NPadded * sizeof(float))`，不随 k_batch 增长。
前一轮执行完成后可以重绑定 buffer；并发 stream 各自拥有独立的 Argument 和 buffer，
输入可只读共享。Argument、输入、输出及 buffer 必须保持存活至对应 stream 完成。
Graph capture 不在当前验证范围内。

开启计时时，接口一包含输出清零和所有计算 kernel，其参数上传在计时重复执行前完成一次；
接口二还包含参数上传、workspace reset、GEMM 和所有 group 的 epilogue。
每次 warmup/repeat 都从新的累加状态开始。

这两个 legacy CK 接口与 Primus-Turbo 使用的 CK Tile device-args API 是不同路径。
以上例子直接验证 CK 接口；PyTorch 接入及框架端测试由其维护者另行完成。
