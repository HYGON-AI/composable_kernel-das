# CK Tile GEMM 算子使用说明

本文介绍 CK Tile GEMM 的命令行用法和 C++ 接口。Basic GEMM 展示 FP16 MMAC V1。Universal GEMM 提供 PyTorch Inductor 所需的 CompV3、Mem、CompV4 三类示例。 [03 号 GEMM 示例](../../example_hcu/ck_tile/03_gemm)

## 1. 功能与接口

算子计算 `C[M,N] = A[M,K] × B[K,N]`，使用 FP32 累加，结果转回输入精度。布局名按 A、B、C 顺序排列，R 表示行主序，C 表示列主序。

| 入口 | 数据类型 | 布局 | 用途 |
| --- | --- | --- | --- |
| [Basic GEMM](../../example_hcu/ck_tile/03_gemm/gemm_basic.cpp) · `tile_example_gemm_basic` | FP16 | RCR | 独立运行 MMAC V1，示例使用 `-b=1` |
| [Universal GEMM](../../example_hcu/ck_tile/03_gemm/universal_gemm.cpp) · `tile_example_gemm_universal` | FP16、BF16 | RCR、RRR | 选择 pipeline、调度和 epilogue，校验结果 |
| [CK Tile 公共头](../../include/ck_tile/ops/gemm.hpp) | 由模板指定 | 由 traits 指定 | 由 C++ 调用方组装 pipeline、partitioner、kernel 和 epilogue |

Universal 示例使用 256×256×32 的 block tile。CompV3 和 CompV4 使用 Intrawave，Mem 可选 Intrawave 或 Interwave。三条 pipeline 都支持 Default 和 CShuffle epilogue。示例使用单分片 `-split_k=1`，并要求 `-persistent=0 -flush_cache=false -rotating_count=1`。

## 2. 构建与运行

从 CK 仓库根目录执行。选择工具链支持的目标架构 `gfx928`、`gfx92a`、`gfx936` 或 `gfx938`。切换架构时使用独立构建目录。

```bash
arch=gfx936
build="build-gemm-$arch"
mkdir -p "$build"
(
  cd "$build"
  CK_GPU_TARGETS="$arch" CK_VERBOSE_MAKEFILE=OFF bash ../script/cmake-ck-dev.sh ..
)
cmake --build "$build" --target tile_example_gemm_basic tile_example_gemm_universal -j8

"$build/bin/tile_example_gemm_basic" \
  -b=1 -prec=fp16 -a_layout=R -b_layout=C -c_layout=R \
  -m=128 -n=128 -k=64 -v=1 -warmup=0 -repeat=0
```

### PyTorch Inductor 需求对应的三个 example

| 需求 | Universal GEMM 参数 | CK Tile 模板 |
| --- | --- | --- |
| **CompV3** | `-pipeline=compv3 -scheduler=intrawave` | [`GemmPipelineAgBgCrCompV3`](../../include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp) |
| **Mem** | `-pipeline=mem`，调度可选 `intrawave` 或 `interwave` | [`GemmPipelineAgBgCrMem`](../../include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp) |
| **CompV4** | `-pipeline=compv4 -scheduler=intrawave` | [`GemmPipelineAgBgCrCompV4`](../../include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp) |

三个示例共用 `tile_example_gemm_universal`，通过参数选择对应的模板实例。以下命令分别运行两种精度，并用 CPU 参考验证输出。

```bash
gemm="$build/bin/tile_example_gemm_universal"
for prec in fp16 bf16; do
  "$gemm" -pipeline=compv3 -scheduler=intrawave -prec="$prec" \
    -m=256 -n=256 -k=128 -b_layout=C -epilogue=default -split_k=1 -v=1
  "$gemm" -pipeline=mem -scheduler=interwave -prec="$prec" \
    -m=256 -n=256 -k=128 -b_layout=C -epilogue=cshuffle -split_k=1 -v=1
  "$gemm" -pipeline=compv4 -scheduler=intrawave -prec="$prec" \
    -m=256 -n=256 -k=128 -b_layout=R -epilogue=default -split_k=1 -v=1
done

bash example_hcu/ck_tile/03_gemm/script/smoke_test.sh "$build"
```

### Universal GEMM 常用参数

| 参数 | 含义 |
| --- | --- |
| `-m/-n/-k` | 正整数，默认 256/256/128 |
| `-a_layout/-b_layout/-c_layout` | 默认 R/C/R。B 也可设为 R |
| `-stride_a/-stride_b/-stride_c` | 以元素为单位的 leading stride。0 表示紧凑存储。A、C 默认分别为 K、N，B 行主序时为 N，列主序时为 K |
| `-init` | 0 为固定种子的随机输入，1 为线性序列，2 为全 1 |
| `-v` | 0 跳过校验，1 使用 CPU 参考，2 使用 GPU 参考。校验覆盖每个输出元素和 C 的行间填充区域 |

维度和 stride 须满足所选实例的向量对齐要求。示例在 launch 前检查参数，对不支持的组合返回非零退出码。带边界和非紧凑 stride 的 RCR 例子为 `-m=264 -n=280 -k=72 -stride_a=80 -stride_b=80 -stride_c=288`。

## 3. C++ 调用

调用方提供设备指针、维度、元素 stride 和 HIP stream，并负责缓冲区生命周期。`UniversalGemmKernelArgs<>` 中的两个空字段分别对应附加张量指针和 stride。以下用 BF16 RCR CompV3 举例，编译时将 03 号示例目录和 CK 的 `include` 加入包含路径。 [完整模板调用](../../example_hcu/ck_tile/03_gemm/universal_gemm_invoker.hpp)

```cpp
#include "universal_gemm_invoker.hpp"

ck_tile::UniversalGemmKernelArgs<> args{
    {a_dev}, {b_dev}, {}, c_dev, m, n, k, {lda}, {ldb}, {}, ldc, 1
};
invoke_universal_gemm<ck_tile::bf16_t,
                      Col,
                      Pipeline::CompV3,
                      ck_tile::GemmPipelineScheduler::Intrawave,
                      false>(args, ck_tile::stream_config{stream});
```

末尾模板参数 `false` 选择 Default，`true` 选择 CShuffle。launcher 根据 K tile 数选择 hot loop 和 tail 实例，`Kernel::IsSupportedArgument` 通过后发射 kernel。生成 C++ 的调用方可使用公共头组装实例。 [pipeline 类型映射](../../example_hcu/ck_tile/03_gemm/gemm_utils.hpp) · [`GemmKernel` 单分片适配](../../include/ck_tile/ops/gemm/kernel/gemm_universal_adapter.hpp)

## 4. 验证范围

| 架构 | Basic 和 Universal 编译链接 | Universal 实机数值 |
| --- | --- | --- |
| gfx936、gfx938、gfx92a | 通过 | 各通过 224 个 CPU 参考用例、1 个 GPU 参考用例及 5 个非法参数检查 |
| gfx928 | 通过 | 待实机验证 |

数值矩阵覆盖两种精度、两种布局、两种 epilogue、Mem 的两种调度、K 循环边界和非紧凑 stride。三个 pipeline 均已在 gfx936 和 gfx938 上通过公共头的独立编译链接。PyTorch Inductor 通过公共头生成并编译 GEMM kernel，以上命令用于独立验收 CK 侧示例。
