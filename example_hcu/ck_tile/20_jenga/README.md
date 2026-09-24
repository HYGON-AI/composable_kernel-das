<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->
<!-- SPDX-License-Identifier: MIT -->
# HCU ck_tile Jenga 稀疏注意力

本目录提供 Jenga block-sparse attention 的独立 ck_tile 示例，包含 FWD、BWD
以及 GPU reference。当前支持 BF16、FP16、head dimension 128 和 64×64 稀疏
block。

## 目录结构

```text
include/ck_tile/ops/
├── jenga.hpp                           # 唯一公共算子入口
└── jenga/
    ├── block/                          # 基础 block GEMM
    ├── pipeline/                       # problem、policy 与 pipeline 实现
    └── kernel/                         # kernel、Kargs 实现

include/ck_tile/host/reference/
└── reference_jenga.hpp                 # 独立 GPU reference

example_hcu/ck_tile/20_jenga/
├── CMakeLists.txt
├── README.md
├── build.sh
├── instances/                          # 显式实例编译单元
│   ├── jenga_fwd_bf16_instance.cpp
│   ├── jenga_fwd_fp16_instance.cpp
│   ├── jenga_bwd_dq_instance.cpp
│   └── jenga_bwd_dkdv_instance.cpp
├── jenga_fwd.cpp                       # FWD 参数解析、调用入口与 main
├── jenga_fwd.hpp                       # FWD 接口声明
├── jenga_fwd_runner_impl.hpp           # FWD Runner 与 Launch 模板实现
├── jenga_bwd.cpp                       # BWD 参数解析、调用入口与 main
├── jenga_bwd.hpp                       # BWD 接口声明
└── jenga_bwd_launch.hpp                # BWD dQ 与 dK/dV 统一 Launch 实现
```

## 算子实现

### FWD 实现

- 使用 CK Tile mask builder 对 Q/K 做 block pooling；
- 计算 Q/K block 相关性；
- 根据 TopK 与概率阈值生成 onehot mask；
- 将 onehot mask 转换为前向 LUT；
- 通过 problem、policy、pipeline 和 kernel 分层实现稀疏注意力；
- 仅计算 LUT 选中的 K/V block；
- 输出 attention 结果以及可选的 LSE。

### BWD 实现

- 使用 CK Tile mask builder 生成 onehot mask；
- 将 onehot mask 转换为前向 LUT 和反向 LUT；
- preprocess 阶段计算 delta；
- dQ 使用前向 LUT 计算 Q 的梯度；
- dK/dV 使用反向 LUT 计算 K/V 的梯度；
- 长序列使用多 stage/split 分摊计算；
- split 计算完成后执行 reduce；
- 输出 dQ、dK/dV 结果。

## 构建

在仓库根目录执行：

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc \
  -DGPU_TARGETS=gfx936 \
  -DBUILD_DEV=ON \
  -DBUILD_EXAMPLE=ON \
  -DBUILD_TEST=OFF

cmake --build build \
  --target tile_example_jenga_fwd tile_example_jenga_bwd \
  -j 8
```

也可以直接运行构建脚本：

```bash
bash example_hcu/ck_tile/20_jenga/build.sh
```

生成文件位于：

```text
build/bin/tile_example_jenga_fwd
build/bin/tile_example_jenga_bwd
```

## 运行

### FWD

```bash
./build/bin/tile_example_jenga_fwd \
  -b=1 -h=40 -s=18048 -d=128 \
  -topk=28 -prec=bf16 -prob_threshold=0.3 \
  -v=0 -warmup=5 -repeat=10
```

### BWD

```bash
./build/bin/tile_example_jenga_bwd \
  -b=1 -h=40 -s=18048 -d=128 \
  -topk=28 -prec=bf16 -prob_threshold=0.3 \
  -v=0 -warmup=5 -repeat=10
```

`s=75648` 时，BWD 默认使用 `dq_kv_stages=5` 和 `dkdv_split_kv=4`；其他长度默认均为 1，也可以通过
-dq_kv_stages 和 -dkdv_split_kv 显式指定。

FP16 正确性测试示例：

```bash
./build/bin/tile_example_jenga_bwd \
  -b=1 -h=1 -s=1024 -d=128 \
  -topk=8 -prec=fp16 -prob_threshold=0.3 \
  -v=1 -warmup=1 -repeat=3
```

## Performance

以下数据来自 OpenDAS MR !6 的历史记录，未用于本次迁移的性能结论。

- GPU target：gfx936
- Precision：BF16
- Probability threshold：0.3

| Kernel | Batch Size | Heads | Head Dim | S | TopK | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FWD | 1 | 40 | 128 | 18048 | 28 | 14.43 ms |
| FWD | 1 | 40 | 128 | 75648 | 118 | 259.67 ms |
| BWD | 1 | 40 | 128 | 18048 | 28 | 59.81 ms |
| BWD | 1 | 40 | 128 | 75648 | 118 | 1088.3 ms |

## 命令行参数

| 参数 | 说明 | FWD 默认值 | BWD 默认值 |
| --- | --- | ---: | ---: |
| `-b` | Batch size | `1` | `1` |
| `-h` | Head 数量 | `1` | `1` |
| `-s` | 序列长度，必须为 64 的整数倍 | `1024` | `1024` |
| `-d` | Head dimension，当前只支持 128 | `128` | `128` |
| `-prec` | 数据类型：`bf16` 或 `fp16` | `bf16` | `bf16` |
| `-topk` | 每个 query block 至少选择的 KV block 数 | `8` | `8` |
| `-prob_threshold` | Mask builder 累计概率阈值 | `0.3` | `0.3` |
| `-text_start_block` | Text block 起点；`-1` 表示全部为 visual block | `-1` | `-1` |
| `-text_blocks` | 始终选择的 text block 数量 | `0` | `0` |
| `-first_frame_blocks` | 始终选择的首帧 block 数量 | `0` | `0` |
| `-store_lse` | 是否保存并检查 LSE | `1` | — |
| `-kv_stages` | FWD KV stages；`0` 表示自动选择 | `0` | — |
| `-dq_kv_stages` | dQ KV stages；`0` 表示自动选择 | — | `0` |
| `-dkdv_split_kv` | dK/dV split 数量；`0` 表示自动选择 | — | `0` |
| `-v` | 是否运行 GPU reference | `1` | `1` |
| `-warmup` | Warmup 次数 | `10` | `10` |
| `-repeat` | 计时重复次数 | `100` | `100` |
| `-seed` | 随机输入种子 | `1` | — |

## 当前限制

* head dimension 固定为 128，block size 固定为 64x64；
* BF16 和 FP16 均有独立 kernel 实例；
* mask builder 不计入前向、反向 kernel 时间。

## 来源与版权

本实现基于 OpenDAS/composable_kernel 的 Jenga sparse attention，直接采用
MR !6 的最终重构版本 `0149068aeab02ce9d6665de41f3337d6282d3fff`，
包含 `362ca48` 的 CK Tile ops 迁移及后续配置整理。
原始 CK 来源为 ROCm/composable_kernel（MIT）；所有原始版权声明保留。
