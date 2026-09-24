<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->
<!-- SPDX-License-Identifier: MIT -->
# HCU ck_tile SLA 稀疏线性注意力

本目录提供 SLA (Sparse Linear Attention) 的独立 ck_tile 示例，包含 FWD、BWD 以及 GPU reference。当前支持 BF16、FP16、head dimension 128 和 64×64 (对于 S=75648 等) / 128×64 (对于 S=18048) 稀疏 block。

## 目录结构

```text
include/ck_tile/ops/
├── sla.hpp                             # 唯一公共算子入口
└── sla/
    ├── block/                          # 基础 block GEMM
    ├── pipeline/                       # problem、policy 与 pipeline 实现
    └── kernel/                         # 各阶段 kernel 与 Kargs 实现

include/ck_tile/host/reference/
└── reference_sla.hpp                   # 独立 GPU reference

example_hcu/ck_tile/22_sla/
├── CMakeLists.txt
├── README.md
├── build.sh
├── instances/                          # 显式实例编译单元
│   ├── sla_fwd_bf16_instance.cpp
│   ├── sla_fwd_fp16_instance.cpp
│   ├── sla_bwd_bf16_instance.cpp
│   └── sla_bwd_fp16_instance.cpp
├── sla_fwd.cpp                         # FWD 参数解析、调用入口与 main
├── sla_fwd.hpp                         # FWD 接口声明
├── sla_fwd_runner_impl.hpp             # FWD Runner 与 Linear Attention Launch 实现
├── sla_bwd.cpp                         # BWD 参数解析、调用入口与 main
├── sla_bwd.hpp                         # BWD 接口声明
├── sla_bwd_launch.hpp                  # BWD preprocess、dQ、dK/dV 统一 Launch 实现
└── sla_sparse_map.hpp                  # sparse map 元数据构建 Launch 实现
```

## 算子实现

### FWD 实现

- 使用 `sparse_map` 构建 Q/K block-wise 的 sparse map 与前向查找表 (LUT)；
- 包含 Sparse Attention 与 Linear Attention 两条分支：
  - **Sparse Attention**：通过 problem、policy、pipeline 和 kernel 分层实现稀疏注意力，仅计算 LUT 选中的 K/V block；
  - **Linear Attention**：对 Q/K/V 进行融合线性注意力计算；
- 支持多 stage (kv_stages) 计算及最后的 stage reduce 规约操作；
- 输出融合后的最终结果。

### BWD 实现

- 依然先构建 `sparse_map`，并生成前向 LUT 和反向查找表 (RLUT) 以及 permutation 元数据；
- preprocess 阶段计算 delta；
- dQ 使用前向 LUT 计算 Q 的梯度，长序列/大 topk 使用多 stage (dq_stages) 计算并执行最终的 reduce；
- dK/dV使用反向 LUT (RLUT) 计算 K/V 的梯度；
- 输出 dQ、dK/dV 梯度结果。

## 构建

在仓库根目录执行：

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc \
  -DGPU_TARGETS=gfx936 \
  -DBUILD_DEV=ON \
  -DBUILD_EXAMPLE=ON \
  -DBUILD_TEST=OFF \
  -DSLA_MATCH_EXTENSION_FLAGS=ON

cmake --build build \
  --target tile_example_sla_fwd tile_example_sla_bwd \
  -j 8
```

也可以直接运行构建脚本：

```bash
bash example_hcu/ck_tile/22_sla/build.sh
```

生成文件位于：

```text
build/bin/tile_example_sla_fwd
build/bin/tile_example_sla_bwd
```

## 运行

### FWD

```bash
./build/bin/tile_example_sla_fwd \
  -b=1 -h=40 -s=18048 -d=128 \
  -topk_ratio=0.1 -prec=bf16 -linear_attn=1 \
  -v=1
```

### BWD

```bash
./build/bin/tile_example_sla_bwd \
  -b=1 -h=40 -s=18048 -d=128 \
  -topk_ratio=0.1 -prec=bf16 \
  -v=1
```

针对 S=75648 的典型测试命令：
```bash
./build/bin/tile_example_sla_bwd \
  -b=1 -h=40 -s=75648 -d=128 \
  -topk_ratio=0.1 -prec=bf16 \
  -v=1
```

## 历史性能记录

以下数据来自 OpenDAS 原实现，未在本次迁移中复测，不代表当前提交性能。

- GPU target：gfx936 (BW151)
- Precision：BF16
- topk_ratio：0.1

| Kernel | Batch Size | Heads | Head Dim | S | M (BlockM) | N_blk (BlockN) | topk_ratio | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FWD | 1 | 40 | 128 | 18048 | 128 | 64 | 0.1 | 4.24 ms |
| FWD | 1 | 40 | 128 | 75648 | 64 | 64 | 0.1 | 83.97 ms |
| BWD | 1 | 40 | 128 | 18048 | 128 | 64 | 0.1 | 18.62 ms |
| BWD | 1 | 40 | 128 | 75648 | 64 | 64 | 0.1 | 330.32 ms |

## 命令行参数

| 参数 | 说明 | FWD 默认值 | BWD 默认值 |
| --- | --- | ---: | ---: |
| `-b` | Batch size | `1` | `1` |
| `-h` | Head 数量 | `1` | `1` |
| `-s` | 序列长度，必须为 64 的整数倍 | `1024` | `1024` |
| `-d` | Head dimension，当前只支持 128 | `128` | `128` |
| `-prec` | 数据类型：`bf16` 或 `fp16` | `bf16` | `bf16` |
| `-topk_ratio` | sparse_map 选取的 topk 比例 | `0.1` | `0.1` |
| `-kv_stages` | FWD 或 BWD dQ 阶段的 KV stage 数；`0` 表示自动选择 | `0` | `0` |
| `-linear_attn` | 是否在 FWD 中运行 linear_attn 分支 | `1` | — |
| `-v` | 是否运行 GPU reference 验证正确性 | `1` | `1` |
| `-warmup` | Warmup 次数 | `10` | `10` |
| `-repeat` | 计时重复次数 | `100` | `100` |
| `-seed` | 随机输入种子 | `1` | `1` |


## 来源

迁移自 OpenDAS Composable Kernel MR !4（`f97100715024afbc09884ab5d7c4f3f2a12e2b58`），
保留原许可证和版权声明，并按本仓库规范补充海光修改声明。
BWD 示例验证 sparse attention 梯度；不提供融合 linear 分支的 backward。
