<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Modified by Hygon Information Technology Co., Ltd. -->
# HCU ck_tile GDN

本目录提供 Gated Delta Network（GDN）的独立 CK Tile 示例，包括 Prefill、
Decode、GPU reference 和 benchmark。当前支持 BF16、FP16，head dimension
固定为 128。

## 正确性回归

完成构建后，在选定的 HCU 卡上运行：

```bash
bash example_hcu/ck_tile/21_gdn/test_gdn_correctness.sh build/bin
```

脚本覆盖 FP16/BF16、固定长度和变长尾块、不同 head 比、状态和 gate
组合、较大输入幅值的中性 gate，以及非法参数拒绝。它校验 output 和
final state，使用独立 GPU reference，不放宽现有数值容差。

## 目录结构

```text
include/ck_tile/ops/
├── gdn.hpp                             # 唯一公共算子入口
└── gdn/
    ├── block/                          # 基础 block GEMM
    ├── pipeline/                       # problem、policy 与 pipeline 实现
    └── kernel/                         # 各阶段 kernel 与 Kargs 实现

include/ck_tile/host/reference/
└── reference_gdn.hpp                   # 独立 GPU reference

example_hcu/ck_tile/21_gdn/
├── CMakeLists.txt
├── README.md
├── build.sh
├── benchmark_gdn_cases.sh
├── gdn_prefill.cpp                     # Prefill 参数解析、计时与校验入口
├── gdn_decode.cpp                      # Decode 参数解析、计时与校验入口
├── prefill/                            # Prefill 各阶段 launch 与实例
│   ├── gdn_prefill_launch.hpp
│   ├── gdn_prefill_launch.cpp
│   ├── gdn_preprocess_instance.cpp     # 前处理与辅助 kernel (beta, indices, l2norm, cp)
│   ├── gdn_cumsum_instance.cpp
│   ├── gdn_kkt_instance.cpp
│   ├── gdn_recompute_instance.cpp
│   ├── gdn_state_instance.cpp
│   ├── gdn_output_instance.cpp
│   └── gdn_fwd_state_static.hpp
└── decode/                             # Decode launch 与实例
    ├── gdn_decode_launch.hpp
    └── gdn_decode_instance.cpp
```

## 算子实现

### Prefill

Prefill 将序列按 64 个 token 划分为 chunk，并按以下顺序执行：

```text
optional preprocess -> cumsum -> kkt_solve
                    -> recompute_w_u -> fwd_state -> fwd_output
```

- optional preprocess：按需执行 Q/K L2Norm、beta sigmoid、varlen chunk
  索引生成或 CP-context state 生成；
- `cumsum`：计算每个 chunk 的 gate 前缀和，并对非 64 对齐的尾部进行
  mask；
- `kkt_solve`：构造并求解 chunk 内的下三角 KKT 系统；
- `recompute_w_u`：根据 K、V、A、beta 和累计 gate 重计算 W/U；
- `fwd_state`：按 chunk 递推 recurrent state，生成 H、`v_new` 和可选的
  final state；
- `fwd_output`：根据 Q/K、H、`v_new` 和累计 gate 计算最终输出。

`use_gk=1` 使用公共 CK fused recurrent kernel 计算完整的 per-key decay，
该路径直接生成 output/final state，不生成 chunk 中间量 H/W/U/v_new，
且不支持与 `cp_context=1` 组合。此路径以功能正确性为目标。
`use_exp2=1` 表示输入 log gate 以 2 为底，`0` 表示自然对数；
chunk 路径内部统一使用 log2 累计 gate。`gate_in_kernel` 从 raw gate
计算自然对数衰减后再转换，`use_g=0` 在所有阶段使用中性 scalar gate。

#### CP-context

`-cp_context=1` 启用单进程 CP-context 功能模拟。`-cp_world_size` 指定模拟
rank 数，`-cp_rank` 指定当前 rank。程序构造
`[rank, HV, 128, 256]` all-gather 缓冲区：

- rank 0 保留本地 preprocess 产生的真实 `[H, M]` summary；
- 其他 rank 基于该 summary 生成非零、可复现且彼此不同的仿射 summary；
- 按 `H_next = H_rank + M_rank * H_prev` 合并当前 rank 之前的 summary；
- rank 0 没有前序 state，因此 initial state 为零。

该路径不创建真实进程，不调用 RCCL/ProcessGroup，也不模拟通信时延，仅用于
验证多 rank payload、内存布局和 merge 逻辑。`h`、`hv` 始终由命令行直接
指定。

### Decode

Decode 使用 fused recurrent kernel，在一个 kernel 中加载 Q/K/V、beta 和
gate，更新 128×128 recurrent state，并输出当前 token 结果和可选 final
state。fixed 模式支持多 batch；packed varlen 通过 `seq_endpoints` 隔离逻辑
序列，物理 batch 必须为 1。满足 fast-path 条件时自动使用专用实现，其余
组合使用通用 fused recurrent 路径。

## 构建

在仓库根目录执行：

```bash
bash example_hcu/ck_tile/21_gdn/build.sh
```

也可以执行：

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc \
  -DGPU_TARGETS=gfx936 \
  -DBUILD_DEV=ON \
  -DBUILD_EXAMPLE=ON \
  -DBUILD_TEST=OFF

cmake --build build \
  --target tile_example_gdn_prefill tile_example_gdn_decode \
  -j 8
```

生成文件：

```text
build/bin/tile_example_gdn_prefill
build/bin/tile_example_gdn_decode
```

## 运行

### Prefill

```bash
./build/bin/tile_example_gdn_prefill \
  -t=8192 -h=2 -hv=8 -head_dim=128 -prec=bf16 \
  -warmup=10 -repeat=100 -v=1
```

显式指定 head 数：

```bash
./build/bin/tile_example_gdn_prefill \
  -t=4096 -h=16 -hv=64 -head_dim=128 \
  -prec=fp16 -v=1
```

packed varlen：

```bash
./build/bin/tile_example_gdn_prefill \
  -seq_endpoints=2048,4096 -h=16 -hv=64 \
  -head_dim=128 -prec=bf16 -v=1
```

### Decode

```bash
./build/bin/tile_example_gdn_decode \
  -b=1 -t=1 -h=16 -hv=64 -head_dim=128 \
  -prec=bf16 -v=1
```

定长多 batch：

```bash
./build/bin/tile_example_gdn_decode \
  -b=4 -t=1 -h=16 -hv=64 -head_dim=128 \
  -prec=bf16 -v=1
```

packed varlen：

```bash
./build/bin/tile_example_gdn_decode \
  -b=1 -seq_endpoints=63,128 -h=16 -hv=64 \
  -head_dim=128 -prec=bf16 -v=1
```

varlen 使用 packed 布局，物理 batch 必须为 1。`seq_endpoints=63,128`
表示两条逻辑序列的长度分别为 63 和 65。

## Benchmark

下述性能记录来自 OpenDAS 原实现，仅供历史参考，不作为本次迁移性能结论。

运行Qwen TP case 和 branches case。

```bash
bash example_hcu/ck_tile/21_gdn/benchmark_gdn_cases.sh \
  --head-dim 128 \
  --warmup 10 \
  --repeat 100 \
  --validation 1
```

保存 CSV：

```bash
bash example_hcu/ck_tile/21_gdn/benchmark_gdn_cases.sh \
  --output-csv gdn_results.csv
```

### Performance

- GPU target：gfx936；
- Head dimension：128；
- Warmup：10，Repeat：100；
- 时间单位：us。

| Suite | Label | H | HV | T | Precision | Status | Time (us) |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| qwen_tp | tp1_T1 | 16 | 64 | 1 | BF16 | PASS | 56.8253 |
| qwen_tp | tp1_T16 | 16 | 64 | 16 | BF16 | PASS | 59.6918 |
| qwen_tp | tp1_T64 | 16 | 64 | 64 | BF16 | PASS | 78.0004 |
| qwen_tp | tp1_T256 | 16 | 64 | 256 | BF16 | PASS | 157.548 |
| qwen_tp | tp1_T1024 | 16 | 64 | 1024 | BF16 | PASS | 350.362 |
| qwen_tp | tp1_T2048 | 16 | 64 | 2048 | BF16 | PASS | 619.828 |
| qwen_tp | tp1_T4096 | 16 | 64 | 4096 | BF16 | PASS | 1185.14 |
| qwen_tp | tp1_T8192 | 16 | 64 | 8192 | BF16 | PASS | 2308.42 |
| qwen_tp | tp2_T1 | 8 | 32 | 1 | BF16 | PASS | 43.3457 |
| qwen_tp | tp2_T16 | 8 | 32 | 16 | BF16 | PASS | 46.4401 |
| qwen_tp | tp2_T64 | 8 | 32 | 64 | BF16 | PASS | 66.2156 |
| qwen_tp | tp2_T256 | 8 | 32 | 256 | BF16 | PASS | 106.346 |
| qwen_tp | tp2_T1024 | 8 | 32 | 1024 | BF16 | PASS | 223.118 |
| qwen_tp | tp2_T2048 | 8 | 32 | 2048 | BF16 | PASS | 373.994 |
| qwen_tp | tp2_T4096 | 8 | 32 | 4096 | BF16 | PASS | 675.16 |
| qwen_tp | tp2_T8192 | 8 | 32 | 8192 | BF16 | PASS | 1287.49 |
| qwen_tp | tp4_T1 | 4 | 16 | 1 | BF16 | PASS | 41.6035 |
| qwen_tp | tp4_T16 | 4 | 16 | 16 | BF16 | PASS | 43.941 |
| qwen_tp | tp4_T64 | 4 | 16 | 64 | BF16 | PASS | 54.6593 |
| qwen_tp | tp4_T256 | 4 | 16 | 256 | BF16 | PASS | 81.1045 |
| qwen_tp | tp4_T1024 | 4 | 16 | 1024 | BF16 | PASS | 161.208 |
| qwen_tp | tp4_T2048 | 4 | 16 | 2048 | BF16 | PASS | 259.779 |
| qwen_tp | tp4_T4096 | 4 | 16 | 4096 | BF16 | PASS | 400.644 |
| qwen_tp | tp4_T8192 | 4 | 16 | 8192 | BF16 | PASS | 699.013 |
| qwen_tp | tp8_T1 | 2 | 8 | 1 | BF16 | PASS | 40.7473 |
| qwen_tp | tp8_T16 | 2 | 8 | 16 | BF16 | PASS | 43.31 |
| qwen_tp | tp8_T64 | 2 | 8 | 64 | BF16 | PASS | 54.4722 |
| qwen_tp | tp8_T256 | 2 | 8 | 256 | BF16 | PASS | 65.0915 |
| qwen_tp | tp8_T1024 | 2 | 8 | 1024 | BF16 | PASS | 142.625 |
| qwen_tp | tp8_T2048 | 2 | 8 | 2048 | BF16 | PASS | 207.139 |
| qwen_tp | tp8_T4096 | 2 | 8 | 4096 | BF16 | PASS | 299.694 |
| qwen_tp | tp8_T8192 | 2 | 8 | 8192 | BF16 | PASS | 457.401 |
| branches | baseline_gva_16_64_1x4096 | 16 | 64 | 4096 | BF16 | PASS | 1186.24 |
| branches | dtype_fp16_gva_16_64 | 16 | 64 | 4096 | FP16 | PASS | 1486.26 |
| branches | dtype_bf16_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1181.48 |
| branches | no_init_state_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1181.19 |
| branches | no_final_state_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1179.43 |
| branches | neutral_gates_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1187.38 |
| branches | l2norm_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1325.71 |
| branches | beta_sigmoid_kernel_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1191.27 |
| branches | gate_in_kernel_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1184.19 |
| branches | scale_1.0_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1187.17 |
| branches | varlen_explicit_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1264.45 |
| branches | varlen_2048+2048_gva_16_64 | 16 | 64 | 4096 | BF16 | PASS | 1284.7 |
| branches | heads_gva_16_32 | 16 | 32 | 4096 | BF16 | PASS | 699.315 |
| branches | heads_sym_16_16 | 16 | 16 | 4096 | BF16 | PASS | 439.692 |
| branches | heads_sym_32_32 | 32 | 32 | 4096 | BF16 | PASS | 765.927 |
| branches | non_aligned_T63_gva_16_64 | 16 | 64 | 63 | BF16 | PASS | 96.522 |
| branches | non_aligned_T127_gva_16_64 | 16 | 64 | 127 | BF16 | PASS | 121.57 |
| branches | non_aligned_T511_gva_16_64 | 16 | 64 | 511 | BF16 | PASS | 245.225 |

## 主要参数

`-` 表示该可执行文件不支持此参数。

| 参数 | 说明 | Prefill 默认值 | Decode 默认值 |
| --- | --- | ---: | ---: |
| `-b` | 物理 batch；Decode varlen 要求为 1 | - | `1` |
| `-t` | Prefill 总 token 数；Decode 为每 batch token 数 | `1024` | `1` |
| `-h` | Q/K head 数 | `16` | `16` |
| `-hv` | Value head 数 | `64` | `64` |
| `-head_dim` | Head dimension，当前只支持 128 | `128` | `128` |
| `-prec` | 输入精度：`bf16` 或 `fp16` | `bf16` | `bf16` |
| `-has_initial_state` | Prefill 是否读取 initial state | `1` | - |
| `-use_initial_state` | Decode 是否读取 initial state | - | `1` |
| `-store_final_state` | 是否保存 final state | `1` | `1` |
| `-save_new_value` | 是否保存 Prefill 的 `v_new` | `1` | - |
| `-use_g` | 是否使用 scalar gate | `1` | `1` |
| `-use_gk` | 是否使用 per-key gate | `0` | `0` |
| `-use_gv` | 是否使用 per-value gate | - | `0` |
| `-beta_headwise` | Decode 是否每个 value head 使用一个 beta | - | `1` |
| `-use_exp2` | 是否使用 log2 gate 表示 | `1` | `0` |
| `-transpose_state` | 是否使用转置 state 布局 | `0` | `0` |
| `-qk_l2norm` | Prefill 是否在执行前归一化 Q/K | `0` | - |
| `-use_qk_l2norm` | Decode 是否在 fused kernel 内归一化 Q/K | - | `0` |
| `-gates` | Prefill 是否生成随机负 log gate；0 使用零 gate | `1` | - |
| `-gate_in_kernel` | 是否在 kernel 内根据 `A_log` 生成 gate | `0` | `0` |
| `-has_dt_bias` | gate-in-kernel 路径是否增加 `dt_bias` | `1` | `0` |
| `-beta_sigmoid_in_kernel` | Prefill 是否在 GPU 上计算 beta sigmoid | `0` | - |
| `-is_varlen` | 是否启用 Prefill packed varlen | `0` | - |
| `-variable_length` | 是否启用 Decode packed varlen | - | `0` |
| `-seq_endpoints` | packed varlen 的累计序列结束位置 | 空 | 空 |
| `-cp_context` | 是否执行功能性 CP-context 路径 | `0` | - |
| `-cp_world_size` | 单进程模拟的 CP rank 数，范围 1–8 | `2` | - |
| `-cp_rank` | 当前模拟 rank，范围 0–`cp_world_size-1` | `1` | - |
| `-input_scale` | 随机 Q/K/V 和 state 的生成范围系数 | `0.1` | - |
| `-scale` | Q scale；0 表示 `1/sqrt(head_dim)` | `0` | - |
| `-warmup` | Warmup 次数 | `10` | `10` |
| `-repeat` | 计时次数 | `100` | `100` |
| `-v` | 是否运行 GPU reference | `1` | `1` |
| `-seed` | 随机数种子 | `42` | `42` |

## 当前限制

- head dimension 固定为 128；
- Prefill 物理 batch 固定为 1；
- Decode varlen 的物理 batch 固定为 1；
- CP-context 是功能性实现，尚未优化，性能较差；

## 来源与版权

本实现基于 OpenDAS/composable_kernel 的 GDN 实现 `2938dbb`，并包含
`1c15e2b` 的配置维度修正（合入节点 `aee604b`）。原始 CK 来源为
ROCm/composable_kernel（MIT）；原始版权声明及 LICENSE 保留。
Modified by Hygon Information Technology Co., Ltd.

本次迁移补齐 FP16 定长部分 chunk 的边界处理，并使 `use_g=0` 在
cumsum、KKT、recompute、state、output 各阶段采用一致的中性 scalar gate。
完整 chunk 保留原有 CK Tile 路径，FP16 部分 chunk 复用原 varlen 的
边界计算 kernel；未放宽 GPU reference 容差。

BF16 长序列的内部状态分组 prefix GEMM 显式等待异步 global-to-LDS
加载完成，再进行 workgroup 同步。正确性脚本覆盖 T=4096/8192 的
重复运行、初始状态开关和多个随机种子，以检查分组边界稳定性。
