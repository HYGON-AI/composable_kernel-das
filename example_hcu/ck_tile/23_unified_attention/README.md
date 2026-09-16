<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Modified by Hygon Information Technology Co., Ltd. -->
# Unified Attention

本示例将 Unified Attention 集成为 CK Tile 公共算子，并分别生成 2D unified_attention 与
3D unified_attention 两个可执行文件。

## 目录结构

```text
include/ck_tile/ops/
├── unified_attention.hpp              # 唯一公共算子入口
└── unified_attention/
    ├── block/                         # 2D MMAC block GEMM
    ├── pipeline/                      # problem、policy 与 2D/3D pipeline
    └── kernel/                        # kernel、Kargs、pack、prepare、split/reduce

include/ck_tile/host/reference/
└── reference_unified_attention.hpp    # 独立 GPU reference

example_hcu/ck_tile/23_unified_attention/
├── CMakeLists.txt
├── build.sh
├── benchmark_unified_attention.sh
├── unified_attention_2d.cpp
├── unified_attention_3d.cpp
├── unified_attention_helper.hpp
└── instances/                              # D192/D256 实现与 FP16/BF16 显式实例
```

## 算子实现

### 2D Unified Attention

- 生成 token-major Q 和全局 paged K/V cache，并通过 block table 访问物理页。
- D192 包含 Q/K/V 数据重排、有效块索引生成和 MMAC 计算路径。
- D256 使用 prepare kernel 生成 active metadata 并完成 Q/K/V pack。
- 支持 FP16/BF16、causal、sliding window、attention sinks、ALiBi、ALiBi sqrt、
  QQ bias、softcap 和 PrefixLM range。

### 3D Unified Attention

- K/V 保存于全局 paged KV cache。
- attention kernel 按 sequence、KV head 和 softmax segment 发射，reduce kernel
  合并分段结果。
- sliding-window 可压缩无效 segment；sink 保留在第一个有效 partial 中参与完整
  running softmax。
- 支持 page size 16/32、head dim 192/256、FP16/BF16，以及与 2D 相同的
  mask/bias 功能开关。


## 编译

```bash
cd example_hcu/ck_tile/23_unified_attention
bash build.sh
```

生成：

```text
build/bin/tile_example_unified_attention_2d
build/bin/tile_example_unified_attention_3d
```


## 运行

```bash
./build/bin/tile_example_unified_attention_2d \
  -nq=49 -nkv=4096 -h=16 -hkv=2 -d=192 -prec=fp16 \
  -mask=sliding -sliding_window=128 -sinks=1

./build/bin/tile_example_unified_attention_2d \
  -nq=563 -nkv=4096 -h=16 -hkv=1 -d=192 -prec=fp16 \
  -query_lens=3,49,511 -kv_lens=4096,4096,4096

./build/bin/tile_example_unified_attention_3d \
  -b=4 -nkv=4096 -h=16 -hkv=2 -d=192 -block_size=16 \
  -segments=16 -prec=fp16 -alibi=1
```

## Benchmark 性能

- GPU：gfx936
- 精度：FP16/BF16

```bash
bash \
  example_hcu/ck_tile/23_unified_attention/benchmark_unified_attention.sh \
  -suite=all -warmup=10 -repeat=100 -v=0
```

| Case | NQ | NKV | H | HKV | Head Dim | Precision | Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| MIMO prefill GQA sink SWA | 49 | 4096 | 16 | 2 | 192 | FP16 | 0.0313655 |
| MIMO prefill MQA | 49 | 4096 | 16 | 1 | 192 | FP16 | 0.0844338 |
| MIMO prefill GQA sink SWA multibatch | 563 | 4096 | 16 | 2 | 192 | FP16 | 0.0604208 |
| MIMO prefill MQA multibatch | 563 | 4096 | 16 | 1 | 192 | FP16 | 0.428984 |
| MIMO decode GQA sink SWA 3D | 1 | 4096 | 16 | 2 | 192 | FP16 | 0.0164479 |
| MIMO decode MQA 3D | 1 | 4096 | 16 | 1 | 192 | FP16 | 0.0187504 |
| MIMO prefill GQA ALiBi | 49 | 4096 | 16 | 2 | 192 | FP16 | 0.122766 |
| MIMO prefill GQA ALiBi sqrt | 49 | 4096 | 16 | 2 | 192 | FP16 | 0.128275 |
| MIMO prefill GQA MM-prefix | 49 | 4096 | 16 | 2 | 192 | FP16 | 0.114981 |
| MIMO decode GQA ALiBi 3D | 1 | 4096 | 16 | 2 | 192 | FP16 | 0.0227296 |
| Qwen FP16 1seq 7392 | 7392 | 7392 | 8 | 1 | 256 | FP16 | 1.56897 |
| Qwen FP16 2seq 7392 | 7392 | 3696 | 8 | 1 | 256 | FP16 | 0.938255 |
| Qwen FP16 2seq 7264 | 7264 | 3632 | 8 | 1 | 256 | FP16 | 0.917868 |
| Qwen FP16 2seq 7395 | 7395 | 3698 | 8 | 1 | 256 | FP16 | 0.962027 |
| Qwen FP16 2seq 4227 | 4227 | 2114 | 8 | 1 | 256 | FP16 | 0.430808 |
| Qwen FP16 2seq 1987 | 1987 | 994 | 8 | 1 | 256 | FP16 | 0.133973 |
| Qwen FP16 2seq 6 | 6 | 3 | 8 | 1 | 256 | FP16 | 0.00948964 |
| Qwen FP16 1seq 3 | 3 | 3 | 8 | 1 | 256 | FP16 | 0.00923043 |
| Qwen BF16 1seq 8000 | 8000 | 8000 | 12 | 2 | 256 | BF16 | 2.54712 |
| Qwen BF16 1seq 7200 | 7200 | 7200 | 12 | 2 | 256 | BF16 | 2.13466 |
| Qwen BF16 1seq 800 | 800 | 800 | 12 | 2 | 256 | BF16 | 0.0933989 |
| Qwen BF16 1seq 3 | 3 | 3 | 12 | 2 | 256 | BF16 | 0.00924958 |
| Qwen FP16 2seq ALiBi | 1987 | 994 | 8 | 1 | 256 | FP16 | 0.136549 |
| Qwen BF16 1seq ALiBi sqrt | 800 | 800 | 12 | 2 | 256 | BF16 | 0.0987711 |
| Qwen BF16 1seq MM-prefix | 800 | 800 | 12 | 2 | 256 | BF16 | 0.198081 |

## 主要参数

| 参数 | 2D 默认值 | 3D 默认值 | 说明 |
| --- | ---: | ---: | --- |
| `-nq` | 64 | - | query token 数 |
| `-nkv` | 64 | 256 | KV 序列长度 |
| `-b` | - | 4 | Decode 序列数，每个序列一个 query token |
| `-h` | 16 | 16 | query head 数 |
| `-hkv` | 2 | 2 | KV head 数，要求 `h % hkv == 0` |
| `-d` | 192 | 192 | head dim，仅支持 192 或 256 |
| `-prec` | fp16 | fp16 | `fp16` 或 `bf16` |
| `-query_lens` | 空 | - | 逗号分隔的各序列 Q 长度；非空启用 multibatch |
| `-kv_lens` | 空 | - | 各序列 KV 长度；为空时均使用 `nkv` |
| `-block_size` | 64 | 16 | paged KV cache 的物理页大小；3D 仅支持 16/32 |
| `-kv_cache_blocks` | 0 | 0 | 物理 KV cache 页数；0 表示使用所需最小值 |
| `-block_table_width` | 0 | 0 | 每序列 block table 宽度；0 表示自动计算 |
| `-segments` | - | 4 | 3D 并行 softmax segment 数 |
| `-mask` | causal | causal | `causal`、`sliding`、`prefix` 或 `sliding_prefix` |
| `-sliding_window` | 0 | 0 | 左侧滑窗大小；0 表示关闭 |
| `-sinks` | 0 | 0 | 启用 per-head attention sinks |
| `-sink_value` | 0 | 0 | sink 初始值 |
| `-alibi` | 0 | 0 | 启用 per-head ALiBi slope |
| `-alibi_sqrt` | 0 | 0 | 使用 sqrt 距离 ALiBi |
| `-qq_bias` | 0 | 0 | 启用 query-query bias |
| `-qq_bias_value` | 0.01 | 0.01 | 随机测试使用的常量 QQ bias |
| `-softcap` | 0 | 0 | 正数启用 logit softcap |
| `-mm_prefix` | 0 | 0 | 启用 PrefixLM range |
| `-prefix_begin/end` | 0/-1 | 0/-1 | PrefixLM 闭区间；`-1` 表示末尾 |
| `-warmup/repeat` | 10/100 | 10/100 | 预热和计时次数 |
| `-v` | 1 | 1 | 1：执行独立 GPU reference 全量验证；0：跳过 |

## Migration provenance and validation

This feature is imported from OpenDAS MR !5, head `7ecbbb639840448f543eb4c5f6eb8085d6fcd0b1`.
Original copyright and license notices are retained. The local migration updates
Hygon modification notices and validates correctness on gfx936 and gfx938.
Historical benchmark data in this document are source results, not performance
acceptance results for this migration.

Run `bash example_hcu/ck_tile/23_unified_attention/test_unified_attention_correctness.sh <build>/bin`
to check 2D prefill and 3D decode, data types, head dimensions, page/segment options,
mask and bias features, and argument rejection. Use `BUILD_DIR`, `CXX`, and
`GPU_TARGETS` to override the build script defaults.

The kernels and GPU reference use infinity sentinels for masked logits and empty
softmax segments. Consumers using fast-math must retain infinity/NaN semantics
(e.g. `-ffast-math -fno-finite-math-only`, as in this example's CMake target).
The D256 tiny specialization requires at most 64 KV tokens; longer contexts use
the tiled path.

For D192 on gfx938, the score stage applies the complete elementwise visibility
predicate directly. The tile-level mask shortcuts produced noncausal results
with the validation toolchain; the gfx936 shortcuts remain available. This
correctness workaround has no performance acceptance claim.
