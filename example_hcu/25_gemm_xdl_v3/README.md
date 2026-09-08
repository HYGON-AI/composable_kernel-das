# GEMM XDL CShuffle V3 HCU 示例

本目录用于在海光 HCU 上验证从 upstream CK 移植过来的 XDL CShuffle v3 device 路径：

- `example_hcu_gemm_xdl_bf16_v3`: `DeviceGemm_Xdl_CShuffleV3`
- `example_hcu_gemm_multi_d_xdl_bf16_v3`: `DeviceGemmMultiD_Xdl_CShuffle_V3`

## 编译

在已配置好的工程构建目录中执行：

```bash
cmake --build build_dev --target example_hcu_gemm_xdl_v3 -j 8
```

也可以分别编译两个可执行文件：

```bash
cmake --build build_dev --target example_hcu_gemm_xdl_bf16_v3 -j 8
cmake --build build_dev --target example_hcu_gemm_multi_d_xdl_bf16_v3 -j 8
```

## 运行

在 `build_dev/bin` 目录下运行。

普通 GEMM v3：

```bash
./example_hcu_gemm_xdl_bf16_v3
./example_hcu_gemm_xdl_bf16_v3 1 1 0
./example_hcu_gemm_xdl_bf16_v3 1 1 0 256 256 128 128 128 256 1
```

MultiD GEMM v3：

```bash
./example_hcu_gemm_multi_d_xdl_bf16_v3
./example_hcu_gemm_multi_d_xdl_bf16_v3 1 1 0
./example_hcu_gemm_multi_d_xdl_bf16_v3 1 1 0 3840 4096 4096 4096 4096 4096 1
```

`example_hcu_gemm_xdl_bf16_v3` 参数格式：

```text
arg1: 是否校验结果 (0=否, 1=是)
arg2: 初始化方式 (0=不初始化, 1=整数随机值, 2=小数随机值)
arg3: 是否计时 kernel (0=否, 1=是)
arg4 到 arg10: M, N, K, StrideA, StrideB, StrideC, KBatch
```

`example_hcu_gemm_multi_d_xdl_bf16_v3` 参数格式：

```text
arg1: 是否校验结果 (0=否, 1=是)
arg2: 初始化方式 (0=不初始化, 1=整数随机值, 2=小数随机值)
arg3: 是否计时 kernel (0=否, 1=是)
arg4 到 arg10: M, N, K, StrideA, StrideB, StrideE, KBatch
```

## 诊断

设置 `CK_LOGGING=1` 后，如果参数被 `IsSupportedArgument` 或 gridwise 合法性检查拒绝，会打印具体拒绝原因：

```bash
CK_LOGGING=1 ./example_hcu_gemm_multi_d_xdl_bf16_v3 1 1 0 128 128 32 32 32 128 1
```

对于 v3 pipeline 示例，如果 `K` 太小，K 循环次数不大于 pipeline 预取 stage 数，参数会被拒绝。
验证 v3 pipeline 时建议使用默认 MultiD shape，或其他具有足够 K-loop 深度的 shape。
