# FP32 / FP64 MMAC 接口回归

覆盖 `hcu_mmac.hpp` 的直接包装、`amd_xdlops.hpp` 的 16×16 MFMA 兼容入口及
`mmac_gemm.hpp::mmac_type` 三条调用路径。用于检查 aicc/dcc 的 builtin 参数和命名兼容。

每种类型、每条路径测试 6 组数据：零 A + 非零 C、基向量、K=4/8/20/64 的确定性随机数据，
共 36 例。比较完整 16×16 输出；FP64 输入含小于 FP32 精度的扰动。
矩阵契约是 A 行主序 [16,K]、B 列主序 [K,16]、C/D 列主序 [16,16]。
`vstep=0` 保持默认累加器布局，不启用 LTS。

直接构建（无需预先生成 CK config）：

```bash
mkdir -p hygon_tmp/mmac_fp32_fp64
hipcc -x hip --offload-arch=gfx936 -std=c++17 -O2 -Iinclude \
  example_hcu/26_mmac_fp32_fp64/mmac_fp32_fp64.cpp -o hygon_tmp/mmac_fp32_fp64/example
HIP_VISIBLE_DEVICES=0 hygon_tmp/mmac_fp32_fp64/example
```

可将 `hipcc` 替换为 `aicc`，或在 gfx938 机器将目标架构替换为 `gfx938`。
CMake 在 `GPU_TARGETS` 全部为 gfx936/gfx938 且启用 HCU examples 时注册
`example_mmac_fp32_fp64`；执行前选择空闲卡。验收为退出码 0，末行
`MMAC_FP32_FP64 PASS cases=36 failures=0`。

该测试证明上述 MMAC 调用的编译和数值正确性，不代表完整 PyTorch 构建或所有 CK 算子通过。
