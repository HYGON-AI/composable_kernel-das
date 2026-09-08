# Change Log for Composable Kernel

Full documentation for Composable Kernel is not yet available.

## CK 0.2.0 for ROCm 5.7.1

### 1. 功能改进
- 更新grouped gemm：覆盖 bf16、bf8、fp16、fp8、int4、int8 等实例路径，并改善 universal/grouped kernel 接口。
- 新增 batched gemm 第一版实现，包含 example_hcu/24_batched_gemm 示例和 host utility 支撑
- 新增 XDL GEMM v3 实现与 HCU 示例，覆盖普通 GEMM 和 multi-D GEMM
- 新增 FMHA forward 与 backward HCU/ck_tile 示例
- 新增 ck4inductor Python 包：支持 universal gemm、batched universal gemm、ck_tile universal gemm、grouped conv、HCU grouped conv 的实例生成与 TorchInductor 集成准备


### 2. 问题修复
- 修复新 AICC 下 rmsnorm/moe quant/int8 grouped gemm 等构建错误
- 修复 example_batchnorm_backward 正确性问题

### 3. 已知问题
- ck4inductor 当前是导入和集成起点，实际 TorchInductor 选择 CK kernel 仍依赖 shape、实例覆盖和 benchmark 结果；能编译/benchmark 不等于一定会被选中
