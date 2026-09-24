# CK GDN 算子使用与性能

本文说明 CK Tile Gated DeltaNet（GDN）的 Prefill、Decode 接口，以及 gfx936/gfx938 的实测性能。支持 BF16、FP16，当前示例的 key/value dimension 均为 128；state 与主要累加使用 FP32，BF16 转换采用最近偶数舍入（RNE）。本文不包含优化过程记录，性能数据的测量版本和范围单独标明。

## 1. 功能与接口层次

GDN 是带门控的递归注意力。对于单个 value head，以 KV state `S[K,V]` 表示：先用 gate 衰减 state，再计算 `delta = beta * (v - k^T S)`，更新 `S += k delta^T`，最后输出 `scale * q^T S`。Prefill 通过 chunk 分解实现同类递推；Decode 使用融合递推 kernel。这里不包含模型的 QKV 投影、卷积、输出投影或推理框架的状态池管理。

|接口层次|入口|用途|
|---|---|---|
|CK Tile 头文件|[`ck_tile/ops/gdn.hpp`](../../include/ck_tile/ops/gdn.hpp)|各阶段 problem/policy/kernel；C++ 模板接口|
|完整 Prefill launcher|[`PrefillArguments`、`launch_prefill_bf16/fp16`](../../example_hcu/ck_tile/21_gdn/prefill/gdn_prefill_launch.hpp)|示例侧封装：预处理及完整 chunk 链路|
|Decode launcher|[`launch_bf16/fp16`、`launch_bf16_raw_beta/fp16_raw_beta`](../../example_hcu/ck_tile/21_gdn/decode/gdn_decode_launch.hpp)|选择通用或专用 fused recurrent kernel|
|Decode 参数|[`GdnFusedRecurrentKargs`](../../include/ck_tile/ops/gdn/kernel/gdn_ck_fused_recurrent_kernel.hpp)|连续张量、gate、state、序列元数据与 dtype 标志|
|独立程序|`tile_example_gdn_prefill`、`tile_example_gdn_decode`|参数校验、设备内存分配、GPU 计时和独立 reference|

完整 launcher 定义在 `example_hcu/ck_tile/21_gdn`，不是已发布的稳定 C ABI 或 Python/Torch 接口。使用 launcher 时需编译并链接对应 `.cpp`，仅包含头文件不能链接成功。它们接受设备指针和 `hipStream_t`，异步提交 GPU 工作，调用方负责分配内存、保证参数合法及 stream/内存生命周期。完整可运行的分配示例见 [`gdn_prefill.cpp`](../../example_hcu/ck_tile/21_gdn/gdn_prefill.cpp) 与 [`gdn_decode.cpp`](../../example_hcu/ck_tile/21_gdn/gdn_decode.cpp)。

## 2. 构建与命令行

在 CK 仓库根目录执行，使用已安装 DTK 的 `aicc`（本文两种架构性能实测所用的编译器）；目标可改成 `gfx938`：

```bash
cmake -S . -B build-gdn \
  -DCMAKE_CXX_COMPILER="$(command -v aicc)" \
  -DGPU_TARGETS=gfx936 \
  -DBUILD_DEV=ON -DBUILD_EXAMPLE=ON -DBUILD_TEST=OFF
cmake --build build-gdn \
  --target tile_example_gdn_prefill tile_example_gdn_decode -j 8

# 完整 Prefill；明确输入 gate 的对数底数。
./build-gdn/bin/tile_example_gdn_prefill \
  -t=1024 -h=2 -hv=8 -head_dim=128 -prec=bf16 \
  -use_exp2=0 -warmup=10 -repeat=100 -v=1

# 固定长度 Decode；transpose_state=1 表示 VK state。
./build-gdn/bin/tile_example_gdn_decode \
  -b=8 -t=1 -h=2 -hv=8 -head_dim=128 -prec=bf16 \
  -transpose_state=1 -warmup=10 -repeat=100 -v=1

# 两条 packed 序列，长度分别为 63、65。
./build-gdn/bin/tile_example_gdn_prefill \
  -seq_endpoints=63,128 -h=2 -hv=8 -prec=fp16 -v=1

bash example_hcu/ck_tile/21_gdn/test_gdn_correctness.sh build-gdn/bin
```

`HIP_VISIBLE_DEVICES` 可选择设备。`-v=1` 检查 output 和 final state；原生脚本包含 190 项正反例。示例程序的 HIP-event 计时与后文 GPU Graph 测量方法不同，不应逐值等同。Decode CLI 默认 beta 已处理，且其 gate 测试数据类型不等于后文 raw-gate 性能适配器；不能直接把默认 CLI 输出当作表中同一工作负载。

## 3. Prefill 输入、工作区与调用

记 `T` 为总 token 数、`H` 为 Q/K head 数、`HV` 为 value head 数、`D=128`、`S` 为逻辑序列数，`C=sum(ceil(length_s/64))`。输入和输出均按表中顺序连续存储；`HV/H` 支持 1、2、4，本文验证覆盖的 `H` 为 2 次幂。固定模式是一条序列；多序列使用 packed varlen。

|成员|dtype|形状 / 元素数|说明|
|---|---|---|---|
|`q`, `k`|BF16 或 FP16|`[T,H,D]`|二者与 V 使用同一 dtype|
|`v`, `output`|同输入 dtype|`[T,HV,D]`|输出参数存储类型为 `uint16_t*`，表示16位元素载荷|
|`g`, `beta`|FP32|`[T,HV]`|逐 token log gate、已 sigmoid 的 beta|
|`initial_state`, `final_state`|FP32|KV:`[S,HV,K,V]`；VK:`[S,HV,V,K]`|K=V=128，但逻辑布局不同|
|`g_cum`|FP32|`T*HV`|内部 chunk 前缀和，统一为 log2|
|`a`|同输入 dtype|`T*HV*64`|KKT 求逆中间结果|
|`w`, `u`, `v_new`|同输入 dtype|各 `T*HV*D`|中间量；按16位元素分配|
|`h`|同输入 dtype|`C*HV*D*D`|内部 state 快照；部分分派使用 packed 布局，不能一律当作逻辑 KV|
|`cu_seqlens`|int64|`S+1`|varlen 时使用，首项0，递增，末项T|
|`chunk_offsets` / `chunk_indices`|int64|`S+1` / `2*C`|varlen 设备工作区，由索引 kernel 生成|

以下是已分配设备缓冲区后的 BF16 固定长度调用示意。`q_dev` 等指针需按上表分配；使用 FP16 时改 launcher 和 `is_bf16`，不是转换指针即可改变数据类型。

```cpp
#include "prefill/gdn_prefill_launch.hpp"
#include <cmath>

gdn_example::PrefillArguments a{};
a.q = q_dev; a.k = k_dev; a.v = v_dev;
a.g = log_gate_dev; a.beta = sigmoid_beta_dev;
a.initial_state = initial_kv_dev;
a.final_state = final_kv_dev;
a.output = output_dev;
a.g_cum = g_cum_dev;
a.a = inverse_dev; a.w = w_dev; a.u = u_dev;
a.h = h_dev; a.v_new = v_new_dev;
a.t = T; a.h_qk = H; a.h_v = HV; a.head_dim = 128;
a.num_sequences = 1; a.num_chunks = (T + 63) / 64;
a.scale = 1.0f / std::sqrt(128.0f);
a.is_bf16 = true;
a.state_has_initial_state = true;
a.output_final_state = a.state_store_final_state = true;
a.state_save_new_value = a.state_use_g = true;
a.state_use_exp2 = false;  // 输入是自然对数 log gate
gdn_example::launch_prefill_bf16(a, stream);
```

完整链为 `cumsum → KKT → W/U → H → O`，另有可选预处理。需要注意：

- `state_use_exp2=false` 表示自然对数 gate，`true` 表示 log2；CLI Prefill 默认 `true`，性能表使用自然对数输入。`state_use_g=false` 使用中性 scalar gate。
- `gate_in_kernel=true` 时 `g` 是 raw gate，并提供 FP32 `a_log[HV]`；有 bias 时提供 FP32 `dt_bias[HV]`。衰减的自然对数为 `-exp(a_log)*softplus(g+dt_bias)`。
- `beta_sigmoid_in_kernel=true` 需提供 `beta_input[T,HV]`、`beta_processed[T,HV]`，并将 `beta` 指向 `beta_processed`。launcher 不会自动替调用方重定向指针。
- `use_qk_l2norm=true` 需提供 `q_input/k_input`、`q_norm/k_norm[T,H,D]`、FP32 `q_rstd/k_rstd[T,H]`，并将 `q/k` 指向归一化输出缓冲区。
- 无初态时将 `state_has_initial_state=false`；不写末态时同时关闭 `state_store_final_state` 和 `output_final_state`。`state_save_new_value=false` 不执行常规 O 阶段，不能当作完整 Prefill 输出接口。
- VK state 使用 `state_transpose_state=true`，另分配 FP32 `state_input_workspace` / `state_final_workspace`，各为 `S*HV*D*D` 个元素。当前 Prefill launcher 的布局转换在 GPU 链内执行；完整链比较时要计入。
- packed varlen 设置 `is_varlen/state_is_varlen`，提供上述三组元数据和正确的 `S/C`；不同逻辑序列的 state 独立。
- `state_use_gk=true` 提供 FP32 `gk[T,HV,D]`，使用通用融合递推路径，不产生常规 H/W/U/v_new；不支持与 `cp_context` 组合。
- `state_cp_groups=0` 可关闭内部 chunk 分组。自动选择仅在示例检查的 BF16、定长、整块、scalar gate 等条件下使用，还需分配 `state_cp_group_a/b`（同输入dtype）与 FP32 `state_cp_group_start`，三者各为 `state_cp_groups*HV*D*D` 个元素。它不同于 `cp_context`：后者仅是单进程多 rank summary 模拟，没有 RCCL/ProcessGroup 通信，不代表分布式推理性能。

独立 `launch_cumsum`、`launch_kkt_*`、`launch_recompute_*`、`launch_state_*`、`launch_output_*` 可用于集成与分阶段诊断；必须沿用相同参数、log2 中间 gate 和 H 布局，不能把任意逻辑 H 直接交给自动选择 packed H 的输出 launcher。

## 4. Decode 输入与调用

固定模式 Q/K 为 `[B,T,H,D]`，V/output 为 `[B,T,HV,D]`；varlen 为物理 `B=1` 的 packed token 数组，设置 `sequences=S` 并提供 `cu_seqlens[S+1]`。虽然命名 Decode，通用递推支持多 token。state 为 FP32，`transpose_state=false` 对应 KV，`true` 对应 VK；此标志描述调用方实际存储布局，不是要求内核先做一次转置。

`GdnFusedRecurrentKargs` 的 gate/beta 是 `void*`，须同时设置 dtype：`Float16=0`、`BFloat16=1`、`Float32=2`。Q/K/V 和 output 的类型由所选 launcher 决定。

|成员|连续形状|含义|
|---|---|---|
|`g`|`[B,T,HV]`|log gate 或 raw gate|
|`gk` / `gv`|`[B,T,HV,K]` / `[B,T,HV,V]`|可选逐 key / value log gate|
|`beta`|headwise:`[B,T,HV]`；否则:`[B,T,HV,V]`|普通 launcher 使用已处理的 beta|
|`a_log` / `dt_bias`|`[HV]`|raw gate 参数|
|`initial_state/final_state`|`[S,HV,K,V]` 或 `[S,HV,V,K]`|定长时S=B；packed时S为逻辑序列数|

对 raw FP32 beta 使用 `launch_bf16_raw_beta(args, raw_beta, stream)` 或 FP16 对应接口，将 sigmoid 融入递推。不要先 sigmoid 后再次调用 raw-beta 入口。后文 Decode 性能即包含 raw gate、beta sigmoid 与 final-state 写回；norm 行还包含 QK L2Norm。

较快 VK/raw-gate 分派要求 K=V=128、scalar gate、headwise beta、初态及末态均启用；`g/dt_bias` 与输入使用相同 BF16/FP16 类型，`beta/a_log` 为 FP32，`gate_in_kernel/has_dt_bias=true`，`use_exp2/use_gk/use_gv=false`。其他合法组合保留通用路径，性能不能外推为相同。

```cpp
#include "decode/gdn_decode_launch.hpp"
#include <cmath>

GdnFusedRecurrentKargs a{};
a.q = q_dev; a.k = k_dev; a.v = v_dev;
a.g = raw_gate_bf16_dev;
a.a_log = a_log_fp32_dev; a.dt_bias = bias_bf16_dev;
a.initial_state = initial_vk_dev; a.final_state = final_vk_dev;
a.output = output_dev;
a.batch = a.sequences = B; a.time = 1;
a.qk_heads = H; a.value_heads = HV;
a.key_dim = a.value_dim = 128;
a.scale = 1.0f / std::sqrt(128.0f);
a.g_dtype = a.dt_bias_dtype = gdn_tensor_dtype_code(GdnTensorDtype::BFloat16);
a.a_log_dtype = a.beta_dtype = gdn_tensor_dtype_code(GdnTensorDtype::Float32);
a.transpose_state = a.use_g = a.beta_headwise = true;
a.use_initial_state = a.store_final_state = true;
a.gate_in_kernel = a.has_dt_bias = true;
gdn_decode_example::launch_bf16_raw_beta(a, raw_beta_fp32_dev, stream);
```

## 5. 性能测量条件

以下均为2026-09-23保存的正式测量，单位 **μs，越小越好**。2026-09-24的死代码清理不产生新的性能数字。各架构使用各自同设备对照，不能用两表的绝对耗时之差推断纯硬件代际增益。

|配置|gfx936|gfx938|
|---|---|---|
|设备报告型号|BW200 / UBB BW1000|BW1101|
|CU数|80|64|
|显存|约64 GiB|约144 GiB|
|L2|8 MiB|8 MiB|
|PyTorch / HIP|2.10.0 / 6.3.26113|2.10.0 / 6.3.26113|
|编译器|DTK aicc，Clang 18.0.0|DTK aicc，Clang 18.0.0|
|vLLM FLA来源版本|`0.18.1+das.dtk2604.torch2100.2606041655.ge5a62f`|`0.21.0+das.dtk2604.torch2100.2606111143.g8c979d`|
|CK性能快照|`2e1414185bdc6aa5c6386b012626ec78d5ad3d7e` 对应已验收源|Prefill:`ab2c12d1bedbfd326c79fef7756fac241cefb999`；Decode:`ffaa2f2e425e1a7fdda1357102f1f020cda14a7f`|

Prefill：B=1、D=128、chunk=64，输入为相同Q/K/V、逐token自然对数gate、sigmoid后beta和固定初态；计时包含cumsum/KKT/WU/H/O及FP32 final state。关闭raw-gate预处理和QK归一化。两种库装载顺序各7轮，GPU Graph每图16次调用，14样本等权中位数。不是独立阶段时间相加。

Decode：T=1、D=128，7轮GPU Graph计时，包含raw gate、beta sigmoid、output/final state；表中同时保留CK的KV和VK时间。比较使用VK，要求调用方直接提供VK布局，输入布局转换不在计时内。Prefill采用各实现约定的合法原生输入布局。JIT、host分配和Python调用开销不计入；计时重用固定初态，不把输出不断回灌。

### AITER 两列具体表示什么

- **Prefill HIP链**：AITER HIP `chunk_gated_delta_rule_fwd_kkt_solve_hip`、`chunk_gated_delta_rule_fwd_vllm_hip_blockdim64`、`chunk_fwd_o_vllm_hip_blockdim64`，加vLLM FLA Triton的cumsum/WU。它不是五阶段全部为HIP。
- **Prefill Triton链**：AITER Triton的KKT/H/O，加相同vLLM FLA cumsum/WU。两条链各自产生中间量，不使用CK生成的W/U或H。
- **Decode HIP/Triton**：分别使用 `vllm_fused_sigmoid_gating_delta_rule_update` 与 `aiter.ops.triton.fla.fused_sigmoid_gating.fused_sigmoid_gating_delta_rule_update` 的适配器，在相同raw gate/beta、状态布局及输出契约下比较。这里不包含框架state-pool索引、packed-QKV拆分或投机解码管理。
- 固定fixture最终output/state通过原检查；部分AITER HIP KKT附加中间诊断不通过，包含T65的重复性审计边界。下表保留全部原始HIP耗时，不通过剔除较快结果宣称全面领先。

性能分类以 `CK/min(HIP,Triton)` 计：小于0.95为领先，大于1.05为落后，其余接近。表中加速比为 `min(HIP,Triton)/CK`；使用同轮原始时间，不跨轮次挑选最快样本。

|范围|配置数|领先|接近|落后|
|---|---:|---:|---:|---:|
|gfx936 完整Prefill|21|21|0|0|
|gfx936 Decode（VK）|17|5|12|0|
|gfx938 完整Prefill|21|13|4|4|
|gfx938 Decode（VK）|17|15|2|0|

## 6. gfx936 完整 Prefill

|dtype|T|H/HV|CK|AITER HIP链|AITER Triton链|对最快加速比|结论|
|---|---:|---|---:|---:|---:|---:|---|
|bfloat16|64|2/8|45.226|64.162|65.589|1.419×|领先|
|bfloat16|256|2/8|54.870|75.172|76.343|1.370×|领先|
|bfloat16|1024|2/8|117.080|130.395|132.514|1.114×|领先|
|bfloat16|4096|2/8|297.707|342.630|396.358|1.151×|领先|
|bfloat16|64|16/64|73.578|100.671|141.854|1.368×|领先|
|bfloat16|256|16/64|141.586|175.534|325.813|1.240×|领先|
|bfloat16|1024|16/64|393.582|499.571|1039.777|1.269×|领先|
|bfloat16|4096|16/64|1346.408|1937.302|4069.416|1.439×|领先|
|float16|64|2/8|39.291|54.672|60.695|1.391×|领先|
|float16|256|2/8|48.566|63.243|70.336|1.302×|领先|
|float16|1024|2/8|93.183|101.415|116.886|1.088×|领先|
|float16|4096|2/8|258.895|273.465|367.187|1.056×|领先|
|float16|64|16/64|65.309|76.948|121.866|1.178×|领先|
|float16|256|16/64|120.065|136.172|288.411|1.134×|领先|
|float16|1024|16/64|357.201|431.985|928.875|1.209×|领先|
|float16|4096|16/64|1236.243|1860.179|3709.923|1.505×|领先|
|bfloat16|63|2/8|44.777|60.917|65.227|1.360×|领先|
|bfloat16|65|2/8|40.117|60.457|69.020|1.507×|领先|
|float16|63|2/8|38.516|51.311|60.193|1.332×|领先|
|float16|65|2/8|34.379|47.852|63.408|1.392×|领先|
|float16|4160|2/8|262.039|276.998|371.004|1.057×|领先|

## 7. gfx936 Decode

|dtype|B|H/HV|norm|CK KV|CK VK|AITER HIP|AITER Triton|对最快加速比（VK）|结论|
|---|---:|---|---|---:|---:|---:|---:|---:|---|
|bfloat16|1|2/8|False|15.527|3.874|3.762|7.807|0.971×|接近|
|bfloat16|8|2/8|False|52.114|7.284|7.512|12.884|1.031×|接近|
|bfloat16|32|2/8|False|175.113|29.846|36.823|34.517|1.157×|领先|
|bfloat16|128|2/8|False|651.036|112.392|263.528|116.784|1.039×|接近|
|bfloat16|1|16/64|False|52.390|7.302|7.448|12.994|1.020×|接近|
|bfloat16|8|16/64|False|334.612|56.575|65.533|58.477|1.034×|接近|
|bfloat16|32|16/64|False|1283.914|232.612|348.420|233.799|1.005×|接近|
|bfloat16|128|16/64|False|5104.181|938.977|2847.949|935.948|0.997×|接近|
|float16|1|2/8|False|15.409|3.829|3.766|7.560|0.984×|接近|
|float16|8|2/8|False|51.661|7.069|7.489|12.665|1.059×|领先|
|float16|32|2/8|False|174.704|29.659|36.024|35.997|1.214×|领先|
|float16|128|2/8|False|651.565|112.799|263.849|121.883|1.081×|领先|
|float16|1|16/64|False|51.945|7.076|7.434|12.692|1.051×|接近|
|float16|8|16/64|False|334.048|56.328|65.482|61.703|1.095×|领先|
|float16|32|16/64|False|1288.521|231.995|348.113|239.792|1.034×|接近|
|float16|128|16/64|False|5123.839|934.773|2999.006|935.931|1.001×|接近|
|bfloat16|32|16/32|True|651.963|113.452|183.786|115.471|1.018×|接近|

## 8. gfx938 完整 Prefill

|dtype|T|H/HV|CK|AITER HIP链|AITER Triton链|对最快加速比|结论|
|---|---:|---|---:|---:|---:|---:|---|
|bfloat16|64|2/8|43.298|60.302|70.859|1.393×|领先|
|bfloat16|256|2/8|56.274|70.231|82.541|1.248×|领先|
|bfloat16|1024|2/8|120.450|114.140|135.676|0.948×|落后|
|bfloat16|4096|2/8|342.758|297.060|370.784|0.867×|落后|
|bfloat16|64|16/64|89.637|85.008|139.615|0.948×|落后|
|bfloat16|256|16/64|167.856|144.234|289.518|0.859×|落后|
|bfloat16|1024|16/64|437.351|428.319|967.976|0.979×|接近|
|bfloat16|4096|16/64|1596.408|1756.680|3844.402|1.100×|领先|
|float16|64|2/8|45.034|60.241|67.583|1.338×|领先|
|float16|256|2/8|56.277|69.680|78.341|1.238×|领先|
|float16|1024|2/8|105.944|112.182|130.753|1.059×|领先|
|float16|4096|2/8|302.895|292.837|360.962|0.967×|接近|
|float16|64|16/64|73.236|83.418|130.236|1.139×|领先|
|float16|256|16/64|140.248|142.456|275.690|1.016×|接近|
|float16|1024|16/64|393.299|428.975|924.909|1.091×|领先|
|float16|4096|16/64|1369.669|1754.312|3676.095|1.281×|领先|
|bfloat16|63|2/8|49.839|58.815|70.452|1.180×|领先|
|bfloat16|65|2/8|48.695|54.124|74.004|1.111×|领先|
|float16|63|2/8|43.534|56.397|67.192|1.295×|领先|
|float16|65|2/8|41.540|53.612|70.680|1.291×|领先|
|float16|4160|2/8|318.770|313.030|390.652|0.982×|接近|

## 9. gfx938 Decode

|dtype|B|H/HV|norm|CK KV|CK VK|AITER HIP|AITER Triton|对最快加速比（VK）|结论|
|---|---:|---|---|---:|---:|---:|---:|---:|---|
|bfloat16|1|2/8|False|18.845|4.305|4.151|10.321|0.964×|接近|
|bfloat16|8|2/8|False|68.135|8.498|9.032|15.710|1.063×|领先|
|bfloat16|32|2/8|False|233.146|27.795|47.226|32.474|1.168×|领先|
|bfloat16|128|2/8|False|888.866|87.901|390.532|111.460|1.268×|领先|
|bfloat16|1|16/64|False|68.262|8.509|8.994|15.752|1.057×|领先|
|bfloat16|8|16/64|False|452.184|48.040|102.646|62.098|1.293×|领先|
|bfloat16|32|16/64|False|1762.744|167.843|696.900|219.159|1.306×|领先|
|bfloat16|128|16/64|False|7008.359|666.075|4004.539|865.026|1.299×|领先|
|float16|1|2/8|False|18.626|4.346|4.210|8.484|0.969×|接近|
|float16|8|2/8|False|65.911|8.332|9.054|14.102|1.087×|领先|
|float16|32|2/8|False|231.786|27.717|47.607|34.958|1.261×|领先|
|float16|128|2/8|False|888.678|87.586|373.375|108.585|1.240×|领先|
|float16|1|16/64|False|65.705|8.328|8.991|13.844|1.080×|领先|
|float16|8|16/64|False|450.858|47.902|101.863|58.322|1.218×|领先|
|float16|32|16/64|False|1767.250|167.623|712.372|204.446|1.220×|领先|
|float16|128|16/64|False|7040.561|664.805|3968.250|764.983|1.151×|领先|
|bfloat16|32|16/32|True|890.142|95.267|296.154|104.172|1.093×|领先|

## 10. 适用边界与验证

- gfx936 的21组完整Prefill均领先，但不代表任意长度都领先。另批边界测量中，FP16 H2/HV8的T4097、8193、8256仍落后HIP；不将这些单顺序测量混入正式主表。
- gfx938 的21组完整Prefill均快于Triton；相对HIP仍有四个BF16落后项：T1024/T4096、H2/HV8，以及T64/T256、H16/HV64。正式表已完整保留，未宣称全部追平或达到硬件极限。
- Prefill历史接受版本通过原生190项、功能边界和122项扩展精度；gfx936功能边界30项、gfx938最后接受版本42项。BF16/FP16原误差门槛分别保留为0.03/0.02，BF16保持RNE，不与早期截断版性能混比。
- Decode较快结果依赖VK及特定gate/beta契约；KV也受支持，但表中可见明显性能差别。这里的独立算子数据不证明已完成任何框架的状态池ABI或模型端到端集成。
- 当前实现的已接受优化包括协作式state计算、按负载选择的输出布局、WU向量写回、短序列融合及架构限定分派；低频形状、FP16、尾块、varlen、gate和布局回退均保留。

性能二进制身份（SHA256）：gfx936 CK `3ec57253a3eccc49adb2d29aea34cdcbc991806bfe373b76ae3824f75059e5b5`，AITER native `ae75a0fed59e0f96657567b7cca581e36f24cdef6bb27f6f72ed76aa1a59caf4`；gfx938 Prefill CK `fd3bec58d882f78958c8508ce2116cb428d736dc5d5bea5d3ca391b50cfc10c3`，Decode CK `b5568846063952b765daf245b54e4ff7668aa4f309680a00c8ef00553b3fe6e6`，AITER native `f8b4d1241cfa41de62d78b33a99821bbcccab0443914c4381d0bb0578f4ba171`。它们标识上述历史实测，不是清理后重新编译的二进制。

源码清理验证：gfx936/gfx938 各10个原生翻译单元重新编译并链接通过；同编译器下，KKT/state/output的设备指令序列集合分别109个和109个，与清理前一致。gfx936原生190项及24项补充边界通过。2026-09-24在gfx938目标设备上，使用清理后的当前源码重新编译全部10个原生翻译单元并链接Prefill/Decode程序，原生190项及24项补充边界全部通过；同步的1,446个源码文件与本地快照逐一核对一致。原有数值检查门槛与BF16 RNE均未修改：Prefill output的rtol/atol为BF16 0.03、FP16 0.02；Decode output为BF16 0.05、FP16 0.03；两者final state均为0.02。本次补验为编译与正确性回归，未重新测量性能，上述性能表仍对应已标明的历史实测版本。
