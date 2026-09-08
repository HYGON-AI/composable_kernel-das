# ck4inductor HCU 移植总结

本文用于记录 `ck4inductor` 从 github CK 移植到 `composable_kernel_hcu` 的必要步骤、与 github CK 的差异点、以及已踩过的坑。本文不记录完整调试流水，后续补充 CK 实例时按这里的流程扩展即可。

## 目标与范围

目标是让 HCU CK 通过 `pip install .` 提供 `ck4inductor` Python 包，使 PyTorch TorchInductor 能 import `ck4inductor`、发现 CK instance、生成 ROCm CK template，并进入 autotune/benchmark。

当前首轮目标是可安装、可枚举、可被 PyTorch CK template 消费。不把当前候选当成 HCU 最优性能配置，`preselected` 暂时禁用，后续必须在 HCU 上重新 benchmark。

当前模块：

- `ck4inductor.universal_gemm`
- `ck4inductor.batched_universal_gemm`
- `ck4inductor.grouped_conv_fwd`
- `ck4inductor.hcu_grouped_conv_fwd`
- `ck4inductor.ck_tile_universal_gemm`

## 移植步骤

### 1. 新增 Python 包骨架

在 `composable_kernel_hcu/python/ck4inductor/` 下建立包结构，新增：

- `__init__.py`
- `util.py`
- `_template_parser.py`
- 各算子子包的 `op.py`、`gen_instances.py`

在 `composable_kernel_hcu/pyproject.toml` 中显式声明 setuptools package，分发名沿用 `rocm-composable-kernel`，Python import 名沿用 `ck4inductor`，避免 PyTorch 侧 import 路径变化。

### 2. 配置 package-data

`pip install .` 后 PyTorch 是从 site-packages 中读取 `ck4inductor.include` 和 `ck4inductor.library`，不是直接读取源码树。因此需要把必要头文件和实例文件放进 package-data。

当前 package-data 保留：

- `ck4inductor.include`: `include/ck/**/*.h`、`include/ck/**/*.hpp`
- `ck4inductor.library`:
  - `library/src/tensor_operation_instance_hcu/gpu/gemm_universal/**/*.hpp`
  - `library/src/tensor_operation_instance_hcu/gpu/gemm_universal_batched/**/*.hpp`
  - `library/src/tensor_operation_instance_hcu/gpu/grouped_conv2d_fwd_gfx*/**/*.cpp`
  - `library/src/tensor_operation_instance_hcu/gpu/grouped_conv2d_fwd_gfx*/**/*.hpp`
  - `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/**/*.cpp`
  - `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/**/*.hpp`

后续新增实例时，先确认实例文件是否能被 package-data 覆盖。若安装后 generator 枚举为空，优先查 package-data。

### 3. 使用通用模板解析器

上游 github CK 的 Python 逻辑偏向简单文本扫描。HCU 版本新增 `_template_parser.py`，用 Python 原生 `pathlib.Path.rglob()` 扫描文件，并用括号深度解析模板参数。

后续新增实例不要写 shell `grep` 依赖，统一走：

- `find_template_instances()`
- `parse_template_instances_from_file()`
- `split_template_args()`

这样能正确处理 `S<...>`、`Sequence<...>`、`Tuple<...>`、多行模板、嵌套逗号。

### 4. 移植 universal_gemm

当前按 PyTorch 需求补的是 HCU instance 路径：

- 路径：`library/src/tensor_operation_instance_hcu/gpu/gemm_universal/`
- 模板：`DeviceGemm_Xdl_CShuffleV3`
- 当前种子实例：BF16，`Row/Col/Row`

与早期计划中扫描标准 `tensor_operation_instance/gpu/gemm` 不同，最终 PyTorch 需求指向 `gemm_universal` 命名，因此 HCU 侧新增 HCU instance 目录，而不是复用标准 GEMM 目录。

注意：

- 当前 `gen_ops_preselected()` 返回空列表。
- `CKGemmOperation` 不能是 frozen dataclass，因为 PyTorch CK template 渲染期间会修改 op 字段。
- Python bool 字段需要规整成 C++ 字面量 `true/false`，否则生成 C++ 会出现 `False`。

### 5. 移植 batched_universal_gemm

当前按 PyTorch 需求补的是 HCU batched instance 路径：

- 路径：`library/src/tensor_operation_instance_hcu/gpu/gemm_universal_batched/`
- 模板：`DeviceBatchedGemmMultiD_Xdl_CShuffle_V3`
- 当前种子实例：BF16，`Row/Col/Row`

注意：

- 当前实例是 PyTorch CK template 能消费的 MultiD CShuffle V3 形式，不是 HCU 老的 `DeviceBatchedGemmXdl` dtype-first 形式。
- `CKBatchedGemmOperation` 同样不能 frozen。
- `a_block_lds_extra_m`、`b_block_lds_extra_n` 等字段需输出 C++ 字面量。

### 6. 移植 grouped_conv_fwd

标准 grouped conv 保留在：

- 路径：`library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/`
- 模板：`DeviceGroupedConvFwdMultipleD_Xdl_CShuffle`

HCU grouped conv 另建模块：

- 模块：`ck4inductor.hcu_grouped_conv_fwd`
- 路径：`library/src/tensor_operation_instance_hcu/gpu/grouped_conv2d_fwd_gfx*/`
- 覆盖 `gfx928/gfx936`，后续可继续扩展 `gfx938`
- 解析 HCU `mmac`、`mmac_v2_cshuffle`、`ck_tile` 等 family

不要把 HCU MMAC grouped conv 强塞进标准 `grouped_conv_fwd` dataclass。两者模板体系不同，混在一起会让字段语义失真。

### 7. 移植 ck_tile_universal_gemm

`ck_tile_universal_gemm` 当前是 Python 枚举，不扫描 C++ instance。HCU 侧只保留已在示例中有依据的组合：

- pipeline: `Mem`、`CompV4`
- scheduler: `Intrawave`
- dtype/layout 只保留当前 allowlist

`CompV3` 暂不进入稳定输出；后续如果 HCU 示例或 benchmark 确认可用，再加入 allowlist。

### 8. 补 header-only 安装闭包

PyTorch CK template 编译的是 header-only package 路径。`pip install .` 不会执行 CK CMake，因此 CMake 生成或全局定义的内容要在 Python 包路径里补齐。

当前已补：

- `include/ck/config.h`
- `pyproject.toml` 中 `ck/**/*.h`
- `include/ck/ck.hpp` 包含 `ck/utility/data_type.hpp`
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp` 补 `MultiplyMultiply`

`config.h` 需要提供默认宏：

- `CK_EXPERIMENTAL_BIT_INT_EXTENSION`
- `CK_ENABLE_INT8`
- `CK_ENABLE_FP8`
- `CK_ENABLE_BF8`
- `CK_ENABLE_FP16`
- `CK_ENABLE_BF16`
- `CK_ENABLE_FP32`
- `CK_ENABLE_FP64`

原因是 CMake 构建会通过 `add_compile_definitions()` 定义这些宏，但 PyTorch JIT 编译 CK template 时不会自动获得它们。

### 9. 验证安装与 generator

本地验证：

```powershell
python -m pip install -e composable_kernel_hcu
python -m unittest composable_kernel_hcu.python.test.test_gen_instances -v
```

预期：

- `ck4inductor` 可 import
- `library_path()` 指向可读 library
- generator 单测通过
- 当前为 21 个 unittest 通过

远端 nmz-1 验证：

```powershell
hcu-remote --profile nmz-1 script --file hygon_tmp\remote_ck4_hcu_validate.sh
```

预期：

- 容器内 `/wksp/ai/composable_kernel` 执行 `pip install -e .`
- 21 个 unittest 通过
- 当前实例数量：
  - `universal_gemm`: 1
  - `batched_universal_gemm`: 1
  - `hcu_grouped_conv_fwd`: 1050

### 10. 验证 PyTorch TorchInductor e2e

PyTorch 侧 CK 选择条件包括：

- `torch._inductor.config.max_autotune` 或 `max_autotune_gemm` 开启
- `max_autotune_gemm_backends` 包含 `CK`
- `torch.version.hip` 存在
- device 是 cuda
- dtype 是 fp16/bf16/fp32
- 当前 GPU arch 命中 `torch._inductor.config.rocm.ck_supported_arch`
- `try_import_ck_lib()` 能 import `ck4inductor`

nmz-1 当前是 `gfx938`，PyTorch 默认 `ck_supported_arch` 不含 `gfx938`。验证时运行时 patch：

```python
from torch._inductor import config

config.max_autotune = True
config.max_autotune_gemm = True
config.max_autotune_gemm_backends = "CK,ATEN"
config.rocm.ck_supported_arch.append("gfx938")
```

正式 PyTorch 侧需要把 `gfx938` 加入 `torch._inductor.config.rocm.ck_supported_arch` 默认列表。

当前 e2e 验证脚本：

```powershell
hcu-remote --profile nmz-1 script --gpu --file hygon_tmp\nmz_torchinductor_ck_rcr_e2e.sh
```

当前结果：

- CK instance 能被 PyTorch 发现。
- CK template 能生成 C++。
- CK `.so` 能编译成功。
- CK 参与 autotune。
- 当前测试 shape 下 ATEN 更快，所以最终输出仍选择 `extern_kernels.mm/bmm`。

这说明“发现 CK 实例并实际调用 benchmark”已经打通；后续要让最终图选择 CK，需要补更合适的 instance 或 benchmark 出更优 shape。

## 与 github CK 的主要差异

### package 与路径差异

github CK 的 package-data 默认不够 HCU 使用，尤其缺 `.h`、HCU instance 路径、grouped conv sidecar 文件。HCU 版本必须显式加入 HCU 路径。

github CK 常见路径：

- `library/src/tensor_operation_instance/gpu/gemm_universal`
- `library/src/tensor_operation_instance/gpu/gemm_universal_batched`

HCU 当前路径：

- `library/src/tensor_operation_instance_hcu/gpu/gemm_universal`
- `library/src/tensor_operation_instance_hcu/gpu/gemm_universal_batched`
- `library/src/tensor_operation_instance_hcu/gpu/grouped_conv2d_fwd_gfx*`

### 模板体系差异

HCU 标准 grouped conv 使用：

- `DeviceGroupedConvFwdMultipleD_Xdl_CShuffle`

github CK/PyTorch 侧常见命名可能带：

- `MultipleABD`
- `_V3`

不要按名字相似强行复用字段顺序。必须以 HCU 实际模板签名为准。

### HCU 与标准 XDL 差异

HCU MMAC/ck_tile grouped conv 与标准 XDL grouped conv 是两套模板体系。HCU 路径要独立 parser/dataclass，字段至少需要带：

- `arch`
- `family`
- `template_name`
- `source_path`
- `raw_args`

这样后续扩展 `gfx938` 或新增 family 时不会污染标准 XDL generator。

### PyTorch 消费端差异

PyTorch CK template 当前直接 import：

- `ck4inductor.universal_gemm`
- `ck4inductor.batched_universal_gemm`

并要求实例字段能直接渲染到 C++。因此 op 字段不是只用于 Python key，还会影响生成 C++，bool、tuple、dtype alias 都要兼容 PyTorch template。

## 已踩坑点

### 1. `pip install .` 不会生成 `ck/config.h`

CK CMake 会从 `include/ck/config.h.in` 生成 `config.h`，但 Python 包安装不走 CMake。缺失时 PyTorch 编译报：

```text
fatal error: 'ck/config.h' file not found
```

解决：新增 `include/ck/config.h`，并在 package-data 中加入 `ck/**/*.h`。

### 2. CMake compile definitions 在 PyTorch JIT 中不存在

`CK_EXPERIMENTAL_BIT_INT_EXTENSION` 在 CMake 中定义，但 PyTorch CK template 编译没有这个宏，会导致：

```text
no type named 'f8_t' in namespace 'ck'
no type named 'bf8_t' in namespace 'ck'
no type named 'pk_i4_t' in namespace 'ck'
```

解决：在 `include/ck/config.h` 默认定义 `CK_EXPERIMENTAL_BIT_INT_EXTENSION`。

### 3. PyTorch 生成 C++ 会引用 `MultiplyMultiply`

即使当前 BF16 pass-through 实例不使用 multiply-multiply epilogue，PyTorch template 头部仍会生成 alias：

```cpp
using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;
```

HCU header 里缺这个 struct 会直接编译失败。解决：补 `MultiplyMultiply` 到 `element_wise_operation.hpp`。

### 4. `ck/utility/data_type.hpp` include 顺序不能太早

`data_type.hpp` 依赖 `ck::index_t`。如果在 `ck.hpp` 顶部 include，会先于 `index_t` 定义而报错。解决：在 `ck.hpp` 定义完 `index_t/long_index_t` 并关闭 `namespace ck` 后再 include。

### 5. editable install 下 `resources.files()` 可能返回 `MultiplexedPath`

远端加了 `ck4inductor/include`、`ck4inductor/library` symlink 后，`importlib.resources.files("ck4inductor.library")` 可能返回 `MultiplexedPath(...)`。直接 `str()` 会变成不可用路径。

解决：`library_path()` 需要识别 `_paths`，选出包含 `src` 的真实 path；若失败再回退到 editable 源码树的 `library`。

### 6. dataclass 不能 frozen

PyTorch CK template 渲染时会改 op 字段。如果 `@dataclass(frozen=True)`，会报 `FrozenInstanceError`。

解决：`CKGemmOperation`、`CKBatchedGemmOperation` 使用普通 dataclass。

### 7. Python bool 会被渲染成非法 C++ `False`

op 字段如果保留 Python bool，生成 C++ 时可能出现：

```cpp
/* a_block_lds_extra_m */ False
```

解决：在 `__post_init__()` 中把相关字段转成 `"true"` 或 `"false"` 字符串。

### 8. 当前实例只覆盖 RCR

当前 HCU GEMM 种子实例是：

- A layout: `Row`
- B layout: `Col`
- C layout: `Row`
- dtype: `BF16`

如果测试输入 B 是 row-major，CK filter 会筛不到实例。e2e 验证要构造 RCR 输入，例如用转置后的 B，使 B stride 呈 col-major。

### 9. `gfx938` 不在 PyTorch 默认 CK supported arch

PyTorch 默认支持列表没有 Hygon `gfx938`，不 patch 时 `use_ck_template()` 直接返回 false。

解决：

- 验证脚本中运行时追加 `gfx938`
- 正式 PyTorch 侧需要把 `gfx938` 加进 `torch._inductor.config.rocm.ck_supported_arch`

### 10. e2e 通过不等于最终选择 CK

TorchInductor 会 autotune 多个 backend。当前 CK 已成功编译并参与 benchmark，但测试 shape 下 ATEN 更快，所以最终代码仍是 `extern_kernels.mm/bmm`。这不是 CK 调用失败，而是正常性能选择结果。

判断是否真正进入 CK 的证据应看日志中是否有：

- `generated 1 ck instances after filter`
- `rocm_ck_gemm_template_0`
- 生成 `.cpp` 和 `.so`
- `AUTOTUNE` 表中出现 `rocm_ck_gemm_template_0`

## 后续补充 CK 实例流程

1. 先确认 PyTorch 需要的 layout/dtype/shape。
2. 在 HCU 现有 example 或 profiler 中找到已能编译运行的模板参数。
3. 把模板参数整理成对应 instance hpp/cpp，放入当前 package-data 覆盖的路径。
4. 若新增路径不在 package-data，先更新 `pyproject.toml`。
5. 更新对应 `gen_instances.py` 的 parser 或 allowlist。
6. 在 `python/test/test_gen_instances.py` 中增加覆盖：
   - 解析 token 数量
   - dtype/layout/family/arch
   - generator 非空与 name 稳定
7. 本地跑 unittest。
8. nmz-1 跑 `remote_ck4_hcu_validate.sh`。
9. 若是 GEMM/Batched GEMM，跑 TorchInductor e2e，确认 CK 进入 autotune。
10. 只有在 HCU 上 benchmark 后，才把实例加入 preselected 或作为推荐候选。

## 常用验证命令

本地：

```powershell
python -m pip install -e composable_kernel_hcu
python -m unittest composable_kernel_hcu.python.test.test_gen_instances -v
```

远端安装与 generator：

```powershell
hcu-remote --profile nmz-1 script --file hygon_tmp\remote_ck4_hcu_validate.sh
```

PyTorch CK 选择条件检查：

```powershell
hcu-remote --profile nmz-1 script --gpu --file hygon_tmp\nmz_torchinductor_ck_selection_probe.sh
```

TorchInductor RCR e2e：

```powershell
hcu-remote --profile nmz-1 script --gpu --file hygon_tmp\nmz_torchinductor_ck_rcr_e2e.sh
```

## 当前状态

- `ck4inductor` package 可安装。
- 本地和 nmz-1 generator 单测通过。
- HCU universal GEMM / batched universal GEMM 当前各有 1 个 BF16 RCR instance。
- HCU grouped conv 当前枚举 1050 个 instance。
- PyTorch TorchInductor 可发现 CK instance、生成 CK C++、编译 `.so` 并进入 autotune。
- 当前 e2e shape 下 CK 比 ATEN 慢，因此最终图选择 ATEN fallback。

后续重点是补更多 HCU GEMM/Batched GEMM instance，并在 `gfx938` 上做 shape 级 benchmark。
