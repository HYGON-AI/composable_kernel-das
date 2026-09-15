<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->
<!-- SPDX-License-Identifier: MIT -->

# FP16 MultiD GEMM V3 on HCU

`gemm_multi_d_f16_v3_config.hpp` provides a `DeviceGemm` alias for
`DeviceGemmMultiD_Xdl_CShuffle_V3` with FP16 A/B/D/E, FP32 accumulation and
shuffle, and the V3 pipeline. A is `[M,K]`, B is `[K,N]`, and D/E are `[M,N]`.
The four executables follow the BF16 example naming convention:

| Filename / target suffix | A/B/E layout |
|---|---|
| `mk_kn_mn` | RRR |
| `mk_nk_mn` | RCR |
| `km_kn_mn` | CRR |
| `km_nk_mn` | CCR |

R/C means row/column major. Layout describes storage order, not a fixed M/N/K.
`DsLayout`, `DsDataType` and the epilogue operation remain template parameters.
Each `.cpp` selects A/B layouts. `gemm_multi_d_f16_v3_common.hpp` contains the
command-line runner, allocation, timing and CPU-reference checks.

After configuring CK for `gfx936` or `gfx938` with HCU examples enabled:

```sh
cmake --build build --target example_hcu_gemm_multi_d_f16_v3 -j4
for layout in mk_kn_mn mk_nk_mn km_kn_mn km_nk_mn; do
    ./build/bin/example_hcu_gemm_multi_d_xdl_f16_${layout}_v3 --test || exit 1
done
```

CTest uses `--test`. In this mode, each executable checks six problems against the CPU reference, with both an
empty D tuple and `E = relu(A*B + D0 + D1)`. The matrix includes small K,
main-loop boundaries, M/N/K block tails, padded leading strides, and different
D0/D1 strides. Comparisons use `rtol=atol=1e-3` after FP16 output conversion.
Two additional empty-D cases exercise KBatch=2/4 with FP16 atomic accumulation;
they use rtol=atol=5e-3 because partial results and atomic sums round to FP16.
Every case prints `PASS` or `FAIL`; unsupported arguments and numerical failures
return a nonzero exit code.

## Single problem and GPU timing

Without arguments, each executable verifies one packed 256x256x128 GEMM.
The first ten positional arguments match the BF16 MultiD example:

```text
verify init time [M N K StrideA StrideB StrideE KBatch [fused StrideD0 StrideD1 [warmup repeat]]]
```

- `verify`, `time`, `fused`: 0 or 1; default 1, 0, 0.
- `init`: 0 initializes zeros, 1 deterministic integers, 2 deterministic fractions (default).
- Strides are leading dimensions in elements. For A use at least K (Row) or M (Col);
  for B use at least N (Row) or K (Col). E/D0/D1 are Row, with strides at least N.
- `KBatch` defaults to 1; `fused=1` selects `relu(A*B+D0+D1)` and requires KBatch=1.
- `warmup` / `repeat` default to 5 / 50; warmup may be zero, repeat must be positive.

```sh
exe=./build/bin/example_hcu_gemm_multi_d_xdl_f16_mk_nk_mn_v3
# BF16-compatible arguments: verify, fractional inputs, GPU timing
"$exe" 1 2 1 256 256 128 128 128 256 1
# Independent A/B/E/D strides, fused epilogue, explicit warmup/repeat
"$exe" 1 2 1 130 132 136 144 152 148 1 1 140 144 2 10
# Empty-D split-K with padded output stride, zero warmup
"$exe" 1 2 1 130 132 512 520 528 148 4 0 140 144 0 10
```

HIP events measure each invoker execution on the default stream; `Perf` reports
mean milliseconds and `2*M*N*K / time` in TFLOPS (GEMM FLOPs only).
Every warmup and measured invocation starts with fresh output. The full output
reset, allocations, H2D/D2H copies and CPU verification are outside the event
interval. The invoker's internal split-K output reset is included. This measures
GPU invoker execution, not Torch dispatch, JIT compilation or end-to-end latency.
Repeated execution's final result is verified when `verify=1`; `verify=0` prints
`RUN` rather than claiming numerical verification.

These are CK example executables, not Torch bindings or registered library
instances. A consumer instantiates the public CK device template with its own
configuration, buffers, strides and stream. The standalone results do not prove
that Torch selected this configuration.

These configurations use `MNKPadding`, K-facing input vectors of 8, other input
vectors of 2, and output vectors of 4. Padding handles block tails; it does not
remove vector-alignment constraints. Call `IsSupportedArgument` before launch.
The V3 configuration requires more than two K blocks (padded K >= 96).
Fused cases use `KBatch=1`; split-K fusion is not supported by this example.
Timing support provides a measurement tool; these configurations have not been
performance-tuned or validated through Torch.
