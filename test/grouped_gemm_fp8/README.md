# gfx938 FP8 grouped GEMM test

This test directly instantiates
`ck::tensor_operation::device::DeviceGroupedGemmMultipleDXdlCShuffleTileLoop`.
It requires this CK checkout, a gfx938-capable HIP compiler/runtime, and a gfx938
GPU. It does not require PyTorch, MSLK, downloaded sources, or files from an
existing `hygon_tmp` directory.

The underlying FP8 XDL path is enabled for gfx92a, gfx938, and gfx946. This
regression target remains gfx938-only; gfx92a and gfx946 have not been compiled
or tested as part of this change. Only 16x16 XDL tiles are supported.

From the CK repository root, configure and run the standalone test without GTest:

```bash
cmake -S test/grouped_gemm_fp8 -B hygon_tmp/grouped_gemm_fp8/build \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/aicc -DCMAKE_BUILD_TYPE=Release
cmake --build hygon_tmp/grouped_gemm_fp8/build -j2
HIP_VISIBLE_DEVICES=0 ctest --test-dir hygon_tmp/grouped_gemm_fp8/build \
  --output-on-failure
```

Adjust the compiler path and visible device for the local installation. The
build directory contains generated output only and may start empty. The full CK
test build also registers `test_grouped_gemm_fp8` when `GPU_TARGETS=gfx938`.

The 11 numerical cases cover OCP E4M3FN inputs, BF16 output, two float scale
tensors, 16x16 XDL tiles, Default/KPadding, v1/v2/v3 pipelines, device-generated
arguments via `SetDeviceKernelArgs`, zero-M groups, changing arguments under graph
replay, and a BF16-input regression. The FP8 reference uses an independent
software decoder, including finite values with exponent 15. The test also checks
host argument validation and output sentinels.
