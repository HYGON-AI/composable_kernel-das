# CK Tile GEMM

This directory builds `tile_example_gemm_basic` and `tile_example_gemm_universal`.

## Build

From the repository root, create a dedicated build directory and select the
architecture of the test device.

```bash
mkdir -p build-gemm
cd build-gemm
CK_GPU_TARGETS=gfx936 bash ../script/cmake-ck-dev.sh ..
cmake --build . --target tile_example_gemm_basic tile_example_gemm_universal -j8
```

## Basic GEMM

The basic example uses MMAC V1 with fp16 and RCR layout.

```bash
./bin/tile_example_gemm_basic -b=1 -m=128 -n=128 -k=64 -v=1 -warmup=0 -repeat=1
```

## Universal GEMM

The universal example is available for gfx928, gfx92a, gfx936 and gfx938 builds.
It computes `C[M,N] = A[M,K] × B[K,N]` with FP32 accumulation.
A and C are row major. B can be row or column major.

```bash
./bin/tile_example_gemm_universal -pipeline=compv3 -prec=fp16 -b_layout=C -v=1
./bin/tile_example_gemm_universal -pipeline=compv4 -prec=bf16 -b_layout=R -v=1
./bin/tile_example_gemm_universal -pipeline=mem -scheduler=interwave -epilogue=cshuffle -v=1
```

| Option | Values |
|---|---|
| `-pipeline` | `compv3`, `compv4`, `mem` |
| `-scheduler` | `intrawave` or `interwave`. Interwave is available for Mem. |
| `-epilogue` | `default` or `cshuffle` |
| `-prec` | `fp16` or `bf16` |
| `-m/-n/-k` | Positive matrix dimensions. Defaults are 256/256/128. |
| `-a_layout/-b_layout/-c_layout` | R/C/R by default. B also accepts R. |
| `-stride_a/-stride_b/-stride_c` | Leading strides in elements. Zero selects packed storage. |
| `-init` | `0` seeded random, `1` linear, `2` ones |
| `-v` | `0` skip, `1` CPU reference, `2` GPU reference |
| `-timer/-warmup/-repeat` | GPU or CPU timer. Zero repeats runs one untimed launch. |
| `-split_k/-persistent/-flush_cache/-rotating_count` | This example accepts 1/0/false/1. |

`-b` is a basic example option. Universal uses `-split_k=1` for a single
2D product. Unsupported layouts, strides and vector alignment return a
nonzero exit status. With verification enabled, the runner checks every
output element and the storage between rows.

From the repository root, pass the exact build directory to the smoke test.
The full test also covers both layouts, both epilogues, K loop boundaries,
and nonpacked leading strides.

```bash
bash example_hcu/ck_tile/03_gemm/script/smoke_test.sh build-gemm
bash example_hcu/ck_tile/03_gemm/script/run_full_test.sh local main "$(hostname)" gfx936 build-gemm
```
