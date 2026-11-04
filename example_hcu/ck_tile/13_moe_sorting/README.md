# moe-sorting

This folder contains example for moe-sorting kernel using ck_tile tile-programming implementation. This kernel is often used in Moe model, before launching the fused-moe-gemm block. The input&weight is a `token*topk` 2d matrix. The op rearange the input weight ids into different experts and feed into fuse moe gemm kernel.

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_moe_sorting -j
```
This will result in an executable `build/bin/tile_example_moe_sorting`

For this HCU tree, select the target with `CK_GPU_TARGETS` (the development
script does not consume a second architecture argument):

```bash
mkdir -p build && cd build
CK_GPU_TARGETS=gfx946 bash ../script/cmake-ck-dev.sh ..
cmake --build . --target tile_example_moe_sorting -j2
```

Use `gfx936` or `gfx938` on the corresponding hardware. The gfx946 CMake target
sets `CK_ENABLE_MULTIDIE=1` for the shared CK launcher in
`ck/host_utility/multidie_kernel_launch.hpp`; runtime HIP properties select the path.
The common interface and standalone regression are documented in
[`00_kernel_launch`](../00_kernel_launch/README.md).

### Correctness without benchmarking

```bash
bash example_hcu/ck_tile/13_moe_sorting/script/validate_multidie.sh \
  "$PWD/build/bin/tile_example_moe_sorting"
```

The script runs ten small cases using the example's CPU reference, including
single-phase/multi-phase sorting, expert masks, invalid routes, padding, and
buffer clearing. It uses int32 indices/FP32 weights, a fixed routing seed, no
warmup, and one untimed call per case. Append `list` or a case name to list or
select cases; separate processes/output directories are suitable for Perf Model.

For gfx946 set the model environment supplied by the installed SDK. Use `sb`
for one XCD or `sbx2` for two XCDs. `AITER_MOE_SORTING_LOG_LAUNCH=1` logs the
launcher; `AITER_MOE_SORTING_DIE_MODE=auto|off|linear` and
`AITER_MOE_SORTING_DIE_CHUNK=1|2` select validation modes. Explicit `linear`
requires gfx946 with at least two XCDs. Validate actual multi-die execution with
per-XCD kernel statistics as well as CPU correctness; launch logs alone are
insufficient. These environment variable names match AITER's CK sorting binding,
but the executable and script do not depend on AITER.

## example
```
args:
          -v    weather do CPU validation or not (default:1)
       -pr_i    index data type. (currently only fp32 supported now) (default:int32)
       -pr_w    output weight data type(currently only fp32 supported now) (default:fp32)
          -t    number of input tokens (default:32)
          -e    number of experts (default:8)
          -k    topk (default:2)
       -st_i    row stride of input, -1 means same as experts (default:-1)
       -seed    seed to be used, -1 means random every time (default:-1)
      -kname    when set to 1 it will print kernel name (default:0)
       -time    time kernel or not (default:1)

```
