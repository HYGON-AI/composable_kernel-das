# Grouped-GEMM performance comparison

Performance comparison for every NT/NN/TN workload in
[`hipc_kernels/bench_grouped_shapes.sh`](./hipc_kernels/bench_grouped_shapes.sh).
All values are TFLOPS. A positive percentage means higher throughput than the
tuned hipBLASLt baseline.

| 测事项 | shape | hipblaslt tuned | hipc | CK original | CK fused | hipc vs hipblaslt tuned | CK original vs hipblaslt tuned | CK fused vs hipblaslt tuned |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NT tp1-fc1-fwd | M=[8152,8152,8257,8257], N=4096, K=4096 | 324.0 | 343.1 | 250.1 | 346.1 | +5.9% | -22.8% | +6.8% |
| NT tp1-fc2-fwd | M=[8152,8152,8257,8257], N=2048, K=4096 | 280.5 | 343.0 | 247.1 | 346.4 | +22.3% | -11.9% | +23.5% |
| NT tp1-fc1-dgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | 320.2 | 339.1 | 247.7 | 342.2 | +5.9% | -22.6% | +6.9% |
| NT tp1-fc2-dgrad | M=[8100,8100,8091,8091], N=4096, K=2048 | 261.4 | 303.8 | 222.7 | 308.3 | +16.2% | -14.8% | +17.9% |
| NT tp1-wgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | 320.1 | 339.0 | 247.6 | 342.1 | +5.9% | -22.6% | +6.9% |
| NT tp4-real | M=[1923,2047,2179,2043], N=4096, K=4096 | 265.2 | 318.9 | 231.5 | 320.6 | +20.3% | -12.7% | +20.9% |
| NT tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | 286.2 | 313.9 | 231.3 | 316.4 | +9.7% | -19.2% | +10.6% |
| NT tp4-mean-2048-fc2 | M=[2048,2048,2048,2048], N=2048, K=4096 | 290.0 | 291.1 | 200.6 | 280.7 | +0.4% | -30.8% | -3.2% |
| NT tp4-step-2112 | M=[2112,2112,2112,2112], N=4096, K=4096 | 269.3 | 290.4 | 208.0 | 292.7 | +7.8% | -22.8% | +8.7% |
| NT tp4-collapse-1875 | M=[1875,1875,1875,1875], N=4096, K=4096 | 243.4 | 294.2 | 58.2 | 295.9 | +20.9% | -76.1% | +21.6% |
| NT tp4-dgrad-a-16384 | M=[16384], N=4096, K=4096 | 352.7 | 342.6 | 249.6 | 344.8 | -2.8% | -29.2% | -2.2% |
| NT ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 352.4 | 344.0 | 250.5 | 345.9 | -2.4% | -28.9% | -1.9% |
| NT ctrl-8group-4096 | M=[4096,4096,4096,4096,4096,4096,4096,4096], N=4096, K=4096 | 288.6 | 341.1 | 250.3 | 341.8 | +18.2% | -13.3% | +18.4% |
| NT ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | 286.3 | 315.7 | 230.2 | 318.1 | +10.3% | -19.6% | +11.1% |
| NN nn-tp1-fc1 | M=[8100,8100,8091,8091], N=4096, K=4096 | 335.1 | 371.1 | 167.0 | 396.3 | +10.8% | -50.2% | +18.3% |
| NN nn-tp1-fc1-k2048 | M=[8100,8100,8091,8091], N=4096, K=2048 | 277.1 | 331.8 | 156.5 | 353.7 | +19.7% | -43.5% | +27.7% |
| NN nn-tp1-fc2 | M=[8100,8100,8091,8091], N=2048, K=4096 | 287.4 | 326.1 | 166.0 | 396.6 | +13.5% | -42.3% | +38.0% |
| NN nn-tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | 301.9 | 327.4 | 156.1 | 365.7 | +8.5% | -48.3% | +21.1% |
| NN nn-ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | 302.0 | 327.3 | 155.9 | 368.6 | +8.4% | -48.4% | +22.1% |
| NN nn-ctrl-16group | M=[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048], N=4096, K=4096 | 301.9 | 327.6 | 168.7 | 395.2 | +8.5% | -44.1% | +30.9% |
| NN nn-ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 372.9 | 407.2 | 168.8 | 403.2 | +9.2% | -54.7% | +8.1% |
| TN tn-tp1-fc1-wgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | N/A | 345.4 | 348.7* | 402.7 | N/A | N/A* | N/A |
| TN tn-tp1-fc2-wgrad | M=[8100,8100,8091,8091], N=2048, K=4096 | N/A | 325.6 | 321.2* | 400.0 | N/A | N/A* | N/A |
| TN tn-tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | N/A | 329.7 | 298.6* | 372.3 | N/A | N/A* | N/A |
| TN tn-ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | N/A | 341.5 | 273.6* | 371.7 | N/A | N/A* | N/A |
| TN tn-ctrl-16group | M=[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048], N=4096, K=4096 | N/A | 329.7 | 301.9* | 405.3 | N/A | N/A* | N/A |
| TN tn-ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 394.2 | 401.1 | 296.0* | 408.1 | +1.8% | N/A* | +3.5% |

## Measurement details

- Device: HCU 5 (80 CUs), selected with `CUDA_VISIBLE_DEVICES=5`; BF16.
- All measurements use 5 warmup iterations and 20 timed iterations.
- Reproduce the full sequential sweep with
  `./bench_grouped_gemm_fused_kernels.sh`. It extracts the same 27-shape list
  from `hipc_kernels/bench_grouped_shapes.sh`, writes raw logs beneath
  `perf-logs/`, and runs tuned hipBLASLt/HIPC, CK original, and CK fused in the
  same order. Use `--suite=hipc|ck-original|ck-fused` or
  `--layout=NT|NN|TN` to run a subset.
- HIPC and hipBLASLt were measured by `hipc_kernels/bf16_gemm` with
  `BF16GEMM_HIPBLASLT_SOLUTION_FILE=/root/workspace/bf16_gemm/tune/hipblaslt_solutions.conf`.
  The tune table selected solution 4095 for the NT baseline and 4396 for NN;
  grouped TN has no hipBLASLt baseline because that API cannot express its
  per-group leading dimension. The single-group TN baseline used solution 2692.
- CK original used `tile_example_grouped_gemm_device_args_benchmark` with
  `--backend=ck --validate=0 --include-args-build=0`, so the reported timing
  excludes the device-args builder kernel. Its values are the reported median
  across 20 samples.
- CK fused workloads use `tile_example_grouped_gemm_fused_test`; `--layout`
  selects WASP for NT and the v7/v8 test for NN/TN. NN and TN verified selected
  instances 5000 and 5001, respectively. The sweep is performance-only; use the functional
  tests separately for correctness validation.
- Large outliers in the first HIPC pass were repeated three times on HCU 5.
  The table uses the median of those repeats for NT `tp1-wgrad`, NT `tp4-real`,
  TN `tn-ctrl-16group`, and TN `tn-ctrl-1group-sum32818`; other HIPC rows use
  the initial sweep value.
- Percentage formula: `(kernel TFLOPS / hipBLASLt tuned TFLOPS - 1) * 100%`.
  TN grouped workloads have no hipBLASLt denominator, so their relative values
  are `N/A`.
- `*` CK original TN values are an equal-FLOP proxy, not exact-shape results:
  the device-args CK interface represents TN `group-lengths` as per-group K,
  whereas this HIPC shape list uses them as per-group M. For these proxy runs,
  CK original used M=4096 and the listed values as per-group K. The absolute
  TFLOPS are shown for reference only; no CK-original-vs-hipBLASLt ratio is
  reported for TN because those operands are not dimension-equivalent.

## Raw logs

- HIPC and tuned hipBLASLt sweep: `/tmp/grouped_gemm_full_hipc_gpu5_20261014.log`
- CK original sweep: `/tmp/grouped_gemm_full_ck_original_gpu5_20261014.log`
- CK fused NT sweep: `/tmp/grouped_gemm_full_ck_fused_nt_gpu5_20261014.log`
- CK fused NN/TN sweep: `/tmp/grouped_gemm_full_ck_fused_nntn_gpu5_20261014.log`
- HIPC anomaly repeats: `/tmp/grouped_gemm_hipc_outlier_repeats_gpu5_20261014.log`

## Refresh — 2026-10-15

This appended section records the 2026-10-15 rerun; the original table above
is preserved unchanged. Device, dtype, warmup, and timing parameters remain
HCU 5, BF16, 5 warmup iterations, and 20 timed iterations/samples.

| 测事项 | shape | hipblaslt tuned | hipc | CK original | CK fused | hipc vs hipblaslt tuned | CK original vs hipblaslt tuned | CK fused vs hipblaslt tuned |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NT tp1-fc1-fwd | M=[8152,8152,8257,8257], N=4096, K=4096 | 324.0 | 343.3 | 249.9 | 346.3 | +6.0% | -22.9% | +6.9% |
| NT tp1-fc2-fwd | M=[8152,8152,8257,8257], N=2048, K=4096 | 280.5 | 343.0 | 247.0 | 346.9 | +22.3% | -11.9% | +23.7% |
| NT tp1-fc1-dgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | 320.1 | 339.1 | 247.7 | 342.6 | +5.9% | -22.6% | +7.0% |
| NT tp1-fc2-dgrad | M=[8100,8100,8091,8091], N=4096, K=2048 | 261.7 | 303.7 | 222.4 | 311.2 | +16.0% | -15.0% | +18.9% |
| NT tp1-wgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | 320.2 | 339.1 | 247.2 | 342.6 | +5.9% | -22.8% | +7.0% |
| NT tp4-real | M=[1923,2047,2179,2043], N=4096, K=4096 | 265.2 | 318.9 | 231.6 | 323.6 | +20.3% | -12.7% | +22.0% |
| NT tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | 286.1 | 313.9 | 231.4 | 318.5 | +9.7% | -19.1% | +11.3% |
| NT tp4-mean-2048-fc2 | M=[2048,2048,2048,2048], N=2048, K=4096 | 290.0 | 291.1 | 200.5 | 287.3 | +0.4% | -30.9% | -1.0% |
| NT tp4-step-2112 | M=[2112,2112,2112,2112], N=4096, K=4096 | 269.5 | 290.3 | 207.5 | 292.1 | +7.7% | -23.0% | +8.4% |
| NT tp4-collapse-1875 | M=[1875,1875,1875,1875], N=4096, K=4096 | 243.3 | 294.3 | 57.9 | 295.7 | +21.0% | -76.2% | +21.6% |
| NT tp4-dgrad-a-16384 | M=[16384], N=4096, K=4096 | 350.8 | 342.6 | 249.1 | 349.2 | -2.3% | -29.0% | -0.5% |
| NT ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 352.2 | 344.0 | 250.1 | 349.6 | -2.3% | -29.0% | -0.7% |
| NT ctrl-8group-4096 | M=[4096,4096,4096,4096,4096,4096,4096,4096], N=4096, K=4096 | 288.5 | 341.0 | 250.2 | 343.8 | +18.2% | -13.3% | +19.2% |
| NT ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | 286.1 | 315.7 | 230.3 | 322.9 | +10.4% | -19.5% | +12.9% |
| NN nn-tp1-fc1 | M=[8100,8100,8091,8091], N=4096, K=4096 | 335.1 | 371.1 | 166.9 | 396.4 | +10.7% | -50.2% | +18.3% |
| NN nn-tp1-fc1-k2048 | M=[8100,8100,8091,8091], N=4096, K=2048 | 277.1 | 331.9 | 156.4 | 353.4 | +19.8% | -43.5% | +27.5% |
| NN nn-tp1-fc2 | M=[8100,8100,8091,8091], N=2048, K=4096 | 287.6 | 326.1 | 165.9 | 396.7 | +13.4% | -42.3% | +37.9% |
| NN nn-tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | 301.7 | 327.4 | 156.1 | 365.4 | +8.5% | -48.3% | +21.1% |
| NN nn-ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | 301.7 | 327.1 | 155.8 | 368.3 | +8.4% | -48.4% | +22.1% |
| NN nn-ctrl-16group | M=[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048], N=4096, K=4096 | 301.9 | 327.6 | 168.7 | 395.5 | +8.5% | -44.1% | +31.0% |
| NN nn-ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 372.4 | 407.2 | 168.8 | 376.8† | +9.3% | -54.7% | N/A† |
| TN tn-tp1-fc1-wgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | N/A | 345.6 | 348.6* | 402.4 | N/A | N/A* | N/A |
| TN tn-tp1-fc2-wgrad | M=[8100,8100,8091,8091], N=2048, K=4096 | N/A | 325.7 | 321.0* | 399.4 | N/A | N/A* | N/A |
| TN tn-tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | N/A | 329.7 | 298.6* | 371.1 | N/A | N/A* | N/A |
| TN tn-ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | N/A | 341.4 | 273.5* | 371.4 | N/A | N/A* | N/A |
| TN tn-ctrl-16group | M=[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048], N=4096, K=4096 | N/A | 329.8 | 301.7* | 405.5 | N/A | N/A* | N/A |
| TN tn-ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 394.2 | 400.7 | 296.0* | 407.9 | +1.7% | N/A* | +3.5% |

### Refresh notes

- The initial refresh measurements for NT `tp4-mean-2048` and TN
  `tn-ctrl-skew-10x` were severe low-throughput outliers. Each implementation
  for those shapes was run three additional times; the table uses the median
  of those stable repeats. The raw first-pass records remain in the log.
- `†` The first-pass CK fused result for NN
  `nn-ctrl-1group-sum32818` was 376.8 TFLOPS, but three immediate repeats
  varied from 54.3 to 71.1 TFLOPS (median 59.8). The measurement is therefore
  unstable and its CK-fused-vs-hipBLASLt ratio is intentionally not reported.
  It should be remeasured on an idle device before drawing a comparison.
- `*` has the same equal-FLOP TN proxy meaning described above; it is not an
  exact-shape CK-original result.
- Refresh logs:
  `/tmp/grouped_gemm_fused_refresh_20261015/hipc_hipblaslt.log`,
  `/tmp/grouped_gemm_fused_refresh_20261015/ck_original.log`,
  `/tmp/grouped_gemm_fused_refresh_20261015/ck_fused_nt.log`,
  `/tmp/grouped_gemm_fused_refresh_20261015/ck_fused_nn.log`,
  `/tmp/grouped_gemm_fused_refresh_20261015/ck_fused_tn.log`, and
  `/tmp/grouped_gemm_fused_refresh_20261015/outlier_repeats.log`.

## Refresh — 2026-10-19 (hipBLASLt 1012)

Third measurement pass, appended after the `build/` tree was deleted and fully
reconfigured. The tables above are preserved unchanged. Device, dtype, warmup,
and timing parameters are identical to the earlier passes: HCU 5, BF16,
5 warmup iterations, and 20 timed iterations/samples.

What changed relative to the 2026-10-15 refresh:

- hipBLASLt install switched from
  `/root/workspace/hipblaslt-install-0820/hipblaslt-install` to
  `/root/workspace/hipblaslt-install-1012/hipblaslt-install`. This lifts the
  `hipblaslt tuned` column itself by +8.2%..+24.2%, so every `vs hipBLASLt`
  ratio here is only comparable within this section.
- Worktree change in the WASP A producer: the buffer descriptor is now built
  with `allow_swizzle=false` (`gemm_pipeline_ag_bg_cr_wasp.hpp`,
  `make_a_prod_src`), which disables `CACHE_SWIZZLE` on the A stream. Only the
  NT / CK fused path is affected.
- Build: fresh configure with `GPU_TARGETS=gfx936;gfx938` and
  `CK_TILE_GROUPED_GEMM_WASP_EPI_CYCLE_COUNT=OFF` (the previous `build/` cache
  had it `ON`, which instruments `s_memtime` in the consumer epilogue).
- Runtime: `LD_LIBRARY_PATH` was prefixed with the 1012 `lib` directory. The ELF
  `RUNPATH` also points there, but the ambient `/opt/dtk/lib` entry otherwise
  shadows it.

| 测事项 | shape | hipblaslt tuned | hipc | CK original | CK fused | hipc vs hipblaslt tuned | CK original vs hipblaslt tuned | CK fused vs hipblaslt tuned |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NT tp1-fc1-fwd | M=[8152,8152,8257,8257], N=4096, K=4096 | 358.4 | 343.6 | 250.1 | 372.9 | -4.1% | -30.2% | +4.1% |
| NT tp1-fc2-fwd | M=[8152,8152,8257,8257], N=2048, K=4096 | 323.9 | 343.0 | 246.7 | 370.2 | +5.9% | -23.8% | +14.3% |
| NT tp1-fc1-dgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | 352.7 | 339.1 | 247.2 | 367.5 | -3.8% | -29.9% | +4.2% |
| NT tp1-fc2-dgrad | M=[8100,8100,8091,8091], N=4096, K=2048 | 309.4 | 303.9 | 222.3 | 321.0 | -1.8% | -28.2% | +3.8% |
| NT tp1-wgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | 353.2 | 339.1 | 247.4 | 367.5 | -4.0% | -30.0% | +4.0% |
| NT tp4-real | M=[1923,2047,2179,2043], N=4096, K=4096 | 315.6 | 318.7 | 231.6 | 340.5 | +1.0% | -26.6% | +7.9% |
| NT tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | 314.5 | 313.9 | 231.4 | 336.2 | -0.2% | -26.4% | +6.9% |
| NT tp4-mean-2048-fc2 | M=[2048,2048,2048,2048], N=2048, K=4096 | 317.9 | 291.2 | 200.3 | 301.7 | -8.4% | -37.0% | -5.1% |
| NT tp4-step-2112 | M=[2112,2112,2112,2112], N=4096, K=4096 | 317.6 | 290.3 | 207.9 | 307.6 | -8.6% | -34.5% | -3.1% |
| NT tp4-collapse-1875 | M=[1875,1875,1875,1875], N=4096, K=4096 | 302.4 | 294.4 | 58.2 | 309.4 | -2.6% | -80.8% | +2.3% |
| NT tp4-dgrad-a-16384 | M=[16384], N=4096, K=4096 | 381.5 | 342.6 | 249.4 | 376.0 | -10.2% | -34.6% | -1.5% |
| NT ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 382.2 | 344.2 | 250.2 | 377.1 | -9.9% | -34.5% | -1.3% |
| NT ctrl-8group-4096 | M=[4096,4096,4096,4096,4096,4096,4096,4096], N=4096, K=4096 | 320.8 | 341.2 | 250.4 | 368.6 | +6.4% | -21.9% | +14.9% |
| NT ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | 320.3 | 315.7 | 230.4 | 344.9 | -1.4% | -28.1% | +7.7% |
| NN nn-tp1-fc1 | M=[8100,8100,8091,8091], N=4096, K=4096 | 371.5 | 370.0 | 166.9 | 396.3 | -0.4% | -55.1% | +6.7% |
| NN nn-tp1-fc1-k2048 | M=[8100,8100,8091,8091], N=4096, K=2048 | 330.2 | 331.2 | 156.4 | 353.2 | +0.3% | -52.6% | +6.9% |
| NN nn-tp1-fc2 | M=[8100,8100,8091,8091], N=2048, K=4096 | 324.9 | 325.0 | 165.8 | 396.6 | +0.0% | -49.0% | +22.1% |
| NN nn-tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | 326.3 | 326.7 | 156.1 | 365.9 | +0.1% | -52.2% | +12.1% |
| NN nn-ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | 352.8 | 326.4 | 155.7 | 368.5 | -7.5% | -55.9% | +4.4% |
| NN nn-ctrl-16group | M=[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048], N=4096, K=4096 | 327.0 | 326.9 | 168.4 | 395.3 | -0.0% | -48.5% | +20.9% |
| NN nn-ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 406.5 | 405.7 | 168.8 | 403.3 | -0.2% | -58.5% | -0.8% |
| TN tn-tp1-fc1-wgrad | M=[8100,8100,8091,8091], N=4096, K=4096 | N/A | 345.5 | 348.4* | 402.5 | N/A | N/A* | N/A |
| TN tn-tp1-fc2-wgrad | M=[8100,8100,8091,8091], N=2048, K=4096 | N/A | 325.5 | 321.2* | 400.0 | N/A | N/A* | N/A |
| TN tn-tp4-mean-2048 | M=[2048,2048,2048,2048], N=4096, K=4096 | N/A | 329.5 | 298.6* | 371.9 | N/A | N/A* | N/A |
| TN tn-ctrl-skew-10x | M=[5120,1536,1024,512], N=4096, K=4096 | N/A | 341.3 | 273.3* | 371.4 | N/A | N/A* | N/A |
| TN tn-ctrl-16group | M=[2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048], N=4096, K=4096 | N/A | 329.6 | 302.0* | 405.5 | N/A | N/A* | N/A |
| TN tn-ctrl-1group-sum32818 | M=[32818], N=4096, K=4096 | 413.1 | 401.0 | 296.0* | 407.8 | -2.9% | N/A* | -1.3% |

### Refresh notes (2026-10-19)

- CK fused NT, the only path touched by the worktree change, gained +4.1%..+9.0%
  over the 2026-10-15 refresh (mean +7.0%; all 14 NT shapes positive). NN and TN
  moved by at most 0.2%, consistent with a change scoped to the WASP A
  descriptor.
- Because the hipBLASLt baseline itself became faster, four NT shapes are now
  below it: `tp4-mean-2048-fc2` -5.1%, `tp4-step-2112` -3.1%,
  `tp4-dgrad-a-16384` -1.5%, `ctrl-1group-sum32818` -1.3%. These ratios must not
  be compared with the two earlier sections, which used the 0820 library.
- CK original reproduced the 2026-10-15 values to within 0.1 TFLOPS on every
  shape (e.g. NT `tp1-fc1-fwd` 250.1), a sanity check that the sweep itself is
  stable across the rebuild.
- The tuned hipBLASLt table still matched (`tuned:` lines appear in the HIPC
  log), but 1012 returns solution indices
  `10234/10240/10243/10252/5814/5827/6285/7300` instead of the `4095`/`4396`
  recorded above, so this column is a re-tuned baseline rather than the same
  plan.
- The sweep is performance-only. Correctness of the changed path was checked
  separately: NT ragged `M=[5120,1536,1024,512], N=512, K=1024` gave
  `wasp vs ck_ref max_abs=0.0625 bad=0`, and the NN smoke test passed.
- `*` has the same equal-FLOP TN proxy meaning described above.
- Logs: `example_hcu/ck_tile/19_grouped_gemm/perf-logs/20261019-182634/`
  (`hipc_hipblaslt.log`, `ck_original.log`, `ck_fused_nt.log`,
  `ck_fused_nn.log`, `ck_fused_tn.log`, `run_configuration.txt`).

## Experiment — 2026-10-20 (WASP Q2: 16 ds_reads then 4 buffer_loads)

Negative result, kept for the record: the Q2 issue order was tried and measured
slower. The tables above are unchanged.

- Change under test: in `c_I2` and `cp_I2`
  (`gemm_pipeline_ag_bg_cr_wasp.hpp`) the next-stage B prefetch
  (`load_B_stage`, four `buffer_load_dwordx4`) used to sit between the two A
  halves; it was moved after both halves, i.e. all 16 `ds_read_b128` first, then
  the 4 `buffer_load`s. The `s_waitcnt lgkmcnt(0)` before `mmac_half` is
  unchanged. Only the NT / WASP path is affected.
- Protocol: device 5, the two binaries differ only in this ordering (the
  pre-change build was kept aside). Three alternating sweep passes per binary
  (before, after, before, after, ...) over the 14 NT shapes, median per shape;
  `ITERS=20`, `WARMUP=5`; `LD_LIBRARY_PATH` prefixed with the 1012 hipBLASLt
  directory. Perf-only.

| NT shape | before (median) | after (median) | delta |
|---|---:|---:|---:|
| tp1-fc1-fwd | 372.8 | 360.2 | -3.4% |
| tp1-fc2-fwd | 370.1 | 353.5 | -4.5% |
| tp1-fc1-dgrad | 367.8 | 355.7 | -3.3% |
| tp1-fc2-dgrad | 321.8 | 317.3 | -1.4% |
| tp1-wgrad | 367.8 | 355.4 | -3.4% |
| tp4-real | 340.3 | 328.8 | -3.4% |
| tp4-mean-2048 | 331.4 | 323.7 | -2.3% |
| tp4-mean-2048-fc2 | 301.9 | 290.7 | -3.7% |
| tp4-step-2112 | 307.4 | 297.2 | -3.3% |
| tp4-collapse-1875 | 309.2 | 299.0 | -3.3% |
| tp4-dgrad-a-16384 | 376.0 | 363.9 | -3.2% |
| ctrl-1group-sum32818 | 376.4 | 364.9 | -3.1% |
| ctrl-8group-4096 | 368.3 | 355.1 | -3.6% |
| ctrl-skew-10x | 344.5 | 333.0 | -3.3% |

### Experiment notes (2026-10-20)

- Mean delta -3.2%, median -3.34%; no shape improves once medians are used
  (0/14). The spread within a configuration was <=0.5 TFLOPS except one
  first-pass `tp4-mean-2048` value of 85.8 TFLOPS (clock ramp) and a first-pass
  `tp1-wgrad` value of 343.1; both are discarded by the median.
- Consequence: the ordering is a regression and should not be adopted. The
  worktree still contains it at the time of writing; the recommendation is to
  restore the interleaved form (`a_read_half_low`, `load_B_stage`,
  `a_read_half_high`) unless a different interleaving is measured better.
- Correctness of the modified path: NT ragged
  `M=[5120,1536,1024,512], N=512, K=1024` gave `wasp vs ck_ref max_abs=0.0625
  bad=0` (`PASS`), so the result is a pure throughput effect.
- Logs: `example_hcu/ck_tile/19_grouped_gemm/perf-logs/ab-before-20261020/`,
  `ab-after-20261020/`, `ab-before-r2/`, `ab-after-r2/`, `ab-before-r3/`,
  `ab-after-r3/` (each contains `ck_fused_nt.log`).
