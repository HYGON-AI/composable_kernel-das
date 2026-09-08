# ck_tile HCU migration and optimization notes

This note records the working pattern used while migrating and tuning ck_tile examples on Hygon HCU. Treat it as a lightweight skill for future ck_tile operator work.

## Workflow

1. Start from the closest `example_hcu/ck_tile` implementation, not the AMD upstream entry alone.
2. Record the current supported data types, shapes, validation status, latency, throughput, bandwidth, GPU id, and compiler before changing code.
3. Sync local edits to the remote workspace, then build and run inside the Docker container. Keep remote temporary scripts, profiles, ISA dumps, and logs under `hygon_tmp/`.
4. Before every benchmark, run `rocm-smi --showuse --showmemuse` and select a low-use HCU with `HIP_VISIBLE_DEVICES`.
5. Add one candidate at a time behind a `-config=` selector. Promote it to default only after a same-run benchmark beats the current champion and validates.
6. Keep older useful configs selectable for A/B comparison.
7. Every local commit for an optimized example should update that example's readme with the measured result.
8. For heavy ck_tile examples, add narrow `EXCLUDE_FROM_ALL` fast targets for one dtype and the common layout while keeping the full target as the release/coverage target.

## HCU ck_tile checks

- HCU grouped GEMM uses MMAC, not MFMA. Search `include/ck_tile` and nearby `example_hcu/ck_tile` code for an existing MMAC alias, dispatcher entry, LDS matrix-load shape, or inline asm before adding a new instruction path.
- For new data type or tile shapes, check both `warp_mmac_gemm.hpp` and `warp_mmac_gemm_dispatcher.hpp`.
- `K_Warp_Tile` changes can be a low-risk first experiment when the underlying MMAC implementation already supports K iteration through `WarpGemmAttribute*IterateK`.
- Persistent grouped GEMM may hide host-side group shape restrictions, but validation should still cover all requested layouts and SplitK behavior.
- Do not treat fp16 performance as valid until correctness is fixed; the current fp16 issue also appears in `03_gemm` for the same shape family.
- Build speed matters: turn resource reports, verbose makefiles, and `-save-temps` on only when inspecting resources or ISA. They should not be default for every iteration.
- If `-jN` does not help, check whether the target is one giant `.cpp`; split by dtype/layout or add fast targets before doing more kernel experiments.

## Benchmark template

Use the default grouped shapes unless testing a production shape:

```bash
rocm-smi --showuse --showmemuse
export HIP_VISIBLE_DEVICES=<idle_gpu>
cmake --build /wksp/ai/composable_kernel_hcu/build_dev --target tile_example_grouped_gemm -j8
/wksp/ai/composable_kernel_hcu/build_dev/bin/tile_example_grouped_gemm \
  -prec=<dtype> -warmup=10 -repeat=100 -validate=1 -group_count=8 -kbatch=1
```

Always record:

- commit or stage name
- GPU id and HCU status
- exact command
- validation result and thresholds when available
- latency, throughput, bandwidth
- speedup vs previous champion
- resource remarks from `-Rpass-analysis=kernel-resource-usage` when a new kernel shape is compiled
