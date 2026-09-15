<!-- Copyright (c) 2026 Hygon Information Technology Co., Ltd. -->
<!-- SPDX-License-Identifier: MIT -->

# HCU grouped convolution with MultipleD

These examples instantiate `DeviceGroupedConvFwdMultipleD_Xdl_CShuffle` for
FP16, BF16 and FP32 on gfx936/gfx938, using native MMAC and FP32 accumulation
and shuffle. `DeviceConv<T, Dim, Fused>` in `grouped_conv_fwd_common.hpp`
contains the complete configuration.

After configuring CK with HCU examples enabled:

```sh
cmake --build build --target example_hcu_grouped_conv_fwd_multiple_d -j3
for dtype in f16 bf16 f32; do
    ./build/bin/example_hcu_grouped_conv_fwd_multiple_d_${dtype} || exit 1
done
```

Each executable runs 12 numerical cases against the CPU convolution reference:
six shapes with an empty D tuple and with `E = relu(Conv(A,B) + bias) + residual`.
The bias broadcasts across batch/spatial dimensions; residual is a full output
tensor with independently initialized values. The matrix covers 1D/2D/3D,
groups 1/2/3/4, 1x1 and larger filters, asymmetric padding, stride, dilation,
and M/N/K block tails. Input/output physical storage uses channel-contiguous
spatial/group ordering, through CK's existing general strided descriptors.

The CPU reference accumulates in FP32, applies the epilogue explicitly, and
converts to the output type. FP16, BF16 and FP32 comparisons use
`rtol=atol=1e-3`, `1e-2` and `1e-5`, respectively. All cases must print `PASS`
and the executable must return zero. The configuration requires C divisible
by 8 for FP16/BF16 (4 for FP32) and output K divisible by 4; always call
`IsSupportedArgument` before launch. Timing and downstream Torch integration
are outside this correctness matrix.
