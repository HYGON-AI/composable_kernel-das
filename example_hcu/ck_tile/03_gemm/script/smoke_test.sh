#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -eu -o pipefail
build=${1:?usage: smoke_test.sh build-directory}
basic="$build/bin/tile_example_gemm_basic"
universal="$build/bin/tile_example_gemm_universal"

basic_result=$("$basic" -b=1 -m=128 -n=128 -k=64 -v=1 -warmup=0 -repeat=1)
printf '%s\n' "$basic_result"
[[ "$basic_result" == *"CPU veification result is: correct"* ]]

for pipeline in compv3 compv4 mem; do
    for precision in fp16 bf16; do
        "$universal" -pipeline="$pipeline" -prec="$precision" \
            -m=256 -n=256 -k=128 -v=1 -warmup=0 -repeat=0 -split_k=1
    done
done
