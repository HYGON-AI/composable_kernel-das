#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

set -eu -o pipefail
env_type=${1:?usage: run_full_test.sh environment branch host architecture build-directory}
branch=${2:?branch required}
host_name=${3:?host required}
gpu_arch=${4:?architecture required}
build=${5:-build-gemm}
printf '%s %s %s %s\n' "$env_type" "$branch" "$host_name" "$gpu_arch"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$script_dir/smoke_test.sh" "$build"
exe="$build/bin/tile_example_gemm_universal"

for pipeline_scheduler in compv3:intrawave compv4:intrawave mem:intrawave mem:interwave; do
    for prec in fp16 bf16; do
        for layout in C R; do
            for epilogue in default cshuffle; do
                args=(-pipeline="${pipeline_scheduler%:*}" -scheduler="${pipeline_scheduler#*:}"
                      -prec="$prec" -b_layout="$layout" -epilogue="$epilogue"
                      -split_k=1 -v=1 -warmup=0 -repeat=0)
                for k in 32 64 96 128 160 352; do
                    "$exe" "${args[@]}" -m=256 -n=256 -k="$k"
                done
                if [[ "$layout" = R ]]; then ldb=288; else ldb=80; fi
                "$exe" "${args[@]}" -m=264 -n=280 -k=72 -stride_a=80 -stride_b="$ldb" -stride_c=288
            done
        done
    done
done

for option in -split_k=2 -stride_a=1 -stride_b=129 -pipeline=unknown -b=1; do
    if "$exe" "$option"; then
        printf 'Expected failure for %s\n' "$option" >&2
        exit 1
    fi
done
