#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bin="${1:-${BUILD_DIR:-$repo_root/build}/bin}"
passed=0
failed=0
run() {
    echo "CASE $*"
    if "$@"; then passed=$((passed+1)); else failed=$((failed+1)); fi
}
reject() {
    echo "REJECT $*"
    local rc=0
    "$@" || rc=$?
    if [[ $rc == 1 ]]; then passed=$((passed+1)); else failed=$((failed+1)); fi
}
fwd="$bin/tile_example_sla_fwd"
bwd="$bin/tile_example_sla_bwd"
run "$bin/tile_example_sla_sizes"
for prec in fp16 bf16; do
    common=(-prec="$prec" -b=1 -h=1 -d=128 -v=1 -warmup=1 -repeat=1)
    for s in 64 256 1024; do
        run "$fwd" "${common[@]}" -s="$s" -linear_attn=1
        run "$fwd" "${common[@]}" -s="$s" -linear_attn=0
        run "$bwd" "${common[@]}" -s="$s"
    done
    for ratio in 0.01 1; do
        run "$fwd" "${common[@]}" -s=512 -topk_ratio="$ratio"
        run "$bwd" "${common[@]}" -s=512 -topk_ratio="$ratio"
    done
    for stages in 1 2 4 16; do
        run "$fwd" "${common[@]}" -s=1024 -kv_stages="$stages"
        run "$bwd" "${common[@]}" -s=1024 -kv_stages="$stages"
    done
    run "$fwd" "${common[@]}" -s=256 -b=2 -h=3
    run "$bwd" "${common[@]}" -s=256 -b=2 -h=3
    run "$fwd" "${common[@]}" -s=18048 -topk_ratio=0.01
    run "$bwd" "${common[@]}" -s=18048 -topk_ratio=0.01
    for exe in "$fwd" "$bwd"; do
        # Reject before HostTensor construction or hipMalloc, including b*h
        # itself overflowing and the first unrepresentable tensor size.
        for shape in '-b=128 -h=128 -s=2048' '-b=8192 -h=1 -s=2048' '-b=65536 -h=65536 -s=64' '-b=2147483647 -h=2147483647 -s=131072'; do
            read -r -a flags <<< "$shape"
            reject "$exe" "${common[@]}" "${flags[@]}"
        done
        for option in '-s=65' '-d=64' '-b=0' '-topk_ratio=0' '-topk_ratio=1.1' '-kv_stages=17' '-warmup=0'; do
            reject "$exe" "${common[@]}" -s=64 "$option"
        done
    done
    reject "$fwd" "${common[@]}" -s=64 -linear_attn=2
done
run "$fwd" -prec=bf16 -b=1 -h=1 -s=75648 -topk_ratio=0.01 -linear_attn=0 -v=1 -warmup=1 -repeat=1
run "$bwd" -prec=bf16 -b=1 -h=1 -s=75648 -topk_ratio=0.01 -v=1 -warmup=1 -repeat=1
echo "SLA_MATRIX passed=$passed failed=$failed"
[[ $failed == 0 ]]
