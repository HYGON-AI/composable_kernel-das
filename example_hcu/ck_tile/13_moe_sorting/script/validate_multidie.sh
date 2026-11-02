#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
set -euo pipefail

# Run the existing CK example and its CPU reference once per case. No AITER,
# generated input files, or timing loop is required. Optional case selection
# allows each Perf Model invocation to use an independent output directory.
exe=${1:?usage: bash validate_multidie.sh EXE [CASE|all|list]}
selection=${2:-all}
cases=(oneshot_min oneshot_buffer oneshot_mask oneshot_bounds oneshot_invalid
       oneshot_padding mp_valid mp_mask_bounds mp_tail mp_invalid)
if [[ "$selection" == list ]]; then
    printf '%s\n' "${cases[@]}"
    exit 0
fi
if [[ ! -x "$exe" ]]; then
    echo "Executable not found: $exe" >&2
    exit 2
fi
run_case() {
    local name=$1
    local -a args
    case "$name" in
        oneshot_min) args=(-t=1 -e=1 -k=1 -unit=1) ;;
        oneshot_buffer) args=(-t=4 -e=8 -k=2 -unit=4 -moe_buf_size=4096) ;;
        oneshot_mask) args=(-t=9 -e=16 -k=4 -unit=8 -local_eid=0,2,7,9,15) ;;
        oneshot_bounds) args=(-t=7 -e=8 -k=3 -unit=3 -invalid=4 -moe_buf_size=64) ;;
        oneshot_invalid) args=(-t=3 -e=8 -k=2 -unit=8 -invalid=1 -moe_buf_size=64) ;;
        oneshot_padding) args=(-t=8 -e=8 -k=2 -unit=4 -invalid=3 -moe_buf_size=64) ;;
        mp_valid) args=(-t=256 -e=64 -k=8 -unit=32 -moe_buf_size=4096) ;;
        mp_mask_bounds) args=(-t=257 -e=64 -k=2 -unit=8 -invalid=4 -local_eid=0,2,7,31,63) ;;
        mp_tail) args=(-t=1025 -e=16 -k=2 -unit=32) ;;
        mp_invalid) args=(-t=257 -e=64 -k=2 -unit=8 -invalid=1) ;;
        *) echo "Unknown case: $name" >&2; return 2 ;;
    esac
    echo "CASE $name"
    "$exe" -v=1 -pr_i=int32 -pr_w=fp32 -seed=20261102 -time=0 -warmup=0 -repeat=1 -kname=1 "${args[@]}"
}
if [[ "$selection" == all ]]; then
    for name in "${cases[@]}"; do run_case "$name"; done
else
    run_case "$selection"
fi
