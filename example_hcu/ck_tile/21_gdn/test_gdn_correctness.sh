#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Run on a selected HCU: bash test_gdn_correctness.sh /path/to/build/bin
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
    local expected="$1"; shift
    echo "REJECT $*"
    set +e
    "$@"
    rc=$?
    set -e
    if [[ $rc == "$expected" ]]; then passed=$((passed+1)); else failed=$((failed+1)); fi
}
for prec in fp16 bf16; do
    common=(-prec="$prec" -head_dim=128 -h=2 -hv=8 -v=1 -warmup=0 -repeat=1)
    pre="$bin/tile_example_gdn_prefill"
    dec="$bin/tile_example_gdn_decode"
    for t in 1 63 64 65 127 256 1024 1025; do
        run "$pre" "${common[@]}" -t="$t"
    done
    for hv in 2 4; do run "$pre" "${common[@]}" -t=128 -hv="$hv"; done
    run "$pre" "${common[@]}" -t=511 -h=8 -hv=32
    run "$pre" "${common[@]}" -t=1024 -h=8 -hv=16
    run "$pre" "${common[@]}" -t=1024 -h=16 -hv=16
    run "$pre" "${common[@]}" -t=4096
    for option in '-has_initial_state=0' '-store_final_state=0' '-save_new_value=0' '-gates=0' '-qk_l2norm=1' '-beta_sigmoid_in_kernel=1' '-gate_in_kernel=1' '-scale=1' '-transpose_state=1' '-use_g=0' '-use_gk=1' '-use_exp2=0'; do
        run "$pre" "${common[@]}" -t=128 "$option"
    done
    run "$pre" "${common[@]}" -seq_endpoints=63,128,257
    run "$pre" "${common[@]}" -seq_endpoints=64,192 -has_initial_state=0 -qk_l2norm=1
    for option in '-has_initial_state=0' '-store_final_state=0' '-save_new_value=0' '-transpose_state=1' '-use_exp2=0' '-use_g=0' '-gate_in_kernel=1' '-beta_sigmoid_in_kernel=1' '-qk_l2norm=1'; do
        run "$pre" "${common[@]}" -seq_endpoints=63,128,257 -use_gk=1 "$option"
    done
    reject 3 "$pre" "${common[@]}" -t=128 -cp_context=1 -use_gk=1
    for t in 64 128 256; do
        for option in '-use_g=1' '-use_g=0' '-use_gk=1'; do
            run "$pre" "${common[@]}" -t="$t" -gates=0 -input_scale=0.2 "$option"
        done
    done
    for rank in 0 1 2; do
        run "$pre" "${common[@]}" -t=128 -cp_context=1 -cp_world_size=3 -cp_rank="$rank"
    done
    run "$dec" "${common[@]}" -b=1 -t=1
    run "$dec" "${common[@]}" -b=4 -t=1
    run "$dec" "${common[@]}" -b=2 -t=4
    for option in '-use_initial_state=0' '-store_final_state=0' '-use_qk_l2norm=1' '-use_g=0' '-use_gk=1' '-use_gv=1' '-beta_headwise=0' '-use_exp2=1' '-transpose_state=1' '-gate_in_kernel=1'; do
        run "$dec" "${common[@]}" -t=4 "$option"
    done
    run "$dec" "${common[@]}" -gate_in_kernel=1 -has_dt_bias=1 -t=4
    run "$dec" "${common[@]}" -seq_endpoints=3,8 -transpose_state=1
    run "$dec" "${common[@]}" -seq_endpoints=63,128 -use_qk_l2norm=1 -use_gk=1 -use_gv=1
    reject 2 "$pre" "${common[@]}" -head_dim=64
    reject 2 "$pre" "${common[@]}" -hv=6
    reject 2 "$pre" "${common[@]}" -seq_endpoints=64,32
    reject 3 "$pre" "${common[@]}" -t=65 -cp_context=1
    reject 2 "$dec" "${common[@]}" -head_dim=64
    reject 2 "$dec" "${common[@]}" -b=2 -seq_endpoints=4,8
    reject 2 "$dec" "${common[@]}" -seq_endpoints=8,4
done
# Repeat the former LDS-reuse race with and without the optimized state path.
for trial in 1 2 3; do
    for option in '-use_g=1' '-use_g=0' '-has_initial_state=0' '-transpose_state=1'; do
        run "$bin/tile_example_gdn_prefill" -prec=fp16 -t=256 -h=2 -hv=8 \
            -gates=0 -input_scale=0.2 -v=1 -warmup=0 -repeat=1 "$option"
    done
done

# Exercise asynchronous prefix-summary loading at internal state-group boundaries.
for trial in 1 2 3 4 5 6 7 8; do
    for tokens in 4096 8192; do
        for initial in 0 1; do
            run "$pre" -prec=bf16 -t="$tokens" -h=2 -hv=8 \
                -has_initial_state="$initial" -seed="$((41 + trial))" \
                -v=1 -warmup=0 -repeat=1
        done
    done
done

echo "GDN_MATRIX passed=$passed failed=$failed"
[[ $failed == 0 ]]
