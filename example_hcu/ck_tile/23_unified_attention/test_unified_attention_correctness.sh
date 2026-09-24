#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bin="${1:-${BUILD_DIR:-$repo_root/build}/bin}"
passed=0; failed=0
run() {
    echo "CASE $*"
    if "$@"; then passed=$((passed+1)); else failed=$((failed+1)); fi
}
reject() {
    echo "REJECT $*"
    local rc=0
    "$@" || rc=$?
    if [[ $rc == 2 ]]; then passed=$((passed+1)); else failed=$((failed+1)); fi
}
pre="$bin/tile_example_unified_attention_2d"
dec="$bin/tile_example_unified_attention_3d"
run "$bin/tile_example_unified_attention_args"
for prec in fp16 bf16; do
 for d in 192 256; do
    common=(-prec="$prec" -d="$d" -h=8 -hkv=2 -v=1 -warmup=1 -repeat=1)
    for nq in 1 63 64 65 128; do
        run "$pre" "${common[@]}" -nq="$nq" -nkv=256
    done
    # Exercise the tiny kernel within its key capacity and the longer fallback.
    for lengths in '1 1' '16 16' '16 64' '1 65' '64 64'; do
        read -r nq nkv <<< "$lengths"
        run "$pre" "${common[@]}" -nq="$nq" -nkv="$nkv"
    done
    run "$pre" "${common[@]}" -nq=16 -nkv=64 -sinks=1 -qq_bias=1 -softcap=0.02
    run "$pre" "${common[@]}" -nq=64 -nkv=256 -h=4 -hkv=4
    run "$pre" "${common[@]}" -nq=64 -nkv=256 -h=16 -hkv=1
    for page in 16 32; do
        run "$pre" "${common[@]}" -nq=65 -nkv=257 -block_size="$page"
    done
    for options in '-sinks=1 -sink_value=3' '-alibi=1' '-alibi=1 -alibi_sqrt=1' '-qq_bias=1 -qq_bias_value=3' '-softcap=0.02' '-mask=sliding -sliding_window=17' '-mask=prefix -prefix_end=32' '-mask=sliding_prefix -sliding_window=17 -prefix_end=32'; do
        read -r -a flags <<< "$options"
        run "$pre" "${common[@]}" -nq=64 -nkv=64 "${flags[@]}"
    done
    run "$pre" "${common[@]}" -query_lens=17,65 -kv_lens=63,129 -block_size=32
    run "$pre" "${common[@]}" -query_lens=17,65 -kv_lens=63,129 -mask=sliding -sliding_window=17
    run "$pre" "${common[@]}" -query_lens=17,65 -kv_lens=17,65 -mask=sliding_prefix -sliding_window=17 -prefix_end=8 -sinks=1 -alibi=1 -qq_bias=1 -softcap=0.02
    run "$pre" "${common[@]}" -nq=256 -nkv=2048
    run "$pre" "${common[@]}" -nq=64 -nkv=256 -kv_cache_blocks=80 -block_table_width=16
    for page in 16 32; do
        for nkv in 1 31 257; do
            run "$dec" "${common[@]}" -b=2 -nkv="$nkv" -block_size="$page"
        done
    done
    for segments in 1 8; do
        run "$dec" "${common[@]}" -b=2 -nkv=512 -segments="$segments"
    done
    for options in '-sinks=1 -sink_value=3' '-alibi=1' '-alibi=1 -alibi_sqrt=1' '-qq_bias=1 -qq_bias_value=3' '-softcap=0.02' '-mask=sliding -sliding_window=17' '-mask=prefix -prefix_end=32' '-mask=sliding_prefix -sliding_window=17 -prefix_end=32'; do
        read -r -a flags <<< "$options"
        run "$dec" "${common[@]}" -b=2 -nkv=257 "${flags[@]}"
    done
    run "$dec" "${common[@]}" -b=2 -nkv=257 -mask=sliding_prefix -sliding_window=17 -prefix_end=32 -sinks=1 -alibi=1 -qq_bias=1 -softcap=0.02
    for exe in "$pre" "$dec"; do
        for option in '-d=128' '-causal=0' '-h=3' '-mask=invalid' '-mask=sliding' '-alibi_sqrt=1' '-kv_cache_blocks=1' '-block_table_width=1'; do
            reject "$exe" "${common[@]}" -nkv=256 "$option"
        done
    done
    for options in '-block_size=0' '-nq=0' '-nq=257' '-query_lens=17,65 -kv_lens=128' '-query_lens=17,-1 -kv_lens=128,128' '-query_lens=17,65 -kv_lens=16,128'; do
        read -r -a flags <<< "$options"
        reject "$pre" "${common[@]}" -nkv=256 "${flags[@]}"
    done
    reject "$dec" "${common[@]}" -block_size=64
    for name in query_lens kv_lens; do
        for value in '17,0' '17,-1' '17,abc' '17,2147483648' '17,65x' ',17' '17,' '17,,65' ','; do
            reject "$pre" "${common[@]}" -nkv=256 "-$name=$value"
        done
    done
    reject "$pre" "${common[@]}" -query_lens=17,65 -kv_lens=128,128,128
    reject "$dec" "${common[@]}" -segments=999
 done
done
echo "UNIFIED_ATTENTION_MATRIX passed=$passed failed=$failed"
[[ $failed == 0 ]]
