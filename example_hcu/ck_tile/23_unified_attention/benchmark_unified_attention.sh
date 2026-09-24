#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build}"
bin2d="${build_dir}/bin/tile_example_unified_attention_2d"
bin3d="${build_dir}/bin/tile_example_unified_attention_3d"

suite="all"
warmup=10
repeat=100
verify=1
for arg in "$@"; do
    case "${arg}" in
        -suite=*) suite="${arg#*=}" ;;
        -warmup=*) warmup="${arg#*=}" ;;
        -repeat=*) repeat="${arg#*=}" ;;
        -v=*) verify="${arg#*=}" ;;
        *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ "${suite}" != "all" && "${suite}" != "mimo" && "${suite}" != "qwen" ]]; then
    echo "suite must be mimo, qwen, or all" >&2
    exit 2
fi
if [[ ! -x "${bin2d}" || ! -x "${bin3d}" ]]; then
    echo "build the examples first: bash example_hcu/ck_tile/23_unified_attention/build.sh" >&2
    exit 2
fi

passed=0
failed=0
declare -a result_case=()
declare -a result_nq=()
declare -a result_nkv=()
declare -a result_h=()
declare -a result_hkv=()
declare -a result_dim=()
declare -a result_time=()

run_case() {
    local label="$1" nq="$2" nkv="$3" h="$4" hkv="$5" dim="$6"
    shift 6
    local output time_ms
    echo "===== ${label} ====="
    if output="$("$@" -warmup="${warmup}" -repeat="${repeat}" -v="${verify}" 2>&1)"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
    printf '%s\n' "${output}"
    time_ms="$(printf '%s\n' "${output}" | sed -n -E 's/.*unified_attention_[23]d Avg Latency: ([0-9.]+) ms.*/\1/p' | tail -n 1)"
    if [[ -z "${time_ms}" ]]; then
        time_ms="$(printf '%s\n' "${output}" | sed -n 's/^ua_[23]d_ms://p' | tail -n 1)"
    fi
    result_case+=("${label}")
    result_nq+=("${nq}")
    result_nkv+=("${nkv}")
    result_h+=("${h}")
    result_hkv+=("${hkv}")
    result_dim+=("${dim}")
    result_time+=("${time_ms:--}")
}

run_2d() {
    local label="$1" nq="$2" nkv="$3" h="$4" hkv="$5" d="$6" prec="$7"
    shift 7
    run_case "${label}" "${nq}" "${nkv}" "${h}" "${hkv}" "${d}" \
        "${bin2d}" -nq="${nq}" -nkv="${nkv}" -h="${h}" \
        -hkv="${hkv}" -d="${d}" -prec="${prec}" "$@"
}

run_3d() {
    local label="$1" nkv="$2" h="$3" hkv="$4" d="$5" prec="$6" page="$7" segments="$8"
    shift 8
    run_case "${label}" 1 "${nkv}" "${h}" "${hkv}" "${d}" \
        "${bin3d}" -b=1 -nkv="${nkv}" -h="${h}" -hkv="${hkv}" \
        -d="${d}" -prec="${prec}" -block_size="${page}" -segments="${segments}" "$@"
}

run_mimo() {
    run_2d mimo/prefill_gqa_sink_swa 49 4096 16 2 192 fp16 \
        -block_size=16 -kv_cache_blocks=2048 -block_table_width=256 \
        -mask=sliding -sliding_window=128 -sinks=1
    run_2d mimo/prefill_mqa 49 4096 16 1 192 fp16 \
        -block_size=32 -kv_cache_blocks=2048 -block_table_width=128
    run_2d mimo/prefill_gqa_sink_swa_multibatch 563 4096 16 2 192 fp16 \
        -query_lens=3,49,511 -block_size=16 \
        -kv_cache_blocks=2048 -block_table_width=256 \
        -mask=sliding -sliding_window=128 -sinks=1
    run_2d mimo/prefill_mqa_multibatch 563 4096 16 1 192 fp16 \
        -query_lens=3,49,511 -block_size=32 \
        -kv_cache_blocks=2048 -block_table_width=128
    run_3d mimo/decode_gqa_sink_swa_3d 4096 16 2 192 fp16 16 16 \
        -kv_cache_blocks=2048 -block_table_width=256 \
        -mask=sliding -sliding_window=128 -sinks=1
    run_3d mimo/decode_mqa_3d 4096 16 1 192 fp16 32 16 \
        -kv_cache_blocks=2048 -block_table_width=128
    run_2d mimo/prefill_gqa_alibi 49 4096 16 2 192 fp16 \
        -block_size=16 -kv_cache_blocks=2048 -block_table_width=256 -alibi=1
    run_2d mimo/prefill_gqa_alibi_sqrt 49 4096 16 2 192 fp16 \
        -block_size=16 -kv_cache_blocks=2048 -block_table_width=256 \
        -alibi=1 -alibi_sqrt=1
    run_2d mimo/prefill_gqa_mm_prefix 49 4096 16 2 192 fp16 \
        -block_size=16 -kv_cache_blocks=2048 -block_table_width=260 \
        -mask=prefix -mm_prefix=1 -prefix_begin=4000 -prefix_end=4080
    run_3d mimo/decode_gqa_alibi_3d 4096 16 2 192 fp16 16 16 \
        -kv_cache_blocks=2048 -block_table_width=256 -alibi=1
}

run_qwen_split() {
    local label="$1" total="$2" seqs="$3" h="$4" hkv="$5" prec="$6"
    shift 6
    local base=$((total / seqs)) rem=$((total % seqs)) i nq max_nq=0 lens=""
    local page=800 cache_blocks=1271 table_width=58
    if ((h == 8)); then
        page=1056
        cache_blocks=2146
        table_width=44
    fi
    for ((i = 0; i < seqs; ++i)); do
        nq="${base}"
        if ((i < rem)); then nq=$((nq + 1)); fi
        if ((nq > max_nq)); then max_nq="${nq}"; fi
        lens="${lens}${lens:+,}${nq}"
    done
    if ((seqs > 1)); then
        run_2d "qwen/${label}" "${total}" "${max_nq}" "${h}" "${hkv}" 256 "${prec}" \
            -query_lens="${lens}" -block_size="${page}" \
            -kv_cache_blocks="${cache_blocks}" -block_table_width="${table_width}" "$@"
    else
        run_2d "qwen/${label}" "${total}" "${max_nq}" "${h}" "${hkv}" 256 "${prec}" \
            -block_size="${page}" -kv_cache_blocks="${cache_blocks}" \
            -block_table_width="${table_width}" "$@"
    fi
}

run_qwen() {
    run_qwen_split fp16-1seq-7392 7392 1 8 1 fp16
    run_qwen_split fp16-2seq-7392 7392 2 8 1 fp16
    run_qwen_split fp16-2seq-7264 7264 2 8 1 fp16
    run_qwen_split fp16-2seq-7395 7395 2 8 1 fp16
    run_qwen_split fp16-2seq-4227 4227 2 8 1 fp16
    run_qwen_split fp16-2seq-1987 1987 2 8 1 fp16
    run_qwen_split fp16-2seq-6 6 2 8 1 fp16
    run_qwen_split fp16-1seq-3 3 1 8 1 fp16
    run_qwen_split bf16-1seq-8000 8000 1 12 2 bf16
    run_qwen_split bf16-1seq-7200 7200 1 12 2 bf16
    run_qwen_split bf16-1seq-800 800 1 12 2 bf16
    run_qwen_split bf16-1seq-3 3 1 12 2 bf16
    run_qwen_split fp16-2seq-alibi 1987 2 8 1 fp16 -alibi=1
    run_qwen_split bf16-1seq-alibi-sqrt 800 1 12 2 bf16 -alibi=1 -alibi_sqrt=1
    run_qwen_split bf16-1seq-mm-prefix 800 1 12 2 bf16 \
        -mask=prefix -mm_prefix=1 -prefix_begin=100 -prefix_end=500
}

if [[ "${suite}" == "all" || "${suite}" == "mimo" ]]; then run_mimo; fi
if [[ "${suite}" == "all" || "${suite}" == "qwen" ]]; then run_qwen; fi

echo "===== summary: passed=${passed}, failed=${failed} ====="
echo
printf '| %-38s | %6s | %6s | %4s | %4s | %5s | %10s |\n' \
    "Case" "nq" "nkv" "h" "hkv" "Dim" "Time (ms)"
printf '|:%-38s-|-%6s:|-%6s:|-%4s:|-%4s:|-%5s:|-%10s:|\n' \
    "--------------------------------------" "------" "------" "----" "----" "-----" "----------"
for ((i = 0; i < ${#result_case[@]}; ++i)); do
    printf '| %-38s | %6s | %6s | %4s | %4s | %5s | %10s |\n' \
        "${result_case[i]}" "${result_nq[i]}" "${result_nkv[i]}" \
        "${result_h[i]}" "${result_hkv[i]}" "${result_dim[i]}" "${result_time[i]}"
done
((failed == 0))
