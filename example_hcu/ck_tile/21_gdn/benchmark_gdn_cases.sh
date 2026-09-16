#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
binary="${repo_root}/build/bin/tile_example_gdn_prefill"
warmup=10
repeat=100
head_dim=128
validation=1
device=""
output_csv=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --warmup) warmup="$2"; shift 2 ;;
    --repeat|--rep) repeat="$2"; shift 2 ;;
    --head-dim) head_dim="$2"; shift 2 ;;
    --validation) validation="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --binary) binary="$2"; shift 2 ;;
    --output-csv) output_csv="$2"; shift 2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${head_dim}" -ne 128 ]]; then
  echo "GDN currently supports only --head-dim 128" >&2
  exit 2
fi
if [[ "${validation}" != "0" && "${validation}" != "1" ]]; then
  echo "--validation must be 0 or 1" >&2
  exit 2
fi

if [[ ! -x "${binary}" ]]; then
  echo "binary not found: ${binary}" >&2
  echo "run: bash example_hcu/ck_tile/21_gdn/build.sh" >&2
  exit 2
fi
if [[ -n "${device}" ]]; then
  export HIP_VISIBLE_DEVICES="${device}"
fi

if [[ -n "${output_csv}" ]]; then
  printf 'suite,label,h,h_v,tokens,dtype,status,time_us\n' >"${output_csv}"
fi

failed=0
passed=0
declare -a res_suite=()
declare -a res_label=()
declare -a res_h=()
declare -a res_hv=()
declare -a res_tokens=()
declare -a res_dtype=()
declare -a res_status=()
declare -a res_time_us=()

record() {
  local suite="$1" label="$2" h="$3" hv="$4" tokens="$5"
  local dtype="$6" status="$7" time_us="$8"
  res_suite+=("${suite}")
  res_label+=("${label}")
  res_h+=("${h}")
  res_hv+=("${hv}")
  res_tokens+=("${tokens}")
  res_dtype+=("${dtype}")
  res_status+=("${status}")
  res_time_us+=("${time_us}")
  if [[ -n "${output_csv}" ]]; then
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${suite}" "${label}" "${h}" "${hv}" "${tokens}" \
      "${dtype}" "${status}" "${time_us}" >>"${output_csv}"
  fi
}

run_case() {
  local suite="$1" label="$2" h="$3" hv="$4" tokens="$5" dtype="$6"
  shift 6
  echo "===== ${suite}/${label} ====="
  local output
  if output="$("${binary}" \
      -t="${tokens}" -h="${h}" -hv="${hv}" -head_dim="${head_dim}" \
      -prec="${dtype}" -warmup="${warmup}" -repeat="${repeat}" \
      -v="${validation}" "$@" 2>&1)"; then
    printf '%s\n' "${output}"
    local time_us
    time_us="$(sed -n -E 's/.*gdn_prefill Avg Latency: [0-9.]+ ms \(([0-9.]+) us\).*/\1/p' <<<"${output}" | tail -n 1)"
    if [[ -z "${time_us}" ]]; then
      time_us="$(sed -n 's/^prefill_us://p' <<<"${output}" | tail -n 1)"
    fi
    local valid_ok=1
    if [[ "${validation}" == "1" ]] &&
       ! grep -q -E '(Verification \(gdn_prefill\): PASSED|^prefill valid:y$)' <<<"${output}"; then
      valid_ok=0
    fi
    if [[ -n "${time_us}" && "${valid_ok}" -eq 1 ]]; then
      record "${suite}" "${label}" "${h}" "${hv}" "${tokens}" \
        "${dtype}" "PASS" "${time_us}"
      passed=$((passed + 1))
    else
      record "${suite}" "${label}" "${h}" "${hv}" "${tokens}" \
        "${dtype}" "ERROR" "-"
      failed=$((failed + 1))
    fi
  else
    printf '%s\n' "${output}" >&2
    record "${suite}" "${label}" "${h}" "${hv}" "${tokens}" \
      "${dtype}" "ERROR" "-"
    failed=$((failed + 1))
  fi
}

# benchmark_gdn_ck.py: qwen suite, default l2norm_values=[False].
for tp in 1 2 4 8; do
  h=$((16 / tp))
  hv=$((64 / tp))
  for tokens in 1 16 64 256 1024 2048 4096 8192; do
    run_case qwen_tp "tp${tp}_T${tokens}" "${h}" "${hv}" "${tokens}" bf16 \
      -has_initial_state=1 -store_final_state=1
  done
done

# benchmark_gdn_ck.py: branches suite.
run_case branches baseline_gva_16_64_1x4096 16 64 4096 bf16
run_case branches dtype_fp16_gva_16_64 16 64 4096 fp16
run_case branches dtype_bf16_gva_16_64 16 64 4096 bf16
run_case branches no_init_state_gva_16_64 16 64 4096 bf16 \
  -has_initial_state=0
run_case branches no_final_state_gva_16_64 16 64 4096 bf16 \
  -store_final_state=0
run_case branches neutral_gates_gva_16_64 16 64 4096 bf16 \
  -gates=0 -input_scale=0.02
run_case branches l2norm_gva_16_64 16 64 4096 bf16 \
  -qk_l2norm=1

run_case branches beta_sigmoid_kernel_gva_16_64 16 64 4096 bf16 \
  -beta_sigmoid_in_kernel=1
run_case branches gate_in_kernel_gva_16_64 16 64 4096 bf16 \
  -gate_in_kernel=1 -has_dt_bias=1 -input_scale=0.02

run_case branches scale_1.0_gva_16_64 16 64 4096 bf16 -scale=1.0
run_case branches varlen_explicit_gva_16_64 16 64 4096 bf16 \
  -is_varlen=1
run_case branches varlen_2048+2048_gva_16_64 16 64 4096 bf16 \
  -seq_endpoints=2048,4096
run_case branches heads_gva_16_32 16 32 4096 bf16
run_case branches heads_sym_16_16 16 16 4096 bf16
run_case branches heads_sym_32_32 32 32 4096 bf16
run_case branches non_aligned_T63_gva_16_64 16 64 63 bf16
run_case branches non_aligned_T127_gva_16_64 16 64 127 bf16
run_case branches non_aligned_T511_gva_16_64 16 64 511 bf16

echo
echo "===== summary: passed=${passed}, failed=${failed} ====="
echo
printf '%-10s %-38s %5s %5s %8s %7s %8s %12s\n' \
  "suite" "label" "h" "h_v" "tokens" "dtype" "status" "time_us"
printf '%s\n' \
  "--------------------------------------------------------------------------------------------------------"
for ((i = 0; i < ${#res_suite[@]}; ++i)); do
  printf '%-10s %-38s %5s %5s %8s %7s %8s %12s\n' \
    "${res_suite[i]}" "${res_label[i]}" "${res_h[i]}" "${res_hv[i]}" \
    "${res_tokens[i]}" "${res_dtype[i]}" "${res_status[i]}" "${res_time_us[i]}"
done
echo
test "${failed}" -eq 0
