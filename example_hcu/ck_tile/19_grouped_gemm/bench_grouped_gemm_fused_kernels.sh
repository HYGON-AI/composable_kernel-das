#!/usr/bin/env bash
# Reproduce the NT/NN/TN commands recorded in grouped_gemm_fused_kernels_perf.md.
#
# This script runs the four columns sequentially on one HCU:
#   1. tuned hipBLASLt + HIPC implementation
#   2. CK original device-args implementation
#   3. CK fused implementation (WASP for NT, v7/v8 for NN/TN)
#
# The workload source of truth is hipc_kernels/bench_grouped_shapes.sh. Do not
# duplicate the 27 shape strings here: this script extracts its SHAPES array.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

DEVICE=${DEVICE:-5}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}
SUITE=${SUITE:-all}
LAYOUT=${LAYOUT:-all}
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/perf-logs/$(date +%Y%m%d-%H%M%S)"}

SHAPES_SCRIPT="$SCRIPT_DIR/hipc_kernels/bench_grouped_shapes.sh"
HIPC_BIN=${HIPC_BIN:-"$SCRIPT_DIR/hipc_kernels/bf16_gemm"}
CK_ORIGINAL_BIN=${CK_ORIGINAL_BIN:-"$REPO_ROOT/build/bin/tile_example_grouped_gemm_device_args_benchmark"}
CK_FUSED_BIN=${CK_FUSED_BIN:-"$REPO_ROOT/build/bin/tile_example_grouped_gemm_fused_test"}
HIPBLASLT_TUNE_FILE=${HIPBLASLT_TUNE_FILE:-"/root/workspace/bf16_gemm/tune/hipblaslt_solutions.conf"}

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Reproduces the 27-shape NT/NN/TN performance sweep in
grouped_gemm_fused_kernels_perf.md.

Options:
  --device=N             HCU index (default: ${DEVICE})
  --warmup=N             Warmup iterations (default: ${WARMUP})
  --iters=N              Timed iterations/samples (default: ${ITERS})
  --suite=NAME           all, hipc, ck-original, or ck-fused (default: ${SUITE})
  --layout=NAME          all, NT, NN, or TN (default: ${LAYOUT})
  --log-dir=PATH         Directory for raw logs (default: ${LOG_DIR})
  --help                 Show this help.

Equivalent environment overrides are supported for DEVICE, WARMUP, ITERS,
SUITE, LAYOUT, LOG_DIR, HIPC_BIN, CK_ORIGINAL_BIN, CK_FUSED_BIN, and HIPBLASLT_TUNE_FILE.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --device=*) DEVICE=${arg#--device=} ;;
        --warmup=*) WARMUP=${arg#--warmup=} ;;
        --iters=*) ITERS=${arg#--iters=} ;;
        --suite=*) SUITE=${arg#--suite=} ;;
        --layout=*) LAYOUT=${arg#--layout=} ;;
        --log-dir=*) LOG_DIR=${arg#--log-dir=} ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

case "$SUITE" in
    all|hipc|ck-original|ck-fused) ;;
    *) echo "--suite must be all, hipc, ck-original, or ck-fused" >&2; exit 2 ;;
esac
case "$LAYOUT" in
    all|NT|NN|TN) ;;
    *) echo "--layout must be all, NT, NN, or TN" >&2; exit 2 ;;
esac

require_executable() {
    local path=$1
    if [[ ! -x "$path" ]]; then
        echo "Required benchmark binary is not executable: $path" >&2
        exit 2
    fi
}

if [[ ! -f "$SHAPES_SCRIPT" ]]; then
    echo "Shape source does not exist: $SHAPES_SCRIPT" >&2
    exit 2
fi

if [[ "$SUITE" == all || "$SUITE" == hipc ]]; then
    require_executable "$HIPC_BIN"
    if [[ ! -f "$HIPBLASLT_TUNE_FILE" ]]; then
        echo "Tuned hipBLASLt solution file does not exist: $HIPBLASLT_TUNE_FILE" >&2
        exit 2
    fi
fi
if [[ "$SUITE" == all || "$SUITE" == ck-original ]]; then
    require_executable "$CK_ORIGINAL_BIN"
fi
if [[ "$SUITE" == all || "$SUITE" == ck-fused ]]; then
    require_executable "$CK_FUSED_BIN"
fi

mapfile -t SHAPES < <(
    awk '
        /^SHAPES=\(/ { in_shapes = 1; next }
        in_shapes && /^\)/ { exit }
        in_shapes && /^[[:space:]]*"/ {
            line = $0
            sub(/^[[:space:]]*"/, "", line)
            sub(/".*$/, "", line)
            print line
        }
    ' "$SHAPES_SCRIPT"
)

if [[ ${#SHAPES[@]} -ne 27 ]]; then
    echo "Expected 27 shapes from $SHAPES_SCRIPT, found ${#SHAPES[@]}" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"

run_logged() {
    local log=$1
    shift

    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    printf '$'
    printf ' %q' "$@"
    printf '\n' >>"$log"
    "$@" 2>&1 | tee -a "$log"
}

run_hipc() {
    local layout=$1 tag=$2 n=$3 k=$4 lengths=$5
    local log="$LOG_DIR/hipc_hipblaslt.log"

    printf '\n--- %s %s N=%s K=%s M=%s ---\n' \
        "$layout" "$tag" "$n" "$k" "$lengths" | tee -a "$log"
    BF16GEMM_HIPBLASLT_SOLUTION_FILE="$HIPBLASLT_TUNE_FILE" \
        CUDA_VISIBLE_DEVICES="$DEVICE" \
        run_logged "$log" "$HIPC_BIN" \
        "--layout=$layout" "--grouped-length=$lengths" "--n=$n" "--k=$k" \
        --bench=1 "--warmup-iters=$WARMUP" "--bench-iters=$ITERS"
}

run_ck_original() {
    local layout=$1 tag=$2 n=$3 k=$4 lengths=$5
    local total_m=0 part
    local log="$LOG_DIR/ck_original.log"

    IFS=',' read -r -a group_lengths <<<"$lengths"
    for part in "${group_lengths[@]}"; do total_m=$((total_m + part)); done

    # The device-args interface models TN group lengths as per-group K. The
    # report therefore used this equal-FLOP proxy (M=4096) for TN, not an
    # exact-shape CK-original comparison.
    local m=1
    if [[ "$layout" == TN ]]; then m=4096; fi

    printf '\n--- %s %s N=%s K=%s lengths=%s ---\n' \
        "$layout" "$tag" "$n" "$k" "$lengths" | tee -a "$log"
    CUDA_VISIBLE_DEVICES="$DEVICE" \
        run_logged "$log" "$CK_ORIGINAL_BIN" \
        "--m=$m" "--n=$n" "--k=$k" "--group-lengths=$lengths" \
        "--variable-capacity=$total_m" --dtype=bf16 "--layout=$layout" \
        --backend=ck --time=1 "--warmup=$WARMUP" "--samples=$ITERS" \
        --validate=0 --include-args-build=0
}

run_ck_fused() {
    local layout=$1 tag=$2 n=$3 k=$4 lengths=$5
    local first_m group_count

    IFS=',' read -r -a group_lengths <<<"$lengths"
    first_m=${group_lengths[0]}
    group_count=${#group_lengths[@]}

    if [[ "$layout" == NT ]]; then
        local log="$LOG_DIR/ck_fused_nt.log"
        printf '\n--- %s %s N=%s K=%s M=%s ---\n' \
            "$layout" "$tag" "$n" "$k" "$lengths" | tee -a "$log"
        CUDA_VISIBLE_DEVICES="$DEVICE" \
            run_logged "$log" "$CK_FUSED_BIN" \
            --layout=NT "--grouped-length=$lengths" "--n=$n" "--k=$k" \
            --bench=1 "--warmup-iters=$WARMUP" "--bench-iters=$ITERS"
        return
    fi

    local expected_instance=5000
    if [[ "$layout" == TN ]]; then expected_instance=5001; fi
    local log="$LOG_DIR/ck_fused_${layout,,}.log"
    printf '\n--- %s %s N=%s K=%s M=%s ---\n' \
        "$layout" "$tag" "$n" "$k" "$lengths" | tee -a "$log"
    local log_start
    log_start=$(wc -l <"$log")
    CUDA_VISIBLE_DEVICES="$DEVICE" \
        run_logged "$log" "$CK_FUSED_BIN" \
        "--layout=$layout" "--m=$first_m" "--n=$n" "--k=$k" \
        "--groups=$group_count" "--group-lengths=$lengths" \
        "--iters=$ITERS" "--warmup=$WARMUP" --check=0 --ref=0 --cpu-ref=0

    if ! tail -n "+$((log_start + 1))" "$log" | grep -q \
        "select layout=$layout instance_id=$expected_instance expected=$expected_instance"; then
        echo "Expected fused $layout instance $expected_instance was not selected for $tag" >&2
        exit 1
    fi
}

printf 'device=%s warmup=%s iters=%s suite=%s layout=%s\n' \
    "$DEVICE" "$WARMUP" "$ITERS" "$SUITE" "$LAYOUT" | tee "$LOG_DIR/run_configuration.txt"
printf 'shape_source=%s\nhipblaslt_tune=%s\n' \
    "$SHAPES_SCRIPT" "$HIPBLASLT_TUNE_FILE" | tee -a "$LOG_DIR/run_configuration.txt"

for entry in "${SHAPES[@]}"; do
    IFS='|' read -r layout tag n k lengths description <<<"$entry"
    if [[ "$LAYOUT" != all && "$LAYOUT" != "$layout" ]]; then continue; fi

    if [[ "$SUITE" == all || "$SUITE" == hipc ]]; then
        run_hipc "$layout" "$tag" "$n" "$k" "$lengths"
    fi
    if [[ "$SUITE" == all || "$SUITE" == ck-original ]]; then
        run_ck_original "$layout" "$tag" "$n" "$k" "$lengths"
    fi
    if [[ "$SUITE" == all || "$SUITE" == ck-fused ]]; then
        run_ck_fused "$layout" "$tag" "$n" "$k" "$lengths"
    fi
done

echo "Completed selected sweep. Raw logs: $LOG_DIR"
