#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MICROLLM_TASK_BUILD_DIR="${MICROLLM_BUILD_DIR:-${PROJECT_DIR}/build}"
MICROLLM_TASK_DEVICE="${MICROLLM_BENCH_DEVICE:-cpu}"
MICROLLM_TASK_OUTPUT="${MICROLLM_BENCH_OUTPUT:-${PROJECT_DIR}/benchmark-results.jsonl}"

"$SCRIPT_DIR/build.sh"
: > "$MICROLLM_TASK_OUTPUT"
for operation in add softmax matmul; do
    size=256
    if [[ "$operation" == "add" ]]; then size=1048576; fi
    "$MICROLLM_TASK_BUILD_DIR/benchmarks/microllm_bench_ops" \
        --op "$operation" --size "$size" --warmup 5 --repetitions 20 \
        --device "$MICROLLM_TASK_DEVICE" >> "$MICROLLM_TASK_OUTPUT"
done
echo "wrote $MICROLLM_TASK_OUTPUT"
