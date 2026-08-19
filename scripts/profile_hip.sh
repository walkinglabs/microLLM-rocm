#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY -- APPLICATION [ARGUMENTS...]" >&2
    exit 2
fi

MICROLLM_TASK_OUTPUT_DIR="$1"
shift 2

if ! command -v rocprofv3 >/dev/null 2>&1; then
    echo "rocprofv3 was not found in PATH" >&2
    exit 1
fi

mkdir -p "$MICROLLM_TASK_OUTPUT_DIR"
rocprofv3 \
    --runtime-trace \
    --stats \
    --summary \
    --summary-per-domain \
    --summary-output-file summary \
    --output-directory "$MICROLLM_TASK_OUTPUT_DIR" \
    --output-format csv json pftrace \
    -- "$@"

echo "trace_directory=$MICROLLM_TASK_OUTPUT_DIR"
