#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MICROLLM_TASK_MODE="${1:-cpu}"
MICROLLM_TASK_BUILD_DIR="${MICROLLM_BUILD_DIR:-${PROJECT_DIR}/build}"

case "$MICROLLM_TASK_MODE" in
    cpu|hip|rccl) ;;
    *) echo "usage: $0 [cpu|hip|rccl]" >&2; exit 2 ;;
esac

python3 - "$PROJECT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
count = 0
for path in sorted((root / "benchmarks" / "results").rglob("*.json")):
    json.loads(path.read_text())
    count += 1
for path in sorted((root / "benchmarks" / "results").rglob("*.jsonl")):
    for line in path.read_text().splitlines():
        if line.strip():
            json.loads(line)
            count += 1
print(f"validated_json_records={count}")
PY

"$SCRIPT_DIR/build.sh"
case "$MICROLLM_TASK_MODE" in
    cpu)
        ctest --test-dir "$MICROLLM_TASK_BUILD_DIR" --output-on-failure -L cpu
        ;;
    hip)
        ctest --test-dir "$MICROLLM_TASK_BUILD_DIR" --output-on-failure -L hip
        ;;
    rccl)
        ctest --test-dir "$MICROLLM_TASK_BUILD_DIR" --output-on-failure -L rccl
        ;;
esac

echo "evidence_mode=${MICROLLM_TASK_MODE} status=pass"
