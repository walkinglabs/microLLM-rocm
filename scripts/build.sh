#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MICROLLM_TASK_BUILD_DIR="${MICROLLM_BUILD_DIR:-${PROJECT_DIR}/build}"
MICROLLM_TASK_JOBS="${MICROLLM_JOBS:-$(nproc 2>/dev/null || echo 4)}"

if [[ ! -f "${MICROLLM_TASK_BUILD_DIR}/CMakeCache.txt" ]]; then
    "$SCRIPT_DIR/configure.sh"
fi
cmake --build "$MICROLLM_TASK_BUILD_DIR" --parallel "$MICROLLM_TASK_JOBS" "$@"
