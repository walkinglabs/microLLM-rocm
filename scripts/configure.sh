#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MICROLLM_TASK_BUILD_DIR="${MICROLLM_BUILD_DIR:-${PROJECT_DIR}/build}"

cmake -B "$MICROLLM_TASK_BUILD_DIR" -S "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE="${MICROLLM_BUILD_TYPE:-Debug}" \
    "$@"
