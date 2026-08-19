#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MICROLLM_TASK_BUILD_DIR="${MICROLLM_BUILD_DIR:-${PROJECT_DIR}/build}"

"$SCRIPT_DIR/build.sh"
ctest --test-dir "$MICROLLM_TASK_BUILD_DIR" --output-on-failure "$@"
