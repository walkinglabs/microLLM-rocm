#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MICROLLM_TASK_OUTPUT="${MICROLLM_TINYSTORIES_OUTPUT:-${PROJECT_DIR}/data/TinyStories-valid-smoke.txt}"
MICROLLM_TASK_REVISION="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
MICROLLM_TASK_BYTES="${MICROLLM_TINYSTORIES_BYTES:-1048576}"
MICROLLM_TASK_SPLIT="${MICROLLM_TINYSTORIES_SPLIT:-validation}"

if [[ "$MICROLLM_TASK_BYTES" -le 0 ]]; then
    echo "MICROLLM_TINYSTORIES_BYTES must be positive" >&2
    exit 2
fi
case "$MICROLLM_TASK_SPLIT" in
    train) MICROLLM_TASK_FILE="TinyStories-train.txt" ;;
    validation) MICROLLM_TASK_FILE="TinyStories-valid.txt" ;;
    *) echo "MICROLLM_TINYSTORIES_SPLIT must be train or validation" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$MICROLLM_TASK_OUTPUT")"
curl --location --fail --show-error \
    --range "0-$((MICROLLM_TASK_BYTES - 1))" \
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/${MICROLLM_TASK_REVISION}/${MICROLLM_TASK_FILE}" \
    --output "$MICROLLM_TASK_OUTPUT"
echo "dataset=roneneldan/TinyStories"
echo "revision=$MICROLLM_TASK_REVISION"
echo "license=cdla-sharing-1.0"
echo "split=$MICROLLM_TASK_SPLIT"
echo "bytes=$(wc -c < "$MICROLLM_TASK_OUTPUT")"
echo "output=$MICROLLM_TASK_OUTPUT"
