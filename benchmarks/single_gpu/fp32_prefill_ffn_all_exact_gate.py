#!/usr/bin/env python3
"""Run the final all-batch exact FP32 prefill FFN gate/up rebuttal."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "fp32_prefill_ffn_all_exact_base",
    Path(__file__).with_name("fp32_prefill_ffn_model_gate.py"))
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

SELECTIVE = {
    batch: {"qk": -1, "pv": -1, "ffn": BASE.SOLUTION}
    for batch in BASE.BATCHES
}
BASE.SELECTIVE = SELECTIVE


def output_directory() -> Path:
    try:
        index = sys.argv.index("--output-directory")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("all-exact runner requires --output-directory") from error


def main() -> int:
    destination = output_directory()
    result = BASE.main()
    summary_path = destination / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["record_type"] = "prefill_ffn_gate_up_all_exact_model_gate"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (destination / "ffn-model-gate.svg").rename(
        destination / "ffn-all-exact-model-gate.svg")
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_prefill_ffn_all_exact_gate: {error}", file=sys.stderr)
        raise SystemExit(2) from error
