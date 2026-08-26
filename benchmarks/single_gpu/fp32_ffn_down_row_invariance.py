#!/usr/bin/env python3
"""Screen real DeepSeek FP32 FFN down solutions across prefill batch M."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "fp32_ffn_down_row_base",
    Path(__file__).with_name("fp32_ffn_row_invariance_matrix.py"))
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

BASE.INNER = 8960
BASE.COLUMNS = 1536


def output_directory() -> Path:
    try:
        index = sys.argv.index("--output-directory")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("FFN down runner requires output directory") from error


def main() -> int:
    destination = output_directory()
    result = BASE.main()
    summary_path = destination / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["record_type"] = "fp32_ffn_down_row_invariance_matrix"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    source = destination / "ffn-row-invariance.svg"
    svg = source.read_text(encoding="utf-8")
    svg = (svg.replace("FP32 FFN gate/up row invariance",
                       "FP32 FFN down row invariance")
           .replace("K1536 · N8960", "K8960 · N1536"))
    (destination / "ffn-down-row-invariance.svg").write_text(
        svg, encoding="utf-8")
    source.unlink()
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_ffn_down_row_invariance: {error}", file=sys.stderr)
        raise SystemExit(2) from error
