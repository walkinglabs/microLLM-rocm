#!/usr/bin/env python3
"""Trace block-0 FFN after exact diagnostic Attention and exact gate/up."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "audit_post_exact_gate_up_base",
    Path(__file__).with_name("audit_prefill_ffn_stages.py"))
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

ORIGINAL_COMMAND = BASE.command
SOLUTION = 296100


def command(args, model: dict, batch: int, trace: Path,
            cache: Path, binary_directory: Path) -> list[str]:
    return ORIGINAL_COMMAND(
        args, model, batch, trace, cache, binary_directory) + [
            "--fp32-prefill-ffn-gate-up-solution-index", str(SOLUTION),
        ]


def require_route(record: dict, batch: int) -> None:
    expected = {
        "status": "pass", "batch": batch, "token_count": 2048,
        "trace_record_count": 55, "trace_binary_record_count": 7,
        "fp32_prefill_q_solution_index": BASE.PARENT.TRACE.Q_SOLUTION,
        "fp32_prefill_kv_solution_index": BASE.PARENT.TRACE.KV_SOLUTION,
        "fp32_prefill_attention_qk_solution_index":
            BASE.PARENT.PARENT.QK_SOLUTION,
        "fp32_prefill_attention_pv_solution_index":
            BASE.PARENT.PARENT.PV_SOLUTION,
        "fp32_prefill_attention_o_solution_index": BASE.PARENT.O_SOLUTION,
        "fp32_prefill_ffn_gate_up_solution_index": SOLUTION,
        "fp32_solution_registered_entries": 6,
        "fp32_solution_cached_algorithms": 6,
        "fp32_solution_registry_hits": 224,
        "fp32_solution_cache_misses": 6,
        "fp32_solution_cache_hits": 218,
        "fp32_solution_dispatches": 224,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(
                f"B{batch} {name} expected {wanted!r}, got {record.get(name)!r}")


def output_directory() -> Path:
    try:
        index = sys.argv.index("--output-directory")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("post-exact gate/up runner requires output directory") from error


def main() -> int:
    destination = output_directory()
    BASE.command = command
    BASE.require_route = require_route
    result = BASE.main()
    summary_path = destination / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["record_type"] = "post_exact_gate_up_ffn_stage_trace_audit"
    summary["ffn_gate_up_solution_index"] = SOLUTION
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (destination / "ffn-stage-trace.svg").rename(
        destination / "post-exact-gate-up-trace.svg")
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_post_exact_gate_up_ffn: {error}", file=sys.stderr)
        raise SystemExit(2) from error
