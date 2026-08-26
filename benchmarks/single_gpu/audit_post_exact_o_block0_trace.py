#!/usr/bin/env python3
"""Locate the first block-0 drift after exact prefill Attention core and O."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PARENT_SPEC = importlib.util.spec_from_file_location(
    "audit_post_exact_o_parent",
    Path(__file__).with_name("audit_post_exact_core_block0_trace.py"))
PARENT = importlib.util.module_from_spec(PARENT_SPEC)
assert PARENT_SPEC.loader is not None
PARENT_SPEC.loader.exec_module(PARENT)

TRACE = PARENT.BASE
COMMON = PARENT.COMMON
BATCHES = PARENT.BATCHES
O_SOLUTION = 296100


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context != 2048 or args.runs != 2 or args.timeout_seconds <= 0):
        parser.error("post-exact-O trace inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, batch: int,
            trace: Path, cache: Path) -> list[str]:
    return PARENT.command(args, model, batch, trace, cache) + [
        "--fp32-prefill-attention-o-solution-index", str(O_SOLUTION),
    ]


def require_route(record: dict, batch: int) -> None:
    expected = {
        "status": "pass", "batch": batch, "token_count": 2048,
        "trace_record_count": 50,
        "fp32_prefill_q_solution_index": TRACE.Q_SOLUTION,
        "fp32_prefill_kv_solution_index": TRACE.KV_SOLUTION,
        "fp32_prefill_attention_qk_solution_index": PARENT.QK_SOLUTION,
        "fp32_prefill_attention_pv_solution_index": PARENT.PV_SOLUTION,
        "fp32_prefill_attention_o_solution_index": O_SOLUTION,
        "fp32_solution_registered_entries": 5,
        "fp32_solution_cached_algorithms": 5,
        "fp32_solution_registry_hits": 168,
        "fp32_solution_cache_misses": 5,
        "fp32_solution_cache_hits": 163,
        "fp32_solution_dispatches": 168,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(
                f"B{batch} {name} expected {wanted!r}, got {record.get(name)!r}")


def render(summary: dict) -> str:
    return (TRACE.render(summary)
            .replace("First drift after exact block-0 cache",
                     "First drift after exact block-0 Attention core + O")
            .replace("Q296100 · KV292135 · complete first two batch rows",
                     "Q/K/V/QK/PV/O scoped exact · complete first two rows"))


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="microllm-post-exact-o-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            reference_trace = temporary / f"b1-r{run}.jsonl"
            reference_cache = temporary / f"b1-r{run}.bin"
            completed = subprocess.run(
                command(args, model, 1, reference_trace, reference_cache),
                text=True, capture_output=True, timeout=args.timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            reference = TRACE.load_trace(reference_trace, 1)
            reference_record = COMMON.last_json(completed.stdout)
            require_route(reference_record, 1)
            for batch in BATCHES:
                if batch == 1:
                    actual = reference
                    record = reference_record
                else:
                    trace = temporary / f"b{batch}-r{run}.jsonl"
                    cache = temporary / f"b{batch}-r{run}.bin"
                    current = subprocess.run(
                        command(args, model, batch, trace, cache),
                        text=True, capture_output=True,
                        timeout=args.timeout_seconds)
                    if current.returncode != 0:
                        raise RuntimeError(
                            current.stderr.strip() or current.stdout.strip())
                    actual = TRACE.load_trace(trace, batch)
                    record = COMMON.last_json(current.stdout)
                    require_route(record, batch)
                processes.append({
                    "schema_version": 1,
                    "record_type": "post_exact_o_block0_trace_process",
                    "status": "pass", "model": args.model,
                    "revision": model["revision"], "context": args.context,
                    "batch": batch, "process_run": run,
                    "trace_record_count": record["trace_record_count"],
                    "stages": TRACE.compare(reference, actual, batch),
                })
                print(json.dumps({"batch": batch, "process_run": run,
                                  "status": "pass"}, sort_keys=True), flush=True)
                if batch != 1:
                    del actual
            del reference
    summary = TRACE.summarize(processes)
    summary["record_type"] = "post_exact_o_block0_trace_audit"
    summary["qk_solution_index"] = PARENT.QK_SOLUTION
    summary["pv_solution_index"] = PARENT.PV_SOLUTION
    summary["o_solution_index"] = O_SOLUTION
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in processes),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "post-exact-o-trace.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_post_exact_o_block0_trace: {error}", file=sys.stderr)
        raise SystemExit(2) from error
