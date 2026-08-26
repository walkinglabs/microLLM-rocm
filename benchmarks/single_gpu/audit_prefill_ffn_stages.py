#!/usr/bin/env python3
"""Locate the first block-0 FFN drift after exact diagnostic Attention."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PARENT = load_module(
    "audit_prefill_ffn_parent", "audit_post_exact_o_block0_trace.py")
CORE = load_module(
    "audit_prefill_ffn_binary", "audit_prefill_attention_core.py")
COMMON = PARENT.COMMON
BATCHES = PARENT.BATCHES
PREFIX = "inference.cached_prefill.blocks.0"
STAGES = (
    PREFIX + ".ffn_norm",
    PREFIX + ".ffn.gate",
    PREFIX + ".ffn.up",
    PREFIX + ".ffn.activated",
    PREFIX + ".ffn.down",
    PREFIX + ".ffn.output",
    PREFIX + ".ffn_output",
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context != 2048 or args.runs != 2 or
            args.timeout_seconds <= 0 or sys.byteorder != "little"):
        parser.error("prefill FFN trace inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, batch: int,
            trace: Path, cache: Path, binary_directory: Path) -> list[str]:
    result = PARENT.command(args, model, batch, trace, cache)
    result[result.index("--trace-value-filter") + 1] = ",".join(STAGES)
    result[result.index("--trace-max-elements") + 1] = "1"
    result.extend([
        "--trace-binary-directory", str(binary_directory),
    ])
    return result


def require_route(record: dict, batch: int) -> None:
    expected = {
        "status": "pass", "batch": batch, "token_count": 2048,
        "trace_record_count": 55, "trace_binary_record_count": len(STAGES),
        "fp32_prefill_q_solution_index": PARENT.TRACE.Q_SOLUTION,
        "fp32_prefill_kv_solution_index": PARENT.TRACE.KV_SOLUTION,
        "fp32_prefill_attention_qk_solution_index": PARENT.PARENT.QK_SOLUTION,
        "fp32_prefill_attention_pv_solution_index": PARENT.PARENT.PV_SOLUTION,
        "fp32_prefill_attention_o_solution_index": PARENT.O_SOLUTION,
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


def load_trace(trace: Path, directory: Path, batch: int,
               context: int) -> dict[str, dict]:
    rows = [json.loads(line) for line in trace.read_text(
        encoding="utf-8").splitlines() if line]
    selected = {row["name"]: row for row in rows if row.get("name") in STAGES}
    if set(selected) != set(STAGES):
        raise ValueError("prefill FFN binary stages changed")
    captured = min(batch, 2)
    for name, row in selected.items():
        width = 8960 if name.endswith((".gate", ".up", ".activated")) else 1536
        shape = [captured, context, width]
        binary = row.get("binary_values")
        if (row.get("shape") != shape or row.get("values") is None or
                len(row["values"]) != 1 or not row.get("values_truncated") or
                not isinstance(binary, dict) or
                binary.get("dtype") != "float32" or
                binary.get("byte_order") != "little" or
                int(binary.get("count", -1)) != math.prod(shape) or
                int(binary.get("bytes", -1)) != math.prod(shape) * 4):
            raise ValueError(f"prefill FFN binary contract changed: {name}")
        path = directory / str(binary.get("file", ""))
        if not path.is_file() or path.stat().st_size != int(binary["bytes"]):
            raise ValueError(f"prefill FFN binary file changed: {name}")
        row["binary_path"] = path
    return selected


def exact_or_difference(left: Path, right: Path, elements: int,
                        left_offset: int = 0,
                        right_offset: int = 0) -> dict:
    byte_count = elements * 4
    with left.open("rb") as left_file, right.open("rb") as right_file:
        left_file.seek(left_offset)
        right_file.seek(right_offset)
        remaining = byte_count
        while remaining:
            size = min(4 * 1024 * 1024, remaining)
            if left_file.read(size) != right_file.read(size):
                return CORE.difference_binary(
                    left, right, elements, left_offset, right_offset)
            remaining -= size
    return {
        "elements": elements, "bitwise_equal": True,
        "first_bitwise_index": None, "first_numeric_index": None,
        "maximum": 0.0, "rms": 0.0, "relative_l2": 0.0,
    }


def compare(reference: dict[str, dict], actual: dict[str, dict],
            batch: int) -> list[dict]:
    output = []
    for name in STAGES:
        left = reference[name]
        right = actual[name]
        row_elements = math.prod(left["shape"])
        if math.prod(right["shape"]) != row_elements * min(batch, 2):
            raise ValueError(f"prefill FFN row shape changed: {name}")
        cross = exact_or_difference(
            left["binary_path"], right["binary_path"], row_elements)
        within = (exact_or_difference(
            right["binary_path"], right["binary_path"], row_elements,
            0, row_elements * 4) if batch > 1 else {
                "elements": row_elements, "bitwise_equal": True,
                "first_bitwise_index": None, "first_numeric_index": None,
                "maximum": 0.0, "rms": 0.0, "relative_l2": 0.0,
            })
        output.append({
            "name": name, "dtype": left["dtype"],
            "shape_b1": left["shape"], "shape_actual": right["shape"],
            "b1_vs_batch_row0": cross,
            "batch_row0_vs_row1": within,
        })
    return output


def summarize(processes: list[dict]) -> dict:
    cases = []
    for batch in BATCHES:
        rows = [row for row in processes if row["batch"] == batch]
        if len(rows) != 2 or rows[0]["stages"] != rows[1]["stages"]:
            raise ValueError(f"B{batch} FFN metrics are not deterministic")
        stages = rows[0]["stages"]
        cases.append({
            "batch": batch, "runs": 2,
            "first_nonzero_stage": next((
                stage["name"] for stage in stages
                if not stage["b1_vs_batch_row0"]["bitwise_equal"]), None),
            "maximum_error": max(
                stage["b1_vs_batch_row0"]["maximum"] for stage in stages),
            "all_within_batch_bitwise_equal": all(
                stage["batch_row0_vs_row1"]["bitwise_equal"]
                for stage in stages),
            "stages": stages,
        })
    first = next((case["first_nonzero_stage"] for case in cases
                  if case["first_nonzero_stage"] is not None), None)
    return {
        "schema_version": 1,
        "record_type": "prefill_ffn_stage_trace_audit",
        "status": "pass", "process_rows": len(processes),
        "case_rows": len(cases), "runs_per_case": 2,
        "context": 2048, "batches": list(BATCHES),
        "captured_batch_rows": 2, "stage_count": len(STAGES),
        "binary_files_retained": 0,
        "first_nonzero_stage": first,
        "all_repeat_metrics_equal": True,
        "all_within_batch_rows_bitwise_equal": all(
            case["all_within_batch_bitwise_equal"] for case in cases),
        "cases": cases,
    }


def render(summary: dict) -> str:
    svg = PARENT.TRACE.render(summary)
    return (svg.replace("First drift after exact block-0 cache",
                        "First drift inside block-0 prefill FFN")
            .replace("DeepSeek T2048 · Q296100 · KV292135 "
                     "· complete first two batch rows",
                     "exact diagnostic Attention · FFN complete first two rows"))


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="microllm-prefill-ffn-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            reference_root = temporary / f"b1-r{run}"
            reference_root.mkdir()
            reference_trace = reference_root / "trace.jsonl"
            reference_cache = reference_root / "cache.bin"
            reference_values = reference_root / "values"
            completed = subprocess.run(
                command(args, model, 1, reference_trace, reference_cache,
                        reference_values),
                text=True, capture_output=True, timeout=args.timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr.strip() or completed.stdout.strip())
            reference_record = COMMON.last_json(completed.stdout)
            require_route(reference_record, 1)
            reference = load_trace(
                reference_trace, reference_values, 1, args.context)
            for batch in BATCHES:
                if batch == 1:
                    actual = reference
                    record = reference_record
                    process_root = None
                else:
                    process_root = temporary / f"b{batch}-r{run}"
                    process_root.mkdir()
                    trace = process_root / "trace.jsonl"
                    cache = process_root / "cache.bin"
                    values = process_root / "values"
                    current = subprocess.run(
                        command(args, model, batch, trace, cache, values),
                        text=True, capture_output=True,
                        timeout=args.timeout_seconds)
                    if current.returncode != 0:
                        raise RuntimeError(
                            current.stderr.strip() or current.stdout.strip())
                    record = COMMON.last_json(current.stdout)
                    require_route(record, batch)
                    actual = load_trace(trace, values, batch, args.context)
                processes.append({
                    "schema_version": 1,
                    "record_type": "prefill_ffn_stage_trace_process",
                    "status": "pass", "model": args.model,
                    "revision": model["revision"], "context": args.context,
                    "batch": batch, "process_run": run,
                    "trace_record_count": record["trace_record_count"],
                    "trace_binary_bytes": record["trace_binary_bytes"],
                    "stages": compare(reference, actual, batch),
                })
                print(json.dumps({
                    "batch": batch, "process_run": run, "status": "pass",
                }, sort_keys=True), flush=True)
                if process_root is not None:
                    del actual
                    shutil.rmtree(process_root)
            del reference
            shutil.rmtree(reference_root)
    summary = summarize(processes)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in processes),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "ffn-stage-trace.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_prefill_ffn_stages: {error}", file=sys.stderr)
        raise SystemExit(2) from error
