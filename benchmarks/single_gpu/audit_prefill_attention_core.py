#!/usr/bin/env python3
"""Compare complete block-0 prefill QK, softmax, and P*V values across batch."""

from __future__ import annotations

import argparse
import array
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BASE_SPEC = importlib.util.spec_from_file_location(
    "audit_prefill_attention_core_base",
    Path(__file__).with_name("audit_post_cache_block0_trace.py"))
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE)

COMMON = BASE.COMMON
BATCHES = BASE.BATCHES
PREFIX = BASE.PREFIX + ".attention"
STAGES = (
    PREFIX + ".scores",
    PREFIX + ".probabilities",
    PREFIX + ".pv_output",
)


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
            args.context != 2048 or args.runs != 2 or
            args.timeout_seconds <= 0):
        parser.error("prefill Attention core inputs are outside the formal contract")
    if sys.byteorder != "little":
        parser.error("prefill Attention binary comparison requires little endian")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, batch: int,
            trace: Path, cache: Path, binary_directory: Path) -> list[str]:
    result = BASE.command(args, model, batch, trace, cache)
    result[result.index("--trace-value-filter") + 1] = ",".join(STAGES)
    result[result.index("--trace-max-elements") + 1] = "1"
    result.extend([
        "--trace-binary-directory", str(binary_directory),
    ])
    return result


def load_binary_records(trace: Path, directory: Path, batch: int,
                        context: int) -> dict[str, dict]:
    rows = [json.loads(line) for line in trace.read_text(
        encoding="utf-8").splitlines() if line]
    selected = {row["name"]: row for row in rows if row.get("name") in STAGES}
    if set(selected) != set(STAGES):
        raise ValueError("prefill Attention core stages changed")
    captured_rows = min(batch, 2)
    for name, row in selected.items():
        shape = [int(value) for value in row.get("shape", [])]
        expected_shape = ([captured_rows, 12, context, context]
                          if name != STAGES[-1]
                          else [captured_rows, 12, context, 128])
        binary = row.get("binary_values")
        if (shape != expected_shape or row.get("dtype") != "float32" or
                row.get("values") is None or len(row["values"]) != 1 or
                not row.get("values_truncated") or not isinstance(binary, dict) or
                binary.get("dtype") != "float32" or
                binary.get("byte_order") != "little" or
                int(binary.get("count", -1)) != math.prod(shape) or
                int(binary.get("bytes", -1)) != math.prod(shape) * 4):
            raise ValueError(f"prefill Attention binary contract changed: {name}")
        path = directory / str(binary.get("file", ""))
        if not path.is_file() or path.stat().st_size != int(binary["bytes"]):
            raise ValueError(f"prefill Attention binary file changed: {name}")
        row["binary_path"] = path
    return selected


def difference_binary(left: Path, right: Path, elements: int,
                      left_offset: int = 0, right_offset: int = 0,
                      causal_sequence: int = 0) -> dict:
    if elements <= 0 or left_offset < 0 or right_offset < 0:
        raise ValueError("binary comparison range must be positive")
    if causal_sequence < 0 or (causal_sequence > 0 and
                               elements % (causal_sequence ** 2) != 0):
        raise ValueError("causal binary comparison shape is invalid")
    byte_count = elements * 4
    if (left_offset + byte_count > left.stat().st_size or
            right_offset + byte_count > right.stat().st_size):
        raise ValueError("binary comparison range exceeds a file")
    maximum = 0.0
    squared = 0.0
    reference_squared = 0.0
    first_bitwise = None
    first_numeric = None
    compared = 0
    selected = 0
    with left.open("rb") as left_file, right.open("rb") as right_file:
        left_file.seek(left_offset)
        right_file.seek(right_offset)
        while compared < byte_count:
            size = min(4 * 1024 * 1024, byte_count - compared)
            left_bytes = left_file.read(size)
            right_bytes = right_file.read(size)
            if len(left_bytes) != size or len(right_bytes) != size:
                raise ValueError("binary comparison ended early")
            left_bits = array.array("I")
            right_bits = array.array("I")
            left_bits.frombytes(left_bytes)
            right_bits.frombytes(right_bytes)
            left_values = array.array("f")
            right_values = array.array("f")
            left_values.frombytes(left_bytes)
            right_values.frombytes(right_bytes)
            element_base = compared // 4
            for local, (left_bit, right_bit, left_value, right_value) in enumerate(
                    zip(left_bits, right_bits, left_values, right_values)):
                flat_index = element_base + local
                if causal_sequence > 0:
                    query = (flat_index // causal_sequence) % causal_sequence
                    source = flat_index % causal_sequence
                    if source > query:
                        continue
                selected += 1
                if left_bit != right_bit and first_bitwise is None:
                    first_bitwise = flat_index
                delta = abs(left_value - right_value)
                if delta != 0.0 and first_numeric is None:
                    first_numeric = flat_index
                if not (math.isfinite(left_value) and math.isfinite(right_value)):
                    raise ValueError("prefill Attention binary contains non-finite values")
                maximum = max(maximum, delta)
                squared += delta * delta
                reference_squared += left_value * left_value
            compared += size
    if selected == 0:
        raise ValueError("binary comparison selected no elements")
    return {
        "elements": selected,
        "bitwise_equal": first_bitwise is None,
        "first_bitwise_index": first_bitwise,
        "first_numeric_index": first_numeric,
        "maximum": maximum,
        "rms": math.sqrt(squared / selected),
        "relative_l2": (math.sqrt(squared / reference_squared)
                        if reference_squared > 0.0 else 0.0),
    }


def compare(reference: dict[str, dict], actual: dict[str, dict],
            batch: int) -> list[dict]:
    output = []
    for name in STAGES:
        left = reference[name]
        right = actual[name]
        row_elements = math.prod(left["shape"])
        if math.prod(right["shape"]) != row_elements * min(batch, 2):
            raise ValueError(f"prefill Attention row shape changed: {name}")
        cross = difference_binary(
            left["binary_path"], right["binary_path"], row_elements)
        within = (difference_binary(
            right["binary_path"], right["binary_path"], row_elements,
            0, row_elements * 4) if batch > 1 else {
                "elements": row_elements,
                "bitwise_equal": True,
                "first_bitwise_index": None,
                "first_numeric_index": None,
                "maximum": 0.0,
                "rms": 0.0,
                "relative_l2": 0.0,
            })
        stage = {
            "name": name,
            "shape_b1": left["shape"],
            "shape_actual": right["shape"],
            "b1_vs_batch_row0": cross,
            "batch_row0_vs_row1": within,
        }
        if name in STAGES[:2]:
            stage["b1_vs_batch_row0_causal_visible"] = difference_binary(
                left["binary_path"], right["binary_path"], row_elements,
                causal_sequence=left["shape"][-1])
            stage["batch_row0_vs_row1_causal_visible"] = (
                difference_binary(
                    right["binary_path"], right["binary_path"], row_elements,
                    0, row_elements * 4, left["shape"][-1])
                if batch > 1 else {
                    "elements": 12 * left["shape"][-1] *
                                (left["shape"][-1] + 1) // 2,
                    "bitwise_equal": True,
                    "first_bitwise_index": None,
                    "first_numeric_index": None,
                    "maximum": 0.0,
                    "rms": 0.0,
                    "relative_l2": 0.0,
                })
        output.append(stage)
    return output


def summarize(processes: list[dict]) -> dict:
    cases = []
    for batch in BATCHES:
        rows = [row for row in processes if row["batch"] == batch]
        if len(rows) != 2 or rows[0]["stages"] != rows[1]["stages"]:
            raise ValueError(f"B{batch} Attention metrics are not deterministic")
        stages = rows[0]["stages"]
        cases.append({
            "batch": batch,
            "runs": 2,
            "first_nonzero_stage": next((
                stage["name"] for stage in stages
                if not stage["b1_vs_batch_row0"]["bitwise_equal"]), None),
            "first_causal_nonzero_stage": next((
                stage["name"] for index, stage in enumerate(stages)
                if not (stage["b1_vs_batch_row0_causal_visible"]
                        if index < 2 else stage["b1_vs_batch_row0"])
                       ["bitwise_equal"]), None),
            "all_within_batch_bitwise_equal": all(
                stage["batch_row0_vs_row1"]["bitwise_equal"]
                for stage in stages),
            "stages": stages,
        })
    return {
        "schema_version": 1,
        "record_type": "prefill_attention_core_audit",
        "status": "pass",
        "process_rows": len(processes),
        "case_rows": len(cases),
        "runs_per_case": 2,
        "context": 2048,
        "batches": list(BATCHES),
        "q_solution_index": BASE.Q_SOLUTION,
        "kv_solution_index": BASE.KV_SOLUTION,
        "stage_count": len(STAGES),
        "binary_files_retained": 0,
        "all_repeat_metrics_equal": True,
        "all_scores_bitwise_equal": all(
            case["stages"][0]["b1_vs_batch_row0"]["bitwise_equal"]
            for case in cases),
        "all_probabilities_bitwise_equal": all(
            case["stages"][1]["b1_vs_batch_row0"]["bitwise_equal"]
            for case in cases),
        "all_causal_scores_bitwise_equal": all(
            case["stages"][0]["b1_vs_batch_row0_causal_visible"]
                ["bitwise_equal"] for case in cases),
        "first_nonzero_stage": next((
            stage for stage_index, stage in enumerate(STAGES)
            if any(not case["stages"][stage_index]
                   ["b1_vs_batch_row0"]["bitwise_equal"]
                   for case in cases)), None),
        "first_nonzero_stage_by_batch": {
            str(case["batch"]): case["first_nonzero_stage"]
            for case in cases
        },
        "first_causal_nonzero_stage": next((
            stage for stage_index, stage in enumerate(STAGES)
            if any(not (case["stages"][stage_index]
                        ["b1_vs_batch_row0_causal_visible"]
                        if stage_index < 2 else case["stages"][stage_index]
                        ["b1_vs_batch_row0"])["bitwise_equal"]
                   for case in cases)), None),
        "first_causal_nonzero_stage_by_batch": {
            str(case["batch"]): case["first_causal_nonzero_stage"]
            for case in cases
        },
        "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1320, 650
    colors = {2: "#38bdf8", 4: "#f97316", 8: "#22c55e"}
    stages = [name.removeprefix(PREFIX + ".") for name in STAGES]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#94a3b8}'
        '.label{font-size:14px}.value{font-size:12px}</style>',
        '<text x="36" y="42" class="title">Complete T2048 prefill Attention core</text>',
        '<text x="36" y="68" class="sub">DeepSeek block 0 · Q296100 · KV292135 · '
        'two fresh processes · full first two rows</text>',
    ]
    for stage_index, stage in enumerate(stages):
        x = 260 + stage_index * 340
        parts.append(f'<text x="{x}" y="118" class="label" text-anchor="middle">{stage}</text>')
        parts.append(f'<line x1="{x}" y1="140" x2="{x}" y2="510" stroke="#334155"/>')
    for row_index, case in enumerate(
            row for row in summary["cases"] if row["batch"] > 1):
        y = 190 + row_index * 130
        parts.append(f'<text x="56" y="{y + 6}" class="label">B1 vs B{case["batch"]}</text>')
        for stage_index, stage in enumerate(case["stages"]):
            x = 260 + stage_index * 340
            metric = (stage["b1_vs_batch_row0_causal_visible"]
                      if stage_index < 2 else stage["b1_vs_batch_row0"])
            exact = metric["bitwise_equal"]
            fill = "#64748b" if exact else colors[case["batch"]]
            radius = 11 if exact else 18
            parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"/>')
            value = "exact" if exact else f'Max {metric["maximum"]:.3e}'
            parts.append(f'<text x="{x}" y="{y + 38}" class="value" '
                         f'text-anchor="middle">{value}</text>')
    parts.extend([
        '<line x1="90" y1="555" x2="1230" y2="555" stroke="#334155" stroke-width="3"/>',
        '<circle cx="230" cy="555" r="9" fill="#64748b"/>',
        '<text x="250" y="560" class="sub">complete bitwise equality</text>',
        '<circle cx="560" cy="555" r="12" fill="#f97316"/>',
        '<text x="580" y="560" class="sub">first numerical drift</text>',
        '<text x="36" y="615" class="label">Decision follows causal-visible values; '
        'masked future scores never count as a root cause.</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def run_process(args: argparse.Namespace, model: dict, batch: int,
                run: int, root: Path) -> tuple[dict[str, dict], dict]:
    process = root / f"b{batch}-r{run}"
    process.mkdir(parents=True)
    trace = process / "trace.jsonl"
    cache = process / "cache.bin"
    binary_directory = process / "values"
    completed = subprocess.run(
        command(args, model, batch, trace, cache, binary_directory),
        text=True, capture_output=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    record = COMMON.last_json(completed.stdout)
    selected = load_binary_records(
        trace, binary_directory, batch, args.context)
    expected_bytes = sum(
        row["binary_values"]["bytes"] for row in selected.values())
    if (record.get("status") != "pass" or
            record.get("trace_record_count") != 54 or
            record.get("trace_binary_record_count") != 3 or
            record.get("trace_binary_bytes") != expected_bytes or
            record.get("fp32_prefill_q_solution_index") != BASE.Q_SOLUTION or
            record.get("fp32_prefill_kv_solution_index") != BASE.KV_SOLUTION or
            record.get("fp32_solution_registered_entries") != 2 or
            record.get("fp32_solution_registry_hits") != 84):
        raise ValueError(f"B{batch} prefill Attention route changed")
    return selected, record


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="microllm-prefill-attention-core-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            reference, reference_record = run_process(args, model, 1, run, temporary)
            reference_root = temporary / f"b1-r{run}"
            for batch in BATCHES:
                if batch == 1:
                    actual = reference
                    record = reference_record
                else:
                    actual, record = run_process(
                        args, model, batch, run, temporary)
                stages = compare(reference, actual, batch)
                processes.append({
                    "schema_version": 1,
                    "record_type": "prefill_attention_core_process",
                    "status": "pass",
                    "model": args.model,
                    "revision": model["revision"],
                    "context": args.context,
                    "batch": batch,
                    "process_run": run,
                    "trace_record_count": record["trace_record_count"],
                    "trace_binary_bytes": record["trace_binary_bytes"],
                    "stages": stages,
                })
                print(json.dumps({
                    "batch": batch,
                    "process_run": run,
                    "first_nonzero_stage": next((
                        stage["name"] for stage in stages
                        if not stage["b1_vs_batch_row0"]["bitwise_equal"]), None),
                    "status": "pass",
                }, sort_keys=True), flush=True)
                if batch != 1:
                    shutil.rmtree(temporary / f"b{batch}-r{run}")
            shutil.rmtree(reference_root)
    summary = summarize(processes)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in processes),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "attention-core.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_prefill_attention_core: {error}", file=sys.stderr)
        raise SystemExit(2) from error
