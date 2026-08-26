#!/usr/bin/env python3
"""Audit block-0 BF16 K/V cache prefixes across full-prefill batch sizes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


COMMON_SPEC = importlib.util.spec_from_file_location(
    "audit_prefill_cache_prefix_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

BATCHES = (1, 2, 4, 8)


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
            args.context <= 0 or args.runs != 2 or args.timeout_seconds <= 0):
        parser.error("prefill cache-prefix inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, batch: int,
            output: Path) -> list[str]:
    return [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", COMMON.expanded(model["inference"]["token_ids"], args.context),
        "--device", "hip", "--top-k", "1", "--batch", str(batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--cache-capacity", str(args.context + 1), "--new-tokens", "1",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--bf16-ffn", "false", "--bf16-attention", "false",
        "--workload", "decode", "--prefill-cache-output", str(output),
        "--prefill-cache-layer", "0",
    ]


def bf16_values(payload: bytes) -> list[float]:
    if not payload or len(payload) % 2:
        raise ValueError("BF16 cache payload is incomplete")
    result = []
    for (bits,) in struct.iter_unpack("<H", payload):
        result.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    return result


def load(path: Path) -> tuple[dict, dict[str, list[bytes]], dict[str, list[list[float]]]]:
    with path.open("rb") as stream:
        header = json.loads(stream.readline())
        payload = stream.read()
    shape = [int(value) for value in header.get("shape", [])]
    if (header.get("schema_version") != 1 or
            header.get("record_type") != "prefill_kv_cache" or
            header.get("layer") != 0 or header.get("dtype") != "bfloat16" or
            len(shape) != 4 or shape[0] <= 0 or shape[2] <= 0 or
            header.get("key_bytes") != header.get("value_bytes") or
            len(payload) != header.get("key_bytes") + header.get("value_bytes")):
        raise ValueError("prefill cache file contract changed")
    batch = shape[0]
    tensor_bytes = int(header["key_bytes"])
    row_bytes = tensor_bytes // batch
    if row_bytes * batch != tensor_bytes:
        raise ValueError("prefill cache row bytes changed")
    raw = {}
    values = {}
    for index, name in enumerate(("key", "value")):
        tensor = payload[index * tensor_bytes:(index + 1) * tensor_bytes]
        raw[name] = [tensor[row * row_bytes:(row + 1) * row_bytes]
                     for row in range(batch)]
        values[name] = [bf16_values(row) for row in raw[name]]
    return header, raw, values


def difference(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("cache-prefix comparison needs equal non-empty rows")
    maximum = 0.0
    square = 0.0
    reference_square = 0.0
    for left_value, right_value in zip(left, right):
        delta = abs(left_value - right_value)
        maximum = max(maximum, delta)
        square += delta * delta
        reference_square += left_value * left_value
    return {
        "elements": len(left), "maximum": maximum,
        "rms": math.sqrt(square / len(left)),
        "relative_l2": math.sqrt(square / reference_square)
        if reference_square > 0.0 else 0.0,
        "bitwise_equal": left == right,
    }


def summarize(processes: list[dict]) -> dict:
    by_key = {(row["batch"], row["process_run"]): row for row in processes}
    cases = []
    for tensor in ("key", "value"):
        reference = by_key[(1, 1)]["values"][tensor][0]
        for batch in BATCHES:
            rows = [by_key[(batch, run)] for run in (1, 2)]
            first = [row["values"][tensor][0] for row in rows]
            cross = [difference(reference, values) for values in first]
            repeat = difference(first[0], first[1])
            raw_rows = [row["raw"][tensor] for row in rows]
            within = [raw == [raw[0]] * len(raw) for raw in raw_rows]
            cases.append({
                "tensor": tensor, "batch": batch, "runs": 2,
                "complete_values_compared_per_run": len(reference),
                "cross_batch_maximum_error": max(item["maximum"] for item in cross),
                "cross_batch_maximum_rms_error": max(item["rms"] for item in cross),
                "cross_batch_maximum_relative_l2": max(
                    item["relative_l2"] for item in cross),
                "cross_batch_bitwise_equal": all(
                    item["bitwise_equal"] for item in cross),
                "repeat_bitwise_equal": repeat["bitwise_equal"],
                "within_batch_bitwise_equal": all(within),
            })
    tensor_summaries = []
    for tensor in ("key", "value"):
        selected = [row for row in cases if row["tensor"] == tensor]
        tensor_summaries.append({
            "tensor": tensor,
            "maximum_cross_batch_error": max(
                row["cross_batch_maximum_error"] for row in selected),
            "maximum_cross_batch_rms_error": max(
                row["cross_batch_maximum_rms_error"] for row in selected),
            "maximum_cross_batch_relative_l2": max(
                row["cross_batch_maximum_relative_l2"] for row in selected),
            "bitwise_case_count": sum(
                row["cross_batch_bitwise_equal"] for row in selected),
        })
    return {
        "schema_version": 1,
        "record_type": "prefill_cache_prefix_audit",
        "status": "pass", "process_rows": len(processes),
        "case_rows": len(cases), "batches": list(BATCHES),
        "runs_per_case": 2, "layer": 0, "dtype": "bfloat16",
        "shape_suffix": [2, 2048, 128],
        "all_repeat_bitwise_equal": all(
            row["repeat_bitwise_equal"] for row in cases),
        "all_within_batch_bitwise_equal": all(
            row["within_batch_bitwise_equal"] for row in cases),
        "tensor_summaries": tensor_summaries, "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1380, 470
    maximum = max(row["maximum_cross_batch_error"]
                  for row in summary["tensor_summaries"])
    scale = 780.0 / maximum if maximum else 1.0
    colors = {"key": "#38bdf8", "value": "#f97316"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:13px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">Block-0 BF16 prefill cache prefix</text>',
        '<text x="30" y="62" class="muted">DeepSeek T2048 · B1 reference · '
        'complete K/V row0 values</text>',
    ]
    for tensor_index, tensor in enumerate(("key", "value")):
        y0 = 100 + tensor_index * 170
        parts.append(f'<text x="30" y="{y0 + 18}" class="label">{tensor}</text>')
        rows = [row for row in summary["cases"] if row["tensor"] == tensor]
        for index, row in enumerate(rows):
            y = y0 + index * 32
            length = max(2.0, row["cross_batch_maximum_error"] * scale)
            parts.extend((
                f'<text x="190" y="{y + 18}" class="label">B{row["batch"]}</text>',
                f'<rect x="230" y="{y}" width="{length:.2f}" height="22" rx="4" '
                f'fill="{colors[tensor]}"/>',
                f'<text x="{245 + length:.2f}" y="{y + 17}" class="label">'
                f'Max {row["cross_batch_maximum_error"]:.3e} · '
                f'RMS {row["cross_batch_maximum_rms_error"]:.3e}</text>',
            ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="microllm-prefill-cache-prefix-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            batch_order = list(BATCHES) if run % 2 else list(reversed(BATCHES))
            for batch in batch_order:
                output = temporary / f"b{batch}-r{run}.bin"
                completed = subprocess.run(
                    command(args, model, batch, output), text=True,
                    capture_output=True, timeout=args.timeout_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(
                        completed.stderr.strip() or completed.stdout.strip())
                record = COMMON.last_json(completed.stdout)
                header, raw, values = load(output)
                required = {
                    "status": "pass", "batch": batch,
                    "token_count": args.context, "decode_tokens": 1,
                    "prefill_cache_exported": True, "prefill_cache_layer": 0,
                    "prefill_cache_dtype": "bfloat16",
                    "prefill_cache_shape": header["shape"],
                    "prefill_cache_key_bytes": header["key_bytes"],
                    "prefill_cache_value_bytes": header["value_bytes"],
                    "cached_attention_materialized_policy": "auto-enabled",
                }
                for name, wanted in required.items():
                    if record.get(name) != wanted:
                        raise ValueError(
                            f"B{batch} {name} expected {wanted!r}, got {record.get(name)!r}")
                processes.append({
                    "schema_version": 1,
                    "record_type": "prefill_cache_prefix_process",
                    "status": "pass", "model": args.model,
                    "revision": model["revision"], "context": args.context,
                    "batch": batch, "process_run": run,
                    "header": header, "raw": raw, "values": values,
                })
                print(json.dumps({"batch": batch, "process_run": run,
                                  "status": "pass"}, sort_keys=True), flush=True)
    summary = summarize(processes)
    raw_records = []
    for row in processes:
        raw_records.append({key: value for key, value in row.items()
                            if key not in {"raw", "values"}})
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "cache-prefix.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_prefill_cache_prefix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
