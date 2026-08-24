#!/usr/bin/env python3
"""Gate direct BTHD BF16 Q/K across sequence and batch cases."""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


CASES = (
    ("b1t256", 1, 256, 256),
    ("b1t1024", 1, 1024, 1024),
    ("b2t512", 2, 512, 1024),
)
POLICIES = ("fp32-boundary", "bf16-qk")
INDICES = {
    ("qwen2.5-0.5b", 256): (64713, 65197, 24),
    ("deepseek-r1-distill-qwen-1.5b", 256): (64713, 65168, 28),
    ("qwen2.5-0.5b", 1024): (64755, 65200, 24),
    ("deepseek-r1-distill-qwen-1.5b", 1024): (64755, 65212, 28),
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-peak-ratio", type=float, default=1.0)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.minimum_speedup <= 1 or result.maximum_peak_ratio <= 0 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("BTHD BF16 Q/K shape options are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != expected:
        raise RuntimeError("BTHD BF16 Q/K shape gate requires pinned models")
    for model in result:
        config = Path(model["config"])
        if not config.is_file() or not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
        model["vocabulary_size"] = int(json.loads(
            config.read_text(encoding="utf-8"))["vocab_size"])
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(length)]


def command(args: argparse.Namespace, model: dict, case: tuple,
            policy: str, logits: Path) -> list[str]:
    _, batch, sequence, rows = case
    qkv_index, gate_up_index, _ = INDICES[(model["name"], rows)]
    tokens = repeated(model["inference"]["token_ids"], sequence)
    return [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", str(batch),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true",
        "--bf16-ffn-arena-minimum-rows", str(rows),
        "--bf16-qkv-arena", "true",
        "--bf16-qkv-arena-minimum-rows", str(rows),
        "--bf16-grouped-qkv-algorithm-index", str(qkv_index),
        "--bf16-grouped-gate-up-algorithm-index", str(gate_up_index),
        "--inference-bthd-attention", "true",
        "--inference-bthd-bf16-qk", "true" if policy == "bf16-qk" else "false",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", str(args.warmup),
        "--prefill-steps", str(args.steps),
        "--prefill-logits", "last", "--logits-output", str(logits),
    ]


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("hf_infer emitted no JSON")


def floats(path: Path) -> array.array:
    result = array.array("f")
    with path.open("rb") as stream:
        result.fromfile(stream, path.stat().st_size // result.itemsize)
    return result


def error(reference: array.array, actual: array.array) -> tuple[float, float, bool]:
    if len(reference) != len(actual) or not reference:
        raise RuntimeError("BTHD BF16 Q/K shape logits changed size")
    differences = [abs(expected - observed)
                   for expected, observed in zip(reference, actual, strict=True)]
    return (max(differences),
            math.sqrt(sum(value * value for value in differences) /
                      len(differences)),
            all(math.isfinite(value) for value in actual))


def row_top(values: array.array, batch: int, vocabulary: int) -> list[int]:
    if len(values) != batch * vocabulary:
        raise RuntimeError("BTHD BF16 Q/K logits do not match batch")
    return [max(range(vocabulary),
                key=lambda index: values[row * vocabulary + index])
            for row in range(batch)]


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    selected_models = models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    outputs: dict[tuple[str, str, str, int], array.array] = {}
    forwards = args.warmup + args.steps
    with tempfile.TemporaryDirectory(prefix="microllm-bthd-bf16-qk-shapes-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            for process_run in range(1, args.runs + 1):
                cases = list(CASES)
                policies = list(POLICIES)
                if process_run % 2 == 0:
                    cases.reverse()
                    policies.reverse()
                for case in cases:
                    case_name, batch, sequence, rows = case
                    _, _, blocks = INDICES[(model["name"], rows)]
                    expected_dispatches = blocks * forwards
                    for policy in policies:
                        logits = temporary / (
                            f"{model['name']}-{process_run}-{case_name}-{policy}.bin")
                        completed = subprocess.run(
                            command(args, model, case, policy, logits), text=True,
                            capture_output=True, check=False)
                        if completed.returncode != 0:
                            raise RuntimeError(completed.stdout + completed.stderr)
                        record = last_json(completed.stdout)
                        expected_retained = (expected_dispatches
                                             if policy == "bf16-qk" else 0)
                        if record.get("status") != "pass" or \
                                record.get("inference_bthd_attention") is not True or \
                                record.get("inference_bthd_bf16_qk") is not (policy == "bf16-qk") or \
                                int(record["bf16_grouped_qkv_dispatches"]) != expected_dispatches or \
                                int(record["bf16_grouped_qkv_retained_query_key_dispatches"]) != expected_retained:
                            raise RuntimeError("invalid BTHD BF16 Q/K shape route")
                        record.update({
                            "record_type": "inference_bthd_bf16_qk_shape_measurement",
                            "model": model["name"], "revision": model["revision"],
                            "case": case_name, "batch": batch,
                            "sequence": sequence, "rows": rows,
                            "policy": policy, "process_run": process_run,
                            "case_order": [item[0] for item in cases],
                            "policy_order": policies,
                        })
                        records.append(record)
                        outputs[(model["name"], case_name,
                                 policy, process_run)] = floats(logits)

    comparisons = []
    for model in selected_models:
        for case_name, batch, sequence, rows in CASES:
            reference = outputs[(model["name"], case_name, "fp32-boundary", 1)]
            reference_top = row_top(reference, batch, model["vocabulary_size"])
            selected = [row for row in records
                        if row["model"] == model["name"] and row["case"] == case_name]
            grouped = {policy: [row for row in selected if row["policy"] == policy]
                       for policy in POLICIES}
            maximum = 0.0
            rms = 0.0
            finite = True
            top_equal = True
            for policy in POLICIES:
                for process_run in range(1, args.runs + 1):
                    actual = outputs[(model["name"], case_name, policy, process_run)]
                    current = error(reference, actual)
                    maximum = max(maximum, current[0])
                    rms = max(rms, current[1])
                    finite = finite and current[2]
                    top_equal = top_equal and reference_top == row_top(
                        actual, batch, model["vocabulary_size"])
            baseline_tps = median(grouped["fp32-boundary"],
                                  "prefill_tokens_per_second")
            candidate_tps = median(grouped["bf16-qk"],
                                   "prefill_tokens_per_second")
            baseline_peak = int(median(grouped["fp32-boundary"],
                                       "engine_peak_bytes"))
            candidate_peak = int(median(grouped["bf16-qk"],
                                        "engine_peak_bytes"))
            comparisons.append({
                "model": model["name"], "revision": model["revision"],
                "case": case_name, "batch": batch,
                "sequence": sequence, "rows": rows,
                "fp32_boundary_tokens_per_second": baseline_tps,
                "bf16_qk_tokens_per_second": candidate_tps,
                "speedup": candidate_tps / baseline_tps,
                "fp32_boundary_peak_bytes": baseline_peak,
                "bf16_qk_peak_bytes": candidate_peak,
                "peak_ratio": candidate_peak / baseline_peak,
                "peak_bytes_saved": baseline_peak - candidate_peak,
                "maximum_absolute_logit_difference": maximum,
                "maximum_rms_logit_difference": rms,
                "finite_complete_logits": finite,
                "top_rows_equal": top_equal,
            })

    correctness = all(row["finite_complete_logits"] and row["top_rows_equal"] and
                      row["maximum_absolute_logit_difference"] == 0 and
                      row["maximum_rms_logit_difference"] == 0
                      for row in comparisons)
    routing = all(int(row["bf16_grouped_qkv_retained_query_key_dispatches"]) > 0
                  for row in records if row["policy"] == "bf16-qk")
    performance = all(row["speedup"] >= args.minimum_speedup
                      for row in comparisons)
    memory = all(row["peak_ratio"] <= args.maximum_peak_ratio
                 for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and routing and memory else "fail",
        "record_type": "inference_bthd_bf16_qk_shape_summary",
        "processes": len(records), "correctness_gate": correctness,
        "routing_gate": routing, "performance_gate": performance,
        "memory_gate": memory, "comparisons": comparisons,
        "decision": ("keep explicit BF16 Q/K across measured shapes"
                     if correctness and routing and performance and memory
                     else "retain only individually passing BF16 Q/K shapes"),
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
