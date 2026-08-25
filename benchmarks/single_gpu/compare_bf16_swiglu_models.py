#!/usr/bin/env python3
"""Full-model gate for the BF16 vectorized SwiGLU default route."""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


GROUPED_T1024 = {
    "qwen2.5-0.5b": (64755, 65200),
    "deepseek-r1-distill-qwen-1.5b": (64755, 65212),
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-binary", required=True, type=Path)
    parser.add_argument("--candidate-binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--candidate-swish", action="store_true")
    parser.add_argument("--candidate-bf16-norm", action="store_true")
    parser.add_argument("--candidate-attention-norm", action="store_true")
    result = parser.parse_args()
    if (not result.manifest.is_file() or not result.baseline_binary.is_file() or
            not result.candidate_binary.is_file() or result.runs <= 0 or
            result.warmup < 0 or result.steps <= 0):
        parser.error("model-gate inputs are outside the measured contract")
    if sum((result.candidate_swish, result.candidate_bf16_norm,
            result.candidate_attention_norm)) > 1:
        parser.error("select only one candidate route")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    if document.get("schema_version") != 1 or \
            {row.get("name") for row in result} != set(GROUPED_T1024):
        raise RuntimeError("SwiGLU gate requires pinned Qwen and DeepSeek")
    for model in result:
        if not Path(model["config"]).is_file() or not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
    return result


def repeated(seed: list[int], length: int) -> str:
    return ",".join(str(seed[index % len(seed)]) for index in range(length))


def command(args: argparse.Namespace, model: dict, policy: str,
            logits: Path) -> list[str]:
    qkv_index, gate_up_index = GROUPED_T1024[model["name"]]
    binary = args.baseline_binary if policy == "baseline" else args.candidate_binary
    result = [
        str(binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", repeated(model["inference"]["token_ids"], 1024),
        "--device", "hip", "--top-k", "10", "--batch", "1",
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true", "--bf16-ffn-arena-minimum-rows", "1024",
        "--bf16-qkv-arena", "true", "--bf16-qkv-arena-minimum-rows", "1024",
        "--bf16-grouped-qkv-algorithm-index", str(qkv_index),
        "--bf16-grouped-gate-up-algorithm-index", str(gate_up_index),
        "--inference-bthd-attention", "true",
        "--inference-bthd-bf16-qk", "true",
        "--inference-bthd-online-attention", "false",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", str(args.warmup),
        "--prefill-steps", str(args.steps),
        "--prefill-logits", "last", "--logits-output", str(logits),
    ]
    if args.candidate_swish:
        result.extend([
            "--bf16-grouped-gate-up-swish",
            "true" if policy == "vectorized" else "false",
        ])
    if args.candidate_bf16_norm:
        result.extend([
            "--bf16-ffn-norm-fusion",
            "true" if policy == "vectorized" else "false",
        ])
    if args.candidate_attention_norm:
        result.extend([
            "--bf16-attention-norm-fusion",
            "true" if policy == "vectorized" else "false",
        ])
    return result


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("hf_infer emitted no JSON object")


def floats(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def errors(reference: array.array, actual: array.array) -> tuple[float, float, bool]:
    if len(reference) != len(actual):
        raise RuntimeError("complete-logit count changed")
    maximum = 0.0
    squared = 0.0
    finite = True
    for expected, observed in zip(reference, actual, strict=True):
        finite = finite and math.isfinite(observed)
        difference = abs(expected - observed)
        maximum = max(maximum, difference)
        squared += difference * difference
    return maximum, math.sqrt(squared / len(reference)), finite


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    selected_models = models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    records = []
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="microllm-swiglu-model-") as temporary:
        root = Path(temporary)
        for model in selected_models:
            for process_run in range(1, args.runs + 1):
                order = ["baseline", "vectorized"]
                if process_run % 2 == 0:
                    order.reverse()
                for policy in order:
                    stem = f"{model['name']}-t1024-p{process_run}-{policy}"
                    logits = root / f"{stem}.bin"
                    completed = subprocess.run(
                        command(args, model, policy, logits), text=True,
                        capture_output=True, check=False)
                    (logs / f"{stem}.stdout.txt").write_text(
                        completed.stdout, encoding="utf-8")
                    (logs / f"{stem}.stderr.txt").write_text(
                        completed.stderr, encoding="utf-8")
                    if completed.returncode != 0:
                        raise RuntimeError(f"{stem} failed: {completed.stderr}")
                    row = last_json(completed.stdout)
                    if (row.get("status") != "pass" or
                            row.get("inference_bthd_attention") is not True or
                            row.get("inference_bthd_bf16_qk") is not True or
                            row.get("inference_bthd_online_attention") is not False):
                        raise RuntimeError(f"{stem} did not run the retained policy")
                    if args.candidate_swish and (
                            row.get("bf16_grouped_gate_up_swish") is not
                            (policy == "vectorized")):
                        raise RuntimeError(
                            f"{stem} did not select the requested swish epilogue")
                    if args.candidate_bf16_norm and (
                            row.get("bf16_ffn_norm_fusion_enabled") is not
                            (policy == "vectorized")):
                        raise RuntimeError(
                            f"{stem} did not select the requested BF16 Norm route")
                    if args.candidate_attention_norm and (
                            row.get("bf16_attention_norm_fusion_enabled") is not
                            (policy == "vectorized")):
                        raise RuntimeError(
                            f"{stem} did not select the requested Attention Norm route")
                    row.update({
                        "record_type": (
                            "bf16_attention_norm_fusion_model_measurement"
                            if args.candidate_attention_norm else
                            "bf16_ffn_norm_fusion_model_measurement"
                            if args.candidate_bf16_norm else
                            "bf16_grouped_swiglu_model_measurement"
                            if args.candidate_swish else
                            "bf16_swiglu_vector_model_measurement"),
                        "model": model["name"], "revision": model["revision"],
                        "policy": policy, "sequence": 1024,
                        "process_run": process_run, "process_order": order,
                    })
                    records.append(row)
                    outputs[(model["name"], policy, process_run)] = floats(logits)

    comparisons = []
    for model in selected_models:
        selected = [row for row in records if row["model"] == model["name"]]
        grouped = {policy: [row for row in selected if row["policy"] == policy]
                   for policy in ("baseline", "vectorized")}
        reference = outputs[(model["name"], "baseline", 1)]
        maximum = 0.0
        rms = 0.0
        finite = True
        for process_run in range(1, args.runs + 1):
            current_max, current_rms, current_finite = errors(
                reference, outputs[(model["name"], "vectorized", process_run)])
            maximum = max(maximum, current_max)
            rms = max(rms, current_rms)
            finite = finite and current_finite
        baseline_speed = median(grouped["baseline"], "prefill_tokens_per_second")
        candidate_speed = median(grouped["vectorized"], "prefill_tokens_per_second")
        row = {
            "model": model["name"], "sequence": 1024, "batch": 1,
            "baseline_tokens_per_second": baseline_speed,
            "candidate_tokens_per_second": candidate_speed,
            "candidate_speedup": candidate_speed / baseline_speed,
            "baseline_engine_peak_bytes": int(median(
                grouped["baseline"], "engine_peak_bytes")),
            "candidate_engine_peak_bytes": int(median(
                grouped["vectorized"], "engine_peak_bytes")),
            "baseline_engine_allocation_calls": int(median(
                grouped["baseline"], "engine_allocation_calls")),
            "candidate_engine_allocation_calls": int(median(
                grouped["vectorized"], "engine_allocation_calls")),
            "maximum_absolute_logit_difference": maximum,
            "maximum_rms_logit_difference": rms,
            "finite_complete_logits": finite,
        }
        row["correctness_passed"] = finite and maximum == 0.0 and rms == 0.0
        row["performance_passed"] = row["candidate_speedup"] >= 1.005
        row["memory_passed"] = (
            row["candidate_engine_peak_bytes"] <= row["baseline_engine_peak_bytes"])
        comparisons.append(row)
    keep = all(row["correctness_passed"] and row["performance_passed"] and
               row["memory_passed"] for row in comparisons)
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": (
            "bf16_attention_norm_fusion_model_summary"
            if args.candidate_attention_norm else
            "bf16_ffn_norm_fusion_model_summary" if args.candidate_bf16_norm else
            "bf16_grouped_swiglu_model_summary" if args.candidate_swish else
            "bf16_swiglu_vector_model_summary"),
        "candidate_swish": args.candidate_swish,
        "candidate_bf16_norm": args.candidate_bf16_norm,
        "candidate_attention_norm": args.candidate_attention_norm,
        "raw_processes": len(records), "warmup": args.warmup,
        "steps": args.steps, "minimum_speedup": 1.005,
        "comparisons": comparisons, "keep_default": keep,
        "decision": (
            "keep BF16 Attention Norm fusion"
            if args.candidate_attention_norm and keep else
            "reject BF16 Attention Norm fusion"
            if args.candidate_attention_norm else
            "keep BF16 FFN Norm fusion" if args.candidate_bf16_norm and keep else
            "reject BF16 FFN Norm fusion" if args.candidate_bf16_norm else
            "keep grouped Swish epilogue" if args.candidate_swish and keep else
            "reject grouped Swish epilogue" if args.candidate_swish else
            "keep BF16 vectorized SwiGLU" if keep else
            "retain explicit operator; reject default route"),
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
