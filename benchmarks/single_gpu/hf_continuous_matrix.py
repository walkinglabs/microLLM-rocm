#!/usr/bin/env python3
"""Fresh-process official-model continuous-serving matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


SUITES = {
    "smoke": {
        "short_s2": {"slots": 2, "prompts": [8, 8], "outputs": [2, 3]},
    },
    "standard": {
        "short_s2": {"slots": 2, "prompts": [8, 8, 32, 32],
                     "outputs": [8, 16, 8, 16]},
        "short_s4": {"slots": 4, "prompts": [8, 8, 16, 16, 32, 32, 64, 64],
                     "outputs": [8, 16, 8, 16, 8, 16, 8, 16]},
        "long_s2": {"slots": 2, "prompts": [512, 512, 2048, 2048],
                    "outputs": [8, 16, 8, 16]},
        "long_s4": {"slots": 4,
                    "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
                    "outputs": [8, 8, 8, 8, 16, 16, 16, 16]},
    },
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--suite", choices=tuple(SUITES), default="standard")
    parser.add_argument("--models")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    result = parser.parse_args()
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    if result.warmup < 0 or result.steps <= 0 or result.runs <= 0 or \
            result.timeout_seconds <= 0:
        parser.error("warmup must be nonnegative and steps/runs/timeout positive")
    result.models = result.models.split(",") if result.models else None
    return result


def load_models(path: Path, selected: list[str] | None = None) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list) or not models:
        raise RuntimeError("manifest needs schema-version-1 models")
    by_name = {model.get("name"): model for model in models}
    names = selected or list(by_name)
    if None in by_name or len(by_name) != len(models) or not set(names) <= set(by_name):
        raise RuntimeError("manifest model names are missing, duplicate, or unknown")
    return [by_name[name] for name in names]


def model_cache_shape(config_path: str | Path) -> tuple[int, int, int]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config["num_key_value_heads"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dimension = int(config.get("head_dim", hidden // heads))
    return layers, kv_heads, head_dimension


def theoretical_cache_bytes(model: dict, case: dict, element_bytes: int = 2) -> int:
    layers, kv_heads, head_dimension = model_cache_shape(model["config"])
    capacity = max(prompt + output for prompt, output in
                   zip(case["prompts"], case["outputs"]))
    return (2 * layers * kv_heads * head_dimension * int(case["slots"]) *
            capacity * element_bytes)


def command(binary: Path, model: dict, case: dict,
            warmup: int, steps: int) -> list[str]:
    tokens = model["inference"]["token_ids"]
    return [
        str(binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "1", "--new-tokens", "0",
        "--warmup", str(warmup), "--steps", str(steps),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--workload", "continuous", "--kv-cache-dtype", "bf16",
        "--continuous-slots", str(case["slots"]),
        "--continuous-prompt-lengths", ",".join(map(str, case["prompts"])),
        "--continuous-new-token-lengths", ",".join(map(str, case["outputs"])),
    ]


def validate(record: dict, model: dict, case_name: str, case: dict,
             warmup: int, steps: int) -> dict:
    expected_tokens = sum(case["outputs"]) * steps
    expected_cache = theoretical_cache_bytes(model, case)
    required = {
        "status": "pass",
        "record_type": "official_continuous_serving_measurement",
        "request_count": len(case["prompts"]),
        "continuous_slots": case["slots"],
        "warmup": warmup,
        "steps": steps,
        "measured_tokens": expected_tokens,
        "prompt_lengths": case["prompts"],
        "new_token_lengths": case["outputs"],
        "deterministic_across_steps": True,
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"{model['name']} {case_name} changed its serving contract")
    if int(record.get("allocated_cache_bytes", -1)) != expected_cache or \
            not 0 < int(record.get("peak_active_cache_bytes", 0)) <= expected_cache or \
            not 0.0 < float(record.get("kv_cache_byte_utilization", 0.0)) <= 1.0 or \
            float(record.get("tokens_per_second", 0.0)) <= 0.0 or \
            int(record.get("engine_peak_bytes", 0)) <= 0 or \
            int(record.get("resident_weight_bytes", 0)) <= 0:
        raise RuntimeError(f"{model['name']} {case_name} has invalid memory/timing evidence")
    return {**record, "model": model["name"], "revision": model["revision"],
            "case": case_name, "expected_cache_bytes": expected_cache}


def main() -> int:
    args = options()
    models = load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for model in models:
        for case_name, case in SUITES[args.suite].items():
            for process_run in range(1, args.runs + 1):
                completed = subprocess.run(
                    command(args.binary, model, case, args.warmup, args.steps),
                    capture_output=True, text=True, timeout=args.timeout_seconds)
                if completed.returncode == 0:
                    lines = [line for line in completed.stdout.splitlines() if line.strip()]
                    if len(lines) != 1:
                        raise RuntimeError("continuous worker must emit one JSON line")
                    record = validate(json.loads(lines[0]), model, case_name, case,
                                      args.warmup, args.steps)
                else:
                    text = completed.stderr.strip() or completed.stdout.strip()
                    status = "oom" if "out of memory" in text.lower() else "failed"
                    record = {"schema_version": 1, "status": status,
                              "record_type": "official_continuous_serving_measurement",
                              "model": model["name"], "revision": model["revision"],
                              "case": case_name, "error": text}
                record["process_run"] = process_run
                rows.append(record)
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
    aggregates = []
    for model in models:
        for case_name in SUITES[args.suite]:
            selected = [row for row in rows if row.get("model") == model["name"] and
                        row.get("case") == case_name and row.get("status") == "pass"]
            aggregate = {"model": model["name"], "case": case_name,
                         "successful_runs": len(selected), "required_runs": args.runs}
            if len(selected) == args.runs:
                checksums = {int(row["token_checksum"]) for row in selected}
                if len(checksums) != 1:
                    raise RuntimeError(f"{model['name']} {case_name} checksum changed across runs")
                throughput = sorted(float(row["tokens_per_second"]) for row in selected)
                peak = sorted(int(row["engine_peak_bytes"]) for row in selected)
                aggregate.update({
                    "status": "pass",
                    "tokens_per_second_min": throughput[0],
                    "tokens_per_second_p50": statistics.median(throughput),
                    "tokens_per_second_max": throughput[-1],
                    "engine_peak_bytes_min": peak[0],
                    "engine_peak_bytes_p50": statistics.median(peak),
                    "engine_peak_bytes_max": peak[-1],
                    "token_checksum": checksums.pop(),
                })
            else:
                aggregate["status"] = "limited"
            aggregates.append(aggregate)
    summary = {
        "schema_version": 1,
        "track": "official_continuous_serving_matrix",
        "suite": args.suite,
        "warmup": args.warmup,
        "steps": args.steps,
        "runs": args.runs,
        "models": [model["name"] for model in models],
        "cases": SUITES[args.suite],
        "status": "pass" if all(row["status"] == "pass" for row in rows)
        else "complete_with_recorded_limits",
        "rows": rows,
        "aggregates": aggregates,
        "pytorch_boundary": "not measured; no variable-position PyTorch serving oracle",
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
