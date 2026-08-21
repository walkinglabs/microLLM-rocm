#!/usr/bin/env python3
"""Compare real-model FP32 and BF16 KV-cache logits after cached decode."""

import argparse
from array import array
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def positive_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return result


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--micro-binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models", default="qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--contexts", type=positive_list, default="32,512,2048")
    parser.add_argument("--batches", type=positive_list, default="1,8")
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--bf16-fp32-layers", default="")
    parser.add_argument("--max-absolute-error", type=float, default=0.25)
    parser.add_argument("--maximum-rmse", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    result = parser.parse_args()
    if result.decode_tokens < 2:
        parser.error("--decode-tokens must be at least two to exercise cached attention")
    if result.warmup < 0 or result.steps <= 0:
        parser.error("warmup must be nonnegative and steps positive")
    return result


def load_models(path: Path, selected: set[str]) -> list[dict]:
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    result = [model for model in models if model["name"] in selected]
    if {model["name"] for model in result} != selected:
        raise RuntimeError("requested model is missing from manifest")
    return result


def expanded_tokens(seed: list[int], context: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(context)]


def run_json(command: list[str], timeout: int) -> dict:
    process = subprocess.run(command, text=True, capture_output=True, timeout=timeout,
                             env=os.environ.copy())
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    for line in reversed(process.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("microLLM did not emit JSON")


def read_float32(path: Path) -> array:
    values = array("f")
    with path.open("rb") as stream:
        values.frombytes(stream.read())
    return values


def one_run(args: argparse.Namespace, model: dict, context: int, batch: int,
            dtype: str, logits_path: Path) -> tuple[dict, array]:
    tokens = expanded_tokens(model["inference"]["token_ids"], context)
    command = [
        str(args.micro_binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in tokens), "--device", "hip",
        "--top-k", "1", "--new-tokens", str(args.decode_tokens),
        "--warmup", str(args.warmup), "--steps", str(args.steps),
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--bf16-ffn", "true", "--bf16-attention", "true", "--workload", "decode",
        "--batch", str(batch), "--use-cache", "true", "--cache-prefill-mode", "full",
        "--batch-argmax-mode", "device", "--kv-cache-dtype", dtype,
        "--cache-logits-output", str(logits_path),
    ]
    if dtype == "bf16" and args.bf16_fp32_layers:
        command.extend(["--kv-cache-fp32-layers", args.bf16_fp32_layers])
    record = run_json(command, args.timeout_seconds)
    values = read_float32(logits_path)
    vocabulary = int(json.loads(Path(model["config"]).read_text(
        encoding="utf-8"))["vocab_size"])
    expected = batch * vocabulary
    if len(values) != expected:
        raise RuntimeError(f"cached logits count {len(values)} != expected {expected}")
    return record, values


def compare_case(args: argparse.Namespace, model: dict, context: int,
                 batch: int, temporary: Path) -> dict:
    raw = {}
    logits = {}
    for dtype in ("fp32", "bf16"):
        raw[dtype], logits[dtype] = one_run(
            args, model, context, batch, dtype,
            temporary / f"{model['name']}-t{context}-b{batch}-{dtype}.bin")
    all_finite = all(math.isfinite(value) for values in logits.values()
                     for value in values)
    differences = [abs(left - right)
                   for left, right in zip(logits["fp32"], logits["bf16"])]
    rmse = math.sqrt(sum(value * value for value in differences) / len(differences))
    maximum_relative = max(
        difference / max(abs(reference), 1.0e-4)
        for difference, reference in zip(differences, logits["fp32"]))
    vocabulary = int(json.loads(Path(model["config"]).read_text(
        encoding="utf-8"))["vocab_size"])
    fp32_top = []
    bf16_top = []
    for row in range(batch):
        start, end = row * vocabulary, (row + 1) * vocabulary
        fp32_top.append(max(range(start, end), key=logits["fp32"].__getitem__) - start)
        bf16_top.append(max(range(start, end), key=logits["bf16"].__getitem__) - start)
    maximum = max(differences)
    result = {
        "schema_version": 1,
        "record_type": "kv_cache_precision_comparison",
        "model": model["name"], "revision": model["revision"],
        "context": context, "batch": batch, "decode_tokens": args.decode_tokens,
        "logit_count": len(differences), "maximum_absolute_error": maximum,
        "maximum_relative_error": maximum_relative, "rmse": rmse,
        "all_logits_finite": all_finite,
        "fp32_top_tokens": fp32_top, "bf16_top_tokens": bf16_top,
        "top_tokens_equal": fp32_top == bf16_top,
        "generated_tokens_equal": raw["fp32"]["generated_tokens"] ==
                                  raw["bf16"]["generated_tokens"],
        "fp32_generated_tokens": raw["fp32"]["generated_tokens"],
        "bf16_generated_tokens": raw["bf16"]["generated_tokens"],
        "fp32_cache_bytes": raw["fp32"]["kv_cache_actual_bytes"],
        "bf16_cache_bytes": raw["bf16"]["kv_cache_actual_bytes"],
        "bf16_fp32_layer_policy": args.bf16_fp32_layers,
        "bf16_fp32_layers": raw["bf16"].get("kv_cache_fp32_layers", 0),
        "bf16_bf16_layers": raw["bf16"].get("kv_cache_bf16_layers", 0),
        "bf16_fp32_bytes": raw["bf16"].get("kv_cache_fp32_bytes", 0),
        "bf16_bf16_bytes": raw["bf16"].get("kv_cache_bf16_bytes", 0),
        "cache_byte_reduction": raw["fp32"]["kv_cache_actual_bytes"] /
                                raw["bf16"]["kv_cache_actual_bytes"],
        "decode_throughput_ratio_bf16_over_fp32":
            raw["bf16"]["decode_tokens_per_second"] /
            raw["fp32"]["decode_tokens_per_second"],
        "peak_memory_ratio_bf16_over_fp32":
            raw["bf16"]["engine_peak_bytes"] / raw["fp32"]["engine_peak_bytes"],
    }
    total_layers = result["bf16_fp32_layers"] + result["bf16_bf16_layers"]
    expected_reduction = 2.0 * total_layers / (
        total_layers + result["bf16_fp32_layers"])
    result["expected_cache_byte_reduction"] = expected_reduction
    result["status"] = "pass" if (
        all_finite and maximum <= args.max_absolute_error and
        rmse <= args.maximum_rmse and
        result["top_tokens_equal"] and result["generated_tokens_equal"] and
        math.isclose(result["cache_byte_reduction"], expected_reduction,
                     rel_tol=1.0e-12, abs_tol=0.0)
    ) else "failed"
    return result


def main() -> int:
    args = options()
    selected = {name for name in args.models.split(",") if name}
    models = load_models(args.manifest, selected)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="microllm-kv-precision-") as directory:
        temporary = Path(directory)
        for model in models:
            for context in args.contexts:
                for batch in args.batches:
                    record = compare_case(args, model, context, batch, temporary)
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n"
                                for record in records), encoding="utf-8")
    summary = {
        "schema_version": 1, "track": "kv_cache_precision",
        "thresholds": {"maximum_absolute_error": args.max_absolute_error,
                       "rmse": args.maximum_rmse},
        "status": "pass" if all(record["status"] == "pass" for record in records)
                  else "failed",
        "records": records,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
