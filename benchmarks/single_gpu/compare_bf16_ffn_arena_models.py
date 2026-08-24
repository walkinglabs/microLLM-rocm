#!/usr/bin/env python3
"""Compare opt-in BF16 FFN Arena on complete official-model inference."""

from __future__ import annotations

import argparse
import array
import json
import statistics
import subprocess
import tempfile
from pathlib import Path


CASES = (
    ("prefill_t32_b1", "prefill", 32, 1),
    ("prefill_t512_b1", "prefill", 512, 1),
    ("prefill_t32_b4", "prefill", 32, 4),
    ("decode_b1", "decode", 0, 1),
    ("decode_b4", "decode", 0, 4),
)
POLICIES = ("baseline", "arena")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--arena-minimum-rows", type=int, default=1)
    parser.add_argument("--comparison-mode", choices=("ffn", "qkv", "core"),
                        default="ffn")
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.arena_minimum_rows <= 0):
        parser.error("runs/steps must be positive and warmup nonnegative")
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    return result


def load_models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError("manifest schema_version must be 1")
    result = document.get("models", [])
    if len(result) != 2:
        raise RuntimeError("formal Arena comparison requires two official models")
    for model in result:
        for key in ("name", "revision", "config", "weights", "inference"):
            if not model.get(key):
                raise RuntimeError(f"model lacks {key}")
        if not Path(model["config"]).is_file() or not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
    return result


def repeated_tokens(seed: list[int], length: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(length)]


def read_floats(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def max_difference(left: array.array, right: array.array) -> float:
    if len(left) != len(right):
        raise RuntimeError("complete-logit count changed")
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("hf_infer emitted no JSON object")


def command_for(args: argparse.Namespace, model: dict, case: tuple,
                policy: str, logits_path: Path) -> list[str]:
    name, workload, context, batch = case
    inference = model["inference"]
    tokens = (repeated_tokens(inference["token_ids"], context)
              if workload == "prefill" else inference["token_ids"])
    use_ffn_arena = policy == "arena" or args.comparison_mode in ("qkv", "core")
    command = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", str(batch),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true" if use_ffn_arena else "false",
        "--logits-output", str(logits_path),
    ]
    if use_ffn_arena:
        command.extend([
            "--bf16-ffn-arena-minimum-rows",
            str(args.arena_minimum_rows),
        ])
    if args.comparison_mode == "qkv":
        command.extend([
            "--bf16-qkv-arena", "true" if policy == "arena" else "false",
        ])
        if policy == "arena":
            command.extend([
                "--bf16-qkv-arena-minimum-rows",
                str(args.arena_minimum_rows),
            ])
    if args.comparison_mode == "core":
        command.extend([
            "--attention-core-arena",
            "true" if policy == "arena" else "false",
        ])
        if policy == "arena":
            command.extend([
                "--attention-core-arena-minimum-sequence",
                str(args.arena_minimum_rows),
            ])
    if workload == "prefill":
        command.extend([
            "--workload", "prefill", "--new-tokens", "0",
            "--warmup", "0", "--steps", "1",
            "--prefill-warmup", str(args.warmup),
            "--prefill-steps", str(args.steps),
            "--prefill-logits", "last",
        ])
    else:
        command.extend([
            "--workload", "both",
            "--new-tokens", str(inference["new_tokens"]),
            "--warmup", str(args.warmup), "--steps", str(args.steps),
            "--prefill-warmup", str(args.warmup),
            "--prefill-steps", str(args.steps),
            "--use-cache", "true", "--cache-prefill-mode", "full",
        ])
    return command


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    models = load_models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    records: list[dict] = []
    logits: dict[tuple[str, str, str, int], array.array] = {}
    with tempfile.TemporaryDirectory(prefix="microllm-bf16-arena-model-") as temporary:
        temporary_path = Path(temporary)
        for model in models:
            for case in CASES:
                case_name = case[0]
                for process_run in range(1, args.runs + 1):
                    order = list(POLICIES)
                    if process_run % 2 == 0:
                        order.reverse()
                    for policy in order:
                        stem = f"{model['name']}-{case_name}-p{process_run}-{policy}"
                        logits_path = temporary_path / f"{stem}.bin"
                        completed = subprocess.run(
                            command_for(args, model, case, policy, logits_path),
                            text=True, capture_output=True, check=False)
                        (logs / f"{stem}.stdout.txt").write_text(
                            completed.stdout, encoding="utf-8")
                        (logs / f"{stem}.stderr.txt").write_text(
                            completed.stderr, encoding="utf-8")
                        if completed.returncode != 0:
                            raise RuntimeError(
                                f"hf_infer failed for {stem}: {completed.stderr}")
                        record = last_json(completed.stdout)
                        if record.get("status") != "pass":
                            raise RuntimeError(f"invalid result for {stem}")
                        arena_field = {
                            "ffn": "bf16_ffn_arena_enabled",
                            "qkv": "bf16_qkv_arena_enabled",
                            "core": "attention_core_arena_enabled",
                        }[args.comparison_mode]
                        if policy == "arena" and not record.get(arena_field):
                            raise RuntimeError(f"Arena did not activate for {stem}")
                        if policy == "baseline" and record.get(arena_field):
                            raise RuntimeError(f"baseline activated Arena for {stem}")
                        if case[1] == "decode" and record.get("generated_tokens") != \
                                model["inference"]["expected_generated_tokens"]:
                            raise RuntimeError(f"generated token mismatch for {stem}")
                        record.update({
                            "record_type": (
                                "bf16_ffn_arena_model_measurement"
                                if args.comparison_mode == "ffn" else
                                "bf16_qkv_arena_model_measurement"
                                if args.comparison_mode == "qkv" else
                                "attention_core_arena_model_measurement"),
                            "model": model["name"],
                            "revision": model["revision"],
                            "case": case_name,
                            "case_kind": case[1],
                            "context": case[2] if case[1] == "prefill"
                                       else len(model["inference"]["token_ids"]),
                            "flattened_rows": (case[2] * case[3]
                                               if case[1] == "prefill"
                                               else case[3]),
                            "policy": policy,
                            "process_run": process_run,
                            "process_order": order,
                            "exact_expected_tokens": True,
                        })
                        records.append(record)
                        logits[(model["name"], case_name, policy, process_run)] = \
                            read_floats(logits_path)

    comparisons: list[dict] = []
    arena_prefix = {
        "ffn": "bf16_ffn_arena",
        "qkv": "bf16_qkv_arena",
        "core": "attention_core_arena",
    }[args.comparison_mode]
    for model in models:
        for case in CASES:
            case_name, kind, _, batch = case
            baseline_reference = logits[(model["name"], case_name, "baseline", 1)]
            selected = [row for row in records
                        if row["model"] == model["name"] and
                        row["case"] == case_name]
            grouped = {policy: [row for row in selected if row["policy"] == policy]
                       for policy in POLICIES}
            metric = "prefill_tokens_per_second" if kind == "prefill" \
                     else "decode_tokens_per_second"
            baseline_speed = median(grouped["baseline"], metric)
            arena_speed = median(grouped["arena"], metric)
            differences = [max_difference(
                logits[(model["name"], case_name, policy, run)],
                baseline_reference) for policy in POLICIES
                for run in range(1, args.runs + 1)]
            comparisons.append({
                "model": model["name"], "revision": model["revision"],
                "case": case_name, "case_kind": kind,
                "context": grouped["arena"][0]["context"], "batch": batch,
                "flattened_rows": grouped["arena"][0]["flattened_rows"],
                "baseline_tokens_per_second": baseline_speed,
                "arena_tokens_per_second": arena_speed,
                "arena_speedup": arena_speed / baseline_speed,
                "baseline_engine_allocation_calls": int(median(
                    grouped["baseline"], "engine_allocation_calls")),
                "arena_engine_allocation_calls": int(median(
                    grouped["arena"], "engine_allocation_calls")),
                "baseline_engine_peak_bytes": int(median(
                    grouped["baseline"], "engine_peak_bytes")),
                "arena_engine_peak_bytes": int(median(
                    grouped["arena"], "engine_peak_bytes")),
                "arena_capacity_bytes": int(median(
                    grouped["arena"], f"{arena_prefix}_capacity_bytes")),
                "arena_entries": int(median(
                    grouped["arena"], f"{arena_prefix}_entries")),
                "arena_hits": int(median(
                    grouped["arena"], f"{arena_prefix}_hits")),
                "arena_misses": int(median(
                    grouped["arena"], f"{arena_prefix}_misses")),
                "arena_eligible_calls": int(median(
                    grouped["arena"], f"{arena_prefix}_eligible_calls")),
                "arena_bypassed_calls": int(median(
                    grouped["arena"], f"{arena_prefix}_bypassed_calls")),
                "maximum_absolute_logit_difference": max(differences),
                "exact_expected_tokens": all(
                    row["exact_expected_tokens"] for row in grouped["arena"]),
            })
    correctness = all(
        row["maximum_absolute_logit_difference"] == 0 and
        row["exact_expected_tokens"] for row in comparisons)
    keep_rows = sum(row["arena_speedup"] >= 1.01 for row in comparisons)
    regressions = sum(row["arena_speedup"] < 0.98 for row in comparisons)
    eligible = [row for row in comparisons
                if row["flattened_rows"] >= args.arena_minimum_rows]
    bypassed = [row for row in comparisons
                if row["flattened_rows"] < args.arena_minimum_rows]
    arena_name = {
        "ffn": "model Arena", "qkv": "QKV Arena",
        "core": "Attention core Arena",
    }[args.comparison_mode]
    if args.arena_minimum_rows == 1:
        decision = ("keep universal model Arena" if regressions == 0 and
                    keep_rows == len(comparisons) else
                    "reject universal model Arena; inspect shape selection")
        if args.comparison_mode == "qkv":
            decision = ("keep universal QKV Arena" if regressions == 0 and
                        keep_rows == len(comparisons) else
                        "reject universal QKV Arena; inspect shape selection")
        elif args.comparison_mode == "core":
            decision = (
                "keep universal Attention core Arena"
                if regressions == 0 and keep_rows == len(comparisons)
                else "reject universal Attention core Arena; inspect shape selection")
    else:
        decision = (
            f"keep rows>={args.arena_minimum_rows} selective {arena_name}"
            if eligible and
            all(row["arena_speedup"] >= 1.01 for row in eligible) and
            all(0.98 <= row["arena_speedup"] <= 1.02 for row in bypassed)
            else f"reject rows>={args.arena_minimum_rows} selective {arena_name}")
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness else "fail",
        "record_type": ("bf16_ffn_arena_model_summary"
                        if args.comparison_mode == "ffn" else
                        "bf16_qkv_arena_model_summary"
                        if args.comparison_mode == "qkv" else
                        "attention_core_arena_model_summary"),
        "comparison_mode": args.comparison_mode,
        "raw_processes": len(records),
        "correctness_gate": correctness,
        "keep_rows": keep_rows,
        "regression_rows": regressions,
        "arena_minimum_rows": args.arena_minimum_rows,
        "eligible_rows": len(eligible),
        "bypassed_rows": len(bypassed),
        "comparisons": comparisons,
        "decision": decision,
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
