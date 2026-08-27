#!/usr/bin/env python3
"""Find minimal Qwen3 FFN BF16 layer combinations that flip the T128 oracle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace


AUDIT_PATH = Path(__file__).with_name("audit_qwen3_bf16_divergence.py")
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_qwen3_bf16_divergence", AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC.loader is not None
AUDIT_SPEC.loader.exec_module(AUDIT)

CONTEXT = 128
BATCH = 2
DECODE_TOKENS = 9
CAPTURE_STEP = 8
FORCED_INPUTS = [14582, 1, 374, 264, 3491, 429, 374, 537, 264]
GROUPS = (
    ("active-14-27", tuple(range(14, 28))),
    ("active-0-13", tuple(range(0, 14))),
    ("active-0-6", tuple(range(0, 7))),
    ("active-7-13", tuple(range(7, 14))),
    ("active-0-2", (0, 1, 2)),
    ("active-3-6", (3, 4, 5, 6)),
)
SINGLES = tuple((f"single-{layer}", (layer,)) for layer in range(7))
PAIRS = tuple(
    (f"pair-{left}-{right}", (left, right))
    for left, right in (
        (0, 1), (0, 2), (1, 2),
        (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)))
REPEATED = (
    ("active-0-2", (0, 1, 2)),
    ("pair-3-4", (3, 4)),
    ("pair-4-6", (4, 6)),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pytorch-python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    for path in (args.manifest, args.binary, args.pytorch_python, AUDIT_PATH):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def audit_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        binary=args.binary, context=CONTEXT, batch=BATCH,
        decode_tokens=DECODE_TOKENS, capture_step=CAPTURE_STEP,
        forced_inputs=FORCED_INPUTS, timeout_seconds=args.timeout_seconds,
        micro_ffn_fp32_layers=[],
    )


def run_selective(args: argparse.Namespace, model: dict, vocabulary: int,
                  oracle: list[float], name: str, active_layers: tuple[int, ...],
                  process_run: int) -> dict:
    case_directory = args.output_directory / "selective"
    case_directory.mkdir(exist_ok=True)
    stem = f"{name}-run{process_run}"
    logits_path = case_directory / f"{stem}.f32"
    all_layers = set(range(28))
    fp32_layers = sorted(all_layers - set(active_layers))
    command = AUDIT.micro_command(
        audit_args(args), model, "micro-ffn-bf16-fp32", logits_path)
    command.extend([
        "--bf16-ffn-fp32-layers", ",".join(str(layer) for layer in fp32_layers),
    ])
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stem} failed: " +
            (completed.stderr.strip() or completed.stdout.strip()))
    source = AUDIT.last_json(completed.stdout)
    if (source.get("status") != "pass" or
            source.get("forced_decode_inputs") is not True or
            source.get("forced_decode_input_count") != len(FORCED_INPUTS) or
            source.get("bf16_attention_converted_tensors") != 0 or
            source.get("bf16_ffn_converted_tensors") != len(active_layers) * 3 or
            source.get("bf16_ffn_fp32_layers") != fp32_layers):
        raise RuntimeError(f"{stem} changed its selective-layer contract")
    rows = AUDIT.read_logit_rows(logits_path, vocabulary, BATCH)
    within = [AUDIT.error(rows[0], row) for row in rows[1:]]
    maximum, rms, bitwise = AUDIT.error(rows[0], oracle)
    top = AUDIT.top_tokens(rows[0])
    record = {
        "schema_version": 1, "record_type": "qwen3_bf16_ffn_layer_sample",
        "status": "pass", "name": name, "process_run": process_run,
        "active_bf16_layers": list(active_layers), "fp32_layers": fp32_layers,
        "converted_tensors": int(source["bf16_ffn_converted_tensors"]),
        "argmax_token": top[0],
        "top3": [{"token": token, "logit": rows[0][token]} for token in top],
        "top1_top2_margin": rows[0][top[0]] - rows[0][top[1]],
        "versus_oracle_maximum_error": maximum,
        "versus_oracle_rms_error": rms,
        "versus_oracle_bitwise_equal": bitwise,
        "within_batch_maximum_error": max(
            (item[0] for item in within), default=0.0),
        "within_batch_rms_error": max(
            (item[1] for item in within), default=0.0),
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def summarize(records: list[dict], oracle_token: int) -> dict:
    by_key = {(row["name"], row["process_run"]): row for row in records}
    if len(by_key) != len(records):
        raise RuntimeError("duplicate layer-search sample")
    first = {name: by_key[(name, 1)] for name, _layers in (*GROUPS, *SINGLES, *PAIRS)}
    single_flips = [
        row["active_bf16_layers"][0] for name, _layers in SINGLES
        if (row := first[name])["argmax_token"] != oracle_token]
    pair_flips = [
        row["active_bf16_layers"] for name, _layers in PAIRS
        if (row := first[name])["argmax_token"] != oracle_token]
    repeat_rows = []
    for name, layers in REPEATED:
        samples = [by_key[(name, run)] for run in (1, 2, 3)]
        repeat_rows.append({
            "name": name, "active_bf16_layers": list(layers),
            "argmax_tokens": [row["argmax_token"] for row in samples],
            "minimum_margin": min(row["top1_top2_margin"] for row in samples),
            "maximum_margin": max(row["top1_top2_margin"] for row in samples),
            "maximum_oracle_error": max(
                row["versus_oracle_maximum_error"] for row in samples),
        })
    repeat_by_name = {row["name"]: row for row in repeat_rows}
    gates = {
        "upper_half_safe": first["active-14-27"]["argmax_token"] == oracle_token,
        "lower_half_flips": first["active-0-13"]["argmax_token"] != oracle_token,
        "both_lower_subgroups_flip":
            first["active-0-2"]["argmax_token"] != oracle_token and
            first["active-3-6"]["argmax_token"] != oracle_token,
        "no_single_layer_flips": not single_flips,
        "only_pair_3_4_flips": pair_flips == [[3, 4]],
        "triple_0_1_2_repeats":
            repeat_by_name["active-0-2"]["argmax_tokens"] == [25, 25, 25],
        "pair_3_4_repeats":
            repeat_by_name["pair-3-4"]["argmax_tokens"] == [25, 25, 25],
        "near_pair_4_6_stays_oracle":
            repeat_by_name["pair-4-6"]["argmax_tokens"] ==
            [oracle_token, oracle_token, oracle_token],
    }
    return {
        "schema_version": 1, "record_type": "qwen3_bf16_ffn_layer_search",
        "status": "pass_minimal_combinations_found" if all(gates.values()) else "fail",
        "model": "Qwen/Qwen3-0.6B", "context": CONTEXT, "batch": BATCH,
        "capture_step": CAPTURE_STEP, "forced_inputs": FORCED_INPUTS,
        "oracle_argmax": oracle_token, "process_rows": len(records),
        "gates": gates, "single_layer_flips": single_flips,
        "pair_layer_flips": pair_flips,
        "minimal_flipping_sets": [[0, 1, 2], [3, 4]],
        "repeat_rows": repeat_rows,
        "groups": [first[name] for name, _layers in GROUPS],
        "singles": [first[name] for name, _layers in SINGLES],
        "pairs": [first[name] for name, _layers in PAIRS],
        "boundary": (
            "fixed forced-input top-2 search; minimal means no tested proper "
            "single/pair subset flips inside the declared groups, not a global proof "
            "over all 2^28 layer combinations"),
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    base = args.output_directory / "base"
    command = [
        str(args.pytorch_python), str(AUDIT_PATH),
        "--manifest", str(args.manifest), "--binary", str(args.binary),
        "--pytorch-python", str(args.pytorch_python),
        "--output-directory", str(base), "--context", str(CONTEXT),
        "--batch", str(BATCH), "--decode-tokens", str(DECODE_TOKENS),
        "--capture-step", str(CAPTURE_STEP), "--forced-inputs",
        ",".join(str(token) for token in FORCED_INPUTS),
        "--micro-policies",
        "micro-fp32-fp32,micro-ffn-bf16-fp32,micro-bf16-bf16",
        "--timeout-seconds", str(args.timeout_seconds),
    ]
    if args.allow_amdsmi_fallback:
        command.append("--allow-amdsmi-fallback")
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    base_summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    if (base_summary.get("status") != "pass_diagnosed_precision_policy" or
            not all(base_summary.get("gates", {}).values())):
        raise RuntimeError("base FFN/oracle audit failed")
    config = json.loads(Path(
        AUDIT.MATRIX.load_models(args.manifest, ["qwen3-0.6b"])[0]["config"]
    ).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    oracle = AUDIT.read_logit_rows(
        base / "logits" / "torch-fp32.f32", vocabulary, BATCH)[0]
    model = AUDIT.MATRIX.load_models(args.manifest, ["qwen3-0.6b"])[0]
    records = []
    for name, layers in (*GROUPS, *SINGLES, *PAIRS):
        records.append(run_selective(
            args, model, vocabulary, oracle, name, layers, 1))
    for name, layers in REPEATED:
        for run in (2, 3):
            records.append(run_selective(
                args, model, vocabulary, oracle, name, layers, run))
    summary = summarize(records, oracle_token=320)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"].startswith("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
