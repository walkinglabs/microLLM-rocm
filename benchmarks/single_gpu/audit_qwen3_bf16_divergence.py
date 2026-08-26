#!/usr/bin/env python3
"""Attribute the first Qwen3 BF16 decode split to a shared FP32 oracle."""

from __future__ import annotations

import argparse
import array
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX_SPEC = importlib.util.spec_from_file_location(
    "hf_inference_shape_matrix",
    Path(__file__).with_name("hf_inference_shape_matrix.py"))
MATRIX = importlib.util.module_from_spec(MATRIX_SPEC)
assert MATRIX_SPEC.loader is not None
MATRIX_SPEC.loader.exec_module(MATRIX)

MICRO_POLICIES = {
    "micro-fp32-fp32": (False, False, "fp32"),
    "micro-fp32-bf16": (False, False, "bf16"),
    "micro-bf16-fp32": (True, True, "fp32"),
    "micro-bf16-bf16": (True, True, "bf16"),
    "micro-ffn-bf16-fp32": (True, False, "fp32"),
    "micro-attention-bf16-fp32": (False, True, "fp32"),
    "micro-ffn-bf16-bf16": (True, False, "bf16"),
    "micro-attention-bf16-bf16": (False, True, "bf16"),
}
TORCH_POLICIES = ("torch-fp32", "torch-bf16")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pytorch-python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--capture-step", type=int, default=1)
    parser.add_argument("--forced-inputs", default="")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    parser.add_argument("--micro-policies", default=",".join(MICRO_POLICIES))
    parser.add_argument("--worker-dtype", choices=("fp32", "bf16"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    for path in (args.manifest, args.binary, args.pytorch_python):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if (args.context <= 0 or args.batch <= 0 or args.decode_tokens <= 0 or
            not 0 <= args.capture_step < args.decode_tokens or
            args.timeout_seconds <= 0):
        parser.error("context/decode/capture/timeout contract is invalid")
    if (args.worker_dtype is None) != (args.worker_output is None):
        parser.error("worker dtype and output must be supplied together")
    if args.worker_dtype is None and args.output_directory.exists() and \
            any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    try:
        args.forced_inputs = ([int(value) for value in args.forced_inputs.split(",")]
                              if args.forced_inputs else [])
    except ValueError:
        parser.error("forced inputs must be comma-separated nonnegative IDs")
    if (any(value < 0 for value in args.forced_inputs) or
            args.forced_inputs and len(args.forced_inputs) != args.decode_tokens):
        parser.error("forced inputs need one nonnegative ID per decode token")
    args.micro_policies = args.micro_policies.split(",")
    if (len(args.micro_policies) != len(set(args.micro_policies)) or
            any(policy not in MICRO_POLICIES for policy in args.micro_policies) or
            not {"micro-fp32-fp32", "micro-bf16-bf16"} <=
                set(args.micro_policies)):
        parser.error("micro policies must be unique known names and include FP32/current")
    return args


def last_json(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"worker emitted {len(lines)} JSON lines instead of one")
    return json.loads(lines[0])


def read_logit_rows(path: Path, vocabulary: int, batch: int) -> list[list[float]]:
    values = array.array("f")
    values.frombytes(path.read_bytes())
    if (batch <= 0 or len(values) != vocabulary * batch or
            not all(math.isfinite(value) for value in values)):
        raise RuntimeError(f"invalid complete logits: {path}")
    return [list(values[row * vocabulary:(row + 1) * vocabulary])
            for row in range(batch)]


def error(left: list[float], right: list[float]) -> tuple[float, float, bool]:
    if len(left) != len(right) or not left:
        raise ValueError("logit comparisons require equal non-empty vectors")
    differences = [left_value - right_value
                   for left_value, right_value in zip(left, right)]
    maximum = max(abs(value) for value in differences)
    rms = math.sqrt(sum(value * value for value in differences) / len(differences))
    return maximum, rms, all(value == 0.0 for value in differences)


def top_tokens(values: list[float], count: int = 3) -> list[int]:
    if count <= 0 or count > len(values):
        raise ValueError("top token count is outside the vector")
    return sorted(range(len(values)), key=values.__getitem__, reverse=True)[:count]


def micro_command(args: argparse.Namespace, model: dict, policy: str,
                  output: Path) -> list[str]:
    bf16_ffn, bf16_attention, cache_dtype = MICRO_POLICIES[policy]
    tokens = ",".join(str(token) for token in MATRIX.expanded_tokens(
        model["inference"]["token_ids"], args.context))
    command = [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", tokens, "--device", "hip", "--top-k", "1",
        "--batch", str(args.batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", cache_dtype,
        "--cache-capacity", str(args.context + args.decode_tokens),
        "--new-tokens", str(args.decode_tokens), "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--bf16-ffn", str(bf16_ffn).lower(),
        "--bf16-attention", str(bf16_attention).lower(),
        "--workload", "decode", "--cache-logits-output", str(output),
        "--cache-logits-step", str(args.capture_step),
    ]
    if args.forced_inputs:
        command.extend([
            "--forced-decode-inputs",
            ",".join(str(token) for token in args.forced_inputs),
        ])
    return command


def run_micro(args: argparse.Namespace, model: dict, vocabulary: int,
              policy: str, logits_path: Path) -> tuple[dict, list[float]]:
    completed = subprocess.run(
        micro_command(args, model, policy, logits_path), capture_output=True,
        text=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    record = last_json(completed.stdout)
    bf16_ffn, bf16_attention, cache_dtype = MICRO_POLICIES[policy]
    required = {
        "status": "pass", "parameter_count": model["parameter_count"],
        "token_count": args.context, "batch": args.batch,
        "decode_tokens": args.decode_tokens, "cache_logits_step": args.capture_step,
        "kv_cache_dtype": cache_dtype,
        "forced_decode_inputs": bool(args.forced_inputs),
        "forced_decode_input_count": len(args.forced_inputs),
    }
    if any(record.get(name) != wanted for name, wanted in required.items()):
        raise RuntimeError(f"{policy} changed the decode capture contract")
    if ((int(record.get("bf16_ffn_converted_tensors", 0)) > 0) != bf16_ffn or
            (int(record.get("bf16_attention_converted_tensors", 0)) > 0) !=
            bf16_attention):
        raise RuntimeError(f"{policy} did not execute its declared weight policy")
    rows = read_logit_rows(logits_path, vocabulary, args.batch)
    within = [error(rows[0], row) for row in rows[1:]]
    record.update({
        "framework_policy": policy, "precision_family": "microllm",
        "bf16_ffn_weights": bf16_ffn,
        "bf16_attention_weights": bf16_attention,
        "captured_rows_bitwise_equal": all(item[2] for item in within),
        "captured_rows_maximum_error": max(
            (item[0] for item in within), default=0.0),
        "captured_rows_rms_error": max(
            (item[1] for item in within), default=0.0),
    })
    return record, rows[0]


def torch_worker(args: argparse.Namespace, model: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    if (torch.version.hip and torch.cuda.device_count() == 0 and
            torch._C._cuda_getDeviceCount() > 0):
        if not args.allow_amdsmi_fallback:
            raise RuntimeError("AMDSMI reports zero devices; enable the explicit fallback")
        torch.cuda._device_count_amdsmi = lambda: -1
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device unavailable")
    dtype = torch.float32 if args.worker_dtype == "fp32" else torch.bfloat16
    device = torch.device("cuda:0")
    loaded = AutoModelForCausalLM.from_pretrained(
        Path(model["config"]).parent, torch_dtype=dtype,
        local_files_only=True, attn_implementation="sdpa").to(device).eval()
    seed = MATRIX.expanded_tokens(model["inference"]["token_ids"], args.context)
    input_ids = torch.tensor([seed], dtype=torch.long, device=device).repeat(
        args.batch, 1)
    suffix_rows = [[] for _ in range(args.batch)]
    captured = None
    with torch.inference_mode():
        prepared = loaded(input_ids=input_ids, use_cache=True)
        selected = torch.argmax(prepared.logits[:, -1, :], dim=-1)
        past = prepared.past_key_values
        for step in range(args.decode_tokens):
            if args.forced_inputs:
                selected = torch.full(
                    (args.batch,), args.forced_inputs[step],
                    dtype=torch.long, device=device)
            current = loaded(
                input_ids=selected[:, None], past_key_values=past, use_cache=True)
            selected = torch.argmax(current.logits[:, -1, :], dim=-1)
            for row, token in enumerate(selected.tolist()):
                suffix_rows[row].append(int(token))
            if step == args.capture_step:
                captured = current.logits[:, -1].float().cpu()
            past = current.past_key_values
    if captured is None:
        raise RuntimeError("PyTorch capture step was not executed")
    generated_rows_equal = all(row == suffix_rows[0] for row in suffix_rows[1:])
    if not generated_rows_equal and not args.forced_inputs:
        raise RuntimeError("identical PyTorch batch rows generated different tokens")
    captured_rows_equal = all(torch.equal(captured[0], row)
                              for row in captured[1:])
    captured.numpy().tofile(args.worker_output)
    parameter_count = sum(parameter.numel() for parameter in loaded.parameters())
    if parameter_count != model["parameter_count"]:
        raise RuntimeError("PyTorch runtime parameter count changed")
    return {
        "schema_version": 1, "record_type": "qwen3_bf16_divergence_worker",
        "status": "pass", "framework_policy": "torch-" + args.worker_dtype,
        "precision_family": "pytorch", "model": model["name"],
        "revision": model["revision"], "context": args.context,
        "batch": args.batch, "decode_tokens": args.decode_tokens,
        "capture_step": args.capture_step,
        "parameter_count": parameter_count,
        "resident_weight_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in loaded.parameters()),
        "generated_tokens": suffix_rows[0],
        "generated_rows_equal": generated_rows_equal,
        "captured_rows_bitwise_equal": captured_rows_equal,
        "forced_decode_inputs": bool(args.forced_inputs),
        "forced_decode_input_count": len(args.forced_inputs),
    }


def run_torch(args: argparse.Namespace, model: dict, vocabulary: int,
              dtype: str, logits_path: Path) -> tuple[dict, list[float]]:
    command = [
        str(args.pytorch_python), str(Path(__file__).resolve()),
        "--manifest", str(args.manifest), "--binary", str(args.binary),
        "--pytorch-python", str(args.pytorch_python),
        "--output-directory", str(args.output_directory), "--model", args.model,
        "--context", str(args.context), "--batch", str(args.batch),
        "--decode-tokens", str(args.decode_tokens),
        "--capture-step", str(args.capture_step),
        "--timeout-seconds", str(args.timeout_seconds),
        "--worker-dtype", dtype, "--worker-output", str(logits_path),
    ]
    if args.forced_inputs:
        command.extend([
            "--forced-inputs",
            ",".join(str(token) for token in args.forced_inputs),
        ])
    if args.allow_amdsmi_fallback:
        command.append("--allow-amdsmi-fallback")
    completed = subprocess.run(command, capture_output=True, text=True,
                               timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    record = last_json(completed.stdout)
    rows = read_logit_rows(logits_path, vocabulary, args.batch)
    within = [error(rows[0], row) for row in rows[1:]]
    record.update({
        "captured_rows_bitwise_equal": all(item[2] for item in within),
        "captured_rows_maximum_error": max(
            (item[0] for item in within), default=0.0),
        "captured_rows_rms_error": max(
            (item[1] for item in within), default=0.0),
    })
    return record, rows[0]


def summarize(samples: dict[str, tuple[dict, list[float]]], vocabulary: int,
              context: int, batch: int, decode_tokens: int,
              capture_step: int,
              forced_inputs: list[int] | tuple[int, ...] = ()) -> dict:
    oracle = samples["torch-fp32"][1]
    policies = []
    policy_order = ["torch-fp32"] + [
        name for name in MICRO_POLICIES if name in samples] + ["torch-bf16"]
    for name in policy_order:
        record, logits = samples[name]
        maximum, rms, bitwise = error(logits, oracle)
        top = top_tokens(logits)
        policies.append({
            "policy": name, "generated_tokens": record["generated_tokens"],
            "complete_logit_elements": len(logits),
            "versus_torch_fp32_maximum_error": maximum,
            "versus_torch_fp32_rms_error": rms,
            "versus_torch_fp32_bitwise_equal": bitwise,
            "argmax_token": top[0],
            "top3": [{"token": token, "logit": logits[token]} for token in top],
            "top1_top2_margin": logits[top[0]] - logits[top[1]],
            "resident_weight_bytes": int(record["resident_weight_bytes"]),
            "captured_rows_bitwise_equal": bool(
                record["captured_rows_bitwise_equal"]),
            "captured_rows_maximum_error": float(
                record["captured_rows_maximum_error"]),
            "captured_rows_rms_error": float(record["captured_rows_rms_error"]),
        })
    by_name = {row["policy"]: row for row in policies}

    def comparison(left: str, right: str) -> dict:
        maximum, rms, bitwise = error(samples[left][1], samples[right][1])
        return {"left": left, "right": right, "maximum_error": maximum,
                "rms_error": rms, "bitwise_equal": bitwise}

    fp32_alignment = comparison("micro-fp32-fp32", "torch-fp32")
    torch_top = by_name["torch-bf16"]["top3"]
    oracle_token = by_name["torch-fp32"]["argmax_token"]
    micro_matches_oracle = \
        by_name["micro-bf16-bf16"]["argmax_token"] == oracle_token
    torch_matches_oracle = by_name["torch-bf16"]["argmax_token"] == oracle_token
    gates = {
        "shared_inputs_before_capture": bool(forced_inputs) or len({
            tuple(row["generated_tokens"][:capture_step]) for row in policies}) == 1,
        "fp32_implementations_aligned":
            fp32_alignment["maximum_error"] <= 2.0e-4 and
            fp32_alignment["rms_error"] <= 4.0e-5,
        "fp32_oracle_argmax_agrees_with_micro_fp32":
            by_name["micro-fp32-fp32"]["argmax_token"] == oracle_token,
        "exactly_one_low_precision_policy_matches_fp32":
            micro_matches_oracle != torch_matches_oracle,
    }
    attribution = {
        "micro_fp32_vs_torch_fp32": fp32_alignment,
    }
    optional_comparisons = (
        ("bf16_cache_effect_with_fp32_weights",
         "micro-fp32-bf16", "micro-fp32-fp32"),
        ("bf16_cache_effect_with_bf16_weights",
         "micro-bf16-bf16", "micro-bf16-fp32"),
        ("bf16_weight_effect_with_fp32_cache",
         "micro-bf16-fp32", "micro-fp32-fp32"),
        ("bf16_weight_effect_with_bf16_cache",
         "micro-bf16-bf16", "micro-fp32-bf16"),
    )
    for label, left, right in optional_comparisons:
        if left in samples and right in samples:
            attribution[label] = comparison(left, right)
    return {
        "schema_version": 1, "record_type": "qwen3_bf16_divergence_audit",
        "status": "pass_diagnosed_precision_policy" if all(gates.values()) else "fail",
        "model": "Qwen/Qwen3-0.6B", "context": context, "batch": batch,
        "decode_tokens": decode_tokens, "capture_step": capture_step,
        "forced_inputs": list(forced_inputs),
        "vocabulary_size": vocabulary, "oracle": "torch-fp32",
        "gates": gates, "policy_rows": policies,
        "observations": {
            "torch_bf16_top_two_tied":
                torch_top[0]["logit"] == torch_top[1]["logit"],
            "micro_mixed_matches_fp32_argmax": micro_matches_oracle,
            "torch_bf16_matches_fp32_argmax": torch_matches_oracle,
        },
        "attribution": attribution,
        "oracle_matching_low_precision_policy":
            "micro-bf16-bf16" if micro_matches_oracle else "torch-bf16",
        "conclusion": (
            "the two low-precision policies choose different low-margin tokens; "
            "the recorded oracle-matching policy is selected by FP32 argmax, not by name"),
        "boundary": "one fixed T32/B1 decode trajectory; no performance claim",
    }


def main() -> int:
    args = options()
    model = MATRIX.load_models(args.manifest, [args.model])[0]
    if args.worker_dtype is not None:
        print(json.dumps(torch_worker(args, model), sort_keys=True))
        return 0
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logits_directory = args.output_directory / "logits"
    logits_directory.mkdir()
    samples = {}
    raw = []
    for policy in args.micro_policies:
        sample = run_micro(
            args, model, vocabulary, policy, logits_directory / f"{policy}.f32")
        samples[policy] = sample
        raw.append(sample[0])
        print(json.dumps(sample[0], sort_keys=True), flush=True)
    for policy in TORCH_POLICIES:
        dtype = policy.removeprefix("torch-")
        sample = run_torch(
            args, model, vocabulary, dtype, logits_directory / f"{policy}.f32")
        samples[policy] = sample
        raw.append(sample[0])
        print(json.dumps(sample[0], sort_keys=True), flush=True)
    summary = summarize(
        samples, vocabulary, args.context, args.batch,
        args.decode_tokens, args.capture_step, args.forced_inputs)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in raw),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"].startswith("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
