#!/usr/bin/env python3
"""Profile the retained B1T512 BF16 training path with load subtraction."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_NAMES = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
CONTEXT = 512
MOMENT_THRESHOLD = 1_048_576
OPTIMIZER_METADATA_BYTES_PER_STEP = {
    "qwen2.5-0.5b": 13_888,
    "deepseek-r1-distill-qwen-1.5b": 12_608,
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--rocprof", default="rocprofv3")
    args = parser.parse_args()
    if not args.manifest.is_file() or not args.binary.is_file():
        parser.error("profile inputs are unavailable")
    return args


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != MODEL_NAMES:
        raise RuntimeError("current training profile requires both pinned models")
    for model in result:
        training = model.get("training", {})
        if not model.get("config") or not model.get("weights") or \
                not training.get("tokens") or not training.get("learning_rate"):
            raise RuntimeError(f"incomplete training manifest for {model.get('name')}")
    return result


def repeated(token_text: str, length: int) -> str:
    seed = [int(value) for value in token_text.split(",")]
    if len(seed) < 2 or any(value < 0 for value in seed):
        raise RuntimeError("training token seed must contain nonnegative IDs")
    return ",".join(str(seed[index % len(seed)]) for index in range(length))


def app_command(args: argparse.Namespace, model: dict, steps: int) -> list[str]:
    training = model["training"]
    return [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"],
        "--tokens", repeated(training["tokens"], CONTEXT + 1),
        "--device", "hip", "--batch", "1",
        "--learning-rate", str(training["learning_rate"]),
        "--warmup", "0", "--steps", str(steps),
        "--linear-precision", "bf16",
        "--bf16-weight-mirrors", "true",
        "--adamw-implementation", "auto",
        "--adamw-moment-precision", "bf16",
        "--adamw-bf16-multi-tensor-threshold", str(MOMENT_THRESHOLD),
        "--tied-embedding-sparse-add", "true",
        "--unique-gradient-inplace-add", "false",
        "--attention-rope-layout-fusion", "true",
        "--attention-context-layout-fusion", "true",
        "--attention-layout-plan-cache", "false",
        "--attention-gemm-scale-fusion", "false",
        "--attention-paired-gqa-repeat", "false",
        "--attention-gqa-value-broadcast", "false",
        "--attention-gqa-forward-value-broadcast", "false",
    ]


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("profiled training application emitted no JSON")


def validate_record(record: dict, steps: int, model_name: str) -> None:
    expected = {
        "status": "pass",
        "compute_dtype": "bf16_linear_fp32_master",
        "bf16_weight_mirrors_enabled": True,
        "adamw_implementation": "auto",
        "adamw_moment_precision": "bf16",
        "adamw_bf16_multi_tensor_threshold": MOMENT_THRESHOLD,
        "tied_embedding_sparse_add": True,
        "unique_gradient_inplace_add": False,
        "attention_rope_layout_fusion": True,
        "attention_context_layout_fusion": True,
        "attention_layout_plan_cache": False,
        "attention_gemm_scale_fusion": False,
        "attention_paired_gqa_repeat": False,
        "attention_gqa_value_broadcast": False,
        "attention_gqa_forward_value_broadcast": False,
        "batch": 1,
        "context": CONTEXT,
        "warmup": 0,
        "steps": steps,
    }
    changed = [key for key, value in expected.items() if record.get(key) != value]
    expected_metadata_bytes = OPTIMIZER_METADATA_BYTES_PER_STEP[model_name] * steps
    if changed or record.get("parameter_changed") is not True or \
            record.get("optimizer_host_to_device_calls") != steps or \
            record.get("optimizer_host_to_device_bytes") != expected_metadata_bytes or \
            record.get("optimizer_device_to_host_calls") != 0 or \
            record.get("optimizer_device_to_host_bytes") != 0:
        raise RuntimeError("profiled training contract changed: " + ", ".join(changed))


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    profiles: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="microllm-current-training-") as temporary:
        scratch = Path(temporary)
        for model in models(args.manifest):
            output = args.output_directory / model["name"]
            output.mkdir(parents=True, exist_ok=True)
            stats: dict[int, Path] = {}
            for steps in (1, 3):
                trace = scratch / f"{model['name']}-{steps}"
                trace.mkdir()
                command = [
                    args.rocprof, "--kernel-trace", "--stats",
                    "--output-format", "csv", "--output-file", "trace",
                    "--output-directory", str(trace), "--",
                    *app_command(args, model, steps),
                ]
                completed = subprocess.run(
                    command, text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stdout + completed.stderr)
                record = last_json(completed.stdout)
                validate_record(record, steps, model["name"])
                record.update({
                    "record_type": "current_training_profile_run",
                    "model": model["name"], "revision": model["revision"],
                    "training_steps_profiled": steps,
                })
                records.append(record)
                (output / f"{steps}-step-run.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
                source = trace / "trace_kernel_stats.csv"
                if not source.is_file():
                    raise RuntimeError("rocprof did not emit training Kernel stats")
                destination = output / f"{steps}-step-kernel-stats.csv"
                shutil.copy2(source, destination)
                stats[steps] = destination
            delta = subprocess.run([
                sys.executable,
                str(Path(__file__).with_name("profile_step_delta.py")),
                "--one-step", str(stats[1]), "--many-step", str(stats[3]),
                "--many-step-count", "3", "--output-directory", str(output),
                "--track", "training_kernel_phase_delta",
            ], text=True, capture_output=True, check=False)
            if delta.returncode != 0:
                raise RuntimeError(delta.stdout + delta.stderr)
            profile = json.loads((output / "profile-delta.json").read_text(
                encoding="utf-8"))
            profile.update({"model": model["name"], "revision": model["revision"]})
            (output / "profile-delta.json").write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            profiles.append(profile)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "current_training_profile_summary",
        "profile_processes": len(records),
        "batch": 1,
        "context": CONTEXT,
        "derived_steps": 2,
        "linear_precision": "bf16",
        "adamw_moment_precision": "bf16",
        "adamw_bf16_multi_tensor_threshold": MOMENT_THRESHOLD,
        "models": profiles,
        "decision": "reselect a training hotspot from the current retained binary",
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"profile_current_training: {error}", file=sys.stderr)
        raise SystemExit(2)
