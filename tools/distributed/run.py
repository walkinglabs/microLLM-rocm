#!/usr/bin/env python3
import argparse
import datetime
import json
import platform
import shutil
import statistics
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a recorded microLLM data-parallel experiment")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=("tiny", "model-s"), default="tiny")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--bucket-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--parameter-check-interval", type=int, default=1)
    parser.add_argument("--inplace-bucket-average", choices=("true", "false"),
                        default="true")
    parser.add_argument("--persistent-gradient-buckets", choices=("true", "false"),
                        default="false")
    parser.add_argument("--gradient-bucket-views", choices=("true", "false"),
                        default="false")
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (args.steps <= 0 or args.bucket_bytes < 4 or args.seed < 0 or
            args.parameter_check_interval < 0 or args.batch <= 0 or args.context < 0):
        parser.error("steps/bucket size must be positive and seed non-negative")
    return args


def command_value(command):
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def version(name):
    executable = shutil.which(name)
    if not executable:
        return None
    completed = subprocess.run([executable, "--version"], text=True, capture_output=True)
    lines = (completed.stdout or completed.stderr).splitlines()
    return lines[0] if lines else None


def median(records, key):
    return statistics.median(record[key] for record in records)


def main():
    args = parse_args()
    repository = {
        "commit": command_value(["git", "rev-parse", "HEAD"]),
        "branch": command_value(["git", "branch", "--show-current"]),
        "dirty": bool(command_value(["git", "status", "--porcelain"])),
    }
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is not empty; pass --overwrite")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "trace.jsonl"
    command = [
        str(args.binary.resolve()),
        "--model", args.model,
        "--steps", str(args.steps),
        "--bucket-bytes", str(args.bucket_bytes),
        "--parameter-check-interval", str(args.parameter_check_interval),
        "--inplace-bucket-average", args.inplace_bucket_average,
        "--persistent-gradient-buckets", args.persistent_gradient_buckets,
        "--gradient-bucket-views", args.gradient_bucket_views,
        "--seed", str(args.seed),
        "--batch", str(args.batch),
        "--context", str(args.context),
        "--trace", str(trace),
    ]
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    (output / "metrics.jsonl").write_text(completed.stdout)
    (output / "stderr.txt").write_text(completed.stderr)
    metrics = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode == 0 and len(metrics) != args.steps:
        raise RuntimeError("distributed CLI did not emit one metric record per step")
    if trace.exists():
        for line in trace.read_text().splitlines():
            if line.strip():
                json.loads(line)
    summary = {
        "schema_version": 1,
        "status": "pass" if completed.returncode == 0 else "fail",
        "steps": len(metrics),
        "initial_loss": metrics[0]["mean_loss"] if metrics else None,
        "final_loss": metrics[-1]["mean_loss"] if metrics else None,
        "maximum_parameter_difference": max(
            (record["parameter_max_difference"] for record in metrics), default=None),
        "median_forward_backward_ms": median(metrics, "forward_backward_ms") if metrics else None,
        "median_communication_ms": median(metrics, "communication_ms") if metrics else None,
        "median_optimizer_ms": median(metrics, "optimizer_ms") if metrics else None,
        "median_verification_ms": median(metrics, "verification_ms") if metrics else None,
        "parameter_checks": sum(
            1 for record in metrics if record["parameter_check_performed"]),
        "median_total_ms": median(metrics, "total_ms") if metrics else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": summary["status"],
        "repository": repository,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cmake": version("cmake"),
            "gcc": version("gcc"),
            "g++": version("g++"),
            "hipcc": version("hipcc"),
            "rocprofv3": version("rocprofv3"),
        },
        "configuration": {
            "binary": str(args.binary.resolve()),
            "model": args.model,
            "steps": args.steps,
            "bucket_bytes": args.bucket_bytes,
            "parameter_check_interval": args.parameter_check_interval,
            "inplace_bucket_average": args.inplace_bucket_average == "true",
            "persistent_gradient_buckets": args.persistent_gradient_buckets == "true",
            "gradient_bucket_views": args.gradient_bucket_views == "true",
            "seed": args.seed,
            "batch": args.batch,
            "context": args.context,
            "devices": [0, 1],
        },
        "command": command,
        "returncode": completed.returncode,
        "artifacts": ["metrics.jsonl", "stderr.txt", "summary.json", "trace.jsonl"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"distributed_output={output}")
    print(f"status={summary['status']}")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
