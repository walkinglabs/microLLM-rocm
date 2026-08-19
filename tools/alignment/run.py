#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run and compare microLLM and PyTorch model traces")
    parser.add_argument("--microllm-binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter containing PyTorch")
    parser.add_argument("--microllm-device", choices=("cpu", "hip"), default="cpu")
    parser.add_argument("--pytorch-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max-captured-elements", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=3.0e-5)
    parser.add_argument("--rtol", type=float, default=3.0e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.warmup < 0 or args.repetitions <= 0:
        parser.error("seed/warmup must be non-negative and repetitions must be positive")
    return args


def git_value(*arguments):
    completed = subprocess.run(
        ["git", *arguments], cwd=PROJECT, text=True, capture_output=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def tool_version(command):
    executable = shutil.which(command)
    if not executable:
        return None
    completed = subprocess.run([executable, "--version"], text=True, capture_output=True)
    text = (completed.stdout or completed.stderr).splitlines()
    return text[0] if text else None


def run_stage(name, command, output_directory, manifest):
    manifest["commands"][name] = command
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    logs = output_directory / "logs"
    logs.mkdir(exist_ok=True)
    (logs / f"{name}.stdout.log").write_text(completed.stdout)
    (logs / f"{name}.stderr.log").write_text(completed.stderr)
    manifest["stages"][name] = {
        "returncode": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "fail",
        "stdout": f"logs/{name}.stdout.log",
        "stderr": f"logs/{name}.stderr.log",
    }
    return completed.returncode


def write_manifest(path, manifest):
    manifest["artifacts"] = sorted(
        str(item.relative_to(path)) for item in path.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    )
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit("output directory is not empty; pass --overwrite to replace its files")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.datetime.now(datetime.UTC).strftime("alignment-%Y%m%dT%H%M%SZ")

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "running",
        "repository": {
            "path": str(PROJECT),
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "cmake": tool_version("cmake"),
            "gcc": tool_version("gcc"),
            "g++": tool_version("g++"),
            "hipcc": tool_version("hipcc"),
            "rocprofv3": tool_version("rocprofv3"),
        },
        "configuration": {
            "microllm_binary": str(args.microllm_binary.resolve()),
            "pytorch_python": str(Path(args.python).resolve()),
            "microllm_device": args.microllm_device,
            "pytorch_device": args.pytorch_device,
            "seed": args.seed,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "max_captured_elements": args.max_captured_elements,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "commands": {},
        "stages": {},
    }

    micro_command = [
        str(args.microllm_binary.resolve()),
        "--output", str(output),
        "--device", args.microllm_device,
        "--run-id", run_id,
        "--seed", str(args.seed),
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
        "--max-captured-elements", str(args.max_captured_elements),
    ]
    torch_command = [
        args.python,
        str(HERE / "pytorch_runner.py"),
        "--input", str(output),
        "--output", str(output),
        "--device", args.pytorch_device,
        "--run-id", run_id,
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
        "--max-captured-elements", str(args.max_captured_elements),
    ]
    compare_command = [
        sys.executable,
        str(HERE / "compare.py"),
        "--microllm", str(output),
        "--pytorch", str(output),
        "--output", str(output),
        "--atol", str(args.atol),
        "--rtol", str(args.rtol),
    ]

    status = 0
    for name, command in (
        ("microllm", micro_command),
        ("pytorch", torch_command),
        ("compare", compare_command),
    ):
        status = run_stage(name, command, output, manifest)
        if status != 0:
            break
    manifest["status"] = "pass" if status == 0 else "fail"
    manifest["completed_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    write_manifest(output, manifest)
    print(f"alignment_directory={output}")
    print(f"status={manifest['status']}")
    if status != 0:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
