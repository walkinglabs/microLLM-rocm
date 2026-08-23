#!/usr/bin/env python3
"""Execute MI300 INT8 hipBLASLt kernels without claiming model-level INT8 support."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


OFFICIAL_PEAK_TOPS = 2614.9
OFFICIAL_BANDWIDTH_TBPS = 5.3


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--sizes", default="128,256,512,1024,2048,4096")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    result = parser.parse_args()
    try:
        result.sizes = [int(value) for value in result.sizes.split(",")]
    except ValueError as error:
        parser.error(f"invalid INT8 size list: {error}")
    if not result.binary.is_file() or not result.sizes or \
            len(set(result.sizes)) != len(result.sizes) or \
            any(size <= 0 or size > 4096 for size in result.sizes) or \
            result.warmup < 0 or result.repetitions <= 0:
        parser.error("invalid INT8 probe options")
    return result


def gpu_state(index: int) -> dict:
    completed = subprocess.run(
        ["rocm-smi", "--showuse", "--showmemuse", "--json"],
        capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("cannot query GPU state")
    card = json.loads(completed.stdout).get(f"card{index}")
    if not isinstance(card, dict):
        raise RuntimeError(f"physical GPU {index} was not reported")
    return {"physical_gpu_index": index,
            "gpu_use_percent": int(card["GPU use (%)"]),
            "vram_percent": int(card["GPU Memory Allocated (VRAM%)"])}


def require_idle(index: int | None, maximum_vram: int,
                 maximum_use: int, boundary: str) -> dict | None:
    if index is None:
        return None
    state = gpu_state(index)
    if state["vram_percent"] > maximum_vram or \
            state["gpu_use_percent"] > maximum_use:
        raise RuntimeError(f"GPU occupied at {boundary}: {state}")
    return state


def enrich(record: dict, size: int) -> dict:
    if record.get("shape") != [size, size, size] or \
            record.get("input_dtype") != "int8" or \
            record.get("output_dtype") != "int32" or \
            record.get("accuracy_passed") is not True or \
            record.get("maximum_sample_error") != 0:
        raise RuntimeError("INT8 worker contract or exact samples failed")
    intensity = size / 3.0  # 2N^3 ops / (2*N^2 int8 + N^2 int32)
    bandwidth_bound = OFFICIAL_BANDWIDTH_TBPS * intensity
    roofline_bound = min(OFFICIAL_PEAK_TOPS, bandwidth_bound)
    achieved = float(record["achieved_tops"])
    return {**record,
            "size": size,
            "arithmetic_intensity_ops_per_byte": intensity,
            "bandwidth_bound_tops": bandwidth_bound,
            "roofline_bound_tops": roofline_bound,
            "roofline_utilization": achieved / roofline_bound}


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for size in args.sizes:
        pre = require_idle(args.physical_gpu_index,
                           args.max_idle_vram_percent,
                           args.max_idle_use_percent, f"size {size} pre")
        completed = subprocess.run(
            [str(args.binary), "--size", str(size),
             "--warmup", str(args.warmup),
             "--repetitions", str(args.repetitions)],
            capture_output=True, text=True)
        post = require_idle(args.physical_gpu_index,
                            args.max_idle_vram_percent,
                            args.max_idle_use_percent, f"size {size} post")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "INT8 worker failed")
        records = [json.loads(line) for line in completed.stdout.splitlines()
                   if line.strip()]
        if len(records) != 1:
            raise RuntimeError("INT8 worker must emit exactly one record")
        row = enrich(records[0], size)
        if pre is not None:
            row["pre_run_gpu_state"] = pre
            row["post_run_gpu_state"] = post
        rows.append(row)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    best = max(rows, key=lambda row: row["achieved_tops"])
    summary = {
        "schema_version": 1,
        "track": "mi300_executed_int8_probe",
        "status": "pass",
        "architecture": "gfx942",
        "sizes": args.sizes,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "official_peak_tops": OFFICIAL_PEAK_TOPS,
        "official_memory_bandwidth_tbps": OFFICIAL_BANDWIDTH_TBPS,
        "rows": rows,
        "best": best,
        "boundary": (
            "raw hipBLASLt INT8xINT8->INT32 square GEMM; five exact CPU samples; "
            "no public Tensor dtype, quantizer, scale/zero-point, or Transformer policy"),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
