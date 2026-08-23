#!/usr/bin/env python3
"""Run the executed MI300 precision GEMM matrix and compute honest roofline bounds."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PEAK_TFLOPS = {
    "fp32_readable": 163.4,
    "fp32": 163.4,
    "fp16": 1307.4,
    "bf16": 1307.4,
    "fp8_e4m3_fnuz": 2614.9,
}
INPUT_BYTES = {
    "fp32_readable": 4,
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8_e4m3_fnuz": 1,
}
OUTPUT_BYTES = {
    "fp32_readable": 4,
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8_e4m3_fnuz": 2,
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--sizes", default="128,256,512,1024")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--reference", choices=("cpu", "fp32"), default="cpu")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    result = parser.parse_args()
    try:
        result.sizes = [int(value) for value in result.sizes.split(",")]
    except ValueError as error:
        parser.error(f"sizes must be comma-separated integers: {error}")
    if not result.binary.is_file() or not result.sizes or \
            any(value <= 0 or value > 4096 for value in result.sizes) or \
            len(set(result.sizes)) != len(result.sizes) or \
            result.warmup < 0 or result.repetitions <= 0 or \
            not 0 <= result.max_idle_vram_percent <= 100 or \
            not 0 <= result.max_idle_use_percent <= 100:
        parser.error("binary, sizes, repetitions or idle thresholds are invalid")
    return result


def parse_gpu_state(text: str, physical_index: int) -> dict:
    card = json.loads(text).get(f"card{physical_index}")
    if not isinstance(card, dict):
        raise RuntimeError(f"rocm-smi did not report physical GPU {physical_index}")
    return {
        "physical_gpu_index": physical_index,
        "gpu_use_percent": int(card["GPU use (%)"]),
        "vram_percent": int(card["GPU Memory Allocated (VRAM%)"]),
    }


def require_idle_gpu(index: int | None, maximum_vram: int,
                     maximum_use: int, boundary: str) -> dict | None:
    if index is None:
        return None
    completed = subprocess.run(
        ["rocm-smi", "--showuse", "--showmemuse", "--json"],
        capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"cannot verify GPU state at {boundary}")
    state = parse_gpu_state(completed.stdout, index)
    if state["vram_percent"] > maximum_vram or \
            state["gpu_use_percent"] > maximum_use:
        raise RuntimeError(
            f"physical GPU {index} is occupied at {boundary}: {state}")
    return state


def roofline_record(record: dict, size: int, bandwidth_tbps: float = 5.3) -> dict:
    dtype = record["dtype"]
    if dtype not in PEAK_TFLOPS or record.get("shape") != [size, size, size]:
        raise RuntimeError("precision worker returned an unknown dtype or shape")
    median_ms = float(record["median_ms"])
    if median_ms <= 0 or record.get("accuracy_passed") is not True:
        raise RuntimeError("precision worker failed timing or accuracy")
    operations = 2.0 * size * size * size
    achieved = operations / (median_ms / 1000.0) / 1.0e12
    moved_bytes = float(size * size) * (
        2 * INPUT_BYTES[dtype] + OUTPUT_BYTES[dtype])
    intensity = operations / moved_bytes
    bandwidth_bound = bandwidth_tbps * intensity
    theoretical = PEAK_TFLOPS[dtype]
    roofline_bound = min(theoretical, bandwidth_bound)
    return {
        **record,
        "size": size,
        "operations": int(operations),
        "arithmetic_intensity_flops_per_byte": intensity,
        "official_peak_tflops": theoretical,
        "bandwidth_bound_tflops": bandwidth_bound,
        "roofline_bound_tflops": roofline_bound,
        "achieved_tflops": achieved,
        "official_peak_utilization": achieved / theoretical,
        "roofline_utilization": achieved / roofline_bound,
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for size in args.sizes:
        pre = require_idle_gpu(
            args.physical_gpu_index, args.max_idle_vram_percent,
            args.max_idle_use_percent, f"size {size} pre")
        completed = subprocess.run(
            [str(args.binary), "--size", str(size),
             "--warmup", str(args.warmup),
             "--repetitions", str(args.repetitions),
             "--reference", args.reference],
            capture_output=True, text=True)
        post = require_idle_gpu(
            args.physical_gpu_index, args.max_idle_vram_percent,
            args.max_idle_use_percent, f"size {size} post")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "precision worker failed")
        records = [json.loads(line) for line in completed.stdout.splitlines()
                   if line.strip()]
        if len(records) != len(PEAK_TFLOPS):
            raise RuntimeError("precision worker returned the wrong dtype count")
        for record in records:
            row = roofline_record(record, size)
            if pre is not None:
                row["pre_run_gpu_state"] = pre
                row["post_run_gpu_state"] = post
            rows.append(row)
            with raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
    by_dtype = {}
    for dtype in PEAK_TFLOPS:
        selected = [row for row in rows if row["dtype"] == dtype]
        best = max(selected, key=lambda row: row["achieved_tflops"])
        by_dtype[dtype] = {
            "best_size": best["size"],
            "best_achieved_tflops": best["achieved_tflops"],
            "best_official_peak_utilization": best[
                "official_peak_utilization"],
            "best_roofline_utilization": best["roofline_utilization"],
            "maximum_absolute_error": max(row["max_abs_error"]
                                          for row in selected),
        }
    summary = {
        "schema_version": 1,
        "track": "mi300_executed_precision_roofline",
        "status": "pass",
        "architecture": "gfx942",
        "official_memory_bandwidth_tbps": 5.3,
        "official_peak_tflops": PEAK_TFLOPS,
        "sizes": args.sizes,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "reference": args.reference,
        "rows": rows,
        "by_dtype": by_dtype,
        "boundary": (
            f"dense square GEMM; reference={args.reference}; "
            "no structured sparsity; Event kernel time; "
            "FP8 output is BF16; INT8/INT4 are not executed by this runner"),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
