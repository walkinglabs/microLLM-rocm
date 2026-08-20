#!/usr/bin/env python3
"""Compare matched microLLM and Python/PyTorch single-device JSONL matrices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microllm", required=True, type=Path)
    parser.add_argument("--pytorch", required=True, type=Path)
    parser.add_argument("--kind", choices=("builtins", "huggingface"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path, record_type: str) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_type") != record_type:
            continue
        if record.get("status") != "pass":
            raise RuntimeError(f"{path}:{number} is not a passing measurement")
        key = (record["model"], record["mode"])
        if key in records:
            raise RuntimeError(f"duplicate measurement {key} in {path}")
        records[key] = record
    if not records:
        raise RuntimeError(f"no {record_type} rows in {path}")
    return records


def require_equal(key: tuple[str, str], micro: dict, torch: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        if micro.get(field) != torch.get(field):
            raise RuntimeError(
                f"{key} is not comparable: {field} "
                f"microLLM={micro.get(field)!r} PyTorch={torch.get(field)!r}"
            )


def positive(record: dict, field: str, key: tuple[str, str]) -> float:
    value = float(record.get(field, 0))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{key} has invalid {field}: {value}")
    return value


def compare_builtin(key: tuple[str, str], micro: dict, torch: dict) -> dict:
    require_equal(
        key, micro, torch,
        ("parameter_count", "fp32_weight_bytes", "dtype", "batch", "context", "steps",
         "warmup", "new_tokens", "measured_tokens", "measurement_profile"),
    )
    if micro.get("measurement_profile") != "comparison":
        raise RuntimeError(f"{key} is a smoke record, not a comparison-grade record")
    micro_throughput = positive(micro, "tokens_per_second", key)
    torch_throughput = positive(torch, "tokens_per_second", key)
    micro_setup = positive(micro, "tokens_per_second_with_setup", key)
    torch_setup = positive(torch, "tokens_per_second_with_setup", key)
    micro_memory = positive(micro, "device_peak_engine_bytes", key)
    torch_memory = positive(torch, "device_peak_allocated_bytes", key)
    return {
        "schema_version": 1,
        "record_type": "single_gpu_framework_comparison",
        "status": "pass",
        "kind": "builtins",
        "model": key[0],
        "mode": key[1],
        "parameter_count": micro["parameter_count"],
        "batch": micro["batch"],
        "context": micro["context"],
        "steps": micro["steps"],
        "warmup": micro["warmup"],
        "measured_tokens": micro["measured_tokens"],
        "microllm_tokens_per_second": micro_throughput,
        "pytorch_tokens_per_second": torch_throughput,
        "throughput_ratio_microllm_over_pytorch": micro_throughput / torch_throughput,
        "microllm_setup_inclusive_tokens_per_second": micro_setup,
        "pytorch_setup_inclusive_tokens_per_second": torch_setup,
        "setup_throughput_ratio_microllm_over_pytorch": micro_setup / torch_setup,
        "microllm_peak_engine_bytes": int(micro_memory),
        "pytorch_peak_allocated_bytes": int(torch_memory),
        "peak_memory_ratio_microllm_engine_over_pytorch_allocated": micro_memory / torch_memory,
        "memory_scope": "microLLM engine allocator versus torch.cuda max_memory_allocated",
    }


def compare_hf(key: tuple[str, str], micro: dict, torch: dict) -> dict:
    require_equal(key, micro, torch,
                  ("parameter_count", "fp32_weight_bytes", "compute_dtype", "loaded_tensors"))
    if key[1] == "infer":
        require_equal(key, micro, torch, ("token_count", "generated_tokens"))
        micro_throughput = positive(micro, "decode_tokens_per_second", key)
        torch_throughput = positive(torch, "decode_tokens_per_second", key)
        throughput_name = "decode_tokens_per_second"
    else:
        require_equal(key, micro, torch, ("trained_tokens",))
        micro_throughput = positive(micro, "tokens_per_second", key)
        torch_throughput = positive(torch, "tokens_per_second", key)
        throughput_name = "training_tokens_per_second"
    micro_memory = positive(micro, "engine_peak_bytes", key)
    torch_memory = positive(torch, "device_peak_allocated_bytes", key)
    return {
        "schema_version": 1,
        "record_type": "single_gpu_framework_comparison",
        "status": "pass",
        "kind": "huggingface",
        "model": key[0],
        "mode": key[1],
        "parameter_count": micro["parameter_count"],
        "throughput_metric": throughput_name,
        "microllm_throughput": micro_throughput,
        "pytorch_throughput": torch_throughput,
        "throughput_ratio_microllm_over_pytorch": micro_throughput / torch_throughput,
        "microllm_peak_engine_bytes": int(micro_memory),
        "pytorch_peak_allocated_bytes": int(torch_memory),
        "peak_memory_ratio_microllm_engine_over_pytorch_allocated": micro_memory / torch_memory,
        "memory_scope": "microLLM engine allocator versus torch.cuda max_memory_allocated",
    }


def main() -> int:
    args = options()
    micro_type = ("single_gpu_model_measurement" if args.kind == "builtins"
                  else "single_gpu_hf_model_measurement")
    torch_type = ("single_gpu_pytorch_model_measurement" if args.kind == "builtins"
                  else "single_gpu_pytorch_hf_model_measurement")
    micro = load(args.microllm, micro_type)
    torch = load(args.pytorch, torch_type)
    if micro.keys() != torch.keys():
        raise RuntimeError(
            f"matrix keys differ: microLLM={sorted(micro)} PyTorch={sorted(torch)}"
        )
    compare = compare_builtin if args.kind == "builtins" else compare_hf
    records = [compare(key, micro[key], torch[key]) for key in sorted(micro)]
    summary = {
        "schema_version": 1,
        "record_type": "single_gpu_framework_comparison_summary",
        "status": "pass",
        "kind": args.kind,
        "comparison_count": len(records),
        "note": "ratios are scoped to matched rows; memory counters have framework-specific scopes",
    }
    lines = [*(json.dumps(record, sort_keys=True) for record in records),
             json.dumps(summary, sort_keys=True)]
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
