#!/usr/bin/env python3
"""PyTorch oracle for zero-copy caller-owned MHA/GQA Attention workspace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def wrap(tensor, device: str):
    from microllm import DType, Tensor

    return Tensor.from_external(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(),
        tuple(tensor.shape), tuple(tensor.stride()), dtype=DType.FLOAT32,
        device=device, owner=tensor)


def tensor_errors(actual, expected) -> tuple[float, float]:
    import torch

    difference = (actual - expected).flatten()
    maximum = float(difference.abs().max().item()) if difference.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(difference * difference)).item()) \
        if difference.numel() else 0.0
    return maximum, rms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    import torch
    from microllm import Stream, causal_gqa_attention_out

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is unavailable")
    report_path = Path(arguments.report)
    if report_path.exists() and not arguments.overwrite:
        raise FileExistsError(f"refusing to replace {report_path}")
    torch.manual_seed(arguments.seed)
    device = f"hip:{arguments.device}"
    stream_owner = torch.cuda.Stream(device=arguments.device)
    stream = Stream.from_external(int(stream_owner.cuda_stream), device=device)
    records = []
    total_bytes = 0
    cases = (
        (1, 2, 2, 1, 4),
        (1, 4, 2, 7, 8),
        (2, 4, 1, 17, 16),
        (1, 4, 2, 64, 32),
        (1, 4, 2, 256, 64),
    )
    for batches, heads, kv_heads, sequence, width in cases:
        repeats = heads // kv_heads
        scale = 1.0 / math.sqrt(width)
        query_owner = torch.randn(
            (batches, heads, sequence, width), device="cuda", dtype=torch.float32)
        key_owner = torch.randn(
            (batches, kv_heads, sequence, width), device="cuda", dtype=torch.float32)
        value_owner = torch.randn(
            (batches, kv_heads, sequence, width), device="cuda", dtype=torch.float32)
        output_owner = torch.empty_like(query_owner)
        scaled_query_owner = torch.zeros_like(query_owner)
        expanded_kv_owner = torch.zeros_like(query_owner)
        probabilities_owner = torch.zeros(
            (batches, heads, sequence, sequence), device="cuda", dtype=torch.float32)
        owners = (query_owner, key_owner, value_owner, output_owner,
                  scaled_query_owner, expanded_kv_owner, probabilities_owner)
        views = tuple(wrap(owner, device) for owner in owners)
        causal_gqa_attention_out(
            views[3], views[4], views[5], views[6], views[0], views[1], views[2],
            repeats=repeats, scale=scale, stream=stream)
        done = torch.cuda.Event()
        done.record(stream_owner)
        pending_at_record = not done.query()
        done.synchronize()

        expanded_key = key_owner.repeat_interleave(repeats, dim=1)
        expanded_value = value_owner.repeat_interleave(repeats, dim=1)
        scaled_query = query_owner * scale
        scores = torch.matmul(scaled_query, expanded_key.transpose(-2, -1))
        causal_mask = torch.triu(
            torch.ones((sequence, sequence), dtype=torch.bool, device="cuda"),
            diagonal=1)
        probabilities = torch.softmax(
            scores.masked_fill(causal_mask, float("-inf")), dim=-1)
        reference = torch.matmul(probabilities, expanded_value)
        maximum, rms = tensor_errors(output_owner, reference)
        workspace_errors = {}
        if sequence >= 256:
            workspace_errors["scaled_query_max"], \
                workspace_errors["scaled_query_rms"] = tensor_errors(
                    scaled_query_owner, scaled_query)
            workspace_errors["expanded_value_max"], \
                workspace_errors["expanded_value_rms"] = tensor_errors(
                    expanded_kv_owner, expanded_value)
            workspace_errors["probabilities_max"], \
                workspace_errors["probabilities_rms"] = tensor_errors(
                    probabilities_owner, probabilities)
        pointer_matches = all(view.data_ptr == owner.data_ptr()
                              for view, owner in zip(views, owners))
        non_owning = all(not view.owning for view in views)
        bytes_value = sum(owner.numel() * owner.element_size() for owner in owners)
        total_bytes += bytes_value
        tolerance = 1.5e-3
        workspace_maximum = max(
            (value for name, value in workspace_errors.items()
             if name.endswith("_max")), default=0.0)
        row = {
            "batches": batches,
            "heads": heads,
            "kv_heads": kv_heads,
            "sequence": sequence,
            "width": width,
            "repeats": repeats,
            "pending_at_record": pending_at_record,
            "pointer_matches": pointer_matches,
            "wrappers_non_owning": non_owning,
            "output_max_error": maximum,
            "output_rms_error": rms,
            "workspace_errors": workspace_errors,
            "tolerance": tolerance,
            "wrapped_payload_bytes": bytes_value,
        }
        if (not pointer_matches or not non_owning or
                not math.isfinite(maximum) or maximum > tolerance or
                workspace_maximum > tolerance):
            raise RuntimeError(f"attention matrix row failed: {json.dumps(row)}")
        records.append(row)
        for view in views:
            view.close()
    stream.close()
    report = {
        "schema_version": 1,
        "status": "pass",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "seed": arguments.seed,
        "records": records,
        "record_count": len(records),
        "maximum_output_error": max(row["output_max_error"] for row in records),
        "maximum_output_rms_error": max(row["output_rms_error"] for row in records),
        "maximum_workspace_error": max(
            (value for row in records for name, value in row["workspace_errors"].items()
             if name.endswith("_max")), default=0.0),
        "total_wrapped_payload_bytes": total_bytes,
        "total_wrapper_copy_bytes": 0,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
