#!/usr/bin/env python3
"""Random PyTorch oracle for zero-copy RoPE, Embedding, and loss outputs."""

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


def wrap(tensor, device: str, dtype):
    from microllm import Tensor

    return Tensor.from_external(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(),
        tuple(tensor.shape), tuple(tensor.stride()), dtype=dtype,
        device=device, owner=tensor)


def errors(actual, expected) -> tuple[float, float]:
    import torch

    difference = (actual.float() - expected.float()).flatten()
    maximum = float(difference.abs().max().item()) if difference.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(difference * difference)).item()) \
        if difference.numel() else 0.0
    return maximum, rms


def rope_reference(input, sequence_dim: int, offset: int, base: float):
    import torch

    width = input.shape[-1]
    pairs = torch.arange(width // 2, device=input.device, dtype=torch.float32)
    frequencies = torch.pow(base, -2.0 * pairs / width)
    positions = torch.arange(
        input.shape[sequence_dim], device=input.device, dtype=torch.float32) + offset
    angles = positions[:, None] * frequencies[None, :]
    shape = [1] * input.ndim
    shape[sequence_dim] = input.shape[sequence_dim]
    shape[-1] = width // 2
    cosine = torch.cos(angles).reshape(shape)
    sine = torch.sin(angles).reshape(shape)
    even = input[..., 0::2]
    odd = input[..., 1::2]
    output = torch.empty_like(input)
    output[..., 0::2] = even * cosine - odd * sine
    output[..., 1::2] = even * sine + odd * cosine
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from microllm import (DType, Stream, cross_entropy_out, embedding_out,
                          rope_out)

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

    def append(operation, shape, actual, expected, tolerance, wrappers,
               metadata=None):
        nonlocal total_bytes
        maximum, rms = errors(actual, expected)
        pointer_matches = all(view.data_ptr == owner.data_ptr()
                              for view, owner in wrappers)
        non_owning = all(not view.owning for view, _ in wrappers)
        bytes_value = sum(owner.numel() * owner.element_size()
                          for _, owner in wrappers)
        total_bytes += bytes_value
        row = {
            "operation": operation,
            "shape": list(shape),
            "max_error": maximum,
            "rms_error": rms,
            "tolerance": tolerance,
            "pointer_matches": pointer_matches,
            "wrappers_non_owning": non_owning,
            "wrapped_payload_bytes": bytes_value,
            **(metadata or {}),
        }
        if (not pointer_matches or not non_owning or
                not math.isfinite(maximum) or maximum > tolerance):
            raise RuntimeError(f"sequence/loss row failed: {json.dumps(row)}")
        records.append(row)

    for shape, offset, base in (
            ((1, 1, 2, 4), 0, 10000.0),
            ((2, 7, 3, 8), 5, 10000.0),
            ((1, 17, 2, 16), 11, 5000.0),
            ((1, 64, 4, 32), 17, 10000.0)):
        with torch.cuda.stream(stream_owner):
            input_owner = torch.randn(shape, device="cuda", dtype=torch.float32)
            output_owner = torch.empty_like(input_owner)
        input_view = wrap(input_owner, device, DType.FLOAT32)
        output_view = wrap(output_owner, device, DType.FLOAT32)
        rope_out(output_view, input_view, sequence_dim=1,
                 position_offset=offset, base=base, stream=stream)
        done = torch.cuda.Event()
        done.record(stream_owner)
        done.synchronize()
        reference = rope_reference(input_owner, 1, offset, base)
        append("rope", shape, output_owner, reference, 3.0e-5,
               ((input_view, input_owner), (output_view, output_owner)),
               {"position_offset": offset, "base": base})
        input_view.close()
        output_view.close()

    for vocabulary, width, index_shape in (
            (7, 3, (5,)),
            (33, 16, (2, 7)),
            (257, 64, (3, 17)),
            (1024, 128, (2, 32))):
        with torch.cuda.stream(stream_owner):
            weight_owner = torch.randn(
                (vocabulary, width), device="cuda", dtype=torch.float32)
            indices_owner = torch.randint(
                0, vocabulary, index_shape, device="cuda", dtype=torch.int32)
            if indices_owner.numel() > 1:
                indices_owner.flatten()[-1] = indices_owner.flatten()[0]
            output_owner = torch.empty(
                (*index_shape, width), device="cuda", dtype=torch.float32)
        weight = wrap(weight_owner, device, DType.FLOAT32)
        indices = wrap(indices_owner, device, DType.INT32)
        output = wrap(output_owner, device, DType.FLOAT32)
        embedding_out(output, weight, indices, stream=stream)
        done = torch.cuda.Event()
        done.record(stream_owner)
        done.synchronize()
        reference = functional.embedding(indices_owner.long(), weight_owner)
        append("embedding", output_owner.shape, output_owner, reference, 0.0,
               ((weight, weight_owner), (indices, indices_owner),
                (output, output_owner)),
               {"vocabulary": vocabulary, "width": width})
        weight.close()
        indices.close()
        output.close()

    for rows, classes in ((1, 7), (3, 64), (17, 513), (64, 4096)):
        with torch.cuda.stream(stream_owner):
            logits_owner = torch.randn(
                (rows, classes), device="cuda", dtype=torch.float32)
            targets_owner = torch.randint(
                0, classes, (rows,), device="cuda", dtype=torch.int32)
            if rows > 1:
                targets_owner[1] = -100
            output_owner = torch.empty((), device="cuda", dtype=torch.float32)
            workspace_owner = torch.empty(
                (rows, 2), device="cuda", dtype=torch.float32)
        logits = wrap(logits_owner, device, DType.FLOAT32)
        targets = wrap(targets_owner, device, DType.INT32)
        output = wrap(output_owner, device, DType.FLOAT32)
        workspace = wrap(workspace_owner, device, DType.FLOAT32)
        cross_entropy_out(output, workspace, logits, targets, stream=stream)
        done = torch.cuda.Event()
        done.record(stream_owner)
        done.synchronize()
        reference = functional.cross_entropy(
            logits_owner, targets_owner.long(), ignore_index=-100)
        append("cross_entropy", (), output_owner, reference, 3.0e-5,
               ((logits, logits_owner), (targets, targets_owner),
                (output, output_owner), (workspace, workspace_owner)),
               {"rows": rows, "classes": classes,
                "ignored_rows": int(torch.sum(targets_owner == -100).item())})
        logits.close()
        targets.close()
        output.close()
        workspace.close()
    stream.close()
    report = {
        "schema_version": 1,
        "status": "pass",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "seed": arguments.seed,
        "records": records,
        "record_count": len(records),
        "maximum_error": max(row["max_error"] for row in records),
        "maximum_rms_error": max(row["rms_error"] for row in records),
        "total_wrapped_payload_bytes": total_bytes,
        "total_wrapper_copy_bytes": 0,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
