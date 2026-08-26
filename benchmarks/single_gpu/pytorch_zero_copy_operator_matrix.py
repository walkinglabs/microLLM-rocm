#!/usr/bin/env python3
"""Random-shape PyTorch oracle for zero-copy caller-owned operator outputs."""

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
    difference = (actual.float() - expected.float()).flatten()
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

    global torch
    import torch
    import torch.nn.functional as functional
    from microllm import (DType, Stream, rms_norm_bf16_out, rms_norm_out,
                          softmax_out, swiglu_out)

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is unavailable")
    report_path = Path(arguments.report)
    if report_path.exists() and not arguments.overwrite:
        raise FileExistsError(f"refusing to replace {report_path}")
    torch.manual_seed(arguments.seed)
    stream_owner = torch.cuda.Stream(device=arguments.device)
    device = f"hip:{arguments.device}"
    stream = Stream.from_external(int(stream_owner.cuda_stream), device=device)
    records = []
    wrapped_bytes = 0

    def record(operation: str, dtype: str, shape, actual, expected,
               tolerance: float, wrappers) -> None:
        maximum, rms = errors(actual, expected)
        pointer_matches = all(wrapper.data_ptr == owner.data_ptr()
                              for wrapper, owner in wrappers)
        non_owning = all(not wrapper.owning for wrapper, _ in wrappers)
        bytes_value = sum(owner.numel() * owner.element_size()
                          for _, owner in wrappers)
        row = {
            "operation": operation,
            "dtype": dtype,
            "shape": list(shape),
            "max_error": maximum,
            "rms_error": rms,
            "tolerance": tolerance,
            "pointer_matches": pointer_matches,
            "wrappers_non_owning": non_owning,
            "wrapped_payload_bytes": bytes_value,
        }
        if not pointer_matches or not non_owning or not math.isfinite(maximum) or \
                not math.isfinite(rms) or maximum > tolerance:
            raise RuntimeError(f"operator matrix row failed: {json.dumps(row)}")
        records.append(row)

    for rows, width in ((1, 7), (3, 64), (17, 513), (64, 1024)):
        owner_input = torch.randn((rows, width), device="cuda", dtype=torch.float32)
        owner_output = torch.empty_like(owner_input)
        input_view = wrap(owner_input, device, DType.FLOAT32)
        output_view = wrap(owner_output, device, DType.FLOAT32)
        wrapped_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in (owner_input, owner_output))
        softmax_out(output_view, input_view, stream=stream)
        done = torch.cuda.Event()
        done.record(stream_owner)
        done.synchronize()
        record("softmax", "fp32", owner_input.shape, owner_output,
               torch.softmax(owner_input, dim=-1), 2.0e-6,
               ((input_view, owner_input), (output_view, owner_output)))
        input_view.close()
        output_view.close()

        norm_input = torch.randn((rows, width), device="cuda", dtype=torch.float32)
        norm_weight = torch.randn((width,), device="cuda", dtype=torch.float32)
        norm_output = torch.empty_like(norm_input)
        norm_bf16_output = torch.empty_like(norm_input, dtype=torch.bfloat16)
        input_view = wrap(norm_input, device, DType.FLOAT32)
        weight_view = wrap(norm_weight, device, DType.FLOAT32)
        output_view = wrap(norm_output, device, DType.FLOAT32)
        bf16_output_view = wrap(norm_bf16_output, device, DType.BFLOAT16)
        wrapped_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in (norm_input, norm_weight, norm_output,
                           norm_bf16_output))
        epsilon = 1.0e-5
        rms_norm_out(output_view, input_view, weight_view,
                     epsilon=epsilon, stream=stream)
        rms_norm_bf16_out(bf16_output_view, input_view, weight_view,
                          epsilon=epsilon, stream=stream)
        done = torch.cuda.Event()
        done.record(stream_owner)
        done.synchronize()
        reference = norm_input * torch.rsqrt(
            torch.mean(norm_input * norm_input, dim=-1, keepdim=True) + epsilon)
        reference = reference * norm_weight
        common = ((input_view, norm_input), (weight_view, norm_weight))
        record("rms_norm", "fp32", norm_input.shape, norm_output,
               reference, 5.0e-5, common + ((output_view, norm_output),))
        record("rms_norm_output", "bf16", norm_input.shape, norm_bf16_output,
               reference.to(torch.bfloat16), 0.0,
               common + ((bf16_output_view, norm_bf16_output),))
        for view in (input_view, weight_view, output_view, bf16_output_view):
            view.close()

    for name, torch_dtype, micro_dtype, tolerance in (
            ("fp32", torch.float32, DType.FLOAT32, 2.0e-6),
            ("fp16", torch.float16, DType.FLOAT16, 5.0e-3),
            ("bf16", torch.bfloat16, DType.BFLOAT16, 7.0e-2)):
        for elements in (7, 4099, 65536):
            gate_owner = torch.randn((elements,), device="cuda", dtype=torch_dtype)
            up_owner = torch.randn((elements,), device="cuda", dtype=torch_dtype)
            output_owner = torch.empty_like(gate_owner)
            gate = wrap(gate_owner, device, micro_dtype)
            up = wrap(up_owner, device, micro_dtype)
            output = wrap(output_owner, device, micro_dtype)
            wrapped_bytes += sum(
                tensor.numel() * tensor.element_size()
                for tensor in (gate_owner, up_owner, output_owner))
            swiglu_out(output, gate, up, stream=stream)
            done = torch.cuda.Event()
            done.record(stream_owner)
            done.synchronize()
            reference = functional.silu(gate_owner) * up_owner
            record("swiglu", name, gate_owner.shape, output_owner,
                   reference, tolerance,
                   ((gate, gate_owner), (up, up_owner), (output, output_owner)))
            gate.close()
            up.close()
            output.close()

    stream.close()
    report = {
        "schema_version": 1,
        "status": "pass",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "seed": arguments.seed,
        "records": records,
        "record_count": len(records),
        "total_wrapped_payload_bytes": wrapped_bytes,
        "total_wrapper_copy_bytes": 0,
        "maximum_error": max(row["max_error"] for row in records),
        "maximum_rms_error": max(row["rms_error"] for row in records),
    }
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
