#!/usr/bin/env python3
"""Validate FP16/BF16 zero-copy external multiply and matmul outputs."""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--elements", type=int, default=4194304)
    parser.add_argument("--matmul-size", type=int, default=1024)
    parser.add_argument("--multiply-iterations", type=int, default=64)
    parser.add_argument("--matmul-iterations", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if (arguments.device < 0 or arguments.elements <= 0 or
            arguments.matmul_size <= 0 or arguments.multiply_iterations <= 1 or
            arguments.matmul_iterations <= 0):
        raise ValueError("device must be non-negative and workload dimensions positive")

    import torch
    from microllm import DType, Stream, matmul_out, multiply_out

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is unavailable")
    report_path = Path(arguments.report)
    if report_path.exists() and not arguments.overwrite:
        raise FileExistsError(f"refusing to replace {report_path}")
    stream_owner = torch.cuda.Stream(device=arguments.device)
    device = f"hip:{arguments.device}"
    stream = Stream.from_external(int(stream_owner.cuda_stream), device=device)
    cases = []
    for name, torch_dtype, micro_dtype, multiply_tolerance, matmul_tolerance in (
            ("fp16", torch.float16, DType.FLOAT16, 0.0, 2.0e-2),
            ("bf16", torch.bfloat16, DType.BFLOAT16, 0.0, 1.5e-1)):
        left_owner = torch.full(
            (arguments.elements,), 1.5, dtype=torch_dtype, device="cuda")
        right_owner = torch.full_like(left_owner, 2.0)
        output_owner = torch.empty_like(left_owner)
        left = wrap(left_owner, device, micro_dtype)
        right = wrap(right_owner, device, micro_dtype)
        output = wrap(output_owner, device, micro_dtype)
        for _ in range(arguments.multiply_iterations):
            multiply_out(output, left, right, stream=stream)
        multiply_finish = torch.cuda.Event()
        multiply_finish.record(stream_owner)
        multiply_pending = not multiply_finish.query()
        multiply_finish.synchronize()
        multiply_error = float((output_owner.float() - 3.0).abs().max().item())

        size = arguments.matmul_size
        matrix_left_owner = torch.full(
            (size, size), 0.01, dtype=torch_dtype, device="cuda")
        matrix_right_owner = torch.full(
            (size, size), 0.02, dtype=torch_dtype, device="cuda")
        matrix_output_owner = torch.empty_like(matrix_left_owner)
        matrix_left = wrap(matrix_left_owner, device, micro_dtype)
        matrix_right = wrap(matrix_right_owner, device, micro_dtype)
        matrix_output = wrap(matrix_output_owner, device, micro_dtype)
        for _ in range(arguments.matmul_iterations):
            matmul_out(matrix_output, matrix_left, matrix_right, stream=stream)
        matmul_finish = torch.cuda.Event()
        matmul_finish.record(stream_owner)
        matmul_pending = not matmul_finish.query()
        matmul_finish.synchronize()
        with torch.cuda.stream(stream_owner):
            reference = torch.mm(matrix_left_owner, matrix_right_owner)
        reference_finish = torch.cuda.Event()
        reference_finish.record(stream_owner)
        reference_finish.synchronize()
        matmul_error = float(
            (matrix_output_owner.float() - reference.float()).abs().max().item())
        pointer_matches = all((
            left.data_ptr == left_owner.data_ptr(),
            right.data_ptr == right_owner.data_ptr(),
            output.data_ptr == output_owner.data_ptr(),
            matrix_left.data_ptr == matrix_left_owner.data_ptr(),
            matrix_right.data_ptr == matrix_right_owner.data_ptr(),
            matrix_output.data_ptr == matrix_output_owner.data_ptr(),
        ))
        wrappers_non_owning = not any((
            left.owning, right.owning, output.owning, matrix_left.owning,
            matrix_right.owning, matrix_output.owning))
        case = {
            "dtype": name,
            "pointer_matches": pointer_matches,
            "wrappers_non_owning": wrappers_non_owning,
            "multiply_pending": multiply_pending,
            "matmul_pending": matmul_pending,
            "multiply_max_error": multiply_error,
            "multiply_tolerance": multiply_tolerance,
            "matmul_max_error": matmul_error,
            "matmul_tolerance": matmul_tolerance,
            "wrapped_payload_bytes": (
                left_owner.numel() * left_owner.element_size() * 3 +
                matrix_left_owner.numel() * matrix_left_owner.element_size() * 3),
            "wrapper_copy_bytes": 0,
        }
        if (not pointer_matches or not wrappers_non_owning or
                not multiply_pending or not matmul_pending or
                multiply_error > multiply_tolerance or
                matmul_error > matmul_tolerance):
            raise RuntimeError(
                f"{name} zero-copy low-precision gate failed: "
                f"{json.dumps(case, sort_keys=True)}")
        cases.append(case)
        for tensor in (left, right, output, matrix_left, matrix_right, matrix_output):
            tensor.close()
    stream.close()
    report = {
        "schema_version": 1,
        "status": "pass",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "cases": cases,
        "case_count": len(cases),
        "total_wrapped_payload_bytes": sum(
            case["wrapped_payload_bytes"] for case in cases),
        "total_wrapper_copy_bytes": 0,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
