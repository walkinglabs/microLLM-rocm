#!/usr/bin/env python3
"""PyTorch autograd oracle for caller-owned zero-copy backward outputs."""

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
    from microllm import (
        DType, Stream, cross_entropy_backward_out, embedding_backward_add,
        rms_norm_backward_out, rope_backward_out, softmax_backward_out,
        swiglu_backward_out)

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
    pointer_count = 0
    wrapped_bytes = 0

    def add_record(operation, target, shape, actual, expected, tolerance):
        maximum, rms = errors(actual, expected)
        row = {
            "operation": operation, "target": target, "shape": list(shape),
            "max_error": maximum, "rms_error": rms, "tolerance": tolerance,
        }
        if not math.isfinite(maximum) or maximum > tolerance:
            raise RuntimeError(f"backward row failed: {json.dumps(row)}")
        records.append(row)

    def account(pairs):
        nonlocal pointer_count, wrapped_bytes
        if not all(view.data_ptr == owner.data_ptr() and not view.owning
                   for view, owner in pairs):
            raise RuntimeError("backward external pointer/ownership gate failed")
        pointer_count += len(pairs)
        wrapped_bytes += sum(owner.numel() * owner.element_size()
                             for _, owner in pairs)

    for rows, width in ((1, 7), (3, 64), (17, 513), (64, 1024)):
        with torch.cuda.stream(stream_owner):
            logits = torch.randn((rows, width), device="cuda", dtype=torch.float32)
            probabilities = torch.softmax(logits, dim=-1)
            gradient = torch.randn_like(probabilities)
            input_gradient = torch.empty_like(probabilities)
        views = tuple(wrap(owner, device, DType.FLOAT32) for owner in
                      (probabilities, gradient, input_gradient))
        softmax_backward_out(views[2], views[0], views[1], stream=stream)
        done = torch.cuda.Event(); done.record(stream_owner); done.synchronize()
        reference_input = logits.detach().clone().requires_grad_(True)
        (torch.softmax(reference_input, dim=-1) * gradient).sum().backward()
        add_record("softmax_backward", "input", input_gradient.shape,
                   input_gradient, reference_input.grad, 3.0e-6)
        account(tuple(zip(views, (probabilities, gradient, input_gradient))))
        for view in views: view.close()

        with torch.cuda.stream(stream_owner):
            norm_input = torch.randn((rows, width), device="cuda", dtype=torch.float32)
            norm_weight = torch.randn((width,), device="cuda", dtype=torch.float32)
            norm_gradient = torch.randn_like(norm_input)
            norm_input_gradient = torch.empty_like(norm_input)
            norm_weight_gradient = torch.empty_like(norm_weight)
            norm_workspace = torch.empty((rows,), device="cuda", dtype=torch.float32)
        owners = (norm_input_gradient, norm_weight_gradient, norm_workspace,
                  norm_input, norm_weight, norm_gradient)
        views = tuple(wrap(owner, device, DType.FLOAT32) for owner in owners)
        epsilon = 1.0e-5
        rms_norm_backward_out(*views, epsilon=epsilon, stream=stream)
        done = torch.cuda.Event(); done.record(stream_owner); done.synchronize()
        ref_input = norm_input.detach().clone().requires_grad_(True)
        ref_weight = norm_weight.detach().clone().requires_grad_(True)
        ref_output = ref_input * torch.rsqrt(
            torch.mean(ref_input * ref_input, dim=-1, keepdim=True) + epsilon)
        (ref_output * ref_weight * norm_gradient).sum().backward()
        add_record("rms_norm_backward", "input", norm_input.shape,
                   norm_input_gradient, ref_input.grad, 8.0e-5)
        add_record("rms_norm_backward", "weight", norm_weight.shape,
                   norm_weight_gradient, ref_weight.grad, 8.0e-5)
        inverse_reference = torch.rsqrt(
            torch.mean(norm_input * norm_input, dim=-1) + epsilon)
        add_record("rms_norm_backward", "row_inverse_rms", norm_workspace.shape,
                   norm_workspace, inverse_reference, 8.0e-5)
        account(tuple(zip(views, owners)))
        for view in views: view.close()

    for elements in (7, 4099, 65536):
        with torch.cuda.stream(stream_owner):
            gate = torch.randn((elements,), device="cuda", dtype=torch.float32)
            up = torch.randn_like(gate)
            gradient = torch.randn_like(gate)
            gate_gradient = torch.empty_like(gate)
            up_gradient = torch.empty_like(gate)
        owners = (gate_gradient, up_gradient, gate, up, gradient)
        views = tuple(wrap(owner, device, DType.FLOAT32) for owner in owners)
        swiglu_backward_out(*views, stream=stream)
        done = torch.cuda.Event(); done.record(stream_owner); done.synchronize()
        ref_gate = gate.detach().clone().requires_grad_(True)
        ref_up = up.detach().clone().requires_grad_(True)
        (functional.silu(ref_gate) * ref_up * gradient).sum().backward()
        add_record("swiglu_backward", "gate", gate.shape,
                   gate_gradient, ref_gate.grad, 3.0e-6)
        add_record("swiglu_backward", "up", up.shape,
                   up_gradient, ref_up.grad, 3.0e-6)
        account(tuple(zip(views, owners)))
        for view in views: view.close()

    for shape, offset, base in (
            ((1, 1, 2, 4), 0, 10000.0),
            ((2, 7, 3, 8), 5, 10000.0),
            ((1, 17, 2, 16), 11, 5000.0),
            ((1, 64, 4, 32), 17, 10000.0)):
        with torch.cuda.stream(stream_owner):
            gradient = torch.randn(shape, device="cuda", dtype=torch.float32)
            input_gradient = torch.empty_like(gradient)
        grad_view = wrap(gradient, device, DType.FLOAT32)
        input_view = wrap(input_gradient, device, DType.FLOAT32)
        rope_backward_out(input_view, grad_view, sequence_dim=1,
                          position_offset=offset, base=base, stream=stream)
        done = torch.cuda.Event(); done.record(stream_owner); done.synchronize()
        reference_input = torch.zeros_like(gradient, requires_grad=True)
        rope_reference(reference_input, 1, offset, base).backward(gradient)
        add_record("rope_backward", "input", shape, input_gradient,
                   reference_input.grad, 3.0e-5)
        account(((grad_view, gradient), (input_view, input_gradient)))
        grad_view.close(); input_view.close()

    for rows, classes in ((1, 7), (3, 64), (17, 513), (64, 4096)):
        with torch.cuda.stream(stream_owner):
            logits = torch.randn((rows, classes), device="cuda", dtype=torch.float32)
            targets = torch.randint(0, classes, (rows,), device="cuda", dtype=torch.int32)
            if rows > 1: targets[1] = -100
            loss_gradient = torch.randn((), device="cuda", dtype=torch.float32)
            logits_gradient = torch.empty_like(logits)
            row_stats = torch.empty((rows, 2), device="cuda", dtype=torch.float32)
            factor = torch.empty((), device="cuda", dtype=torch.float32)
        owners = (logits_gradient, row_stats, factor, logits, targets, loss_gradient)
        dtypes = (DType.FLOAT32, DType.FLOAT32, DType.FLOAT32,
                  DType.FLOAT32, DType.INT32, DType.FLOAT32)
        views = tuple(wrap(owner, device, dtype) for owner, dtype in zip(owners, dtypes))
        cross_entropy_backward_out(*views, stream=stream)
        done = torch.cuda.Event(); done.record(stream_owner); done.synchronize()
        reference_logits = logits.detach().clone().requires_grad_(True)
        loss = functional.cross_entropy(
            reference_logits, targets.long(), ignore_index=-100)
        (loss * loss_gradient).backward()
        add_record("cross_entropy_backward", "logits", logits.shape,
                   logits_gradient, reference_logits.grad, 5.0e-5)
        valid_rows = torch.sum(targets != -100)
        add_record("cross_entropy_backward", "factor", (), factor,
                   loss_gradient / valid_rows, 5.0e-5)
        account(tuple(zip(views, owners)))
        for view in views: view.close()

    for vocabulary, width, index_shape in (
            (7, 3, (5,)), (33, 16, (2, 7)),
            (257, 64, (3, 17)), (1024, 128, (2, 32))):
        with torch.cuda.stream(stream_owner):
            indices = torch.randint(
                0, vocabulary, index_shape, device="cuda", dtype=torch.int32)
            if indices.numel() > 1: indices.flatten()[-1] = indices.flatten()[0]
            gradient = torch.randn(
                (*index_shape, width), device="cuda", dtype=torch.float32)
            weight_gradient = torch.zeros(
                (vocabulary, width), device="cuda", dtype=torch.float32)
        owners = (weight_gradient, gradient, indices)
        views = (wrap(weight_gradient, device, DType.FLOAT32),
                 wrap(gradient, device, DType.FLOAT32),
                 wrap(indices, device, DType.INT32))
        embedding_backward_add(*views, stream=stream)
        done = torch.cuda.Event(); done.record(stream_owner); done.synchronize()
        reference_weight = torch.zeros(
            (vocabulary, width), device="cuda", dtype=torch.float32,
            requires_grad=True)
        functional.embedding(indices.long(), reference_weight).backward(gradient)
        add_record("embedding_backward", "weight", weight_gradient.shape,
                   weight_gradient, reference_weight.grad, 3.0e-6)
        account(tuple(zip(views, owners)))
        for view in views: view.close()

    stream.close()
    report = {
        "schema_version": 1, "status": "pass",
        "torch_version": torch.__version__, "torch_hip_version": torch.version.hip,
        "seed": arguments.seed, "records": records,
        "record_count": len(records), "pointer_count": pointer_count,
        "maximum_error": max(row["max_error"] for row in records),
        "maximum_rms_error": max(row["rms_error"] for row in records),
        "total_wrapped_payload_bytes": wrapped_bytes,
        "total_wrapper_copy_bytes": 0,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
